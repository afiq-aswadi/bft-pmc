"""Generate the canonical BAU evaluation dataset.

Writes:
  {output_dir}/shared.npz       generalising tokens (Dirichlet prior, shared across all runs)
  {output_dir}/{run_id}.npz     memorising tokens for each run (uses that run's theta_pool)
  {output_dir}/metadata.json    provenance (alpha, seq_len, batch_size, seed, generated_at)

Each call is idempotent: existing files are left alone unless --overwrite is passed.
This is the SOLE producer of BAU eval datasets; analysis scripts only load.
"""

import datetime
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from balls_and_urns.data import make_bau_generator, make_generator_from_pool
from balls_and_urns.dataset import (
    load_generalising_dataset,
    load_memorising_dataset,
    save_generalising_dataset,
    save_memorising_dataset,
)
from analysis.checkpoints import find_latest_checkpoint, load_run_info
from pfn_transformerlens import sample_batch


@dataclass(slots=True)
class DatasetConfig:
    output_dir: Path
    checkpoint_root: Path
    alpha_value: float = 1.0
    seq_len: int = 16
    batch_size: int = 256
    seed: int = 42
    overwrite: bool = False

    def validate(self) -> None:
        if not self.checkpoint_root.is_dir():
            raise FileNotFoundError(
                f"Checkpoint root not found: {self.checkpoint_root}"
            )
        if self.alpha_value <= 0:
            raise ValueError("alpha_value must be positive.")
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")


def _check_cache_compat(
    cached: dict[str, object],
    expected: dict[str, object],
    path: Path,
    fields: tuple[str, ...],
) -> None:
    """Fail loudly if a cached .npz was generated with different config than now requested."""
    mismatches = [
        f"{f}: cached={cached[f]!r} requested={expected[f]!r}"
        for f in fields
        if cached[f] != expected[f]
    ]
    assert not mismatches, (
        f"cached {path.name} was generated with different settings:\n  "
        + "\n  ".join(mismatches)
        + f"\nDelete {path} or pass --overwrite to regenerate."
    )


def main(config: DatasetConfig) -> None:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # filter to run dirs that actually contain checkpoints (skip e.g. logs/)
    run_dirs = sorted(
        d
        for d in config.checkpoint_root.iterdir()
        if d.is_dir() and find_latest_checkpoint(d) is not None
    )
    assert run_dirs, f"No checkpoint-containing run dirs in {config.checkpoint_root}"

    # vocab_size is fixed across runs; read from any one
    first_ckpt = find_latest_checkpoint(run_dirs[0])
    assert first_ckpt is not None  # guaranteed by run_dirs filter
    first_info = load_run_info(first_ckpt, device="cpu")
    vocab_size = first_info["tasks"].shape[1]
    del first_info

    alpha = torch.ones(vocab_size) * config.alpha_value

    # -- shared generalising dataset --
    shared_path = config.output_dir / "shared.npz"
    if shared_path.exists() and not config.overwrite:
        cached = load_generalising_dataset(shared_path)
        _check_cache_compat(
            cached,
            {
                "seq_len": config.seq_len,
                "batch_size": config.batch_size,
                "vocab_size": vocab_size,
            },
            shared_path,
            ("seq_len", "batch_size", "vocab_size"),
        )
        # alpha is an array; check separately for value match
        assert np.allclose(cached["alpha"], alpha.numpy()), (
            f"cached {shared_path.name} alpha {cached['alpha']} != requested {alpha.numpy()};"
            f" delete or pass --overwrite"
        )
    else:
        generalising_gen = make_bau_generator(alpha)
        _, tokens = sample_batch(
            generalising_gen, batch_size=config.batch_size, seq_len=config.seq_len
        )
        assert tokens is not None
        save_generalising_dataset(
            shared_path,
            generalising_tokens=tokens.numpy(),
            alpha=alpha.numpy(),
            vocab_size=vocab_size,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
        )

    # -- per-run memorising datasets --
    for run_dir in run_dirs:
        run_id = run_dir.name
        run_path = config.output_dir / f"{run_id}.npz"

        ckpt = find_latest_checkpoint(run_dir)
        assert ckpt is not None  # guaranteed by run_dirs filter above
        info = load_run_info(ckpt, device="cpu")
        theta_pool = info["tasks"]
        num_tasks = info["num_tasks"]
        del info

        if run_path.exists() and not config.overwrite:
            cached = load_memorising_dataset(run_path)
            assert cached["num_tasks"] == num_tasks, (
                f"cached {run_path.name} num_tasks {cached['num_tasks']} != current {num_tasks};"
                f" delete or pass --overwrite"
            )
            assert cached["memorising_tokens"].shape == (
                config.batch_size,
                config.seq_len,
            ), (
                f"cached {run_path.name} shape {cached['memorising_tokens'].shape}"
                f" != ({config.batch_size}, {config.seq_len}); delete or pass --overwrite"
            )
            continue

        gen = make_generator_from_pool(theta_pool)
        _, tokens = sample_batch(
            gen, batch_size=config.batch_size, seq_len=config.seq_len
        )
        assert tokens is not None
        save_memorising_dataset(
            run_path,
            memorising_tokens=tokens.numpy(),
            theta_pool=theta_pool.numpy(),
            num_tasks=num_tasks,
        )

    # -- metadata --
    metadata = {
        "alpha_value": config.alpha_value,
        "seq_len": config.seq_len,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_root": str(config.checkpoint_root),
    }
    metadata_path = config.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(tyro.cli(DatasetConfig))
