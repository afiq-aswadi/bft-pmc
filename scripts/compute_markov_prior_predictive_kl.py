"""First-step predictive KL: model P(s_1 | s_0) vs the two prior baselines.

Length-1 dataset = a single observed state s_0. For each run:
  - Model: feed [BOS, s_0], take softmax of logits at position 1 -> P_model(s_1 | s_0).
  - Memorising prior: bi_ret_predictive with no observed transitions yet,
    which equals (1/M) sum_m T_m[s_0, :].
  - Generalising prior: bi_inf_predictive with no observed transitions, which
    equals uniform 1/k under Dirichlet(1) row priors.

We sym-KL the model against each baseline and average over s_0 in {0, ..., k-1}
(uniform). One scalar per (run, baseline) pair.

This is the predictive-space analog of the prior ED/SW plot. The two metrics
live in different spaces (T-matrix samples vs single-row predictives), so this
script writes its own sidecar CSV and a 1x3 plotter consumes both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tyro

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markov.analysis_common import load_markov_state_dict
from markov.config import load_config
from markov.model import MarkovTransformer
from metrics import symmetrised_kl


@dataclass(slots=True)
class PriorKLConfig:
    """Configuration for the Markov prior-predictive KL sidecar."""

    manifest_csv: Path = Path("outputs/markov/pmc_replotted/manifest.csv")
    out_csv: Path = Path("outputs/markov/pmc_replotted/prior_predictive_kl.csv")
    device: str | None = None

    def validate(self) -> None:
        if not self.manifest_csv.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_config_path(row: pd.Series, output_dir: Path) -> Path:
    raw = str(row.get("config_path", "")).strip()
    if raw and Path(raw).exists():
        return Path(raw)
    sibling = output_dir / "resolved_config.yaml"
    if sibling.exists():
        return sibling
    fallback = (
        Path("outputs/markov/training") / output_dir.name / "resolved_config.yaml"
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No resolved_config.yaml for run {output_dir.name}.")


def _resolve_checkpoint_path(row: pd.Series, output_dir: Path) -> Path:
    raw = str(row.get("checkpoint_path", "")).strip()
    if raw and Path(raw).exists():
        return Path(raw)
    fallback = (
        Path("checkpoints/markov/task_diversity")
        / output_dir.name
        / "checkpoint_step_100000.pt"
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No final checkpoint for run {output_dir.name}.")


def _resolve_training_matrices(output_dir: Path) -> Path:
    """Locate the training task pool for a run."""
    direct = output_dir / "transition_matrices.npy"
    if direct.exists():
        return direct
    # The pmc_replotted dir doesn't carry the npy; the training run does.
    fallback = (
        Path("outputs/markov/training") / output_dir.name / "transition_matrices.npy"
    )
    if fallback.exists():
        return fallback
    bundle = output_dir / "pmc_eval_bundle.npz"
    if bundle.exists():
        return bundle
    raise FileNotFoundError(
        f"Cannot locate training_matrices for run {output_dir.name}: "
        "neither transition_matrices.npy nor pmc_eval_bundle.npz found."
    )


def _load_training_matrices(path: Path, device: torch.device) -> torch.Tensor:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            arr = archive["training_matrices"]
    else:
        arr = np.load(path)
    return torch.from_numpy(arr).to(device).float()


@torch.inference_mode()
def _first_step_kl(
    *,
    model: MarkovTransformer,
    training_matrices: torch.Tensor,
    k: int,
    bos_token_id: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return (sym_KL_vs_memorising, sym_KL_vs_generalising) at length-1 context."""
    s0 = torch.arange(k, device=device).unsqueeze(1)  # (k, 1)
    bos = torch.full((k, 1), fill_value=bos_token_id, dtype=torch.long, device=device)
    model_inputs = torch.cat([bos, s0], dim=1)  # (k, 2) = [[BOS, s_0]]

    logits = model(model_inputs)  # (k, 2, vocab_size)
    model_probs = F.softmax(logits[:, 1, :k], dim=-1)  # (k, k): row i = P(. | s_0=i)

    # Closed-form priors at zero observed transitions. The bi_*_predictive helpers
    # assume seq_len >= 2 (they prepend a zero slab over S-1 transitions), so we
    # write these out directly. Memorising: uniform weights over {T_m} -> row mean
    # of the training pool. Generalising: Dirichlet(1) row prior with no counts ->
    # uniform 1/k.
    mem_preds = training_matrices.mean(dim=0)  # (k, k); row i = (1/M) sum_m T_m[i,:]
    gen_preds = torch.full((k, k), 1.0 / k, device=device, dtype=model_probs.dtype)

    kl_mem = symmetrised_kl(model_probs, mem_preds)
    kl_gen = symmetrised_kl(model_probs, gen_preds)
    return kl_mem, kl_gen


def main(config: PriorKLConfig) -> None:
    """Compute one prior-predictive KL row per manifest entry."""
    config.validate()
    device = _resolve_device(config.device)
    manifest = pd.read_csv(config.manifest_csv)

    rows: list[dict[str, float | int | str]] = []
    for _, row in manifest.iterrows():
        run_name = str(row["run_name"])
        n_chains = int(row["n_chains"])
        output_dir = Path(str(row["output_dir"]))

        config_path = _resolve_config_path(row, output_dir)
        run_config = load_config(config_path)
        checkpoint_path = _resolve_checkpoint_path(row, output_dir)
        tm_path = _resolve_training_matrices(output_dir)
        training_matrices = _load_training_matrices(tm_path, device)
        if training_matrices.shape[0] != n_chains:
            raise ValueError(
                f"{run_name}: training_matrices has {training_matrices.shape[0]} chains, "
                f"manifest says {n_chains}."
            )

        model = MarkovTransformer(
            vocab_size=run_config.k + 1,
            d_model=run_config.d_model,
            seq_len=run_config.seq_len,
            num_layers=run_config.num_layers,
            num_heads=run_config.num_heads,
            expansion_factor=run_config.expansion_factor,
            rope_theta=run_config.rope_theta,
        ).to(device)
        model.load_state_dict(load_markov_state_dict(checkpoint_path, device))
        model.eval()

        kl_mem, kl_gen = _first_step_kl(
            model=model,
            training_matrices=training_matrices,
            k=run_config.k,
            bos_token_id=run_config.k,
            device=device,
        )

        rows.append(
            {
                "run_name": run_name,
                "n_chains": n_chains,
                "kl_vs_memorising": kl_mem,
                "kl_vs_generalising": kl_gen,
            }
        )
    if not rows:
        raise RuntimeError("The Markov manifest contains no runs.")
    config.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("n_chains").to_csv(config.out_csv, index=False)


if __name__ == "__main__":
    main(tyro.cli(PriorKLConfig))
