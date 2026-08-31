"""Next-token fidelity of BAU checkpoints against the exact predictive.

For sequences drawn from the population prior (fresh Dirichlet urns), compares
the model's next-token predictive at every position with the exact
Dirichlet-multinomial predictive, reporting the symmetrised KL averaged over
positions and sequences. One row per checkpoint, plus a per-position CSV.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.checkpoints import load_run_info
from balls_and_urns.baselines import generalising_predictive, memorising_predictive
from metrics import symmetrised_kl
from pfn_transformerlens.model.PFN import UnsupervisedPFN


@dataclass
class FidelityConfig:
    checkpoint_paths: list[str] = field(default_factory=list)
    output_dir: str = "outputs/bau_predictive_fidelity"
    n_sequences: int = 256
    seq_len: int = 255  # data tokens; the model input adds BOS in front
    alpha_value: float = 1.0
    batch_size: int = 64
    seed: int = 42
    device: str = "auto"


def fidelity_for_checkpoint(
    checkpoint_path: str, config: FidelityConfig
) -> dict[str, tuple[float, np.ndarray]]:
    """Fidelity under both data laws.

    in_distribution: sequences from pool urns, compared with the memorizing
    predictive (the Bayes optimum for the training law). population:
    sequences from fresh Dirichlet urns, compared with the generalizing
    predictive.
    """
    run_info = load_run_info(Path(checkpoint_path), device=config.device)
    model = run_info["model"]
    assert isinstance(model, UnsupervisedPFN)
    device = next(model.parameters()).device
    theta_pool = torch.as_tensor(run_info["tasks"], dtype=torch.float32)
    vocab_size = int(theta_pool.shape[1])
    bos_token = vocab_size
    alpha = torch.ones(vocab_size) * config.alpha_value

    population_urns = torch.distributions.Dirichlet(alpha).sample(
        (config.n_sequences,)
    )
    pool_urns = theta_pool[
        torch.randint(theta_pool.shape[0], (config.n_sequences,))
    ]

    results: dict[str, tuple[float, np.ndarray]] = {}
    for name, urns in [
        ("in_distribution", pool_urns),
        ("population", population_urns),
    ]:
        tokens = torch.multinomial(urns, config.seq_len, replacement=True)
        per_position_sum = torch.zeros(config.seq_len)
        total_kl = 0.0
        batches = 0
        with torch.no_grad():
            for start in range(0, config.n_sequences, config.batch_size):
                batch = tokens[start : start + config.batch_size]
                bos = torch.full((batch.shape[0], 1), bos_token, dtype=batch.dtype)
                model_input = torch.cat([bos, batch], dim=1).to(
                    device=device, dtype=torch.float32
                )
                prediction = model.predict_on_prompt(model_input)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                model_probs = prediction.probs.cpu()[:, :-1]  # align to data
                if name == "in_distribution":
                    exact = memorising_predictive(batch, theta_pool)
                else:
                    exact = generalising_predictive(batch, alpha)
                total_kl += symmetrised_kl(exact, model_probs)
                for position in range(config.seq_len):
                    per_position_sum[position] += symmetrised_kl(
                        exact[:, position : position + 1],
                        model_probs[:, position : position + 1],
                    )
                batches += 1
        results[name] = (total_kl / batches, (per_position_sum / batches).numpy())
    return results


def main(config: FidelityConfig) -> None:
    if not config.checkpoint_paths:
        raise ValueError("Provide at least one checkpoint path.")
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    per_position_frames = {}
    for checkpoint_path in config.checkpoint_paths:
        results = fidelity_for_checkpoint(checkpoint_path, config)
        path = Path(checkpoint_path)
        label = path.parent.name if path.stem.startswith("checkpoint_step") else path.stem
        row = {"checkpoint": label, "path": checkpoint_path}
        for name, (mean_kl, per_position) in results.items():
            row[f"sym_kl_{name}"] = mean_kl
            per_position_frames[f"{label}_{name}"] = per_position
            print(f"{label} [{name}]: symmetrised KL = {mean_kl:.6f}")
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "fidelity.csv", index=False)
    positions = np.arange(config.seq_len)
    frame = pd.DataFrame({"position": positions, **per_position_frames})
    frame.to_csv(output_dir / "fidelity_per_position.csv", index=False)
    print(f"written to {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(FidelityConfig))
