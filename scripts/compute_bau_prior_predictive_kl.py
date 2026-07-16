"""Prior predictive KL for BAU: model P(x_1 | BOS) vs the two prior baselines.

Length-0 context = no observations. For each run:
  - Model: feed [BOS], take softmax of logits at position 0 -> P_model(x_1 | BOS).
  - Memorising prior predictive: marginal under uniform-on-pool prior,
        P(x_1) = (1/M) sum_m theta_m.
  - Generalising prior predictive: marginal under Dirichlet(alpha),
        P(x_1) = alpha / alpha.sum() (uniform 1/V when alpha = ones).

BAU sequences are i.i.d. given theta, so the prior predictive already encodes
the full first-token distribution under each prior -- no need to condition on
an observation. One scalar per (run, baseline) pair.

This is the predictive-space analog of the prior ED/SW plot. The two metrics
live in different spaces (theta samples vs single-token marginals), so this
script writes its own sidecar CSV and a 1x3 plotter consumes both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
import torch
import tyro

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.checkpoints import find_latest_checkpoint, load_run_info
from metrics import symmetrised_kl
from pfn_transformerlens.model.PFN import DistributionPrediction, UnsupervisedPFN


@dataclass(slots=True)
class PriorKLConfig:
    """Configuration for the BAU prior-predictive KL sidecar."""

    checkpoint_root: Path = Path("checkpoints/bau/task_diversity")
    out_csv: Path = Path("outputs/bau/sweep_analysis/prior_predictive_kl.csv")
    alpha_value: float = 1.0
    device: str | None = None

    def validate(self) -> None:
        if not self.checkpoint_root.is_dir():
            raise FileNotFoundError(
                f"Checkpoint root not found: {self.checkpoint_root}"
            )
        if self.alpha_value <= 0:
            raise ValueError("alpha_value must be positive.")


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def _prior_predictive_kl(
    *,
    model: UnsupervisedPFN,
    thetas: torch.Tensor,
    alpha: torch.Tensor,
    bos_token: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return (sym_KL_vs_memorising, sym_KL_vs_generalising) at zero observations."""
    inputs = torch.tensor([[bos_token]], device=device)  # (1, 1)

    pred = model.predict_on_prompt(inputs)
    if isinstance(pred, tuple):
        pred = pred[0]
    assert isinstance(pred, DistributionPrediction)
    # d_vocab_out=V (no BOS), so probs is (1, 1, V); position 0 = P(x_1 | BOS).
    model_probs = pred.probs[:, 0, :].cpu().float()  # (1, V)

    thetas_cpu = thetas.detach().cpu().float()
    # Memorising prior predictive: (1/M) sum_m theta_m, the row-mean of the pool.
    mem_preds = thetas_cpu.mean(dim=0, keepdim=True)  # (1, V)

    # Generalising prior predictive: alpha / alpha.sum() (uniform 1/V if alpha=ones).
    alpha_cpu = alpha.detach().cpu().float()
    gen_preds = (alpha_cpu / alpha_cpu.sum()).unsqueeze(0)  # (1, V)

    kl_mem = symmetrised_kl(model_probs, mem_preds)
    kl_gen = symmetrised_kl(model_probs, gen_preds)
    return kl_mem, kl_gen


def main(config: PriorKLConfig) -> None:
    """Compute one prior-predictive KL row per checkpoint run."""
    config.validate()
    device = _resolve_device(config.device)
    run_dirs = sorted(
        path for path in config.checkpoint_root.iterdir() if path.is_dir()
    )

    rows: list[dict[str, float | int | str]] = []
    for run_dir in run_dirs:
        ckpt = find_latest_checkpoint(run_dir)
        if ckpt is None:
            continue
        info = load_run_info(ckpt, str(device))
        model = info["model"]
        assert isinstance(model, UnsupervisedPFN)
        thetas = info["tasks"].to(device)
        num_tasks = int(info["num_tasks"])
        V = thetas.shape[1]
        alpha = torch.full((V,), config.alpha_value, device=device)
        bos_token = V

        model.eval()
        kl_mem, kl_gen = _prior_predictive_kl(
            model=model,
            thetas=thetas,
            alpha=alpha,
            bos_token=bos_token,
            device=device,
        )
        rows.append(
            {
                "run_id": run_dir.name,
                "num_tasks": num_tasks,
                "kl_vs_memorising": kl_mem,
                "kl_vs_generalising": kl_gen,
            }
        )
    if not rows:
        raise RuntimeError("No BAU checkpoints were analyzed.")
    config.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("num_tasks").to_csv(config.out_csv, index=False)


if __name__ == "__main__":
    main(tyro.cli(PriorKLConfig))
