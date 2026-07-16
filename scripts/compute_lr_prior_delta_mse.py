"""Prior-mode Delta MSE for LR: zero-context model prediction vs both baselines.

Length-0 context = no observations. For each run we feed a batch of x_0 ~ N(0, I)
to the model as a length-1 prompt, take the position-0 expected y prediction, and
compare against:
  - Memorising baseline (dMMSE at k=0): (1/M) sum_m w_m . x_0
  - Generalising baseline (ridge at k=0): 0  (zero-mean Gaussian prior on w)

Both are derived from `baselines.{memorising_predictor, generalising_predictor}`
evaluated at sequence position 0 -- where they reduce to their prior predictive
means without any conditioning data.

This is the prediction-space analog of the prior ED/SW sweep plot. Writes a
sidecar CSV that the 1x3 prior plotter reads alongside the sweep metrics.csv.
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

from analysis.checkpoints import find_latest_checkpoint
from linear_regression.analysis.data import load_run_info
from linear_regression.baselines import generalising_predictor, memorising_predictor
from linear_regression.evals import mse
from linear_regression.priors import DiscretePrior
from pfn_transformerlens.model.PFN import DistributionPrediction, SupervisedPFN


@dataclass(slots=True)
class PriorDeltaConfig:
    """Configuration for the linear-regression prior delta sidecar."""

    checkpoint_root: Path = Path("checkpoints/lr/task_diversity")
    out_csv: Path = Path("outputs/lr/sweep_analysis/prior_delta_mse.csv")
    noise_std: float = 0.5
    batch_size: int = 4096
    seed: int = 42
    device: str | None = None

    def validate(self) -> None:
        if not self.checkpoint_root.is_dir():
            raise FileNotFoundError(
                f"Checkpoint root not found: {self.checkpoint_root}"
            )
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def _prior_delta_mse(
    *,
    model: SupervisedPFN,
    prior: DiscretePrior,
    task_size: int,
    noise_variance: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Return prior-mode (length-0 context) delta MSE vs mem and gen baselines."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    xs = torch.randn(batch_size, 1, task_size, generator=g)  # (B, 1, D)
    dummy_ys = torch.zeros(batch_size, 1, 1)  # not used at position 0

    xs_dev = xs.to(device)
    dummy_dev = dummy_ys.to(device)

    pred = model.predict_on_prompt(xs_dev, dummy_dev.squeeze(-1))
    if isinstance(pred, tuple):
        pred = pred[0]
    assert isinstance(pred, DistributionPrediction)
    # Expected-y under the model's bucketised distribution at position 0.
    model_preds = (
        (pred.probs * pred.y_grid).sum(dim=-1).unsqueeze(-1).cpu()
    )  # (B, 1, 1)

    mem_preds = memorising_predictor(xs, dummy_ys, prior, noise_variance).cpu()
    gen_preds = generalising_predictor(xs, dummy_ys, noise_variance).cpu()

    d = task_size
    return {
        "model_mse_self": mse(model_preds, torch.zeros_like(model_preds)).item() / d,
        "baseline_memorising_mse_self": mse(
            mem_preds, torch.zeros_like(mem_preds)
        ).item()
        / d,
        "delta_vs_memorising": mse(model_preds, mem_preds).item() / d,
        "delta_vs_generalising": mse(model_preds, gen_preds).item() / d,
    }


def main(config: PriorDeltaConfig) -> None:
    """Compute one prior-predictive delta row per checkpoint run."""
    config.validate()
    device = _resolve_device(config.device)
    noise_variance = config.noise_std**2
    run_dirs = sorted(
        path for path in config.checkpoint_root.iterdir() if path.is_dir()
    )

    rows: list[dict[str, float | int | str]] = []
    for run_dir in run_dirs:
        ckpt = find_latest_checkpoint(run_dir)
        if ckpt is None:
            continue
        info = load_run_info(ckpt, device)
        model = info["model"]
        assert isinstance(model, SupervisedPFN)
        tasks = info["tasks"].to(device)
        task_size = int(info["task_size"])
        num_tasks = int(info["num_tasks"])

        prior = DiscretePrior(task_size=task_size, tasks=tasks, device=str(device))

        model.eval()
        metrics = _prior_delta_mse(
            model=model,
            prior=prior,
            task_size=task_size,
            noise_variance=noise_variance,
            batch_size=config.batch_size,
            device=device,
            seed=config.seed,
        )
        row = {
            "run_id": run_dir.name,
            "num_tasks": num_tasks,
            **metrics,
        }
        rows.append(row)
    if not rows:
        raise RuntimeError("No linear-regression checkpoints were analyzed.")
    config.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("num_tasks").to_csv(config.out_csv, index=False)


if __name__ == "__main__":
    main(tyro.cli(PriorDeltaConfig))
