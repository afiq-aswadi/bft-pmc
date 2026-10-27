"""Evaluate final BAU checkpoints across task diversity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro
from tqdm import tqdm

from bft_pmc.analysis.checkpoints import find_latest_checkpoint, load_run_info
from bft_pmc.balls_and_urns.analysis import (
    compute_distribution_metrics,
    compute_predictive_metrics,
    load_evaluation_tokens,
    precompute_baseline_samples,
    prepend_bos,
)
from bft_pmc.balls_and_urns.plot_sweep_combined import PlotConfig
from bft_pmc.balls_and_urns.plot_sweep_combined import main as plot_sweep
from pfn_transformerlens.model.PFN import UnsupervisedPFN


@dataclass(slots=True)
class SweepConfig:
    """Evaluate the final checkpoint of every run in a balls-and-urns sweep.

    Prediction-space metrics (symmetrised KL between the model's next-token
    predictive and each reference's) come from the fixed evaluation prompts;
    latent-space metrics from PMC rollouts after them. Results go to a timestamped directory under
    output_dir.
    """

    checkpoint_root: str = "checkpoints/bau/task_diversity"
    """Directory holding one run directory per task diversity; the latest
    checkpoint of each is evaluated."""
    eval_dataset_dir: str = ""
    """Fixed evaluation prompts, from eval.py bau-eval-dataset or, for the
    original paper checkpoints, paper_data/bau/eval_datasets (required)."""
    seed: int = 42
    """Seed for rollouts and reference draws."""
    compute_distribution_metrics: bool = True
    """Run PMC and the latent-space comparison, writing sample bundles; off
    gives prediction-space metrics only."""
    compute_prior_mode: bool = True
    """Include the prior (BOS-only prompt)."""
    prior_only_distribution: bool = False
    """Run PMC for the prior only, skipping the posterior prompt sources."""
    n_samples: int = 100
    """Rollouts per posterior prompt."""
    n_samples_prior: int = 1024
    """Rollouts for the prior."""
    n_projections: int = 100
    """Random projections for sliced Wasserstein."""
    predictive_steps: int = 223
    """Rollout length N, capped so that the prompt and rollout fit the model
    context."""
    chunk_size: int = 100
    """Rollouts generated per forward batch."""
    alpha_value: float = 1.0
    """Concentration of the symmetric Dirichlet population prior, which defines
    the generalizing reference."""
    output_dir: str = "outputs/bau/sweep_analysis"
    """Parent of the timestamped results directory."""
    device: str | None = None
    """Torch device; None picks cuda, then mps, then cpu."""
    save_rollouts: bool = True
    """Write the rollout token streams beside each sample bundle as
    <bundle>_rollouts.npz; turn off when only the estimates are needed."""

    def validate(self) -> None:
        if not self.eval_dataset_dir:
            raise ValueError("eval_dataset_dir must be provided.")
        for name in [
            "n_samples",
            "n_samples_prior",
            "n_projections",
            "predictive_steps",
            "chunk_size",
        ]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.alpha_value <= 0:
            raise ValueError("alpha_value must be positive.")


def _resolve_device(config: SweepConfig) -> str:
    if config.device is not None:
        return config.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_analysis(config: SweepConfig, output_dir: Path) -> pd.DataFrame:
    """Evaluate each run's latest checkpoint against fixed evaluation data."""
    config.validate()
    checkpoint_root = Path(config.checkpoint_root)
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")
    dataset_dir = Path(config.eval_dataset_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    results: list[dict[str, float | int | str]] = []
    run_dirs = sorted(path for path in checkpoint_root.iterdir() if path.is_dir())
    for run_dir in tqdm(run_dirs, desc="Processing runs"):
        checkpoint_path = find_latest_checkpoint(run_dir)
        if checkpoint_path is None:
            continue
        run_info = load_run_info(checkpoint_path, device)
        model = run_info["model"]
        assert isinstance(model, UnsupervisedPFN)
        theta_pool = run_info["tasks"]
        task_count = int(run_info["num_tasks"])
        vocab_size = theta_pool.shape[1]
        alpha = torch.full((vocab_size,), config.alpha_value)
        generalising_tokens, memorising_tokens = load_evaluation_tokens(
            dataset_dir,
            run_dir.name,
            theta_pool,
            alpha,
        )
        checkpoint_step = int(checkpoint_path.stem.split("_")[-1])
        eval_tokens = {
            "data_memorising": prepend_bos(memorising_tokens, vocab_size),
            "data_generalising": prepend_bos(generalising_tokens, vocab_size),
        }
        base_row: dict[str, float | int | str] = {
            "run_id": run_dir.name,
            "num_tasks": task_count,
            "checkpoint_step": checkpoint_step,
            **compute_predictive_metrics(model, eval_tokens, alpha, theta_pool),
        }

        if not config.compute_distribution_metrics:
            results.append({**base_row, "prompt_source": "none"})
            continue

        prompt_sources: dict[str, torch.Tensor | None] = {}
        if config.compute_prior_mode:
            prompt_sources["prior"] = None
        if not config.prior_only_distribution:
            prompt_sources["data_generalising"] = generalising_tokens
            prompt_sources["data_memorising"] = memorising_tokens

        context_size = int(model.transformer.cfg.n_ctx)
        for source, prompts in prompt_sources.items():
            prompt_length = 0 if prompts is None else prompts.shape[1]
            effective_steps = min(
                config.predictive_steps,
                context_size - prompt_length - 1,
            )
            sample_count = (
                config.n_samples_prior if prompts is None else config.n_samples
            )
            baseline_samples = precompute_baseline_samples(
                sample_count,
                alpha,
                theta_pool,
                prompts,
            )
            source_metrics, _ = compute_distribution_metrics(
                model,
                vocab_size=vocab_size,
                bos_token=vocab_size,
                alpha=alpha,
                theta_pool=theta_pool,
                effective_steps=effective_steps,
                n_samples=sample_count,
                n_projections=config.n_projections,
                chunk_size=config.chunk_size,
                prompts=prompts,
                baseline_samples=baseline_samples,
                samples_save_path=samples_dir / f"{run_dir.name}__source_{source}.npz",
                step=checkpoint_step,
                prompt_source=source,
                save_rollouts=config.save_rollouts,
            )
            results.append({**base_row, "prompt_source": source, **source_metrics})

    if not results:
        raise RuntimeError("No BAU checkpoints were analyzed.")
    return pd.DataFrame(results).sort_values("num_tasks").reset_index(drop=True)


def main(config: SweepConfig) -> None:
    """Run the BAU sweep analysis and generate aggregate plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    metrics = run_analysis(config, output_dir)
    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    if config.compute_distribution_metrics:
        plot_sweep(
            PlotConfig(metrics_csv=str(metrics_path), output_dir=str(output_dir))
        )


if __name__ == "__main__":
    main(tyro.cli(SweepConfig))
