"""Evaluate one BAU run throughout training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro
from tqdm import tqdm

from analysis.checkpoints import find_all_checkpoints, load_run_info
from balls_and_urns.analysis import (
    compute_distribution_metrics,
    compute_predictive_metrics,
    load_evaluation_tokens,
    precompute_baseline_samples,
    prepend_bos,
)
from balls_and_urns.plot_dynamics_combined import PlotConfig
from balls_and_urns.plot_dynamics_combined import main as plot_dynamics
from pfn_transformerlens.model.PFN import UnsupervisedPFN


@dataclass(slots=True)
class DynamicsConfig:
    run_id: str = ""
    checkpoint_root: str = "checkpoints/bau/task_diversity"
    eval_dataset_dir: str = ""
    compute_delta: bool = True
    compute_distribution: bool = True
    compute_prior_mode: bool = True
    n_samples: int = 100
    predictive_steps: int = 256
    n_projections: int = 100
    chunk_size: int = 100
    alpha_value: float = 1.0
    checkpoint_subsample: int = 1
    output_dir: str = "outputs/bau/distribution_dynamics"
    seed: int = 42
    device: str | None = None

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be provided.")
        if not self.eval_dataset_dir:
            raise ValueError("eval_dataset_dir must be provided.")
        for name in [
            "n_samples",
            "predictive_steps",
            "n_projections",
            "chunk_size",
            "checkpoint_subsample",
        ]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.alpha_value <= 0:
            raise ValueError("alpha_value must be positive.")


def _resolve_device(config: DynamicsConfig) -> str:
    if config.device is not None:
        return config.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dynamics_prediction_names(metrics: dict[str, float]) -> dict[str, float]:
    """Convert shared sweep keys to the historical dynamics column schema."""
    renamed: dict[str, float] = {}
    for key, value in metrics.items():
        source, metric = key.split("/", maxsplit=1)
        renamed[f"{metric}_on_{source}"] = value
    return renamed


def run_analysis(config: DynamicsConfig) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """Evaluate every selected checkpoint from one BAU run."""
    config.validate()
    run_dir = Path(config.checkpoint_root) / config.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    checkpoints = find_all_checkpoints(run_dir)[:: config.checkpoint_subsample]
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir}")

    device = _resolve_device(config)
    first_info = load_run_info(checkpoints[0], device)
    theta_pool = first_info["tasks"]
    task_count = int(first_info["num_tasks"])
    vocab_size = theta_pool.shape[1]
    alpha = torch.full((vocab_size,), config.alpha_value)

    dataset_dir = Path(config.eval_dataset_dir)
    generalising_tokens, memorising_tokens = load_evaluation_tokens(
        dataset_dir,
        config.run_id,
        theta_pool,
        alpha,
    )
    eval_tokens = {
        "data_memorising": prepend_bos(memorising_tokens, vocab_size),
        "data_generalising": prepend_bos(generalising_tokens, vocab_size),
    }

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    prompt_sources: dict[str, torch.Tensor | None] = {}
    baseline_samples: dict[
        str,
        dict[str, np.ndarray] | list[dict[str, np.ndarray]],
    ] = {}
    if config.compute_distribution:
        if config.compute_prior_mode:
            prompt_sources["prior"] = None
        prompt_sources["generalising"] = generalising_tokens
        prompt_sources["memorising"] = memorising_tokens
        baseline_samples = {
            source: precompute_baseline_samples(
                config.n_samples,
                alpha,
                theta_pool,
                prompts,
            )
            for source, prompts in prompt_sources.items()
        }

    samples_dir = Path(config.output_dir) / config.run_id / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float | int]] = []
    per_prompt_results: list[dict[str, float | int | str]] = []

    for checkpoint_path in tqdm(checkpoints, desc="Processing checkpoints"):
        run_info = load_run_info(checkpoint_path, device)
        assert int(run_info["num_tasks"]) == task_count
        assert torch.equal(run_info["tasks"].cpu(), theta_pool.cpu())
        model = run_info["model"]
        assert isinstance(model, UnsupervisedPFN)
        step = int(checkpoint_path.stem.split("_")[-1])
        metrics: dict[str, float] = {}

        if config.compute_distribution:
            context_size = int(model.transformer.cfg.n_ctx)
            for source, prompts in prompt_sources.items():
                prompt_length = 0 if prompts is None else prompts.shape[1]
                effective_steps = min(
                    config.predictive_steps,
                    context_size - prompt_length - 1,
                )
                save_source = "prior" if prompts is None else f"data_{source}"
                source_metrics, source_per_prompt = compute_distribution_metrics(
                    model,
                    vocab_size=vocab_size,
                    bos_token=vocab_size,
                    alpha=alpha,
                    theta_pool=theta_pool,
                    effective_steps=effective_steps,
                    n_samples=config.n_samples,
                    n_projections=config.n_projections,
                    chunk_size=config.chunk_size,
                    prompts=prompts,
                    baseline_samples=baseline_samples[source],
                    samples_save_path=samples_dir
                    / f"step{step:08d}__source_{save_source}.npz",
                    step=step,
                    prompt_source=save_source,
                )
                suffix = "" if prompts is None else f"_from_prompts_{source}"
                metrics.update(
                    {
                        f"{key.removeprefix('dist/')}{suffix}": value
                        for key, value in source_metrics.items()
                    }
                )
                for prompt_index, prompt_metrics in enumerate(source_per_prompt):
                    per_prompt_results.append(
                        {
                            "step": step,
                            "prompt_source": source,
                            "prompt_idx": prompt_index,
                            **{
                                f"{key.removeprefix('dist/')}{suffix}": value
                                for key, value in prompt_metrics.items()
                            },
                        }
                    )

        if config.compute_delta:
            metrics.update(
                _dynamics_prediction_names(
                    compute_predictive_metrics(model, eval_tokens, alpha, theta_pool)
                )
            )
        results.append({"step": step, **metrics})

    return pd.DataFrame(results), task_count, pd.DataFrame(per_prompt_results)


def main(config: DynamicsConfig) -> None:
    """Run BAU dynamics analysis and render the aggregate plot."""
    output_dir = Path(config.output_dir) / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    metrics, _, per_prompt = run_analysis(config)
    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    if not per_prompt.empty:
        per_prompt.to_csv(output_dir / "per_prompt_metrics.csv", index=False)
    if config.compute_delta and config.compute_distribution:
        plot_dynamics(
            PlotConfig(metrics_csv=str(metrics_path), output_dir=str(output_dir))
        )


if __name__ == "__main__":
    main(tyro.cli(DynamicsConfig))
