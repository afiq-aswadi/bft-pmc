"""Evaluate one linear-regression run throughout training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro
from tqdm import tqdm

from analysis.checkpoints import find_all_checkpoints
from linear_regression.analysis.config import SweepConfig
from linear_regression.analysis.data import load_run_info
from linear_regression.analysis.metrics import (
    compute_all_predictive_metrics,
    compute_distribution_metrics_single,
)
from linear_regression.analysis.runner import (
    build_run_eval_inputs,
    load_or_create_shared_eval_context,
)
from linear_regression.plot_dynamics_combined import PlotConfig
from linear_regression.plot_dynamics_combined import main as plot_dynamics
from linear_regression.predictive_monte_carlo import prepare_model_for_long_rollout
from linear_regression.priors import DiscretePrior
from pfn_transformerlens.model.PFN import SupervisedPFN


SOURCE_NAMES = {
    "discrete": "memorising",
    "gaussian": "generalising",
}


@dataclass(slots=True)
class DynamicsConfig:
    """Configuration for one checkpoint-dynamics analysis."""

    run_id: str = ""
    checkpoint_root: str = "checkpoints/lr/task_diversity"

    prompt_length: int = 8
    n_samples: int = 100
    n_prompts: int = 50
    predictive_steps: int = 256
    n_projections: int = 100

    compute_distribution: bool = True
    compute_delta: bool = True
    eval_batch_size: int = 64
    eval_seq_len: int = 64
    eval_position: int | None = None
    include_random_eval: bool = True
    separate_eval_prompts: bool = False
    eval_n_prompts: int = 50
    eval_prompt_length: int = 8

    noise_std: float = 0.5
    seed: int = 42
    output_dir: str = "outputs/lr/distribution_dynamics"
    eval_dataset_dir: str | None = None
    device: str | None = None

    @property
    def noise_variance(self) -> float:
        return self.noise_std**2

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be provided.")
        if self.prompt_length < 0:
            raise ValueError("prompt_length must be non-negative.")
        for name in ["n_samples", "n_prompts", "predictive_steps", "n_projections"]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.eval_batch_size < 1 or self.eval_seq_len < 1:
            raise ValueError(
                "Evaluation batch size and sequence length must be positive."
            )


def _to_sweep_config(config: DynamicsConfig) -> SweepConfig:
    """Map dynamics options onto the shared LR analysis configuration."""
    return SweepConfig(
        checkpoint_root=config.checkpoint_root,
        eval_batch_size=config.eval_batch_size,
        seq_len=config.eval_seq_len,
        noise_std=config.noise_std,
        seed=config.seed,
        eval_position=config.eval_position,
        compute_distribution_metrics=config.compute_distribution,
        prompt_sources=tuple(SOURCE_NAMES),
        prompt_lengths=(config.prompt_length,),
        n_prompts=(config.n_prompts,),
        n_samples=(config.n_samples,),
        n_samples_prior=(config.n_samples,),
        n_projections=config.n_projections,
        predictive_steps=config.predictive_steps,
        include_random_eval=config.include_random_eval,
        separate_eval_prompts=config.separate_eval_prompts,
        eval_n_prompts=config.eval_n_prompts,
        eval_prompt_length=config.eval_prompt_length,
        eval_dataset_dir=config.eval_dataset_dir,
    )


def _resolve_device(config: DynamicsConfig) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_analysis(
    config: DynamicsConfig,
    samples_dir: Path | None = None,
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """Evaluate every checkpoint from one run using shared LR estimators."""
    config.validate()
    run_dir = Path(config.checkpoint_root) / config.run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    checkpoints = find_all_checkpoints(run_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir}")

    device = _resolve_device(config)
    first_info = load_run_info(checkpoints[0], device=device)
    prior = DiscretePrior(
        task_size=int(first_info["task_size"]),
        tasks=first_info["tasks"],
        device="cpu",
    )
    num_tasks = int(first_info["num_tasks"])
    sweep_config = _to_sweep_config(config)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    shared_context = load_or_create_shared_eval_context(sweep_config, [run_dir])
    eval_data, prompt_data = build_run_eval_inputs(
        sweep_config,
        config.run_id,
        first_info,
        prior,
        shared_context,
    )

    results: list[dict[str, float | int]] = []
    per_prompt_results: list[dict[str, float | int | str]] = []
    for checkpoint_path in tqdm(checkpoints, desc="Processing checkpoints"):
        run_info = load_run_info(checkpoint_path, device=device)
        assert int(run_info["num_tasks"]) == num_tasks
        assert torch.equal(run_info["tasks"].cpu(), prior.tasks.cpu())
        model = run_info["model"]
        assert isinstance(model, SupervisedPFN)
        step = int(checkpoint_path.stem.split("_")[-1])

        metrics: dict[str, float] = {}
        if config.compute_delta:
            predictive_metrics = compute_all_predictive_metrics(
                model,
                prior,
                sweep_config,
                eval_data=eval_data,
            )
            for key, value in predictive_metrics.items():
                source, metric = key.split("/", maxsplit=1)
                if not metric.startswith("delta_vs_baseline_"):
                    continue
                assert source in {
                    "data_memorising",
                    "data_generalising",
                    "data_random",
                }
                output_source = "random" if source == "data_random" else source
                metrics[f"{metric}_on_{output_source}"] = value

        if config.compute_distribution:
            effective_steps = prepare_model_for_long_rollout(
                model,
                rollout_length=config.predictive_steps,
                prompt_length=config.prompt_length,
            )
            sources = ["N/A"] if config.prompt_length == 0 else list(SOURCE_NAMES)
            for source in sources:
                source_metrics, source_samples, source_per_prompt = (
                    compute_distribution_metrics_single(
                        model,
                        prior,
                        noise_std=config.noise_std,
                        noise_variance=config.noise_variance,
                        n_projections=config.n_projections,
                        prompt_source=source,
                        prompt_length=config.prompt_length,
                        predictive_steps=effective_steps,
                        n_samples=config.n_samples if config.prompt_length > 0 else 0,
                        n_samples_prior=config.n_samples
                        if config.prompt_length == 0
                        else 0,
                        n_prompts=config.n_prompts if config.prompt_length > 0 else 0,
                        model_prepared=True,
                        prompt_data=prompt_data,
                    )
                )
                suffix = (
                    "" if source == "N/A" else f"_from_prompts_{SOURCE_NAMES[source]}"
                )
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
                            "source": SOURCE_NAMES.get(source, source),
                            "prompt_idx": prompt_index,
                            **{
                                key.removeprefix("dist/"): value
                                for key, value in prompt_metrics.items()
                            },
                        }
                    )
                if samples_dir is not None:
                    samples_dir.mkdir(parents=True, exist_ok=True)
                    sample_suffix = SOURCE_NAMES.get(source, "prior")
                    np.savez(
                        samples_dir / f"step_{step}_{sample_suffix}.npz",
                        **source_samples,
                    )

        results.append({"step": step, **metrics})

    per_prompt_df = pd.DataFrame(per_prompt_results)
    return pd.DataFrame(results), num_tasks, per_prompt_df


def main(config: DynamicsConfig) -> None:
    """Run dynamics analysis and render the aggregate figure."""
    output_dir = Path(config.output_dir) / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    metrics, _, per_prompt = run_analysis(config, samples_dir=output_dir / "samples")
    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    if not per_prompt.empty:
        per_prompt.to_csv(output_dir / "per_prompt_metrics.csv", index=False)
    if config.compute_distribution and (
        config.prompt_length == 0 or config.compute_delta
    ):
        plot_dynamics(
            PlotConfig(metrics_csv=str(metrics_path), output_dir=str(output_dir))
        )


if __name__ == "__main__":
    main(tyro.cli(DynamicsConfig))
