"""Evaluate Markov sweep checkpoints with distribution-distance metrics.

The analysis emits the shared energy-distance and sliced-Wasserstein layouts
for the Markov transformer setting:

- prior sweep:      1 x 2 (ED, SW) over M
- posterior sweep:  2 x 2 (rows = ID/OOD prompts; cols = ED, SW) over M
- prior dynamics:   1 x 2 (ED, SW) over training step
- posterior dynamics:
                    2 x 2 (rows = ID/OOD prompts; cols = ED, SW) over step

The green curve denotes the in-distribution reference and the blue curve denotes
the out-of-distribution reference, matching the memorising/generalising colors
used in the LR code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import tyro

from markov.analysis_common import load_markov_state_dict, load_trained_markov_artifacts
from markov.plotting import (
    plot_distribution_distance_dynamics,
    plot_distribution_distance_sweep,
    training_posterior_weights,
    transition_count_matrix,
)
from markov.predictive_monte_carlo import (
    predictive_monte_carlo_transition_matrix_chunked,
)
from markov.samples_saving import PMCSamplingConfig, resolve_pmc_sampling_config
from metrics import energy_distance_multidim, sliced_wasserstein


_PROMPT_SOURCES = ("in_distribution", "out_of_distribution")


@dataclass(slots=True)
class RunSpec:
    """Resolved metadata for one Markov run to analyze."""

    run_name: str
    checkpoint_dir: Path
    latest_checkpoint_path: Path
    config_path: Path | None = None
    expected_n_chains: int | None = None


@dataclass(slots=True)
class SweepConfig:
    """CLI arguments for Markov distribution-distance analysis."""

    summary_csv_path: str | None = None
    checkpoint_root: str = "checkpoints/markov/task_diversity"
    training_output_root: str = "outputs/markov/training"
    output_dir: str = "outputs/markov/sweep_analysis"

    run_name_contains: str = ""
    device: str | None = None
    allow_seed_rehydration: bool = False

    n_samples: int = 128
    n_prompts: int = 16
    prompt_len: int = 8
    generation_length: int = 400
    n_projections: int = 100
    chunk_size: int = 64
    seed: int = 0

    def validate(self) -> None:
        if self.n_samples < 1:
            raise ValueError("n_samples must be positive.")
        if self.n_prompts < 0:
            raise ValueError("n_prompts must be non-negative.")
        if self.prompt_len < 0:
            raise ValueError("prompt_len must be non-negative.")
        if self.generation_length < 2:
            raise ValueError("generation_length must be at least 2.")
        if self.n_projections < 1:
            raise ValueError("n_projections must be positive.")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive.")


@dataclass(slots=True)
class PriorReferenceBundle:
    """Fixed prior-mode reference samples reused across checkpoints for one run."""

    archive: dict[str, np.ndarray]


@dataclass(slots=True)
class PosteriorReferenceBundle:
    """Fixed posterior-mode prompts and reference samples reused across checkpoints."""

    prompt_source: str
    prompts: torch.Tensor
    archive: dict[str, np.ndarray]


def _resolve_device(device: str | None) -> torch.device:
    """Resolve the requested torch device."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _step_from_checkpoint_name(path: Path) -> int:
    """Extract the training step from a checkpoint filename."""
    return int(path.stem.split("_")[-1])


def _latest_checkpoint_in_dir(checkpoint_dir: Path) -> Path:
    """Return the numerically latest checkpoint in one run directory."""
    candidates = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint_step_*.pt files found in {checkpoint_dir}."
        )
    return max(candidates, key=_step_from_checkpoint_name)


def _candidate_run_output_dirs(
    config: SweepConfig,
    run_name: str,
) -> list[Path]:
    """Return plausible output directories containing resolved configs and data caches."""
    return [Path(config.training_output_root) / run_name]


def _resolve_run_config_path(
    config: SweepConfig,
    run_name: str,
) -> Path | None:
    """Find a local resolved_config.yaml for a discovered run, if present."""
    for run_output_dir in _candidate_run_output_dirs(config, run_name):
        candidate = run_output_dir / "resolved_config.yaml"
        if candidate.exists():
            return candidate
    return None


def _csv_text(value: object) -> str:
    """Normalize an optional summary-CSV cell to a stripped string."""
    return "" if pd.isna(value) else str(value).strip()


def _discover_run_specs(config: SweepConfig) -> list[RunSpec]:
    """Resolve the run list from either a summary CSV or a checkpoint tree."""
    if config.summary_csv_path is not None:
        summary_df = pd.read_csv(config.summary_csv_path)
        run_specs: list[RunSpec] = []
        for _, row in summary_df.iterrows():
            run_name = _csv_text(row.get("run_name", ""))
            if config.run_name_contains and config.run_name_contains not in run_name:
                continue

            latest_checkpoint_raw = _csv_text(row.get("latest_checkpoint_path", ""))
            checkpoint_dir_raw = _csv_text(row.get("checkpoint_dir", ""))
            if latest_checkpoint_raw:
                latest_checkpoint_path = Path(latest_checkpoint_raw)
                checkpoint_dir = latest_checkpoint_path.parent
            elif checkpoint_dir_raw:
                checkpoint_dir = Path(checkpoint_dir_raw)
                latest_checkpoint_path = _latest_checkpoint_in_dir(checkpoint_dir)
            else:
                raise ValueError(
                    "Each summary CSV row must provide `latest_checkpoint_path` or `checkpoint_dir`."
                )

            output_dir_raw = _csv_text(row.get("output_dir", ""))
            config_path = None
            if output_dir_raw:
                candidate = Path(output_dir_raw) / "resolved_config.yaml"
                if candidate.exists():
                    config_path = candidate
            if config_path is None:
                config_path = _resolve_run_config_path(
                    config, run_name or checkpoint_dir.name
                )

            expected_n_chains = None
            if "n_chains" in row and pd.notna(row["n_chains"]):
                expected_n_chains = int(row["n_chains"])

            run_specs.append(
                RunSpec(
                    run_name=run_name or checkpoint_dir.name,
                    checkpoint_dir=checkpoint_dir,
                    latest_checkpoint_path=latest_checkpoint_path,
                    config_path=config_path,
                    expected_n_chains=expected_n_chains,
                )
            )

        if not run_specs:
            raise FileNotFoundError(
                "No runs matched the requested filters in the supplied summary CSV."
            )
        return run_specs

    checkpoint_root = Path(config.checkpoint_root)
    run_dirs = sorted(
        {
            path.parent.resolve()
            for path in checkpoint_root.rglob("checkpoint_step_*.pt")
        }
    )
    if config.run_name_contains:
        run_dirs = [path for path in run_dirs if config.run_name_contains in path.name]
    if not run_dirs:
        detail = (
            f" containing {config.run_name_contains!r}"
            if config.run_name_contains
            else ""
        )
        raise FileNotFoundError(
            f"No checkpoint runs{detail} found under {checkpoint_root}."
        )

    return [
        RunSpec(
            run_name=run_dir.name,
            checkpoint_dir=run_dir,
            latest_checkpoint_path=_latest_checkpoint_in_dir(run_dir),
            config_path=_resolve_run_config_path(config, run_dir.name),
        )
        for run_dir in run_dirs
    ]


def _flatten_transition_samples(samples: np.ndarray) -> np.ndarray:
    """Flatten `[n_samples, k, k]` transition matrices into `[n_samples, k^2]`."""
    return samples.reshape(samples.shape[0], -1)


def _sample_discrete_reference(
    training_matrices: np.ndarray,
    *,
    n_samples: int,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Sample from the empirical training-pool distribution over transition matrices."""
    num_training = training_matrices.shape[0]
    if weights is None:
        indices = rng.integers(0, num_training, size=n_samples)
    else:
        indices = rng.choice(num_training, size=n_samples, replace=True, p=weights)
    return np.asarray(training_matrices[indices], dtype=np.float32)


def _sample_dirichlet_reference(
    k: int,
    *,
    n_samples: int,
    rng: np.random.Generator,
    counts: np.ndarray | None = None,
) -> np.ndarray:
    """Sample from the row-wise Dirichlet population prior or posterior."""
    alpha = np.ones((k, k), dtype=np.float64)
    if counts is not None:
        alpha += counts.astype(np.float64)

    samples = np.empty((n_samples, k, k), dtype=np.float32)
    for src in range(k):
        samples[:, src, :] = rng.dirichlet(alpha[src], size=n_samples).astype(
            np.float32
        )
    return samples


def _sample_prompt_batch(
    dataset,
    *,
    prompt_source: str,
    prompt_len: int,
    n_prompts: int,
    seed: int,
) -> torch.Tensor:
    """Sample a batch of ID or OOD prompts."""
    torch.manual_seed(seed)

    prompts: list[torch.Tensor] = []
    for _ in range(n_prompts):
        if prompt_source == "in_distribution":
            chain_index = int(
                torch.randint(0, dataset.n_chains, (1,), device=dataset.device).item()
            )
            transition_matrix = dataset.transition_matrices[chain_index]
        elif prompt_source == "out_of_distribution":
            transition_matrix = dataset.sample_ood_matrix()
        else:
            raise ValueError(f"Unsupported prompt source: {prompt_source!r}")

        prompts.append(dataset.sample_eval_chains(transition_matrix, prompt_len))

    return torch.stack(prompts, dim=0)


def _prepare_prior_reference_bundle(
    dataset,
    *,
    n_samples: int,
    id_seed: int,
    ood_seed: int,
) -> PriorReferenceBundle:
    """Precompute fixed prior-mode reference samples for one Markov run."""
    training_matrices = (
        dataset.transition_matrices.detach().cpu().numpy().astype(np.float32)
    )
    baseline_in_distribution = _sample_discrete_reference(
        training_matrices,
        n_samples=n_samples,
        rng=np.random.default_rng(id_seed),
    )[None, :, :, :]
    baseline_out_of_distribution = _sample_dirichlet_reference(
        dataset.k,
        n_samples=n_samples,
        rng=np.random.default_rng(ood_seed),
    )[None, :, :, :]
    n_training = training_matrices.shape[0]
    prior_alpha = np.ones((dataset.k, dataset.k), dtype=np.float64)
    return PriorReferenceBundle(
        archive={
            "baseline_in_distribution": baseline_in_distribution,
            "baseline_out_of_distribution": baseline_out_of_distribution,
            "posterior_training_weights": np.full(
                (1, n_training),
                1.0 / n_training,
                dtype=np.float64,
            ),
            "posterior_dirichlet_alpha": prior_alpha[None, :, :],
            "prior_dirichlet_alpha": prior_alpha,
            "training_transition_matrices": training_matrices,
            "prompt_tokens": np.empty((1, 0), dtype=np.int64),
        }
    )


def _prepare_posterior_reference_bundle(
    dataset,
    *,
    prompt_source: str,
    sampling: PMCSamplingConfig,
    n_prompts: int,
    prompt_seed: int,
    baseline_seed: int,
) -> PosteriorReferenceBundle:
    """Precompute fixed posterior-mode prompts and reference samples for one run."""
    prompts = _sample_prompt_batch(
        dataset,
        prompt_source=prompt_source,
        prompt_len=sampling.prompt_len,
        n_prompts=n_prompts,
        seed=prompt_seed,
    )

    prompt_array = prompts.detach().cpu().numpy()
    training_matrices = (
        dataset.transition_matrices.detach().cpu().numpy().astype(np.float32)
    )
    n_training = training_matrices.shape[0]
    prior_alpha = np.ones((dataset.k, dataset.k), dtype=np.float64)

    baseline_in_distribution = np.empty(
        (n_prompts, sampling.num_samples, dataset.k, dataset.k),
        dtype=np.float32,
    )
    baseline_out_of_distribution = np.empty_like(baseline_in_distribution)
    posterior_training_weights = np.empty((n_prompts, n_training), dtype=np.float64)
    posterior_dirichlet_alpha = np.empty(
        (n_prompts, dataset.k, dataset.k), dtype=np.float64
    )

    for prompt_index in range(n_prompts):
        prompt_rng = np.random.default_rng(baseline_seed + prompt_index)
        weights = training_posterior_weights(
            training_matrices, prompt_array[prompt_index]
        )
        counts = transition_count_matrix(prompt_array[prompt_index], dataset.k)
        posterior_alpha = prior_alpha + counts.astype(np.float64)

        baseline_in_distribution[prompt_index] = _sample_discrete_reference(
            training_matrices,
            n_samples=sampling.num_samples,
            rng=prompt_rng,
            weights=weights,
        )
        baseline_out_of_distribution[prompt_index] = _sample_dirichlet_reference(
            dataset.k,
            n_samples=sampling.num_samples,
            rng=prompt_rng,
            counts=counts,
        )
        posterior_training_weights[prompt_index] = weights
        posterior_dirichlet_alpha[prompt_index] = posterior_alpha

    return PosteriorReferenceBundle(
        prompt_source=prompt_source,
        prompts=prompts,
        archive={
            "baseline_in_distribution": baseline_in_distribution,
            "baseline_out_of_distribution": baseline_out_of_distribution,
            "posterior_training_weights": posterior_training_weights,
            "posterior_dirichlet_alpha": posterior_dirichlet_alpha,
            "prior_dirichlet_alpha": prior_alpha,
            "training_transition_matrices": training_matrices,
            "prompt_tokens": prompt_array,
        },
    )


def _save_predictive_samples(
    path: Path,
    *,
    model_samples: np.ndarray,
    baseline_in_distribution: np.ndarray,
    baseline_out_of_distribution: np.ndarray,
    posterior_training_weights: np.ndarray,
    posterior_dirichlet_alpha: np.ndarray,
    prior_dirichlet_alpha: np.ndarray,
    training_transition_matrices: np.ndarray,
    prompt_tokens: np.ndarray,
    step: int,
    prompt_source: str,
    n_chains: int,
) -> None:
    """Save a bundled predictive-sampling artifact for later re-scoring/re-plotting."""
    n_prompts, _n_samples, k_rows, k_cols = model_samples.shape
    assert k_rows == k_cols, (
        f"expected square transition matrices, got {model_samples.shape}"
    )
    assert (
        baseline_in_distribution.shape
        == baseline_out_of_distribution.shape
        == model_samples.shape
    ), (
        "shape mismatch: "
        f"model={model_samples.shape} "
        f"id={baseline_in_distribution.shape} "
        f"ood={baseline_out_of_distribution.shape}"
    )
    assert posterior_training_weights.shape == (
        n_prompts,
        training_transition_matrices.shape[0],
    ), (
        "posterior_training_weights shape "
        f"{posterior_training_weights.shape} does not match "
        f"({n_prompts}, {training_transition_matrices.shape[0]})"
    )
    assert posterior_dirichlet_alpha.shape == (n_prompts, k_rows, k_cols), (
        "posterior_dirichlet_alpha shape "
        f"{posterior_dirichlet_alpha.shape} does not match ({n_prompts}, {k_rows}, {k_cols})"
    )
    assert prior_dirichlet_alpha.shape == (k_rows, k_cols), (
        "prior_dirichlet_alpha shape "
        f"{prior_dirichlet_alpha.shape} does not match ({k_rows}, {k_cols})"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        model_samples=model_samples,
        baseline_in_distribution=baseline_in_distribution,
        baseline_out_of_distribution=baseline_out_of_distribution,
        posterior_training_weights=posterior_training_weights,
        posterior_dirichlet_alpha=posterior_dirichlet_alpha,
        prior_dirichlet_alpha=prior_dirichlet_alpha,
        training_transition_matrices=training_transition_matrices,
        prompt_tokens=prompt_tokens,
        step=np.int64(step),
        prompt_source=np.array(prompt_source),
        n_chains=np.int64(n_chains),
    )


def _compute_prior_metrics(
    *,
    model,
    dataset,
    sampling: PMCSamplingConfig,
    n_projections: int,
    chunk_size: int,
    seed: int,
    references: PriorReferenceBundle,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compute prior ED/SW metrics against ID and OOD references."""
    torch.manual_seed(seed)
    model_samples = predictive_monte_carlo_transition_matrix_chunked(
        model=model,
        dataset=dataset,
        forward_recursion_steps=sampling.generation_length,
        forward_recursion_samples=sampling.num_samples,
        chunk_size=chunk_size,
        prompt=None,
    )
    assert isinstance(model_samples, np.ndarray)
    model_samples = cast(np.ndarray, model_samples)

    in_distribution_reference = references.archive["baseline_in_distribution"][0]
    out_of_distribution_reference = references.archive["baseline_out_of_distribution"][
        0
    ]

    model_flat = _flatten_transition_samples(model_samples)
    in_flat = _flatten_transition_samples(in_distribution_reference)
    out_flat = _flatten_transition_samples(out_of_distribution_reference)
    metrics = {
        "ed_vs_baseline_in_distribution": energy_distance_multidim(model_flat, in_flat),
        "sw_vs_baseline_in_distribution": sliced_wasserstein(
            model_flat,
            in_flat,
            n_projections=n_projections,
            seed=seed,
        ),
        "ed_vs_baseline_out_of_distribution": energy_distance_multidim(
            model_flat, out_flat
        ),
        "sw_vs_baseline_out_of_distribution": sliced_wasserstein(
            model_flat,
            out_flat,
            n_projections=n_projections,
            seed=seed,
        ),
    }
    samples = {
        "model_samples": model_samples[None, :, :, :],
        **references.archive,
    }
    return metrics, samples


def _compute_posterior_metrics(
    *,
    model,
    dataset,
    references: PosteriorReferenceBundle,
    sampling: PMCSamplingConfig,
    n_projections: int,
    chunk_size: int,
    seed: int,
) -> tuple[
    dict[str, float],
    list[dict[str, float | int | str]],
    dict[str, np.ndarray],
]:
    """Compute posterior ED/SW metrics averaged across many prompts."""
    torch.manual_seed(seed + 17)
    model_samples = predictive_monte_carlo_transition_matrix_chunked(
        model=model,
        dataset=dataset,
        forward_recursion_steps=sampling.generation_length,
        forward_recursion_samples=sampling.num_samples,
        chunk_size=chunk_size,
        prompt=references.prompts,
    )
    assert isinstance(model_samples, np.ndarray)

    per_prompt_rows: list[dict[str, float | int | str]] = []
    averaged_metrics: list[dict[str, float]] = []
    baseline_in_distribution = references.archive["baseline_in_distribution"]
    baseline_out_of_distribution = references.archive["baseline_out_of_distribution"]

    n_prompts = model_samples.shape[0]
    for prompt_index in range(n_prompts):
        prompt_seed = seed + prompt_index
        in_distribution_reference = baseline_in_distribution[prompt_index]
        out_of_distribution_reference = baseline_out_of_distribution[prompt_index]
        model_flat = _flatten_transition_samples(model_samples[prompt_index])
        in_flat = _flatten_transition_samples(in_distribution_reference)
        out_flat = _flatten_transition_samples(out_of_distribution_reference)
        metrics = {
            "ed_vs_baseline_in_distribution": energy_distance_multidim(
                model_flat, in_flat
            ),
            "sw_vs_baseline_in_distribution": sliced_wasserstein(
                model_flat,
                in_flat,
                n_projections=n_projections,
                seed=prompt_seed,
            ),
            "ed_vs_baseline_out_of_distribution": energy_distance_multidim(
                model_flat, out_flat
            ),
            "sw_vs_baseline_out_of_distribution": sliced_wasserstein(
                model_flat,
                out_flat,
                n_projections=n_projections,
                seed=prompt_seed,
            ),
        }
        averaged_metrics.append(metrics)
        per_prompt_rows.append(
            {
                "prompt_source": references.prompt_source,
                "prompt_idx": prompt_index,
                **metrics,
            }
        )

    reduced_metrics = {
        key: float(np.mean([row[key] for row in averaged_metrics]))
        for key in averaged_metrics[0]
    }
    samples = {
        "model_samples": model_samples,
        **references.archive,
    }
    return reduced_metrics, per_prompt_rows, samples


def _with_prompt_suffix(
    metrics: dict[str, float], prompt_source: str
) -> dict[str, float]:
    """Append the prompt-source suffix used by the combined dynamics plot."""
    return {
        f"{key}_from_prompts_{prompt_source}": value for key, value in metrics.items()
    }


def _build_sweep_rows(
    *,
    final_row: pd.Series,
    run_name: str,
    n_chains: int,
    prompt_length: int,
    n_samples: int,
    n_prompts: int,
) -> list[dict[str, float | int | str]]:
    """Convert one run's final dynamics row into sweep-style rows."""
    checkpoint_step = int(final_row["step"])
    rows: list[dict[str, float | int | str]] = [
        {
            "run_name": run_name,
            "n_chains": n_chains,
            "checkpoint_step": checkpoint_step,
            "prompt_source": "N/A",
            "prompt_length": 0,
            "n_samples": n_samples,
            "n_prompts": 0,
            "dist/ed_vs_baseline_in_distribution": float(
                final_row["ed_vs_baseline_in_distribution"]
            ),
            "dist/sw_vs_baseline_in_distribution": float(
                final_row["sw_vs_baseline_in_distribution"]
            ),
            "dist/ed_vs_baseline_out_of_distribution": float(
                final_row["ed_vs_baseline_out_of_distribution"]
            ),
            "dist/sw_vs_baseline_out_of_distribution": float(
                final_row["sw_vs_baseline_out_of_distribution"]
            ),
        }
    ]

    if prompt_length <= 0:
        return rows

    for prompt_source in _PROMPT_SOURCES:
        suffix = f"_from_prompts_{prompt_source}"
        rows.append(
            {
                "run_name": run_name,
                "n_chains": n_chains,
                "checkpoint_step": checkpoint_step,
                "prompt_source": prompt_source,
                "prompt_length": prompt_length,
                "n_samples": n_samples,
                "n_prompts": n_prompts,
                "dist/ed_vs_baseline_in_distribution": float(
                    final_row[f"ed_vs_baseline_in_distribution{suffix}"]
                ),
                "dist/sw_vs_baseline_in_distribution": float(
                    final_row[f"sw_vs_baseline_in_distribution{suffix}"]
                ),
                "dist/ed_vs_baseline_out_of_distribution": float(
                    final_row[f"ed_vs_baseline_out_of_distribution{suffix}"]
                ),
                "dist/sw_vs_baseline_out_of_distribution": float(
                    final_row[f"sw_vs_baseline_out_of_distribution{suffix}"]
                ),
            }
        )

    return rows


def _analyze_run(
    run_spec: RunSpec,
    *,
    config: SweepConfig,
    device: torch.device,
    runs_output_root: Path,
    sweep_samples_dir: Path,
) -> tuple[int, list[dict[str, float | int | str]]]:
    """Analyze one run across all checkpoints and emit dynamics artifacts."""
    artifacts = load_trained_markov_artifacts(
        run_spec.config_path,
        run_spec.latest_checkpoint_path,
        device=device,
        allow_seed_rehydration=config.allow_seed_rehydration,
    )
    if (
        run_spec.expected_n_chains is not None
        and run_spec.expected_n_chains != artifacts.config.n_chains
    ):
        raise ValueError(
            f"Run {run_spec.run_name} expected n_chains={run_spec.expected_n_chains}, "
            f"but checkpoint resolved to n_chains={artifacts.config.n_chains}."
        )

    sampling = resolve_pmc_sampling_config(
        PMCSamplingConfig(
            num_samples=config.n_samples,
            prompt_len=config.prompt_len,
            generation_length=config.generation_length,
        ),
        seq_len=artifacts.config.seq_len,
    )

    checkpoint_paths = sorted(
        run_spec.checkpoint_dir.glob("checkpoint_step_*.pt"),
        key=_step_from_checkpoint_name,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {run_spec.checkpoint_dir}.")

    run_seed = config.seed + 10_000 * artifacts.config.n_chains
    prior_references = _prepare_prior_reference_bundle(
        artifacts.dataset,
        n_samples=sampling.num_samples,
        id_seed=run_seed + 1,
        ood_seed=config.seed + 1,
    )
    posterior_references: dict[str, PosteriorReferenceBundle] = {}
    if sampling.prompt_len > 0 and config.n_prompts > 0:
        posterior_references["in_distribution"] = _prepare_posterior_reference_bundle(
            artifacts.dataset,
            prompt_source="in_distribution",
            sampling=sampling,
            n_prompts=config.n_prompts,
            prompt_seed=run_seed + 1_000,
            baseline_seed=run_seed + 2_000,
        )
        posterior_references["out_of_distribution"] = (
            _prepare_posterior_reference_bundle(
                artifacts.dataset,
                prompt_source="out_of_distribution",
                sampling=sampling,
                n_prompts=config.n_prompts,
                prompt_seed=config.seed + 3_000,
                baseline_seed=config.seed + 4_000,
            )
        )

    run_output_dir = runs_output_root / run_spec.run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_samples_dir = run_output_dir / "samples"
    final_prior_samples: dict[str, np.ndarray] | None = None
    final_posterior_samples: dict[str, dict[str, np.ndarray]] = {}
    dynamics_rows: list[dict[str, float | int]] = []
    per_prompt_rows: list[dict[str, float | int | str]] = []
    for checkpoint_path in checkpoint_paths:
        step = _step_from_checkpoint_name(checkpoint_path)
        artifacts.model.load_state_dict(
            load_markov_state_dict(checkpoint_path, artifacts.device)
        )
        artifacts.model.eval()

        base_seed = config.seed + 10_000 * artifacts.config.n_chains + step
        prior_metrics, prior_samples = _compute_prior_metrics(
            model=artifacts.model,
            dataset=artifacts.dataset,
            sampling=sampling,
            n_projections=config.n_projections,
            chunk_size=config.chunk_size,
            seed=base_seed,
            references=prior_references,
        )
        _save_predictive_samples(
            run_samples_dir / f"step{step:08d}__source_prior.npz",
            **prior_samples,
            step=step,
            prompt_source="prior",
            n_chains=artifacts.config.n_chains,
        )
        final_prior_samples = prior_samples
        row: dict[str, float | int] = {
            "step": step,
            "n_chains": artifacts.config.n_chains,
            **prior_metrics,
        }

        if sampling.prompt_len > 0 and config.n_prompts > 0:
            for prompt_index, prompt_source in enumerate(_PROMPT_SOURCES):
                posterior_metrics, prompt_rows, posterior_samples = (
                    _compute_posterior_metrics(
                        model=artifacts.model,
                        dataset=artifacts.dataset,
                        sampling=sampling,
                        references=posterior_references[prompt_source],
                        n_projections=config.n_projections,
                        chunk_size=config.chunk_size,
                        seed=base_seed + 100 * (prompt_index + 1),
                    )
                )
                _save_predictive_samples(
                    run_samples_dir
                    / f"step{step:08d}__source_data_{prompt_source}.npz",
                    **posterior_samples,
                    step=step,
                    prompt_source=f"data_{prompt_source}",
                    n_chains=artifacts.config.n_chains,
                )
                final_posterior_samples[prompt_source] = posterior_samples
                row.update(_with_prompt_suffix(posterior_metrics, prompt_source))
                for prompt_row in prompt_rows:
                    per_prompt_rows.append({"step": step, **prompt_row})

        dynamics_rows.append(row)
    dynamics_df = pd.DataFrame(dynamics_rows).sort_values("step")
    dynamics_df.to_csv(run_output_dir / "metrics.csv", index=False)
    plot_distribution_distance_dynamics(
        dynamics_df,
        run_output_dir / "dynamics_combined_prior.png",
        mode="prior",
    )
    plot_distribution_distance_dynamics(
        dynamics_df,
        run_output_dir / "dynamics_combined_prior_logx.png",
        mode="prior",
        log_xscale=True,
    )

    has_posterior = (
        sampling.prompt_len > 0
        and "ed_vs_baseline_in_distribution_from_prompts_in_distribution"
        in dynamics_df.columns
    )
    if has_posterior:
        plot_distribution_distance_dynamics(
            dynamics_df,
            run_output_dir / "dynamics_combined_posterior.png",
            mode="posterior",
        )
        plot_distribution_distance_dynamics(
            dynamics_df,
            run_output_dir / "dynamics_combined_posterior_logx.png",
            mode="posterior",
            log_xscale=True,
        )

    if per_prompt_rows:
        pd.DataFrame(per_prompt_rows).to_csv(
            run_output_dir / "per_prompt_metrics.csv",
            index=False,
        )

    final_row = dynamics_df.iloc[-1]
    final_step = int(final_row["step"])
    assert final_prior_samples is not None
    _save_predictive_samples(
        sweep_samples_dir / f"M{artifacts.config.n_chains}_prior.npz",
        **final_prior_samples,
        step=final_step,
        prompt_source="prior",
        n_chains=artifacts.config.n_chains,
    )
    for prompt_source in _PROMPT_SOURCES:
        if prompt_source in final_posterior_samples:
            _save_predictive_samples(
                sweep_samples_dir
                / f"M{artifacts.config.n_chains}_{prompt_source}_L{sampling.prompt_len}.npz",
                **final_posterior_samples[prompt_source],
                step=final_step,
                prompt_source=prompt_source,
                n_chains=artifacts.config.n_chains,
            )

    sweep_rows = _build_sweep_rows(
        final_row=final_row,
        run_name=run_spec.run_name,
        n_chains=artifacts.config.n_chains,
        prompt_length=sampling.prompt_len,
        n_samples=sampling.num_samples,
        n_prompts=config.n_prompts if sampling.prompt_len > 0 else 0,
    )
    return artifacts.config.n_chains, sweep_rows


def main(config: SweepConfig) -> None:
    """Compute Markov ED/SW metrics and render the aggregate plots."""
    config.validate()
    device = _resolve_device(config.device)
    run_specs = _discover_run_specs(config)

    output_dir = Path(config.output_dir)
    runs_output_root = output_dir / "runs"
    sweep_samples_dir = output_dir / "samples"
    runs_output_root.mkdir(parents=True, exist_ok=True)
    sweep_samples_dir.mkdir(parents=True, exist_ok=True)

    seen_n_chains: dict[int, str] = {}
    sweep_rows: list[dict[str, float | int | str]] = []
    for run_spec in run_specs:
        n_chains, run_rows = _analyze_run(
            run_spec,
            config=config,
            device=device,
            runs_output_root=runs_output_root,
            sweep_samples_dir=sweep_samples_dir,
        )
        if n_chains in seen_n_chains:
            raise ValueError(
                f"Multiple runs discovered for n_chains={n_chains}: "
                f"{seen_n_chains[n_chains]} and {run_spec.run_name}. "
                "Filter with --run-name-contains to pick one."
            )
        seen_n_chains[n_chains] = run_spec.run_name
        sweep_rows.extend(run_rows)

    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        by=["prompt_length", "n_chains", "prompt_source"]
    )
    sweep_df.to_csv(output_dir / "metrics.csv", index=False)

    plot_distribution_distance_sweep(
        sweep_df,
        output_dir / "sweep_combined_prior.png",
        prompt_length=0,
    )
    if config.prompt_len > 0 and config.n_prompts > 0:
        plot_distribution_distance_sweep(
            sweep_df,
            output_dir / "sweep_combined_posterior.png",
            prompt_length=config.prompt_len,
        )


if __name__ == "__main__":
    main(tyro.cli(SweepConfig))
