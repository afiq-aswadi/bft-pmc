"""Rollout-length and rollout-count diagnostics for linear-regression Predictive Monte Carlo.

The linear-regression counterpart of balls_and_urns/rollout_diagnostics.py and
markov/rollout_diagnostics.py. For one checkpoint, rollouts are generated once
with the sweep's own sampler (predictive_monte_carlo_beta_chunked: x ~ N(0, 1)
per coordinate, y drawn from the model), and the least-squares weights on the
prompt plus the first N generated pairs, the estimator that sampler solves, are
read off a rollout-length grid:
  1. recovery error (energy distance, sliced Wasserstein-1) against the
     generalising (Gaussian prior or ridge posterior) and memorising (task pool)
     baselines as a function of rollout length N at fixed R;
  2. drift vs N from the prompt-only least-squares estimate (posterior) or each
     rollout's own estimate at N = anchor_length (prior), divided by the weight
     dimension;
  3. recovery error as a function of rollout count R at the final N, alongside a
     Monte Carlo floor from fresh baseline draws of the same size.

A rollout of N pairs after an n-pair prompt occupies 2(n + N) token positions,
so n + N may not exceed half the model context; main() checks this before
generating anything. --n-ctx-override builds the model with a longer context
(see pfn_transformerlens.checkpointing.load_checkpoint) to roll out past the
trained one. The output directory holds the same files as the other
families, with {source}_trajectories.npz storing the (x, y) pairs, and
<run>_rollouts.npz feeding the jellyfish plot.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.pmc_diagnostics import (
    check_rollouts_fit_context,
    count_curves_with_resamples,
    on_isolated_torch_stream,
    drift_curves,
    plot_counts,
    plot_drift,
    plot_recovery,
    record_figure,
    recovery_curves,
    save_path_stability_bundle,
    write_run_record,
)
from linear_regression.analysis.data import load_run_info
from linear_regression.analysis.metrics import (
    compute_dmmse_weights,
    gaussian_posterior_params,
    sample_dmmse_posterior,
    sample_ridge_posterior,
)
from linear_regression.predictive_monte_carlo import (
    predictive_monte_carlo_beta_chunked,
)
from linear_regression.priors import DiscretePrior
from models.pos_encoding import positional_encoding_of
from pfn_transformerlens.model.PFN import SupervisedPFN


@dataclass
class LinearRegressionDiagnosticsConfig:
    checkpoint_path: str
    output_dir: str = "outputs/lr_rollout_diagnostics"
    noise_std: float = 0.5

    # rollout geometry: the camera-ready PMC setting, a 32-pair prompt and 223
    # generated pairs, which fills the 256 pairs of a 512-token context
    steps: int = 223
    num_rollouts: int = 1024
    length_grid: list[int] = field(
        default_factory=lambda: [8, 16, 32, 64, 128, 223]
    )
    # token context to build the model with; rolling out n + N pairs needs
    # 2(n + N) tokens, e.g. 1064 for N=500 after a 32-pair prompt
    n_ctx_override: int | None = None

    # prompt sources: prior (no prompt) and/or one in-distribution prompt
    prompt_sources: list[str] = field(default_factory=lambda: ["prior", "posterior"])
    prompt_length: int = 32
    anchor_length: int = 16  # prior drift anchor: estimate after this many pairs

    # rollout-count panel
    count_grid: list[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024]
    )
    n_resamples: int = 50
    # Rollout-count panel: draw an independent pool of rollouts per repeat
    # rather than resampling one saved pool. The repeats then share no
    # rollouts, so the spread is Monte Carlo error rather than subset-selection
    # noise, at roughly n_resamples times the rollout cost. Off by default:
    # only the figures that report that spread need it.
    independent_count_pools: bool = False

    # jellyfish bundle: one curve per realisation of the observed data, as in
    # TabMGP Figure 8, each realisation its own prompt and rollouts.
    jellyfish_prompts: int = 20
    jellyfish_rollouts: int = 128

    n_projections: int = 100
    chunk_size: int = 128
    temperature: float = 1.0
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        if self.steps < 1 or self.num_rollouts < 1:
            raise ValueError("steps and num_rollouts must be positive.")
        if self.jellyfish_prompts < 1 or self.jellyfish_rollouts < 1:
            raise ValueError(
                "jellyfish_prompts and jellyfish_rollouts must be positive."
            )
        if max(self.length_grid) > self.steps:
            raise ValueError("length_grid entries cannot exceed steps.")
        if max(self.count_grid) > self.num_rollouts:
            raise ValueError("count_grid entries cannot exceed num_rollouts.")
        if self.anchor_length not in self.length_grid:
            raise ValueError("anchor_length must be a length_grid entry.")
        if self.noise_std <= 0:
            raise ValueError("noise_std must be positive.")
        unknown = set(self.prompt_sources) - {"prior", "posterior"}
        if unknown:
            raise ValueError(f"Unknown prompt sources: {sorted(unknown)}")


def sample_in_distribution_prompt(
    tasks: torch.Tensor,
    prompt_length: int,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """An n-pair prompt from one task of the training pool, like the sweep's discrete prompts.

    Returns (xs, ys, task_index) with xs of shape (prompt_length, dim).
    """
    task_index = int(rng.integers(tasks.shape[0]))
    xs = torch.randn(prompt_length, tasks.shape[1])
    ys = xs @ tasks[task_index].cpu() + noise_std * torch.randn(prompt_length)
    return xs, ys, task_index


def jellyfish_rollouts(
    model: SupervisedPFN,
    *,
    tasks: torch.Tensor,
    config: LinearRegressionDiagnosticsConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rollouts from several prompt realisations, for the jellyfish plot.

    TabMGP draws the path-stability curve once per realisation of the observed
    data (their Figure 8), so each realisation here is a fresh prompt from a
    fresh task. Returns x and y of shape (realisations, rollouts, prompt_length
    + steps, dim) and (realisations, rollouts, prompt_length + steps), plus the
    task behind each realisation.
    """
    all_xs: list[np.ndarray] = []
    all_ys: list[np.ndarray] = []
    task_indices: list[int] = []
    for _ in range(config.jellyfish_prompts):
        prompt_x, prompt_y, task_index = sample_in_distribution_prompt(
            tasks, config.prompt_length, config.noise_std, rng
        )
        xs, ys = rollout_pairs(
            model,
            steps=config.steps,
            num_rollouts=config.jellyfish_rollouts,
            prompt_x=prompt_x,
            prompt_y=prompt_y,
            temperature=config.temperature,
            chunk_size=min(config.chunk_size, config.jellyfish_rollouts),
        )
        all_xs.append(xs)
        all_ys.append(ys)
        task_indices.append(task_index)
    return (
        np.stack(all_xs),
        np.stack(all_ys),
        np.asarray(task_indices, dtype=np.int32),
    )


def rollout_pairs(
    model: SupervisedPFN,
    *,
    steps: int,
    num_rollouts: int,
    prompt_x: torch.Tensor | None,
    prompt_y: torch.Tensor | None,
    temperature: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate rollouts and return their (x, y) pairs, prompt included.

    Returns float32 arrays of shape (num_rollouts, prompt_len + steps, dim) and
    (num_rollouts, prompt_len + steps).
    """
    device = next(model.parameters()).device
    _, xs, ys = predictive_monte_carlo_beta_chunked(
        model=model,
        x_distribution=torch.distributions.Normal(0.0, 1.0),
        forward_recursion_steps=steps,
        forward_recursion_samples=num_rollouts,
        chunk_size=chunk_size,
        temperature=temperature,
        init_x=None if prompt_x is None else prompt_x.to(device),
        init_y=None if prompt_y is None else prompt_y.to(device),
        save_rollouts=True,
    )
    return xs.astype(np.float32), ys.astype(np.float32)


def least_squares(xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    """Least-squares weights for (..., pairs, dim) inputs, in float64.

    The rank-revealing driver returns the minimum-norm solution when a prefix
    has fewer pairs than weights.
    """
    solution = torch.linalg.lstsq(
        xs.double(), ys.double().unsqueeze(-1), driver="gelsd"
    ).solution
    return solution.squeeze(-1).numpy().astype(np.float32)


def running_least_squares(
    xs: np.ndarray, ys: np.ndarray, prompt_length: int, grid: list[int]
) -> np.ndarray:
    """Least-squares weights from the prompt plus the first N generated pairs.

    Returns float32 array of shape (num_rollouts, len(grid), dim).
    """
    x_tensor = torch.from_numpy(xs)
    y_tensor = torch.from_numpy(ys)
    estimates = np.empty((xs.shape[0], len(grid), xs.shape[2]), dtype=np.float32)
    for j, n in enumerate(grid):
        upto = prompt_length + n
        estimates[:, j, :] = least_squares(x_tensor[:, :upto], y_tensor[:, :upto])
    return estimates


def analyze_source(
    *,
    source: str,
    xs: np.ndarray,
    ys: np.ndarray,
    prompt_x: torch.Tensor | None,
    prompt_y: torch.Tensor | None,
    prior: DiscretePrior,
    config: LinearRegressionDiagnosticsConfig,
    output_dir: Path,
    rng: np.random.Generator,
    fresh_model_sampler=None,
) -> None:
    dimension = int(prior.task_size)
    prompt_length = 0 if prompt_x is None else int(prompt_x.shape[0])
    grid = sorted(config.length_grid)
    noise_variance = config.noise_std**2
    pool = prior.tasks.cpu().numpy()

    if prompt_x is None or prompt_y is None:

        def fresh_baseline_sampler(r: int) -> dict[str, np.ndarray]:
            return {
                "baseline_generalising": np.random.randn(r, dimension),
                "baseline_memorising": pool[
                    np.random.randint(0, prior.num_tasks, size=r)
                ],
            }

        posterior_parameters: dict[str, np.ndarray] = {}
    else:
        context_x, context_y = prompt_x, prompt_y

        def fresh_baseline_sampler(r: int) -> dict[str, np.ndarray]:
            return {
                "baseline_generalising": sample_ridge_posterior(
                    context_x, context_y, noise_variance, r
                ),
                "baseline_memorising": sample_dmmse_posterior(
                    context_x, context_y, prior, noise_variance, r
                ),
            }

        mean, covariance = gaussian_posterior_params(
            context_x, context_y, noise_variance
        )
        posterior_parameters = {
            "generalising_mean": mean,
            "generalising_covariance": covariance,
            "memorising_weights": compute_dmmse_weights(
                context_x, context_y, prior, noise_variance
            ),
        }

    baselines = fresh_baseline_sampler(config.num_rollouts)
    np.savez_compressed(
        output_dir / f"{source}_baselines.npz",
        **{name: np.asarray(samples) for name, samples in baselines.items()},
        task_pool=pool,
        noise_std=np.asarray(config.noise_std),
        **posterior_parameters,
    )

    estimates = running_least_squares(xs, ys, prompt_length, grid)

    recovery = recovery_curves(estimates, baselines, grid, config.n_projections)
    recovery.to_csv(output_dir / f"{source}_recovery_vs_length.csv", index=False)
    recovery_title = (
        f"linear regression {source}: recovery error vs rollout length "
        f"(R={config.num_rollouts})"
    )
    plot_recovery(
        recovery,
        output_dir / f"{source}_recovery_vs_length",
        recovery_title,
    )
    record_figure(
        output_dir,
        f"{source}_recovery_vs_length",
        kind="recovery",
        title=recovery_title,
        csv=f"{source}_recovery_vs_length.csv",
    )

    if prompt_x is None or prompt_y is None:
        anchor_stat = estimates[:, grid.index(config.anchor_length), :]
        anchor_label = f"path estimate at N={config.anchor_length}"
    else:
        anchor_stat = least_squares(prompt_x, prompt_y)
        anchor_label = "prompt-only estimate"
    drift, per_path = drift_curves(estimates, grid, anchor_stat, scale=dimension)
    drift.to_csv(output_dir / f"{source}_drift_vs_length.csv", index=False)
    np.savez_compressed(
        output_dir / f"{source}_drift_paths.npz",
        rollout_lengths=np.asarray(grid),
        per_path_drift=per_path,
        estimates_at_grid=estimates,
        anchor_stat=anchor_stat,
    )
    drift_title = f"linear regression {source}: drift from {anchor_label}"
    plot_drift(
        drift,
        per_path,
        grid,
        output_dir / f"{source}_drift_vs_length",
        drift_title,
    )
    record_figure(
        output_dir,
        f"{source}_drift_vs_length",
        kind="drift",
        title=drift_title,
        csv=f"{source}_drift_vs_length.csv",
        paths=f"{source}_drift_paths.npz",
    )

    counts_frame, resamples = count_curves_with_resamples(
        estimates[:, -1, :],
        baselines,
        fresh_baseline_sampler,
        sorted(config.count_grid),
        config.n_resamples,
        config.n_projections,
        rng,
        fresh_model_sampler=fresh_model_sampler,
    )
    counts_frame.to_csv(output_dir / f"{source}_recovery_vs_count.csv", index=False)
    resamples.to_csv(
        output_dir / f"{source}_recovery_vs_count_resamples.csv", index=False
    )
    counts_title = (
        f"linear regression {source}: recovery error vs rollout count (N={grid[-1]})"
    )
    plot_counts(
        counts_frame,
        output_dir / f"{source}_recovery_vs_count",
        counts_title,
    )
    record_figure(
        output_dir,
        f"{source}_recovery_vs_count",
        kind="counts",
        title=counts_title,
        csv=f"{source}_recovery_vs_count.csv",
        resamples=f"{source}_recovery_vs_count_resamples.csv",
    )


def main(config: LinearRegressionDiagnosticsConfig) -> None:
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_info = load_run_info(
        Path(config.checkpoint_path),
        device=None if config.device == "auto" else config.device,
        n_ctx_override=config.n_ctx_override,
    )
    model = run_info["model"]
    assert isinstance(model, SupervisedPFN)
    context_length = int(model.transformer.cfg.n_ctx)
    check_rollouts_fit_context(
        steps=config.steps,
        prompt_length=config.prompt_length,
        prompt_sources=config.prompt_sources,
        context_length=context_length // 2,
        unit="(x, y) pairs",
    )
    prior = DiscretePrior(
        task_size=int(run_info["task_size"]), tasks=run_info["tasks"], device="cpu"
    )
    dimension = int(prior.task_size)
    write_run_record(
        output_dir,
        asdict(config),
        inputs=[config.checkpoint_path],
        family="lr",
        device=str(next(model.parameters()).device),
        dimension=dimension,
        latent_dimension=dimension,
        drift_scale=dimension,
        num_tasks=int(run_info["num_tasks"]),
        model_context=context_length,
        pair_capacity=context_length // 2,
        pos_encoding=positional_encoding_of(model.config),
        checkpoint_metadata=asdict(run_info["metadata"]),
    )

    for source in config.prompt_sources:
        if source == "prior":
            task_index = -1
            prompt_x = prompt_y = None
        else:
            prompt_x, prompt_y, task_index = sample_in_distribution_prompt(
                prior.tasks, config.prompt_length, config.noise_std, rng
            )

        xs, ys = rollout_pairs(
            model,
            steps=config.steps,
            num_rollouts=config.num_rollouts,
            prompt_x=prompt_x,
            prompt_y=prompt_y,
            temperature=config.temperature,
            chunk_size=config.chunk_size,
        )
        np.savez_compressed(
            output_dir / f"{source}_trajectories.npz",
            rollout_x=xs,
            rollout_y=ys,
            prompt_x=(
                np.empty((0, dimension), dtype=np.float32)
                if prompt_x is None
                else prompt_x.numpy()
            ),
            prompt_y=(
                np.empty((0,), dtype=np.float32)
                if prompt_y is None
                else prompt_y.numpy()
            ),
            task_index=np.asarray(task_index),
            steps=np.asarray(config.steps),
            checkpoint_path=np.asarray(str(config.checkpoint_path)),
        )
        def fresh_model_sampler(r: int, _px=prompt_x, _py=prompt_y) -> np.ndarray:
            # An independent pool of r rollouts, not a subset of the saved one.
            fresh_x, fresh_y = rollout_pairs(
                model,
                steps=config.steps,
                num_rollouts=r,
                prompt_x=_px,
                prompt_y=_py,
                temperature=config.temperature,
                chunk_size=config.chunk_size,
            )
            prompt_len = 0 if _px is None else int(_px.shape[0])
            return running_least_squares(
                fresh_x, fresh_y, prompt_len, sorted(config.length_grid)
            )[:, -1, :]

        analyze_source(
            source=source,
            xs=xs,
            ys=ys,
            prompt_x=prompt_x,
            prompt_y=prompt_y,
            prior=prior,
            config=config,
            output_dir=output_dir,
            rng=rng,
            fresh_model_sampler=(
                on_isolated_torch_stream(
                    fresh_model_sampler, config.seed + 10_000
                )
                if config.independent_count_pools
                else None
            ),
        )
        print(f"[{source}] diagnostics written to {output_dir}")

    # After every diagnostic, so the rollouts above keep their random stream:
    # the jellyfish bundle, one prompt realisation per curve.
    if "posterior" in config.prompt_sources:
        bundle_x, bundle_y, task_indices = jellyfish_rollouts(
            model,
            tasks=prior.tasks,
            config=config,
            rng=rng,
        )
        save_path_stability_bundle(
            output_dir,
            prompt_length=config.prompt_length,
            checkpoint_path=config.checkpoint_path,
            arrays={"rollout_x": bundle_x, "rollout_y": bundle_y},
            prompt_source="in_distribution",
            num_tasks=int(run_info["num_tasks"]),
            task_index=task_indices,
        )


if __name__ == "__main__":
    main(tyro.cli(LinearRegressionDiagnosticsConfig))
