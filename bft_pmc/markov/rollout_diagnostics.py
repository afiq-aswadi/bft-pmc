"""Rollout-length and rollout-count diagnostics for 1-Markov Predictive Monte Carlo.

Mirrors bft_pmc/balls_and_urns/rollout_diagnostics.py on the paper's Markov
checkpoints: rollout state trajectories are generated once, the running
Laplace-smoothed transition-matrix estimate is evaluated on a rollout-length
grid, and the shared analysis (recovery error vs N, drift diagnostic vs N,
recovery error vs R with a Monte Carlo floor) is read off the same
trajectories. Rollouts cannot exceed the model context: the prompt plus N
generated states must fit in seq_len. The camera-ready models (seq_len 256)
therefore stop at N = 248 after the default 8-state prompt, unless
--seq-len-override builds the model with a longer rotary table, which needs no
retraining and reuses the trained weights unchanged. --rope-scaling then picks
what happens past the trained context: "interpolate" rescales every position
into the trained range (safer at long N, but it moves the trained positions too,
so a whole figure must be regenerated together), "extrapolate" leaves positions
as trained (short rollouts reproduce exactly, long ones use unseen angles).
main() checks the geometry before generating anything.

The output directory holds the same files as the balls-and-urns diagnostics,
with {source}_baselines.npz carrying the chain pool rather than urns, so every
figure can be redrawn without the model (bft_pmc/analysis/replot_pmc_diagnostics.py),
and <run>_rollouts.npz feeding the jellyfish plot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from bft_pmc.analysis.pmc_diagnostics import (
    check_rollouts_fit_context,
    count_curves_with_resamples,
    drift_curves,
    on_isolated_torch_stream,
    plot_counts,
    plot_drift,
    plot_recovery,
    record_figure,
    recovery_curves,
    save_path_stability_bundle,
    write_run_record,
)
from bft_pmc.markov.analysis_common import load_trained_markov_artifacts
from bft_pmc.markov.baselines import sample_bi_inf_posterior, sample_bi_ret_posterior
from bft_pmc.markov.predictive_monte_carlo import (
    _estimate_transition_matrices_from_rollouts,
    predictive_monte_carlo_transition_matrix,
)
from bft_pmc.markov.sweep_analysis import _sample_prompt_batch


@dataclass
class MarkovDiagnosticsConfig:
    """Rollout-length and rollout-count diagnostics for one 1-Markov checkpoint."""

    checkpoint_path: str
    """Checkpoint to diagnose."""
    config_path: str | None = None
    """The run's resolved_config.yaml, written by train.py markov; it
    locates the training chain pool and fixes k. Required for the camera-ready
    runs."""
    output_dir: str = "outputs/markov_rollout_diagnostics"
    """Directory for the curves, figures, rollout bundles and run record."""
    steps: int = 500
    """Rollout length N: states generated after the prompt."""
    num_rollouts: int = 1024
    """Rollouts R per prompt source."""
    length_grid: list[int] = field(
        default_factory=lambda: [8, 16, 32, 64, 128, 256, 384, 500]
    )
    """Rollout lengths at which recovery error and drift are read off; at most
    steps."""
    seq_len_override: int | None = None
    """Build the model with a longer context than it was trained on, so
    rollouts can pass it; None keeps the trained seq_len. See
    rope_scaling."""
    rope_scaling: str = "interpolate"
    """Positions past the trained context: interpolate rescales every
    position into the trained range, extrapolate leaves positions as
    trained. Only matters with seq_len_override."""
    prompt_sources: list[str] = field(default_factory=lambda: ["prior", "posterior"])
    """prior (BOS only) and/or posterior (one in-distribution prompt)."""
    prompt_length: int = 8
    """States in the posterior prompt (the camera-ready runs pass 32)."""
    anchor_length: int = 16
    """Prior drift is measured from each rollout's own estimate at this length
    (a length_grid entry); posterior drift from the prompt-only estimate."""
    count_grid: list[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024]
    )
    """Rollout counts R of the recovery-against-count panel, read at the last
    length_grid entry."""
    n_resamples: int = 50
    """Subsets of rollouts drawn at each count below the full pool."""
    independent_count_pools: bool = False
    """Draw an independent pool of rollouts per repeat rather than resampling
    one saved pool. The repeats then share no rollouts, so the spread is Monte
    Carlo error rather than subset-selection noise, at roughly n_resamples
    times the rollout cost. Only the figures that report that spread need it."""
    jellyfish_prompts: int = 20
    """Prompt realisations in the path-stability (jellyfish) bundle, each with
    its own rollouts, as in TabMGP Figure 8."""
    jellyfish_rollouts: int = 128
    """Rollouts per jellyfish prompt."""
    n_projections: int = 100
    """Random projections for sliced Wasserstein."""
    chunk_size: int = 128
    """Rollouts generated per forward batch."""
    smoothing: float = 1.0
    """Laplace pseudo-count added to every transition count before each row is
    normalised into the PMC estimate."""
    temperature: float = 1.0
    """Sampling temperature; 1.0 samples the model's own predictive."""
    seed: int = 42
    """Seed for prompts, rollouts and reference draws."""

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
        unknown = set(self.prompt_sources) - {"prior", "posterior"}
        if unknown:
            raise ValueError(f"Unknown prompt sources: {sorted(unknown)}")
        if self.rope_scaling not in {"interpolate", "extrapolate"}:
            raise ValueError(
                f"Unknown rope_scaling {self.rope_scaling!r}; expected "
                "'interpolate' or 'extrapolate'."
            )


def jellyfish_rollouts(
    model,
    dataset,
    *,
    config: MarkovDiagnosticsConfig,
) -> np.ndarray:
    """Rollouts from several prompt realisations, for the jellyfish plot.

    TabMGP draws the path-stability curve once per realisation of the observed
    data (their Figure 8), so each realisation here is a fresh prompt from a
    fresh chain; the first repeats the posterior run's own prompt, which shares
    the seed. Returns (realisations, rollouts, prompt_length + steps) states.
    """
    prompts = _sample_prompt_batch(
        dataset,
        prompt_source="in_distribution",
        prompt_len=config.prompt_length,
        n_prompts=config.jellyfish_prompts,
        seed=config.seed,
    )
    realisations = [
        rollout_state_trajectories(
            model,
            dataset,
            steps=config.steps,
            num_rollouts=config.jellyfish_rollouts,
            prompt=prompt.detach().cpu(),
            temperature=config.temperature,
            smoothing=config.smoothing,
            chunk_size=min(config.chunk_size, config.jellyfish_rollouts),
        )
        for prompt in prompts
    ]
    return np.stack(realisations)


def rollout_state_trajectories(
    model,
    dataset,
    *,
    steps: int,
    num_rollouts: int,
    prompt: torch.Tensor | None,
    temperature: float,
    smoothing: float,
    chunk_size: int,
) -> np.ndarray:
    """Generate rollouts in chunks, returning state trajectories (BOS stripped).

    Returns int16 array of shape (num_rollouts, prompt_len + steps): prompt
    states (if any) followed by generated states.
    """
    chunks = []
    remaining = num_rollouts
    while remaining > 0:
        chunk = min(remaining, chunk_size)
        _, rollouts = predictive_monte_carlo_transition_matrix(
            model,
            dataset,
            forward_recursion_steps=steps,
            forward_recursion_samples=chunk,
            prompt=prompt,
            sample=True,
            temperature=temperature,
            smoothing=smoothing,
            save_rollouts=True,
        )
        chunks.append(rollouts.astype(np.int16))
        remaining -= chunk
    return np.concatenate(chunks, axis=0)


def running_transition_estimates(
    trajectories: np.ndarray,
    *,
    k: int,
    prompt_length: int,
    grid: list[int],
    smoothing: float,
) -> np.ndarray:
    """Laplace-smoothed transition estimates over the first prompt_length + N states.

    Returns float32 array of shape (num_rollouts, len(grid), k * k), the
    flattened matrices the paper's metrics operate on.
    """
    rollouts = trajectories.shape[0]
    states = torch.from_numpy(trajectories.astype(np.int64))
    estimates = np.empty((rollouts, len(grid), k * k), dtype=np.float32)
    for j, n in enumerate(grid):
        matrices = _estimate_transition_matrices_from_rollouts(
            states[:, : prompt_length + n],
            k=k,
            smoothing=smoothing,
        )
        estimates[:, j, :] = matrices.reshape(rollouts, k * k).numpy()
    return estimates


def main(config: MarkovDiagnosticsConfig) -> None:
    config.validate()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_trained_markov_artifacts(
        config.config_path,
        Path(config.checkpoint_path),
        seq_len_override=config.seq_len_override,
        rope_scaling=config.rope_scaling,
    )
    model = artifacts.model
    check_rollouts_fit_context(
        steps=config.steps,
        prompt_length=config.prompt_length,
        prompt_sources=config.prompt_sources,
        context_length=model.max_seq_len,
    )
    dataset = artifacts.dataset
    pool = artifacts.transition_matrices.detach().cpu().float()
    k = int(dataset.k)
    alpha = torch.ones(k, k)
    grid = sorted(config.length_grid)
    free_parameters = k * (k - 1)
    write_run_record(
        output_dir,
        asdict(config),
        inputs=[str(path) for path in artifacts.input_files],
        family="markov",
        device=str(next(model.parameters()).device),
        k=k,
        latent_dimension=free_parameters,
        drift_scale=free_parameters,
        num_chains=int(pool.shape[0]),
        model_context=int(model.max_seq_len),
        trained_context=int(artifacts.config.seq_len),
        rope_position_scale=float(model.position_scale),
        task_distribution_source=artifacts.task_distribution_source,
    )

    for source in config.prompt_sources:
        if source == "prior":
            prompt_tensor = None
            prompt_states = np.empty((0,), dtype=np.int16)
        else:
            prompt_batch = _sample_prompt_batch(
                dataset,
                prompt_source="in_distribution",
                prompt_len=config.prompt_length,
                n_prompts=1,
                seed=config.seed,
            )
            prompt_tensor = prompt_batch[0].detach().cpu()
            prompt_states = prompt_tensor.to(torch.int16).numpy()

        trajectories = rollout_state_trajectories(
            model,
            dataset,
            steps=config.steps,
            num_rollouts=config.num_rollouts,
            prompt=prompt_tensor,
            temperature=config.temperature,
            smoothing=config.smoothing,
            chunk_size=config.chunk_size,
        )
        np.savez_compressed(
            output_dir / f"{source}_trajectories.npz",
            trajectories=trajectories,
            prompt_tokens=prompt_states,
            k=np.asarray(k),
            steps=np.asarray(config.steps),
            checkpoint_path=np.asarray(str(config.checkpoint_path)),
        )

        prompt_length = int(prompt_states.shape[0])
        estimates = running_transition_estimates(
            trajectories,
            k=k,
            prompt_length=prompt_length,
            grid=grid,
            smoothing=config.smoothing,
        )

        if source == "prior":

            def fresh_baseline_sampler(r: int) -> dict[str, np.ndarray]:
                generalising = (
                    torch.distributions.Dirichlet(alpha).sample((r,)).reshape(r, -1)
                )
                pool_index = torch.randint(pool.shape[0], (r,))
                memorising = pool[pool_index].reshape(r, -1)
                return {
                    "baseline_generalising": generalising.numpy(),
                    "baseline_memorising": memorising.numpy(),
                }

        else:
            prompt_long = torch.from_numpy(prompt_states.astype(np.int64))

            def fresh_baseline_sampler(r: int) -> dict[str, np.ndarray]:
                generalising = sample_bi_inf_posterior(prompt_long, alpha, r).reshape(
                    r, -1
                )
                memorising = sample_bi_ret_posterior(prompt_long, pool, r).reshape(
                    r, -1
                )
                return {
                    "baseline_generalising": generalising.numpy(),
                    "baseline_memorising": memorising.numpy(),
                }

        baselines = fresh_baseline_sampler(config.num_rollouts)
        np.savez_compressed(
            output_dir / f"{source}_baselines.npz",
            **baselines,
            task_pool=pool.numpy(),
            alpha=alpha.numpy(),
            prompt_tokens=prompt_states,
        )

        recovery = recovery_curves(estimates, baselines, grid, config.n_projections)
        recovery.to_csv(output_dir / f"{source}_recovery_vs_length.csv", index=False)
        recovery_title = (
            f"1-Markov {source}: recovery error vs rollout length "
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

        if source == "prior":
            anchor_index = grid.index(config.anchor_length)
            anchor_stat = estimates[:, anchor_index, :]
            anchor_label = f"path estimate at N={config.anchor_length}"
        else:
            anchor_stat = (
                _estimate_transition_matrices_from_rollouts(
                    torch.from_numpy(prompt_states.astype(np.int64)).unsqueeze(0),
                    k=k,
                    smoothing=config.smoothing,
                )
                .reshape(-1)
                .numpy()
                .astype(np.float32)
            )
            anchor_label = "prompt-only estimate"
        drift, per_path = drift_curves(
            estimates, grid, anchor_stat, scale=free_parameters
        )
        drift.to_csv(output_dir / f"{source}_drift_vs_length.csv", index=False)
        np.savez_compressed(
            output_dir / f"{source}_drift_paths.npz",
            rollout_lengths=np.asarray(grid),
            per_path_drift=per_path,
            estimates_at_grid=estimates,
            anchor_stat=anchor_stat,
        )
        drift_title = f"1-Markov {source}: drift from {anchor_label}"
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

        def fresh_model_sampler(r: int, _prompt=prompt_tensor) -> np.ndarray:
            # An independent pool of r rollouts, not a subset of the saved one.
            fresh = rollout_state_trajectories(
                model,
                dataset,
                steps=config.steps,
                num_rollouts=r,
                prompt=_prompt,
                temperature=config.temperature,
                smoothing=config.smoothing,
                chunk_size=config.chunk_size,
            )
            return running_transition_estimates(
                fresh,
                k=k,
                prompt_length=prompt_length,
                grid=grid,
                smoothing=config.smoothing,
            )[:, -1, :]

        counts_frame, resamples = count_curves_with_resamples(
            estimates[:, -1, :],
            baselines,
            fresh_baseline_sampler,
            sorted(config.count_grid),
            config.n_resamples,
            config.n_projections,
            rng,
            fresh_model_sampler=(
                on_isolated_torch_stream(fresh_model_sampler, config.seed + 10_000)
                if config.independent_count_pools
                else None
            ),
        )
        counts_frame.to_csv(output_dir / f"{source}_recovery_vs_count.csv", index=False)
        resamples.to_csv(
            output_dir / f"{source}_recovery_vs_count_resamples.csv", index=False
        )
        counts_title = (
            f"1-Markov {source}: recovery error vs rollout count (N={grid[-1]})"
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
        print(f"[{source}] diagnostics written to {output_dir}")

    # After every diagnostic, so the rollouts above keep their random stream:
    # the jellyfish bundle, one prompt realisation per curve.
    if "posterior" in config.prompt_sources:
        realisations = jellyfish_rollouts(model, dataset, config=config)
        save_path_stability_bundle(
            output_dir,
            prompt_length=config.prompt_length,
            checkpoint_path=config.checkpoint_path,
            arrays={"rollout_states": realisations},
            prompt_source="in_distribution",
            n_chains=int(pool.shape[0]),
        )


if __name__ == "__main__":
    main(tyro.cli(MarkovDiagnosticsConfig))
