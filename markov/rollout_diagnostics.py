"""Rollout-length and rollout-count diagnostics for 1-Markov Predictive Monte Carlo.

Mirrors balls_and_urns/rollout_diagnostics.py on the paper's Markov
checkpoints: rollout state trajectories are generated once, the running
Laplace-smoothed transition-matrix estimate is evaluated on a rollout-length
grid, and the shared analysis (recovery error vs N, drift diagnostic vs N,
recovery error vs R with a Monte Carlo floor) is read off the same
trajectories. The paper's context length (1024) already covers rollouts to
N = 500, so no context override is involved.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.pmc_diagnostics import (
    count_curves,
    drift_curves,
    plot_counts,
    plot_drift,
    plot_recovery,
    recovery_curves,
)
from markov.analysis_common import load_trained_markov_artifacts
from markov.baselines import sample_bi_inf_posterior, sample_bi_ret_posterior
from markov.predictive_monte_carlo import (
    _estimate_transition_matrices_from_rollouts,
    predictive_monte_carlo_transition_matrix,
)
from markov.sweep_analysis import _sample_prompt_batch


@dataclass
class MarkovDiagnosticsConfig:
    checkpoint_path: str
    config_path: str | None = None
    output_dir: str = "outputs/markov_rollout_diagnostics"

    # rollout geometry
    steps: int = 500
    num_rollouts: int = 1024
    length_grid: list[int] = field(
        default_factory=lambda: [8, 16, 32, 64, 128, 256, 384, 500]
    )

    # prompt sources: prior (BOS only) and/or one in-distribution prompt
    prompt_sources: list[str] = field(default_factory=lambda: ["prior", "posterior"])
    prompt_length: int = 8
    anchor_length: int = 16  # prior drift anchor: estimate after this many states

    # rollout-count panel
    count_grid: list[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024]
    )
    n_resamples: int = 50

    n_projections: int = 100
    chunk_size: int = 128
    smoothing: float = 1.0
    temperature: float = 1.0
    seed: int = 42

    def validate(self) -> None:
        if self.steps < 1 or self.num_rollouts < 1:
            raise ValueError("steps and num_rollouts must be positive.")
        if max(self.length_grid) > self.steps:
            raise ValueError("length_grid entries cannot exceed steps.")
        if max(self.count_grid) > self.num_rollouts:
            raise ValueError("count_grid entries cannot exceed num_rollouts.")
        if self.anchor_length not in self.length_grid:
            raise ValueError("anchor_length must be a length_grid entry.")
        unknown = set(self.prompt_sources) - {"prior", "posterior"}
        if unknown:
            raise ValueError(f"Unknown prompt sources: {sorted(unknown)}")


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
    )
    model = artifacts.model
    dataset = artifacts.dataset
    pool = artifacts.transition_matrices.detach().cpu().float()
    k = int(dataset.k)
    alpha = torch.ones(k, k)
    grid = sorted(config.length_grid)
    free_parameters = k * (k - 1)

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
                generalising = sample_bi_inf_posterior(
                    prompt_long, alpha, r
                ).reshape(r, -1)
                memorising = sample_bi_ret_posterior(
                    prompt_long, pool, r
                ).reshape(r, -1)
                return {
                    "baseline_generalising": generalising.numpy(),
                    "baseline_memorising": memorising.numpy(),
                }

        baselines = fresh_baseline_sampler(config.num_rollouts)

        recovery = recovery_curves(estimates, baselines, grid, config.n_projections)
        recovery.to_csv(output_dir / f"{source}_recovery_vs_length.csv", index=False)
        plot_recovery(
            recovery,
            output_dir / f"{source}_recovery_vs_length",
            f"1-Markov {source}: recovery error vs rollout length "
            f"(R={config.num_rollouts})",
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
        plot_drift(
            drift,
            per_path,
            grid,
            output_dir / f"{source}_drift_vs_length",
            f"1-Markov {source}: drift from {anchor_label}",
        )

        counts_frame = count_curves(
            estimates[:, -1, :],
            baselines,
            fresh_baseline_sampler,
            sorted(config.count_grid),
            config.n_resamples,
            config.n_projections,
            rng,
        )
        counts_frame.to_csv(output_dir / f"{source}_recovery_vs_count.csv", index=False)
        plot_counts(
            counts_frame,
            output_dir / f"{source}_recovery_vs_count",
            f"1-Markov {source}: recovery error vs rollout count (N={grid[-1]})",
        )
        print(f"[{source}] diagnostics written to {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(MarkovDiagnosticsConfig))
