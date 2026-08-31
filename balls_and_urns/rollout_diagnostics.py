"""Rollout-length and rollout-count diagnostics for BAU Predictive Monte Carlo.

For one checkpoint, generates rollout trajectories once, then reads off:
  1. recovery error (energy distance, sliced Wasserstein-1) against the
     analytic baselines as a function of rollout length N at fixed R;
  2. the path-stability drift diagnostic: expected scaled L1 norm between
     the anchored estimate and the depth-N estimate, with per-path traces;
  3. recovery error as a function of rollout count R at fixed N, by
     subsampling the rollout set, alongside a Monte Carlo floor from fresh
     baseline draws of the same size.

Everything is saved under --output-dir: trajectories and per-path drift as
.npz, curves as .csv, figures as .png/.pdf.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.checkpoints import load_run_info
from analysis.pmc_diagnostics import (
    count_curves,
    drift_curves,
    plot_counts,
    plot_drift,
    plot_recovery,
    recovery_curves,
)
from balls_and_urns.analysis import precompute_baseline_samples
from balls_and_urns.baselines import (
    sample_generalising_posterior,
    sample_memorising_posterior,
)
from pfn_transformerlens.model.PFN import UnsupervisedPFN


@dataclass
class DiagnosticsConfig:
    checkpoint_path: str
    output_dir: str = "outputs/bau_rollout_diagnostics"
    alpha_value: float = 1.0

    # rollout geometry
    steps: int = 500
    num_rollouts: int = 1024
    length_grid: list[int] = field(
        default_factory=lambda: [8, 16, 32, 64, 128, 256, 384, 500]
    )
    n_ctx_override: int | None = 1024

    # prompt sources: prior (BOS only) and/or one in-distribution prompt
    prompt_sources: list[str] = field(default_factory=lambda: ["prior", "posterior"])
    prompt_length: int = 16
    anchor_length: int = 16  # prior drift anchor: estimate after this many tokens

    # rollout-count panel
    count_grid: list[int] = field(
        default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024]
    )
    n_resamples: int = 50

    n_projections: int = 100
    chunk_size: int = 256
    temperature: float = 1.0
    seed: int = 42
    device: str = "auto"

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


def rollout_trajectories(
    model: UnsupervisedPFN,
    *,
    steps: int,
    num_rollouts: int,
    bos_token: int,
    prompt: torch.Tensor | None,
    temperature: float,
    chunk_size: int,
) -> np.ndarray:
    """Generate rollouts and return data-token trajectories (BOS stripped).

    Returns int16 array of shape (num_rollouts, prompt_len + steps): the
    prompt tokens (if any) followed by the generated tokens, exactly the
    token sequence the paper's estimator averages over.
    """
    device = next(model.parameters()).device
    parts = [torch.tensor([bos_token], dtype=torch.float32, device=device)]
    if prompt is not None:
        parts.append(prompt.to(device=device, dtype=torch.float32))
    full_prompt = torch.cat(parts)

    chunks = []
    remaining = num_rollouts
    with torch.no_grad():
        while remaining > 0:
            chunk = min(remaining, chunk_size)
            generated = model.generate(
                num_generate=steps,
                prompt=full_prompt,
                sample=True,
                temperature=temperature,
                num_rollouts=chunk,
            )
            if generated.dim() == 1:
                generated = generated.unsqueeze(0)
            chunks.append(generated[:, 1:].to(torch.int16).cpu().numpy())
            remaining -= chunk
    return np.concatenate(chunks, axis=0)


def running_theta(
    trajectories: np.ndarray, vocab_size: int, prompt_length: int, grid: list[int]
) -> np.ndarray:
    """Empirical-frequency estimates over the first prompt_length + N tokens.

    Returns float32 array of shape (num_rollouts, len(grid), vocab_size).
    """
    rollouts, total_len = trajectories.shape
    one_hot = np.zeros((rollouts, total_len, vocab_size), dtype=np.int32)
    rows = np.arange(rollouts)[:, None]
    cols = np.arange(total_len)[None, :]
    one_hot[rows, cols, trajectories.astype(np.int64)] = 1
    counts = np.cumsum(one_hot, axis=1)

    thetas = np.empty((rollouts, len(grid), vocab_size), dtype=np.float32)
    for j, n in enumerate(grid):
        upto = prompt_length + n
        thetas[:, j, :] = counts[:, upto - 1, :] / float(upto)
    return thetas


def analyze_source(
    *,
    source: str,
    trajectories: np.ndarray,
    prompt_tokens: np.ndarray,
    alpha: torch.Tensor,
    theta_pool: torch.Tensor,
    config: DiagnosticsConfig,
    output_dir: Path,
    rng: np.random.Generator,
) -> None:
    vocab_size = int(alpha.shape[0])
    prompt_length = int(prompt_tokens.shape[0])
    grid = sorted(config.length_grid)

    if source == "prior":
        baselines = precompute_baseline_samples(
            config.num_rollouts, alpha, theta_pool, None
        )

        def fresh_baseline_sampler(r: int) -> dict[str, np.ndarray]:
            return precompute_baseline_samples(r, alpha, theta_pool, None)

    else:
        prompt_tensor = torch.from_numpy(prompt_tokens.astype(np.int64))

        def _posterior_samples(r: int) -> dict[str, np.ndarray]:
            return {
                "baseline_generalising": sample_generalising_posterior(
                    prompt_tensor, alpha, r
                ).numpy(),
                "baseline_memorising": sample_memorising_posterior(
                    prompt_tensor, theta_pool, r
                ).numpy(),
            }

        baselines = _posterior_samples(config.num_rollouts)
        fresh_baseline_sampler = _posterior_samples

    thetas = running_theta(trajectories, vocab_size, prompt_length, grid)

    recovery = recovery_curves(thetas, baselines, grid, config.n_projections)
    recovery.to_csv(output_dir / f"{source}_recovery_vs_length.csv", index=False)
    plot_recovery(
        recovery,
        output_dir / f"{source}_recovery_vs_length",
        f"{source}: recovery error vs rollout length (R={config.num_rollouts})",
    )

    if source == "prior":
        anchor_index = grid.index(config.anchor_length)
        anchor_theta = thetas[:, anchor_index, :]
        anchor_label = f"path estimate at N={config.anchor_length}"
    else:
        counts = np.bincount(prompt_tokens.astype(np.int64), minlength=vocab_size)
        anchor_theta = (counts / prompt_length).astype(np.float32)
        anchor_label = "prompt-only estimate"
    drift, per_path = drift_curves(thetas, grid, anchor_theta, scale=vocab_size - 1)
    drift.to_csv(output_dir / f"{source}_drift_vs_length.csv", index=False)
    np.savez_compressed(
        output_dir / f"{source}_drift_paths.npz",
        rollout_lengths=np.asarray(grid),
        per_path_drift=per_path,
        thetas_at_grid=thetas,
        anchor_theta=anchor_theta,
    )
    plot_drift(
        drift,
        per_path,
        grid,
        output_dir / f"{source}_drift_vs_length",
        f"{source}: drift from {anchor_label}",
    )

    counts_frame = count_curves(
        thetas[:, -1, :],
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
        f"{source}: recovery error vs rollout count (N={grid[-1]})",
    )


def main(config: DiagnosticsConfig) -> None:
    config.validate()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_info = load_run_info(
        Path(config.checkpoint_path),
        device=config.device,
        n_ctx_override=config.n_ctx_override,
    )
    model = run_info["model"]
    assert isinstance(model, UnsupervisedPFN)
    theta_pool = torch.as_tensor(run_info["tasks"], dtype=torch.float32)
    vocab_size = int(theta_pool.shape[1])
    bos_token = vocab_size
    alpha = torch.ones(vocab_size) * config.alpha_value

    for source in config.prompt_sources:
        if source == "prior":
            prompt_tensor = None
            prompt_tokens = np.empty((0,), dtype=np.int16)
        else:
            urn_index = int(rng.integers(theta_pool.shape[0]))
            urn = theta_pool[urn_index]
            prompt_tensor = torch.multinomial(
                urn, config.prompt_length, replacement=True
            ).to(torch.float32)
            prompt_tokens = prompt_tensor.to(torch.int16).numpy()

        trajectories = rollout_trajectories(
            model,
            steps=config.steps,
            num_rollouts=config.num_rollouts,
            bos_token=bos_token,
            prompt=prompt_tensor,
            temperature=config.temperature,
            chunk_size=config.chunk_size,
        )
        np.savez_compressed(
            output_dir / f"{source}_trajectories.npz",
            trajectories=trajectories,
            prompt_tokens=prompt_tokens,
            vocab_size=np.asarray(vocab_size),
            steps=np.asarray(config.steps),
            checkpoint_path=np.asarray(str(config.checkpoint_path)),
        )
        analyze_source(
            source=source,
            trajectories=trajectories,
            prompt_tokens=prompt_tokens,
            alpha=alpha,
            theta_pool=theta_pool,
            config=config,
            output_dir=output_dir,
            rng=rng,
        )
        print(f"[{source}] diagnostics written to {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(DiagnosticsConfig))
