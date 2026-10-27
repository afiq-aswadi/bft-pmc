"""Rollout-length and rollout-count diagnostics for BAU Predictive Monte Carlo.

For one checkpoint, generates rollout trajectories once, then reads off:
  1. recovery error (energy distance, sliced Wasserstein-1) against the
     analytic baselines as a function of rollout length N at fixed R;
  2. the path-stability drift diagnostic: expected scaled L1 norm between
     the anchored estimate and the depth-N estimate, with per-path traces;
  3. recovery error as a function of rollout count R at fixed N, by
     subsampling the rollout set, alongside a Monte Carlo floor from fresh
     baseline draws of the same size.

Everything is saved under --output-dir, per prompt source, so every figure
can be redrawn (bft_pmc/analysis/replot_pmc_diagnostics.py) or every curve recomputed
without the model:
  - {source}_trajectories.npz: the rollout tokens, prompt and urn index;
  - {source}_baselines.npz: the baseline sample sets, task pool and alpha;
  - {source}_recovery_vs_length.csv, {source}_drift_vs_length.csv and
    {source}_recovery_vs_count.csv: the plotted curves;
  - {source}_drift_paths.npz: per-path drift, running estimates and anchor;
  - {source}_recovery_vs_count_resamples.csv: every draw behind the count panel;
  - run_record.json and figures.json: settings, provenance and figure index;
  - <run>_rollouts.npz: the posterior rollouts, for the jellyfish plot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from bft_pmc.analysis.checkpoints import load_run_info
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
from bft_pmc.balls_and_urns.analysis import precompute_baseline_samples
from bft_pmc.balls_and_urns.baselines import (
    sample_generalising_posterior,
    sample_memorising_posterior,
)
from bft_pmc.pos_encoding import positional_encoding_of
from pfn_transformerlens.model.PFN import UnsupervisedPFN


@dataclass
class DiagnosticsConfig:
    """Rollout-length and rollout-count diagnostics for one balls-and-urns checkpoint."""

    checkpoint_path: str
    """Checkpoint to diagnose."""
    output_dir: str = "outputs/bau_rollout_diagnostics"
    """Directory for the curves, figures, rollout bundles and run record."""
    alpha_value: float = 1.0
    """Concentration of the symmetric Dirichlet population prior, which defines
    the generalizing reference."""
    steps: int = 500
    """Rollout length N: tokens generated after the prompt."""
    num_rollouts: int = 1024
    """Rollouts R per prompt source."""
    length_grid: list[int] = field(
        default_factory=lambda: [8, 16, 32, 64, 128, 256, 384, 500]
    )
    """Rollout lengths at which recovery error and drift are read off; at most
    steps."""
    n_ctx_override: int | None = 1024
    """Context length to build the model with, so rollouts can run past the
    trained context; None keeps the trained context."""
    prompt_sources: list[str] = field(default_factory=lambda: ["prior", "posterior"])
    """prior (BOS only) and/or posterior (one in-distribution prompt)."""
    prompt_length: int = 16
    """Tokens in the posterior prompt."""
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
    chunk_size: int = 256
    """Rollouts generated per forward batch."""
    temperature: float = 1.0
    """Sampling temperature; 1.0 samples the model's own predictive."""
    seed: int = 42
    """Seed for prompts, rollouts and reference draws."""
    device: str = "auto"
    """auto, cpu, cuda or mps."""

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


def jellyfish_rollouts(
    model: UnsupervisedPFN,
    *,
    theta_pool: torch.Tensor,
    bos_token: int,
    config: DiagnosticsConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Rollouts from several prompt realisations, for the jellyfish plot.

    TabMGP draws the path-stability curve once per realisation of the observed
    data (their Figure 8), so each realisation here is a fresh prompt from a
    fresh urn. Returns trajectories of shape (realisations, rollouts,
    prompt_length + steps) and the urn behind each.
    """
    realisations: list[np.ndarray] = []
    urns: list[int] = []
    for _ in range(config.jellyfish_prompts):
        urn_index = int(rng.integers(theta_pool.shape[0]))
        prompt = torch.multinomial(
            theta_pool[urn_index], config.prompt_length, replacement=True
        ).to(torch.float32)
        realisations.append(
            rollout_trajectories(
                model,
                steps=config.steps,
                num_rollouts=config.jellyfish_rollouts,
                bos_token=bos_token,
                prompt=prompt,
                temperature=config.temperature,
                chunk_size=min(config.chunk_size, config.jellyfish_rollouts),
            )
        )
        urns.append(urn_index)
    return np.stack(realisations), np.asarray(urns, dtype=np.int32)


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
    fresh_model_sampler=None,
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

    np.savez_compressed(
        output_dir / f"{source}_baselines.npz",
        **{name: np.asarray(samples) for name, samples in baselines.items()},
        task_pool=theta_pool.numpy(),
        alpha=alpha.numpy(),
        prompt_tokens=prompt_tokens,
    )

    thetas = running_theta(trajectories, vocab_size, prompt_length, grid)

    recovery = recovery_curves(thetas, baselines, grid, config.n_projections)
    recovery.to_csv(output_dir / f"{source}_recovery_vs_length.csv", index=False)
    recovery_title = (
        f"{source}: recovery error vs rollout length (R={config.num_rollouts})"
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
    drift_title = f"{source}: drift from {anchor_label}"
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
        thetas[:, -1, :],
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
    counts_title = f"{source}: recovery error vs rollout count (N={grid[-1]})"
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
    context_length = int(model.transformer.cfg.n_ctx)
    check_rollouts_fit_context(
        steps=config.steps,
        prompt_length=config.prompt_length,
        prompt_sources=config.prompt_sources,
        context_length=context_length,
    )
    theta_pool = torch.as_tensor(run_info["tasks"], dtype=torch.float32)
    vocab_size = int(theta_pool.shape[1])
    bos_token = vocab_size
    alpha = torch.ones(vocab_size) * config.alpha_value
    write_run_record(
        output_dir,
        asdict(config),
        inputs=[config.checkpoint_path],
        family="bau",
        device=str(next(model.parameters()).device),
        vocab_size=vocab_size,
        latent_dimension=vocab_size - 1,
        drift_scale=vocab_size - 1,
        num_tasks=int(theta_pool.shape[0]),
        bos_token=bos_token,
        model_context=context_length,
        pos_encoding=positional_encoding_of(model.config),
        checkpoint_metadata=asdict(run_info["metadata"]),
    )

    for source in config.prompt_sources:
        if source == "prior":
            urn_index = -1
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
            urn_index=np.asarray(urn_index),
            vocab_size=np.asarray(vocab_size),
            steps=np.asarray(config.steps),
            checkpoint_path=np.asarray(str(config.checkpoint_path)),
        )

        def fresh_model_sampler(r: int, _prompt=prompt_tensor) -> np.ndarray:
            # An independent pool of r rollouts, not a subset of the saved one.
            fresh = rollout_trajectories(
                model,
                steps=config.steps,
                num_rollouts=r,
                bos_token=bos_token,
                prompt=_prompt,
                temperature=config.temperature,
                chunk_size=config.chunk_size,
            )
            return running_theta(
                fresh,
                int(alpha.shape[0]),
                int(prompt_tokens.shape[0]),
                sorted(config.length_grid),
            )[:, -1, :]

        analyze_source(
            source=source,
            trajectories=trajectories,
            prompt_tokens=prompt_tokens,
            alpha=alpha,
            theta_pool=theta_pool,
            config=config,
            output_dir=output_dir,
            rng=rng,
            fresh_model_sampler=(
                on_isolated_torch_stream(fresh_model_sampler, config.seed + 10_000)
                if config.independent_count_pools
                else None
            ),
        )
        print(f"[{source}] diagnostics written to {output_dir}")

    # After every diagnostic, so the rollouts above keep their random stream:
    # the jellyfish bundle, one prompt realisation per curve.
    if "posterior" in config.prompt_sources:
        realisations, urn_indices = jellyfish_rollouts(
            model,
            theta_pool=theta_pool,
            bos_token=bos_token,
            config=config,
            rng=rng,
        )
        save_path_stability_bundle(
            output_dir,
            prompt_length=config.prompt_length,
            checkpoint_path=config.checkpoint_path,
            arrays={"rollout_tokens": realisations},
            prompt_source="in_distribution",
            num_tasks=int(theta_pool.shape[0]),
            urn_index=urn_indices,
        )


if __name__ == "__main__":
    main(tyro.cli(DiagnosticsConfig))
