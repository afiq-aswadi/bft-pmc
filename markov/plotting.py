"""Shared plotting utilities for Markov experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde
from scipy.stats import beta as beta_dist

from plotting.paper_style import apply_paper_style


MEMORISING_COLOR = "tab:green"
GENERALISING_COLOR = "tab:blue"
MISSPECIFIED_COLOR = "tab:red"
PMC_COLOR = "goldenrod"
IN_DISTRIBUTION_LABEL = "In-distribution"
OUT_OF_DISTRIBUTION_LABEL = "Out-of-distribution"
ID_LABEL = "ID"
OOD_LABEL = "OOD"
TARGET_COLOR = "0.35"
MEMORISING_LABEL = r"$\Pi_M$"
GENERALISING_LABEL = r"$\Pi_\infty$"
MEMORISING_LABEL_POSTERIOR = r"$\Pi^{\mathrm{mem}}(\cdot \mid c)$"
GENERALISING_LABEL_POSTERIOR = r"$\Pi^{\mathrm{gen}}(\cdot \mid c)$"


def _style_axis(
    ax: plt.Axes,
    *,
    xlabel: str,
    ylabel: str | None = None,
    log_x: bool = False,
    log_x_base2: bool = False,
    log_y: bool = False,
    symlog_y: bool = False,
    linthresh: float = 0.01,
    show_legend: bool = True,
    legend_loc: str = "best",
    grid: bool = True,
) -> None:
    """Apply the shared visual style used by the experiment plots."""
    if log_x_base2:
        ax.set_xscale("log", base=2)
    elif log_x:
        ax.set_xscale("log")

    if symlog_y:
        ax.set_yscale("symlog", linthresh=linthresh)
    elif log_y:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, loc=legend_loc)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, alpha=0.3, linestyle="--")


def plot_task_diversity(
    csv_path: str | Path,
    save_path: str | Path,
    *,
    max_steps: int = 1000,
    context_len: int = 400,
) -> None:
    """Plot task-diversity results in the Figure 1a style."""
    dataframe = pd.read_csv(csv_path).sort_values(by="n_chains")

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(
        dataframe["n_chains"],
        dataframe["final_id_kl"],
        "o-",
        label="ID",
        color=MEMORISING_COLOR,
        markersize=5,
    )
    axis.plot(
        dataframe["n_chains"],
        dataframe["final_ood_kl"],
        "s-",
        label="OOD",
        color=GENERALISING_COLOR,
        markersize=5,
    )
    axis.set_title(
        f"Data Diversity Threshold (steps={max_steps}, Context={context_len})"
    )
    _style_axis(
        axis,
        xlabel=r"$M$",
        ylabel="KL",
        log_x_base2=True,
        log_y=True,
    )
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def _format_chain_label(n_chains: int) -> str:
    """Format training-chain counts as powers of two when possible."""
    if n_chains > 0 and n_chains & (n_chains - 1) == 0:
        exponent = int(np.log2(n_chains))
        return rf"$2^{{{exponent}}}$"
    return str(n_chains)


def _nearest_positions(values: np.ndarray, targets: np.ndarray) -> list[int]:
    """Return unique nearest indices in `values` for each target."""
    positions: list[int] = []
    for target in targets:
        nearest = int(np.abs(values - target).argmin())
        if nearest not in positions:
            positions.append(nearest)
    return positions


@dataclass(slots=True)
class HeatmapThresholdLocation:
    """Resolved `b)` and `c)` locations used in the task-diversity heatmap."""

    threshold_n_chains: int | None
    threshold_step: int | None


def load_task_diversity_history(
    summary_csv_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sweep summary rows plus concatenated per-run training histories."""
    summary_df = pd.read_csv(summary_csv_path).sort_values(by="n_chains")
    if summary_df.empty:
        raise ValueError(f"No rows found in {summary_csv_path}.")

    run_frames: list[pd.DataFrame] = []
    for _, row in summary_df.iterrows():
        run_df = pd.read_csv(row["csv_path"]).sort_values(by="step")
        run_df = run_df.assign(
            n_chains=int(row["n_chains"]),
            generalization_gap=lambda frame: frame["ood_kl"] - frame["id_kl"],
        )
        run_frames.append(run_df)

    combined = pd.concat(run_frames, ignore_index=True)
    return summary_df, combined


def resolve_task_diversity_heatmap_location(
    summary_df: pd.DataFrame,
    combined: pd.DataFrame,
    *,
    gap_tolerance: float,
) -> HeatmapThresholdLocation:
    """Return the threshold locations used for the `b)` and `c)` heatmap lines."""
    threshold_rows = summary_df[
        summary_df["generalization_gap"].astype(float) <= gap_tolerance
    ]
    if threshold_rows.empty:
        return HeatmapThresholdLocation(
            threshold_n_chains=None,
            threshold_step=None,
        )

    threshold_n_chains = int(threshold_rows.iloc[0]["n_chains"])
    threshold_run = combined[combined["n_chains"] == threshold_n_chains]
    threshold_hits = threshold_run[threshold_run["generalization_gap"] <= gap_tolerance]
    threshold_step = (
        None if threshold_hits.empty else int(threshold_hits.iloc[0]["step"])
    )
    return HeatmapThresholdLocation(
        threshold_n_chains=threshold_n_chains,
        threshold_step=threshold_step,
    )


def plot_task_diversity_heatmap(
    summary_csv_path: str | Path,
    save_path: str | Path,
    *,
    gap_tolerance: float = 0.05,
) -> None:
    """Plot task diversity as a KL heatmap over training steps and chain count."""
    summary_df, combined = load_task_diversity_history(summary_csv_path)
    heatmap_df = (
        combined.pivot(index="n_chains", columns="step", values="ood_kl")
        .sort_index()
        .sort_index(axis=1)
    )

    chain_values = heatmap_df.index.to_numpy(dtype=int)
    step_values = heatmap_df.columns.to_numpy(dtype=int)
    heatmap_values = heatmap_df.to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    image = axis.pcolormesh(
        step_values,
        chain_values,
        heatmap_values,
        cmap="inferno",
        shading="nearest",
    )

    axis.set_xscale("log")
    axis.set_yscale("log", base=2)
    axis.text(
        -0.16,
        1.05,
        "a)",
        transform=axis.transAxes,
        fontsize=22,
        fontweight="bold",
        va="top",
    )

    axis.set_yticks(chain_values)
    axis.set_yticklabels(
        [_format_chain_label(value) for value in chain_values],
        fontsize=12,
    )
    axis.set_ylabel(r"$M$", fontsize=18)

    if len(step_values) == 1:
        x_tick_values = [step_values[0]]
    elif len(step_values) == 2:
        x_tick_values = step_values.tolist()
    else:
        log_targets = np.geomspace(step_values[1], step_values[-1], num=6)
        x_positions = _nearest_positions(
            step_values,
            np.concatenate(([step_values[0]], log_targets)),
        )
        x_tick_values = step_values[x_positions]
    axis.set_xticks(x_tick_values)
    axis.set_xticklabels([f"{value:d}" for value in x_tick_values], fontsize=10)
    axis.minorticks_off()
    axis.set_xlabel("Training step", fontsize=18)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    threshold_location = resolve_task_diversity_heatmap_location(
        summary_df,
        combined,
        gap_tolerance=gap_tolerance,
    )
    threshold_n_chains = threshold_location.threshold_n_chains
    threshold_step = threshold_location.threshold_step
    if threshold_n_chains is not None:
        axis.axhline(
            y=threshold_n_chains,
            color="#66ff99",
            linestyle=(0, (2, 2)),
            linewidth=2.0,
        )
        axis.text(
            0.02,
            threshold_n_chains * 1.15,
            "c)",
            transform=axis.get_yaxis_transform(),
            color="#66ff99",
            fontsize=16,
            fontweight="bold",
            va="bottom",
        )

        if threshold_step is not None:
            axis.axvline(
                x=threshold_step,
                color="#ff33ff",
                linestyle=(0, (2, 2)),
                linewidth=2.0,
            )
            axis.text(
                threshold_step * 1.2,
                0.93,
                "b)",
                transform=axis.get_xaxis_transform(),
                color="#ff33ff",
                fontsize=16,
                fontweight="bold",
                va="top",
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("KL", rotation=270, labelpad=18, fontsize=18)
    colorbar.ax.tick_params(labelsize=12)

    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def _to_numpy(values: torch.Tensor | np.ndarray | list[float]) -> np.ndarray:
    """Convert tensors and sequences to NumPy arrays."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _default_transitions(k: int, num_dims: int = 4) -> list[tuple[int, int]]:
    """Select the first few flattened transition entries for plotting."""
    entries: list[tuple[int, int]] = []
    for index in range(min(num_dims, k * k)):
        entries.append((index // k, index % k))
    return entries


def _stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution for one transition matrix."""
    eigvals, eigvecs = np.linalg.eig(matrix.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    stationary = np.real(eigvecs[:, idx])
    if stationary.sum() < 0:
        stationary = -stationary
    stationary = np.clip(stationary, 1e-12, None)
    stationary /= stationary.sum()
    return stationary


def training_posterior_weights(
    training_matrices: np.ndarray,
    prompt_tokens: np.ndarray,
) -> np.ndarray:
    """Compute the discrete posterior over training matrices for a prompt."""
    if prompt_tokens.size == 0:
        return np.full(training_matrices.shape[0], 1.0 / training_matrices.shape[0])

    first_state = int(prompt_tokens[0])
    log_likelihoods = np.empty(training_matrices.shape[0], dtype=np.float64)
    for index, matrix in enumerate(training_matrices):
        stationary = _stationary_distribution(matrix)
        log_prob = np.log(np.clip(stationary[first_state], 1e-12, 1.0))
        for src, dst in zip(prompt_tokens[:-1], prompt_tokens[1:]):
            log_prob += np.log(np.clip(matrix[int(src), int(dst)], 1e-12, 1.0))
        log_likelihoods[index] = log_prob

    log_likelihoods -= log_likelihoods.max()
    weights = np.exp(log_likelihoods)
    weights /= weights.sum()
    return weights


def transition_count_matrix(prompt_tokens: np.ndarray, k: int) -> np.ndarray:
    """Count observed row-wise transitions in a prompt."""
    counts = np.zeros((k, k), dtype=np.int64)
    if prompt_tokens.size < 2:
        return counts
    for src, dst in zip(prompt_tokens[:-1], prompt_tokens[1:]):
        counts[int(src), int(dst)] += 1
    return counts


def _pmc_figsize(num_dims: int) -> tuple[float, float]:
    """Return a readable width for a row of marginal panels."""
    return (max(3.0 * num_dims, 10.0), 4.2)


def _sample_reference_values(
    values: np.ndarray,
    *,
    sample_size: int,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Draw scalar samples from the empirical ID reference distribution."""
    if weights is None:
        indices = rng.integers(0, len(values), size=sample_size)
    else:
        indices = rng.choice(len(values), size=sample_size, replace=True, p=weights)
    return np.asarray(values[indices], dtype=np.float64)


def _empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return empirical CDF coordinates for unweighted samples."""
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    cdf = np.arange(1, len(sorted_values) + 1, dtype=float) / len(sorted_values)
    return sorted_values, cdf


def _weighted_atomic_cdf(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form CDF for a discrete distribution on `values` with `weights`.

    The result is exact (no Monte Carlo error) and renders as a step plot via
    ``ax.step(..., where='post')``.
    """
    order = np.argsort(np.asarray(values, dtype=np.float64))
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    cdf = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    return sorted_values, cdf


def plot_pmc_matrix_summary(
    samples: torch.Tensor | np.ndarray,
    reference_matrices: torch.Tensor | np.ndarray,
    save_path: str | Path,
    *,
    title: str,
    target_matrix: torch.Tensor | np.ndarray | None = None,
) -> None:
    """Plot average transition matrices for quick PMC inspection."""
    sample_array = _to_numpy(samples)
    reference_array = _to_numpy(reference_matrices)
    if sample_array.ndim == 2:
        sample_array = sample_array[None, ...]
    if reference_array.ndim == 2:
        reference_array = reference_array[None, ...]

    sample_mean = sample_array.mean(axis=0)
    reference_mean = (
        _to_numpy(target_matrix)
        if target_matrix is not None
        else reference_array.mean(axis=0)
    )
    difference = np.abs(sample_mean - reference_mean)

    probability_vmax = float(max(sample_mean.max(), reference_mean.max(), 1e-8))
    error_vmax = float(max(difference.max(), 1e-8))

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    image_probability = axes[0].imshow(
        sample_mean,
        cmap="viridis",
        vmin=0.0,
        vmax=probability_vmax,
        aspect="auto",
    )
    axes[0].set_title("PMC mean")
    axes[1].imshow(
        reference_mean,
        cmap="viridis",
        vmin=0.0,
        vmax=probability_vmax,
        aspect="auto",
    )
    axes[1].set_title("Target" if target_matrix is not None else "Training-pool mean")
    image_error = axes[2].imshow(
        difference,
        cmap="magma",
        vmin=0.0,
        vmax=error_vmax,
        aspect="auto",
    )
    axes[2].set_title("|Difference|")

    for axis in axes:
        axis.set_xlabel("Next state")
        axis.set_ylabel("Current state")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.suptitle(title, fontsize="medium")
    figure.colorbar(
        image_probability,
        ax=axes[:2],
        fraction=0.046,
        pad=0.04,
        label="Probability",
    )
    figure.colorbar(
        image_error,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
        label="Absolute error",
    )
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_pmc_distributions(
    samples: torch.Tensor | np.ndarray,
    training_matrices: torch.Tensor | np.ndarray,
    save_path: str | Path,
    *,
    title_prefix: str = "Prior",
    prompt_tokens: torch.Tensor | np.ndarray | None = None,
    target_matrix: torch.Tensor | np.ndarray | None = None,
    transitions_to_plot: list[tuple[int, int]] | None = None,
) -> None:
    """Plot PMC marginals and CDFs in the shared prior/posterior style."""
    sample_array = _to_numpy(samples)
    training_array = _to_numpy(training_matrices)
    if sample_array.ndim == 2:
        sample_array = sample_array[None, ...]
    if training_array.ndim == 2:
        training_array = training_array[None, ...]

    if sample_array.ndim != 3 or training_array.ndim != 3:
        raise ValueError("Expected [n_samples, k, k] arrays for PMC plotting.")

    k = training_array.shape[1]
    prompt_array = (
        np.empty(0, dtype=np.int64)
        if prompt_tokens is None
        else _to_numpy(prompt_tokens).astype(np.int64).reshape(-1)
    )
    transitions = transitions_to_plot or _default_transitions(k)

    if prompt_array.size == 0:
        discrete_weights = np.full(
            training_array.shape[0], 1.0 / training_array.shape[0]
        )
        figure_title = "Prior samples"
    else:
        discrete_weights = training_posterior_weights(training_array, prompt_array)
        figure_title = f"Posterior samples (prompt len={prompt_array.size})"

    transition_counts = transition_count_matrix(prompt_array, k)
    figure, axes = plt.subplots(
        2,
        len(transitions),
        figsize=_pmc_figsize(len(transitions)),
        constrained_layout=True,
        sharex="col",
        sharey="row",
    )
    if len(transitions) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    figure.suptitle(title_prefix or figure_title, fontsize="medium")

    for column_index, (src, dst) in enumerate(transitions):
        density_axis = axes[0, column_index]
        cdf_axis = axes[1, column_index]

        sample_values = sample_array[:, src, dst]
        training_values = training_array[:, src, dst]
        row_counts = transition_counts[src]
        alpha = 1 + row_counts[dst]
        beta = (k - 1) + row_counts.sum() - row_counts[dst]

        # Memorising marginal is a discrete distribution on the training atoms
        # `training_values` with closed-form weights `discrete_weights`; no sampling needed.
        all_sample_values = np.concatenate([sample_values, training_values])
        lo, hi = np.percentile(all_sample_values, [0.5, 99.5])
        span = float(hi - lo)
        margin = 0.15 * span if span > 1e-9 else 0.03
        lo = max(0.0, float(lo - margin))
        hi = min(1.0, float(hi + margin))
        if target_matrix is not None:
            target_value = float(_to_numpy(target_matrix)[src, dst])
            lo = min(lo, max(0.0, target_value - 0.03))
            hi = max(hi, min(1.0, target_value + 0.03))
        x_grid = np.linspace(lo, hi, 500)
        bins = np.linspace(lo, hi, 80)
        continuous_ymax = 0.0
        analytic_pdf = beta_dist.pdf(x_grid, alpha, beta)
        analytic_cdf = beta_dist.cdf(x_grid, alpha, beta)
        # Histograms excluded from ymax (concentrated samples can spike a single
        # bin). KDE on a degenerate posterior can also spike absurdly high; cap
        # its contribution to a multiple of the analytic Beta height.
        finite_pdf = analytic_pdf[np.isfinite(analytic_pdf)]
        analytic_max = float(np.max(finite_pdf)) if finite_pdf.size else 0.0
        if finite_pdf.size:
            continuous_ymax = max(continuous_ymax, float(np.percentile(finite_pdf, 99)))
        kde_cap = 1.5 * analytic_max

        density_axis.hist(
            sample_values,
            bins=bins,
            density=True,
            alpha=0.35,
            color=PMC_COLOR,
            label="PMC",
            histtype="stepfilled",
        )
        if np.std(sample_values) > 1e-8:
            kde = gaussian_kde(sample_values)
            kde_values = kde(x_grid)
            density_axis.plot(
                x_grid,
                kde_values,
                color=PMC_COLOR,
                linewidth=1.5,
            )
            continuous_ymax = max(
                continuous_ymax, min(float(np.max(kde_values)), kde_cap)
            )
        else:
            density_axis.axvline(
                float(sample_values[0]),
                color=PMC_COLOR,
                linewidth=1.5,
            )

        density_axis.hist(
            training_values,
            bins=bins,
            weights=discrete_weights,
            density=True,
            alpha=0.5,
            color=MEMORISING_COLOR,
            label=ID_LABEL,
            histtype="stepfilled",
        )
        density_axis.plot(
            x_grid,
            analytic_pdf,
            color=GENERALISING_COLOR,
            linewidth=1.5,
            label=OOD_LABEL,
        )

        sample_cdf_x, sample_cdf_y = _empirical_cdf(sample_values)
        cdf_axis.plot(
            sample_cdf_x, sample_cdf_y, color=PMC_COLOR, label="PMC", linewidth=1.2
        )
        id_cdf_x, id_cdf_y = _weighted_atomic_cdf(training_values, discrete_weights)
        cdf_axis.step(
            id_cdf_x,
            id_cdf_y,
            where="post",
            color=MEMORISING_COLOR,
            label=ID_LABEL,
            linewidth=1.2,
        )
        cdf_axis.plot(
            x_grid,
            analytic_cdf,
            color=GENERALISING_COLOR,
            linewidth=1.2,
            label=OOD_LABEL,
        )

        if target_matrix is not None:
            density_axis.axvline(
                target_value,
                color=TARGET_COLOR,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )
            cdf_axis.axvline(
                target_value,
                color=TARGET_COLOR,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )

        density_axis.set_title(f"$P_{{{src} \\to {dst}}}$", fontsize="small")
        density_axis.set_xlim(lo, hi)
        cdf_axis.set_xlim(lo, hi)
        cdf_axis.set_ylim(0.0, 1.0)
        if continuous_ymax > 0:
            density_axis.set_ylim(0.0, continuous_ymax * 1.15)

        density_axis.spines["top"].set_visible(False)
        density_axis.spines["right"].set_visible(False)
        cdf_axis.spines["top"].set_visible(False)
        cdf_axis.spines["right"].set_visible(False)

        if column_index == 0:
            density_axis.set_ylabel("Density")
            cdf_axis.set_ylabel("CDF")
            density_axis.legend(frameon=False, fontsize="x-small")

    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_pmc_distribution_matrix(
    samples: torch.Tensor | np.ndarray,
    training_matrices: torch.Tensor | np.ndarray,
    save_path: str | Path,
    *,
    title_prefix: str = "Prior",
    prompt_tokens: torch.Tensor | np.ndarray | None = None,
    mode: str = "density",
    panel_size: float = 1.4,
    share_axes: bool = False,
    max_classes: int | None = None,
    dpi: int = 400,
    print_frac: float = 0.95,
) -> None:
    """Plot PMC marginals as a full K x K matrix mirroring the transition matrix.

    Each panel (src, dst) shows the marginal distribution of the (src, dst)
    transition probability. ``mode='density'`` plots the PMC histogram and KDE alongside
    the ID empirical resample and the OOD analytic Beta posterior; ``mode='cdf'``
    plots their empirical/analytic CDFs.

    By default each panel follows its own data and analytic reference, matching
    the LR and BAU marginal-cell convention. Set ``share_axes=True`` to force
    common x limits across the matrix.
    Density y limits always follow each cell's own continuous curves.

    ``max_classes`` restricts the displayed grid to the first ``max_classes``
    rows/cols. The underlying posterior (transition counts, Beta priors,
    discrete weights) still uses the full K — only the rendered panels are
    sliced.
    """
    if mode not in {"density", "cdf"}:
        raise ValueError("mode must be 'density' or 'cdf'.")

    sample_array = _to_numpy(samples)
    training_array = _to_numpy(training_matrices)
    if sample_array.ndim == 2:
        sample_array = sample_array[None, ...]
    if training_array.ndim == 2:
        training_array = training_array[None, ...]
    if sample_array.ndim != 3 or training_array.ndim != 3:
        raise ValueError("Expected [n_samples, k, k] arrays for PMC plotting.")

    k = training_array.shape[1]
    prompt_array = (
        np.empty(0, dtype=np.int64)
        if prompt_tokens is None
        else _to_numpy(prompt_tokens).astype(np.int64).reshape(-1)
    )

    if prompt_array.size == 0:
        discrete_weights = np.full(
            training_array.shape[0], 1.0 / training_array.shape[0]
        )
    else:
        discrete_weights = training_posterior_weights(training_array, prompt_array)

    transition_counts = transition_count_matrix(prompt_array, k)

    kd = k if max_classes is None else min(k, max_classes)

    cell_data: dict[tuple[int, int], dict] = {}
    global_lo = 1.0
    global_hi = 0.0
    for src in range(kd):
        for dst in range(kd):
            sample_values = sample_array[:, src, dst]
            training_values = training_array[:, src, dst]
            row_counts = transition_counts[src]
            alpha = 1 + row_counts[dst]
            beta = (k - 1) + row_counts.sum() - row_counts[dst]
            # same data-following x-range as plotting.marginal_cell.cell_xrange:
            # union of the PMC [0.5,99.5] quantiles and the Π_∞ Beta(alpha,beta)
            # [0.5,99.5] quantiles, +15% margin, clipped to [0,1]. Following the
            # data + the reference avoids both dead whitespace and clipping Π_∞.
            plo, phi = np.percentile(sample_values, [0.5, 99.5])
            rlo, rhi = beta_dist.ppf([0.005, 0.995], alpha, beta)
            lo, hi = min(float(plo), float(rlo)), max(float(phi), float(rhi))
            span = hi - lo
            margin = 0.15 * span if span > 1e-9 else 0.03
            lo = max(0.0, lo - margin)
            hi = min(1.0, hi + margin)
            cell_data[(src, dst)] = {
                "sample_values": sample_values,
                "training_values": training_values,
                "alpha": int(alpha),
                "beta": int(beta),
                "lo": lo,
                "hi": hi,
            }
            global_lo = min(global_lo, lo)
            global_hi = max(global_hi, hi)

    if share_axes:
        for cell in cell_data.values():
            cell["lo"] = global_lo
            cell["hi"] = global_hi

    if mode == "density":
        for cell in cell_data.values():
            x_grid = np.linspace(cell["lo"], cell["hi"], 500)
            cell["x_grid"] = x_grid
            cell["bins"] = np.linspace(cell["lo"], cell["hi"], 61)
            cell["analytic_pdf"] = beta_dist.pdf(x_grid, cell["alpha"], cell["beta"])
            # Histograms excluded from ymax (concentrated samples can spike a
            # single bin). Per-cell ylim with full KDE peak (no cap) and no
            # cross-cell sharing — a degenerate PMC mode in one cell no longer
            # crushes the Beta references in the rest of the K x K grid.
            finite_pdf = cell["analytic_pdf"][np.isfinite(cell["analytic_pdf"])]
            analytic_99 = (
                float(np.percentile(finite_pdf, 99)) if finite_pdf.size else 0.0
            )
            ymax = analytic_99
            if np.std(cell["sample_values"]) > 1e-8:
                kde_values = gaussian_kde(cell["sample_values"])(x_grid)
                cell["kde_values"] = kde_values
                ymax = max(ymax, float(np.max(kde_values)))
            else:
                cell["kde_values"] = None
            cell["ymax"] = ymax

    apply_paper_style(panel_size * kd, print_frac)

    figure, axes = plt.subplots(
        kd,
        kd,
        figsize=(panel_size * kd, panel_size * kd),
        constrained_layout=True,
        sharex=share_axes,
        sharey=(share_axes if mode == "cdf" else False),
    )
    if kd == 1:
        axes = np.asarray(axes).reshape(1, 1)
    # No suptitle: the figure-level legend sits at the top, and the paper
    # caption identifies the prompt source and task diversity.

    for src in range(kd):
        for dst in range(kd):
            ax = axes[src, dst]
            cell = cell_data[(src, dst)]
            lo, hi = cell["lo"], cell["hi"]

            if mode == "density":
                x_grid = cell["x_grid"]
                bins = cell["bins"]
                ax.hist(
                    cell["sample_values"],
                    bins=bins,
                    density=True,
                    alpha=0.35,
                    color=PMC_COLOR,
                    histtype="stepfilled",
                )
                if cell["kde_values"] is not None:
                    ax.plot(x_grid, cell["kde_values"], color=PMC_COLOR, linewidth=1.0)
                else:
                    ax.axvline(
                        float(cell["sample_values"][0]),
                        color=PMC_COLOR,
                        linewidth=1.0,
                    )
                # Memorising marginal (same convention as the LR/BAU plotters,
                # mirrors plotting.marginal_cell: for the prior only, per-atom
                # vlines at low M (<= 8 tasks, matching BAU's cutoff); otherwise a
                # weighted histogram (posteriors always; M >= 16; the green fill
                # then converges to the Π_∞ Beta). Opacity ∝ weight; excluded from
                # the y-limit.
                n_atoms = cell["training_values"].size
                is_prior_cell = prompt_array.size == 0
                if is_prior_cell and n_atoms and n_atoms <= 8:
                    max_w = float(np.max(discrete_weights))
                    if max_w > 0:
                        alphas = np.clip(
                            np.asarray(discrete_weights) / max_w * 0.6, 0.05, 1.0
                        )
                        rgba = np.tile(
                            np.array(mcolors.to_rgba(MEMORISING_COLOR)), (n_atoms, 1)
                        )
                        rgba[:, 3] = alphas
                        ax.vlines(
                            cell["training_values"],
                            0.0,
                            1.0,
                            transform=ax.get_xaxis_transform(),
                            colors=rgba,
                            linewidth=0.5,
                        )
                elif discrete_weights.size:
                    ax.hist(
                        cell["training_values"],
                        bins=bins,
                        weights=discrete_weights,
                        density=True,
                        alpha=0.5,
                        color=MEMORISING_COLOR,
                        histtype="stepfilled",
                    )
                ax.plot(
                    x_grid,
                    cell["analytic_pdf"],
                    color=GENERALISING_COLOR,
                    linewidth=1.0,
                )
                if cell["ymax"] > 0:
                    ax.set_ylim(0.0, cell["ymax"] * 1.15)
            else:
                x_grid = np.linspace(lo, hi, 400)
                analytic_cdf = beta_dist.cdf(x_grid, cell["alpha"], cell["beta"])
                sample_cdf_x, sample_cdf_y = _empirical_cdf(cell["sample_values"])
                ax.plot(sample_cdf_x, sample_cdf_y, color=PMC_COLOR, linewidth=1.0)
                id_cdf_x, id_cdf_y = _weighted_atomic_cdf(
                    cell["training_values"], discrete_weights
                )
                ax.step(
                    id_cdf_x,
                    id_cdf_y,
                    where="post",
                    color=MEMORISING_COLOR,
                    linewidth=1.0,
                )
                ax.plot(x_grid, analytic_cdf, color=GENERALISING_COLOR, linewidth=1.0)
                ax.set_ylim(0.0, 1.0)

            ax.set_xlim(lo, hi)
            ax.tick_params(axis="both", which="both", labelsize=8, length=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if dst == 0:
                ax.set_ylabel(f"src={src}", fontsize="medium")
            elif mode == "density":
                # density-mode ylim is per-cell, so keep tick labels visible
                # everywhere — readers need each panel's scale.
                pass
            elif share_axes:
                ax.tick_params(axis="y", labelleft=False)
            else:
                ax.set_yticklabels([])
            if src == 0:
                ax.set_title(f"dst={dst}", fontsize="large")
            if src != kd - 1:
                if share_axes:
                    ax.tick_params(axis="x", labelbottom=False)
                else:
                    ax.set_xticklabels([])

    is_prior_figure = prompt_array.size == 0
    mem_label = MEMORISING_LABEL if is_prior_figure else MEMORISING_LABEL_POSTERIOR
    gen_label = GENERALISING_LABEL if is_prior_figure else GENERALISING_LABEL_POSTERIOR
    legend_handles = [
        plt.Line2D([0], [0], color=PMC_COLOR, lw=1.5, label="PMC"),
        plt.Line2D([0], [0], color=MEMORISING_COLOR, lw=1.5, label=mem_label),
        plt.Line2D([0], [0], color=GENERALISING_COLOR, lw=1.5, label=gen_label),
    ]
    figure.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=len(legend_handles),
        frameon=False,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_transient(
    csv_path: str | Path,
    save_path: str | Path,
    *,
    n_chains: int,
) -> None:
    """Plot transient training dynamics for a single run."""
    dataframe = pd.read_csv(csv_path)
    filtered = dataframe[dataframe["n_chains"] == n_chains].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for n_chains={n_chains} in {csv_path}.")

    filtered = filtered.sort_values(by="step")

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(
        filtered["step"],
        filtered["id_kl"],
        "-",
        label="ID",
        color=MEMORISING_COLOR,
    )
    axis.plot(
        filtered["step"],
        filtered["ood_kl"],
        "-",
        label="OOD",
        color=GENERALISING_COLOR,
    )
    axis.set_title(f"Transient Nature of ICL (N={n_chains}, Context=400)")
    _style_axis(
        axis,
        xlabel="Training step",
        ylabel="KL",
        log_x=True,
        log_y=True,
    )
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


_DISTANCE_SWEEP_SERIES = [
    (IN_DISTRIBUTION_LABEL, "o-", MEMORISING_COLOR, 5),
    (OUT_OF_DISTRIBUTION_LABEL, "s-", GENERALISING_COLOR, 5),
]

_DISTANCE_DYNAMICS_SERIES = [
    (IN_DISTRIBUTION_LABEL, "-", MEMORISING_COLOR),
    (OUT_OF_DISTRIBUTION_LABEL, "-", GENERALISING_COLOR),
]


def _style_distance_sweep_axis(ax: plt.Axes, ylabel: str | None = None) -> None:
    """Render the shared Markov distance-sweep layout."""
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$M$")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def _plot_distance_sweep_series(
    ax: plt.Axes,
    x: pd.Series,
    columns: list[str],
    dataframe: pd.DataFrame,
) -> None:
    """Plot the in-distribution and out-of-distribution sweep curves."""
    for column, (label, fmt, color, markersize) in zip(columns, _DISTANCE_SWEEP_SERIES):
        if column in dataframe.columns and dataframe[column].notna().any():
            ax.plot(
                x,
                dataframe[column],
                fmt,
                label=label,
                color=color,
                markersize=markersize,
            )


def plot_distribution_distance_sweep(
    metrics: pd.DataFrame | str | Path,
    save_path: str | Path,
    *,
    prompt_length: int,
) -> None:
    """Plot Markov ED/SW sweep curves in the same style as the LR combined plots."""
    dataframe = (
        pd.read_csv(metrics) if isinstance(metrics, (str, Path)) else metrics.copy()
    )
    dist_df = dataframe[dataframe["prompt_length"] == prompt_length].sort_values(
        "n_chains"
    )

    if prompt_length > 0:
        figure, axes = plt.subplots(
            2, 2, figsize=(12, 8), constrained_layout=True, sharey="col"
        )
        row_labels = [IN_DISTRIBUTION_LABEL, OUT_OF_DISTRIBUTION_LABEL]
        ylabels = ["Energy distance", "Sliced Wasserstein"]

        indist_df = dist_df[dist_df["prompt_source"] == "in_distribution"].sort_values(
            "n_chains"
        )
        ood_df = dist_df[dist_df["prompt_source"] == "out_of_distribution"].sort_values(
            "n_chains"
        )

        _plot_distance_sweep_series(
            axes[0, 0],
            indist_df["n_chains"],
            [
                "dist/ed_vs_baseline_in_distribution",
                "dist/ed_vs_baseline_out_of_distribution",
            ],
            indist_df,
        )
        _plot_distance_sweep_series(
            axes[0, 1],
            indist_df["n_chains"],
            [
                "dist/sw_vs_baseline_in_distribution",
                "dist/sw_vs_baseline_out_of_distribution",
            ],
            indist_df,
        )
        _plot_distance_sweep_series(
            axes[1, 0],
            ood_df["n_chains"],
            [
                "dist/ed_vs_baseline_in_distribution",
                "dist/ed_vs_baseline_out_of_distribution",
            ],
            ood_df,
        )
        _plot_distance_sweep_series(
            axes[1, 1],
            ood_df["n_chains"],
            [
                "dist/sw_vs_baseline_in_distribution",
                "dist/sw_vs_baseline_out_of_distribution",
            ],
            ood_df,
        )

        for column_index, ylabel in enumerate(ylabels):
            for row_index in range(2):
                _style_distance_sweep_axis(axes[row_index, column_index], ylabel)

        for row_index, row_label in enumerate(row_labels):
            axes[row_index, 0].annotate(
                row_label,
                xy=(-0.25, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize="large",
                rotation=90,
            )
    else:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        prior_df = dist_df[
            dist_df["prompt_source"].isna() | (dist_df["prompt_source"] == "N/A")
        ].sort_values("n_chains")

        _plot_distance_sweep_series(
            axes[0],
            prior_df["n_chains"],
            [
                "dist/ed_vs_baseline_in_distribution",
                "dist/ed_vs_baseline_out_of_distribution",
            ],
            prior_df,
        )
        _plot_distance_sweep_series(
            axes[1],
            prior_df["n_chains"],
            [
                "dist/sw_vs_baseline_in_distribution",
                "dist/sw_vs_baseline_out_of_distribution",
            ],
            prior_df,
        )

        _style_distance_sweep_axis(axes[0], "Energy distance")
        _style_distance_sweep_axis(axes[1], "Sliced Wasserstein")

    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _style_distance_dynamics_axis(
    ax: plt.Axes,
    ylabel: str | None = None,
    *,
    log_xscale: bool = False,
) -> None:
    """Render the shared Markov distance-dynamics layout."""
    ax.set_yscale("log")
    if log_xscale:
        ax.set_xscale("log")
    ax.set_xlabel("Training step")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def _plot_distance_dynamics_series(
    ax: plt.Axes,
    x: pd.Series,
    columns: list[str],
    dataframe: pd.DataFrame,
) -> None:
    """Plot the in-distribution and out-of-distribution dynamics curves."""
    for column, (label, fmt, color) in zip(columns, _DISTANCE_DYNAMICS_SERIES):
        if column in dataframe.columns and dataframe[column].notna().any():
            ax.plot(x, dataframe[column], fmt, label=label, color=color)


def plot_distribution_distance_dynamics(
    metrics: pd.DataFrame | str | Path,
    save_path: str | Path,
    *,
    mode: str = "posterior",
    log_xscale: bool = False,
) -> None:
    """Plot Markov ED/SW dynamics in prior or posterior combined layouts."""
    dataframe = (
        pd.read_csv(metrics) if isinstance(metrics, (str, Path)) else metrics.copy()
    )

    if mode == "posterior":
        required_column = "ed_vs_baseline_in_distribution_from_prompts_in_distribution"
        if required_column not in dataframe.columns:
            raise ValueError(
                "Posterior dynamics requested, but per-source posterior columns were not found."
            )

        figure, axes = plt.subplots(
            2, 2, figsize=(12, 8), constrained_layout=True, sharey="col"
        )
        row_labels = [IN_DISTRIBUTION_LABEL, OUT_OF_DISTRIBUTION_LABEL]
        ylabels = ["Energy distance", "Sliced Wasserstein"]

        _plot_distance_dynamics_series(
            axes[0, 0],
            dataframe["step"],
            [
                "ed_vs_baseline_in_distribution_from_prompts_in_distribution",
                "ed_vs_baseline_out_of_distribution_from_prompts_in_distribution",
            ],
            dataframe,
        )
        _plot_distance_dynamics_series(
            axes[0, 1],
            dataframe["step"],
            [
                "sw_vs_baseline_in_distribution_from_prompts_in_distribution",
                "sw_vs_baseline_out_of_distribution_from_prompts_in_distribution",
            ],
            dataframe,
        )
        _plot_distance_dynamics_series(
            axes[1, 0],
            dataframe["step"],
            [
                "ed_vs_baseline_in_distribution_from_prompts_out_of_distribution",
                "ed_vs_baseline_out_of_distribution_from_prompts_out_of_distribution",
            ],
            dataframe,
        )
        _plot_distance_dynamics_series(
            axes[1, 1],
            dataframe["step"],
            [
                "sw_vs_baseline_in_distribution_from_prompts_out_of_distribution",
                "sw_vs_baseline_out_of_distribution_from_prompts_out_of_distribution",
            ],
            dataframe,
        )

        for column_index, ylabel in enumerate(ylabels):
            for row_index in range(2):
                _style_distance_dynamics_axis(
                    axes[row_index, column_index],
                    ylabel,
                    log_xscale=log_xscale,
                )

        for row_index, row_label in enumerate(row_labels):
            axes[row_index, 0].annotate(
                row_label,
                xy=(-0.25, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize="large",
                rotation=90,
            )
    elif mode == "prior":
        figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        _plot_distance_dynamics_series(
            axes[0],
            dataframe["step"],
            [
                "ed_vs_baseline_in_distribution",
                "ed_vs_baseline_out_of_distribution",
            ],
            dataframe,
        )
        _plot_distance_dynamics_series(
            axes[1],
            dataframe["step"],
            [
                "sw_vs_baseline_in_distribution",
                "sw_vs_baseline_out_of_distribution",
            ],
            dataframe,
        )

        _style_distance_dynamics_axis(axes[0], "Energy distance", log_xscale=log_xscale)
        _style_distance_dynamics_axis(
            axes[1], "Sliced Wasserstein", log_xscale=log_xscale
        )
    else:
        raise ValueError(f"Unsupported dynamics mode: {mode!r}")

    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
