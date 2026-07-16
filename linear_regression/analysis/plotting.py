"""Plotting helpers for the linear-regression sweep analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from linear_regression.analysis.config import SOURCE_DISPLAY_LABELS, SweepConfig
from plotting.marginal_cell import (
    LR_VLINE_MAX_M,
    cell_xrange,
    draw_cdf_cell,
    draw_density_cell,
    legend_handles,
    ref_quantiles,
)


PUBLICATION_DPI = 300


def _style_axis(
    ax: plt.Axes,
    *,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    log_y: bool = False,
) -> None:
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("symlog", linthresh=0.001)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize="large")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_transition(
    df: pd.DataFrame,
    output_dir: Path,
    eval_position: int | None = None,
) -> None:
    """Plot model distance from each Bayes-optimal predictor across task diversity."""
    prediction_df = df.drop_duplicates(
        subset=["run_id", "num_tasks", "checkpoint_step"]
    )
    position_suffix = (
        f" (position {eval_position})" if eval_position is not None else ""
    )
    panels = [
        ("memorising", f"In-distribution{position_suffix}"),
        ("generalising", f"Out-of-distribution{position_suffix}"),
    ]
    if "data_random/delta_vs_baseline_memorising" in prediction_df.columns:
        panels.append(("random", f"Random{position_suffix}"))

    for log_y, suffix in ((False, ""), (True, "_log")):
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(5 * len(panels), 4),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, (distribution, title) in zip(axes[0], panels, strict=True):
            ax.plot(
                prediction_df["num_tasks"],
                prediction_df[f"data_{distribution}/delta_vs_baseline_memorising"],
                "o-",
                label="Memorising",
                color="tab:green",
                markersize=5,
            )
            ax.plot(
                prediction_df["num_tasks"],
                prediction_df[f"data_{distribution}/delta_vs_baseline_generalising"],
                "s-",
                label="Generalising",
                color="tab:blue",
                markersize=5,
            )
            _style_axis(
                ax,
                xlabel=r"$M$",
                ylabel=r"$\Delta$",
                title=title,
                log_y=log_y,
            )

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncols=len(labels),
            frameon=False,
        )
        _save_figure(fig, output_dir / f"transition_plot{suffix}.png")


def plot_mse_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    eval_position: int | None = None,
) -> None:
    """Plot model and baseline MSE across task diversity."""
    prediction_df = df.drop_duplicates(
        subset=["run_id", "num_tasks", "checkpoint_step"]
    )
    position_suffix = (
        f" (position {eval_position})" if eval_position is not None else ""
    )
    panels = [
        ("memorising", f"In-distribution{position_suffix}"),
        ("generalising", f"Out-of-distribution{position_suffix}"),
    ]
    if "data_random/model_mse" in prediction_df.columns:
        panels.append(("random", f"Random{position_suffix}"))

    series = (
        ("model_mse", "PMC", "o-", "goldenrod"),
        ("baseline_memorising_mse", "Memorising", "x--", "tab:green"),
        ("baseline_generalising_mse", "Generalising", "+--", "tab:blue"),
    )
    for log_y, suffix in ((False, ""), (True, "_log")):
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(5 * len(panels), 4),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, (distribution, title) in zip(axes[0], panels, strict=True):
            for column, label, style, color in series:
                ax.plot(
                    prediction_df["num_tasks"],
                    prediction_df[f"data_{distribution}/{column}"],
                    style,
                    label=label,
                    color=color,
                    markersize=5,
                )
            _style_axis(
                ax,
                xlabel=r"$M$",
                ylabel=r"$\mathrm{MSE}/D$",
                title=title,
                log_y=log_y,
            )

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncols=len(labels),
            frameon=False,
        )
        _save_figure(fig, output_dir / f"mse_comparison{suffix}.png")


def _plot_distance_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    memorising_column: str,
    generalising_column: str,
    ylabel: str,
    log_scale: bool,
    title: str | None = None,
) -> None:
    ax.plot(
        data["num_tasks"],
        data[memorising_column],
        "o-",
        label="Memorising",
        color="tab:green",
        markersize=4,
    )
    ax.plot(
        data["num_tasks"],
        data[generalising_column],
        "s-",
        label="Generalising",
        color="tab:blue",
        markersize=4,
    )
    _style_axis(
        ax,
        xlabel=r"$M$",
        ylabel=ylabel,
        title=title,
        log_y=log_scale,
    )


def plot_distribution_metrics_by_length(
    df: pd.DataFrame,
    output_dir: Path,
    log_scale: bool = True,
) -> None:
    """Plot energy distance and sliced Wasserstein for each prompt setting."""
    metrics = (
        (
            "dist/ed_vs_baseline_memorising",
            "dist/ed_vs_baseline_generalising",
            "Energy distance",
        ),
        (
            "dist/sw_vs_baseline_memorising",
            "dist/sw_vs_baseline_generalising",
            "Sliced Wasserstein",
        ),
    )
    suffix = "_log" if log_scale else "_linear"

    prior_df = df[df["prompt_length"] == 0]
    for n_samples_prior in sorted(prior_df["n_samples_prior"].unique()):
        subset = (
            prior_df[prior_df["n_samples_prior"] == n_samples_prior]
            .drop_duplicates(subset=["num_tasks"])
            .sort_values("num_tasks")
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, (memorising_column, generalising_column, ylabel) in zip(
            axes,
            metrics,
            strict=True,
        ):
            _plot_distance_panel(
                ax,
                subset,
                memorising_column=memorising_column,
                generalising_column=generalising_column,
                ylabel=ylabel,
                title=ylabel,
                log_scale=log_scale,
            )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncols=len(labels),
            frameon=False,
        )
        _save_figure(
            fig,
            output_dir / f"dist_prompt0_nprior{n_samples_prior}{suffix}.png",
        )

    posterior_df = df[df["prompt_length"] > 0]
    combinations = posterior_df[
        ["prompt_length", "n_samples", "n_prompts"]
    ].drop_duplicates()
    for combination in combinations.itertuples(index=False):
        subset = posterior_df[
            (posterior_df["prompt_length"] == combination.prompt_length)
            & (posterior_df["n_samples"] == combination.n_samples)
            & (posterior_df["n_prompts"] == combination.n_prompts)
        ]
        prompt_sources = sorted(subset["prompt_source"].unique())
        fig, axes = plt.subplots(
            2,
            len(prompt_sources),
            figsize=(4 * len(prompt_sources), 8),
            constrained_layout=True,
            sharey="row",
            squeeze=False,
        )
        for row, (memorising_column, generalising_column, ylabel) in enumerate(metrics):
            for column, source in enumerate(prompt_sources):
                source_df = subset[subset["prompt_source"] == source].sort_values(
                    "num_tasks"
                )
                _plot_distance_panel(
                    axes[row, column],
                    source_df,
                    memorising_column=memorising_column,
                    generalising_column=generalising_column,
                    ylabel=ylabel if column == 0 else "",
                    title=(
                        SOURCE_DISPLAY_LABELS.get(source, source)
                        if row == 0
                        else ylabel
                    ),
                    log_scale=log_scale,
                )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncols=len(labels),
            frameon=False,
        )
        _save_figure(
            fig,
            output_dir
            / (
                f"dist_prompt{combination.prompt_length}_nsamp{combination.n_samples}"
                f"_nprompt{combination.n_prompts}{suffix}.png"
            ),
        )


def plot_marginal_distributions(
    data: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    plot_memorising: bool = True,
) -> None:
    """Plot prior marginal densities and CDFs for up to four task dimensions."""
    assert data["pt"].ndim == 2, data["pt"].shape
    assert "theta_pool" in data and "dmmse_weights" in data
    num_dimensions = min(4, data["pt"].shape[1])
    fig, axes = plt.subplots(
        2,
        num_dimensions,
        figsize=(3 * num_dimensions, 4),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize="medium")

    for dimension in range(num_dimensions):
        samples = data["pt"][:, dimension]
        atoms = data["theta_pool"][:, dimension] if plot_memorising else None
        weights = data["dmmse_weights"] if plot_memorising else None
        lower, upper = cell_xrange(
            samples,
            ref=ref_quantiles(norm.ppf),
        )
        draw_density_cell(
            axes[0, dimension],
            pmc_vals=samples,
            atoms=atoms,
            weights=weights,
            gen_pdf=norm.pdf,
            lo=lower,
            hi=upper,
            is_prior=True,
            vline_max_m=LR_VLINE_MAX_M,
        )
        draw_cdf_cell(
            axes[1, dimension],
            pmc_vals=samples,
            atoms=atoms,
            weights=weights,
            gen_cdf=norm.cdf,
            lo=lower,
            hi=upper,
        )
        axes[0, dimension].set_title(f"Dimension {dimension}", fontsize="small")
        for row in range(2):
            axes[row, dimension].spines["top"].set_visible(False)
            axes[row, dimension].spines["right"].set_visible(False)

    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("CDF")
    axes[0, 0].legend(handles=legend_handles(), frameon=False, fontsize="small")
    _save_figure(fig, output_path)


def plot_per_prompt_marginals(
    data: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    plot_memorising: bool = True,
) -> None:
    """Plot posterior marginals for the first prompt and up to four dimensions."""
    assert data["pt"].ndim == 3, data["pt"].shape
    assert "baseline_generalising_posterior_means" in data
    assert "baseline_generalising_posterior_covs" in data
    assert "theta_pool" in data and "dmmse_weights" in data
    num_dimensions = min(4, data["pt"].shape[2])
    fig, axes = plt.subplots(
        2,
        num_dimensions,
        figsize=(3 * num_dimensions, 4),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize="medium")

    for dimension in range(num_dimensions):
        samples = data["pt"][0, :, dimension]
        mean = float(data["baseline_generalising_posterior_means"][0, dimension])
        covariance = data["baseline_generalising_posterior_covs"][0]
        standard_deviation = float(np.sqrt(covariance[dimension, dimension]))
        atoms = data["theta_pool"][:, dimension] if plot_memorising else None
        weights = data["dmmse_weights"][0] if plot_memorising else None
        reference = norm(loc=mean, scale=standard_deviation)
        lower, upper = cell_xrange(
            samples,
            ref=ref_quantiles(reference.ppf),
        )
        draw_density_cell(
            axes[0, dimension],
            pmc_vals=samples,
            atoms=atoms,
            weights=weights,
            gen_pdf=reference.pdf,
            lo=lower,
            hi=upper,
        )
        draw_cdf_cell(
            axes[1, dimension],
            pmc_vals=samples,
            atoms=atoms,
            weights=weights,
            gen_cdf=reference.cdf,
            lo=lower,
            hi=upper,
        )
        axes[0, dimension].set_title(f"Dimension {dimension}", fontsize="small")
        for row in range(2):
            axes[row, dimension].spines["top"].set_visible(False)
            axes[row, dimension].spines["right"].set_visible(False)

    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("CDF")
    axes[0, 0].legend(handles=legend_handles(), frameon=False, fontsize="small")
    _save_figure(fig, output_path)


def plot_results(df: pd.DataFrame, output_dir: Path, config: SweepConfig) -> None:
    """Generate all figures for a completed sweep-analysis run."""
    plot_transition(df, output_dir, eval_position=config.eval_position)
    plot_mse_comparison(df, output_dir, eval_position=config.eval_position)

    if (
        config.compute_distribution_metrics
        and "dist/ed_vs_baseline_memorising" in df.columns
    ):
        plot_distribution_metrics_by_length(df, output_dir, log_scale=True)
        plot_distribution_metrics_by_length(df, output_dir, log_scale=False)

    samples_dir = output_dir / "samples"
    if not samples_dir.exists():
        return

    for npz_path in sorted(samples_dir.glob("*.npz")):
        with np.load(npz_path) as archive:
            data = dict(archive)
        if "pt" not in data:
            continue

        stem = npz_path.stem
        num_tasks = stem.split("_")[0][1:]
        if data["pt"].ndim == 2:
            plot_marginal_distributions(
                data,
                output_dir / f"marginal_{stem}.png",
                title=rf"$M = {num_tasks}$",
                plot_memorising=config.plot_memorising_marginals,
            )
            continue

        assert data["pt"].ndim == 3, data["pt"].shape
        parts = stem.split("_")
        assert len(parts) == 3, stem
        source, prompt_length = parts[1], parts[2][1:]
        plot_per_prompt_marginals(
            data,
            output_dir / f"marginal_{stem}.png",
            title=(
                rf"$M = {num_tasks}$, {SOURCE_DISPLAY_LABELS.get(source, source)}, "
                rf"$n_{{\mathrm{{prompt}}}} = {prompt_length}$"
            ),
            plot_memorising=config.plot_memorising_marginals,
        )
