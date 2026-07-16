"""Combine the LR task-diversity sweep and training dynamics in one figure.

Left half: task-diversity sweep (Δ_MSE, ED) vs M.
Right half: training-time dynamics (Δ_MSE, ED) vs step at fixed M.

Rows are In-distribution (top) and Out-of-distribution (bottom). Single
in-axes legend in the upper-right of the rightmost panel.

Usage:
    uv run python -m linear_regression.plot_stitched_sweep_dynamics \
        --sweep-csv paper_data/lr/sweep/metrics.csv \
        --dynamics-csv paper_data/lr/dynamics/metrics.csv \
        --out-path paper_data/lr/figure_lr_sweep_and_dynamics.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SERIES_SWEEP = [
    ("Memorizing", "o-", "tab:green", 5),
    ("Generalizing", "s-", "tab:blue", 5),
]
SERIES_DYN = [
    ("Memorizing", "-", "tab:green"),
    ("Generalizing", "-", "tab:blue"),
]


def _style(
    ax: plt.Axes,
    *,
    xlabel: str,
    ylabel: str | None,
    xscale: str,
    label_fs: int,
    tick_fs: int,
) -> None:
    ax.set_xscale(xscale)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=label_fs)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def _plot_sweep_panel(ax: plt.Axes, df: pd.DataFrame, cols: list[str]) -> None:
    if df.empty:
        raise ValueError("cannot plot an empty sweep table.")
    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise KeyError(f"missing sweep columns: {missing}")
    for col, (label, fmt, color, ms) in zip(cols, SERIES_SWEEP):
        if df[col].isna().any():
            raise ValueError(f"sweep column {col!r} contains missing values.")
        ax.plot(df["num_tasks"], df[col], fmt, label=label, color=color, markersize=ms)


def _plot_dyn_panel(ax: plt.Axes, df: pd.DataFrame, cols: list[str]) -> None:
    if df.empty:
        raise ValueError("cannot plot an empty dynamics table.")
    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise KeyError(f"missing dynamics columns: {missing}")
    for col, (label, fmt, color) in zip(cols, SERIES_DYN):
        if df[col].isna().any():
            raise ValueError(f"dynamics column {col!r} contains missing values.")
        ax.plot(df["step"], df[col], fmt, label=label, color=color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-csv", type=Path, required=True)
    parser.add_argument("--dynamics-csv", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=8,
        help="Sweep posterior prompt length to filter on.",
    )
    args = parser.parse_args()

    sweep_df = pd.read_csv(args.sweep_csv)
    dyn_df = pd.read_csv(args.dynamics_csv).sort_values("step")

    pred_df = sweep_df.drop_duplicates(
        subset=["run_id", "num_tasks", "checkpoint_step"]
    ).sort_values("num_tasks")
    dist_df = sweep_df[sweep_df["prompt_length"] == args.prompt_length].sort_values(
        "num_tasks"
    )
    indist_sweep = dist_df[dist_df["prompt_source"] == "memorising"].sort_values(
        "num_tasks"
    )
    ood_sweep = dist_df[dist_df["prompt_source"] == "generalising"].sort_values(
        "num_tasks"
    )

    label_fs = 13
    tick_fs = 11
    legend_fs = 11
    header_fs = 14

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(17.5, 6.25),
        constrained_layout=True,
        sharey="col",
        gridspec_kw={"wspace": 0.05},
    )

    ylabels = [r"$\Delta_{\mathrm{MSE}}$", "Energy distance"]
    row_labels = ["In-distribution", "Out-of-distribution"]

    # Sweep half (cols 0, 1).
    _plot_sweep_panel(
        axes[0, 0],
        pred_df,
        [
            "data_memorising/delta_vs_baseline_memorising",
            "data_memorising/delta_vs_baseline_generalising",
        ],
    )
    _plot_sweep_panel(
        axes[0, 1],
        indist_sweep,
        [
            "dist/ed_vs_baseline_memorising",
            "dist/ed_vs_baseline_generalising",
        ],
    )
    _plot_sweep_panel(
        axes[1, 0],
        pred_df,
        [
            "data_generalising/delta_vs_baseline_memorising",
            "data_generalising/delta_vs_baseline_generalising",
        ],
    )
    _plot_sweep_panel(
        axes[1, 1],
        ood_sweep,
        [
            "dist/ed_vs_baseline_memorising",
            "dist/ed_vs_baseline_generalising",
        ],
    )

    # Dynamics half (cols 2, 3).
    _plot_dyn_panel(
        axes[0, 2],
        dyn_df,
        [
            "delta_vs_baseline_memorising_on_data_memorising",
            "delta_vs_baseline_generalising_on_data_memorising",
        ],
    )
    _plot_dyn_panel(
        axes[0, 3],
        dyn_df,
        [
            "ed_vs_baseline_memorising_from_prompts_memorising",
            "ed_vs_baseline_generalising_from_prompts_memorising",
        ],
    )
    _plot_dyn_panel(
        axes[1, 2],
        dyn_df,
        [
            "delta_vs_baseline_memorising_on_data_generalising",
            "delta_vs_baseline_generalising_on_data_generalising",
        ],
    )
    _plot_dyn_panel(
        axes[1, 3],
        dyn_df,
        [
            "ed_vs_baseline_memorising_from_prompts_generalising",
            "ed_vs_baseline_generalising_from_prompts_generalising",
        ],
    )

    # styling: every column gets its metric ylabel so each panel is self-describing
    for col_in_half, ylabel in enumerate(ylabels):
        for row in (0, 1):
            sweep_col = col_in_half  # 0 or 1
            _style(
                axes[row, sweep_col],
                xlabel=r"$M$",
                ylabel=ylabel,
                xscale="log",
                label_fs=label_fs,
                tick_fs=tick_fs,
            )
            dyn_col = col_in_half + 2
            _style(
                axes[row, dyn_col],
                xlabel="Training step",
                ylabel=ylabel,
                xscale="log",
                label_fs=label_fs,
                tick_fs=tick_fs,
            )

    # log-base-2 specifically for sweep side
    for row in (0, 1):
        for col in (0, 1):
            axes[row, col].set_xscale("log", base=2)

    # row labels (left of leftmost panels)
    for row, row_label in enumerate(row_labels):
        axes[row, 0].annotate(
            row_label,
            xy=(-0.32, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=label_fs,
            rotation=90,
        )

    # group headers above each half
    fig.text(
        0.27,
        1.02,
        "(a) Task-diversity sweep at end of training",
        ha="center",
        va="bottom",
        fontsize=header_fs,
    )
    fig.text(
        0.77,
        1.02,
        r"(b) Training dynamics at $M=32$",
        ha="center",
        va="bottom",
        fontsize=header_fs,
    )

    # single legend, upper-right of the rightmost panel
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, -1].legend(loc="upper right", frameon=False, fontsize=legend_fs)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
