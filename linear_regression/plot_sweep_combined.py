"""Combined sweep plot for the linear regression setting.

Produces a 2x3 grid: rows = (In-distribution, Out-of-distribution),
cols = (Delta MSE, Energy distance, Sliced Wasserstein).

The CSV has 3 rows per num_tasks value:
  - prompt_length=0, prompt_source="N/A": prior mode (no in-dist/OOD for dist metrics)
  - prompt_length>0, prompt_source=memorising: posterior, in-distribution prompts
  - prompt_length>0, prompt_source=generalising: posterior, OOD prompts

Delta (prediction) metrics live in `data_memorising/...` and `data_generalising/...`
columns. Distribution metrics are bare per-baseline; prompt_source disambiguates rows.

Usage:
    uv run python -m linear_regression.plot_sweep_combined --metrics-csv outputs/lr/sweep_analysis/.../metrics.csv
    uv run python -m linear_regression.plot_sweep_combined --metrics-csv ... --prompt-length 8
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tyro


@dataclass
class PlotConfig:
    metrics_csv: str
    output_dir: str | None = None
    prompt_length: int = 0


SERIES = [
    ("Memorizing", "o-", "tab:green", 5),
    ("Generalizing", "s-", "tab:blue", 5),
]
SERIES_PRIOR = [
    (r"$\Pi_M$", "o-", "tab:green", 5),
    (r"$\Pi_\infty$", "s-", "tab:blue", 5),
]


def style_ax(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$M$", fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def plot_series(
    ax: plt.Axes,
    x: pd.Series,
    cols: list[str],
    df: pd.DataFrame,
    series: list[tuple] = SERIES,
) -> None:
    if df.empty:
        raise ValueError("cannot plot an empty metrics table.")
    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise KeyError(f"missing plot columns: {missing}")
    for col, (label, fmt, color, ms) in zip(cols, series):
        if df[col].isna().any():
            raise ValueError(f"plot column {col!r} contains missing values.")
        ax.plot(x, df[col], fmt, label=label, color=color, markersize=ms)


def main(config: PlotConfig) -> None:
    csv_path = Path(config.metrics_csv)
    df = pd.read_csv(csv_path)
    output_dir = Path(config.output_dir) if config.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # delta metrics: one per (run_id, num_tasks) -- deduplicate across prompt configs
    pred_df = df.drop_duplicates(
        subset=["run_id", "num_tasks", "checkpoint_step"]
    ).sort_values("num_tasks")

    # distribution metrics: filter to chosen prompt_length
    dist_df = df[df["prompt_length"] == config.prompt_length].sort_values("num_tasks")

    is_posterior = config.prompt_length > 0

    if is_posterior:
        indist_df = dist_df[dist_df["prompt_source"] == "memorising"].sort_values(
            "num_tasks"
        )
        ood_df = dist_df[dist_df["prompt_source"] == "generalising"].sort_values(
            "num_tasks"
        )
        if indist_df.empty or ood_df.empty:
            raise ValueError(
                f"no complete posterior rows for prompt_length={config.prompt_length}."
            )

        # Emit both layouts: 2x3 (Delta, ED, SW) preserves the historical figure;
        # 2x2 (Delta, ED) drops SW for the paper-ready version.
        for layout, suffix in [("2x3", ""), ("2x2", "_2x2")]:
            n_cols = 3 if layout == "2x3" else 2
            ylabels = [
                r"$\Delta_{\mathrm{MSE}}$",
                "Energy distance",
                "Sliced Wasserstein",
            ][:n_cols]
            row_labels = ["In-distribution", "Out-of-distribution"]
            figsize = (5.5 * n_cols, 8)

            fig, axes = plt.subplots(
                2, n_cols, figsize=figsize, constrained_layout=True, sharey="col"
            )

            # row 0: in-distribution
            plot_series(
                axes[0, 0],
                pred_df["num_tasks"],
                [
                    "data_memorising/delta_vs_baseline_memorising",
                    "data_memorising/delta_vs_baseline_generalising",
                ],
                pred_df,
            )
            plot_series(
                axes[0, 1],
                indist_df["num_tasks"],
                ["dist/ed_vs_baseline_memorising", "dist/ed_vs_baseline_generalising"],
                indist_df,
            )
            if n_cols == 3:
                plot_series(
                    axes[0, 2],
                    indist_df["num_tasks"],
                    [
                        "dist/sw_vs_baseline_memorising",
                        "dist/sw_vs_baseline_generalising",
                    ],
                    indist_df,
                )

            # row 1: out-of-distribution
            plot_series(
                axes[1, 0],
                pred_df["num_tasks"],
                [
                    "data_generalising/delta_vs_baseline_memorising",
                    "data_generalising/delta_vs_baseline_generalising",
                ],
                pred_df,
            )
            plot_series(
                axes[1, 1],
                ood_df["num_tasks"],
                ["dist/ed_vs_baseline_memorising", "dist/ed_vs_baseline_generalising"],
                ood_df,
            )
            if n_cols == 3:
                plot_series(
                    axes[1, 2],
                    ood_df["num_tasks"],
                    [
                        "dist/sw_vs_baseline_memorising",
                        "dist/sw_vs_baseline_generalising",
                    ],
                    ood_df,
                )

            for col_idx, ylabel in enumerate(ylabels):
                for row_idx in range(2):
                    style_ax(axes[row_idx, col_idx], ylabel)

            for row_idx, row_label in enumerate(row_labels):
                axes[row_idx, 0].annotate(
                    row_label,
                    xy=(-0.28, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=16,
                    rotation=90,
                )

            if axes[0, 0].get_legend_handles_labels()[0]:
                axes[0, -1].legend(loc="upper right", frameon=False, fontsize=14)

            out_path = output_dir / f"sweep_combined{suffix}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return
    else:
        # prior mode: 1 row x 2 cols (ED, SW) -- no prediction metrics
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        ylabels = ["Energy distance", "Sliced Wasserstein"]
        # pandas parses the literal N/A prompt-source marker as a missing value
        prior_df = dist_df[
            dist_df["prompt_source"].isna() | (dist_df["prompt_source"] == "N/A")
        ].sort_values("num_tasks")
        if prior_df.empty:
            raise ValueError("no prior rows found in the metrics table.")

        plot_series(
            axes[0],
            prior_df["num_tasks"],
            ["dist/ed_vs_baseline_memorising", "dist/ed_vs_baseline_generalising"],
            prior_df,
            series=SERIES_PRIOR,
        )
        plot_series(
            axes[1],
            prior_df["num_tasks"],
            ["dist/sw_vs_baseline_memorising", "dist/sw_vs_baseline_generalising"],
            prior_df,
            series=SERIES_PRIOR,
        )

        for col_idx, ylabel in enumerate(ylabels):
            style_ax(axes[col_idx], ylabel)

        if axes[0].get_legend_handles_labels()[0]:
            axes[-1].legend(loc="upper right", frameon=False, fontsize=14)

    out_path = output_dir / "sweep_combined.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(tyro.cli(PlotConfig))
