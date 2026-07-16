"""Combined sweep plot for the balls-and-urns setting.

Usage:
    uv run python -m balls_and_urns.plot_sweep_combined --metrics-csv outputs/bau/sweep_analysis/.../metrics.csv
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


SERIES = [
    ("Memorizing", "s-", "tab:green", 5),
    ("Generalizing", "o-", "tab:blue", 5),
]
SERIES_PRIOR = [
    (r"$\Pi_M$", "s-", "tab:green", 5),
    (r"$\Pi_\infty$", "o-", "tab:blue", 5),
]


def style_ax(
    ax: plt.Axes,
    ylabel: str | None = None,
    log_scale: bool = True,
    symlog_thresh: float | None = None,
) -> None:
    ax.set_xscale("log", base=2)
    if symlog_thresh is not None:
        ax.set_yscale("symlog", linthresh=symlog_thresh)
    elif log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(r"$M$", fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def _add_inset_legend(target_ax: plt.Axes, source_ax: plt.Axes) -> None:
    handles, labels = source_ax.get_legend_handles_labels()
    if handles:
        target_ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=14)


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

    # delta metrics: deduped per run (independent of prompt_source)
    pred_df = df.drop_duplicates(
        subset=["run_id", "num_tasks", "checkpoint_step"]
    ).sort_values("num_tasks")

    indist_df = df[df["prompt_source"] == "data_memorising"].sort_values("num_tasks")
    ood_df = df[df["prompt_source"] == "data_generalising"].sort_values("num_tasks")
    prior_df = df[df["prompt_source"] == "prior"].sort_values("num_tasks")

    if indist_df.empty != ood_df.empty:
        raise ValueError("BAU posterior metrics require both prompt sources.")
    if indist_df.empty and prior_df.empty:
        raise ValueError("no posterior or prior rows found in the metrics table.")

    if not indist_df.empty and not ood_df.empty:
        # Emit both layouts: 2x3 (Sym KL, ED, SW) preserves the historical figure;
        # 2x2 (Sym KL, ED) drops SW for the paper-ready version.
        for layout, suffix in [("2x3", ""), ("2x2", "_2x2")]:
            n_cols = 3 if layout == "2x3" else 2
            fig, axes = plt.subplots(
                2,
                n_cols,
                figsize=(6 * n_cols, 8),
                constrained_layout=True,
                sharey="col",
            )
            row_labels = ["In-distribution", "Out-of-distribution"]
            ylabels = ["Symmetrised KL", "Energy distance", "Sliced Wasserstein"][
                :n_cols
            ]

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

            _add_inset_legend(axes[0, -1], axes[0, 0])

            out_path = output_dir / f"sweep_posterior{suffix}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    if not prior_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
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
        ed_cols = ["dist/ed_vs_baseline_memorising", "dist/ed_vs_baseline_generalising"]
        sw_cols = ["dist/sw_vs_baseline_memorising", "dist/sw_vs_baseline_generalising"]
        # the unbiased ED estimate can dip below zero near convergence; check
        # each distance panel and use symlog only when its data is non-positive
        thresholds = [
            1e-3 if (prior_df[ed_cols].to_numpy() <= 0).any() else None,
            1e-3 if (prior_df[sw_cols].to_numpy() <= 0).any() else None,
        ]
        for col_idx, (ylabel, thresh) in enumerate(
            zip(["Energy distance", "Sliced Wasserstein"], thresholds)
        ):
            style_ax(axes[col_idx], ylabel, symlog_thresh=thresh)

        _add_inset_legend(axes[-1], axes[0])

        out_path = output_dir / "sweep_prior.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main(tyro.cli(PlotConfig))
