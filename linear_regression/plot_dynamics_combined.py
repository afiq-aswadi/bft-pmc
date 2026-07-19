"""Combined delta metrics + distribution distance dynamics plot.

Usage:
    uv run python -m linear_regression.plot_dynamics_combined --metrics-csv outputs/lr/distribution_dynamics/num_tasks_32/metrics.csv
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tyro

from plotting.paper_style import apply_paper_style


@dataclass
class PlotConfig:
    metrics_csv: str
    output_dir: str | None = None


SERIES = [
    ("Memorizing", "-", "tab:green"),
    ("Generalizing", "-", "tab:blue"),
]
SERIES_PRIOR = [
    ("Memorizing", "-", "tab:green"),
    ("Generalizing", "-", "tab:blue"),
]


def style_ax(ax: plt.Axes, ylabel: str | None = None, log_xscale: bool = False) -> None:
    ax.set_yscale("log")
    if log_xscale:
        ax.set_xscale("log")
    ax.set_xlabel("Training step")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def plot_series(
    ax: plt.Axes,
    x: pd.Series,
    cols: list[str],
    df: pd.DataFrame,
    series: list[tuple[str, str, str]],
) -> None:
    if df.empty:
        raise ValueError("cannot plot an empty metrics table.")
    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise KeyError(f"missing plot columns: {missing}")
    for col, (label, fmt, color) in zip(cols, series):
        if df[col].isna().any():
            raise ValueError(f"plot column {col!r} contains missing values.")
        ax.plot(x, df[col], fmt, label=label, color=color)


def _plot_dynamics(df: pd.DataFrame, out_path: Path, log_xscale: bool = False) -> None:
    has_per_source_dist = (
        "ed_vs_baseline_memorising_from_prompts_memorising" in df.columns
    )

    if has_per_source_dist:
        # Emit both 2x3 (Delta, ED, SW) and 2x2 (Delta, ED) layouts.
        for layout, suffix in [("2x3", ""), ("2x2", "_2x2")]:
            n_cols = 3 if layout == "2x3" else 2
            ylabels = [
                r"$\Delta_{\mathrm{MSE}}$",
                "Energy distance",
                "Sliced Wasserstein",
            ][:n_cols]
            row_labels = ["In-distribution", "Out-of-distribution"]
            figsize = (4.2 * n_cols, 6.2)

            apply_paper_style(figsize[0], 0.49 if n_cols == 2 else 0.95)
            fig, axes = plt.subplots(
                2, n_cols, figsize=figsize, constrained_layout=True, sharey="col"
            )

            # row 0: in-distribution
            plot_series(
                axes[0, 0],
                df["step"],
                [
                    "delta_vs_baseline_memorising_on_data_memorising",
                    "delta_vs_baseline_generalising_on_data_memorising",
                ],
                df,
                SERIES,
            )
            plot_series(
                axes[0, 1],
                df["step"],
                [
                    "ed_vs_baseline_memorising_from_prompts_memorising",
                    "ed_vs_baseline_generalising_from_prompts_memorising",
                ],
                df,
                SERIES,
            )
            if n_cols == 3:
                plot_series(
                    axes[0, 2],
                    df["step"],
                    [
                        "sw_vs_baseline_memorising_from_prompts_memorising",
                        "sw_vs_baseline_generalising_from_prompts_memorising",
                    ],
                    df,
                    SERIES,
                )

            # row 1: out-of-distribution
            plot_series(
                axes[1, 0],
                df["step"],
                [
                    "delta_vs_baseline_memorising_on_data_generalising",
                    "delta_vs_baseline_generalising_on_data_generalising",
                ],
                df,
                SERIES,
            )
            plot_series(
                axes[1, 1],
                df["step"],
                [
                    "ed_vs_baseline_memorising_from_prompts_generalising",
                    "ed_vs_baseline_generalising_from_prompts_generalising",
                ],
                df,
                SERIES,
            )
            if n_cols == 3:
                plot_series(
                    axes[1, 2],
                    df["step"],
                    [
                        "sw_vs_baseline_memorising_from_prompts_generalising",
                        "sw_vs_baseline_generalising_from_prompts_generalising",
                    ],
                    df,
                    SERIES,
                )

            for col_idx, ylabel in enumerate(ylabels):
                style_ax(axes[0, col_idx], ylabel, log_xscale=log_xscale)
                style_ax(axes[1, col_idx], ylabel, log_xscale=log_xscale)

            for row_idx, row_label in enumerate(row_labels):
                axes[row_idx, 0].annotate(
                    row_label,
                    xy=(-0.55, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=plt.rcParams["axes.labelsize"],
                    rotation=90,
                )

            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.12),
                    ncol=len(handles),
                    frameon=False,
                )

            layout_path = out_path.with_name(out_path.stem + suffix + out_path.suffix)
            fig.savefig(layout_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        return
    else:
        # prior mode: 1 row x 2 cols (ED, SW)
        apply_paper_style(12, 0.95)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        ylabels = ["Energy distance", "Sliced Wasserstein"]

        plot_series(
            axes[0],
            df["step"],
            ["ed_vs_baseline_memorising", "ed_vs_baseline_generalising"],
            df,
            SERIES_PRIOR,
        )
        style_ax(axes[0], ylabels[0], log_xscale=log_xscale)

        plot_series(
            axes[1],
            df["step"],
            ["sw_vs_baseline_memorising", "sw_vs_baseline_generalising"],
            df,
            SERIES_PRIOR,
        )
        style_ax(axes[1], ylabels[1], log_xscale=log_xscale)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.12),
                ncol=len(handles),
                frameon=False,
            )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(config: PlotConfig) -> None:
    csv_path = Path(config.metrics_csv)
    df = pd.read_csv(csv_path)
    output_dir = Path(config.output_dir) if config.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_dynamics(df, output_dir / "dynamics_combined.png")
    _plot_dynamics(df, output_dir / "dynamics_combined_logx.png", log_xscale=True)


if __name__ == "__main__":
    main(tyro.cli(PlotConfig))
