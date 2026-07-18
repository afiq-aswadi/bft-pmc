"""Combined dynamics plot for the balls-and-urns setting.

Usage:
    uv run python -m balls_and_urns.plot_dynamics_combined --metrics-csv outputs/bau/distribution_dynamics/.../metrics.csv
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


def _add_shared_legend(fig: plt.Figure, ax: plt.Axes, anchor_y: float = 1.06) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, anchor_y),
            ncol=len(handles),
            frameon=False,
        )


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
    for col, (label, fmt, color) in zip(cols, series):
        if df[col].isna().any():
            raise ValueError(f"plot column {col!r} contains missing values.")
        ax.plot(x, df[col], fmt, label=label, color=color)


def _plot_dynamics(df: pd.DataFrame, out_path: Path, log_xscale: bool = False) -> None:
    has_dist_split = (
        "ed_vs_baseline_generalising_from_prompts_generalising" in df.columns
    )
    is_posterior = has_dist_split

    if is_posterior:
        # Emit both layouts: 2x3 (Sym KL, ED, SW) preserves the historical figure;
        # 2x2 (Sym KL, ED) drops SW for the paper-ready version.
        for layout, suffix in [("2x3", ""), ("2x2", "_2x2")]:
            n_cols = 3 if layout == "2x3" else 2
            apply_paper_style(4.2 * n_cols, 0.49 if n_cols == 2 else 0.95)
            fig, axes = plt.subplots(
                2,
                n_cols,
                figsize=(4.2 * n_cols, 6.2),
                constrained_layout=True,
                sharey="col",
            )

            row_labels = ["In-distribution", "Out-of-distribution"]
            ylabels = ["Symmetrized KL", "Energy distance", "Sliced Wasserstein"][
                :n_cols
            ]

            # row 0: in-distribution (memorising prompt source)
            plot_series(
                axes[0, 0],
                df["step"],
                [
                    "delta_vs_baseline_memorising_on_data_memorising",
                    "delta_vs_baseline_generalising_on_data_memorising",
                ],
                df,
            )
            plot_series(
                axes[0, 1],
                df["step"],
                [
                    "ed_vs_baseline_memorising_from_prompts_memorising",
                    "ed_vs_baseline_generalising_from_prompts_memorising",
                ],
                df,
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
                )

            # row 1: out-of-distribution (generalising prompt source)
            plot_series(
                axes[1, 0],
                df["step"],
                [
                    "delta_vs_baseline_memorising_on_data_generalising",
                    "delta_vs_baseline_generalising_on_data_generalising",
                ],
                df,
            )
            plot_series(
                axes[1, 1],
                df["step"],
                [
                    "ed_vs_baseline_memorising_from_prompts_generalising",
                    "ed_vs_baseline_generalising_from_prompts_generalising",
                ],
                df,
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
                )

            for col_idx, ylabel in enumerate(ylabels):
                for row_idx in range(2):
                    style_ax(axes[row_idx, col_idx], ylabel, log_xscale=log_xscale)

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

            _add_shared_legend(fig, axes[0, 0])

            suffixed = out_path.with_name(out_path.stem + suffix + out_path.suffix)
            fig.savefig(suffixed, dpi=300, bbox_inches="tight")
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
            series=SERIES_PRIOR,
        )
        plot_series(
            axes[1],
            df["step"],
            ["sw_vs_baseline_memorising", "sw_vs_baseline_generalising"],
            df,
            series=SERIES_PRIOR,
        )

        for col_idx, ylabel in enumerate(ylabels):
            style_ax(axes[col_idx], ylabel, log_xscale=log_xscale)

        _add_shared_legend(fig, axes[0], anchor_y=1.08)

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
