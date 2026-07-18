"""1x3 LR prior sweep: zero-context Delta MSE / Energy distance / Sliced Wasserstein vs M.

Delta MSE is the prediction-space metric (model's first-step prediction at length-0
context vs the analytic memorising/generalising priors). ED and SW are the
distribution-space metrics over theta samples from Predictive Monte Carlo. Delta
comes from a sidecar CSV produced by `scripts/compute_lr_prior_delta_mse.py`;
ED and SW come from the prior rows of the sweep `metrics.csv`.

Layout matches the prior panels in `linear_regression/plot_sweep_combined.py`,
with a Delta MSE panel inserted on the left.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plotting.paper_style import apply_paper_style


SERIES = [
    ("Memorizing", "o-", "tab:green", 5),
    ("Generalizing", "s-", "tab:blue", 5),
]


def _style_ax(
    ax: plt.Axes,
    ylabel: str | None = None,
    symlog_thresh: float | None = None,
) -> None:
    ax.set_xscale("log", base=2)
    if symlog_thresh is not None:
        ax.set_yscale("symlog", linthresh=symlog_thresh)
    else:
        ax.set_yscale("log")
    ax.set_xlabel(r"$M$")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def _plot_panel(ax: plt.Axes, x: pd.Series, df: pd.DataFrame, cols: list[str]) -> None:
    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise KeyError(f"missing plot columns: {missing}")
    for col, (label, fmt, color, ms) in zip(cols, SERIES):
        if df[col].isna().any():
            raise ValueError(f"plot column {col!r} contains missing values.")
        ax.plot(x, df[col], fmt, label=label, color=color, markersize=ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        required=True,
        help="metrics.csv from linear_regression/sweep_analysis.py.",
    )
    parser.add_argument(
        "--delta-csv",
        type=Path,
        default=Path("outputs/lr/sweep_analysis/prior_delta_mse.csv"),
        help="CSV produced by scripts/compute_lr_prior_delta_mse.py.",
    )
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics_csv)
    prior_df = metrics[metrics["prompt_length"] == 0].sort_values("num_tasks")
    assert not prior_df.empty, f"no prior rows in {args.metrics_csv}"

    delta_df = pd.read_csv(args.delta_csv).sort_values("num_tasks")
    assert not delta_df.empty, f"no delta rows in {args.delta_csv}"

    common_M = sorted(set(prior_df["num_tasks"]).intersection(delta_df["num_tasks"]))
    assert common_M, "prior metrics and delta sidecar have no task counts in common"
    prior_df = prior_df[prior_df["num_tasks"].isin(common_M)].sort_values("num_tasks")
    delta_df = delta_df[delta_df["num_tasks"].isin(common_M)].sort_values("num_tasks")

    apply_paper_style(18, 0.95)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

    _plot_panel(
        axes[0],
        delta_df["num_tasks"],
        delta_df,
        ["delta_vs_memorising", "delta_vs_generalising"],
    )
    _plot_panel(
        axes[1],
        prior_df["num_tasks"],
        prior_df,
        ["dist/ed_vs_baseline_memorising", "dist/ed_vs_baseline_generalising"],
    )
    _plot_panel(
        axes[2],
        prior_df["num_tasks"],
        prior_df,
        ["dist/sw_vs_baseline_memorising", "dist/sw_vs_baseline_generalising"],
    )

    delta_cols = ["delta_vs_memorising", "delta_vs_generalising"]
    ed_cols = ["dist/ed_vs_baseline_memorising", "dist/ed_vs_baseline_generalising"]
    sw_cols = ["dist/sw_vs_baseline_memorising", "dist/sw_vs_baseline_generalising"]
    delta_thresh = 1e-6 if (delta_df[delta_cols].to_numpy() <= 0).any() else None
    ed_thresh = 1e-3 if (prior_df[ed_cols].to_numpy() <= 0).any() else None
    sw_thresh = 1e-3 if (prior_df[sw_cols].to_numpy() <= 0).any() else None

    _style_ax(axes[0], r"$\Delta_{\mathrm{MSE}}$", symlog_thresh=delta_thresh)
    _style_ax(axes[1], "Energy distance", symlog_thresh=ed_thresh)
    _style_ax(axes[2], "Sliced Wasserstein", symlog_thresh=sw_thresh)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.16),
            ncol=len(handles),
            frameon=False,
        )

    out_path = args.out_path or args.metrics_csv.parent / "sweep_prior_delta_ed_sw.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
