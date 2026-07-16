"""1x3 BAU prior sweep: symmetrised KL / Energy distance / Sliced Wasserstein vs M.

KL is the predictive-space metric (model's first-step prediction at length-1
context vs the analytic memorising/generalising priors). ED and SW are the
distribution-space metrics over theta samples from Predictive Monte Carlo. The
two live in different geometries, so KL comes from a sidecar CSV produced by
`scripts/compute_bau_prior_predictive_kl.py`; ED and SW come from the prior
rows of the sweep `metrics.csv`.

Layout matches the prior panels in `balls_and_urns/plot_sweep_combined.py`,
with a KL panel inserted on the left.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SERIES = [
    (r"$\Pi_M$", "s-", "tab:green", 5),
    (r"$\Pi_\infty$", "o-", "tab:blue", 5),
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
    ax.set_xlabel(r"$M$", fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=13)
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
        help="metrics.csv from balls_and_urns/sweep_analysis.py (must contain prior rows).",
    )
    parser.add_argument(
        "--kl-csv",
        type=Path,
        default=Path("outputs/bau/sweep_analysis/prior_predictive_kl.csv"),
        help="CSV produced by scripts/compute_bau_prior_predictive_kl.py.",
    )
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics_csv)
    prior_df = metrics[metrics["prompt_source"] == "prior"].sort_values("num_tasks")
    assert not prior_df.empty, f"no prior rows in {args.metrics_csv}"

    kl_df = pd.read_csv(args.kl_csv).sort_values("num_tasks")
    assert not kl_df.empty, f"no KL rows in {args.kl_csv}"

    common_M = sorted(set(prior_df["num_tasks"]).intersection(kl_df["num_tasks"]))
    assert common_M, "prior metrics and KL sidecar have no task counts in common"
    prior_df = prior_df[prior_df["num_tasks"].isin(common_M)].sort_values("num_tasks")
    kl_df = kl_df[kl_df["num_tasks"].isin(common_M)].sort_values("num_tasks")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

    _plot_panel(
        axes[0], kl_df["num_tasks"], kl_df, ["kl_vs_memorising", "kl_vs_generalising"]
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

    ed_cols = ["dist/ed_vs_baseline_memorising", "dist/ed_vs_baseline_generalising"]
    sw_cols = ["dist/sw_vs_baseline_memorising", "dist/sw_vs_baseline_generalising"]
    # the unbiased ED estimate can dip below zero near convergence; check each
    # distance panel and use symlog only when its data is non-positive
    ed_thresh = 1e-3 if (prior_df[ed_cols].to_numpy() <= 0).any() else None
    sw_thresh = 1e-3 if (prior_df[sw_cols].to_numpy() <= 0).any() else None

    _style_ax(axes[0], "Symmetrised KL")
    _style_ax(axes[1], "Energy distance", symlog_thresh=ed_thresh)
    _style_ax(axes[2], "Sliced Wasserstein", symlog_thresh=sw_thresh)

    if axes[0].get_legend_handles_labels()[0]:
        axes[-1].legend(loc="upper right", frameon=False, fontsize=14)

    out_path = args.out_path or args.metrics_csv.parent / "sweep_prior_kl_ed_sw.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
