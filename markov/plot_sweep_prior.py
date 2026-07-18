"""Plot the Markov prior sweep as KL, ED, and SW panels.

KL is the predictive-space metric (model's first-step prediction at length-1
context vs the analytic memorising/generalising priors). ED and SW are the
distribution-space metrics over transition-matrix samples from Predictive Monte Carlo.
The two live in different geometries, so KL comes from a sidecar CSV produced
by `scripts/compute_markov_prior_predictive_kl.py`; ED and SW come from the
prior rows emitted by `markov/sweep_analysis.py`.

The three panels use the same styling as the posterior sweep figure.
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


def _style_ax(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$M$")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def _plot_panel(ax: plt.Axes, df: pd.DataFrame, cols: list[str]) -> None:
    for col, (label, fmt, color, ms) in zip(cols, SERIES):
        ax.plot(df["n_chains"], df[col], fmt, label=label, color=color, markersize=ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("outputs/markov/sweep_analysis/metrics_prior_n1024.csv"),
        help="CSV produced by markov/sweep_analysis.py (prior rows).",
    )
    parser.add_argument(
        "--kl-csv",
        type=Path,
        default=None,
        help="Optional CSV from scripts/compute_markov_prior_predictive_kl.py.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs/markov/sweep_analysis/sweep_prior_kl_ed_sw.png"),
    )
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics_csv)
    prior = metrics[
        (metrics["prompt_source"].isna() | (metrics["prompt_source"] == "N/A"))
        & (metrics["prompt_length"] == 0)
    ].sort_values("n_chains")
    assert not prior.empty, f"no prior rows in {args.metrics_csv}"

    kl = None
    if args.kl_csv is not None:
        kl = pd.read_csv(args.kl_csv).sort_values("n_chains")
        assert not kl.empty, f"no KL rows in {args.kl_csv}"
        common_m = sorted(set(prior["n_chains"]).intersection(kl["n_chains"]))
        assert common_m, "prior metrics and KL sidecar have no chain counts in common"
        prior = prior[prior["n_chains"].isin(common_m)].sort_values("n_chains")
        kl = kl[kl["n_chains"].isin(common_m)].sort_values("n_chains")

    num_columns = 3 if kl is not None else 2
    apply_paper_style(6 * num_columns, 0.95)

    fig, axes = plt.subplots(
        1,
        num_columns,
        figsize=(6 * num_columns, 4),
        constrained_layout=True,
    )
    offset = 0
    if kl is not None:
        _plot_panel(axes[0], kl, ["kl_vs_memorising", "kl_vs_generalising"])
        _style_ax(axes[0], "Symmetrised KL")
        offset = 1

    _plot_panel(
        axes[offset],
        prior,
        [
            "dist/ed_vs_baseline_in_distribution",
            "dist/ed_vs_baseline_out_of_distribution",
        ],
    )
    _plot_panel(
        axes[offset + 1],
        prior,
        [
            "dist/sw_vs_baseline_in_distribution",
            "dist/sw_vs_baseline_out_of_distribution",
        ],
    )

    _style_ax(axes[offset], "Energy distance")
    _style_ax(axes[offset + 1], "Sliced Wasserstein")

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

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
