"""Plot KL + ED + SW dynamics across training time per M.

Reads per-run CSVs from
`outputs/markov/sweep_analysis/runs/markov_*_chains{n}_*/`:
  - metrics.csv          ED/SW per step, with prompt-source variants
  - wandb_kl_history.csv symmetric KL per step against the four baselines

Produces one 2x3 figure per M, matching the LR / BAU layout
(`linear_regression/plot_dynamics_combined.py`, `balls_and_urns/plot_dynamics_combined.py`):
  rows = In-distribution / Out-of-distribution prompts
  cols = KL / Energy distance / Sliced Wasserstein
  per-panel lines:
    green = memorising baseline (training-pool discrete posterior)
    blue  = generalising baseline (analytic Dirichlet posterior)

KL uses the wellspec (order-1 bigram) baselines: `kl/{id,ood}/wellspec/{memorising,generalising}`.
Training step axis is always log-scaled because runs span many decades.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plotting.paper_style import apply_paper_style


SERIES = [
    ("Memorizing", "-", "tab:green"),
    ("Generalizing", "-", "tab:blue"),
]


def _style_ax(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def _plot_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    metric: str,
) -> None:
    mem_col = f"{metric}_vs_baseline_in_distribution"
    gen_col = f"{metric}_vs_baseline_out_of_distribution"
    cols = [mem_col, gen_col]
    for col, (label, fmt, color) in zip(cols, SERIES):
        if col not in df.columns:
            raise KeyError(f"missing plot column: {col}")
        if df[col].isna().any():
            raise ValueError(f"plot column {col!r} contains missing values.")
        ax.plot(df["step"], df[col], fmt, label=label, color=color)


def _plot_kl(
    ax: plt.Axes,
    df: pd.DataFrame,
    source: str,
) -> None:
    src = "id" if source == "in_distribution" else "ood"
    cols = [f"kl/{src}/wellspec/memorising", f"kl/{src}/wellspec/generalising"]
    for col, (label, fmt, color) in zip(cols, SERIES):
        ax.plot(df["step"], df[col], fmt, label=label, color=color)


def _filter_prompt_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Project the per-prompt-source columns under the canonical metric names."""
    out = df[["step", "n_chains"]].copy()
    for metric in ("ed", "sw"):
        for baseline in ("in_distribution", "out_of_distribution"):
            src_col = f"{metric}_vs_baseline_{baseline}_from_prompts_{source}"
            dst_col = f"{metric}_vs_baseline_{baseline}"
            if src_col not in df.columns:
                raise KeyError(f"missing prompt-source column: {src_col}")
            out[dst_col] = df[src_col]
    return out


def _process_run(
    run_dir: Path,
    out_dir: Path,
    *,
    n_chains: int,
) -> None:
    metrics_path = run_dir / "metrics.csv"
    wandb_path = run_dir / "wandb_kl_history.csv"
    df = pd.read_csv(metrics_path).sort_values("step")
    kl_df = pd.read_csv(wandb_path).sort_values("step")

    # Emit both layouts: 2x3 (KL, ED, SW) preserves the historical figure;
    # 2x2 (KL, ED) drops SW for the paper-ready version.
    for layout, suffix in [("2x3", ""), ("2x2", "_2x2")]:
        n_cols = 3 if layout == "2x3" else 2
        apply_paper_style(4.2 * n_cols, 0.49 if n_cols == 2 else 0.95)
        fig, axes = plt.subplots(
            2, n_cols, figsize=(4.2 * n_cols, 6.2), constrained_layout=True, sharey="col"
        )

        row_labels = ["In-distribution", "Out-of-distribution"]
        ylabels = ["Symmetrized KL", "Energy distance", "Sliced Wasserstein"][:n_cols]

        for row_idx, source in enumerate(("in_distribution", "out_of_distribution")):
            sub = _filter_prompt_source(df, source)
            _plot_kl(axes[row_idx, 0], kl_df, source)
            _plot_metric(axes[row_idx, 1], sub, metric="ed")
            if n_cols == 3:
                _plot_metric(axes[row_idx, 2], sub, metric="sw")

        for col_idx, ylabel in enumerate(ylabels):
            for row_idx in range(2):
                _style_ax(axes[row_idx, col_idx], ylabel)

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

        handles, labels = axes[0, 1].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.12),
                ncol=len(handles),
                frameon=False,
            )

        out_path = out_dir / f"dynamics_M{n_chains}{suffix}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("outputs/markov/sweep_analysis/runs"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/markov/distribution_dynamics"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(path.parent for path in args.runs_dir.glob("*/metrics.csv"))
    assert run_dirs, f"no run metrics found under {args.runs_dir}"
    for run_dir in run_dirs:
        token = next(
            (t for t in run_dir.name.split("_") if t.startswith("chains")), None
        )
        if token is None:
            raise ValueError(
                f"cannot parse n_chains from run directory {run_dir.name!r}"
            )
        n_chains = int(token.removeprefix("chains"))
        _process_run(run_dir, args.out_dir, n_chains=n_chains)


if __name__ == "__main__":
    main()
