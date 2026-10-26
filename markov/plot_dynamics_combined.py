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

Beside each 2x3 figure it writes two paper-ready 2x2 twins:
`dynamics_M{n}_2x2` (KL, energy distance) and `dynamics_M{n}_2x2_sw` (KL,
sliced Wasserstein).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from markov.kl_history import load_kl_history
from markov.run_names import parse_n_chains
from plotting.paper_style import apply_paper_style, every_decade_ticks


SERIES = [
    ("Memorizing", "-", "tab:green"),
    ("Generalizing", "-", "tab:blue"),
]

# Figure-name suffix and the metric of each column: the full 2x3 figure, then
# the two paper-ready 2x2 twins pairing KL with one posterior metric each.
LAYOUTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("", ("kl", "ed", "sw")),
    ("_2x2", ("kl", "ed")),
    ("_2x2_sw", ("kl", "sw")),
)
YLABELS = {"kl": "Symmetrised KL", "ed": "Energy distance", "sw": "Sliced Wasserstein"}


def _style_ax(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_yscale("log")
    ax.set_xscale("log")
    # These runs log from step 1, so the axis spans six decades in a
    # half-width panel; every decade would run the labels together.
    every_decade_ticks(ax.xaxis, max_ticks=4)
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
    training_root: Path | None = None,
) -> None:
    metrics_path = run_dir / "metrics.csv"
    df = pd.read_csv(metrics_path).sort_values("step")
    kl_df = load_kl_history(run_dir, training_root=training_root)
    if kl_df is None:
        raise FileNotFoundError(
            f"no KL history for {run_dir.name}: expected "
            f"{run_dir / 'wandb_kl_history.csv'} or a training_log.csv under "
            "the training root (see markov.kl_history)."
        )

    for suffix, columns in LAYOUTS:
        n_cols = len(columns)
        apply_paper_style(4.2 * n_cols, 0.49 if n_cols == 2 else 0.95)
        fig, axes = plt.subplots(
            2, n_cols, figsize=(4.2 * n_cols, 6.2), constrained_layout=True, sharey="col"
        )

        row_labels = ["In-distribution", "Out-of-distribution"]

        for row_idx, source in enumerate(("in_distribution", "out_of_distribution")):
            sub = _filter_prompt_source(df, source)
            for col_idx, metric in enumerate(columns):
                if metric == "kl":
                    _plot_kl(axes[row_idx, col_idx], kl_df, source)
                else:
                    _plot_metric(axes[row_idx, col_idx], sub, metric=metric)

        for col_idx, metric in enumerate(columns):
            for row_idx in range(2):
                _style_ax(axes[row_idx, col_idx], YLABELS[metric])

        for row_idx, row_label in enumerate(row_labels):
            axes[row_idx, 0].annotate(
                row_label,
                xy=(-0.72, 0.5),
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
    parser.add_argument(
        "--training-root",
        type=Path,
        default=None,
        help="Root holding <run>/training_log.csv; used when a run has no "
        "wandb_kl_history.csv. Default: outputs/markov/training.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(path.parent for path in args.runs_dir.glob("*/metrics.csv"))
    assert run_dirs, f"no run metrics found under {args.runs_dir}"
    for run_dir in run_dirs:
        n_chains = parse_n_chains(run_dir.name)
        _process_run(
            run_dir, args.out_dir, n_chains=n_chains, training_root=args.training_root
        )


if __name__ == "__main__":
    main()
