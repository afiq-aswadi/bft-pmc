"""2x3 sweep: KL / Energy distance / Sliced Wasserstein vs M.

Rows = prompt source (In-distribution / Out-of-distribution).
Cols = KL / ED / SW.
Per-panel lines:
  green = vs memorising baseline (training-pool discrete posterior)
  blue  = vs generalising baseline (analytic Dirichlet posterior)

ED and SW are read from the top-level
`outputs/markov/sweep_analysis/metrics.csv`.
KL uses the last-step wellspec values from each run's
`runs/<run>/wandb_kl_history.csv` (`kl/{id,ood}/wellspec/{memorising,generalising}`).

Layout matches the linear-regression and balls-and-urns sweep figures.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from markov.kl_history import load_kl_history
from markov.run_names import parse_n_chains
from plotting.paper_style import apply_paper_style, task_diversity_ticks


SERIES = [
    ("Memorizing", "o-", "tab:green", 5),
    ("Generalizing", "s-", "tab:blue", 5),
]


@dataclass(slots=True)
class SweepRow:
    n_chains: int
    kl_vs_memorising: float
    kl_vs_generalising: float
    ed_vs_memorising: float
    ed_vs_generalising: float
    sw_vs_memorising: float
    sw_vs_generalising: float



ALL_PANELS: tuple[tuple[str, str, str], ...] = (
    ("Symmetrised KL", "kl_vs_memorising", "kl_vs_generalising"),
    ("Energy distance", "ed_vs_memorising", "ed_vs_generalising"),
    ("Sliced Wasserstein", "sw_vs_memorising", "sw_vs_generalising"),
)


def available_panels(
    rows_by_source: dict[str, list[SweepRow]],
) -> list[tuple[str, str, str]]:
    """Panels whose metric is present in at least one row."""
    every_row = [row for rows in rows_by_source.values() for row in rows]
    return [
        spec
        for spec in ALL_PANELS
        if any(not math.isnan(getattr(row, spec[1])) for row in every_row)
    ]


def _last_step_kl(history: pd.DataFrame | None, source: str) -> tuple[float, float]:
    """Last-step (memorising_kl, generalising_kl) for a run, using wellspec.

    Returns NaNs when neither KL source exists, so the figure drops that column
    rather than failing. See :mod:`markov.kl_history` for the two sources.
    """
    if history is None:
        return (float("nan"), float("nan"))
    df = history
    src = "id" if source == "in_distribution" else "ood"
    last = df.iloc[-1]
    return (
        float(last[f"kl/{src}/wellspec/memorising"]),
        float(last[f"kl/{src}/wellspec/generalising"]),
    )


def _build_rows(
    runs_dir: Path, metrics_csv: Path, training_root: Path | None = None
) -> dict[str, list[SweepRow]]:
    metrics = pd.read_csv(metrics_csv)
    metrics["n_chains"] = metrics["n_chains"].astype(int)

    run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())

    rows_by_source: dict[str, list[SweepRow]] = {
        "in_distribution": [],
        "out_of_distribution": [],
    }
    for run_dir in run_dirs:
        n = parse_n_chains(run_dir.name)
        history = load_kl_history(run_dir, training_root=training_root)

        for source in ("in_distribution", "out_of_distribution"):
            kl_mem, kl_gen = _last_step_kl(history, source)

            mask = (metrics["n_chains"] == n) & (metrics["prompt_source"] == source)
            sub = metrics[mask]
            assert len(sub) == 1, (
                f"expected one metrics row for M={n} source={source}, got {len(sub)}"
            )
            metric_row = sub.iloc[0]
            ed_mem = float(metric_row["dist/ed_vs_baseline_in_distribution"])
            ed_gen = float(metric_row["dist/ed_vs_baseline_out_of_distribution"])
            sw_mem = float(metric_row["dist/sw_vs_baseline_in_distribution"])
            sw_gen = float(metric_row["dist/sw_vs_baseline_out_of_distribution"])

            rows_by_source[source].append(
                SweepRow(
                    n_chains=n,
                    kl_vs_memorising=kl_mem,
                    kl_vs_generalising=kl_gen,
                    ed_vs_memorising=ed_mem,
                    ed_vs_generalising=ed_gen,
                    sw_vs_memorising=sw_mem,
                    sw_vs_generalising=sw_gen,
                )
            )
    for source in rows_by_source:
        rows_by_source[source].sort(key=lambda r: r.n_chains)
        if not rows_by_source[source]:
            raise ValueError(f"no complete rows found for {source} prompts.")
    return rows_by_source


def _style_ax(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_xscale("log", base=2)
    task_diversity_ticks(ax.xaxis)
    ax.set_yscale("log")
    ax.set_xlabel(r"$M$")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")


def _plot_panel(
    ax: plt.Axes,
    rows: list[SweepRow],
    *,
    mem_attr: str,
    gen_attr: str,
) -> None:
    if not rows:
        raise ValueError("cannot plot an empty Markov sweep.")
    xs = np.array([r.n_chains for r in rows])
    series_data = [
        np.array([getattr(r, mem_attr) for r in rows]),
        np.array([getattr(r, gen_attr) for r in rows]),
    ]
    for ys, (label, fmt, color, ms) in zip(series_data, SERIES):
        ax.plot(xs, ys, fmt, label=label, color=color, markersize=ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("outputs/markov/sweep_analysis/runs"),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("outputs/markov/sweep_analysis/metrics.csv"),
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs/markov/sweep_analysis/sweep_combined_with_kl.png"),
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=None,
        help="Root holding <run>/training_log.csv; used when a run has no "
        "wandb_kl_history.csv. Default: outputs/markov/training.",
    )
    args = parser.parse_args()

    rows_by_source = _build_rows(args.runs_dir, args.metrics_csv, args.training_root)

    panels = available_panels(rows_by_source)
    if len(panels) < len(ALL_PANELS):
        missing = [spec[0] for spec in ALL_PANELS if spec not in panels]
        print(
            f"[markov-sweep] no data for {', '.join(missing)}; "
            f"emitting {len(panels)} columns instead of {len(ALL_PANELS)}.",
            file=sys.stderr,
        )

    # Emit the full figure and, when there is a column to spare, the narrower
    # paper-ready twin that drops the last metric.
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    layouts = [(len(panels), "")]
    if len(panels) > 2:
        layouts.append((len(panels) - 1, "_2x2"))

    for n_cols, suffix in layouts:
        apply_paper_style(4.2 * n_cols, 0.49 if n_cols == 2 else 0.95)
        fig, axes = plt.subplots(
            2, n_cols, figsize=(4.2 * n_cols, 6.2), constrained_layout=True, sharey="col"
        )

        row_labels = ["In-distribution", "Out-of-distribution"]
        shown = panels[:n_cols]

        for row_idx, source in enumerate(("in_distribution", "out_of_distribution")):
            rows = rows_by_source[source]
            for col_idx, (_, mem_attr, gen_attr) in enumerate(shown):
                _plot_panel(
                    axes[row_idx, col_idx],
                    rows,
                    mem_attr=mem_attr,
                    gen_attr=gen_attr,
                )

        for col_idx, (ylabel, _, _) in enumerate(shown):
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

        suffixed = args.out_path.with_name(
            args.out_path.stem + suffix + args.out_path.suffix
        )
        fig.savefig(suffixed, dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
