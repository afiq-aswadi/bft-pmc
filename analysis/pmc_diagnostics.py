"""Shared machinery for PMC rollout-length and rollout-count diagnostics.

Family modules (balls_and_urns/rollout_diagnostics.py,
markov/rollout_diagnostics.py) produce running rollout-statistic arrays of
shape (num_rollouts, len(grid), dim); everything here operates on those flat
sample matrices: recovery-error curves against baseline sample sets, the
path-stability drift diagnostic, the rollout-count subsampling panel with a
fresh-draw Monte Carlo floor, and the corresponding figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import energy_distance_multidim, sliced_wasserstein


def distance_metrics(
    model_samples: np.ndarray,
    baseline_samples: dict[str, np.ndarray],
    n_projections: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, references in baseline_samples.items():
        metrics[f"dist/ed_vs_{name}"] = energy_distance_multidim(
            model_samples,
            references,
        )
        metrics[f"dist/sw_vs_{name}"] = sliced_wasserstein(
            model_samples,
            references,
            n_projections=n_projections,
        )
    return metrics


def recovery_curves(
    stats: np.ndarray,
    baselines: dict[str, np.ndarray],
    grid: list[int],
    n_projections: int,
) -> pd.DataFrame:
    rows = []
    for j, n in enumerate(grid):
        metrics = distance_metrics(stats[:, j, :], baselines, n_projections)
        rows.append({"rollout_length": n, **metrics})
    return pd.DataFrame(rows)


def drift_curves(
    stats: np.ndarray,
    grid: list[int],
    anchor_stat: np.ndarray,
    scale: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Scaled L1 drift from the anchor estimate at each grid length.

    anchor_stat: (num_rollouts, dim) for a per-path anchor, or (dim,) for a
    shared anchor. scale defaults to the statistic dimension; pass the free
    parameter count when that differs (e.g. transition matrices).
    Returns the summary frame and the per-path drift matrix (R, len(grid)).
    """
    if scale is None:
        scale = stats.shape[-1]
    if anchor_stat.ndim == 1:
        anchor = anchor_stat[None, None, :]
    else:
        anchor = anchor_stat[:, None, :]
    per_path = np.abs(stats - anchor).sum(axis=-1) / scale

    frame = pd.DataFrame(
        {
            "rollout_length": grid,
            "drift_mean": per_path.mean(axis=0),
            "drift_q10": np.quantile(per_path, 0.10, axis=0),
            "drift_q50": np.quantile(per_path, 0.50, axis=0),
            "drift_q90": np.quantile(per_path, 0.90, axis=0),
        }
    )
    return frame, per_path


def count_curves(
    stat_final: np.ndarray,
    baselines: dict[str, np.ndarray],
    fresh_baseline_sampler: Callable[[int], dict[str, np.ndarray]],
    count_grid: list[int],
    n_resamples: int,
    n_projections: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Recovery error vs rollout count R, with a fresh-draw Monte Carlo floor."""
    rows = []
    total = stat_final.shape[0]
    for r in count_grid:
        resamples = 1 if r == total else n_resamples
        model_metrics: dict[str, list[float]] = {}
        floor_metrics: dict[str, list[float]] = {}
        for _ in range(resamples):
            index = rng.choice(total, size=r, replace=False)
            for key, value in distance_metrics(
                stat_final[index], baselines, n_projections
            ).items():
                model_metrics.setdefault(key, []).append(value)
            fresh = fresh_baseline_sampler(r)
            for name, reference in baselines.items():
                for key, value in distance_metrics(
                    fresh[name], {name: reference}, n_projections
                ).items():
                    floor_metrics.setdefault(key, []).append(value)
        row: dict[str, float] = {"num_rollouts": r}
        for key, values in model_metrics.items():
            row[f"{key}/mean"] = float(np.mean(values))
            row[f"{key}/sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for key, values in floor_metrics.items():
            row[f"floor/{key}/mean"] = float(np.mean(values))
            row[f"floor/{key}/sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_recovery(frame: pd.DataFrame, out_stem: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for ax, metric, label in zip(
        axes, ["ed", "sw"], ["Energy distance", "Sliced Wasserstein-1"]
    ):
        for baseline, style in [
            ("baseline_generalising", "-o"),
            ("baseline_memorising", "--s"),
        ]:
            ax.plot(
                frame["rollout_length"],
                frame[f"dist/{metric}_vs_{baseline}"],
                style,
                markersize=3,
                label=baseline.replace("baseline_", ""),
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("rollout length $N$")
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_drift(
    frame: pd.DataFrame,
    per_path: np.ndarray,
    grid: list[int],
    out_stem: Path,
    title: str,
    max_traces: int = 200,
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    shown = min(max_traces, per_path.shape[0])
    ax.plot(grid, per_path[:shown].T, color="tab:blue", alpha=0.05, linewidth=0.6)
    ax.plot(
        frame["rollout_length"],
        frame["drift_mean"],
        color="black",
        linewidth=1.8,
        label="mean",
    )
    ax.fill_between(
        frame["rollout_length"],
        frame["drift_q10"],
        frame["drift_q90"],
        color="black",
        alpha=0.12,
        label="10–90%",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rollout length $N$")
    ax.set_ylabel("scaled L1 drift from anchor")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_counts(frame: pd.DataFrame, out_stem: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for ax, metric, label in zip(
        axes, ["ed", "sw"], ["Energy distance", "Sliced Wasserstein-1"]
    ):
        for baseline, color in [
            ("baseline_generalising", "tab:blue"),
            ("baseline_memorising", "tab:orange"),
        ]:
            key = f"dist/{metric}_vs_{baseline}"
            ax.errorbar(
                frame["num_rollouts"],
                frame[f"{key}/mean"],
                yerr=frame[f"{key}/sd"],
                fmt="-o",
                markersize=3,
                color=color,
                label=baseline.replace("baseline_", ""),
            )
            ax.errorbar(
                frame["num_rollouts"],
                frame[f"floor/{key}/mean"],
                yerr=frame[f"floor/{key}/sd"],
                fmt=":",
                color=color,
                alpha=0.7,
                label=f"{baseline.replace('baseline_', '')} floor",
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("rollout count $R$")
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)
