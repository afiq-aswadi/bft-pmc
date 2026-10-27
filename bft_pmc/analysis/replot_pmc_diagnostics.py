"""Redraw PMC diagnostics figures from the data their runs saved.

Every figure written by ``bft_pmc/balls_and_urns/rollout_diagnostics.py``,
``bft_pmc/markov/rollout_diagnostics.py`` and ``bft_pmc/analysis/dimension_sweep.py`` is
registered in its directory's ``figures.json`` together with its title and the
files it was drawn from. This reads those entries and draws each figure again
in place, through the same plotting functions, so restyling a figure needs no
checkpoint, rollouts or GPU.

Usage:
    uv run eval.py replot-pmc-diagnostics --roots outputs/pmc_diagnostics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import tyro

from bft_pmc.analysis import dimension_sweep
from bft_pmc.analysis.pmc_diagnostics import (
    FIGURE_INDEX,
    plot_counts,
    plot_drift,
    plot_recovery,
)


@dataclass
class ReplotConfig:
    """Directories to search, recursively, for ``figures.json`` indexes."""

    roots: list[str] = field(default_factory=lambda: ["outputs/pmc_diagnostics"])


def find_figure_indexes(roots: list[str]) -> list[Path]:
    """Every ``figures.json`` at or below the given roots, in path order."""
    found: set[Path] = set()
    for root in roots:
        found.update(Path(root).rglob(FIGURE_INDEX))
    return sorted(found)


def replot_figure(directory: Path, stem: str, spec: dict[str, Any]) -> None:
    """Draw one indexed figure again from its saved data.

    Each figure is drawn inside its own rc context: the dimension plot applies
    the paper style globally, which would otherwise restyle every figure
    redrawn after it in the same process.
    """
    kind = spec["kind"]
    title = spec["title"]
    out_stem = directory / stem
    with matplotlib.rc_context():
        if kind == "recovery":
            plot_recovery(pd.read_csv(directory / spec["csv"]), out_stem, title)
        elif kind == "drift":
            with np.load(directory / spec["paths"]) as archive:
                per_path = archive["per_path_drift"]
                grid = [int(length) for length in archive["rollout_lengths"]]
            frame = pd.read_csv(directory / spec["csv"])
            plot_drift(frame, per_path, grid, out_stem, title)
        elif kind == "counts":
            plot_counts(pd.read_csv(directory / spec["csv"]), out_stem, title)
        elif kind in dimension_sweep.AXIS_BY_KIND:
            frame = pd.read_csv(directory / spec["csv"])
            dimension_sweep.plot(
                frame,
                list(spec["sources"]),
                out_stem,
                title,
                dimension_sweep.AXIS_BY_KIND[kind],
            )
        else:
            raise ValueError(f"unknown figure kind {kind!r} for {stem} in {directory}")


def replot_directory(directory: Path) -> list[str]:
    """Redraw every figure indexed in one directory; returns their stems."""
    index = json.loads((directory / FIGURE_INDEX).read_text())
    for stem in sorted(index):
        replot_figure(directory, stem, index[stem])
    return sorted(index)


def main(config: ReplotConfig) -> None:
    indexes = find_figure_indexes(config.roots)
    if not indexes:
        raise FileNotFoundError(f"no {FIGURE_INDEX} under {', '.join(config.roots)}")
    total = 0
    for index in indexes:
        stems = replot_directory(index.parent)
        total += len(stems)
        print(f"{index.parent}: redrew {len(stems)} figures")
    print(f"redrew {total} figures from {len(indexes)} directories")


if __name__ == "__main__":
    main(tyro.cli(ReplotConfig))
