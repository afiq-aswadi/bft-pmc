"""Recovery error as a function of latent task dimension or task diversity.

The per-run diagnostics (``balls_and_urns/rollout_diagnostics.py``,
``markov/rollout_diagnostics.py`` and ``linear_regression/rollout_diagnostics.py``)
each write one directory per model. The dimension analysis promised to Reviewer
XX72 reads across those directories: one point per model, at a shared rollout
length, plotted against the dimension of the latent task. The same aggregation,
with ``--x-axis task-diversity``, reads the reliability runs repeated at a few
task diversities M near each family's threshold, also promised to XX72.

Inputs are given as ``<value>=<results_dir>``. Against dimension, ``value`` is
the number of free parameters of the latent task, so the script stays family
agnostic:

- balls-and-urns: ``|Y| - 1``  (state space 6, 12, 24, 48 -> 5, 11, 23, 47)
- 1-Markov:       ``k(k - 1)`` (state space 3, 4, 6, 8 -> 6, 12, 30, 56)

Against task diversity, ``value`` is the number of tasks M the model was trained
on.

Each results directory is expected to hold the CSVs written by those scripts,
``{prior,posterior}_recovery_vs_length.csv``. By default the comparison is made
at the largest rollout length present in every input, so the curves are read at
a common budget.

Usage:
    uv run eval.py recovery-vs-dimension \
        --inputs 6=outputs/pmc_diagnostics/markov_k3 \
                 12=outputs/pmc_diagnostics/markov_k4 \
        --output-dir outputs/pmc_diagnostics/markov_dimension_sweep
    uv run eval.py recovery-vs-dimension --x-axis task-diversity \
        --inputs 32=outputs/pmc_diagnostics/markov_m32_ctx256 \
                 64=outputs/pmc_diagnostics/markov_m64_ctx256 \
        --output-dir outputs/pmc_diagnostics/markov_task_diversity
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.pmc_diagnostics import record_figure, write_run_record
from plotting.paper_style import apply_paper_style


# Column stems written by the rollout-diagnostics scripts, in plot order.
METRICS = [("ed", "energy distance"), ("sw", "sliced Wasserstein-1")]
BASELINES = [
    ("memorising", "memorizing", "tab:blue"),
    ("generalising", "generalizing", "tab:red"),
]
# Characters per suptitle line; longer titles wrap instead of running off the
# two-panel figure.
TITLE_WIDTH = 60


@dataclass(frozen=True)
class Axis:
    """What the aggregate is read against."""

    column: str  # CSV column holding each run's value
    name: str  # axis label, and the quantity named in error messages
    stem: str  # output CSV and figure stem
    kind: str  # figures.json kind, which the replot dispatches on
    title: str  # default figure title


AXES = {
    "dimension": Axis(
        column="latent_dim",
        name="latent dimension",
        stem="recovery_vs_dimension",
        kind="dimension",
        title="Recovery error vs latent dimension",
    ),
    "task-diversity": Axis(
        column="num_tasks",
        name="task diversity",
        stem="recovery_vs_task_diversity",
        kind="task_diversity",
        title="Recovery error vs task diversity",
    ),
}
# figures.json kind -> the axis to redraw it against.
AXIS_BY_KIND = {axis.kind: name for name, axis in AXES.items()}


@dataclass
class DimensionSweepConfig:
    """Inputs for the recovery-error-vs-dimension aggregation."""

    inputs: list[str] = field(default_factory=list)
    output_dir: str = "outputs/pmc_diagnostics/dimension_sweep"
    sources: list[str] = field(default_factory=lambda: ["prior", "posterior"])
    # Rollout length at which every run is read. None selects the largest
    # length present in all inputs.
    at_length: int | None = None
    # What each input's value is: the latent dimension of the model's task, or
    # the task diversity M it was trained on.
    x_axis: Literal["dimension", "task-diversity"] = "dimension"
    # None uses the axis's own title.
    title: str | None = None

    def validate(self) -> None:
        if self.x_axis not in AXES:
            raise ValueError(f"unknown x axis {self.x_axis!r}")
        if len(self.inputs) < 2:
            raise ValueError(
                "need at least two --inputs entries to plot against "
                f"{AXES[self.x_axis].name}"
            )
        for source in self.sources:
            if source not in {"prior", "posterior"}:
                raise ValueError(f"unknown source {source!r}")


def parse_inputs(entries: list[str], x_axis: str = "dimension") -> dict[int, Path]:
    """Parse ``<value>=<results_dir>`` entries into a value-keyed map."""
    axis = AXES[x_axis]
    parsed: dict[int, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"input {entry!r} is not of the form <{axis.column}>=<results_dir>"
            )
        raw_value, raw_path = entry.split("=", 1)
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ValueError(
                f"input {entry!r} does not start with an integer {axis.name}"
            ) from error
        if value < 1:
            raise ValueError(f"{axis.name} must be positive, got {value}")
        if value in parsed:
            raise ValueError(f"duplicate {axis.name} {value}")
        parsed[value] = Path(raw_path)
    return dict(sorted(parsed.items()))


def load_source_frames(
    inputs: dict[int, Path], source: str
) -> dict[int, pd.DataFrame]:
    """Read one source's recovery-vs-length CSV from every input directory."""
    frames: dict[int, pd.DataFrame] = {}
    for value, directory in inputs.items():
        path = directory / f"{source}_recovery_vs_length.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        frames[value] = pd.read_csv(path)
    return frames


def common_length(frames: dict[int, pd.DataFrame], at_length: int | None) -> int:
    """The rollout length at which all runs are compared."""
    available = [set(frame["rollout_length"].astype(int)) for frame in frames.values()]
    shared = set.intersection(*available)
    if not shared:
        raise ValueError("inputs share no common rollout length")
    if at_length is None:
        return max(shared)
    if at_length not in shared:
        raise ValueError(
            f"rollout length {at_length} is not present in every input; "
            f"shared lengths are {sorted(shared)}"
        )
    return at_length


def collect(
    inputs: dict[int, Path],
    sources: list[str],
    at_length: int | None,
    x_axis: str = "dimension",
) -> tuple[pd.DataFrame, int]:
    """One row per (source, input) with both metrics against both baselines."""
    column = AXES[x_axis].column
    per_source = {source: load_source_frames(inputs, source) for source in sources}
    all_frames = {
        value: frame
        for frames in per_source.values()
        for value, frame in frames.items()
    }
    length = common_length(all_frames, at_length)

    rows = []
    for source in sources:
        for value, frame in per_source[source].items():
            selected = frame[frame["rollout_length"].astype(int) == length]
            row: dict[str, object] = {
                "source": source,
                column: value,
                "rollout_length": length,
            }
            for metric, _ in METRICS:
                for baseline, _, _ in BASELINES:
                    metric_column = f"dist/{metric}_vs_baseline_{baseline}"
                    if metric_column not in selected:
                        raise KeyError(f"{metric_column} missing from {source} input")
                    row[f"{metric}_{baseline}"] = float(selected[metric_column].iloc[0])
            rows.append(row)
    return pd.DataFrame(rows), length


def plot(
    frame: pd.DataFrame,
    sources: list[str],
    out_stem: Path,
    title: str,
    x_axis: str = "dimension",
) -> None:
    """Grid of source rows by metric columns, each panel a line per baseline."""
    axis = AXES[x_axis]
    fig_width = 3.2 * len(METRICS)
    apply_paper_style(fig_width)
    fig, axes = plt.subplots(
        len(sources),
        len(METRICS),
        figsize=(fig_width, 2.9 * len(sources)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, source in enumerate(sources):
        subset = frame[frame["source"] == source].sort_values(axis.column)
        for column_index, (metric, metric_label) in enumerate(METRICS):
            axes_cell = axes[row_index][column_index]
            for baseline, baseline_label, color in BASELINES:
                axes_cell.plot(
                    subset[axis.column],
                    subset[f"{metric}_{baseline}"],
                    marker="o",
                    color=color,
                    label=baseline_label,
                )
            axes_cell.set_xscale("log", base=2)
            axes_cell.set_yscale("log")
            axes_cell.set_xlabel(axis.name)
            axes_cell.set_ylabel(f"{source}: {metric_label}")
            axes_cell.spines["top"].set_visible(False)
            axes_cell.spines["right"].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False, ncols=2)
    fig.suptitle(textwrap.fill(title, TITLE_WIDTH))
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def main(config: DimensionSweepConfig) -> None:
    config.validate()
    axis = AXES[config.x_axis]
    inputs = parse_inputs(config.inputs, config.x_axis)
    frame, length = collect(inputs, config.sources, config.at_length, config.x_axis)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_record(
        output_dir,
        asdict(config),
        inputs=[
            str(path / f"{source}_recovery_vs_length.csv")
            for path in inputs.values()
            for source in config.sources
        ],
        family=axis.kind,
        rollout_length=length,
        input_directories={value: str(path) for value, path in inputs.items()},
    )
    csv_path = output_dir / f"{axis.stem}.csv"
    frame.to_csv(csv_path, index=False)
    title = f"{config.title or axis.title} (rollout length N={length})"
    plot(frame, config.sources, output_dir / axis.stem, title, config.x_axis)
    record_figure(
        output_dir,
        axis.stem,
        kind=axis.kind,
        title=title,
        csv=csv_path.name,
        sources=config.sources,
    )
    print(f"read {len(inputs)} runs at rollout length {length}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(DimensionSweepConfig))
