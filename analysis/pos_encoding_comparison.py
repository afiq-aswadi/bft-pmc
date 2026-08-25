"""Compare the learned / rope / none positional-encoding variants.

The LR and balls-and-urns sweeps train every task count under all three
encodings, each into its own checkpoint root and its own analysis output. This
entry point joins those per-variant artifacts on (task count, prompt source) so
the encodings can be read against each other, and reports where the choice
actually moves a number.

Two input families, either or both:

- training logs (``--training-logs``): the ``--log-file`` JSON written by the
  LR and BAU trainers, giving train/eval loss against step. Answers "does
  dropping the positional embedding cost fit quality?".
- sweep metrics (``--sweep-metrics``): the ``metrics.csv`` written by
  ``eval.py {lr,bau,markov}-sweep``, giving every PMC distance against task
  diversity. Answers "does it change the recovered prior and posterior?".

Both take ``label=path`` entries, where a path is a single file or a directory
that is searched for the usual filenames.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

from analysis.path_stability import parse_input
from plotting.paper_style import apply_paper_style, task_diversity_ticks


# One curve per encoding, in the order they are reported.
ENCODING_COLORS: dict[str, str] = {
    "learned": "tab:blue",
    "rope": "goldenrod",
    "none": "tab:red",
}
_FALLBACK_COLOR = "0.4"


def _num_tasks_from_name(path: Path) -> int | None:
    """Recover the task count from a ``num_tasks_<M>.json`` log filename."""
    stem = path.stem
    prefix = "num_tasks_"
    if not stem.startswith(prefix):
        return None
    suffix = stem.removeprefix(prefix)
    return int(suffix) if suffix.isdigit() else None


def load_training_logs(entries: tuple[str, ...]) -> pd.DataFrame:
    """Read labelled trainer log files into one tidy frame."""
    rows: list[dict[str, object]] = []
    for entry in entries:
        label, path = parse_input(entry)
        files = [path] if path.is_file() else sorted(path.rglob("num_tasks_*.json"))
        if not files:
            raise FileNotFoundError(f"No trainer log files under {path}.")
        for log_path in files:
            with log_path.open(encoding="utf-8") as handle:
                records = json.load(handle)
            for record in records:
                rows.append(
                    {
                        "pos_encoding": label,
                        "num_tasks": _num_tasks_from_name(log_path),
                        **record,
                    }
                )
    return pd.DataFrame(rows)


def load_sweep_metrics(entries: tuple[str, ...]) -> pd.DataFrame:
    """Read labelled sweep ``metrics.csv`` files into one tidy frame."""
    frames: list[pd.DataFrame] = []
    for entry in entries:
        label, path = parse_input(entry)
        files = [path] if path.is_file() else sorted(path.rglob("metrics.csv"))
        if not files:
            raise FileNotFoundError(f"No metrics.csv under {path}.")
        for metrics_path in files:
            frame = pd.read_csv(metrics_path)
            # Markov names its task-diversity axis n_chains; align the families.
            if "n_chains" in frame.columns and "num_tasks" not in frame.columns:
                frame = frame.rename(columns={"n_chains": "num_tasks"})
            frame.insert(0, "pos_encoding", label)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def final_training_losses(training: pd.DataFrame) -> pd.DataFrame:
    """Last recorded evaluation loss per (encoding, task count)."""
    if "eval/loss" not in training.columns:
        raise ValueError(
            "Trainer logs carry no eval/loss; rerun training with --eval-every set."
        )
    evaluated = training.dropna(subset=["eval/loss"])
    latest = evaluated.sort_values("step").groupby(
        ["pos_encoding", "num_tasks"], dropna=False, as_index=False
    )
    return latest.last()[
        ["pos_encoding", "num_tasks", "step", "eval/loss", "train/loss"]
    ]


def metric_columns(metrics: pd.DataFrame) -> list[str]:
    """Metric columns are the namespaced numeric ones, e.g. ``dist/ed_vs_...``."""
    return [
        column
        for column in metrics.columns
        if "/" in column and pd.api.types.is_numeric_dtype(metrics[column])
    ]


def compare_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per (task count, prompt source, metric), one column per encoding.

    ``spread`` is max minus min across encodings: large values are where the
    positional encoding actually changes the recovered distribution.
    """
    columns = metric_columns(metrics)
    if not columns:
        raise ValueError("No namespaced metric columns found in the sweep metrics.")
    index = [name for name in ("num_tasks", "prompt_source") if name in metrics.columns]
    melted = metrics.melt(
        id_vars=[*index, "pos_encoding"],
        value_vars=columns,
        var_name="metric",
        value_name="value",
    )
    wide = melted.pivot_table(
        index=[*index, "metric"],
        columns="pos_encoding",
        values="value",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    encodings = [name for name in wide.columns if name in metrics["pos_encoding"].values]
    wide["spread"] = wide[encodings].max(axis=1) - wide[encodings].min(axis=1)
    return wide.sort_values(["metric", *index]).reset_index(drop=True)


def _encoding_color(encoding: str) -> str:
    return ENCODING_COLORS.get(encoding, _FALLBACK_COLOR)


def plot_training_curves(training: pd.DataFrame, save_path: str | Path) -> Path:
    """Evaluation loss against step, one line per encoding."""
    evaluated = training.dropna(subset=["eval/loss"])
    if evaluated.empty:
        raise ValueError("No eval/loss rows to plot.")
    apply_paper_style(3.3, 0.6)
    figure, axes = plt.subplots(figsize=(3.3, 2.1))
    for encoding, group in evaluated.groupby("pos_encoding"):
        ordered = group.sort_values("step")
        axes.plot(
            ordered["step"],
            ordered["eval/loss"],
            label=encoding,
            color=_encoding_color(str(encoding)),
            linewidth=1.0,
        )
    axes.set_xlabel("training step")
    axes.set_ylabel("eval loss")
    axes.legend(frameon=False)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_metric_panels(
    metrics: pd.DataFrame,
    save_path: str | Path,
    *,
    max_panels: int = 6,
) -> Path:
    """One panel per metric: distance against task diversity, per encoding."""
    columns = metric_columns(metrics)[:max_panels]
    if not columns:
        raise ValueError("No namespaced metric columns found in the sweep metrics.")
    apply_paper_style(6.5, 1.0)
    panels = len(columns)
    figure, axes = plt.subplots(
        1, panels, figsize=(2.0 * panels, 1.9), squeeze=False, sharex=True
    )
    for index, column in enumerate(columns):
        axis = axes[0][index]
        for encoding, group in metrics.groupby("pos_encoding"):
            aggregated = (
                group.groupby("num_tasks", as_index=False)[column].mean().sort_values(
                    "num_tasks"
                )
            )
            axis.plot(
                aggregated["num_tasks"],
                aggregated[column],
                label=encoding,
                color=_encoding_color(str(encoding)),
                marker="o",
                markersize=2,
                linewidth=1.0,
            )
        axis.set_xscale("log", base=2)
        task_diversity_ticks(axis.xaxis)
        axis.set_title(column.split("/")[-1], fontsize="small")
        axis.set_xlabel("M")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0][0].set_ylabel("distance")
    axes[0][-1].legend(frameon=False)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_summary(
    final_losses: pd.DataFrame | None,
    comparison: pd.DataFrame | None,
) -> str:
    """Human-readable summary of where the encodings differ."""
    lines: list[str] = ["# Positional-encoding comparison", ""]
    if final_losses is not None:
        lines += ["## Final evaluation loss", ""]
        pivot = final_losses.pivot_table(
            index="num_tasks", columns="pos_encoding", values="eval/loss", dropna=False
        )
        lines += [pivot.to_string(), ""]
        best = final_losses.loc[final_losses["eval/loss"].idxmin()]
        lines += [
            f"lowest eval loss: {best['pos_encoding']} "
            f"({best['eval/loss']:.4f} at step {int(best['step'])})",
            "",
        ]
    if comparison is not None:
        lines += ["## Largest disagreements between encodings", ""]
        ranked = comparison.sort_values("spread", ascending=False).head(10)
        lines += [ranked.to_string(index=False), ""]
    return "\n".join(lines)


@dataclass(slots=True)
class ComparisonConfig:
    """CLI arguments for the positional-encoding comparison."""

    training_logs: tuple[str, ...] = ()
    sweep_metrics: tuple[str, ...] = ()
    output_dir: str = "outputs/pos_encoding_comparison"

    def validate(self) -> None:
        if not self.training_logs and not self.sweep_metrics:
            raise ValueError(
                "Pass --training-logs and/or --sweep-metrics as label=path entries."
            )


def main(config: ComparisonConfig) -> None:
    """Join the per-encoding artifacts, then write tables, figures, and a summary."""
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    final_losses = None
    comparison = None

    if config.training_logs:
        training = load_training_logs(config.training_logs)
        training.to_csv(output_dir / "training_curves.csv", index=False)
        final_losses = final_training_losses(training)
        final_losses.to_csv(output_dir / "final_training_loss.csv", index=False)
        plot_training_curves(training, output_dir / "training_curves.png")

    if config.sweep_metrics:
        metrics = load_sweep_metrics(config.sweep_metrics)
        metrics.to_csv(output_dir / "sweep_metrics.csv", index=False)
        comparison = compare_metrics(metrics)
        comparison.to_csv(output_dir / "metric_comparison.csv", index=False)
        plot_metric_panels(metrics, output_dir / "metric_panels.png")

    summary = render_summary(final_losses, comparison)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main(tyro.cli(ComparisonConfig))
