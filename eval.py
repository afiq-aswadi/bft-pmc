"""Unified evaluation and figure-regeneration entry point."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGETS = {
    "lr-sweep": ROOT / "linear_regression" / "sweep_analysis.py",
    "bau-sweep": ROOT / "balls_and_urns" / "sweep_analysis.py",
    "markov-sweep": ROOT / "markov" / "sweep_analysis.py",
    "markov-threshold": ROOT / "markov" / "task_diversity_threshold.py",
    "plot-lr-sweep": ROOT / "linear_regression" / "plot_sweep_combined.py",
    "plot-lr-dynamics": ROOT / "linear_regression" / "plot_dynamics_combined.py",
    "plot-bau-sweep": ROOT / "balls_and_urns" / "plot_sweep_combined.py",
    "plot-bau-dynamics": ROOT / "balls_and_urns" / "plot_dynamics_combined.py",
    "plot-markov-sweep": ROOT / "markov" / "plot_sweep_combined.py",
    "plot-markov-dynamics": ROOT / "markov" / "plot_dynamics_combined.py",
}


def _print_help() -> None:
    print(
        """Usage: uv run eval.py <command> [command args...]

Commands:
  lr-sweep          Evaluate the linear-regression sweep from saved checkpoints.
  bau-sweep         Evaluate the balls-and-urns sweep from saved checkpoints.
  markov-sweep      Evaluate the Markov sweep from saved checkpoints.
  markov-threshold  Run the Markov task-diversity threshold experiment.
  plot-lr-sweep     Regenerate the linear-regression sweep figure from metrics.csv.
  plot-lr-dynamics  Regenerate the linear-regression dynamics figure from metrics.csv.
  plot-bau-sweep    Regenerate the balls-and-urns sweep figure from metrics.csv.
  plot-bau-dynamics Regenerate the balls-and-urns dynamics figure from metrics.csv.
  plot-markov-sweep Regenerate the Markov sweep figure from saved metrics/history.
  plot-markov-dynamics Regenerate the Markov dynamics figures from saved metrics/history.

Examples:
  uv run eval.py lr-sweep --checkpoint-root checkpoints/lr/task_diversity
  uv run eval.py markov-sweep --checkpoint-root checkpoints/markov/task_diversity
  uv run eval.py plot-lr-sweep --metrics-csv paper_data/lr/sweep/metrics.csv
  uv run eval.py plot-markov-sweep --runs-dir paper_data/markov/sweep/runs --metrics-csv paper_data/markov/sweep/metrics.csv
  uv run eval.py plot-bau-dynamics --metrics-csv paper_data/bau/dynamics/metrics.csv

Pass `--help` after a command to see the underlying experiment-specific CLI.
"""
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    command = args.pop(0)
    target = TARGETS.get(command)
    if target is None:
        valid = ", ".join(sorted(TARGETS))
        raise SystemExit(f"Unknown eval command '{command}'. Expected one of: {valid}")

    sys.argv = [target.name, *args]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
