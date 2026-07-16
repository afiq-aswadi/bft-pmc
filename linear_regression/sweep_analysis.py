"""Sweep analysis entry point for the LR memorisation/generalisation study."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import tyro

from linear_regression.analysis.config import SOURCE_DISPLAY_LABELS, SweepConfig
from linear_regression.analysis.data import find_latest_checkpoint, load_run_info
from linear_regression.analysis.plotting import plot_results
from linear_regression.analysis.runner import run_analysis

__all__ = [
    "SweepConfig",
    "SOURCE_DISPLAY_LABELS",
    "find_latest_checkpoint",
    "load_run_info",
    "plot_results",
    "run_analysis",
]


def main(config: SweepConfig) -> None:
    """Run sweep analysis with the given configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w") as file:
        json.dump(asdict(config), file, indent=2)

    df, per_prompt_df = run_analysis(config, output_dir)
    df.to_csv(output_dir / "metrics.csv", index=False)
    if not per_prompt_df.empty:
        per_prompt_df.to_csv(output_dir / "per_prompt_metrics.csv", index=False)
    plot_results(df, output_dir, config)


if __name__ == "__main__":
    main(tyro.cli(SweepConfig))
