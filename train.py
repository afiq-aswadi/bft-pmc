"""Unified training entry point for the public research-code release."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGETS = {
    "lr": ROOT / "linear_regression" / "train.py",
    "bau": ROOT / "balls_and_urns" / "train.py",
    "beta-bernoulli": ROOT / "balls_and_urns" / "beta_bernoulli.py",
    "markov": ROOT / "markov" / "train.py",
}


def _print_help() -> None:
    print(
        """Usage: uv run train.py <setting> [setting args...]

Settings:
  lr       Linear-regression PFN training.
  bau      Balls-and-urns PFN training.
  beta-bernoulli  Continuous-prior Beta-Bernoulli training and PMC evaluation.
  markov   Markov transient experiment training.

Examples:
  uv run train.py lr --num-tasks 32 --num-steps 1000 --no-use-wandb
  uv run train.py bau --num-tasks 32 --num-steps 1000 --no-use-wandb
  uv run train.py beta-bernoulli --output-dir outputs/bau/beta_bernoulli
  uv run train.py markov --config-path markov/train.yaml --n-chains 256

Pass `--help` after a setting to see the underlying experiment-specific CLI.
"""
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    setting = args.pop(0)
    target = TARGETS.get(setting)
    if target is None:
        valid = ", ".join(sorted(TARGETS))
        raise SystemExit(
            f"Unknown training setting '{setting}'. Expected one of: {valid}"
        )

    sys.argv = [target.name, *args]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
