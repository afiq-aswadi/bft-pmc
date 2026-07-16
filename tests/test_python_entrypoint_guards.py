from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
from typing import NoReturn

import pytest
import tyro


TYRO_ENTRYPOINTS = [
    "balls_and_urns/distribution_dynamics.py",
    "balls_and_urns/plot_dynamics_combined.py",
    "balls_and_urns/plot_marginals.py",
    "balls_and_urns/plot_stitched_marginals.py",
    "balls_and_urns/plot_sweep_combined.py",
    "balls_and_urns/sweep_analysis.py",
    "balls_and_urns/train.py",
    "linear_regression/distribution_dynamics.py",
    "linear_regression/plot_dynamics_combined.py",
    "linear_regression/plot_marginals.py",
    "linear_regression/plot_single_marginal.py",
    "linear_regression/plot_stitched_marginals.py",
    "linear_regression/plot_sweep_combined.py",
    "linear_regression/sweep_analysis.py",
    "linear_regression/train.py",
    "markov/run_pmc.py",
    "markov/sweep_analysis.py",
    "markov/task_diversity_threshold.py",
    "markov/train.py",
    "sweeps/run_yaml_sweep.py",
]

ARGPARSE_ENTRYPOINTS = [
    "balls_and_urns/plot_sweep_prior.py",
    "linear_regression/plot_stitched_sweep_dynamics.py",
    "linear_regression/plot_sweep_prior.py",
    "markov/plot_dynamics_combined.py",
    "markov/plot_matrix_marginals.py",
    "markov/plot_sweep_combined.py",
    "markov/plot_sweep_prior.py",
]


def _stop(*args: object, **kwargs: object) -> NoReturn:
    raise SystemExit("coverage stop")


@pytest.mark.parametrize("path", TYRO_ENTRYPOINTS)
def test_tyro_entrypoint_guard(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tyro, "cli", _stop)
    with pytest.raises(SystemExit, match="coverage stop"):
        runpy.run_path(path, run_name="__main__")


@pytest.mark.parametrize("path", ARGPARSE_ENTRYPOINTS)
def test_argparse_entrypoint_guard(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _stop)
    with pytest.raises(SystemExit, match="coverage stop"):
        runpy.run_path(path, run_name="__main__")


@pytest.mark.parametrize("path", ["train.py", "eval.py"])
def test_root_dispatcher_guard(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [Path(path).name])
    runpy.run_path(path, run_name="__main__")
    assert "Usage:" in capsys.readouterr().out
