"""The ``train.py`` / ``eval.py`` command registry and the modules it runs."""

from __future__ import annotations

import argparse
import importlib.util
import re
import runpy
import sys
from pathlib import Path
from typing import NoReturn

import pytest
import tyro

from bft_pmc import cli

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = (*cli.TRAIN_SECTIONS, *cli.EVAL_SECTIONS)
MAIN_GUARD = re.compile(r"""^if __name__\s*==\s*["']__main__["']\s*:""", re.M)
MODULES = sorted(
    {command.module for section in SECTIONS for command in section.commands}
)


def _stop(*args: object, **kwargs: object) -> NoReturn:
    raise SystemExit("coverage stop")


def test_command_names_are_unique_within_each_entry_point() -> None:
    for sections in (cli.TRAIN_SECTIONS, cli.EVAL_SECTIONS):
        names = [command.name for section in sections for command in section.commands]
        assert len(names) == len(set(names))


def test_every_runnable_package_module_has_a_command() -> None:
    """A module with a ``__main__`` guard must be reachable from train.py or eval.py."""
    runnable = {
        ".".join(path.relative_to(ROOT).with_suffix("").parts)
        for path in (ROOT / "bft_pmc").rglob("*.py")
        if MAIN_GUARD.search(path.read_text())
    }
    assert runnable == set(MODULES)


@pytest.mark.parametrize("module", MODULES)
def test_every_command_module_guards_its_cli(
    module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run each module as a script, stopping at its argument parser."""
    monkeypatch.setattr(tyro, "cli", _stop)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _stop)
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None
    with pytest.raises(SystemExit, match="coverage stop"):
        runpy.run_path(spec.origin, run_name="__main__")


@pytest.mark.parametrize("path", ["train.py", "eval.py"])
def test_root_scripts_print_grouped_help(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [path])
    runpy.run_path(str(ROOT / path), run_name="__main__")
    out = capsys.readouterr().out
    assert out.startswith(f"Usage: uv run {path}")
    sections = cli.TRAIN_SECTIONS if path == "train.py" else cli.EVAL_SECTIONS
    for section in sections:
        assert f"{section.title}:" in out
        for command in section.commands:
            assert command.name in out


def test_dispatch_runs_the_named_module_as_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, str | None, bool, list[str]]] = []

    def record(
        module: str, run_name: str | None = None, alter_sys: bool = False
    ) -> None:
        calls.append((module, run_name, alter_sys, list(sys.argv)))

    monkeypatch.setattr(cli.runpy, "run_module", record)
    cli.train_main(["lr", "--num-tasks", "2"])
    cli.eval_main(["plot-lr-sweep", "--help"])
    monkeypatch.setattr(sys, "argv", ["eval.py", "path-stability", "--stride", "4"])
    cli.eval_main(None)

    assert calls == [
        (
            "bft_pmc.linear_regression.train",
            "__main__",
            True,
            ["bft_pmc.linear_regression.train", "--num-tasks", "2"],
        ),
        (
            "bft_pmc.linear_regression.plot_sweep_combined",
            "__main__",
            True,
            ["bft_pmc.linear_regression.plot_sweep_combined", "--help"],
        ),
        (
            "bft_pmc.analysis.path_stability",
            "__main__",
            True,
            ["bft_pmc.analysis.path_stability", "--stride", "4"],
        ),
    ]

    for argv in ([], ["-h"], ["help"]):
        cli.train_main(argv)
        assert "Usage: uv run train.py <setting>" in capsys.readouterr().out


def test_dispatch_rejects_unknown_commands() -> None:
    with pytest.raises(SystemExit, match="Unknown setting 'bad'"):
        cli.train_main(["bad"])
    with pytest.raises(SystemExit, match="Unknown command 'bad'.*plot-lr-sweep"):
        cli.eval_main(["bad"])
