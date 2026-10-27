"""Repository locations used by the tools that read saved diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from bft_pmc import paths


def test_repo_root_holds_the_entry_points() -> None:
    assert (paths.REPO_ROOT / "train.py").is_file()
    assert (paths.REPO_ROOT / "eval.py").is_file()
    assert paths.PMC_DIAGNOSTICS == paths.REPO_ROOT / "outputs" / "pmc_diagnostics"
    assert paths.PAPER_DATA.is_dir()


def test_repo_relative_shortens_paths_inside_the_repository() -> None:
    inside = paths.PMC_DIAGNOSTICS / "lr_m32_none" / "run_record.json"
    assert (
        paths.repo_relative(inside)
        == "outputs/pmc_diagnostics/lr_m32_none/run_record.json"
    )


def test_repo_relative_keeps_outside_paths_absolute(tmp_path: Path) -> None:
    assert paths.repo_relative(tmp_path / "x.csv") == str(tmp_path / "x.csv")


def test_repo_relative_resolves_relative_paths_against_the_working_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(paths.REPO_ROOT)
    assert paths.repo_relative(Path("paper_data/README.md")) == "paper_data/README.md"
