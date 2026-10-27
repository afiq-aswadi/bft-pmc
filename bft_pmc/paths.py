"""Locations inside the repository checkout.

Most command-line defaults are relative to the working directory and assume
the repository root, as in ``uv run eval.py ...``. The few tools that read the
saved PMC diagnostics regardless of where they are launched from resolve paths
against :data:`REPO_ROOT` instead.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
"""The repository root: the directory holding ``train.py`` and ``eval.py``."""

OUTPUTS = REPO_ROOT / "outputs"
"""Untracked experiment outputs (checkpoint evaluations, figures, diagnostics)."""

PMC_DIAGNOSTICS = OUTPUTS / "pmc_diagnostics"
"""Per-run reliability diagnostics written by ``eval.py *-rollout-diagnostics``."""

PAPER_DATA = REPO_ROOT / "paper_data"
"""Frozen, committed figure inputs."""


def repo_relative(path: Path) -> str:
    """``path`` relative to the repository root when it lies inside it, else absolute.

    Symlinks are deliberately not resolved, so a diagnostics directory that
    links into scratch storage is still recorded at its repository path.
    """
    absolute = Path(path).absolute()
    try:
        return str(absolute.relative_to(REPO_ROOT))
    except ValueError:
        return str(absolute)
