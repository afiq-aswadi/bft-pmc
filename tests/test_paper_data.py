"""The frozen camera-ready figure inputs are complete and unaltered."""

from __future__ import annotations

import hashlib
from pathlib import Path

from bft_pmc.analysis.plot_reliability_appendix import draw_all

ROOT = Path(__file__).resolve().parents[1]
CAMERA_READY = ROOT / "paper_data" / "camera_ready"


def _listed() -> dict[str, str]:
    """SHA256SUMS as {path relative to camera_ready/: digest}."""
    entries: dict[str, str] = {}
    for line in (CAMERA_READY / "SHA256SUMS").read_text().splitlines():
        digest, rest = line.split("  ", 1)
        entries[rest.split("  <- ")[0]] = digest
    return entries


def test_camera_ready_inputs_match_their_checksums() -> None:
    listed = _listed()
    on_disk = {
        str(path.relative_to(CAMERA_READY))
        for path in CAMERA_READY.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    } - {"SHA256SUMS", "README.md"}
    assert on_disk == set(listed)
    for relative, digest in listed.items():
        actual = hashlib.sha256((CAMERA_READY / relative).read_bytes()).hexdigest()
        assert actual == digest, relative


def test_frozen_diagnostics_hold_every_reliability_input(tmp_path: Path) -> None:
    figures = draw_all(CAMERA_READY / "pmc_diagnostics", tmp_path)
    assert len(figures.figures) == 41
    for entry in figures.figures:
        for source in entry["inputs"]:
            assert source["path"].startswith("paper_data/camera_ready/pmc_diagnostics/")
