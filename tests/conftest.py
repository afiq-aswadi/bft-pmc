from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure


@pytest.fixture
def saved_figures(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Path]]:
    """Record figure destinations without rasterizing high-resolution test plots."""
    paths: list[Path] = []

    def record_save(
        self: Figure,
        filename: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        del self, args, kwargs
        paths.append(Path(filename))

    monkeypatch.setattr(Figure, "savefig", record_save)
    yield paths
    plt.close("all")
