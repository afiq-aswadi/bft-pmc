from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from analysis.pmc_diagnostics import (
    FIGURE_INDEX,
    count_curves,
    drift_curves,
    record_figure,
    recovery_curves,
)
from analysis.replot_pmc_diagnostics import (
    ReplotConfig,
    find_figure_indexes,
    main,
    replot_directory,
    replot_figure,
)


GRID = [8, 16, 32]
BASELINE_NAMES = ("baseline_generalising", "baseline_memorising")


def _simplex_samples(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    draws = rng.random((n, dim)) + 1e-3
    return (draws / draws.sum(axis=1, keepdims=True)).astype(np.float32)


def _fresh_sampler(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(n)
    return {name: _simplex_samples(rng, n, 3) for name in BASELINE_NAMES}


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """One figure of each per-run kind, saved the way the diagnostics save them."""
    rng = np.random.default_rng(0)
    stats = np.stack([_simplex_samples(rng, 16, 3) for _ in GRID], axis=1)
    baselines = {name: _simplex_samples(rng, 16, 3) for name in BASELINE_NAMES}
    directory = tmp_path / "pmc_diagnostics" / "bau_m32_learned"
    directory.mkdir(parents=True)

    recovery_curves(stats, baselines, GRID, 8).to_csv(
        directory / "prior_recovery_vs_length.csv", index=False
    )
    record_figure(
        directory,
        "prior_recovery_vs_length",
        kind="recovery",
        title="recovery",
        csv="prior_recovery_vs_length.csv",
    )

    drift, per_path = drift_curves(stats, GRID, stats[:, 0, :])
    drift.to_csv(directory / "prior_drift_vs_length.csv", index=False)
    np.savez_compressed(
        directory / "prior_drift_paths.npz",
        rollout_lengths=np.asarray(GRID),
        per_path_drift=per_path,
    )
    record_figure(
        directory,
        "prior_drift_vs_length",
        kind="drift",
        title="drift",
        csv="prior_drift_vs_length.csv",
        paths="prior_drift_paths.npz",
    )

    counts = count_curves(
        stats[:, -1, :], baselines, _fresh_sampler, [8, 16], 2, 8, rng
    )
    counts.to_csv(directory / "prior_recovery_vs_count.csv", index=False)
    record_figure(
        directory,
        "prior_recovery_vs_count",
        kind="counts",
        title="counts",
        csv="prior_recovery_vs_count.csv",
    )
    return directory


@pytest.fixture
def dimension_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "pmc_diagnostics" / "bau_dimension_sweep"
    directory.mkdir(parents=True)
    rows = [
        {
            "source": "prior",
            "latent_dim": dimension,
            "rollout_length": 32,
            "ed_memorising": 0.1 * dimension,
            "ed_generalising": 0.2,
            "sw_memorising": 0.3,
            "sw_generalising": 0.4,
        }
        for dimension in (5, 11)
    ]
    pd.DataFrame(rows).to_csv(directory / "recovery_vs_dimension.csv", index=False)
    record_figure(
        directory,
        "recovery_vs_dimension",
        kind="dimension",
        title="dimension",
        csv="recovery_vs_dimension.csv",
        sources=["prior"],
    )
    return directory


def test_replot_directory_redraws_every_indexed_figure(
    run_dir: Path, saved_figures: list[Path]
) -> None:
    stems = replot_directory(run_dir)
    assert stems == [
        "prior_drift_vs_length",
        "prior_recovery_vs_count",
        "prior_recovery_vs_length",
    ]
    assert saved_figures == [
        run_dir / f"{stem}{suffix}" for stem in stems for suffix in (".png", ".pdf")
    ]


def test_replotting_the_dimension_aggregate_restores_the_rc_params(
    dimension_dir: Path, saved_figures: list[Path]
) -> None:
    """The aggregate applies the paper style globally; later figures must not inherit it."""
    before = dict(matplotlib.rcParams)
    replot_directory(dimension_dir)
    assert [path.name for path in saved_figures] == [
        "recovery_vs_dimension.png",
        "recovery_vs_dimension.pdf",
    ]
    assert dict(matplotlib.rcParams) == before


def test_replot_draws_the_task_diversity_aggregate_against_m(
    tmp_path: Path, saved_figures: list[Path]
) -> None:
    directory = tmp_path / "pmc_diagnostics" / "lr_task_diversity_none"
    directory.mkdir(parents=True)
    rows = [
        {
            "source": "posterior",
            "num_tasks": num_tasks,
            "rollout_length": 223,
            "ed_memorising": 0.1,
            "ed_generalising": 0.2,
            "sw_memorising": 0.3,
            "sw_generalising": 0.4,
        }
        for num_tasks in (32, 128, 256)
    ]
    pd.DataFrame(rows).to_csv(directory / "recovery_vs_task_diversity.csv", index=False)
    record_figure(
        directory,
        "recovery_vs_task_diversity",
        kind="task_diversity",
        title="task diversity",
        csv="recovery_vs_task_diversity.csv",
        sources=["posterior"],
    )
    assert replot_directory(directory) == ["recovery_vs_task_diversity"]
    assert [path.name for path in saved_figures] == [
        "recovery_vs_task_diversity.png",
        "recovery_vs_task_diversity.pdf",
    ]


def test_replot_rejects_an_unknown_figure_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown figure kind 'violin'"):
        replot_figure(tmp_path, "figure", {"kind": "violin", "title": "title"})


def test_figure_indexes_are_found_once_below_every_root(
    run_dir: Path, dimension_dir: Path
) -> None:
    root = run_dir.parent
    assert find_figure_indexes([str(root), str(run_dir)]) == [
        dimension_dir / FIGURE_INDEX,
        run_dir / FIGURE_INDEX,
    ]


def test_main_redraws_every_directory_under_the_roots(
    run_dir: Path,
    dimension_dir: Path,
    saved_figures: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(ReplotConfig(roots=[str(run_dir.parent)]))
    assert len(saved_figures) == 2 * 4
    assert "redrew 4 figures from 2 directories" in capsys.readouterr().out


def test_main_reports_a_root_without_figures(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no figures.json"):
        main(ReplotConfig(roots=[str(tmp_path)]))
