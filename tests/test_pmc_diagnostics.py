from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import pmc_diagnostics
from analysis.pmc_diagnostics import (
    FIGURE_INDEX,
    RUN_RECORD,
    check_rollouts_fit_context,
    count_curves,
    count_curves_with_resamples,
    distance_metrics,
    drift_curves,
    plot_counts,
    plot_drift,
    plot_recovery,
    record_figure,
    recovery_curves,
    save_path_stability_bundle,
    write_run_record,
)


BASELINE_NAMES = ("baseline_generalising", "baseline_memorising")
GRID = [8, 16, 32]


def _simplex_samples(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Rows on the probability simplex, matching the rollout statistics' support."""
    draws = rng.random((n, dim)) + 1e-3
    return (draws / draws.sum(axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def baselines() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {name: _simplex_samples(rng, 16, 3) for name in BASELINE_NAMES}


@pytest.fixture
def stats() -> np.ndarray:
    rng = np.random.default_rng(1)
    return np.stack(
        [_simplex_samples(rng, 16, 3) for _ in GRID],
        axis=1,
    )


def test_distance_metrics_reports_both_metrics_per_baseline(
    baselines: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(2)
    metrics = distance_metrics(_simplex_samples(rng, 16, 3), baselines, 8)
    assert set(metrics) == {
        f"dist/{metric}_vs_{name}" for metric in ("ed", "sw") for name in BASELINE_NAMES
    }
    assert all(np.isfinite(value) for value in metrics.values())


def test_distance_metrics_vanish_against_the_same_samples(
    baselines: dict[str, np.ndarray],
) -> None:
    """A sample set is at distance zero from itself, up to the U-statistic bias."""
    same = baselines["baseline_generalising"]
    metrics = distance_metrics(same, {"baseline_generalising": same}, 8)
    assert metrics["dist/sw_vs_baseline_generalising"] == pytest.approx(0.0)
    assert metrics["dist/ed_vs_baseline_generalising"] < 1e-6


def test_recovery_curves_report_one_row_per_grid_point(
    stats: np.ndarray,
    baselines: dict[str, np.ndarray],
) -> None:
    frame = recovery_curves(stats, baselines, GRID, 8)
    assert list(frame["rollout_length"]) == GRID
    for name in BASELINE_NAMES:
        assert f"dist/ed_vs_{name}" in frame.columns
        assert f"dist/sw_vs_{name}" in frame.columns


def test_drift_curves_default_to_the_statistic_dimension() -> None:
    values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    frame, per_path = drift_curves(values, GRID, np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(per_path, values.sum(axis=-1) / 4)
    assert list(frame["rollout_length"]) == GRID
    np.testing.assert_allclose(frame["drift_mean"], per_path.mean(axis=0))


def test_drift_curves_scale_by_the_requested_free_parameter_count() -> None:
    """Passing ``scale`` divides by that count rather than the ambient dimension."""
    values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    anchor = np.zeros(4, dtype=np.float32)
    _, by_dimension = drift_curves(values, GRID, anchor)
    _, by_free_parameters = drift_curves(values, GRID, anchor, scale=3)
    np.testing.assert_allclose(by_free_parameters, by_dimension * 4 / 3)


def test_drift_curves_accept_a_per_path_anchor(stats: np.ndarray) -> None:
    """A per-path anchor puts that path at zero drift on its own grid point."""
    _, per_path = drift_curves(stats, GRID, stats[:, 1, :])
    np.testing.assert_allclose(per_path[:, 1], 0.0, atol=1e-7)
    assert (per_path[:, 0] > 0).all()


def test_drift_curves_report_ordered_quantiles(stats: np.ndarray) -> None:
    frame, _ = drift_curves(stats, GRID, stats[:, 0, :])
    assert (frame["drift_q10"] <= frame["drift_q50"]).all()
    assert (frame["drift_q50"] <= frame["drift_q90"]).all()


def _fresh_sampler(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(n)
    return {name: _simplex_samples(rng, n, 3) for name in BASELINE_NAMES}


def test_count_curves_report_a_monte_carlo_floor_for_every_metric(
    stats: np.ndarray,
    baselines: dict[str, np.ndarray],
) -> None:
    frame = count_curves(
        stats[:, -1, :],
        baselines,
        _fresh_sampler,
        [8, 16],
        2,
        8,
        np.random.default_rng(3),
    )
    assert list(frame["num_rollouts"]) == [8, 16]
    for name in BASELINE_NAMES:
        for metric in ("ed", "sw"):
            key = f"dist/{metric}_vs_{name}"
            assert f"{key}/mean" in frame.columns
            assert f"floor/{key}/mean" in frame.columns
            assert f"floor/{key}/sd" in frame.columns


def test_count_curves_report_zero_spread_for_the_full_sample(
    stats: np.ndarray,
    baselines: dict[str, np.ndarray],
) -> None:
    """At R equal to the rollout count there is one subset, so no spread to report."""
    frame = count_curves(
        stats[:, -1, :],
        baselines,
        _fresh_sampler,
        [16],
        5,
        8,
        np.random.default_rng(4),
    )
    row = frame.iloc[0]
    assert row["dist/ed_vs_baseline_generalising/sd"] == 0.0
    assert row["floor/dist/ed_vs_baseline_generalising/sd"] == 0.0


def test_count_resamples_hold_every_draw_behind_the_summary(
    stats: np.ndarray,
    baselines: dict[str, np.ndarray],
) -> None:
    frame, resamples = count_curves_with_resamples(
        stats[:, -1, :],
        baselines,
        _fresh_sampler,
        [8, 16],
        3,
        8,
        np.random.default_rng(6),
    )
    assert list(resamples.columns) == pmc_diagnostics.RESAMPLE_COLUMNS
    # R=8 draws three subsets and R=16, the full sample, one; each draw scores
    # four metrics for the model subset and four for the fresh-draw floor.
    assert len(resamples) == (3 + 1) * 4 * 2
    key = "dist/ed_vs_baseline_generalising"
    for kind, prefix in (("model", ""), ("floor", "floor/")):
        draws = resamples[(resamples["kind"] == kind) & (resamples["metric"] == key)]
        np.testing.assert_allclose(
            draws.groupby("num_rollouts")["value"].mean().to_numpy(),
            frame[f"{prefix}{key}/mean"].to_numpy(),
        )


def test_collecting_resamples_leaves_the_count_curves_unchanged(
    stats: np.ndarray,
    baselines: dict[str, np.ndarray],
) -> None:
    arguments = (stats[:, -1, :], baselines, _fresh_sampler, [8, 16], 3, 8)
    plain = count_curves(*arguments, np.random.default_rng(7))
    detailed, _ = count_curves_with_resamples(*arguments, np.random.default_rng(7))
    pd.testing.assert_frame_equal(plain, detailed)


def test_a_balls_and_urns_rollout_may_fill_the_context() -> None:
    """BOS takes a position but the last token is never fed back: 16 + 240 = 256."""
    check_rollouts_fit_context(
        steps=240,
        prompt_length=16,
        prompt_sources=["prior", "posterior"],
        context_length=256,
    )
    with pytest.raises(ValueError, match=r"prompt of 16 needs 257 positions.*--steps 240"):
        check_rollouts_fit_context(
            steps=241,
            prompt_length=16,
            prompt_sources=["prior", "posterior"],
            context_length=256,
        )


def test_the_context_check_counts_in_the_caller_s_unit() -> None:
    """Linear regression checks (x, y) pairs against half its token context."""
    with pytest.raises(
        ValueError,
        match=r"prompt of 32 needs 257 \(x, y\) pairs, but the model context holds 256\. "
        r"Pass --steps 224",
    ):
        check_rollouts_fit_context(
            steps=225,
            prompt_length=32,
            prompt_sources=["posterior"],
            context_length=256,
            unit="(x, y) pairs",
        )


def test_path_stability_bundle_is_named_after_the_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "bau_m32_learned"
    run_dir.mkdir()
    tokens = np.zeros((3, 5), dtype=np.int16)
    path = save_path_stability_bundle(
        run_dir,
        prompt_length=2,
        checkpoint_path="checkpoints/bau/M32/checkpoint_step_100000.pt",
        arrays={"rollout_tokens": tokens},
        num_tasks=32,
    )
    assert path == run_dir / "bau_m32_learned_rollouts.npz"
    with np.load(path) as archive:
        assert set(archive.files) == {"prompt_len", "rollout_tokens", "num_tasks", "step"}
        assert int(archive["prompt_len"]) == 2
        assert int(archive["step"]) == 100000


def test_path_stability_bundle_omits_a_step_the_name_lacks(tmp_path: Path) -> None:
    path = save_path_stability_bundle(
        tmp_path,
        prompt_length=2,
        checkpoint_path="run/model.pt",
        arrays={"rollout_states": np.zeros((3, 5), dtype=np.int16)},
    )
    with np.load(path) as archive:
        assert "step" not in archive.files


def test_run_record_saves_config_context_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("PMC_SUBMISSION_DIR", "outputs/pmc_diagnostics/submissions/1")
    checkpoint = tmp_path / "checkpoint_step_100000.pt"
    checkpoint.write_bytes(b"weights")
    path = write_run_record(
        tmp_path,
        {"steps": 239, "length_grid": [8, 239]},
        inputs=[str(checkpoint), None, str(tmp_path / "absent.yaml")],
        family="bau",
        checkpoint=checkpoint,
    )
    record = json.loads(path.read_text())
    assert path == tmp_path / RUN_RECORD
    assert record["config"] == {"steps": 239, "length_grid": [8, 239]}
    assert record["context"] == {"family": "bau", "checkpoint": str(checkpoint)}
    assert record["slurm_job_id"] == "12345"
    assert record["environment"]["PMC_SUBMISSION_DIR"] == (
        "outputs/pmc_diagnostics/submissions/1"
    )
    assert set(record["packages"]) == {"numpy", "pandas", "torch", "matplotlib"}
    assert record["installed_distributions"]["numpy"] == np.__version__
    assert set(record["hardware"]) == {
        "platform",
        "cpu_count",
        "torch",
        "cuda",
        "cudnn",
        "gpus",
    }
    assert set(record["git"]) == {"commit", "changed_files"}
    assert record["inputs"] == {
        str(checkpoint): hashlib.sha256(b"weights").hexdigest(),
        str(tmp_path / "absent.yaml"): None,
    }


def test_file_sha256_hashes_bytes_and_skips_missing_files(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(range(256)) * 5000)
    assert pmc_diagnostics.file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert pmc_diagnostics.file_sha256(tmp_path / "missing.bin") is None


def test_git_state_reports_the_commit_and_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = iter(["abc123\n", " M eval.py\n?? slurm/new.slurm\n"])

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(command, 0, stdout=next(outputs))

    monkeypatch.setattr(pmc_diagnostics.subprocess, "run", fake_run)
    assert pmc_diagnostics._git_state(tmp_path) == {
        "commit": "abc123",
        "changed_files": [" M eval.py", "?? slurm/new.slurm"],
    }


def test_git_state_is_null_outside_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(pmc_diagnostics.subprocess, "run", fail)
    assert pmc_diagnostics._git_state(tmp_path) == {
        "commit": None,
        "changed_files": None,
    }


def test_package_versions_report_a_missing_package_as_null() -> None:
    versions = pmc_diagnostics._package_versions(("numpy", "no-such-package-pmc"))
    assert versions == {"numpy": np.__version__, "no-such-package-pmc": None}


def test_record_figure_merges_entries_into_the_index(tmp_path: Path) -> None:
    record_figure(
        tmp_path, "prior_recovery_vs_length", kind="recovery", title="old", csv="a.csv"
    )
    record_figure(
        tmp_path,
        "prior_drift_vs_length",
        kind="drift",
        title="drift",
        csv="d.csv",
        paths="d.npz",
    )
    record_figure(
        tmp_path, "prior_recovery_vs_length", kind="recovery", title="new", csv="a.csv"
    )
    assert json.loads((tmp_path / FIGURE_INDEX).read_text()) == {
        "prior_drift_vs_length": {
            "kind": "drift",
            "title": "drift",
            "csv": "d.csv",
            "paths": "d.npz",
        },
        "prior_recovery_vs_length": {"kind": "recovery", "title": "new", "csv": "a.csv"},
    }


def test_plot_recovery_writes_both_formats(
    stats: np.ndarray,
    baselines: dict[str, np.ndarray],
    saved_figures: list[Path],
    tmp_path: Path,
) -> None:
    frame = recovery_curves(stats, baselines, GRID, 8)
    plot_recovery(frame, tmp_path / "recovery", "recovery")
    assert [path.suffix for path in saved_figures] == [".png", ".pdf"]


def test_plot_drift_writes_both_formats(
    stats: np.ndarray,
    saved_figures: list[Path],
    tmp_path: Path,
) -> None:
    frame, per_path = drift_curves(stats, GRID, stats[:, 0, :])
    plot_drift(frame, per_path, GRID, tmp_path / "drift", "drift", max_traces=4)
    assert [path.suffix for path in saved_figures] == [".png", ".pdf"]


def test_plot_counts_writes_both_formats(
    stats: np.ndarray,
    baselines: dict[str, np.ndarray],
    saved_figures: list[Path],
    tmp_path: Path,
) -> None:
    frame = count_curves(
        stats[:, -1, :],
        baselines,
        _fresh_sampler,
        [8, 16],
        2,
        8,
        np.random.default_rng(5),
    )
    plot_counts(frame, tmp_path / "counts", "counts")
    assert [path.suffix for path in saved_figures] == [".png", ".pdf"]
