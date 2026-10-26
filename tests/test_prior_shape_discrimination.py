"""Gaussian against variance-matched Laplace prior, from saved prior rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bft_pmc.linear_regression import prior_shape_discrimination as shape
from bft_pmc.linear_regression.prior_shape_discrimination import (
    LAPLACE_SCALE,
    main,
    pmc_prior_samples,
    prediction_space_gap,
)

ROLLOUTS, PAIRS, DIM = 256, 24, 2


def _prior_run(root: Path, m: int, weight_scale: float = 1.0) -> Path:
    """A diagnostics run whose prior rollouts are noiseless y = w^T x, w ~ N(0, s^2 I)."""
    rng = np.random.default_rng(m)
    run = root / f"lr_m{m}_none_n500"
    run.mkdir(parents=True)
    x = rng.standard_normal((ROLLOUTS, PAIRS, DIM))
    w = weight_scale * rng.standard_normal((ROLLOUTS, DIM))
    y = np.einsum("rni,ri->rn", x, w)
    np.savez(run / "prior_trajectories.npz", rollout_x=x, rollout_y=y)
    return run


def test_least_squares_recovers_each_rollout_s_weights(tmp_path: Path) -> None:
    run = _prior_run(tmp_path, 32)
    with np.load(run / "prior_trajectories.npz") as bundle:
        x, y = bundle["rollout_x"], bundle["rollout_y"]
    w = np.linalg.lstsq(x[0], y[0], rcond=None)[0]
    assert np.allclose(pmc_prior_samples(run, PAIRS)[0], w)


def test_laplace_reference_matches_the_gaussian_variance() -> None:
    assert 2 * LAPLACE_SCALE**2 == pytest.approx(1.0)


def test_prediction_space_cannot_separate_zero_mean_priors() -> None:
    control = prediction_space_gap(np.random.default_rng(0), 4096, DIM)
    assert control["population_gap"] == 0.0
    assert control["sampled_rms_gap"] < 0.2


@pytest.mark.parametrize("match_variance", [False, True])
def test_comparison_writes_tables_figure_and_provenance(
    match_variance: bool, tmp_path: Path, saved_figures: list[Path]
) -> None:
    root = tmp_path / "diagnostics"
    _prior_run(root, 32)
    _prior_run(root, 128, weight_scale=0.8)  # under-dispersed, as the PMC priors are
    output_dir = tmp_path / "out"
    args = ["--diagnostics-root", str(root), "--output-dir", str(output_dir)]
    # M = 512 has no run and is skipped.
    args += ["--task-diversities", "32", "128", "512", "--length", str(PAIRS)]
    args += ["--n-blocks", "4", "--max-ed-samples", "128"]
    main(args + (["--match-variance"] if match_variance else []))

    table = pd.read_csv(output_dir / "prior_shape_discrimination.csv")
    assert table["task_diversity"].tolist() == [32, 128]
    assert table["variance_matched"].tolist() == [match_variance] * 2
    for column in ("ed_margin", "sw_margin", "ed_margin_se", "sw_margin_oracle"):
        assert np.isfinite(table[column]).all()
    # The samples are Gaussian, so the Gaussian reference is the closer one.
    assert (table["sw_margin_oracle"] > 0).all()

    record = json.loads((output_dir / "prior_shape_discrimination.json").read_text())
    assert record["generator"] == (
        "bft_pmc/linear_regression/prior_shape_discrimination.py"
    )
    assert record["prediction_space_control"]["population_gap"] == 0.0
    assert [Path(entry["path"]).parent.name for entry in record["inputs"]] == [
        "lr_m32_none_n500",
        "lr_m128_none_n500",
    ]
    assert [path.name for path in saved_figures] == ["prior_shape_discrimination.pdf"]


def test_variance_matching_rescales_to_unit_variance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved_figures: list[Path]
) -> None:
    root = tmp_path / "diagnostics"
    _prior_run(root, 32, weight_scale=0.5)
    seen: list[float] = []
    real_evaluate = shape.evaluate

    def spy(samples: np.ndarray, *args: object) -> dict[str, float]:
        seen.append(float(samples.var()))
        return real_evaluate(samples, *args)

    monkeypatch.setattr(shape, "evaluate", spy)
    args = ["--diagnostics-root", str(root), "--output-dir", str(tmp_path / "out")]
    main(args + ["--task-diversities", "32", "--length", str(PAIRS), "--n-blocks", "2"])
    main(
        args
        + [
            "--task-diversities",
            "32",
            "--length",
            str(PAIRS),
            "--n-blocks",
            "2",
            "--match-variance",
        ]
    )
    raw, oracle_raw, matched, oracle_matched = seen
    assert raw == pytest.approx(0.25, rel=0.2)
    assert matched == pytest.approx(1.0)


def test_no_runs_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no prior rollouts found"):
        main(["--diagnostics-root", str(tmp_path), "--task-diversities", "32"])
