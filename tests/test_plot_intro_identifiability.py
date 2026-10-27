from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

import bft_pmc.linear_regression.plot_intro_identifiability as figure2


@pytest.fixture(scope="module")
def slab_var() -> float:
    return figure2.calibrated_slab_variance()


def _grid_mean(log_prior_1d) -> np.ndarray:
    """Posterior mean by brute-force integration over a wide, fine grid."""
    grid = np.linspace(-7.0, 7.0, 1601)
    w1, w2 = np.meshgrid(grid, grid, indexing="xy")
    log_post = (
        -0.5 * (figure2.Y_OBS - (w1 + w2)) ** 2 / figure2.SIGMA2
        + log_prior_1d(w1)
        + log_prior_1d(w2)
    )
    post = np.exp(log_post - log_post.max())
    post /= post.sum()
    return np.array([(w1 * post).sum(), (w2 * post).sum()])


def test_calibration_makes_the_two_posterior_means_identical(slab_var: float) -> None:
    gaussian = figure2.posterior_mean(figure2.gaussian_posterior())
    spike_slab = figure2.posterior_mean(figure2.spike_slab_posterior(slab_var))
    np.testing.assert_allclose(spike_slab, gaussian, rtol=0.0, atol=1e-12)
    # Under N(0, I) with one observation, the mean is x y / (x^T x + sigma^2).
    np.testing.assert_allclose(gaussian, [1 / 2.05, 1 / 2.05])


def test_point_predictions_coincide_at_the_orthogonal_query(slab_var: float) -> None:
    for components in (
        figure2.gaussian_posterior(),
        figure2.spike_slab_posterior(slab_var),
    ):
        prediction = figure2.X_QUERY @ figure2.posterior_mean(components)
        assert prediction == pytest.approx(0.0, abs=1e-12)


def test_closed_form_means_agree_with_numerical_integration(slab_var: float) -> None:
    np.testing.assert_allclose(
        _grid_mean(lambda w: figure2._log_prior_ss_1d(w, slab_var)),
        figure2.posterior_mean(figure2.spike_slab_posterior(slab_var)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _grid_mean(lambda w: -0.5 * w**2 / figure2.TAU2),
        figure2.posterior_mean(figure2.gaussian_posterior()),
        atol=1e-6,
    )


def test_predictives_are_unimodal_and_bimodal_around_the_shared_prediction(
    slab_var: float,
) -> None:
    y = np.linspace(-15.0, 15.0, 30001)
    step = y[1] - y[0]
    gaussian = figure2.predictive_density(figure2.gaussian_posterior(), y)
    spike_slab = figure2.predictive_density(figure2.spike_slab_posterior(slab_var), y)
    assert gaussian.sum() * step == pytest.approx(1.0, abs=1e-6)
    assert spike_slab.sum() * step == pytest.approx(1.0, abs=1e-6)
    at_prediction = int(np.argmin(np.abs(y)))
    assert int(gaussian.argmax()) == at_prediction
    assert spike_slab[at_prediction] < 0.5 * spike_slab.max()


def test_main_draws_the_figure(
    saved_figures: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "figure.png"
    monkeypatch.setattr(
        sys, "argv", ["plot_intro_identifiability.py", "--out-path", str(out_path)]
    )
    with matplotlib.rc_context():
        figure2.main()
    assert saved_figures == [out_path]
    assert "shared posterior mean [0.48780488 0.48780488]" in capsys.readouterr().out


def test_main_refuses_posteriors_whose_means_differ(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(figure2, "calibrated_slab_variance", lambda: 1.0)
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_intro_identifiability.py", "--out-path", str(tmp_path / "figure.png")],
    )
    with pytest.raises(AssertionError, match="posterior means differ"):
        figure2.main()
