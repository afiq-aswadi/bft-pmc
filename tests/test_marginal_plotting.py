from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import norm

from balls_and_urns.dataset import save_predictive_samples
from balls_and_urns import plot_marginals as bau_marginals
from balls_and_urns import plot_stitched_marginals as bau_stitched
from linear_regression import plot_marginals as lr_marginals
from linear_regression import plot_single_marginal as lr_single
from linear_regression import plot_stitched_marginals as lr_stitched
from markov import plot_matrix_marginals as markov_matrix
from plotting.marginal_cell import (
    _analytic_peak,
    _validate_cell_inputs,
    cell_xrange,
    draw_atom_vlines,
    draw_cdf_cell,
    draw_density_cell,
    legend_handles,
    ref_quantiles,
)


def _write_lr_samples(sweep_dir: Path) -> tuple[Path, Path]:
    samples_dir = sweep_dir / "samples"
    samples_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    theta_pool = rng.normal(size=(4, 2))
    prior_path = samples_dir / "T4_prior.npz"
    np.savez(
        prior_path,
        pt=rng.normal(size=(32, 2)),
        theta_pool=theta_pool,
        dmmse_weights=np.full(4, 0.25),
    )
    posterior_path = samples_dir / "T4_memorising_L2.npz"
    np.savez(
        posterior_path,
        pt=rng.normal(size=(1, 32, 2)),
        theta_pool=theta_pool,
        dmmse_weights=np.full((1, 4), 0.25),
        baseline_generalising_posterior_means=np.zeros((1, 2)),
        baseline_generalising_posterior_covs=np.eye(2)[None],
    )
    return prior_path, posterior_path


def _write_bau_sample(path: Path, source: str) -> None:
    rng = np.random.default_rng(4)
    model_samples = rng.dirichlet(np.ones(2), size=(1, 32))
    theta_pool = rng.dirichlet(np.ones(2), size=4)
    save_predictive_samples(
        path,
        model_samples,
        model_samples.copy(),
        model_samples.copy(),
        np.array([[3.0, 4.0]]),
        np.full((1, 4), 0.25),
        np.array([1.0, 1.0]),
        theta_pool,
        np.array([[0, 1]]),
        step=2,
        prompt_source=source,
    )


def test_shared_marginal_rendering_paths() -> None:
    samples = np.linspace(-1.0, 1.0, 32)
    lo, hi = cell_xrange(samples)
    clipped_lo, clipped_hi = cell_xrange(samples, clip=(-0.5, 0.5), ref=(-2.0, 2.0))
    assert lo < hi
    assert (clipped_lo, clipped_hi) == (-0.5, 0.5)
    assert ref_quantiles(norm.ppf)[0] < 0
    assert _analytic_peak(np.array([np.inf, np.nan])) == 0

    fig, axes = plt.subplots(2, 3)
    draw_atom_vlines(axes[0, 0], np.array([0.0]), np.array([0.0]))
    draw_density_cell(
        axes[0, 0],
        pmc_vals=samples,
        atoms=np.array([-0.5, 0.5]),
        weights=np.array([0.5, 0.5]),
        gen_pdf=norm.pdf,
        lo=lo,
        hi=hi,
        is_prior=True,
        vline_max_m=2,
    )
    draw_density_cell(
        axes[0, 1],
        pmc_vals=samples,
        atoms=np.linspace(-1, 1, 4),
        weights=np.full(4, 0.25),
        gen_pdf=norm.pdf,
        lo=lo,
        hi=hi,
        is_prior=True,
        vline_max_m=2,
    )
    draw_density_cell(
        axes[0, 2],
        pmc_vals=np.ones(16),
        atoms=None,
        weights=None,
        gen_pdf=lambda x: np.zeros_like(x),
        lo=0.0,
        hi=2.0,
    )
    draw_cdf_cell(
        axes[1, 0],
        pmc_vals=samples,
        atoms=np.array([0.0]),
        weights=np.array([1.0]),
        gen_cdf=norm.cdf,
        lo=lo,
        hi=hi,
    )
    draw_cdf_cell(
        axes[1, 1],
        pmc_vals=samples,
        atoms=None,
        weights=None,
        gen_cdf=norm.cdf,
        lo=lo,
        hi=hi,
    )
    assert len(legend_handles()) == 3
    plt.close(fig)


def test_shared_marginal_inputs_fail_fast() -> None:
    with pytest.raises(ValueError, match="pmc_vals"):
        cell_xrange(np.array([]))

    valid = np.array([0.0, 1.0])
    invalid_cases = [
        (np.array([]), None, None, 0.0, 1.0, "pmc_vals"),
        (valid, None, None, 1.0, 0.0, "x-range"),
        (valid, None, np.array([1.0]), 0.0, 1.0, "without atoms"),
        (valid, np.array([0.0]), None, 0.0, 1.0, "weights are required"),
        (
            valid,
            np.array([0.0, 1.0]),
            np.array([1.0]),
            0.0,
            1.0,
            "equal shape",
        ),
        (
            valid,
            np.array([np.nan]),
            np.array([1.0]),
            0.0,
            1.0,
            "finite",
        ),
        (
            valid,
            np.array([0.0, 1.0]),
            np.array([0.8, -0.2]),
            0.0,
            1.0,
            "sum to one",
        ),
    ]
    for pmc, atoms, weights, lo, hi, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            _validate_cell_inputs(pmc, atoms, weights, lo, hi)


def test_lr_marginal_entry_points(
    tmp_path: Path,
    saved_figures: list[Path],
) -> None:
    sweep_dir = tmp_path / "lr"
    prior_path, posterior_path = _write_lr_samples(sweep_dir)
    lr_marginals.main(sweep_dir, output_subdir="figures", num_dims=1)
    lr_stitched.main(sweep_dir, output_subdir="figures")
    lr_single.main(prior_path, tmp_path / "single_prior.pdf", title="Prior")
    lr_single.main(posterior_path, tmp_path / "single_posterior.pdf")
    assert len(saved_figures) == 8


def test_lr_marginal_legacy_and_validation_paths(
    tmp_path: Path,
    saved_figures: list[Path],
) -> None:
    sweep_dir = tmp_path / "legacy"
    samples_dir = sweep_dir / "samples"
    samples_dir.mkdir(parents=True)
    np.savez(samples_dir / "ignored.npz", unrelated=np.ones(1))
    np.savez(samples_dir / "T2_generalising_L2.npz", pt=np.ones((1, 4, 1)))
    with pytest.raises(AssertionError, match="analytic generalising means"):
        lr_marginals.main(sweep_dir, plot_dmmse=False)
    with pytest.raises(AssertionError, match="analytic generalising means"):
        lr_stitched.main(sweep_dir, plot_dmmse=False)
    assert not saved_figures

    ignored_dir = tmp_path / "ignored"
    (ignored_dir / "samples").mkdir(parents=True)
    np.savez(ignored_dir / "samples/ignored.npz", unrelated=np.ones(1))
    with pytest.raises(AssertionError, match="missing 'pt'"):
        lr_marginals.main(ignored_dir, plot_dmmse=False)
    with pytest.raises(AssertionError, match="missing 'pt'"):
        lr_stitched.main(ignored_dir, plot_dmmse=False)

    empty_dir = tmp_path / "empty"
    (empty_dir / "samples").mkdir(parents=True)
    with pytest.raises(AssertionError, match="no recognized"):
        lr_stitched.main(empty_dir, plot_dmmse=False)

    with pytest.raises(AssertionError, match="prior grid expects"):
        lr_marginals.plot_prior_grid(
            {"pt": np.ones((1, 2, 3))},
            tmp_path / "bad.png",
            "bad",
            False,
        )
    with pytest.raises(AssertionError, match="analytic generalising"):
        lr_marginals.plot_posterior_grid(
            {"pt": np.ones((1, 2, 1))},
            tmp_path / "bad.png",
            "bad",
            False,
        )
    with pytest.raises(AssertionError, match="no entries"):
        lr_stitched.plot_stitched([], tmp_path / "bad.png", "density", True, False)
    with pytest.raises(AssertionError):
        lr_stitched.plot_stitched(
            [(1, {"pt": np.ones((2, 1))})], tmp_path / "bad.png", "x", True, False
        )
    assert lr_stitched._parse_stem("T8_prior") == (8, "prior", True)
    lr_stitched.plot_stitched(
        [(1, {"pt": np.linspace(-1, 1, 16)[:, None]})],
        tmp_path / "one_dimension.png",
        "density",
        True,
        False,
    )


def test_bau_marginal_entry_points(
    tmp_path: Path,
    saved_figures: list[Path],
) -> None:
    sweep_dir = tmp_path / "bau"
    samples_dir = sweep_dir / "samples"
    _write_bau_sample(samples_dir / "M4_prior.npz", "prior")
    _write_bau_sample(samples_dir / "M4_data_memorising.npz", "data_memorising")

    bau_marginals.main(sweep_dir, output_subdir="figures", num_dims=1)
    bau_stitched.main(sweep_dir, output_subdir="figures", num_dims=1)
    assert len(saved_figures) == 12


def test_bau_marginal_validation_paths(tmp_path: Path) -> None:
    data = {
        "model_samples": np.ones((1, 2, 2)) / 2,
        "theta_pool": np.ones((2, 3)) / 3,
        "posterior_pool_weights": np.ones((1, 2)) / 2,
        "prompt_source": "prior",
        "prior_dirichlet_alpha": np.ones(2),
    }
    with pytest.raises(AssertionError):
        bau_marginals.plot_grid(data, tmp_path / "bad.png", 0)
    with pytest.raises(AssertionError):
        bau_stitched.plot_stitched([], tmp_path / "bad.png", "density", 0)
    with pytest.raises(AssertionError):
        bau_stitched.plot_stitched([(1, data)], tmp_path / "bad.png", "invalid", 0)


def test_markov_matrix_marginal_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    calls: list[tuple[Path, str]] = []

    def record_plot(
        samples: np.ndarray,
        training: np.ndarray,
        output_path: Path,
        **kwargs: object,
    ) -> None:
        assert samples.ndim == 3
        assert training.ndim == 3
        calls.append((output_path, str(kwargs["mode"])))

    monkeypatch.setattr(markov_matrix, "plot_pmc_distribution_matrix", record_plot)
    with pytest.raises(FileNotFoundError, match="missing Markov sample bundles"):
        markov_matrix._process_run(
            samples_dir,
            tmp_path,
            2,
            prompt_index=0,
            panel_size=1.0,
            max_classes=1,
        )

    rng = np.random.default_rng(5)
    training = rng.dirichlet(np.ones(2), size=(2, 2))
    model_samples = rng.dirichlet(np.ones(2), size=(1, 4, 2))
    prior_samples = model_samples.reshape(1, 4, 2, 2)
    np.savez(
        samples_dir / "M2_prior.npz",
        training_transition_matrices=training,
        model_samples=prior_samples,
    )
    for source in ["in_distribution", "out_of_distribution"]:
        np.savez(
            samples_dir / f"M2_{source}_L8.npz",
            training_transition_matrices=training,
            model_samples=prior_samples,
            prompt_tokens=np.array([[0, 1]]),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_matrix_marginals.py",
            "--samples-dir",
            str(samples_dir),
            "--out-dir",
            str(tmp_path / "figures"),
            "--max-classes",
            "1",
        ],
    )
    markov_matrix.main()
    assert len(calls) == 6
    assert {mode for _, mode in calls} == {"density", "cdf"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_matrix_marginals.py",
            "--samples-dir",
            str(samples_dir),
            "--n-chains",
            "2",
            "--max-classes",
            "1",
        ],
    )
    markov_matrix.main()
    assert len(calls) == 12
    assert (tmp_path / "outputs/markov/matrix_marginals_K1").is_dir()
