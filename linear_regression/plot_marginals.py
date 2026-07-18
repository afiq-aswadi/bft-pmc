"""Regenerate full-D marginal grids from a completed sweep's saved samples.

Reads `.npz` files under `<sweep_dir>/samples/` (written by
`linear_regression/sweep_analysis.py`) and produces companion
`grid_marginal_<stem>.png` figures that show every weight dimension (not just
dims 0-3 like the main-text 4-dim figures).

No retraining or PMC computation -- purely an offline plotter. Every cell is drawn by
the shared `marginal_cell` helper, so a given (dim, M, source) object renders
identically here, in the 1x1 panel, and in the stitched multi-M grids.

Usage:
    uv run python -m linear_regression.plot_marginals \
        --sweep-dir /abs/path/to/outputs/lr/sweep_analysis/sweep_YYYYMMDD_HHMMSS
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.stats import norm

from linear_regression.analysis.config import SOURCE_DISPLAY_LABELS
from plotting.marginal_cell import (
    LR_VLINE_MAX_M,
    cell_xrange,
    draw_cdf_cell,
    draw_density_cell,
    legend_handles,
    ref_quantiles,
)
from plotting.paper_style import apply_paper_style


def _figsize_for_D(D: int) -> tuple[float, float]:
    # matches the feel of the 4-dim (12, 4) figure: ~3.0 in per column, 4.2 tall
    return (max(3.0 * D, 10.0), 4.2)


def _hide_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_prior_grid(
    data: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    plot_dmmse: bool,
    num_dims: int | None = None,
) -> None:
    pt_all = data["pt"]
    assert pt_all.ndim == 2, (
        f"prior grid expects 2D pt samples, got shape {pt_all.shape}"
    )
    D = min(num_dims, pt_all.shape[1]) if num_dims else pt_all.shape[1]

    if plot_dmmse:
        assert "theta_pool" in data and "dmmse_weights" in data, (
            f"{output_path}: missing 'theta_pool'/'dmmse_weights'. The memorising "
            "marginal is analytically available and must not be drawn from MC samples. "
            "Rerun linear_regression/sweep_analysis.py to populate "
            "these fields, or pass --no-plot-dmmse to drop the memorising layer."
        )

    figsize = _figsize_for_D(D)
    apply_paper_style(figsize[0], 0.9)
    fig, axes = plt.subplots(2, D, figsize=figsize, constrained_layout=True)
    axes = axes.reshape(2, D)
    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("CDF")

    for d in range(D):
        pmc = pt_all[:, d]
        atoms = data["theta_pool"][:, d] if plot_dmmse else None
        weights = np.asarray(data["dmmse_weights"]) if plot_dmmse else None
        lo, hi = cell_xrange(pmc, ref=ref_quantiles(lambda q: norm.ppf(q, 0.0, 1.0)))

        axes[0, d].set_title(f"dim {d}", fontsize="small")
        draw_density_cell(
            axes[0, d],
            pmc_vals=pmc,
            atoms=atoms,
            weights=weights,
            gen_pdf=lambda x: norm.pdf(x, 0.0, 1.0),
            lo=lo,
            hi=hi,
            is_prior=True,
            vline_max_m=LR_VLINE_MAX_M,
        )
        draw_cdf_cell(
            axes[1, d],
            pmc_vals=pmc,
            atoms=atoms,
            weights=weights,
            gen_cdf=lambda x: norm.cdf(x, 0.0, 1.0),
            lo=lo,
            hi=hi,
        )
        _hide_spines(axes[0, d])
        _hide_spines(axes[1, d])

    fig.legend(
        handles=legend_handles(), loc="outside upper center", ncol=3, frameon=False
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_posterior_grid(
    data: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    plot_dmmse: bool,
    num_dims: int | None = None,
) -> None:
    pt_all = data["pt"]
    assert pt_all.ndim == 3, (
        f"posterior grid expects 3D pt samples, got shape {pt_all.shape}"
    )
    D = min(num_dims, pt_all.shape[2]) if num_dims else pt_all.shape[2]

    assert "baseline_generalising_posterior_means" in data, (
        f"{output_path}: posterior grid needs analytic generalising posterior params."
    )
    if plot_dmmse:
        assert "theta_pool" in data and "dmmse_weights" in data, (
            f"{output_path}: missing 'theta_pool'/'dmmse_weights'. The memorising "
            "posterior is analytically available and must not be drawn from MC samples."
        )

    means = data["baseline_generalising_posterior_means"][0]
    covs = data["baseline_generalising_posterior_covs"][0]

    figsize = _figsize_for_D(D)
    apply_paper_style(figsize[0], 0.9)
    fig, axes = plt.subplots(2, D, figsize=figsize, constrained_layout=True)
    axes = axes.reshape(2, D)
    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("CDF")

    for d in range(D):
        pmc = pt_all[0, :, d]
        mu = float(means[d])
        sigma = float(np.sqrt(covs[d, d]))
        atoms = data["theta_pool"][:, d] if plot_dmmse else None
        weights = np.asarray(data["dmmse_weights"])[0] if plot_dmmse else None
        lo, hi = cell_xrange(
            pmc, ref=ref_quantiles(lambda q, mu=mu, sigma=sigma: norm.ppf(q, mu, sigma))
        )

        axes[0, d].set_title(f"dim {d}", fontsize="small")
        draw_density_cell(
            axes[0, d],
            pmc_vals=pmc,
            atoms=atoms,
            weights=weights,
            gen_pdf=lambda x, mu=mu, sigma=sigma: norm.pdf(x, mu, sigma),
            lo=lo,
            hi=hi,
            is_prior=False,  # posterior: always histogram
        )
        draw_cdf_cell(
            axes[1, d],
            pmc_vals=pmc,
            atoms=atoms,
            weights=weights,
            gen_cdf=lambda x, mu=mu, sigma=sigma: norm.cdf(x, mu, sigma),
            lo=lo,
            hi=hi,
        )
        _hide_spines(axes[0, d])
        _hide_spines(axes[1, d])

    fig.legend(
        handles=legend_handles(posterior=True),
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@dataclass
class MarginalPlotConfig:
    sweep_dir: Path
    """Absolute path to a sweep run directory (containing a samples/ subdir)."""

    plot_dmmse: bool = True
    """Include memorising (dMMSE) baseline."""

    output_subdir: str = ""
    """Optional subdir under sweep_dir to write into. Default '' writes beside marginal_*.png."""


def main(
    sweep_dir: Path,
    plot_dmmse: bool = True,
    output_subdir: str = "",
    num_dims: int | None = None,
) -> None:
    """Generate per-T grid marginals from saved sweep samples.

    Args:
        num_dims: limit to first N dims. None = all. Output files get a
            '{N}dim_' prefix when set, so they don't overwrite the all-D versions.
    """
    config = MarginalPlotConfig(
        sweep_dir=sweep_dir,
        plot_dmmse=plot_dmmse,
        output_subdir=output_subdir,
    )
    samples_dir = config.sweep_dir / "samples"
    assert samples_dir.is_dir(), f"no samples/ under {config.sweep_dir}"

    out_dir = (
        config.sweep_dir / config.output_subdir
        if config.output_subdir
        else config.sweep_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(samples_dir.glob("*.npz"))
    assert npz_paths, f"no .npz files under {samples_dir}"

    prefix = f"grid_marginal_{num_dims}dim_" if num_dims else "grid_marginal_"

    for npz_path in npz_paths:
        data = dict(np.load(npz_path))
        assert "pt" in data, f"{npz_path}: sample bundle is missing 'pt'"

        stem = npz_path.stem
        output_path = out_dir / f"{prefix}{stem}.png"

        is_prior = data["pt"].ndim == 2
        if is_prior:
            M = stem.split("_")[0][1:]
            plot_prior_grid(
                data=data,
                output_path=output_path,
                title=rf"$M = {M}$",
                plot_dmmse=config.plot_dmmse,
                num_dims=num_dims,
            )
        else:
            assert "baseline_generalising_posterior_means" in data, (
                f"{npz_path}: posterior bundle is missing analytic generalising means; "
                "rerun linear_regression/sweep_analysis.py"
            )
            parts = stem.split("_")
            M, source, L = parts[0][1:], parts[1], parts[2][1:]
            plot_posterior_grid(
                data=data,
                output_path=output_path,
                title=rf"$M = {M}$, {SOURCE_DISPLAY_LABELS.get(source, source)}, $n_{{\mathrm{{prompt}}}} = {L}$",
                plot_dmmse=config.plot_dmmse,
                num_dims=num_dims,
            )


if __name__ == "__main__":
    tyro.cli(main)
