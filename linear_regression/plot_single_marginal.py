"""1x1 LR prior-marginal panel for publication.

Reads a single sweep `samples/<stem>.npz` and emits a tight single-axes density
figure via the shared `marginal_cell` helper: PMC histogram+KDE, Pi_infty
weighted histogram (analytic, no MC), and the population Gaussian. Because it uses the
same helper as the grid/stitched plotters, the (dim, M, variant) cell here is
identical to its appearance in those figures.

Usage:
    uv run python -m linear_regression.plot_single_marginal \\
        --npz-path outputs/lr/sweep_analysis/sweep_20260323_082051/samples/T8_prior.npz \\
        --dim 0 \\
        --out-path paper_data/lr/sweep/prior_marginal_M8_dim0.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.stats import norm

from plotting.marginal_cell import (
    LR_VLINE_MAX_M,
    cell_xrange,
    draw_density_cell,
    legend_handles,
    ref_quantiles,
)
from plotting.paper_style import apply_paper_style


def main(
    npz_path: Path,
    out_path: Path,
    dim: int = 0,
    figsize: tuple[float, float] = (3.2, 2.2),
    title: str | None = None,
) -> None:
    """Render a single density panel for one (M, variant, dim).

    Args:
        npz_path: Path to a sweep_analysis samples/*.npz file (prior or posterior).
        out_path: Output figure path. Suffix decides format (.pdf, .png, ...).
        dim: weight dimension to render.
        figsize: matplotlib figsize. Default sized for a single LaTeX subfigure.
        title: optional title; default omits.
    """
    data = dict(np.load(npz_path))
    assert "pt" in data, f"no 'pt' samples in {npz_path}"
    is_prior = data["pt"].ndim == 2

    pmc = data["pt"][:, dim] if is_prior else data["pt"][0, :, dim]

    assert "theta_pool" in data and "dmmse_weights" in data, (
        f"{npz_path}: missing 'theta_pool'/'dmmse_weights'. The memorising marginal "
        "is analytically available and must not be drawn from MC samples. Rerun "
        "linear_regression/sweep_analysis.py to populate these fields."
    )
    atoms = data["theta_pool"][:, dim]
    weights_all = np.asarray(data["dmmse_weights"])
    weights = weights_all if is_prior else weights_all[0]

    if is_prior:
        generalising = norm(loc=0.0, scale=1.0)
    else:
        mu = float(data["baseline_generalising_posterior_means"][0, dim])
        sigma = float(
            np.sqrt(data["baseline_generalising_posterior_covs"][0, dim, dim])
        )
        generalising = norm(loc=mu, scale=sigma)
    lo, hi = cell_xrange(pmc, ref=ref_quantiles(generalising.ppf))

    apply_paper_style(figsize[0], 0.40)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    draw_density_cell(
        ax,
        pmc_vals=pmc,
        atoms=atoms,
        weights=weights,
        gen_pdf=generalising.pdf,
        lo=lo,
        hi=hi,
        is_prior=is_prior,
        vline_max_m=LR_VLINE_MAX_M,
    )
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title, fontsize="medium")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=legend_handles(posterior=not is_prior),
        frameon=False,
        fontsize="small",
        loc="upper left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    tyro.cli(main)
