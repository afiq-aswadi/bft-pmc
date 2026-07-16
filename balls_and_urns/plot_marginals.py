"""Marginal-distribution grids for BAU sweep-analysis predictive samples.

Reads `.npz` files under `<sweep_dir>/samples/` (written by the BAU sweep-analysis
experiment) and produces a `grid_marginal_<stem>.png` per file. Each grid is
`2 x vocab_size`: density on top, CDF on bottom, columns are vocab dims.

For BAU each sample is a probability vector on the simplex, so the per-dim marginal
of the Dirichlet generalising posterior is closed-form Beta. The memorising posterior
is a discrete mixture of point masses on `theta_pool` whose closed-form weights are
saved as `posterior_pool_weights` (uniform 1/M for the prior). All cells are drawn by
the shared `marginal_cell` helper on the fixed [0, 1] simplex axis, so a given
(class, M, source) cell is identical here and in the stitched BAU grids.

Usage:
    uv run python -m balls_and_urns.plot_marginals \\
        --sweep-dir outputs/bau/sweep_analysis/YYYYMMDD_HHMMSS
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.stats import beta as beta_dist

from balls_and_urns.dataset import load_predictive_samples
from plotting.marginal_cell import (
    BAU_VLINE_MAX_M,
    cell_xrange,
    draw_cdf_cell,
    draw_density_cell,
    legend_handles,
    ref_quantiles,
)

PUB_DPI = 400


def _save_publication(fig, out_path: Path) -> None:
    """Save PNG (high DPI) and PDF (vector) alongside each other."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PUB_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def _figsize_for_V(V: int) -> tuple[float, float]:
    return (max(3.0 * V, 10.0), 4.2)


def plot_grid(
    data: dict[str, np.ndarray],
    output_path: Path,
    prompt_idx: int,
    num_dims: int | None = None,
) -> None:
    model = data["model_samples"]
    theta_pool = data["theta_pool"]
    pool_weights = data["posterior_pool_weights"]
    prompt_source = str(data["prompt_source"])

    n_prompts, _, V_full = model.shape
    M = theta_pool.shape[0]
    assert theta_pool.shape == (M, V_full)
    assert pool_weights.shape == (n_prompts, M)
    assert 0 <= prompt_idx < n_prompts

    if prompt_source == "prior":
        alpha = np.asarray(data["prior_dirichlet_alpha"])
    else:
        alpha = np.asarray(data["posterior_dirichlet_alpha"][prompt_idx])
    alpha_sum = float(alpha.sum())

    V = min(num_dims, V_full) if num_dims else V_full
    mem_weights = pool_weights[prompt_idx]

    fig, axes = plt.subplots(2, V, figsize=_figsize_for_V(V), constrained_layout=True)
    axes = axes.reshape(2, V)

    for k in range(V):
        ax_d, ax_c = axes[0, k], axes[1, k]
        ax_d.set_title(f"Class {k + 1}", fontsize="large")
        a, b = float(alpha[k]), float(alpha_sum - alpha[k])
        atoms = theta_pool[:, k]
        pmc = model[prompt_idx, :, k]
        lo, hi = cell_xrange(
            pmc,
            clip=(0.0, 1.0),
            ref=ref_quantiles(lambda q, a=a, b=b: beta_dist.ppf(q, a, b)),
        )

        draw_density_cell(
            ax_d,
            pmc_vals=pmc,
            atoms=atoms,
            weights=mem_weights,
            gen_pdf=lambda x, a=a, b=b: beta_dist.pdf(x, a, b),
            lo=lo,
            hi=hi,
            is_prior=(prompt_source == "prior"),
            vline_max_m=BAU_VLINE_MAX_M,
        )
        draw_cdf_cell(
            ax_c,
            pmc_vals=pmc,
            atoms=atoms,
            weights=mem_weights,
            gen_cdf=lambda x, a=a, b=b: beta_dist.cdf(x, a, b),
            lo=lo,
            hi=hi,
        )

        if k == 0:
            ax_d.set_ylabel("Density", fontsize="medium")
            ax_c.set_ylabel("CDF", fontsize="medium")
        for ax in (ax_d, ax_c):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    axes[0, 0].legend(
        handles=legend_handles(), frameon=False, fontsize="small", loc="best"
    )
    _save_publication(fig, output_path)
    plt.close(fig)


def main(
    sweep_dir: Path,
    prompt_idx: int = 0,
    output_subdir: str = "",
    num_dims: int | None = None,
) -> None:
    """Plot marginal-distribution grids for every .npz under sweep_dir/samples/.

    Args:
        sweep_dir: sweep-analysis run directory containing a samples/ subdir.
        prompt_idx: which prompt to render for posterior sources (prior is always prompt 0).
        output_subdir: optional subdir under sweep_dir for outputs; default writes alongside samples/.
        num_dims: limit to first N class dims. None = all. Output files get a
            '{N}dim_' prefix when set, so they don't overwrite the all-V versions.
    """
    samples_dir = sweep_dir / "samples"
    assert samples_dir.is_dir(), f"no samples/ under {sweep_dir}"

    out_dir = sweep_dir / output_subdir if output_subdir else sweep_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(samples_dir.glob("*.npz"))
    assert npz_paths, f"no .npz files under {samples_dir}"

    prefix = f"grid_marginal_{num_dims}dim_" if num_dims else "grid_marginal_"

    for npz_path in npz_paths:
        data = load_predictive_samples(npz_path)
        stem = npz_path.stem
        source = str(data["prompt_source"])
        output_path = out_dir / f"{prefix}{stem}.png"
        plot_grid(
            data=data,
            output_path=output_path,
            prompt_idx=0 if source == "prior" else prompt_idx,
            num_dims=num_dims,
        )


if __name__ == "__main__":
    tyro.cli(main)
