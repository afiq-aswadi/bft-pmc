"""Stitched marginal grids across all task diversities (M) for a BAU sweep.

Companion to `plot_marginals.py`. That script writes one figure per
(run_id, source). This one reads the same `<sweep_dir>/samples/*.npz` files and
writes one stitched figure per (source, mode), where

    rows = M values (sorted low -> high)
    cols = V class dims
    mode = {density, cdf}

So for a sweep with sources prior / data_memorising / data_generalising this
emits 6 files: `stitched_bau_{mode}_{source}.png`.

Each cell is drawn by the shared `marginal_cell` helper on the fixed [0, 1]
simplex axis (no `sharex`/`sharey`), so the (class, M, source) cell matches the
1x1 / grid plotters.

Usage:
    uv run python -m balls_and_urns.plot_stitched_marginals \\
        --sweep-dir outputs/bau/sweep_analysis/YYYYMMDD_HHMMSS
"""

from collections import defaultdict
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

SOURCE_LABELS = {
    "prior": "Prior rollout",
    "data_generalising": "Generalizing prompt",
    "data_memorising": "Memorizing prompt",
}
PUB_DPI = 400


def _save_publication(fig, out_path: Path) -> None:
    """Save PNG (high DPI) and PDF (vector) alongside each other."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PUB_DPI, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def _alpha_for_cell(data: dict, prompt_idx: int) -> np.ndarray:
    if str(data["prompt_source"]) == "prior":
        return np.asarray(data["prior_dirichlet_alpha"])
    return np.asarray(data["posterior_dirichlet_alpha"][prompt_idx])


def plot_stitched(
    entries: list[tuple[int, dict]],
    output_path: Path,
    mode: str,
    prompt_idx: int,
    num_dims: int | None = None,
) -> None:
    assert mode in ("density", "cdf"), mode
    assert entries, "no entries for stitched figure"

    V_full = entries[0][1]["model_samples"].shape[2]
    V = min(num_dims, V_full) if num_dims else V_full
    n_M = len(entries)

    fig, axes = plt.subplots(
        n_M,
        V,
        figsize=(1.9 * V, 1.3 * n_M),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    if n_M == 1:
        axes = axes.reshape(1, -1)
    if V == 1:
        axes = axes.reshape(-1, 1)

    for row, (M, data) in enumerate(entries):
        alpha = _alpha_for_cell(data, prompt_idx)
        alpha_sum = float(alpha.sum())
        atoms_all = data["theta_pool"]
        mem_weights = data["posterior_pool_weights"][prompt_idx]
        model = data["model_samples"]
        is_prior = str(data["prompt_source"]) == "prior"
        for k in range(V):
            ax = axes[row, k]
            a, b = float(alpha[k]), float(alpha_sum - alpha[k])
            pmc = model[prompt_idx, :, k]
            atoms = atoms_all[:, k]
            lo, hi = cell_xrange(
                pmc,
                clip=(0.0, 1.0),
                ref=ref_quantiles(lambda q, a=a, b=b: beta_dist.ppf(q, a, b)),
            )
            if mode == "density":
                draw_density_cell(
                    ax,
                    pmc_vals=pmc,
                    atoms=atoms,
                    weights=mem_weights,
                    gen_pdf=lambda x, a=a, b=b: beta_dist.pdf(x, a, b),
                    lo=lo,
                    hi=hi,
                    is_prior=is_prior,
                    vline_max_m=BAU_VLINE_MAX_M,
                )
            else:
                draw_cdf_cell(
                    ax,
                    pmc_vals=pmc,
                    atoms=atoms,
                    weights=mem_weights,
                    gen_cdf=lambda x, a=a, b=b: beta_dist.cdf(x, a, b),
                    lo=lo,
                    hi=hi,
                )

            if k == 0:
                ax.set_ylabel(f"$M={M}$", fontsize="small")
            if row == 0:
                ax.set_title(f"Class {k + 1}", fontsize="small")
            ax.tick_params(labelsize="xx-small")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    axes[0, 0].legend(
        handles=legend_handles(), frameon=False, fontsize="xx-small", loc="best"
    )
    _save_publication(fig, output_path)
    plt.close(fig)


def main(
    sweep_dir: Path,
    prompt_idx: int = 0,
    output_subdir: str = "",
    num_dims: int | None = None,
) -> None:
    """Write stitched marginal grids (rows = M, cols = V classes) per source and mode.

    Args:
        sweep_dir: sweep-analysis run directory containing a samples/ subdir.
        prompt_idx: prompt index to render for posterior sources (prior is always 0).
        output_subdir: optional subdir under sweep_dir for outputs; default writes alongside samples/.
        num_dims: limit to first N class dims. None = all. Output files get a
            '{N}dim_' prefix when set, so they don't overwrite the all-V versions.
    """
    samples_dir = sweep_dir / "samples"
    assert samples_dir.is_dir(), f"no samples/ under {sweep_dir}"

    out_dir = sweep_dir / output_subdir if output_subdir else sweep_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for npz_path in sorted(samples_dir.glob("*.npz")):
        data = load_predictive_samples(npz_path)
        source = str(data["prompt_source"])
        M = int(data["theta_pool"].shape[0])
        groups[source].append((M, data))

    assert groups, f"no recognized .npz files in {samples_dir}"

    suffix = f"_{num_dims}dim" if num_dims else ""

    for source, items in groups.items():
        items.sort(key=lambda t: t[0])
        effective_pidx = 0 if source == "prior" else prompt_idx
        for mode in ("density", "cdf"):
            output_path = out_dir / f"stitched_bau_{mode}_{source}{suffix}.png"
            plot_stitched(
                entries=items,
                output_path=output_path,
                mode=mode,
                prompt_idx=effective_pidx,
                num_dims=num_dims,
            )


if __name__ == "__main__":
    tyro.cli(main)
