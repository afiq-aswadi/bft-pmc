"""Stitched marginal plots across all task diversities (T) for the sweep.

Companion to `plot_marginals.py`. That script writes one figure per (T,
variant). This one reads the same `<sweep_dir>/samples/*.npz` files and
writes one *stitched* figure per (variant, mode), where

    rows = Ts (sorted low -> high)
    cols = all D weight dims
    mode = {density, cdf}

So for a sweep with variants prior / gaussian_L8 / discrete_L8 this emits 6
files: `stitched_density_{variant}.png` and `stitched_cdf_{variant}.png`.

Every cell is drawn by the shared `marginal_cell` helper from its own data only
(no `sharex`/`sharey`), so the (dim, M, variant) cell here is identical to the
same object in the 1x1 and grid plotters.

Usage:
    uv run python -m linear_regression.plot_stitched_marginals \
        --sweep-dir /abs/path/to/outputs/lr/sweep_analysis/sweep_YYYYMMDD_HHMMSS
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.stats import norm

from plotting.marginal_cell import (
    LR_VLINE_MAX_M,
    cell_xrange,
    draw_cdf_cell,
    draw_density_cell,
    legend_handles,
    ref_quantiles,
)


def _pick(data: dict, key: str, dim: int, is_prior: bool) -> np.ndarray:
    arr = data[key]
    return arr[:, dim] if is_prior else arr[0, :, dim]


def _has_analytic_mem(data: dict) -> bool:
    return "theta_pool" in data and "dmmse_weights" in data


def _mem_atomic(data: dict, dim: int, is_prior: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (atoms, weights) for the analytic memorising marginal at dim."""
    atoms = data["theta_pool"][:, dim]
    weights = np.asarray(data["dmmse_weights"])
    if not is_prior:
        weights = weights[0]
    return atoms, weights


def _cell_spec(data: dict, dim: int, is_prior: bool):
    """Return (lo, hi, gen_pdf, gen_cdf) for one (dim, variant) object."""
    pmc = _pick(data, "pt", dim, is_prior)
    if is_prior:
        lo, hi = cell_xrange(pmc, ref=ref_quantiles(lambda q: norm.ppf(q, 0.0, 1.0)))
        return (
            lo,
            hi,
            (lambda x: norm.pdf(x, 0.0, 1.0)),
            (lambda x: norm.cdf(x, 0.0, 1.0)),
        )
    mu = float(data["baseline_generalising_posterior_means"][0, dim])
    sigma = float(np.sqrt(data["baseline_generalising_posterior_covs"][0, dim, dim]))
    lo, hi = cell_xrange(pmc, ref=ref_quantiles(lambda q: norm.ppf(q, mu, sigma)))
    return (
        lo,
        hi,
        (lambda x, mu=mu, sigma=sigma: norm.pdf(x, mu, sigma)),
        (lambda x, mu=mu, sigma=sigma: norm.cdf(x, mu, sigma)),
    )


def plot_stitched(
    entries: list[tuple[int, dict]],
    output_path: Path,
    mode: str,
    is_prior: bool,
    plot_dmmse: bool,
) -> None:
    assert mode in ("density", "cdf"), mode
    assert entries, "no entries for stitched figure"

    first = entries[0][1]
    D = first["pt"].shape[1] if is_prior else first["pt"].shape[2]
    n_T = len(entries)

    fig, axes = plt.subplots(
        n_T,
        D,
        figsize=(1.5 * D, 1.0 * n_T),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    if n_T == 1:
        axes = axes.reshape(1, -1)
    if D == 1:
        axes = axes.reshape(-1, 1)

    for row, (M, data) in enumerate(entries):
        for d in range(D):
            ax = axes[row, d]
            pmc = _pick(data, "pt", d, is_prior)
            atoms, weights = (
                _mem_atomic(data, d, is_prior) if plot_dmmse else (None, None)
            )
            lo, hi, gen_pdf, gen_cdf = _cell_spec(data, d, is_prior)

            if mode == "density":
                draw_density_cell(
                    ax,
                    pmc_vals=pmc,
                    atoms=atoms,
                    weights=weights,
                    gen_pdf=gen_pdf,
                    lo=lo,
                    hi=hi,
                    is_prior=is_prior,
                    vline_max_m=LR_VLINE_MAX_M,
                )
            else:
                draw_cdf_cell(
                    ax,
                    pmc_vals=pmc,
                    atoms=atoms,
                    weights=weights,
                    gen_cdf=gen_cdf,
                    lo=lo,
                    hi=hi,
                )

            if d == 0:
                ax.set_ylabel(f"$M={M}$", fontsize="small")
            if row == 0:
                ax.set_title(f"dim {d}", fontsize="small")
            ax.tick_params(labelsize="xx-small")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    axes[0, 0].legend(handles=legend_handles(), frameon=False, fontsize="xx-small")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _parse_stem(stem: str) -> tuple[int, str, bool]:
    # T{M}_prior or T{M}_{source}_L{L}
    parts = stem.split("_")
    M = int(parts[0][1:])  # strip leading "T"
    if parts[1] == "prior":
        return M, "prior", True
    source = parts[1]
    L = parts[2][1:]
    return M, f"{source}_L{L}", False


def main(
    sweep_dir: Path,
    plot_dmmse: bool = True,
    output_subdir: str = "",
) -> None:
    samples_dir = sweep_dir / "samples"
    assert samples_dir.is_dir(), f"no samples/ under {sweep_dir}"

    out_dir = sweep_dir / output_subdir if output_subdir else sweep_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # group: variant -> list[(M, data, is_prior)]
    groups: dict[str, list[tuple[int, dict, bool]]] = defaultdict(list)
    for npz_path in sorted(samples_dir.glob("*.npz")):
        data = dict(np.load(npz_path))
        assert "pt" in data, f"{npz_path}: sample bundle is missing 'pt'"
        M, variant, is_prior = _parse_stem(npz_path.stem)
        assert is_prior or "baseline_generalising_posterior_means" in data, (
            f"{npz_path}: posterior bundle is missing analytic generalising means; "
            "rerun linear_regression/sweep_analysis.py"
        )
        if plot_dmmse:
            assert _has_analytic_mem(data), (
                f"{npz_path}: missing 'theta_pool'/'dmmse_weights'. The memorising "
                "marginal is analytically available and must not be drawn from MC samples. "
                "Rerun linear_regression/sweep_analysis.py to "
                "populate these fields, or pass --no-plot-dmmse to drop it."
            )
        groups[variant].append((M, data, is_prior))

    assert groups, f"no recognized .npz files in {samples_dir}"

    for variant, items in groups.items():
        items.sort(key=lambda t: t[0])
        is_prior = items[0][2]
        entries = [(M, data) for M, data, _ in items]

        for mode in ("density", "cdf"):
            output_path = out_dir / f"stitched_{mode}_{variant}.png"
            plot_stitched(
                entries=entries,
                output_path=output_path,
                mode=mode,
                is_prior=is_prior,
                plot_dmmse=plot_dmmse,
            )


if __name__ == "__main__":
    tyro.cli(main)
