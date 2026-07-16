"""Shared rendering primitives for one posterior or prior marginal cell.

Every marginal plotter (LR / BAU single-panel, grid, and stitched multi-M
figures, plus the Markov matrix panels) draws the same three curves:

  - PMC      (goldenrod)  empirical histogram + KDE / empirical CDF.
  - Pi_M     (tab:green)  discrete *memorising* prior/posterior on the finite
                          training pool of M tasks, with closed-form weights.
                          Rendered as a `density=True` weighted histogram
                          (density panel) and a closed-form weighted step CDF
                          (CDF panel). As M grows this fills in toward the
                          Pi_infty curve -- the memorising->generalising story --
                          so we keep the histogram rather than atom markers.
  - Pi_infty (tab:blue)   analytic *generalising* population prior/posterior
                          (continuous: Beta for BAU/Markov, Gaussian for LR),
                          passed in as a callable so this module is domain-agnostic.

Crucial invariant (so the *same object* looks identical regardless of which
figure it appears in): a cell's x-range, bin edges, and y-limit depend **only on
that object's own data** — never on neighbouring cells. Callers must therefore
NOT use matplotlib `sharex`/`sharey`; each cell is self-contained.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# Canonical colours + labels, matching the paper:
#   Pi_M    = empirical training prior (discrete, memorising)   -> green
#   Pi_infty = population prior (continuous, generalising)        -> blue curve
COLORS = {"pmc": "goldenrod", "mem": "tab:green", "gen": "tab:blue"}
LABELS = {"pmc": "PMC", "mem": r"$\Pi_M$", "gen": r"$\Pi_\infty$"}

N_BINS = 60  # histogram bins over the cell range
N_GRID = 500  # points for analytic curves
# Prior-only Pi_M vline cutoffs (M <= cutoff -> vlines, else weighted histogram).
# Domain-specific so each family switches to the histogram at the right diversity.
VLINE_MAX_M = 32  # default
LR_VLINE_MAX_M = 16  # LR: histogram from M >= 32
BAU_VLINE_MAX_M = 8  # BAU: histogram from M >= 16


def ref_quantiles(ppf, q: float = 0.005) -> tuple[float, float]:
    """Central [q, 1-q] quantiles of the Π_∞ reference via its ppf (inverse CDF)."""
    return float(ppf(q)), float(ppf(1.0 - q))


def cell_xrange(
    pmc_vals: np.ndarray,
    *,
    clip: tuple[float, float] | None = None,
    ref: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Deterministic, data-following x-range for one object.

    The range is the union of where the PMC samples and the Π_∞ reference have
    mass, so it neither leaves dead whitespace (e.g. a BAU Beta lives in ~[0,0.4],
    not [0,1]) nor clips the Π_∞ curve. A 15% margin is added and the result is
    clamped to the variable's support.

    Args:
        pmc_vals: this object's PMC samples (1-D).
        clip: hard [min, max] the range may never exceed (e.g. (0, 1) for a
            probability / simplex coordinate).
        ref: the Π_∞ reference's central quantiles (use `ref_quantiles(dist.ppf)`).
            For the prior Π_∞ is M-independent, so this fixes the range across M.
    """
    pmc_vals = np.asarray(pmc_vals)
    if pmc_vals.ndim != 1 or pmc_vals.size == 0 or not np.isfinite(pmc_vals).all():
        raise ValueError("pmc_vals must be a non-empty finite vector.")
    lo, hi = np.percentile(pmc_vals, [0.5, 99.5])
    if ref is not None:
        lo, hi = min(lo, ref[0]), max(hi, ref[1])
    span = hi - lo
    margin = 0.15 * span if span > 1e-12 else max(0.03, 0.05 * abs(float(lo)))
    lo, hi = float(lo - margin), float(hi + margin)
    if clip is not None:
        lo, hi = max(lo, clip[0]), min(hi, clip[1])
    return lo, hi


def _validate_cell_inputs(
    pmc_vals: np.ndarray,
    atoms: np.ndarray | None,
    weights: np.ndarray | None,
    lo: float,
    hi: float,
) -> None:
    """Validate one marginal cell's samples, support, and plotting range."""
    pmc_vals = np.asarray(pmc_vals)
    if pmc_vals.ndim != 1 or pmc_vals.size == 0 or not np.isfinite(pmc_vals).all():
        raise ValueError("pmc_vals must be a non-empty finite vector.")
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError(f"expected a finite increasing x-range, got ({lo}, {hi}).")
    if atoms is None:
        if weights is not None:
            raise ValueError("weights cannot be provided without atoms.")
        return
    if weights is None:
        raise ValueError("weights are required when atoms are provided.")

    atoms = np.asarray(atoms)
    weights = np.asarray(weights)
    if atoms.ndim != 1 or weights.shape != atoms.shape or atoms.size == 0:
        raise ValueError("atoms and weights must be non-empty vectors of equal shape.")
    if not np.isfinite(atoms).all() or not np.isfinite(weights).all():
        raise ValueError("atoms and weights must be finite.")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("weights must be non-negative and sum to one.")


def _kde_or_none(vals: np.ndarray, x_grid: np.ndarray) -> np.ndarray | None:
    return gaussian_kde(vals)(x_grid) if np.std(vals) > 1e-8 else None


def draw_atom_vlines(ax: Axes, atoms: np.ndarray, weights: np.ndarray) -> None:
    """Π_M as full-panel-height vertical lines, opacity ∝ weight.

    Used at low M (few tasks) where discrete markers read more clearly than a
    histogram: prior weights are uniform so all lines share one opacity; a
    posterior's favoured task stands out while near-zero-weight tasks fade. Lines
    span the panel in axes coords, so they're independent of the density y-limit.
    """
    w = np.asarray(weights, dtype=float)
    max_w = float(w.max()) if w.size else 0.0
    if max_w <= 0:
        return
    alphas = np.clip(w / max_w * 0.6, 0.05, 1.0)
    rgba = np.tile(np.array(mcolors.to_rgba(COLORS["mem"])), (len(atoms), 1))
    rgba[:, 3] = alphas
    ax.vlines(
        atoms,
        0.0,
        1.0,
        transform=ax.get_xaxis_transform(),
        colors=rgba.tolist(),
        linewidth=1.0,
    )


def _analytic_peak(curve: np.ndarray) -> float:
    """99th-percentile of a finite pdf curve.

    The percentile (not the max) keeps a Beta density that spikes at 0/1 from
    blowing up the y-limit, while being ~the peak for a smooth Gaussian.
    """
    finite = curve[np.isfinite(curve)]
    return float(np.percentile(finite, 99)) if finite.size else 0.0


def draw_density_cell(
    ax: Axes,
    *,
    pmc_vals: np.ndarray,
    atoms: np.ndarray | None,
    weights: np.ndarray | None,
    gen_pdf: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
    is_prior: bool = False,
    vline_max_m: int = VLINE_MAX_M,
) -> None:
    """Draw one density cell and set its x/y limits deterministically.

    Args:
        pmc_vals: PMC samples for this object.
        atoms / weights: task-pool atom values and closed-form weights for
            Pi_M. Pass None to omit the memorising layer.
        gen_pdf: callable mapping an x-grid to the Pi_infty density.
        lo, hi: x-range (use `cell_xrange`).
        is_prior: only the *prior* may use vlines; posteriors always use the
            weighted histogram (the alpha-by-weight vlines read poorly when the
            posterior weights vary).
        vline_max_m: prior uses vlines when M <= this, else the histogram. The
            cutoff is domain-specific (LR 16, BAU 8, Markov per caller).
    """
    _validate_cell_inputs(pmc_vals, atoms, weights, lo, hi)
    x_grid = np.linspace(lo, hi, N_GRID)
    bins = np.linspace(lo, hi, N_BINS + 1)

    ax.hist(
        pmc_vals,
        bins=bins.tolist(),
        density=True,
        alpha=0.35,
        color=COLORS["pmc"],
        label=LABELS["pmc"],
        histtype="stepfilled",
    )
    kde = _kde_or_none(pmc_vals, x_grid)
    if kde is not None:
        ax.plot(x_grid, kde, color=COLORS["pmc"], linewidth=1.0)

    if atoms is not None:
        assert weights is not None
        if is_prior and len(atoms) <= vline_max_m:
            draw_atom_vlines(ax, atoms, weights)
        else:
            ax.hist(
                atoms,
                bins=bins.tolist(),
                weights=weights,
                density=True,
                alpha=0.5,
                color=COLORS["mem"],
                label=LABELS["mem"],
                histtype="stepfilled",
            )

    pdf = gen_pdf(x_grid)
    ax.plot(x_grid, pdf, color=COLORS["gen"], label=LABELS["gen"], linewidth=1.0)

    # y-limit shows both the Pi_infty peak and the *full* PMC-KDE peak, so the
    # top of the yellow PMC curve is always visible. Histograms are excluded
    # (the Pi_M weighted-histogram spikes are position-carrying, not heights).
    analytic_max = _analytic_peak(pdf)
    ymax = analytic_max
    if kde is not None:
        ymax = max(ymax, float(np.max(kde)))

    ax.set_xlim(lo, hi)
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.15)


def draw_cdf_cell(
    ax: Axes,
    *,
    pmc_vals: np.ndarray,
    atoms: np.ndarray | None,
    weights: np.ndarray | None,
    gen_cdf: Callable[[np.ndarray], np.ndarray],
    lo: float,
    hi: float,
) -> None:
    """Draw one CDF cell for PMC, finite-pool, and population references."""
    _validate_cell_inputs(pmc_vals, atoms, weights, lo, hi)
    x_grid = np.linspace(lo, hi, N_GRID)

    sorted_pmc = np.sort(pmc_vals)
    emp = np.arange(1, sorted_pmc.size + 1) / sorted_pmc.size
    ax.plot(sorted_pmc, emp, color=COLORS["pmc"], label=LABELS["pmc"], linewidth=1.0)

    if atoms is not None:
        assert weights is not None
        order = np.argsort(atoms)
        sx = np.asarray(atoms)[order]
        sy = np.cumsum(np.asarray(weights)[order])
        # Anchor the step at the panel edges so the discrete CDF renders a visible
        # 0 -> 1 staircase even for a single atom (M=1): start at (lo, 0), rise at
        # each atom, hold the final mass out to (hi, 1). Without the (lo,0)/(hi,*)
        # anchors a one-point step draws nothing.
        xs = np.concatenate(([lo], sx, [hi]))
        ys = np.concatenate(([0.0], sy, [sy[-1]]))
        ax.step(
            xs,
            ys,
            where="post",
            color=COLORS["mem"],
            label=LABELS["mem"],
            linewidth=1.0,
        )

    ax.plot(
        x_grid, gen_cdf(x_grid), color=COLORS["gen"], label=LABELS["gen"], linewidth=1.0
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.0, 1.0)


def legend_handles() -> list[Line2D]:
    """Return legend handles in fixed PMC, finite-pool, population order."""
    return [
        Line2D([0], [0], color=COLORS["pmc"], lw=2, label=LABELS["pmc"]),
        Line2D([0], [0], color=COLORS["mem"], lw=2, label=LABELS["mem"]),
        Line2D([0], [0], color=COLORS["gen"], lw=2, label=LABELS["gen"]),
    ]
