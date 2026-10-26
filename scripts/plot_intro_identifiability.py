"""Intro identifiability figure: Gaussian vs spike-and-slab posterior.

Generates `intro_identifiability_spike_slab.png` (main-text Figure 2): 2-D
Bayesian linear regression with a single observation (x1, y1) = ([1, 1], 1).
The Gaussian and spike-and-slab posteriors have identical means by
construction but different shapes (left, middle): the slab variance is set so
that the spike-and-slab posterior mean equals the Gaussian one. At the
orthogonal query x' = [1, -1] the predictive distributions are unimodal vs
bimodal (right), yet the posterior-mean predictions coincide. The same cyan
cross marks the shared posterior mean (left, middle) and the shared point
prediction (right).

Both priors are Gaussian mixtures, so the posteriors, their means and the
predictive densities are computed in closed form; only the heat maps are
evaluated on a grid.

Ported from notes/laplace_gaussian_identifiability.ipynb in the project
folder so the figure has a committed generator.

Usage:
    uv run python scripts/plot_intro_identifiability.py \
        --out-path outputs/paper_figures/misc/intro_identifiability_spike_slab.png
"""

import argparse
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from plotting.paper_style import apply_paper_style

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# Observation and model
X_OBS = np.array([1.0, 1.0])
Y_OBS = 1.0
SIGMA2 = 0.05
# Orthogonal query of the right panel
X_QUERY = np.array([1.0, -1.0])

# Priors. The slab variance is not fixed here: calibrated_slab_variance() sets
# it so that both posteriors share one mean.
TAU2 = 1.0          # Gaussian prior variance
PI_SPIKE = 0.5      # spike probability per coordinate
EPS2 = 0.03 ** 2    # spike variance

GRID_LIM = 2.0
G = 601

# The shared-mean and shared-prediction marker, identical in all three panels.
MARKER = dict(c="cyan", s=90, marker="x", lw=2.0, zorder=10)

# One Gaussian component of a posterior: weight, mean, covariance.
Component = tuple[float, np.ndarray, np.ndarray]


def conjugate_update(prior_cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Posterior mean and covariance, and log evidence, under w ~ N(0, prior_cov)."""
    evidence_var = X_OBS @ prior_cov @ X_OBS + SIGMA2
    gain = prior_cov @ X_OBS / evidence_var
    mean = gain * Y_OBS
    cov = prior_cov - np.outer(gain, prior_cov @ X_OBS)
    log_evidence = -0.5 * np.log(2 * np.pi * evidence_var) - 0.5 * Y_OBS**2 / evidence_var
    return mean, cov, log_evidence


def gaussian_posterior() -> list[Component]:
    mean, cov, _ = conjugate_update(TAU2 * np.eye(2))
    return [(1.0, mean, cov)]


def spike_slab_posterior(slab_var: float) -> list[Component]:
    """One Gaussian per spike/slab assignment of the two coordinates."""
    per_coordinate = [(PI_SPIKE, EPS2), (1 - PI_SPIKE, slab_var)]
    parts = []
    for p1, v1 in per_coordinate:
        for p2, v2 in per_coordinate:
            mean, cov, log_evidence = conjugate_update(np.diag([v1, v2]))
            parts.append((np.log(p1 * p2) + log_evidence, mean, cov))
    log_weights = np.array([part[0] for part in parts])
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    return [(float(w), mean, cov) for w, (_, mean, cov) in zip(weights, parts)]


def posterior_mean(components: list[Component]) -> np.ndarray:
    return sum(w * mean for w, mean, _ in components)


def calibrated_slab_variance() -> float:
    """Slab variance at which the spike-and-slab posterior mean equals the Gaussian one.

    Both priors and the observation are symmetric under swapping w1 and w2, so
    both means lie on the diagonal and matching one coordinate matches both.
    """
    target = posterior_mean(gaussian_posterior())[0]
    return brentq(
        lambda v: posterior_mean(spike_slab_posterior(v))[0] - target,
        1.0,
        20.0,
        xtol=1e-14,
    )


def predictive_density(components: list[Component], y_grid: np.ndarray) -> np.ndarray:
    """Density of y' = x'^T w + noise at X_QUERY: a Gaussian mixture."""
    return sum(
        w * norm.pdf(y_grid, loc=X_QUERY @ mean, scale=np.sqrt(X_QUERY @ cov @ X_QUERY + SIGMA2))
        for w, mean, cov in components
    )


def _log_prior_ss_1d(x: np.ndarray, slab_var: float) -> np.ndarray:
    lp_spike = np.log(PI_SPIKE) - 0.5 * np.log(2 * np.pi * EPS2) - 0.5 * x**2 / EPS2
    lp_slab = (
        np.log(1 - PI_SPIKE) - 0.5 * np.log(2 * np.pi * slab_var) - 0.5 * x**2 / slab_var
    )
    m = np.maximum(lp_spike, lp_slab)
    return m + np.log(np.exp(lp_spike - m) + np.exp(lp_slab - m))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-path", type=_Path, default=_Path("outputs/intro_identifiability_spike_slab.png")
    )
    args = parser.parse_args()

    slab_var = calibrated_slab_variance()
    gaussian = gaussian_posterior()
    spike_slab = spike_slab_posterior(slab_var)
    mean_gauss = posterior_mean(gaussian)
    mean_ss = posterior_mean(spike_slab)
    if not np.allclose(mean_gauss, mean_ss, rtol=0.0, atol=1e-10):
        raise AssertionError(f"posterior means differ: {mean_gauss} vs {mean_ss}")
    prediction_gauss = float(X_QUERY @ mean_gauss)
    prediction_ss = float(X_QUERY @ mean_ss)
    print(
        f"slab variance {slab_var:.6f}; shared posterior mean {mean_gauss}; "
        f"point prediction at x'={X_QUERY}: {prediction_gauss:.3g} (Gaussian), "
        f"{prediction_ss:.3g} (spike-and-slab)"
    )

    w_grid = np.linspace(-GRID_LIM, GRID_LIM, G)
    W1, W2 = np.meshgrid(w_grid, w_grid, indexing="xy")
    log_lik = -0.5 * (Y_OBS - (W1 + W2)) ** 2 / SIGMA2
    log_post_gauss = log_lik - 0.5 * (W1**2 + W2**2) / TAU2
    log_post_ss = log_lik + _log_prior_ss_1d(W1, slab_var) + _log_prior_ss_1d(W2, slab_var)
    post_gauss = np.exp(log_post_gauss - log_post_gauss.max())
    post_ss = np.exp(log_post_ss - log_post_ss.max())

    y_grid = np.linspace(-3, 3, 801)
    pred_gauss = predictive_density(gaussian, y_grid)
    pred_ss = predictive_density(spike_slab, y_grid)

    fig_w = 9.0
    apply_paper_style(fig_w, 1.0)
    fig, axes = plt.subplots(
        1, 3, figsize=(fig_w, 3.1), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1.3]},
    )

    for ax, post, mean, title in [
        (axes[0], post_gauss, mean_gauss, "Gaussian prior"),
        (axes[1], post_ss, mean_ss, "Spike-and-slab prior"),
    ]:
        ax.pcolormesh(W1, W2, post, cmap="magma", rasterized=True, shading="auto")
        ax.scatter([mean[0]], [mean[1]], **MARKER)
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(-0.7, 1.7)
        ax.set_aspect("equal")
        ax.set_xlabel("$w_1$")
        ax.set_ylabel("$w_2$")
        ax.set_title(title)

    ax = axes[2]
    ax.plot(y_grid, pred_gauss, lw=1.5, color="tab:blue", label="Gaussian prior")
    ax.plot(y_grid, pred_ss, lw=1.5, color="tab:red", label="Spike-and-slab prior")
    # Both point predictions, drawn on the y' axis: they coincide, so the two
    # crosses land on the same spot. The guide line stops at the taller density
    # there, below the legend.
    guide_top = max(np.interp(prediction_gauss, y_grid, pred_gauss), np.interp(prediction_ss, y_grid, pred_ss))
    ax.vlines(prediction_gauss, 0.0, guide_top, colors="cyan", linestyles="--", lw=1.0, zorder=1)
    ax.scatter(
        [prediction_gauss], [0.0], clip_on=False,
        label="Posterior-mean prediction (both priors)", **MARKER,
    )
    ax.scatter([prediction_ss], [0.0], clip_on=False, **MARKER)
    # Headroom above the peaks keeps the legend off the curves.
    ax.set_ylim(0.0, 1.45 * max(pred_gauss.max(), pred_ss.max()))
    ax.set_xlabel(r"$y'$ at query $x' = [1, -1]$")
    ax.set_ylabel(r"$p(y' \mid x',\, (x_1, y_1))$")
    ax.set_title("Predictive at orthogonal query")
    ax.legend(frameon=False, loc="upper left", fontsize="x-small")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
