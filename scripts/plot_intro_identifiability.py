"""Intro identifiability figure: Gaussian vs spike-and-slab posterior.

Generates `intro_identifiability_spike_slab.png` (main-text Figure 2): 2-D
Bayesian linear regression with a single observation (x1, y1) = ([1, 1], 1).
The Gaussian and spike-and-slab priors yield posteriors with the same mean
but different shapes (left, middle), and at the orthogonal query
x' = [1, -1] their predictive distributions are unimodal vs bimodal (right).

Ported from notes/laplace_gaussian_identifiability.ipynb in the project
folder so the figure has a committed generator.

Usage:
    uv run python scripts/plot_intro_identifiability.py \
        --out-path ../arxiv-paper/figures/intro_identifiability_spike_slab.png
"""

import argparse
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from plotting.paper_style import apply_paper_style

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Observation and model
X_OBS = np.array([1.0, 1.0])
Y_OBS = 1.0
SIGMA2 = 0.05

# Priors
TAU2 = 1.0          # Gaussian prior variance
PI_SPIKE = 0.5      # spike probability per coordinate
EPS2 = 0.03 ** 2    # spike variance
SIGMA_W2 = 1.0      # slab variance

GRID_LIM = 2.0
G = 601


def _log_prior_ss_1d(x: np.ndarray) -> np.ndarray:
    lp_spike = np.log(PI_SPIKE) - 0.5 * np.log(2 * np.pi * EPS2) - 0.5 * x**2 / EPS2
    lp_slab = (
        np.log(1 - PI_SPIKE) - 0.5 * np.log(2 * np.pi * SIGMA_W2) - 0.5 * x**2 / SIGMA_W2
    )
    m = np.maximum(lp_spike, lp_slab)
    return m + np.log(np.exp(lp_spike - m) + np.exp(lp_slab - m))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-path", type=_Path, default=_Path("outputs/intro_identifiability_spike_slab.png")
    )
    args = parser.parse_args()

    w_grid = np.linspace(-GRID_LIM, GRID_LIM, G)
    W1, W2 = np.meshgrid(w_grid, w_grid, indexing="xy")
    cell_area = (w_grid[1] - w_grid[0]) ** 2

    resid = Y_OBS - (W1 + W2)
    log_lik = -0.5 * resid**2 / SIGMA2

    log_post_gauss = log_lik - 0.5 * (W1**2 + W2**2) / TAU2
    post_gauss = np.exp(log_post_gauss - log_post_gauss.max())
    post_gauss /= post_gauss.sum() * cell_area

    log_post_ss = log_lik + _log_prior_ss_1d(W1) + _log_prior_ss_1d(W2)
    post_ss = np.exp(log_post_ss - log_post_ss.max())
    post_ss /= post_ss.sum() * cell_area

    mu_gauss = np.array([(W1 * post_gauss).sum(), (W2 * post_gauss).sum()]) * cell_area

    # Predictive densities at the orthogonal query
    x_star = np.array([1.0, -1.0])
    y_grid = np.linspace(-3, 3, 801)
    mean_at_w = W1 * x_star[0] + W2 * x_star[1]
    norm_pdf = norm.pdf(y_grid[None, None, :], loc=mean_at_w[:, :, None], scale=np.sqrt(SIGMA2))
    pred_gauss = (norm_pdf * post_gauss[:, :, None]).sum(axis=(0, 1)) * cell_area
    pred_ss = (norm_pdf * post_ss[:, :, None]).sum(axis=(0, 1)) * cell_area

    fig_w = 9.0
    apply_paper_style(fig_w, 1.0)
    fig, axes = plt.subplots(
        1, 3, figsize=(fig_w, 3.1), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1.3]},
    )

    for ax, post, title in [
        (axes[0], post_gauss, "Gaussian prior"),
        (axes[1], post_ss, "Spike-and-slab prior"),
    ]:
        ax.pcolormesh(W1, W2, post, cmap="magma", rasterized=True, shading="auto")
        ax.scatter(
            [mu_gauss[0]], [mu_gauss[1]], c="cyan", s=90, marker="x", lw=2.0, zorder=10
        )
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(-0.7, 1.7)
        ax.set_aspect("equal")
        ax.set_xlabel("$w_1$")
        ax.set_ylabel("$w_2$")
        ax.set_title(title)

    ax = axes[2]
    ax.plot(y_grid, pred_gauss, lw=1.5, color="tab:blue", label="Gaussian prior")
    ax.plot(y_grid, pred_ss, lw=1.5, color="tab:red", label="Spike-and-slab prior")
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
