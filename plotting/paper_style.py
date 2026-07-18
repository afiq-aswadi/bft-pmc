"""Shared matplotlib style for all figures that appear in the paper.

Every plotting script that produces a paper figure imports this module and
calls ``apply_paper_style(fig_width_in, print_frac)`` before creating its
figure. Font sizes are scaled by the ratio of the figure's canvas width to
its printed width, so that text renders at the target point sizes below
once the figure is scaled into the paper. This is the single source of
truth for figure typography; do not hardcode font sizes in individual
plotters.

``print_frac`` is the fraction of the paper's text width the figure
occupies in the LaTeX source (e.g. 0.49 for a half-width subfigure,
0.95 for a near-full-width appendix figure).
"""

from __future__ import annotations

import matplotlib

# NeurIPS/arXiv body text width in inches.
TEXT_WIDTH_IN = 5.5

# Target sizes at print scale, in points.
BASE_FONT_PT = 7.5    # axis labels, titles, row annotations
TICK_FONT_PT = 6.5    # tick labels
LEGEND_FONT_PT = 7.0  # legends


def font_scale(fig_width_in: float, print_frac: float = 1.0) -> float:
    """Ratio between canvas size and printed size."""
    return fig_width_in / (TEXT_WIDTH_IN * print_frac)


def apply_paper_style(fig_width_in: float, print_frac: float = 1.0) -> float:
    """Set rcParams so text prints at the target sizes; returns the scale.

    Call before creating the figure. The returned scale can be used for
    any remaining manual ``fontsize=`` arguments (multiply a target print
    size by it).
    """
    s = font_scale(fig_width_in, print_frac)
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": BASE_FONT_PT * s,
            "axes.labelsize": BASE_FONT_PT * s,
            "axes.titlesize": BASE_FONT_PT * s,
            "xtick.labelsize": TICK_FONT_PT * s,
            "ytick.labelsize": TICK_FONT_PT * s,
            "legend.fontsize": LEGEND_FONT_PT * s,
            "savefig.dpi": 300,
        }
    )
    return s
