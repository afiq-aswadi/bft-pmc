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

import math

import matplotlib
from matplotlib.ticker import Locator, LogLocator

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


class OddPowerOfTwoLocator(Locator):
    """Ticks at odd powers of two, thinned to a budget over the visible range.

    The published sweep figures tick 2^3, 2^5, ... 2^11. Neither matplotlib
    locator produces that: the base-2 LogLocator aligns strides to even powers,
    and a base-4 locator offset to odd powers returns nothing at small tick
    budgets. Both also thin against the font scale, which at paper sizes
    collapses the axis to two labels.

    Thinning here is against the *visible* power range, so a sweep ending at
    2^11 keeps every other power as published, while linear regression's 2^0
    to 2^16 steps further instead of colliding.
    """

    def __init__(self, max_ticks: int = 6) -> None:
        self.max_ticks = max_ticks

    def __call__(self) -> list[float]:
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin: float, vmax: float) -> list[float]:
        if vmin <= 0 or vmax <= 0 or not math.isfinite(vmin) or not math.isfinite(vmax):
            return []
        low, high = sorted((vmin, vmax))
        powers = [
            power
            for power in range(
                math.floor(math.log2(low)), math.ceil(math.log2(high)) + 1
            )
            if power % 2
        ]
        stride = 1
        while len(powers[::stride]) > self.max_ticks:
            stride += 1
        return [float(2**power) for power in powers[::stride]]


def task_diversity_ticks(axis: object, max_ticks: int = 6) -> None:
    """Put a base-2 task-diversity axis on odd powers of two."""
    axis.set_major_locator(OddPowerOfTwoLocator(max_ticks))


def every_decade_ticks(axis: object) -> None:
    """Label every power of ten on a log axis.

    Matplotlib thins log ticks as soon as labels would collide, and the paper
    font scaling is large enough relative to the panels that it drops every
    second decade (10^1, 10^3, 10^5 instead of 10^0..10^5). Pinning the locator
    keeps the axis matching the published figures at any font size.
    """
    axis.set_major_locator(LogLocator(base=10.0, numticks=99))
