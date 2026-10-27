from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from bft_pmc.plotting.paper_style import (
    DecadeLocator,
    OddPowerOfTwoLocator,
    every_decade_ticks,
    task_diversity_ticks,
)


def _powers(values: list[float]) -> list[int]:
    return [int(round(math.log2(value))) for value in values]


def test_locator_matches_the_published_tick_pattern() -> None:
    """A sweep ending at 2^11 ticks every other power, as the paper does."""
    assert _powers(OddPowerOfTwoLocator(6).tick_values(4, 2048)) == [3, 5, 7, 9, 11]
    assert _powers(OddPowerOfTwoLocator(6).tick_values(1, 4096)) == [1, 3, 5, 7, 9, 11]


def test_locator_thins_wide_sweeps_instead_of_colliding() -> None:
    """Linear regression spans 2^0-2^16; eight labels do not fit a half panel."""
    ticks = OddPowerOfTwoLocator(6).tick_values(1, 65536)
    assert _powers(ticks) == [1, 5, 9, 13]
    assert len(ticks) <= 6
    # a tighter budget thins further, and never below one tick
    assert len(OddPowerOfTwoLocator(2).tick_values(1, 65536)) <= 2


def test_locator_rejects_ranges_a_log_axis_cannot_show() -> None:
    locator = OddPowerOfTwoLocator()
    assert locator.tick_values(0, 1024) == []
    assert locator.tick_values(-4, 1024) == []
    assert locator.tick_values(float("nan"), 1024) == []
    assert locator.tick_values(1, float("inf")) == []
    # reversed limits are read as a range, not an error
    assert _powers(locator.tick_values(2048, 4)) == [3, 5, 7, 9, 11]


def test_task_diversity_ticks_drives_a_live_axis() -> None:
    figure, axes = plt.subplots()
    axes.set_xscale("log", base=2)
    axes.set_xlim(4, 2048)
    task_diversity_ticks(axes.xaxis)
    assert _powers(list(axes.get_xticks())) == [3, 5, 7, 9, 11]
    plt.close(figure)


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ((1.0, 1000.0), [1.0, 10.0, 100.0, 1000.0]),  # within budget: every decade
        ((1.0, 1e5), [1.0, 100.0, 10000.0]),  # six decades: every second
        ((1e5, 1.0), [1.0, 100.0, 10000.0]),  # reversed limits
        ((0.0, 10.0), []),  # a log axis cannot show zero
        ((1.0, float("inf")), []),
    ],
)
def test_decade_locator_thins_decades_to_the_tick_budget(
    limits: tuple[float, float], expected: list[float]
) -> None:
    assert DecadeLocator(max_ticks=4).tick_values(*limits) == expected


def test_every_decade_ticks_caps_the_label_count_when_asked() -> None:
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1e5)
    every_decade_ticks(ax.xaxis, max_ticks=4)
    assert isinstance(ax.xaxis.get_major_locator(), DecadeLocator)
    assert list(ax.get_xticks()) == [1.0, 100.0, 10000.0]
    every_decade_ticks(ax.xaxis)
    assert not isinstance(ax.xaxis.get_major_locator(), DecadeLocator)
    plt.close(fig)
