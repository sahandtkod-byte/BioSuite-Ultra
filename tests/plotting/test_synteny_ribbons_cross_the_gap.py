"""The ribbons on a synteny map connect the two gene strips.

Each line must start at the centre of its gene block on the top strip's lower
edge and end at the centre on the bottom strip's upper edge.  The axes use
different x-scales, so reading a gene index as an axes fraction (the old bug)
put the line at several times the figure width.
"""
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting.synteny import plot_synteny

FORWARD = ["A", "B", "C", "D"]
FLIPPED = ["D", "C", "B", "A"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _display_endpoints(link):
    path = link.arrow_patch.get_path().transformed(link.arrow_patch.get_transform())
    return path.vertices[0], path.vertices[-1]


def test_every_ribbon_joins_its_two_gene_centres():
    fig = plot_synteny(FORWARD, FLIPPED)
    fig.canvas.draw()
    ax1, ax2 = fig.axes
    links = [text for text in ax1.texts if not text.get_text()]

    assert len(links) == len(FORWARD)
    for link in links:
        start, end = _display_endpoints(link)
        # The annotation starts at the top strip and points at the bottom one;
        # its stored coordinates name the gene, not the axes fraction.
        i1 = round(link.xyann[0] - 0.5)
        i2 = round(link.xy[0] - 0.5)
        expected_start = ax1.transData.transform((i1 + 0.5, 0))
        expected_end = ax2.transData.transform((i2 + 0.5, 1))
        assert np.hypot(*(start - expected_start)) < 5.0, (
            f"ribbon for gene {i1} starts {start} not {expected_start}")
        assert np.hypot(*(end - expected_end)) < 5.0, (
            f"ribbon for gene {i2} ends {end} not {expected_end}")


def test_every_ribbon_stays_inside_the_figure():
    fig = plot_synteny(FORWARD, FLIPPED)
    fig.canvas.draw()
    width, height = fig.bbox.width, fig.bbox.height
    links = [text for text in fig.axes[0].texts if not text.get_text()]

    for link in links:
        for x, y in _display_endpoints(link):
            assert 0 <= x <= width, f"ribbon x={x} is outside a {width}px figure"
            assert 0 <= y <= height, f"ribbon y={y} is outside a {height}px figure"


def test_the_synteny_map_does_not_ask_tight_layout_to_redo_its_spacing():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = plot_synteny(FORWARD, FLIPPED)
        fig.canvas.draw()

    tight = [w for w in caught if "tight_layout" in str(w.message)]
    assert tight == [], f"the map warned about its own layout: {tight}"
