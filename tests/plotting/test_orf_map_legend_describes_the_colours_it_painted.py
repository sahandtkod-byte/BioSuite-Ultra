"""The ORF map's colour bar has to describe the colours the bars were painted with.

The bars are coloured over the middle of viridis (``0.2 + 0.75 * length/max``)
so that short ORFs stay visible, but the colour bar was built from the raw
colormap with ``Normalize(0, max_len)``.  For a 10 aa ORF in a set whose longest
is 120 aa the bar painted (0.224, 0.335, 0.548) and its own legend said a 10 aa
ORF looks like (0.283, 0.121, 0.441): every length read off the legend was wrong,
and the mismatch grew as the ORFs got shorter.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import sequence_viewer as sv

ORFS = [{"frame": 1, "start": 0, "end": 30, "length": 10, "terminated": True},
        {"frame": 2, "start": 40, "end": 400, "length": 120, "terminated": True},
        {"frame": 3, "start": 200, "end": 300, "length": 33, "terminated": True}]
SEQ = "A" * 600


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _legend_mesh(fig):
    return fig.axes[1].collections[-1]


def test_a_bar_and_the_legend_now_agree_on_every_length_drawn():
    fig = sv.draw_orf_map(SEQ, orfs=ORFS)

    mesh = _legend_mesh(fig)
    for patch, orf in zip(fig.axes[0].patches, ORFS):
        painted = np.round(patch.get_facecolor()[:3], 3)
        claimed = np.round(mesh.cmap(orf["length"] / mesh.norm.vmax)[:3], 3)
        assert painted == pytest.approx(claimed, abs=0.01), \
            f"{orf['length']} aa: bar {painted} != legend {claimed}"


def test_the_legend_span_is_the_length_span_it_labels():
    fig = sv.draw_orf_map(SEQ, orfs=ORFS)

    mesh = _legend_mesh(fig)
    assert (mesh.norm.vmin, mesh.norm.vmax) == (0.0, 120.0)
    assert fig.axes[1].yaxis.get_label().get_text() == "ORF length (aa)"


def test_the_legend_colours_are_a_ramp_the_short_orfs_can_survive():
    """The squeeze is kept on purpose: the darkest fifth of viridis is unreadable."""
    fig = sv.draw_orf_map(SEQ, orfs=ORFS)

    mesh = _legend_mesh(fig)
    shortest = np.round(mesh.cmap(10 / 120)[:3], 3)
    raw_viridis = np.round(matplotlib.colormaps["viridis"](10 / 120)[:3], 3)
    assert shortest != pytest.approx(raw_viridis, abs=0.05), \
        f"the ramp stopped being the squeezed one: {shortest} vs {raw_viridis}"
    assert shortest.sum() > raw_viridis.sum(), "the squeezed colour is the lighter one"


def test_a_set_of_zero_length_orfs_still_gets_a_finite_legend():
    orfs = [{"frame": 1, "start": 0, "end": 10, "length": 0}]

    fig = sv.draw_orf_map(SEQ, orfs=orfs)

    assert _legend_mesh(fig).norm.vmax == 1
    assert len(fig.axes[0].patches) == 1


def test_an_empty_autodetected_set_is_reported_on_the_map():
    fig = sv.draw_orf_map("AAAAAA")

    assert "no ORF of 10 aa or more in 6 bp" in fig.axes[0].get_title(), \
        fig.axes[0].get_title()
