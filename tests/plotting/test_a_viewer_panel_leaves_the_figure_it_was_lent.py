"""A panel that is handed someone else's axes must not re-lay-out their figure.

Each drawing function ended with ``fig.tight_layout()`` — including when the
figure belonged to the dashboard, or to a caller with a gridspec of its own.  The
borrowed figure's other panels were resized around whatever had just been drawn,
and matplotlib said so: ``This figure includes Axes that are not compatible with
tight_layout``, once per embedded panel, every render.
"""
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import sequence_viewer as sv

SEQ = "ATG" + "GGG" * 100 + "TAA"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("draw", [
    sv.draw_sequence_view, sv.draw_gc_content_plot, sv.draw_orf_map,
])
def test_a_neighbour_keeps_the_box_it_had(draw):
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 2)
    left = fig.add_subplot(gs[0])
    left.set_title("sibling")
    right = fig.add_subplot(gs[1])
    before = tuple(left.get_position().bounds)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        draw(SEQ, ax=right)

    assert tuple(left.get_position().bounds) == before, \
        f"{draw.__name__} moved a panel it was not given"
    assert not [w for w in caught if "tight_layout" in str(w.message)], \
        f"{draw.__name__} warned about the figure it was lent"


def test_the_translation_panel_is_the_same_case():
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 2)
    left = fig.add_subplot(gs[0])
    right = fig.add_subplot(gs[1])
    before = tuple(left.get_position().bounds)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sv.draw_translation_view(SEQ, frame=1, ax=right)

    assert tuple(left.get_position().bounds) == before
    assert not [w for w in caught if "tight_layout" in str(w.message)]


def test_the_dashboard_that_owns_its_figure_lays_it_out_once_and_quietly():
    """The other half of the rule: the owner still gets all four panels."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = sv.create_sequence_overview("ATGC" * 120)

    assert len(fig.axes) == 6, [a.get_title() for a in fig.axes]
    assert not [w for w in caught if "tight_layout" in str(w.message)], \
        [str(w.message) for w in caught]


def test_the_panel_that_is_given_a_slot_keeps_its_neighbours_alive():
    """An earlier regression, still true: the slot is used, the figure is not wiped."""
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)
    keep = fig.add_subplot(gs[0, 0])
    keep.set_title("keeper", loc="left", fontsize=6)
    fig.add_subplot(gs[0, 1])
    fig.add_subplot(gs[1, 0])
    slot = fig.add_subplot(gs[1, 1])

    sv.draw_translation_view("ATGCCCGCTAAATG", frame=1, ax=slot)

    assert len(fig.axes) == 5, "two replacement axes in the slot, three untouched"
    assert any("keeper" in ax.get_title(loc="left") for ax in fig.axes)


def test_the_sequence_view_does_not_delete_the_figure_it_was_invited_to():
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 2)
    sentinel = fig.add_subplot(gs[0])
    sentinel.plot([0, 1], [0, 1])
    slot = fig.add_subplot(gs[1])

    sv.draw_sequence_view("ATGC" * 40, ax=slot)

    assert sentinel in fig.axes, "the other panel was removed to make room"
    assert len(fig.axes) == 9, "sentinel plus the eight tracks of the view"


def test_the_panels_still_do_not_overlap_each_other():
    """The positive control: dropping tight_layout did not let the panels touch."""
    fig = sv.create_sequence_overview("ATGC" * 120)

    boxes = [ax.get_position().bounds for ax in fig.axes]
    for i, (ax0, ay0, aw, ah) in enumerate(boxes):
        for (bx0, by0, bw, bh) in boxes[i + 1:]:
            touches = ax0 < bx0 + bw and bx0 < ax0 + aw and ay0 < by0 + bh \
                and by0 < ay0 + ah
            assert not touches, f"axes {i} overlaps a later panel: {boxes}"
