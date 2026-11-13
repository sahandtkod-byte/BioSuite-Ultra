"""A plasmid map needs a size to have a scale, and a polar axes to be circular.

A map with no size was drawn as if it were 1000 bp — and the centre label printed
that invented number.  A Cartesian axes accepted the polar data and drew the whole
map on a straight line from (0, 0) to (2π, 1).  Both drawers also called
``fig.tight_layout()`` on a figure they were handed, re-flowing the caller's other
parcels.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import plasmid_map as pm

DRAWERS = [pm.draw_plasmid, pm.draw_plasmid_with_annotations]
IDS = [fn.__name__ for fn in DRAWERS]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("draw", DRAWERS, ids=IDS)
def test_a_map_with_no_size_is_refused_instead_of_assuming_one(draw):
    with pytest.raises(ValueError, match="has no size"):
        draw(pm.PlasmidMap(name="empty"))


@pytest.mark.parametrize("draw", DRAWERS, ids=IDS)
def test_a_cartesian_axes_is_refused_by_name(draw):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    with pytest.raises(ValueError, match="needs a polar axes"):
        draw(pm.PlasmidMap(name="p", size=100), ax=ax)


@pytest.mark.parametrize("draw", DRAWERS, ids=IDS)
def test_a_polar_axes_is_accepted(draw):
    """The positive control for the refusal above."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="polar")

    drawn = draw(pm.PlasmidMap(name="p", size=100), ax=ax)

    assert drawn is fig
    assert len(fig.axes) == 1


@pytest.mark.parametrize("draw", DRAWERS, ids=IDS)
def test_a_borrowed_figure_keeps_its_layout(draw):
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 2)
    left = fig.add_subplot(gs[0])
    right = fig.add_subplot(gs[1], projection="polar")
    before = tuple(left.get_position().bounds)

    draw(pm.PlasmidMap(name="p", size=100), ax=right)

    assert tuple(left.get_position().bounds) == before, \
        "tight_layout re-flowed a figure the drawer does not own"


def test_the_centre_label_quotes_the_size_it_was_given():
    fig = pm.draw_plasmid(pm.PlasmidMap(name="p", size=2500))

    assert [t.get_text() for t in fig.axes[0].texts][-1] == "p\n2500 bp"
