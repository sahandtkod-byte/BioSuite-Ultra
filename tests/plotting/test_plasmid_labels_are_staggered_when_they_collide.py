"""Labels that would sit on top of each other move out to another ring.

Every feature label sat at one radius, so the sample plasmid's multiple cloning
site — five features inside 52 bp, measured angular gaps down to 0.003 rad — drew
its names over each other in a single illegible pile.
"""
import math

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import plasmid_map as pm


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _feature_label_radii(fig):
    """The radii of the feature labels (not the size ruler, not the centre)."""
    return [t.get_position()[1] for t in fig.axes[0].texts
            if t.get_position()[1] > 1.05]


def _feature_labels(fig):
    return {t.get_text(): t.get_position() for t in fig.axes[0].texts
            if t.get_position()[1] > 1.05}


def test_the_sample_plasmid_spreads_its_clustered_labels_over_several_rings():
    fig = pm.draw_plasmid(pm.create_sample_plasmid())

    rings = {round(r, 3) for r in _feature_label_radii(fig)}
    assert len(rings) >= 3, f"the cloning-site labels still share a ring: {rings}"


def test_a_map_with_room_keeps_one_ring():
    """The positive control: staggering is for collisions, not decoration."""
    plasmid = pm.PlasmidMap(name="p", size=1000)
    for i, name in enumerate(["a", "b", "c"]):
        plasmid.add_feature(pm.PlasmidFeature(name, i * 300, i * 300 + 100))

    fig = pm.draw_plasmid(plasmid)

    assert {round(r, 3) for r in _feature_label_radii(fig)} == {1.28}


def test_each_label_stays_at_its_own_feature_angle():
    plasmid = pm.PlasmidMap(name="p", size=1000)
    for i, name in enumerate(["a", "b", "c"]):
        plasmid.add_feature(pm.PlasmidFeature(name, i * 300, i * 300 + 100))

    fig = pm.draw_plasmid(plasmid)

    positions = _feature_labels(fig)
    for i, name in enumerate(["a", "b", "c"]):
        middle = (i * 300 + i * 300 + 100) / 2
        assert positions[name][0] == pytest.approx(middle / 1000 * 2 * math.pi)


def test_the_annotated_drawing_staggers_too():
    fig = pm.draw_plasmid_with_annotations(pm.create_sample_plasmid())

    rings = {round(r, 3) for r in _feature_label_radii(fig)}
    assert len(rings) >= 3, rings


def test_a_label_never_moves_past_the_top_of_the_axes():
    """However many collide, the rings stop where the axes end."""
    plasmid = pm.PlasmidMap(name="p", size=100)
    for i in range(12):
        plasmid.add_feature(pm.PlasmidFeature(f"f{i}", i, i + 2))

    fig = pm.draw_plasmid(plasmid)

    assert max(_feature_label_radii(fig)) <= 1.48, \
        "0.28 + 0.20 is the last ring, under the 1.6 axis top"
