"""The plasmid data model and the two arcs it can be drawn as.

``PlasmidMap`` accepts either a sequence or a bare size, ``add_feature`` takes an
object or keyword arguments, and the setters hand the map back so a caller can chain
them.  Both drawers wrap a feature that crosses the origin, so its arc runs past
2π instead of collapsing to nothing, and both return the figure the caller already
owns when an axes is passed in.
"""
import math

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting.plasmid_map import (
    PlasmidFeature,
    PlasmidMap,
    _bp_to_angle,
    create_sample_plasmid,
    draw_plasmid,
    draw_plasmid_with_annotations,
    format_plasmid_report,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _polar_axes():
    return plt.subplots(subplot_kw={"projection": "polar"})


# ─── the feature ─────────────────────────────────────────────────────────────

def test_a_feature_picks_its_colour_from_its_type():
    assert PlasmidFeature("ampR", 100, 900, feature_type="antibiotic_resistance").color == "#9C27B0"
    assert PlasmidFeature("odd", 1, 2, feature_type="replication_origin").color == "#607D8B", \
        "a type outside the palette falls back to the colour 'other' uses"


def test_a_feature_colours_itself_when_the_caller_says_so():
    assert PlasmidFeature("x", 1, 2, color="#010203").color == "#010203"


def test_a_feature_is_named_after_itself_unless_it_is_labelled():
    assert PlasmidFeature("lacZ", 1, 2).label == "lacZ"
    assert PlasmidFeature("lacZ", 1, 2, label="lacZ α").label == "lacZ α"


# ─── the map ─────────────────────────────────────────────────────────────────

def test_a_sequence_given_map_takes_its_length_and_uppercases_it():
    plasmid = PlasmidMap(name="pX", sequence="atgcATGC")

    assert plasmid.size == 8
    assert plasmid.sequence == "ATGCATGC"


def test_a_size_given_map_invents_a_sequence_of_that_length():
    plasmid = PlasmidMap(name="pX", size=40)

    assert plasmid.size == 40
    assert set(plasmid.sequence) <= set("ATCG"), "the filler is made of DNA letters"


def test_an_empty_map_holds_no_sequence_and_no_features():
    plasmid = PlasmidMap(name="pX")

    assert (plasmid.size, plasmid.sequence, plasmid.features) == (0, "", [])


def test_a_feature_arrives_as_an_object_or_as_keyword_arguments():
    plasmid = PlasmidMap(name="pX", size=100)

    plasmid.add_feature(PlasmidFeature("ori", 1, 50, feature_type="origin"))
    plasmid.add_feature("lacZ", start=51, end=99, feature_type="CDS")

    assert [f.name for f in plasmid.features] == ["ori", "lacZ"]
    assert [f.color for f in plasmid.features] == ["#FF9800", "#2196F3"]


def test_the_setters_hand_the_map_back_so_they_chain():
    plasmid = PlasmidMap(name="pX", size=10)

    returned = plasmid.set_name("pY").set_description("a vector").set_size(20)

    assert returned is plasmid
    assert (plasmid.name, plasmid.description, plasmid.size) == ("pY", "a vector", 20)


# ─── the two drawings ────────────────────────────────────────────────────────

def test_a_feature_crossing_the_origin_is_drawn_as_one_wrapping_arc():
    plasmid = PlasmidMap(name="pWrap", size=3000)
    plasmid.add_feature(PlasmidFeature("ori", 2800, 200, feature_type="origin"))

    fig = draw_plasmid(plasmid)

    angles = fig.axes[0].collections[0].get_paths()[0].vertices[:, 0]
    assert angles.min() == pytest.approx(2800 / 3000 * 2 * math.pi)
    assert angles.max() > 2 * math.pi, "the arc runs past the top instead of collapsing"
    assert "ori" in [t.get_text() for t in fig.axes[0].texts]


def test_the_annotated_drawing_wraps_a_feature_the_same_way():
    plasmid = PlasmidMap(name="pWrap", size=3000)
    plasmid.add_feature(PlasmidFeature("ori", 2800, 200, feature_type="origin"))

    fig = draw_plasmid_with_annotations(plasmid)

    angles = fig.axes[0].collections[0].get_paths()[0].vertices[:, 0]
    assert angles.max() > 2 * math.pi
    assert "ori (2800-200)" in [t.get_text() for t in fig.axes[0].texts], \
        "the annotated map names the feature and its span"


def test_the_plain_map_draws_on_the_axes_the_caller_owns():
    outer, ax = _polar_axes()

    fig = draw_plasmid(PlasmidMap(name="pX", size=10), ax=ax)

    assert fig is outer
    assert len(outer.axes) == 1, "no second axes appears behind the caller's back"


def test_the_annotated_map_draws_on_the_axes_the_caller_owns():
    outer, ax = _polar_axes()

    fig = draw_plasmid_with_annotations(PlasmidMap(name="pX", size=10), ax=ax)

    assert fig is outer
    assert len(outer.axes) == 1


def test_an_unannotated_map_still_draws_its_backbone():
    # The size is given here: an unsized map used to be drawn as if it were 1000 bp
    # and print that invented number in the middle of the figure, which is now a
    # refusal (test_an_unsized_plasmid_is_refused_rather_than_invented).
    fig = draw_plasmid_with_annotations(PlasmidMap(name="empty", size=1000))

    assert len(fig.axes[0].collections) == 0, "there is no feature to fill"
    backbone, *markers = fig.axes[0].lines
    assert len(backbone.get_xdata()) == 500, "the backbone is a circle of 500 points"
    assert len(markers) == 2, "one tick every 500 bp of the 1000 bp it was given"
    assert fig.axes[0].get_legend() is None, "a legend of nothing is no legend"
    assert [t.get_text() for t in fig.axes[0].texts] == ["0", "500", "empty\n1000 bp"], \
        "the ruler it drew, then the name over the middle"


# ─── pins moved here from the stacked plasmid/browser/conservation file ──────

def test_a_position_maps_to_the_angle_the_circle_gives_it():
    assert _bp_to_angle(1343, 2686) == pytest.approx(math.pi)


def test_the_sample_roundtrips_through_both_drawers():
    sample = create_sample_plasmid()

    assert len(sample.features) == 10
    assert draw_plasmid(sample) is not None
    assert draw_plasmid_with_annotations(sample) is not None


def test_arrows_use_polar_data_coordinates():
    """Regression: old arrows passed Cartesian (r*cos, r*sin) as (theta, r)."""
    plasmid = PlasmidMap(name="t", size=1000, sequence="A" * 1000)
    plasmid.add_feature(PlasmidFeature("f1", 100, 400))

    fig = draw_plasmid(plasmid)
    ax = fig.axes[0]
    arrows = [a for a in ax.texts if getattr(a, "arrow_patch", None) is not None]

    assert arrows, "no arrows drawn"
    for arrow in arrows:
        theta, radius = arrow.get_position()
        assert 0 <= theta <= 2 * math.pi + 1e-6
        assert 0.9 <= radius <= 1.1


def test_the_sample_report_names_its_size_and_its_ampicillin_gene():
    report = format_plasmid_report(create_sample_plasmid())

    assert "AmpR" in report and "2686" in report


def test_a_wrapped_span_needs_the_map_size_to_wrap_around():
    feature = PlasmidFeature("wrap", 600, 300)

    assert feature.span(1000) == 700
    assert feature.span() == -300, "with no map there is no circle to wrap around"
