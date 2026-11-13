"""A feature is checked when it is built, not when it is drawn.

``SequenceAnnotation`` used to be a bare dataclass and the dict wrapper used to
feed it defaults, so three things happened silently: a backwards span became a
box with negative width, a dict without ``end`` became no box at all, and the
text coordinates a parsed feature table hands over crashed the drawing loop
with a ``TypeError`` about ``-``.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import sequence_viewer as sv

SEQ = "A" * 30


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_a_backwards_feature_is_refused_where_it_is_written():
    with pytest.raises(ValueError, match="spans 20..5: end must be after start"):
        sv.SequenceAnnotation("promoter", 20, 5)


def test_a_zero_length_feature_is_the_same_mistake():
    with pytest.raises(ValueError, match="spans 7..7"):
        sv.SequenceAnnotation("cut site", 7, 7)


def test_coordinates_that_arrive_as_text_are_positions_after_all():
    """GFF and BED parsers hand over strings; that is not an error, it is a cast."""
    ann = sv.SequenceAnnotation("rrnB", "123", "456")

    assert (ann.start, ann.end) == (123, 456)


def test_a_coordinate_that_is_not_a_position_says_which_feature_it_came_from():
    with pytest.raises(ValueError, match="annotation 'rrnB' end must be a base "
                                        "position, got '1-2'"):
        sv.SequenceAnnotation("rrnB", 3, "1-2")


def test_a_feature_without_an_end_is_named_instead_of_vanishing():
    with pytest.raises(ValueError, match="unnamed annotation end must be a base "
                                        "position, got None"):
        sv.draw_sequence_view_with_annotations(SEQ, [{"start": 3}])


def test_a_list_entry_that_is_not_a_feature_is_pointed_at_by_number():
    with pytest.raises(ValueError, match="annotation #1 is a str, not a dict"):
        sv.draw_sequence_view_with_annotations(SEQ, ["promoter"])


def test_a_strand_the_view_cannot_draw_is_refused():
    with pytest.raises(ValueError, match="has strand '-1': use '\\+' or '-'"):
        sv.SequenceAnnotation("oriC", 1, 5, strand="-1")


def test_the_wrapper_passes_the_fields_it_was_given_through():
    """The positive control, and the reason the wrapper stopped dropping them."""
    fig = sv.draw_sequence_view_with_annotations(
        SEQ, [{"name": "lac", "start": "4", "end": "12", "color": "#00ff00",
               "strand": "-", "feature_type": "CDS"}])

    ann = fig.axes[2]
    assert [t.get_text() for t in ann.texts] == ["lac"]
    box = ann.patches[0]
    assert (round(box.get_x(), 2), round(box.get_width(), 2)) == (4, 8)
    assert box.get_facecolor()[:3] == pytest.approx((0.0, 1.0, 0.0))


def test_the_sequence_itself_is_still_validated_through_the_wrapper():
    with pytest.raises(ValueError, match="must not be empty"):
        sv.draw_sequence_view_with_annotations("", [{"name": "x", "start": 1, "end": 2}])
