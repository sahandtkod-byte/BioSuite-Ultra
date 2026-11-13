"""Two features that overlap cannot share a row.

The old packing counted how many already-placed features overlapped the new one
and called that number the row — which is not the same question as "is this row
free here".  With A(0-10), B(8-14) and C(12-20), B and C both landed on row 1 and
were drawn through each other, labels included.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import sequence_viewer as sv

SEQ = "A" * 24


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _rows(track):
    """``{row_y: [(name, from, to)]}`` as the figure actually placed them."""
    rows = {}
    for text, patch in zip(track.texts, track.patches):
        rows.setdefault(round(patch.get_y(), 3), []).append(
            (text.get_text(), round(patch.get_x(), 2),
             round(patch.get_x() + patch.get_width(), 2)))
    return {y: sorted(spans) for y, spans in rows.items()}


def _collisions(rows):
    out = []
    for spans in rows.values():
        for i, left in enumerate(spans):
            for right in spans[i + 1:]:
                if right[1] < left[2]:
                    out.append((left[0], right[0]))
    return out


def test_the_feature_that_forced_its_neighbour_up_does_not_take_the_vacated_row():
    annots = [sv.SequenceAnnotation("A", 0, 10), sv.SequenceAnnotation("B", 8, 14),
              sv.SequenceAnnotation("C", 12, 20)]

    fig = sv.draw_sequence_view(SEQ, annotations=annots, end=24)

    rows = _rows(fig.axes[2])
    assert _collisions(rows) == [], f"features drawn through each other: {rows}"
    assert len(rows) == 2, f"A and C share a row they do not overlap: {rows}"


def test_features_that_do_not_touch_stay_on_the_lowest_row():
    """The positive control: packing is real, not one row per feature."""
    annots = [sv.SequenceAnnotation("a", 0, 5), sv.SequenceAnnotation("b", 10, 15),
              sv.SequenceAnnotation("c", 16, 20)]

    fig = sv.draw_sequence_view(SEQ, annotations=annots, end=24)

    rows = _rows(fig.axes[2])
    assert len(rows) == 1, f"three non-overlapping features needed one row: {rows}"
    assert sorted(name for spans in rows.values() for name, _, _ in spans) == \
        ["a", "b", "c"]


def test_a_feature_that_ends_before_the_window_is_counted_out_loudly():
    annots = [sv.SequenceAnnotation("inside", 0, 5),
              sv.SequenceAnnotation("beyond", 200, 240)]

    fig = sv.draw_sequence_view(SEQ, annotations=annots, end=24)

    track = fig.axes[2]
    assert [t.get_text() for t in track.texts] == ["inside"]
    assert track.get_title(loc="left") == \
        "Annotations  (1 of 2 features reach this window)"


def test_a_track_that_holds_everything_says_nothing_about_it():
    """The positive control: the count only appears when something is missing."""
    fig = sv.draw_sequence_view(
        SEQ, annotations=[sv.SequenceAnnotation("inside", 4, 9)], end=24)

    assert fig.axes[2].get_title(loc="left") == "Annotations"
