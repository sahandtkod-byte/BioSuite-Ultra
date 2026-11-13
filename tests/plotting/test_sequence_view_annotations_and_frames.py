"""The sequence view's window, its annotation rows and the short-sequence panels.

``draw_sequence_view`` slices the sequence before it lays anything out, so a feature
outside the window is dropped rather than drawn at a negative position, and two
features that overlap have to find separate rows.  The dashboard and the translation
view both have a "too short to show" panel, and the GC plot accepts its window
under two different names.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import sequence_viewer as sv

#: 16 bases: ATG AAA CCC GGG TTT A — a start codon and three full codons in frame 1.
SEQ = "ATGAAACCCGGGTTTA"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _tracks(fig):
    """The axes of a sequence view in order: ruler, nts, [annotations], 6 frames."""
    return fig.axes


# ─── the window and the annotations ─────────────────────────────────────────

def test_the_view_ends_where_the_caller_asked():
    fig = sv.draw_sequence_view(SEQ, end=10)

    assert len(_tracks(fig)) == 8, "ruler, nucleotides and six frames"
    assert _tracks(fig)[0].get_xlim() == (0, 10), "the ruler stops at the window"


def test_a_feature_outside_the_window_is_left_out_rather_than_pinned_to_the_edge():
    inside = sv.SequenceAnnotation("inside", 0, 5)
    beyond = sv.SequenceAnnotation("beyond", 50, 60)

    fig = sv.draw_sequence_view(SEQ, annotations=[inside, beyond])

    track = _tracks(fig)[2]
    assert len(track.patches) == 1, "only the feature the window contains was drawn"
    assert [t.get_text() for t in track.texts] == ["inside"]


def test_two_overlapping_features_find_rows_of_their_own():
    first = sv.SequenceAnnotation("a", 0, 5)
    second = sv.SequenceAnnotation("b", 2, 7)

    fig = sv.draw_sequence_view(SEQ, annotations=[first, second])

    track = _tracks(fig)[2]
    bottoms = [round(patch.get_y(), 2) for patch in track.patches]
    assert len(bottoms) == 2
    assert bottoms[1] > bottoms[0], "the overlapping feature moved up a row"
    assert [t.get_text() for t in track.texts] == ["a", "b"]


def test_annotations_can_arrive_as_plain_dicts():
    fig = sv.draw_sequence_view_with_annotations(
        SEQ, [{"name": "promoter", "start": 0, "end": 6, "color": "#ff0000"}])

    track = _tracks(fig)[2]
    assert [t.get_text() for t in track.texts] == ["promoter"]
    assert track.patches[0].get_facecolor()[:3] == pytest.approx((1.0, 0.0, 0.0))


# ─── the translation view ───────────────────────────────────────────────────

def test_a_sequence_shorter_than_a_codon_says_what_it_is_short_for():
    fig = sv.draw_translation_view("AT")

    assert [t.get_text() for t in fig.axes[0].texts] == [
        "Sequence too short for translation: 2 bases leave no complete codon "
        "in frame 1"], "the length and the frame are the two things it could be"


def test_the_reading_frame_moves_where_the_translation_starts():
    fig = sv.draw_translation_view("ATGAAA", frame=3)

    dna_row, amino_row = fig.axes
    assert [t.get_text() for t in amino_row.texts] == ["E"], "GAA, not ATG"
    assert [t.get_text() for t in dna_row.texts if t.get_text() in ("G", "A", "T", "C")] == ["G", "A", "A"]


# ─── the GC plot and the ORF map ────────────────────────────────────────────

def test_the_window_size_alias_selects_the_same_window_as_the_argument():
    as_window = sv.draw_gc_content_plot("GCTATGCG", window=3)
    as_alias = sv.draw_gc_content_plot("GCTATGCG", window_size=3)

    first = np.asarray(as_window.axes[0].lines[0].get_ydata(), dtype=float)
    second = np.asarray(as_alias.axes[0].lines[0].get_ydata(), dtype=float)
    assert np.allclose(first, second), "the two names must not drift apart"
    assert first[0] == pytest.approx(66.66666667, abs=1e-4), "GC of GCT is two thirds"


def test_an_orf_in_a_frame_the_map_does_not_have_is_left_out():
    orfs = [{"frame": 7, "start": 0, "end": 5, "length": 5},
            {"frame": 1, "start": 0, "end": 5, "length": 5}]

    fig = sv.draw_orf_map(SEQ, orfs=orfs)

    assert len(fig.axes[0].patches) == 1, "only frame 1 has a row to sit on"


def test_an_orf_too_thin_to_read_gets_no_label():
    long_seq = "ATG" + "A" * 2000
    orfs = [{"frame": 1, "start": 0, "end": 1, "length": 30},
            {"frame": 2, "start": 10, "end": 400, "length": 130}]

    fig = sv.draw_orf_map(long_seq, orfs=orfs)

    labels = [t.get_text() for t in fig.axes[0].texts if t.get_text().endswith("aa")]
    assert labels == ["130aa"], "a one-base bar cannot carry its own length"


# ─── the dashboard ──────────────────────────────────────────────────────────

def test_a_two_letter_sequence_shortens_both_of_the_panels_that_need_more():
    fig = sv.create_sequence_overview("AC")

    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert texts.count("Sequence too short") == 2, texts
    assert any("Length:  2 bp" in t for t in texts), "the statistics panel still reports what it has"
