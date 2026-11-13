"""A per-base view draws bases, and only as many as a screen can hold.

Measured on this machine before the budget existed: about 1.1 ms of build time
and 3.2 text artists per base — 2,031 bp took 2.3 s and 6,396 artists, and the
CLI's "DNA sequence (or FASTA path)" option hands the view the whole record.
The picture is useless long before that: the adaptive font floors at 3 pt around
270 bases, so anything past roughly a thousand bases is one grey band that the
figure never admitted being.
"""
import time

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import sequence_viewer as sv


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _random_dna(length, seed=0):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), length))


def _bases_drawn(track):
    return len(track.texts)


def test_a_sequence_of_two_thousand_bases_draws_the_first_thousand():
    seq = _random_dna(2_031)

    fig = sv.draw_sequence_view(seq)

    track = fig.axes[1]
    assert _bases_drawn(track) == sv._MAX_VIEW_BASES
    assert [t.get_text() for t in track.texts] == list(seq[:sv._MAX_VIEW_BASES]), \
        "the bases shown have to be the bases it starts with"


def test_the_cut_is_written_on_the_track_that_was_cut():
    fig = sv.draw_sequence_view(_random_dna(2_031))

    title = fig.axes[1].get_title(loc="left")
    assert title.startswith("Sequence  (bases 0"), title
    assert "1,000" in title and "2,031" in title and "start=1,000" in title, title


def test_a_twenty_thousand_base_record_takes_the_same_time_as_a_thousand():
    """The CLI path: a whole FASTA record, and the figure stays a figure."""
    t0 = time.perf_counter()
    fig = sv.draw_sequence_view(_random_dna(20_000, seed=1))
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"a bounded view took {elapsed:.1f}s"
    assert _bases_drawn(fig.axes[1]) == sv._MAX_VIEW_BASES
    assert fig.axes[0].texts[-1].get_text() == "990", "the ruler stops with the bases"


def test_a_window_the_caller_chose_is_not_reported_as_a_loss():
    """The positive control: 10 of 16 bases asked for is not a cut-down view."""
    fig = sv.draw_sequence_view("ATGAAACCCGGGTTTA", end=10)

    assert fig.axes[1].get_title(loc="left") == "Sequence"
    assert _bases_drawn(fig.axes[1]) == 10


def test_the_six_frames_are_cut_to_the_same_number_of_bases_as_the_letters():
    fig = sv.draw_sequence_view(_random_dna(1_500))

    frames = fig.axes[2:]
    assert len(frames) == 6, "ruler, nucleotides and six frames"
    for ax, label in zip(frames, ["+1", "+2", "+3", "-1", "-2", "-3"]):
        last = max(t.get_position()[0] for t in ax.texts)
        assert last < sv._MAX_VIEW_BASES, f"frame {label} drew past the window"
        assert ax.get_xlim()[1] == sv._MAX_VIEW_BASES, label


def test_a_view_the_budget_needs_is_left_alone():
    fig = sv.draw_sequence_view(_random_dna(200, seed=2))

    assert _bases_drawn(fig.axes[1]) == 200
    assert fig.axes[1].get_title(loc="left") == "Sequence"
