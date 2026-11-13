"""``draw_translation_view`` reads frame 1, 2 or 3 — and nothing else was ever checked.

An offset is `frame - 1`, so the values a caller reaches for by accident each
produced a *plausible* plot rather than an error: frame 0 gained a stray ``X``
codon and read from the second base, frame 4 returned frame 1 minus its first
codon, frame -1 read frame 2 behind an ``X``, and frame 99 came back saying
"Sequence too short for translation" about a sequence that was long enough.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import sequence_viewer as sv

#: 18 bases: six complete codons in frame 1 (M K P G F K).
CODING = "ATGAAACCCGGGTTTAAA"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("frame", [0, 4, -1, 99])
def test_a_frame_outside_one_two_three_is_named(frame):
    before = set(plt.get_fignums())

    with pytest.raises(ValueError, match=f"frame must be 1, 2 or 3, got {frame}"):
        sv.draw_translation_view(CODING, frame=frame)

    assert set(plt.get_fignums()) == before, "a refused frame drew a panel anyway"


@pytest.mark.parametrize("frame", [2.0, None, "1"])
def test_a_frame_that_is_not_a_frame_number_is_refused_too(frame):
    with pytest.raises(ValueError, match="frame must be a whole number"):
        sv.draw_translation_view(CODING, frame=frame)


def test_the_three_frames_it_does_accept_are_the_three_it_claims():
    """The positive control: the offset really is ``frame - 1`` and the codons follow."""
    read = {}
    for frame in (1, 2, 3):
        fig = sv.draw_translation_view(CODING, frame=frame)
        read[frame] = [t.get_text() for t in fig.axes[1].texts]
        plt.close(fig)

    assert read[1] == ["M", "K", "P", "G", "F", "K"], "ATG AAA CCC GGG TTT AAA"
    assert read[2] == ["*", "N", "P", "G", "L"], "TGA AAC CCG GGT TTA"
    assert read[3] == ["E", "T", "R", "V", "*"], "GAA ACC CGG GTT TAA"
