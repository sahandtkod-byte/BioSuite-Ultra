"""ORF records a caller hands the map are checked, and the ones it cannot place are counted.

The map reads four keys out of every dict.  A missing one died as
``KeyError: 'length'`` somewhere inside the drawing loop, a frame outside
``-3..3`` was dropped with no trace (a caller counting rows saw fewer bars than
records), and passing a generator rather than a list drew an empty figure — the
length scan consumed it and the drawing loop found nothing.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import sequence_viewer as sv

SEQ = "A" * 600


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_a_record_without_a_length_is_named_by_position():
    orfs = [{"frame": 1, "start": 0, "end": 30, "length": 10},
            {"frame": 2, "start": 0, "end": 30}]

    with pytest.raises(ValueError, match="ORF #2 has no length"):
        sv.draw_orf_map(SEQ, orfs=orfs)


def test_a_record_that_is_not_a_record_at_all_says_so():
    with pytest.raises(ValueError, match="ORF #1 is a str, not a dict"):
        sv.draw_orf_map(SEQ, orfs=["ATGGGG"])


def test_a_row_the_map_does_not_have_is_counted_out_loudly():
    orfs = [{"frame": 1, "start": 0, "end": 30, "length": 10},
            {"frame": 9, "start": 40, "end": 100, "length": 20},
            {"frame": 0, "start": 40, "end": 100, "length": 20}]

    fig = sv.draw_orf_map(SEQ, orfs=orfs)

    assert len(fig.axes[0].patches) == 1
    title = fig.axes[0].get_title()
    assert "2 of 3 records have no row in" in title, title


def test_a_generator_of_records_draws_the_records_it_holds():
    orfs = [{"frame": 1, "start": 0, "end": 30, "length": 10}]

    fig = sv.draw_orf_map(SEQ, orfs=(o for o in orfs))

    assert len(fig.axes[0].patches) == 1, "the scan consumed the generator first"


def test_the_autodetected_frames_land_on_the_rows_the_map_labels():
    """Positive control: ORFs at offsets 0 and 44 are frames +1 and +3."""
    orf1 = "ATG" + "GGG" * 12 + "TAA"           # 42 bp, offset 0   -> frame +1
    orf2 = "ATG" + "AAA" * 12 + "TGA"           # 42 bp, offset 44  -> frame +3
    seq = orf1 + "CC" + orf2 + "TT"

    fig = sv.draw_orf_map(seq)

    rows = sorted(round(patch.get_y() + 0.4) for patch in fig.axes[0].patches)
    assert 0 in rows and 2 in rows, f"+1 is row 0 and +3 is row 2, got {rows}"
    assert "records have no row" not in fig.axes[0].get_title()
    assert [(o["frame"], o["start"]) for o in sv._find_orfs_forward(seq, min_aa=5)] \
        == [(1, 0), (3, 44)]


def test_an_empty_sequence_is_refused_before_it_divides_by_its_own_length():
    before = set(plt.get_fignums())

    with pytest.raises(ValueError, match="must not be empty"):
        sv.draw_orf_map("", orfs=[{"frame": 1, "start": 0, "end": 0, "length": 1}])

    assert set(plt.get_fignums()) == before
