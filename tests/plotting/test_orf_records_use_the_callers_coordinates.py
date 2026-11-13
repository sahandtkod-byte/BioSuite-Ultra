"""An ORF record's coordinates are the caller's, not the scanner's working space.

The reverse scan runs on the reverse complement and has to map back.  The version
before this subtracted the frame as well, which moved every reverse ORF up to two
bases off; the pins below are the arithmetic that replaced it, kept here so a
future "tidy up" of the mapping has something to break.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.core.sequence import reverse_complement
from biosuite.plotting import sequence_viewer as sv


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_a_reverse_orf_is_reported_where_it_sits_in_the_original_sequence():
    rc = "ATGCCGCCGCCCTAA" + "TTT"           # M P P P stop on the complement strand
    seq = reverse_complement(rc)
    length = len(seq)

    found = sv._find_orfs_reverse(seq, min_aa=1)

    assert len(found) == 1, found
    orf = found[0]
    assert orf["frame"] == -1
    assert orf["length"] == 4
    assert orf["start"] == length - 12, "the rc span [0, 12) is the tail here"
    assert orf["end"] == length - 0
    assert orf["strand"] == "-"
    assert orf["terminated"] is True


def test_a_forward_orf_ends_at_its_stop_codon_without_including_it():
    found = sv._find_orfs_forward("ATGCCGCCCTAA" + "ATG", min_aa=1)

    assert found[0]["length"] == 3, found
    assert found[0]["end"] == 9, "TAA is at 9..12 and is not part of the coding span"
    assert found[0]["start"] == 0


def test_the_three_forward_frames_start_where_their_names_say():
    """An ATG at offset 1, 2 and 3 is found by frame +2, +3 and... +1 of the next round."""
    for offset in (0, 1, 2):
        seq = "C" * offset + "ATG" + "GGG" * 11 + "TAA" + "C" * 3

        found = sv._find_orfs_forward(seq, min_aa=5)

        assert [o["frame"] for o in found] == [offset + 1], (offset, found)
        assert found[0]["start"] == offset, found


def test_the_default_floor_hides_the_short_orf_rather_than_deleting_it():
    short_then_long = "ATG" + "GGG" * 3 + "TAA" + "ATG" + "GGG" * 30 + "TAA"

    assert [o["length"] for o in sv._find_orfs_forward(short_then_long)] == [31], \
        "the 4 aa ORF is under the default floor and the 31 aa one is not"
    assert [o["length"] for o in sv._find_orfs_forward(short_then_long, min_aa=1)] == \
        [4, 31]
    assert [o["length"] for o in sv._find_orfs_forward(short_then_long, min_aa=4)] == \
        [4, 31]


def test_the_reverse_scan_applies_the_same_floor_it_was_given():
    """The complement of that sequence carries the same two ORFs, backwards."""
    seq = reverse_complement("ATG" + "GGG" * 3 + "TAA" + "ATG" + "GGG" * 30 + "TAA")

    assert [o["length"] for o in sv._find_orfs_reverse(seq)] == [31]
    assert [o["length"] for o in sv._find_orfs_reverse(seq, min_aa=1)] == [4, 31]
    assert all(o["strand"] == "-" for o in sv._find_orfs_reverse(seq, min_aa=1))
