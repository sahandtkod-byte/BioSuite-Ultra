"""The reverse-strand work reads the same sequence the forward-strand work reads.

``_find_orfs_forward`` folded U into T and ``_find_orfs_reverse`` did not, so an
RNA string got forward ORFs and an empty reverse list, and the reverse-complement
helper it uses was the module's own ACGT-only table: ``R, Y, S, W, K, M, B, D, H,
V`` came back merely reversed and ``U`` came back as ``U``, so a contig with
ambiguity codes had its -1/-2/-3 frames translated from something that was only
half complemented.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.core.sequence import reverse_complement
from biosuite.plotting import sequence_viewer as sv

#: A coding strand whose reverse complement carries ATG GAA TTT TAA.
RC_WANTED = "ATGAAATTTTAA"
DNA = reverse_complement(RC_WANTED)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_the_same_sequence_written_as_rna_finds_the_same_reverse_orfs():
    rna = DNA.replace("T", "U")

    assert sv._find_orfs_reverse(rna, min_aa=1) == sv._find_orfs_reverse(DNA, min_aa=1)
    assert sv._find_orfs_forward(rna) == sv._find_orfs_forward(DNA)


def test_the_reverse_orf_the_sequence_actually_hides_is_found():
    """The positive control for the equality above: it is not two empty lists."""
    found = sv._find_orfs_reverse(DNA, min_aa=1)

    assert found == [{"frame": -1, "start": 3, "end": 12, "length": 3,
                      "strand": "-", "terminated": True}], found


@pytest.mark.parametrize("seq", ["ATGRYSKMBDHVN", "atgryskmbdhvn", "ACGT"])
def test_the_viewer_complements_ambiguity_codes_the_way_the_package_does(seq):
    assert sv._revcomp(seq) == reverse_complement(seq.upper()), \
        "the view folds case first, then complements every code it knows"


def test_letters_that_are_not_nucleotides_are_refused_not_reversed():
    with pytest.raises(ValueError, match="not IUPAC nucleotides"):
        sv._revcomp("XYZ123")


def test_a_complement_code_that_survives_is_still_read_in_the_right_frame():
    """An IUPAC code inside a codon is an unknown amino acid, never a wrong one."""
    seq = "ATGRYGTAA" + "A" * 30          # ATG RYG TAA -> M ? stop

    amino = sv._translate_frame(seq, 0)

    assert amino[0] == "M" and amino[2] == "*", amino
    assert amino[1] == "X", "RYG is not a codon the table knows"
