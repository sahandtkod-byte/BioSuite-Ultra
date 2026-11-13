"""Case folding in the p-distance functions and the inputs they refuse.

A soft-masked FASTA — the normal shape of a UCSC or RepeatMasker download — is
written in lower case.  Before the comparison was folded, `p_distance('ACGT',
'acgt')` measured **1.0**: the identical sequence read as maximally divergent, and
`distance_matrix(['ACGTACGT', 'acgtacgt', 'ACGTACGA'])` returned
`[[0, 1, .125], [1, 0, 1], [.125, 1, 0]]`, placing the masked taxon farthest from
everything it is identical to.
"""
import numpy as np
import pytest

from biosuite.core.phylogeny import distance_matrix, p_distance


def test_identical_sequences_in_different_case_are_zero_apart():
    assert p_distance("ACGT", "acgt") == 0.0
    assert p_distance("acgt", "ACGT") == 0.0
    assert p_distance("aCgT", "AcGt") == 0.0


def test_one_real_difference_is_one_site_of_four():
    assert p_distance("ACGT", "ACGA") == pytest.approx(0.25)
    assert p_distance("acgt", "ACGA") == pytest.approx(0.25)


def test_gaps_still_do_not_count_as_sites():
    assert p_distance("ACGT", "AC-T") == 0.0
    assert p_distance("----", "ACGT") == 0.0
    assert p_distance("AC-GT", "AC-GT") == 0.0


def test_the_matrix_ignores_case_in_every_pair():
    mat = distance_matrix(["ACGTACGT", "acgtacgt", "ACGTACGA"])
    assert mat.tolist() == [[0.0, 0.0, 0.125], [0.0, 0.0, 0.125], [0.125, 0.125, 0.0]]
    assert np.array_equal(mat, mat.T), "the matrix stays symmetric"


def test_a_single_sequence_gives_a_zero_by_itself():
    assert distance_matrix(["ACGT"]).tolist() == [[0.0]]
    assert distance_matrix([]).shape == (0, 0)


@pytest.mark.parametrize("pair", [["ACGT", "AC"], ["AC", "ACGT"], ["ACGT", "ACG", "A"]])
def test_unaligned_input_is_named_rather_than_broadcast_wrong(pair):
    """numpy used to answer 'setting an array element with a sequence. The
    requested array has an inhomogeneous shape after 1 dimensions'."""
    with pytest.raises(ValueError) as exc:
        distance_matrix(pair)
    assert str(exc.value) == (
        "distance_matrix needs sequences of equal length, got lengths "
        f"{sorted({len(s) for s in pair})}; align them first "
        "(see biosuite.core.msa).")


def test_p_distance_still_refuses_unequal_lengths():
    with pytest.raises(ValueError, match="Sequences must be same length"):
        p_distance("ACGT", "ACG")


def test_a_lowercase_taxon_no_longer_drifts_to_the_root():
    """End-to-end consequence of the fold: with the masked sequence the tree
    merged it last and at the greatest height; folded, the two identical
    sequences merge first at height 0."""
    from biosuite.core.phylogeny import upgma_tree
    mat = distance_matrix(["ATCG", "atcg", "AAAA", "TTTT"])
    assert upgma_tree(mat)[0].tolist() == [0.0, 1.0, 0.0, 2.0]
    assert np.array_equal(mat, distance_matrix(["ATCG", "ATCG", "AAAA", "TTTT"]))
