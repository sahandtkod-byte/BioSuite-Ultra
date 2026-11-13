"""`verify_insert`'s backbone argument, which used to be accepted and ignored.

The docstring promised "if provided, checks that the insert is between vector
segments".  The parameter was never read: passing a backbone changed nothing in the
result, so a product that carried no vector at all, or the wrong one, passed the
check exactly as a good clone does.
"""
import pytest

from biosuite.core import cloning as cl

BACKBONE = "AAAGGGCCCTTTAAAGGGCCCTTTAAAGGG"          # 30 bp, all A/C/G/T
INSERT = "GATTACA"


def test_a_backbone_opened_at_one_site_passes_without_a_warning():
    product = BACKBONE[:10] + INSERT + BACKBONE[10:]
    result = cl.verify_insert(product, INSERT, vector_backbone=BACKBONE)
    assert result["insert_found"] is True
    assert (result["insert_start"], result["insert_end"]) == (10, 17)
    assert result["warnings"] == []


def test_the_same_check_survives_a_cut_at_either_end_of_the_backbone():
    """The rotation search must not depend on where the vector was opened."""
    for cut in (0, 1, 15, 29):
        product = BACKBONE[cut:] + INSERT + BACKBONE[:cut]
        result = cl.verify_insert(product, INSERT, vector_backbone=BACKBONE)
        assert result["warnings"] == [], cut


def test_flanks_that_are_not_the_given_backbone_are_reported():
    product = "TTTT" + INSERT + "AAAA"
    result = cl.verify_insert(product, INSERT, vector_backbone="AAAGGGCCCTTT")
    assert result["warnings"] == [
        "The sequence flanking the insert is not contiguous in the given backbone: "
        "the insert is not between vector segments."]


def test_a_product_that_is_only_the_insert_is_reported():
    result = cl.verify_insert(INSERT, INSERT, vector_backbone="AAAGGGCCCTTT")
    assert result["warnings"] == [
        "No backbone material remains outside the insert: the product is the "
        "insert alone."]


def test_more_vector_than_the_backbone_handed_in_is_reported():
    product = BACKBONE + INSERT + BACKBONE
    result = cl.verify_insert(product, INSERT, vector_backbone=BACKBONE)
    assert result["warnings"] == [
        "Vector material outside the insert is 60 bp but the backbone is 30 bp: "
        "the product carries more vector than was handed in."]


def test_a_reverse_oriented_insert_still_keeps_its_flanks_contiguous():
    from biosuite.core.sequence import reverse_complement
    rc = reverse_complement(INSERT)
    product = BACKBONE[:8] + rc + BACKBONE[8:]
    result = cl.verify_insert(product, INSERT, vector_backbone=BACKBONE)
    assert result["forward_orientation"] is False
    assert result["warnings"] == []


def test_no_backbone_check_runs_when_the_insert_is_absent():
    """The truncation warnings are the story here; a backbone verdict on a product
    with no insert found would only repeat it."""
    result = cl.verify_insert("TTTTTTTT", "ACGTACGTACGTACGTAGCTAGCT",
                              vector_backbone=BACKBONE)
    assert result["insert_found"] is False
    assert any("Insert NOT found" in w for w in result["warnings"])


def test_a_backbone_outside_the_iupac_alphabet_is_refused_by_name():
    with pytest.raises(ValueError,
                       match=r"Invalid characters in vector_backbone: Z\. "
                             r"Only IUPAC nucleotide codes are accepted\."):
        cl.verify_insert(BACKBONE[:10] + INSERT + BACKBONE[10:], INSERT,
                         vector_backbone="AAAGGGCCZZZ")


def test_ambiguity_codes_are_valid_dna_not_a_refusal():
    """N is an IUPAC code, so a masked backbone must pass validation."""
    product = "AAANNNGGG" + INSERT + "AAANNNGGG"
    result = cl.verify_insert(product, INSERT, vector_backbone="AAANNNGGGTTTCCCAAG")
    assert result["insert_found"] is True
