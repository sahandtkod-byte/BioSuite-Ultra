"""The bounds of `design_primers`' primer_length argument.

``seq[:0]`` is empty while ``seq[-0:]`` is the *whole* template, so a length of
zero used to return an empty forward primer plus a "reverse" primer as long as
the target, and a negative length quietly trimmed both ends.
"""
import pytest

from biosuite.core import cloning as cl

TEMPLATE = "ATGCATGCATGCATGCATGCATGC"          # 24 nt, 50 % GC


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_a_non_positive_primer_length_is_refused_by_value(bad):
    with pytest.raises(ValueError) as exc:
        cl.design_primers(TEMPLATE, primer_length=bad)
    assert str(exc.value) == f"Primer length must be at least 1 nt, got {bad}."


def test_a_one_base_primer_is_allowed_and_reads_from_both_ends():
    pair = cl.design_primers("ATGC", primer_length=1)
    assert (pair["forward"], pair["reverse"]) == ("A", "G")
    assert (pair["fwd_tm"], pair["rev_tm"]) == (2.0, 4.0)
    assert (pair["fwd_gc"], pair["rev_gc"]) == (0.0, 100.0)
    assert pair["warnings"] == [
        "Forward primer GC content (0.0%) outside ideal range (40-60%).",
        "Reverse primer GC content (100.0%) outside ideal range (40-60%).",
    ]


def test_a_primer_as_long_as_the_template_takes_the_whole_molecule():
    pair = cl.design_primers("ATGC", primer_length=4)
    assert pair["forward"] == "ATGC"
    assert pair["reverse"] == cl._reverse_complement("ATGC") == "GCAT"
    assert pair["warnings"] == [
        "Target sequence (4 bp) shorter than 2× primer length (8 bp). "
        "Primers may overlap."]


def test_a_target_one_base_shorter_than_the_primer_is_still_refused():
    with pytest.raises(ValueError) as exc:
        cl.design_primers("ATGC", primer_length=5)
    assert str(exc.value) == ("Target sequence (4 bp) is shorter than primer "
                              "length (5 bp).")


def test_the_default_length_still_designs_from_a_normal_template():
    template = ("ATGCGTACGTTAGCGTACGTACAGTACTGCATGCATCGTACGTACGATCGACTGCATGCA")
    assert len(template) == 60
    pair = cl.design_primers(template)
    assert pair == {
        "forward": "ATGCGTACGTTAGCGTACGT",
        "reverse": "TGCATGCAGTCGATCGTACG",
        "fwd_tm": 51.8, "rev_tm": 53.8, "fwd_gc": 50.0, "rev_gc": 55.0,
        "warnings": []}
