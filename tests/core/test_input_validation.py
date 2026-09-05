"""Input-validation regression tests for the core sequence functions (BSU-020).

The failure mode these guard against is not a crash - it is a *plausible
number*.  ``gc_content("ACGT!@#")`` returned 28.57 %, ``reverse_complement
("XYZ123")`` returned ``"321ZYX"`` as if it had been complemented, and
``codon_usage_table("")`` reported ``total_codons: 1``.
"""
import pytest

from biosuite.core.codon_usage import (
    codon_usage_table, kmer_composition, sequence_complexity,
)
from biosuite.core.popgen import hardy_weinberg_test
from biosuite.core.sequence import (
    gc_content, reverse_complement, sequence_stats, translate,
    validate_nucleotide_sequence,
)

NOT_SEQUENCES = ["XYZ123", "ACGT!@#", "MKFLVQPW", ">chr1 header", "ACGT\x00"]
NOT_STRINGS = [None, 123, 4.5, ["A", "C"], {"seq": "ACGT"}, b"ACGT"]


# ── the shared validator ────────────────────────────────────────────────────

@pytest.mark.parametrize("value", NOT_STRINGS)
def test_validator_rejects_non_strings(value):
    with pytest.raises(TypeError):
        validate_nucleotide_sequence(value)


@pytest.mark.parametrize("value", NOT_SEQUENCES)
def test_validator_rejects_non_nucleotides(value):
    with pytest.raises(ValueError, match="IUPAC"):
        validate_nucleotide_sequence(value)


@pytest.mark.parametrize("value", [
    "ACGT", "acgt", "ACGU", "ACGTRYSWKMBDHVN", "AC-GT", "ACGT\n ACGT",
])
def test_validator_accepts_legitimate_sequences(value):
    assert validate_nucleotide_sequence(value)


def test_validator_strips_whitespace_and_newlines():
    assert validate_nucleotide_sequence("AC GT\nAC") == "ACGTAC"


def test_empty_is_allowed_by_default_and_refusable():
    assert validate_nucleotide_sequence("") == ""
    with pytest.raises(ValueError):
        validate_nucleotide_sequence("", allow_empty=False)


# ── gc_content ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", NOT_SEQUENCES)
def test_gc_content_refuses_non_nucleotides(value):
    with pytest.raises(ValueError):
        gc_content(value)


@pytest.mark.parametrize("value", NOT_STRINGS)
def test_gc_content_refuses_non_strings(value):
    with pytest.raises(TypeError):
        gc_content(value)


def test_gc_content_known_values():
    assert gc_content("ACGT") == pytest.approx(50.0)
    assert gc_content("GGCC") == pytest.approx(100.0)
    assert gc_content("AATT") == pytest.approx(0.0)
    assert gc_content("") == 0.0


def test_gc_content_ignores_gaps_in_the_denominator():
    """Gaps are not bases, so they must not dilute the percentage."""
    assert gc_content("GC--") == pytest.approx(100.0)
    assert gc_content("GCAT") == gc_content("GC-AT-".replace("-", ""))


def test_gc_content_is_case_insensitive():
    assert gc_content("acgt") == gc_content("ACGT")


# ── reverse_complement ──────────────────────────────────────────────────────

@pytest.mark.parametrize("value", NOT_SEQUENCES)
def test_reverse_complement_refuses_non_nucleotides(value):
    with pytest.raises(ValueError):
        reverse_complement(value)


def test_reverse_complement_known_values():
    assert reverse_complement("ACGT") == "ACGT"
    assert reverse_complement("AAAA") == "TTTT"
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("") == ""


def test_reverse_complement_is_an_involution():
    for seq in ("ACGTACGGTA", "N" * 5, "ACGTRYSWKM"):
        assert reverse_complement(reverse_complement(seq)) == seq


def test_reverse_complement_handles_iupac_ambiguity_codes():
    # R (A/G) complements to Y (C/T) and vice versa.
    assert reverse_complement("R") == "Y"
    assert reverse_complement("Y") == "R"
    assert reverse_complement("S") == "S"          # S (G/C) is self-complementary
    assert reverse_complement("W") == "W"          # W (A/T) is self-complementary
    assert reverse_complement("B") == "V"


# ── translate ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", NOT_SEQUENCES)
def test_translate_refuses_non_nucleotides(value):
    with pytest.raises(ValueError):
        translate(value)


def test_translate_known_values():
    assert translate("ATGAAATTTTAA") == "MKF*"
    assert translate("ATGNNNTAA") == "MX*"
    assert translate("") == ""


@pytest.mark.parametrize("frame", [0, 4, -4, 7, 1.5, "1"])
def test_translate_rejects_invalid_frames(frame):
    with pytest.raises((ValueError, TypeError)):
        translate("ATGAAATTTTAA", frame=frame)


def test_translate_rejects_unimplemented_genetic_codes():
    """table= was accepted and then ignored, silently using the standard code."""
    with pytest.raises(ValueError):
        translate("ATGAAATTTTAA", table=11)


def test_negative_frame_translates_the_reverse_strand():
    forward = "ATGAAATTTTAA"
    assert translate(forward, frame=-1) == translate(reverse_complement(forward))


# ── sequence_stats ──────────────────────────────────────────────────────────

def test_sequence_stats_counts_add_up():
    stats = sequence_stats("AACCGGTTNN")
    assert stats["length"] == 10
    assert stats["A"] == 2 and stats["C"] == 2
    assert stats["G"] == 2 and stats["T"] == 2 and stats["N"] == 2
    assert stats["GC"] == pytest.approx(40.0)


# ── codon usage ─────────────────────────────────────────────────────────────

def test_total_codons_is_zero_for_an_empty_sequence():
    """It used to report 1: the divide-by-zero guard leaked into the count."""
    assert codon_usage_table("")["total_codons"] == 0


def test_total_codons_counts_real_codons():
    assert codon_usage_table("ATGAAATTTTAA")["total_codons"] == 4


@pytest.mark.parametrize("func", [codon_usage_table, kmer_composition,
                                  sequence_complexity])
def test_codon_module_refuses_non_nucleotides(func):
    with pytest.raises(ValueError):
        func("XYZ123")


@pytest.mark.parametrize("k", [0, -1, 1.5])
def test_kmer_composition_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        kmer_composition("ACGTACGT", k=k)


def test_kmer_frequencies_sum_to_one():
    result = kmer_composition("ACGTACGTAC", k=3)
    assert sum(v["frequency"] for v in result.values()) == pytest.approx(1.0, abs=1e-6)


# ── Hardy-Weinberg ──────────────────────────────────────────────────────────

def test_hwe_rejects_an_empty_population():
    """Returning chi2=0/p=1 declared a population of nobody to be in HWE."""
    with pytest.raises(ValueError, match="empty population"):
        hardy_weinberg_test({"AA": 0, "Aa": 0, "aa": 0})


@pytest.mark.parametrize("counts", [
    {"AA": -1, "Aa": 10, "aa": 5},
    {"AA": 10, "Aa": -10, "aa": 5},
    {"AA": 10, "Aa": 10, "aa": -5},
])
def test_hwe_rejects_negative_counts(counts):
    with pytest.raises(ValueError, match="non-negative"):
        hardy_weinberg_test(counts)


@pytest.mark.parametrize("counts", [
    {"AA": "10", "Aa": 10, "aa": 5},
    {"AA": None, "Aa": 10, "aa": 5},
    {"AA": [1], "Aa": 10, "aa": 5},
])
def test_hwe_rejects_non_numeric_counts(counts):
    with pytest.raises(TypeError):
        hardy_weinberg_test(counts)


def test_hwe_reports_whether_the_chi_square_approximation_is_valid():
    """Every expected count must be >= 5 for the chi-square test to be usable."""
    tiny = hardy_weinberg_test({"AA": 2, "Aa": 1, "aa": 0})
    assert tiny["chi2_approximation_valid"] is False
    large = hardy_weinberg_test({"AA": 100, "Aa": 200, "aa": 100})
    assert large["chi2_approximation_valid"] is True


def test_hwe_p_value_does_not_underflow_to_zero():
    """`1 - cdf` returns exactly 0 for extreme chi2; `sf` keeps precision."""
    result = hardy_weinberg_test({"AA": 5000, "Aa": 0, "aa": 5000})
    assert result["chi2"] > 100
    assert 0.0 <= result["p_value"] < 1e-5
