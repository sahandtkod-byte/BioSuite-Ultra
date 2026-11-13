"""Codon usage: frame and window validation, and the synonymous-codon table.

`codon_usage_table` accepts a reading frame, `sequence_complexity` a window size;
both had only ever been called with valid values, so the guards that name the
accepted set were unexecuted.  The second gap is subtler: a residue encoded by
several codons has to accumulate them under one amino-acid key, and only a
sequence with a synonymous pair runs the "already present" branch.
"""
import pytest

from biosuite.core.codon_usage import (
    codon_usage_table,
    kmer_composition,
    sequence_complexity,
)


# ── reading frame ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("frame", [0, 4, -4, 7, 1.5, None, "1"])
def test_a_frame_outside_the_six_is_refused_by_name(frame):
    with pytest.raises(ValueError, match=r"frame must be one of 1, 2, 3, -1, -2, -3"):
        codon_usage_table("ATGAAATTT", frame=frame)


@pytest.mark.parametrize("frame", [1, 2, 3, -1, -2, -3])
def test_every_valid_frame_produces_a_table(frame):
    result = codon_usage_table("ATGAAATTGCCCGGG", frame=frame)
    assert set(result) == {"codon_usage", "total_codons", "amino_acids", "codon_per_aa"}
    assert result["total_codons"] >= 1


def test_frame_two_starts_one_base_later_than_frame_one():
    seq = "AATGAAATTT"
    first = list(codon_usage_table(seq, frame=1)["codon_usage"])
    second = list(codon_usage_table(seq, frame=2)["codon_usage"])
    assert first != second
    # Frame 2 reads ATG/AAA/TTT from index 1; the keys come back sorted.
    assert second == ["AAA", "ATG", "TTT"]


def test_a_negative_frame_reads_the_reverse_complement():
    from biosuite.core.sequence import reverse_complement
    seq = "ATGAAATTGCCCGGG"
    rev = codon_usage_table(seq, frame=-1)
    assert rev == codon_usage_table(reverse_complement(seq), frame=1)
    assert rev["total_codons"] == codon_usage_table(seq, frame=1)["total_codons"]


@pytest.mark.parametrize("frame", [-1, -2, -3])
def test_each_negative_frame_equals_its_positive_counterpart_on_the_complement(
        frame, seq_for_frames):
    """-3 is not "the same as 3": it is the reverse complement read in frame 3."""
    from biosuite.core.sequence import reverse_complement
    assert codon_usage_table(seq_for_frames, frame=frame) == \
        codon_usage_table(reverse_complement(seq_for_frames), frame=-frame)


@pytest.fixture()
def seq_for_frames():
    return "AATGAAATTGCCCGGGTA"


# ── the synonymous case ──────────────────────────────────────────────────────

def test_two_codons_for_one_amino_acid_share_one_key():
    """TTG and CTC both code leucine, so `codon_per_aa` must collect them under
    'L' instead of the second overwriting the first."""
    result = codon_usage_table("TTGCTC")
    assert result["total_codons"] == 2
    assert set(result["codon_per_aa"]["L"]) == {"TTG", "CTC"}
    assert result["codon_per_aa"]["L"] == {"TTG": 50.0, "CTC": 50.0}
    assert result["amino_acids"]["L"] == 2


def test_percentages_of_a_synonymous_pair_are_taken_over_the_whole_cds():
    result = codon_usage_table("TTGCTCTTTAAA")
    assert result["total_codons"] == 4
    assert result["codon_per_aa"]["L"] == pytest.approx({"TTG": 25.0, "CTC": 25.0})
    assert result["codon_usage"]["TTG"] == 25.0


def test_a_sequence_of_unrecognised_codons_counts_zero_not_one():
    """'NNN' is not a codon; the divide-by-zero guard must not leak into the
    reported total."""
    result = codon_usage_table("NNNNNN")
    assert result["total_codons"] == 0
    assert result["codon_usage"] == {}
    assert result["codon_per_aa"] == {}
    assert result["amino_acids"] == {"?": 2}


def test_a_trailing_partial_codon_is_not_counted_but_is_translated():
    result = codon_usage_table("ATGAA")
    assert result["total_codons"] == 1
    assert "ATG" in result["codon_usage"]


# ── complexity window validation ────────────────────────────────────────────

@pytest.mark.parametrize("window", [0, 1, -2, 2.5, True, "20"])
def test_an_invalid_complexity_window_is_refused_by_name(window):
    with pytest.raises(ValueError, match=r"window must be an integer >= 2"):
        sequence_complexity("ATGCATGC", window=window)


def test_a_repeating_sequence_is_low_complexity_and_a_mixed_one_is_not():
    repeats = sequence_complexity("AT" * 40, window=8)
    mixed = sequence_complexity("ATGCGTACCTGAGCTC" * 3, window=8)
    assert repeats["is_low_complexity"] is True
    assert repeats["average_complexity"] < mixed["average_complexity"]
    assert mixed["is_low_complexity"] is False
    assert isinstance(repeats["is_low_complexity"], bool)


def test_a_window_longer_than_the_sequence_is_measured_over_its_whole_length():
    """A 20-base window never fit a 4-base sequence, so the loop produced no
    regions and the function reported 0.0 / is_low_complexity True — a diverse
    molecule declared an artifact."""
    result = sequence_complexity("ATGC", window=20)
    assert result["regions"] == [{"position": 0, "complexity": 1.0}]
    assert result["average_complexity"] == pytest.approx(1.0)
    assert result["is_low_complexity"] is False
    repeats = sequence_complexity("ATAT", window=20)
    assert repeats["regions"] == [{"position": 0, "complexity": 0.6667}]
    assert repeats["average_complexity"] < result["average_complexity"], \
        "the dinucleotide repeat must score below the molecule with three distinct ones"


def test_regions_are_stepped_by_half_a_window():
    result = sequence_complexity("ACGT" * 10, window=8)
    positions = [region["position"] for region in result["regions"]]
    assert positions[:3] == [0, 4, 8]
    assert all(positions[i] < positions[i + 1] for i in range(len(positions) - 1))


# ── k-mer composition guards ────────────────────────────────────────────────

@pytest.mark.parametrize("k", [0, -1, 1.5, True, "3"])
def test_an_invalid_k_is_refused(k):
    with pytest.raises(ValueError, match=r"k must be a positive integer"):
        kmer_composition("ATGCATGC", k=k)


def test_k_of_one_counts_each_base_once_per_position():
    counts = kmer_composition("AAGC", k=1)
    assert counts["A"]["count"] == 2
    assert counts["A"]["frequency"] == pytest.approx(0.5)
    assert sum(c["count"] for c in counts.values()) == 4


def test_frequencies_sum_to_one():
    counts = kmer_composition("ATGCGTACGATCG", k=3)
    assert sum(c["frequency"] for c in counts.values()) == pytest.approx(1.0, abs=1e-6)
