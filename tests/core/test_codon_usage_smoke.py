"""Smoke tests pinning codon_usage.py behaviour (reviewed clean)."""
import pytest

from biosuite.core.codon_usage import (
    codon_usage_table, kmer_composition, sequence_complexity,
    format_codon_usage, format_kmer_composition,
)


def test_codon_usage_counts_frame_correctly():
    seq = "X" + "ATGAAATTT" + "GG"  # frame=2: ATG AAA TTT
    res = codon_usage_table(seq, frame=2)
    assert res['codon_usage'].get('ATG') == pytest.approx(100 / 3, abs=0.01)
    assert res['total_codons'] == 3
    assert res['amino_acids'] == {'M': 1, 'K': 1, 'F': 1}


def test_codon_usage_negative_frame():
    seq = "AAACCC"  # revcomp = GGGTTT
    fwd = codon_usage_table("AAACCC", frame=1)
    rev = codon_usage_table("AAACCC", frame=-1)
    assert set(rev['codon_usage']) != set(fwd['codon_usage']) or rev['codon_usage'] != fwd['codon_usage']


def test_kmer_composition_normalized():
    res = kmer_composition("AAAA", k=2)
    assert res['AA'] == {'count': 3, 'frequency': 1.0}


def test_sequence_complexity_bounds():
    low = sequence_complexity("A" * 100, window=20)
    assert low['is_low_complexity'] is True
    import random
    random.seed(0)
    high = sequence_complexity(''.join(random.choice('ACGT') for _ in range(400)), window=20)
    assert high['average_complexity'] > low['average_complexity']
    assert 0 <= high['average_complexity'] <= 1


def test_formatters_no_crash():
    res = codon_usage_table("ATGAAATTTTAA")
    txt = format_codon_usage(res)
    assert "Codon Usage" in txt
    km = kmer_composition("ACGTACGTACGT")
    assert "K-mer" in format_kmer_composition(km)
