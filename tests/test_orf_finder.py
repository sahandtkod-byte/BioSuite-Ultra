"""
Unit tests for biosuite.core.orf_finder module.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biosuite.core.orf_finder import (
    find_orfs, find_restriction_sites, design_primers,
    _reverse_complement, _calculate_tm,
    format_orf_results, format_restriction_sites, format_primers
)


class TestORFFinder:
    def test_simple_orf(self):
        seq = "TTTATGAAATTTTAA"  # ATG AAA TTT TAA
        orfs = find_orfs(seq, min_length=1)
        assert len(orfs) >= 1
        assert any(o.has_start_codon for o in orfs)

    def test_no_orf(self):
        seq = "TTTTTTTTTT"
        orfs = find_orfs(seq, min_length=5)
        assert len(orfs) == 0

    def test_multiple_frames(self):
        seq = "ATGAAATTTTAA"  # Frame 1: ATG AAA TTT TAA
        orfs = find_orfs(seq, min_length=1)
        assert len(orfs) >= 1
        assert orfs[0].frame == 1

    def test_translation(self):
        seq = "ATGAAATTTTAA"
        orfs = find_orfs(seq, min_length=1)
        # ATG=M, AAA=K, TTT=F, TAA=*
        assert orfs[0].protein == "MKF"


class TestRestrictionSites:
    def test_ecori(self):
        seq = "GAATTC"
        sites = find_restriction_sites(seq, enzymes=['EcoRI'])
        assert len(sites) >= 1
        assert sites[0].enzyme == 'EcoRI'

    def test_multiple_sites(self):
        seq = "GAATTCGAATTC"
        sites = find_restriction_sites(seq, enzymes=['EcoRI'])
        assert len(sites) == 2

    def test_no_sites(self):
        seq = "ACGTACGT"
        sites = find_restriction_sites(seq, enzymes=['EcoRI'])
        assert len(sites) == 0


class TestPrimerDesign:
    def test_basic_design(self):
        seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        fwd, rev = design_primers(seq, min_tm=40, max_tm=80)
        assert fwd is not None or rev is not None

    def test_primer_properties(self):
        seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        fwd, rev = design_primers(seq)
        if fwd:
            assert fwd.gc_content > 0
            assert fwd.tm > 0
            assert len(fwd.sequence) > 0


class TestHelpers:
    def test_reverse_complement(self):
        assert _reverse_complement("ACGT") == "ACGT"
        assert _reverse_complement("AAAA") == "TTTT"
        assert _reverse_complement("AATT") == "AATT"

    def test_tm_calculation(self):
        tm = _calculate_tm("ACGTACGTACGTACGT")
        assert tm > 40
        assert tm < 80

    def test_format_results(self):
        formatted = format_orf_results([])
        assert 'ORFs' in formatted
        assert 'Found 0' in formatted
