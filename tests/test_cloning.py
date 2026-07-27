"""Tests for biosuite.core.cloning module — restriction enzymes, primers, PCR."""
import pytest


class TestRestrictionSites:
    """Tests for find_restriction_sites()."""

    def test_find_sites(self):
        from biosuite.core.cloning import find_restriction_sites
        # EcoRI site: GAATTC
        sites = find_restriction_sites("ATCGAATTCGATCG")
        assert isinstance(sites, dict)

    def test_find_sites_empty(self):
        from biosuite.core.cloning import find_restriction_sites
        sites = find_restriction_sites("AAAA")
        assert isinstance(sites, dict)


class TestDigestion:
    """Tests for simulate_digestion()."""

    def test_digestion_returns_result(self):
        from biosuite.core.cloning import simulate_digestion
        seq = "ATCG" * 100 + "GAATTC" + "ATCG" * 100
        result = simulate_digestion(seq, enzyme="EcoRI")
        assert result is not None


class TestPCR:
    """Tests for simulate_pcr()."""

    def test_pcr_basic(self):
        from biosuite.core.cloning import simulate_pcr
        template = "ATCGATCGATCGATCGATCGATCG" + "A" * 50 + "GCTAGCTAGCTAGCTAGCTAGCTA"
        fwd_primer = "ATCGATCGATCG"
        rev_primer = "GCTAGCTAGCTAG"
        result = simulate_pcr(template, fwd_primer, rev_primer)
        assert result is not None


class TestPrimerDesign:
    """Tests for design_primers()."""

    def test_design_primers(self):
        from biosuite.core.cloning import design_primers
        target = "ATCGATCG" * 50 + "GAATTC" + "ATCGATCG" * 50
        result = design_primers(target)
        assert result is not None


class TestFormatReports:
    """Tests for format_*_report() functions."""

    def test_format_primer_report(self):
        from biosuite.core.cloning import format_primer_report
        result = format_primer_report({
            "forward": {"sequence": "ATCGATCG", "tm": 55.0},
            "reverse": {"sequence": "GCTAGCTA", "tm": 56.0}
        })
        assert isinstance(result, str)
