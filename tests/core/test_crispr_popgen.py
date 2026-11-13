"""
Tests for the CRISPR guide design and population genetics modules.

The epitope predictions this file used to hold are covered, with real assertions, by
the dedicated test_epitope_* files.
"""
import numpy as np


class TestCRISPR:
    def test_design_guides_basic(self):
        from biosuite.core.crispr import design_guides
        seq = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        result = design_guides(seq, pam_type="SpCas9")
        assert result.num_guides >= 0
        assert hasattr(result, 'guides')

    def test_pam_types(self):
        from biosuite.core.crispr import design_guides
        no_sites = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        positives = {
            "SpCas9": no_sites + "AGG",
            "SaCas9": no_sites + "AAGAAT",
            "Cas12a": "TTTA" + "ATCG" * 5,  # Cas12a PAM sits upstream of its guide
        }
        for pam, seq in positives.items():
            empty = design_guides(no_sites, pam_type=pam)
            assert empty.engine == "builtin"
            assert empty.num_guides == 0
            found = design_guides(seq, pam_type=pam)
            assert found.num_guides >= 1
            assert found.guides[0].pam

    def test_empty_sequence(self):
        from biosuite.core.crispr import design_guides
        result = design_guides("", pam_type="SpCas9")
        assert result.num_guides == 0

    def test_format_report(self):
        from biosuite.core.crispr import design_guides, format_crispr_report
        seq = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        result = design_guides(seq, pam_type="SpCas9")
        report = format_crispr_report(result)
        assert isinstance(report, str)
        assert len(report) > 0


class TestPopulationGenetics:
    def test_full_analysis(self):
        from biosuite.core.popgen import full_analysis
        np.random.seed(42)
        matrix = np.random.randint(0, 3, size=(10, 20))
        report = full_analysis(matrix)
        assert report.num_sites == 20

    def test_format_report(self):
        from biosuite.core.popgen import format_popgen_report, full_analysis
        np.random.seed(42)
        matrix = np.random.randint(0, 3, size=(10, 20))
        report = full_analysis(matrix)
        text = format_popgen_report(report)
        assert isinstance(text, str)
        assert "Sites" in text
