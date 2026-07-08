"""
Tests for CRISPR, Population Genetics, and Epitope prediction modules.
"""
import pytest
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
        seq = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        for pam in ["SpCas9", "SaCas9", "Cas12a"]:
            result = design_guides(seq, pam_type=pam)
            assert hasattr(result, 'num_guides')

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
        from biosuite.core.popgen import full_analysis, format_popgen_report
        np.random.seed(42)
        matrix = np.random.randint(0, 3, size=(10, 20))
        report = full_analysis(matrix)
        text = format_popgen_report(report)
        assert isinstance(text, str)
        assert "Sites" in text


class TestEpitope:
    def test_t_cell_epitopes(self):
        from biosuite.core.epitope import predict_t_cell_epitopes
        seq = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS"
        result = predict_t_cell_epitopes(seq, mhc_type="A0201")
        assert isinstance(result, list)

    def test_b_cell_epitopes(self):
        from biosuite.core.epitope import predict_b_cell_epitopes
        seq = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS"
        result = predict_b_cell_epitopes(seq)
        assert isinstance(result, list)

    def test_format_report(self):
        from biosuite.core.epitope import (predict_t_cell_epitopes,
                                              predict_b_cell_epitopes,
                                              format_epitope_report)
        seq = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS"
        tc = predict_t_cell_epitopes(seq, mhc_type="A0201")
        bc = predict_b_cell_epitopes(seq)
        report = format_epitope_report(tc, bc, "test_protein")
        assert isinstance(report, str)
        assert len(report) > 0
