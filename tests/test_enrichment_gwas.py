"""
Tests for Enrichment analysis and GWAS modules.
"""
import pytest
import numpy as np
import pandas as pd


class TestGWAS:
    def test_generate_gwas_data(self):
        from biosuite.core.gwas import generate_gwas_data
        data = generate_gwas_data(n_snps=100)
        assert len(data) > 0
        assert 'chrom' in data.columns
        assert 'pos' in data.columns
        assert 'snp_id' in data.columns

    def test_run_gwas(self):
        from biosuite.core.gwas import generate_gwas_data, run_gwas
        data = generate_gwas_data(n_snps=100)
        results = run_gwas(data)
        assert len(results) > 0
        assert 'p_value' in results.columns
        assert 'p_adjusted' in results.columns

    def test_detect_lead_snps(self):
        from biosuite.core.gwas import generate_gwas_data, run_gwas, detect_lead_snps
        data = generate_gwas_data(n_snps=200)
        results = run_gwas(data)
        leads = detect_lead_snps(results)
        assert isinstance(leads, pd.DataFrame)

    def test_format_report(self):
        from biosuite.core.gwas import (generate_gwas_data, run_gwas,
                                           detect_lead_snps, format_gwas_report)
        data = generate_gwas_data(n_snps=100)
        results = run_gwas(data)
        leads = detect_lead_snps(results)
        report = format_gwas_report(results, leads)
        assert isinstance(report, str)
        assert len(report) > 0


class TestEnrichment:
    def test_ora_basic(self):
        from biosuite.core.enrichment import run_ora
        genes = ['BRCA1', 'TP53', 'MYC', 'EGFR', 'PTEN']
        result = run_ora(genes, ontology='BP')
        assert hasattr(result, 'results')
        assert isinstance(result.results, list)

    def test_format_enrichment_report(self):
        from biosuite.core.enrichment import run_ora, format_enrichment_report
        genes = ['BRCA1', 'TP53', 'MYC', 'EGFR', 'PTEN']
        result = run_ora(genes, ontology='BP')
        report = format_enrichment_report(result)
        assert isinstance(report, str)
