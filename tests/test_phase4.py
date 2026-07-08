"""
Unit tests for Phase 4 modules:
  - workflow/pipeline.py
  - workflow/batch.py
  - workflow/report.py
  - go_browser.py
  - pathway_viz.py
  - gwas.py
  - epitope.py
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─── Pipeline ────────────────────────────────────────────────────────────────

class TestPipeline:
    def test_pipeline_create_and_run(self):
        from biosuite.core.workflow.pipeline import Pipeline
        p = Pipeline("test")
        p.add_step("step1", lambda: 42)
        p.run()
        assert p.results["step1"] == 42

    def test_pipeline_summary(self):
        from biosuite.core.workflow.pipeline import Pipeline
        p = Pipeline("test")
        p.add_step("s1", lambda: "done")
        p.run()
        summary = p.summary()
        assert "test" in summary
        assert "Done" in summary

    def test_pipeline_stop_on_error(self):
        from biosuite.core.workflow.pipeline import Pipeline
        p = Pipeline("test")
        p.add_step("fail", lambda: 1/0)
        p.add_step("after", lambda: "ok")
        p.run(stop_on_error=True)
        assert "after" not in p.results

    def test_pipeline_context(self):
        from biosuite.core.workflow.pipeline import Pipeline
        p = Pipeline("test")
        p.set_context(val=10)
        p.add_step("double", lambda val: val * 2, kwargs={})
        p.run()
        assert p.results["double"] == 20

    def test_pipeline_to_dict(self):
        from biosuite.core.workflow.pipeline import Pipeline
        p = Pipeline("test")
        p.add_step("s1", lambda: 1)
        p.run()
        d = p.to_dict()
        assert d["name"] == "test"
        assert len(d["steps"]) == 1

    def test_pipeline_report(self):
        from biosuite.core.workflow.pipeline import Pipeline, format_pipeline_report
        p = Pipeline("test")
        p.add_step("s1", lambda: 1)
        p.run()
        report = format_pipeline_report(p)
        assert "Pipeline" in report


# ─── Batch Processor ─────────────────────────────────────────────────────────

class TestBatchProcessor:
    def test_batch_run(self):
        from biosuite.core.workflow.batch import BatchProcessor
        bp = BatchProcessor("test")
        bp.add_samples(["s1", "s2", "s3"], lambda sid: f"done_{sid}")
        bp.run(max_workers=1)
        results = bp.get_results()
        assert results["s1"] == "done_s1"
        assert results["s2"] == "done_s2"

    def test_batch_summary(self):
        from biosuite.core.workflow.batch import BatchProcessor
        bp = BatchProcessor("test")
        bp.add_samples(["s1"], lambda sid: sid)
        bp.run(max_workers=1)
        summary = bp.summary()
        assert "test" in summary

    def test_batch_failures(self):
        from biosuite.core.workflow.batch import BatchProcessor
        def bad_func(sid):
            if sid == "bad":
                raise ValueError("fail")
            return sid
        bp = BatchProcessor("test")
        bp.add_samples(["good", "bad"], bad_func)
        bp.run(max_workers=1)
        failures = bp.get_failures()
        assert len(failures) == 1
        assert failures[0][0] == "bad"

    def test_batch_to_dict(self):
        from biosuite.core.workflow.batch import BatchProcessor
        bp = BatchProcessor("test")
        bp.add_samples(["s1", "s2"], lambda sid: sid)
        bp.run(max_workers=1)
        d = bp.to_dict()
        assert d["done"] == 2
        assert d["failed"] == 0

    def test_batch_report(self):
        from biosuite.core.workflow.batch import BatchProcessor, format_batch_report
        bp = BatchProcessor("test")
        bp.add_samples(["s1"], lambda x: x)
        bp.run(max_workers=1)
        report = format_batch_report(bp)
        assert "Batch" in report


# ─── Report Generator ────────────────────────────────────────────────────────

class TestReportGenerator:
    def test_html_report(self):
        from biosuite.core.workflow.report import HTMLReport
        r = HTMLReport("Test Report")
        r.add_section("Intro", "<p>Hello</p>")
        r.add_stats({"Count": 42})
        html = r.to_html()
        assert "Test Report" in html
        assert "42" in html

    def test_report_save(self, tmp_path):
        from biosuite.core.workflow.report import HTMLReport
        r = HTMLReport("Test")
        r.add_section("Section 1", "<p>Content</p>")
        path = str(tmp_path / "report.html")
        r.save(path)
        assert os.path.exists(path)

    def test_report_with_plot(self):
        from biosuite.core.workflow.report import HTMLReport
        r = HTMLReport("Test")
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        r.add_plot(fig, "My Plot")
        html = r.to_html()
        assert "data:image/png;base64" in html

    def test_report_with_table(self):
        import pandas as pd
        from biosuite.core.workflow.report import HTMLReport
        r = HTMLReport("Test")
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        r.add_table(df, "My Table")
        html = r.to_html()
        assert "My Table" in html

    def test_generate_pipeline_report(self, tmp_path):
        from biosuite.core.workflow.pipeline import Pipeline
        from biosuite.core.workflow.report import generate_pipeline_report
        p = Pipeline("test")
        p.add_step("s1", lambda: 1)
        p.run()
        path = str(tmp_path / "pipeline.html")
        generate_pipeline_report(p, path)
        assert os.path.exists(path)

    def test_generate_batch_report(self, tmp_path):
        from biosuite.core.workflow.batch import BatchProcessor
        from biosuite.core.workflow.report import generate_batch_report
        bp = BatchProcessor("test")
        bp.add_samples(["s1"], lambda x: x)
        bp.run(max_workers=1)
        path = str(tmp_path / "batch.html")
        generate_batch_report(bp, path)
        assert os.path.exists(path)


# ─── GO Browser ──────────────────────────────────────────────────────────────

class TestGOBrowser:
    def test_builtin_terms(self):
        from biosuite.core.go_browser import GOBrowser
        go = GOBrowser()
        assert len(go.terms) > 0
        assert "GO:0008150" in go.terms

    def test_search(self):
        from biosuite.core.go_browser import GOBrowser
        go = GOBrowser()
        results = go.search("kinase")
        assert len(results) > 0
        assert any("kinase" in t.name.lower() for t in results)

    def test_get_term(self):
        from biosuite.core.go_browser import GOBrowser
        go = GOBrowser()
        term = go.get_term("GO:0004672")
        assert term is not None
        assert term.name == "protein kinase activity"

    def test_get_parents(self):
        from biosuite.core.go_browser import GOBrowser
        go = GOBrowser()
        parents = go.get_parents("GO:0004672")
        assert len(parents) > 0
        assert any(p.name == "catalytic activity" for p in parents)

    def test_get_children(self):
        from biosuite.core.go_browser import GOBrowser
        go = GOBrowser()
        children = go.get_children("GO:0003824")
        assert len(children) > 0

    def test_get_namespace_terms(self):
        from biosuite.core.go_browser import GOBrowser
        go = GOBrowser()
        bp = go.get_namespace_terms("BP")
        assert len(bp) > 0
        assert all(t.namespace == "BP" for t in bp)

    def test_format_results(self):
        from biosuite.core.go_browser import GOBrowser, format_go_results
        go = GOBrowser()
        results = go.search("kinase")
        formatted = format_go_results(results)
        assert "GO ID" in formatted


# ─── Pathway Visualization ───────────────────────────────────────────────────

class TestPathwayViz:
    def test_create_pathway(self):
        from biosuite.core.pathway_viz import PathwayMap
        pm = PathwayMap("test")
        pm.add_node("A", "GeneA", 0, 0)
        pm.add_node("B", "GeneB", 3, 0)
        pm.add_edge("A", "B")
        assert len(pm.nodes) == 2
        assert len(pm.edges) == 1

    def test_kegg_pathway(self):
        from biosuite.core.pathway_viz import create_kegg_style_pathway
        pm = create_kegg_style_pathway()
        assert len(pm.nodes) > 5
        assert len(pm.edges) > 5

    def test_draw_pathway(self):
        from biosuite.core.pathway_viz import create_kegg_style_pathway, draw_pathway
        pm = create_kegg_style_pathway()
        fig = draw_pathway(pm)
        assert fig is not None
        plt.close(fig)

    def test_custom_pathway(self):
        from biosuite.core.pathway_viz import create_custom_pathway, draw_pathway
        pm = create_custom_pathway(["A", "B", "C", "D"])
        fig = draw_pathway(pm)
        assert fig is not None
        plt.close(fig)

    def test_format_report(self):
        from biosuite.core.pathway_viz import create_kegg_style_pathway, format_pathway_report
        pm = create_kegg_style_pathway()
        report = format_pathway_report(pm)
        assert "Pathway" in report
        assert "Nodes" in report

    def test_set_expression(self):
        from biosuite.core.pathway_viz import PathwayMap
        pm = PathwayMap("test")
        pm.add_node("A", "GeneA")
        pm.set_expression({"A": 1.5})
        assert pm.nodes["A"].color is not None


# ─── GWAS ────────────────────────────────────────────────────────────────────

class TestGWAS:
    def test_generate_data(self):
        from biosuite.core.gwas import generate_gwas_data
        data = generate_gwas_data(n_snps=100, n_chromosomes=1)
        assert len(data) > 0
        assert "chrom" in data.columns

    def test_run_gwas(self):
        from biosuite.core.gwas import run_gwas, generate_gwas_data
        data = generate_gwas_data(n_snps=200)
        results = run_gwas(data)
        assert "p_value" in results.columns
        assert "neg_log10" in results.columns
        assert len(results) > 0

    def test_detect_lead_snps(self):
        from biosuite.core.gwas import run_gwas, detect_lead_snps, generate_gwas_data
        data = generate_gwas_data(n_snps=500)
        results = run_gwas(data)
        leads = detect_lead_snps(results, p_threshold=0.05)
        assert isinstance(leads, type(results))

    def test_format_report(self):
        from biosuite.core.gwas import run_gwas, generate_gwas_data, format_gwas_report
        data = generate_gwas_data(n_snps=200)
        results = run_gwas(data)
        report = format_gwas_report(results)
        assert "GWAS" in report
        assert "SNPs" in report

    def test_chi_squared(self):
        from biosuite.core.gwas import gwas_chi_squared
        res = gwas_chi_squared(100, 150, 200, 200)
        assert "p_value" in res
        assert "odds_ratio" in res
        assert 0 < res["p_value"] <= 1

    def test_bh_correction(self):
        from biosuite.core.gwas import _benjamini_hochberg
        p = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        adj = _benjamini_hochberg(p)
        assert len(adj) == len(p)
        assert all(0 <= a <= 1 for a in adj)


# ─── Epitope Prediction ─────────────────────────────────────────────────────

class TestEpitope:
    def test_t_cell_prediction(self):
        from biosuite.core.epitope import predict_t_cell_epitopes
        seq = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS"
        results = predict_t_cell_epitopes(seq, mhc_type="A0201")
        assert len(results) > 0
        assert all(8 <= len(e.peptide) <= 11 for e in results)

    def test_b_cell_prediction(self):
        from biosuite.core.epitope import predict_b_cell_epitopes
        seq = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS"
        results = predict_b_cell_epitopes(seq)
        assert len(results) > 0

    def test_linear_epitopes(self):
        from biosuite.core.epitope import predict_linear_epitopes
        seq = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS"
        results = predict_linear_epitopes(seq)
        assert len(results) > 0

    def test_cleavage_sites(self):
        from biosuite.core.epitope import cleavage_site_prediction
        sites = cleavage_site_prediction("MKWVTFISLLFLFSSAYSR")
        assert len(sites) > 0
        assert all("position" in s for s in sites)

    def test_format_report(self):
        from biosuite.core.epitope import (predict_t_cell_epitopes, predict_b_cell_epitopes,
                                              format_epitope_report)
        seq = "MKWVTFISLLFLFSSAYSRGVFRRDAHK"
        tc = predict_t_cell_epitopes(seq)
        bc = predict_b_cell_epitopes(seq)
        report = format_epitope_report(tc, bc, "test")
        assert "T-cell" in report
        assert "B-cell" in report

    def test_epitope_result_dict(self):
        from biosuite.core.epitope import EpitopeResult
        e = EpitopeResult("AAAA", 0, 4, 0.8, "T-cell")
        d = e.to_dict()
        assert d["peptide"] == "AAAA"
        assert d["score"] == 0.8


# ─── CLI Integration ─────────────────────────────────────────────────────────

class TestPhase4CLI:
    def test_cli_menu_compiles(self):
        import importlib
        mod = importlib.import_module('biosuite.cli.menu')
        assert hasattr(mod, 'print_menu')

    def test_menu_has_phase4_options(self):
        import io, contextlib
        from biosuite.cli.menu import print_menu
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_menu()
        output = f.getvalue()
        assert '92' in output
        assert '93' in output
        assert '94' in output
        assert '95' in output
        assert '96' in output
        assert '97' in output
        assert '98' in output
        assert 'Pipeline' in output
        assert 'GWAS' in output
        assert 'Epitope' in output
        assert 'GO Browser' in output


# ─── GUI Integration ─────────────────────────────────────────────────────────

class TestPhase4GUI:
    def test_gui_has_frame_builders(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert hasattr(BioSuiteApp, '_build_pipeline_frame')
        assert hasattr(BioSuiteApp, '_build_batch_frame')
        assert hasattr(BioSuiteApp, '_build_gobrowser_frame')
        assert hasattr(BioSuiteApp, '_build_pathway_frame')
        assert hasattr(BioSuiteApp, '_build_gwas_frame')
        assert hasattr(BioSuiteApp, '_build_epitope_frame')

    def test_gui_has_action_methods(self):
        from biosuite.gui.main_window import BioSuiteApp
        assert hasattr(BioSuiteApp, '_run_pipeline')
        assert hasattr(BioSuiteApp, '_run_batch')
        assert hasattr(BioSuiteApp, '_go_search')
        assert hasattr(BioSuiteApp, '_draw_pathway')
        assert hasattr(BioSuiteApp, '_gwas_demo')
        assert hasattr(BioSuiteApp, '_run_epitope')
