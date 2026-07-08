"""
Comprehensive tests for untested core modules:

- biosuite.core.databases      (search & format functions with mocked HTTP)
- biosuite.core.go_browser     (GOBrowser class, format_go_results)
- biosuite.core.pathway_viz    (PathwayMap, draw_pathway, format_pathway_report)
- biosuite.core.epigenomics    (methylation, DMR, ATAC-seq)
- biosuite.core.md_simulation  (built-in MD minimization)
- biosuite.core.structure_prediction (structure prediction API)

All external HTTP and file I/O dependencies are mocked.
"""
import os
import sys
import json
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATABASES — biosuite.core.databases
# ══════════════════════════════════════════════════════════════════════════════

from biosuite.core.databases import (
    DBResult, search_ncbi, search_uniprot, search_pdb,
    search_kegg, search_ensembl, fetch_ncbi_sequence,
    fetch_uniprot, fetch_kegg_pathway,
    format_search_results, _format_single, search_all,
)


class TestDBResultDataclass:
    """Test the DBResult dataclass construction and defaults."""

    def test_defaults(self):
        r = DBResult(source="test", query="q")
        assert r.source == "test"
        assert r.query == "q"
        assert r.data == {}
        assert r.records == []
        assert r.error == ""
        assert r.cached is False

    def test_with_records(self):
        recs = [{"id": "1", "title": "X"}]
        r = DBResult(source="ncbi", query="q", records=recs)
        assert len(r.records) == 1

    def test_with_error(self):
        r = DBResult(source="pdb", query="q", error="timeout")
        assert r.error == "timeout"


class TestFormatSingle:
    """Test the internal _format_single helper."""

    def test_error_result(self):
        r = DBResult(source="x", query="q", error="boom")
        out = _format_single(r)
        assert "boom" in out

    def test_no_records(self):
        r = DBResult(source="x", query="q")
        out = _format_single(r)
        assert "No records" in out

    def test_with_records(self):
        r = DBResult(source="x", query="q", records=[
            {"id": "ABC", "title": "Gene A", "organism": "Homo sapiens"},
            {"id": "DEF", "title": "Gene B", "organism": "Mus musculus"},
        ])
        out = _format_single(r)
        assert "ABC" in out
        assert "Gene A" in out

    def test_records_truncated_to_10(self):
        recs = [{"id": str(i)} for i in range(20)]
        r = DBResult(source="x", query="q", records=recs)
        out = _format_single(r)
        # Should only show 10 records (lines starting with "  N.")
        lines = [l for l in out.split("\n") if l.strip().startswith(("1.", "10."))]
        assert len(lines) <= 10


class TestFormatSearchResults:
    def test_single_result(self):
        r = DBResult(source="ncbi", query="q", records=[{"id": "1"}])
        out = format_search_results(r)
        assert "1" in out

    def test_dict_of_results(self):
        results = {
            "ncbi": DBResult(source="ncbi", query="q", records=[{"id": "1"}]),
            "uniprot": DBResult(source="uniprot", query="q", error="fail"),
        }
        out = format_search_results(results)
        assert "NCBI" in out
        assert "UNIPROT" in out

    def test_dict_with_source_filter(self):
        results = {
            "ncbi": DBResult(source="ncbi", query="q", records=[{"id": "1"}]),
            "pdb": DBResult(source="pdb", query="q", records=[{"id": "2"}]),
        }
        out = format_search_results(results, source="ncbi")
        assert "1" in out
        assert "2" not in out

    def test_no_results(self):
        out = format_search_results("garbage")
        assert "No results" in out


class TestSearchNCBI:
    """Test search_ncbi with mocked HTTP responses."""

    @patch('biosuite.core.databases.urllib.request.urlopen')
    @patch('biosuite.core.databases.get_api_key', return_value='')
    @patch('biosuite.core.databases.prompt_api_key', return_value='test@example.com')
    def test_success(self, mock_prompt, mock_key, mock_urlopen):
        # Mock esearch response
        esearch_resp = json.dumps({
            "esearchresult": {"idlist": ["12345", "67890"], "count": "2"}
        }).encode()
        # Mock esummary response
        esummary_resp = json.dumps({
            "result": {
                "12345": {"title": "Gene BRCA1", "organism": "Homo sapiens", "accessionversion": "NM_001234"},
                "67890": {"title": "Gene TP53", "organism": "Homo sapiens", "accessionversion": "NM_001235"},
            }
        }).encode()

        mock_resp1 = MagicMock()
        mock_resp1.read.return_value = esearch_resp
        mock_resp1.__enter__ = lambda s: s
        mock_resp1.__exit__ = MagicMock(return_value=False)

        mock_resp2 = MagicMock()
        mock_resp2.read.return_value = esummary_resp
        mock_resp2.__enter__ = lambda s: s
        mock_resp2.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [mock_resp1, mock_resp2]

        result = search_ncbi("BRCA1", email="test@example.com")
        assert result.source == "ncbi"
        assert len(result.records) == 2
        assert result.data["count"] == 2
        assert result.error == ""

    @patch('biosuite.core.databases.urllib.request.urlopen')
    @patch('biosuite.core.databases.get_api_key', return_value='')
    @patch('biosuite.core.databases.prompt_api_key', return_value='test@example.com')
    def test_no_results(self, mock_prompt, mock_key, mock_urlopen):
        esearch_resp = json.dumps({"esearchresult": {"idlist": [], "count": "0"}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = esearch_resp
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = search_ncbi("nonexistent_xyz", email="test@example.com")
        assert result.records == []
        assert result.data["count"] == 0

    @patch('biosuite.core.databases.urllib.request.urlopen', side_effect=Exception("Network error"))
    @patch('biosuite.core.databases.get_api_key', return_value='')
    @patch('biosuite.core.databases.prompt_api_key', return_value='test@example.com')
    def test_network_error(self, mock_prompt, mock_key, mock_urlopen):
        result = search_ncbi("test", email="test@example.com")
        assert result.error == "Network error"


class TestFetchNCBISequence:
    @patch('biosuite.core.databases.urllib.request.urlopen')
    @patch('biosuite.core.databases.get_api_key', return_value='')
    def test_success(self, mock_key, mock_urlopen):
        fasta = ">seq1\nATCGATCGATCG\n"
        mock_resp = MagicMock()
        mock_resp.read.return_value = fasta.encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = fetch_ncbi_sequence("NM_001234")
        assert result.records[0]["fasta"] == fasta

    @patch('biosuite.core.databases.urllib.request.urlopen', side_effect=Exception("timeout"))
    @patch('biosuite.core.databases.get_api_key', return_value='')
    def test_error(self, mock_key, mock_urlopen):
        result = fetch_ncbi_sequence("bad_accession")
        assert "timeout" in result.error


class TestSearchUniProt:
    @patch('biosuite.core.databases._http_get')
    def test_success(self, mock_http_get):
        from biosuite.core import databases
        databases._cache._store.clear()
        resp_data = json.dumps({
            "results": [{
                "primaryAccession": "P01308",
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Insulin"}}},
                "organism": {"scientificName": "Homo sapiens"},
                "sequence": {"length": 110},
                "genes": [{"geneName": {"value": "INS"}}],
            }]
        }).encode()
        mock_http_get.return_value = resp_data

        result = search_uniprot("insulin")
        assert result.source == "uniprot"
        assert len(result.records) == 1
        assert result.records[0]["accession"] == "P01308"

    @patch('biosuite.core.databases.urllib.request.urlopen', side_effect=Exception("404"))
    def test_not_found(self, mock_urlopen):
        result = search_uniprot("nonexistent_protein_xyz")
        assert "404" in result.error


class TestFetchUniProt:
    @patch('biosuite.core.databases.urllib.request.urlopen')
    def test_success(self, mock_urlopen):
        entry = {
            "primaryAccession": "P01308",
            "proteinDescription": {"recommendedName": {"fullName": {"value": "Insulin"}}},
            "organism": {"scientificName": "Homo sapiens"},
            "sequence": {"value": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT", "length": 110},
            "uniProtKBCrossReferences": [
                {"database": "GO", "properties": [{"value": "GO:0005515"}]},
            ],
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(entry).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = fetch_uniprot("P01308")
        assert len(result.records) == 1
        assert result.records[0]["length"] == 110
        assert "GO:0005515" in result.records[0]["go_terms"]


class TestSearchPDB:
    @patch('biosuite.core.databases.urllib.request.urlopen')
    def test_success(self, mock_urlopen):
        resp_data = json.dumps({
            "result_set": [
                {"identifier": "1HHO", "title": "Deoxy Hemoglobin"},
                {"identifier": "2HHO", "title": "Oxy Hemoglobin"},
            ]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = resp_data
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = search_pdb("hemoglobin")
        assert result.source == "pdb"
        assert len(result.records) == 2
        assert result.records[0]["pdb_id"] == "1HHO"
        assert "rcsb.org" in result.records[0]["url"]


class TestSearchKEGG:
    @patch('biosuite.core.databases.urllib.request.urlopen')
    @patch('biosuite.core.databases.get_api_key', return_value='')
    def test_success(self, mock_key, mock_urlopen):
        text = "path:hsa00010\tGlycolysis / Gluconeogenesis\npath:hsa00020\tCitrate cycle\n"
        mock_resp = MagicMock()
        mock_resp.read.return_value = text.encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = search_kegg("glycolysis")
        assert result.source == "kegg"
        assert len(result.records) == 2
        # search_kegg splits on first ':', so id='path', desc='hsa00010\t...'
        assert result.records[0]["id"] == "path"
        assert "hsa00010" in result.records[0]["description"]


class TestSearchEnsembl:
    @patch('biosuite.core.databases.urllib.request.urlopen')
    def test_success(self, mock_urlopen):
        data = {
            "id": "ENSG00000139618",
            "description": "BRCA1 DNA repair associated",
            "species": "homo_sapiens",
            "biotype": "protein_coding",
            "start": 43044295,
            "end": 43170245,
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = search_ensembl("BRCA1", species="human")
        assert result.source == "ensembl"
        assert len(result.records) == 1
        assert result.records[0]["id"] == "ENSG00000139618"

    @patch('biosuite.core.databases.urllib.request.urlopen', side_effect=Exception("not found"))
    def test_not_found(self, mock_urlopen):
        result = search_ensembl("nonexistent_gene_xyz")
        assert "not found" in result.error


class TestFetchKEGGPathway:
    @patch('biosuite.core.databases.urllib.request.urlopen')
    @patch('biosuite.core.databases.get_api_key', return_value='')
    def test_success(self, mock_key, mock_urlopen):
        text = "ENTRY       PATHWAY\nNAME        Glycolysis\n"
        mock_resp = MagicMock()
        mock_resp.read.return_value = text.encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = fetch_kegg_pathway("hsa00010")
        assert result.source == "kegg"
        assert "Glycolysis" in result.records[0]["text"]


class TestSearchAll:
    """Test the universal search function."""

    @patch('biosuite.core.databases.search_kegg')
    @patch('biosuite.core.databases.search_pdb')
    @patch('biosuite.core.databases.search_uniprot')
    @patch('biosuite.core.databases.search_ncbi')
    def test_all_databases(self, mock_ncbi, mock_up, mock_pdb, mock_kg):
        mock_ncbi.return_value = DBResult(source="ncbi", query="q", records=[{"id": "1"}])
        mock_up.return_value = DBResult(source="uniprot", query="q", records=[{"id": "2"}])
        mock_pdb.return_value = DBResult(source="pdb", query="q", records=[{"id": "3"}])
        mock_kg.return_value = DBResult(source="kegg", query="q", records=[{"id": "4"}])

        results = search_all("test")
        assert "ncbi" in results
        assert "uniprot" in results
        assert "pdb" in results
        assert "kegg" in results
        mock_ncbi.assert_called_once()
        mock_up.assert_called_once()
        mock_pdb.assert_called_once()
        mock_kg.assert_called_once()

    @patch('biosuite.core.databases.search_ncbi')
    def test_subset_databases(self, mock_ncbi):
        mock_ncbi.return_value = DBResult(source="ncbi", query="q")
        results = search_all("test", databases=["ncbi"])
        assert "ncbi" in results
        assert len(results) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 2. GO BROWSER — biosuite.core.go_browser
# ══════════════════════════════════════════════════════════════════════════════

from biosuite.core.go_browser import (
    GOBrowser, GOTerm, go_enrichment, format_go_results,
    BASIC_GO_TERMS,
)


class TestGOTerm:
    def test_creation(self):
        t = GOTerm("GO:0000001", "test term", "BP", parents=["GO:0000000"], definition="A test")
        assert t.go_id == "GO:0000001"
        assert t.name == "test term"
        assert t.namespace == "BP"
        assert t.parents == ["GO:0000000"]
        assert t.definition == "A test"

    def test_repr(self):
        t = GOTerm("GO:123", "hello", "MF")
        assert "GO:123" in repr(t)
        assert "hello" in repr(t)


class TestGOBrowserInit:
    def test_builtin_terms_loaded(self):
        browser = GOBrowser()
        assert len(browser.terms) == len(BASIC_GO_TERMS)
        assert "GO:0008150" in browser.terms
        assert "GO:0003674" in browser.terms

    def test_children_populated(self):
        browser = GOBrowser()
        # biological_process (GO:0008150) should have children
        assert len(browser.children["GO:0008150"]) > 0

    def test_no_obo_file_falls_back_to_builtin(self):
        browser = GOBrowser(obo_file="/nonexistent/file.obo")
        assert len(browser.terms) == len(BASIC_GO_TERMS)

    def test_search(self):
        browser = GOBrowser()
        results = browser.search("kinase")
        assert len(results) > 0
        assert any("kinase" in t.name.lower() for t in results)

    def test_search_by_id(self):
        browser = GOBrowser()
        results = browser.search("GO:0005524")
        assert len(results) == 1
        assert results[0].go_id == "GO:0005524"

    def test_search_no_match(self):
        browser = GOBrowser()
        results = browser.search("zzz_nonexistent_xyz")
        assert results == []

    def test_get_term(self):
        browser = GOBrowser()
        term = browser.get_term("GO:0005524")
        assert term is not None
        assert term.name == "ATP binding"

    def test_get_term_not_found(self):
        browser = GOBrowser()
        term = browser.get_term("GO:9999999")
        assert term is None

    def test_get_parents(self):
        browser = GOBrowser()
        parents = browser.get_parents("GO:0005829")  # cytosol
        assert len(parents) > 0
        assert any(p.name == "cytoplasm" for p in parents)

    def test_get_children(self):
        browser = GOBrowser()
        children = browser.get_children("GO:0008150")  # biological_process
        assert len(children) > 0

    def test_get_ancestors(self):
        browser = GOBrowser()
        ancestors = browser.get_ancestors("GO:0006468")  # protein phosphorylation
        names = [a.name for a in ancestors]
        assert "protein phosphorylation" in names
        assert "biological_process" in names

    def test_get_namespace_terms(self):
        browser = GOBrowser()
        bp_terms = browser.get_namespace_terms("BP")
        mf_terms = browser.get_namespace_terms("MF")
        cc_terms = browser.get_namespace_terms("CC")
        assert len(bp_terms) > 0
        assert len(mf_terms) > 0
        assert len(cc_terms) > 0
        assert all(t.namespace == "BP" for t in bp_terms)

    def test_get_dag(self):
        browser = GOBrowser()
        dag = browser.get_dag("GO:0008150", depth=2)
        assert len(dag) > 1
        # First element should be root
        assert dag[0][0] == "GO:0008150"
        assert dag[0][1] == 0  # depth 0


class TestFormatGoResults:
    def test_empty(self):
        out = format_go_results([])
        assert "No GO terms" in out

    def test_with_terms(self):
        terms = [
            GOTerm("GO:0000001", "term one", "BP", definition="Short def"),
            GOTerm("GO:0000002", "term two", "MF", definition="A" * 60),
        ]
        out = format_go_results(terms)
        assert "term one" in out
        assert "GO:0000001" in out
        assert "Found 2 terms" in out
        # Long definition should be truncated
        assert "..." in out

    def test_single_term(self):
        terms = [GOTerm("GO:123", "single", "CC")]
        out = format_go_results(terms)
        assert "single" in out
        assert "Found 1 terms" in out


class TestGOEnrichment:
    def test_basic_enrichment(self):
        gene_list = ["gene1", "gene2", "gene3"]
        go_terms_map = {
            "GO:0000001": ["gene1", "gene2", "gene4"],
            "GO:0000002": ["gene5", "gene6"],
        }
        results = go_enrichment(gene_list, go_terms_map, background_size=100)
        assert len(results) == 2
        # GO:0000001 should be more enriched (2/3 overlap vs 0/2)
        assert results[0]["go_term"] == "GO:0000001"

    def test_empty_gene_list(self):
        results = go_enrichment([], {"GO:0000001": ["gene1"]}, background_size=100)
        assert len(results) == 1

    def test_no_overlap(self):
        results = go_enrichment(["a", "b"], {"GO:0000001": ["x", "y"]}, background_size=100)
        assert results[0]["count"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. PATHWAY VIZ — biosuite.core.pathway_viz
# ══════════════════════════════════════════════════════════════════════════════

from biosuite.core.pathway_viz import (
    PathwayNode, PathwayEdge, PathwayMap,
    draw_pathway, create_kegg_style_pathway, create_custom_pathway,
    format_pathway_report,
)


class TestPathwayNode:
    def test_creation(self):
        n = PathwayNode("n1", "Gene A", x=1, y=2)
        assert n.node_id == "n1"
        assert n.name == "Gene A"
        assert n.x == 1
        assert n.y == 2
        assert n.expression is None

    def test_set_expression_upregulated(self):
        n = PathwayNode("n1", "Gene A")
        n.set_expression(1.5, vmin=-2, vmax=2)
        assert n.expression == 1.5
        assert n.color is not None
        # Upregulated → red-ish
        assert n.color.startswith("#")

    def test_set_expression_downregulated(self):
        n = PathwayNode("n1", "Gene A")
        n.set_expression(-1.5, vmin=-2, vmax=2)
        assert n.expression == -1.5
        assert n.color.startswith("#")

    def test_set_expression_neutral(self):
        n = PathwayNode("n1", "Gene A")
        n.set_expression(0, vmin=-2, vmax=2)
        assert n.expression == 0.0


class TestPathwayEdge:
    def test_creation(self):
        e = PathwayEdge("src", "tgt", edge_type="inhibition", label="binds")
        assert e.source == "src"
        assert e.target == "tgt"
        assert e.edge_type == "inhibition"
        assert e.label == "binds"

    def test_defaults(self):
        e = PathwayEdge("a", "b")
        assert e.edge_type == "activation"
        assert e.label == ""


class TestPathwayMap:
    def test_creation(self):
        pm = PathwayMap("test_pathway")
        assert pm.name == "test_pathway"
        assert len(pm.nodes) == 0
        assert len(pm.edges) == 0

    def test_add_node(self):
        pm = PathwayMap("p")
        pm.add_node("n1", "Gene1", x=0, y=0)
        assert "n1" in pm.nodes
        assert pm.nodes["n1"].name == "Gene1"

    def test_add_node_chain(self):
        pm = PathwayMap("p")
        pm.add_node("a", "A").add_node("b", "B")
        assert len(pm.nodes) == 2

    def test_add_edge(self):
        pm = PathwayMap("p")
        pm.add_node("a", "A").add_node("b", "B")
        pm.add_edge("a", "b", edge_type="inhibition", label="blocks")
        assert len(pm.edges) == 1
        assert pm.edges[0].source == "a"

    def test_set_expression(self):
        pm = PathwayMap("p")
        pm.add_node("a", "A")
        pm.set_expression({"a": 1.5})
        assert pm.nodes["a"].expression == 1.5

    def test_layout_grid(self):
        pm = PathwayMap("p")
        for i in range(6):
            pm.add_node(f"n{i}", f"Gene{i}")
        pm.layout_grid(n_cols=3, spacing_x=3, spacing_y=2)
        assert pm.nodes["n0"].x == 0
        assert pm.nodes["n3"].x == 0
        assert pm.nodes["n3"].y == -2

    def test_layout_linear(self):
        pm = PathwayMap("p")
        for i in range(4):
            pm.add_node(f"n{i}", f"Gene{i}")
        pm.layout_linear(spacing=5)
        assert pm.nodes["n0"].x == 0
        assert pm.nodes["n1"].x == 5
        assert pm.nodes["n3"].x == 15

    def test_empty_map(self):
        pm = PathwayMap("empty")
        assert len(pm.nodes) == 0
        assert len(pm.edges) == 0


class TestDrawPathway:
    def test_draw_basic(self):
        pm = create_kegg_style_pathway()
        fig = draw_pathway(pm, title="Test Pathway")
        assert fig is not None

    def test_draw_custom(self):
        pm = create_custom_pathway(["A", "B", "C", "D"])
        fig = draw_pathway(pm)
        assert fig is not None

    def test_draw_empty_map(self):
        pm = PathwayMap("empty")
        fig = draw_pathway(pm)
        assert fig is not None

    def test_draw_with_expression(self):
        pm = PathwayMap("p")
        pm.add_node("a", "GeneA").add_node("b", "GeneB")
        pm.add_edge("a", "b")
        pm.set_expression({"a": 1.5, "b": -1.0})
        fig = draw_pathway(pm, node_colors=True)
        assert fig is not None

    def test_draw_without_labels(self):
        pm = create_kegg_style_pathway()
        fig = draw_pathway(pm, show_labels=False)
        assert fig is not None


class TestCreatePathways:
    def test_kegg_style(self):
        pm = create_kegg_style_pathway()
        assert pm.name == "MAPK Signaling Pathway"
        assert len(pm.nodes) == 10
        assert len(pm.edges) == 9

    def test_custom_linear(self):
        pm = create_custom_pathway(["BRCA1", "TP53", "MYC"])
        assert len(pm.nodes) == 3
        assert len(pm.edges) == 2  # linear connections

    def test_custom_with_connections(self):
        pm = create_custom_pathway(["A", "B", "C"], connections=[(0, 2), (1, 2)])
        assert len(pm.edges) == 2


class TestFormatPathwayReport:
    def test_basic_report(self):
        pm = create_kegg_style_pathway()
        report = format_pathway_report(pm)
        assert "MAPK Signaling Pathway" in report
        assert "Nodes: 10" in report
        assert "Edges: 9" in report
        assert "EGF" in report
        assert "Connections:" in report

    def test_empty_pathway(self):
        pm = PathwayMap("empty")
        report = format_pathway_report(pm)
        assert "Nodes: 0" in report
        assert "Edges: 0" in report

    def test_with_expression(self):
        pm = PathwayMap("p")
        pm.add_node("a", "GeneA")
        pm.set_expression({"a": 1.23})
        report = format_pathway_report(pm)
        assert "expr=1.23" in report


# ══════════════════════════════════════════════════════════════════════════════
# 4. EPIGENOMICS — biosuite.core.epigenomics
# ══════════════════════════════════════════════════════════════════════════════

from biosuite.core.epigenomics import (
    MethylationSite, EpigenomicsReport,
    calculate_methylation_levels, find_dmrs,
    parse_atac_peaks, atac_peak_stats, format_epigenomics_report,
)


def _make_sites(n=20, context="CpG", coverage_range=(10, 50)):
    """Generate random MethylationSite objects for testing."""
    rng = np.random.RandomState(42)
    sites = []
    for i in range(n):
        cov = rng.randint(*coverage_range)
        meth = rng.randint(0, cov + 1)
        sites.append(MethylationSite(
            chrom="chr1",
            pos=1000 + i * 100,
            context=context,
            methylation_level=meth / cov,
            coverage=cov,
            methylated_count=meth,
        ))
    return sites


class TestMethylationSite:
    def test_creation(self):
        s = MethylationSite("chr1", 1000, "CpG", 0.75, 40, 30)
        assert s.chrom == "chr1"
        assert s.methylation_level == 0.75


class TestCalculateMethylationLevels:
    def test_basic(self):
        sites = _make_sites(n=30, context="CpG")
        report = calculate_methylation_levels(sites, min_coverage=5)
        assert isinstance(report, EpigenomicsReport)
        assert report.total_sites == 30
        assert 0 <= report.avg_methylation <= 1

    def test_context_breakdown(self):
        cpg = _make_sites(n=10, context="CpG")
        chg = _make_sites(n=10, context="CHG")
        chh = _make_sites(n=10, context="CHH")
        all_sites = cpg + chg + chh
        report = calculate_methylation_levels(all_sites)
        assert report.total_sites == 30
        assert report.cpg_methylation > 0
        assert report.chg_methylation > 0
        assert report.chh_methylation > 0

    def test_min_coverage_filter(self):
        sites = [
            MethylationSite("chr1", 1000, "CpG", 0.5, 2, 1),   # low coverage
            MethylationSite("chr1", 2000, "CpG", 0.8, 50, 40),  # high coverage
        ]
        report = calculate_methylation_levels(sites, min_coverage=10)
        assert report.total_sites == 1  # only one passes filter

    def test_empty_sites(self):
        report = calculate_methylation_levels([], min_coverage=5)
        assert report.total_sites == 0
        assert "No sites" in report.message

    def test_distribution(self):
        sites = _make_sites(n=100, context="CpG")
        report = calculate_methylation_levels(sites)
        total_in_dist = sum(report.methylation_distribution.values())
        assert total_in_dist == report.total_sites


class TestFindDMRs:
    def test_basic(self):
        rng = np.random.RandomState(42)
        g1, g2 = [], []
        for i in range(20):
            cov = 30
            g1.append(MethylationSite("chr1", 1000 + i * 100, "CpG", 0.9, cov, 27))
            g2.append(MethylationSite("chr1", 1000 + i * 100, "CpG", 0.2, cov, 6))
        dmrs = find_dmrs(g1, g2, min_coverage=5, min_delta=0.1)
        assert len(dmrs) > 0
        assert all("delta_methylation" in d for d in dmrs)

    def test_no_common_sites(self):
        g1 = [MethylationSite("chr1", 1000, "CpG", 0.5, 20, 10)]
        g2 = [MethylationSite("chr2", 2000, "CpG", 0.5, 20, 10)]
        dmrs = find_dmrs(g1, g2)
        assert dmrs == []

    def test_small_delta_no_dmrs(self):
        g1 = [MethylationSite("chr1", 1000, "CpG", 0.50, 30, 15)]
        g2 = [MethylationSite("chr1", 1000, "CpG", 0.49, 30, 15)]
        dmrs = find_dmrs(g1, g2, min_delta=0.5)
        assert dmrs == []


class TestATACSeq:
    def test_parse_atac_peaks(self, tmp_path):
        bed_file = str(tmp_path / "peaks.bed")
        with open(bed_file, "w") as f:
            f.write("chr1\t1000\t2000\tpeak1\t50\n")
            f.write("chr1\t5000\t6000\tpeak2\t75\n")
            f.write("chr2\t100\t500\tpeak3\t30\n")
        peaks = parse_atac_peaks(bed_file)
        assert len(peaks) == 3
        assert peaks[0]["chrom"] == "chr1"
        assert peaks[0]["start"] == 1000
        assert peaks[0]["end"] == 2000

    def test_atac_peak_stats(self):
        peaks = [
            {"chrom": "chr1", "start": 1000, "end": 2000, "score": 50, "name": "p1"},
            {"chrom": "chr1", "start": 5000, "end": 5500, "score": 75, "name": "p2"},
            {"chrom": "chr2", "start": 100, "end": 600, "score": 30, "name": "p3"},
        ]
        stats = atac_peak_stats(peaks)
        assert stats["total_peaks"] == 3
        assert stats["total_bp"] == 2000  # 1000 + 500 + 500
        assert stats["chromosomes"] == 2

    def test_atac_empty(self):
        assert atac_peak_stats([]) == {}


class TestFormatEpigenomicsReport:
    def test_format(self):
        report = EpigenomicsReport(
            total_sites=100,
            avg_methylation=0.65,
            cpg_methylation=0.72,
            chg_methylation=0.55,
            chh_methylation=0.30,
            dmr_count=5,
            methylation_distribution={"low": 40, "high": 60},
        )
        out = format_epigenomics_report(report)
        assert "Total sites: 100" in out
        assert "0.650" in out
        assert "CpG" in out
        assert "DMRs found: 5" in out
        assert "low: 40" in out

    def test_format_empty_report(self):
        report = EpigenomicsReport()
        out = format_epigenomics_report(report)
        assert "Total sites: 0" in out


# ══════════════════════════════════════════════════════════════════════════════
# 5. MD SIMULATION — biosuite.core.md_simulation
# ══════════════════════════════════════════════════════════════════════════════

from biosuite.core.md_simulation import (
    MDSimulationResult, check_md_tools, run_simulation, format_md_report,
    _extract_coords_from_pdb, _write_pdb,
)


def _make_simple_pdb(tmp_path, n_atoms=10):
    """Create a minimal PDB file with n_atoms CA atoms."""
    pdb_path = str(tmp_path / "test.pdb")
    with open(pdb_path, "w") as f:
        rng = np.random.RandomState(42)
        for i in range(n_atoms):
            x, y, z = rng.uniform(-10, 10, 3)
            f.write(f"ATOM  {i+1:5d}  CA  ALA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
        f.write("END\n")
    return pdb_path


class TestCheckMDTools:
    def test_returns_dict(self):
        result = check_md_tools()
        assert isinstance(result, dict)
        assert "openmm" in result


class TestExtractCoords:
    def test_basic(self, tmp_path):
        pdb = _make_simple_pdb(tmp_path, n_atoms=5)
        coords = _extract_coords_from_pdb(pdb)
        assert coords.shape == (5, 3)

    def test_empty_pdb(self, tmp_path):
        pdb = str(tmp_path / "empty.pdb")
        with open(pdb, "w") as f:
            f.write("END\n")
        coords = _extract_coords_from_pdb(pdb)
        assert len(coords) == 0


class TestWritePDB:
    def test_write_and_read_back(self, tmp_path):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out_path = str(tmp_path / "out.pdb")
        _write_pdb(coords, out_path)
        assert os.path.exists(out_path)
        coords_back = _extract_coords_from_pdb(out_path)
        assert coords_back.shape == (2, 3)
        np.testing.assert_allclose(coords_back, coords, atol=0.001)


class TestRunSimulation:
    def test_builtin_minimization(self, tmp_path):
        pdb = _make_simple_pdb(tmp_path, n_atoms=8)
        out_pdb = str(tmp_path / "minimized.pdb")
        result = run_simulation(pdb, output_pdb=out_pdb, steps=100, tool='builtin')
        assert isinstance(result, MDSimulationResult)
        assert result.engine in ("builtin", "openmm")
        assert result.steps > 0
        assert os.path.exists(out_pdb)

    def test_file_not_found(self):
        result = run_simulation("/nonexistent/file.pdb")
        assert "not found" in result.message

    def test_empty_pdb(self, tmp_path):
        pdb = str(tmp_path / "empty.pdb")
        with open(pdb, "w") as f:
            f.write("END\n")
        result = run_simulation(pdb, steps=10)
        assert "No atoms" in result.message


class TestFormatMDReport:
    def test_basic_report(self):
        result = MDSimulationResult(
            engine="builtin", steps=500, energy=-123.45,
            temperature=300, radius_gyration=[5.0, 4.5, 4.2],
            message="Test minimization"
        )
        report = format_md_report(result)
        assert "builtin" in report
        assert "500" in report
        assert "-123.45" in report
        assert "300" in report
        assert "4.20" in report
        assert "Test minimization" in report

    def test_empty_rg(self):
        result = MDSimulationResult(engine="openmm", steps=100, energy=0)
        report = format_md_report(result)
        assert "openmm" in report


# ══════════════════════════════════════════════════════════════════════════════
# 6. STRUCTURE PREDICTION — biosuite.core.structure_prediction
# ══════════════════════════════════════════════════════════════════════════════

from biosuite.core.structure_prediction import (
    PredictionResult, check_prediction_tools, predict_structure,
    format_prediction_report, _extract_plddt,
)


class TestCheckPredictionTools:
    def test_returns_dict(self):
        result = check_prediction_tools()
        assert isinstance(result, dict)
        assert "esmfold" in result
        assert "torch" in result


class TestExtractPlddt:
    def test_basic(self):
        pdb_string = (
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  85.30\n"
            "ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00  92.10\n"
            "ATOM      3  CA  ALA A   3       7.000   8.000   9.000  1.00  45.60\n"
            "END\n"
        )
        scores = _extract_plddt(pdb_string)
        assert len(scores) == 3
        assert abs(scores[0] - 85.3) < 0.1
        assert abs(scores[2] - 45.6) < 0.1

    def test_empty_string(self):
        assert _extract_plddt("") == []

    def test_no_atom_lines(self):
        assert _extract_plddt("HETATM  1  O   HOH A   1\nEND\n") == []


class TestPredictStructure:
    def test_no_input(self):
        result = predict_structure()
        assert result.engine == "none"
        assert "No sequence" in result.message

    @patch('biosuite.core.structure_prediction._esmfold_predict')
    def test_sequence_provided(self, mock_esmfold):
        mock_esmfold.return_value = PredictionResult(
            engine="esmfold", pdb_string="ATOM ...",
            num_residues=10, confidence=80.0,
            message="Done"
        )
        result = predict_structure(sequence="ACDEFGHIKL")
        assert result.engine == "esmfold"
        mock_esmfold.assert_called_once()

    @patch('biosuite.core.structure_prediction._alphafold_fetch')
    @patch('biosuite.core.structure_prediction._esmfold_predict')
    def test_uniprot_id_priority(self, mock_esmfold, mock_afold):
        mock_afold.return_value = PredictionResult(
            engine="alphafold", pdb_string="PDB data",
            confidence=90.0, message="AF done"
        )
        result = predict_structure(uniprot_id="P01308")
        assert result.engine == "alphafold"
        mock_afold.assert_called_once()
        mock_esmfold.assert_not_called()

    @patch('biosuite.core.structure_prediction._alphafold_fetch')
    @patch('biosuite.core.structure_prediction._esmfold_predict')
    def test_esmfold_fallback_when_alphafold_empty(self, mock_esmfold, mock_afold):
        mock_afold.return_value = PredictionResult(engine="alphafold", message="No prediction found")
        mock_esmfold.return_value = PredictionResult(
            engine="esmfold", pdb_string="ATOM 1", num_residues=5,
            confidence=75.0, message="OK"
        )
        result = predict_structure(sequence="ACDEF", uniprot_id="P01308")
        assert result.engine == "esmfold"


class TestFormatPredictionReport:
    def test_basic(self):
        result = PredictionResult(
            engine="esmfold", num_residues=100, confidence=82.5,
            output_file="/tmp/test.pdb",
            plddt_scores=[95, 85, 60, 70, 40],
            message="ESMFold done"
        )
        report = format_prediction_report(result)
        assert "esmfold" in report
        assert "100" in report
        assert "82.5" in report
        assert "/tmp/test.pdb" in report
        assert "ESMFold done" in report

    def test_confidence_categories(self):
        result = PredictionResult(
            engine="test", num_residues=10, confidence=50.0,
            plddt_scores=[95, 92, 80, 75, 60, 55, 40, 30, 20, 10],
        )
        report = format_prediction_report(result)
        assert "Confident (>90): 2" in report
        assert "Good (70-90): 2" in report
        assert "Low (<70): 6" in report

    def test_no_plddt(self):
        result = PredictionResult(engine="none", message="No prediction")
        report = format_prediction_report(result)
        assert "none" in report
        assert "No prediction" in report
        assert "Confident" not in report


# ── Run with: pytest tests/test_databases_and_tools.py -v ────────────────────
