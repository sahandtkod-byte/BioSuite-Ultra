"""
Tests for the BioSuite REST API (biosuite/api/__init__.py).

Uses fastapi.testclient.TestClient to hit every major endpoint.
All tests are self-contained and use inline fixture data.
"""
import os

# Ensure the project root is importable
# Project root is already on sys.path via pyproject.toml / pytest config
import pytest
from fastapi.testclient import TestClient

from biosuite.api import app

API_KEY = os.environ.get("BIOSUITE_API_KEY", "changeme-dev-key")
client = TestClient(app, raise_server_exceptions=False)
client.headers["X-API-Key"] = API_KEY


# ── Health & Info ────────────────────────────────────────────────────────────


class TestHealthAndInfo:
    """Root, health, and module listing endpoints."""

    def test_root_returns_html(self):
        """GET / should return an HTML page."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_root_contains_title(self):
        """Landing page should mention BioSuite."""
        resp = client.get("/")
        assert "BioSuite" in resp.text

    def test_health_healthy(self):
        """GET /health should report healthy status."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        # Pin to the package version rather than a literal, so a release bump
        # cannot leave /health reporting a stale number.
        from biosuite import __version__
        assert data["version"] == __version__
        assert "timestamp" in data

    def test_health_module_count_matches_the_installation(self):
        """The count used to be the hard-coded literal 47."""
        import pathlib

        import biosuite.core as core_pkg

        resp = client.get("/health")
        root = pathlib.Path(core_pkg.__file__).parent
        expected = len({p.stem for p in root.glob("*.py") if p.stem != "__init__"}) \
            + len({p.stem for p in (root / "workflow").glob("*.py")
                   if p.stem != "__init__"})
        assert resp.json()["modules"] == expected


    def test_modules_list(self):
        """GET /api/v1/modules should return a list of modules."""
        resp = client.get("/api/v1/modules")
        assert resp.status_code == 200
        data = resp.json()
        assert "modules" in data
        assert len(data["modules"]) > 0
        names = [m["name"] for m in data["modules"]]
        assert "sequence" in names
        assert "alignment" in names


# ── Sequence Analysis ────────────────────────────────────────────────────────


class TestSequenceEndpoints:
    """POST endpoints under /api/v1/sequence/."""

    def test_gc_content(self):
        """GC content of ATCGATCG should be 50%."""
        resp = client.post(
            "/api/v1/sequence/gc-content",
            json={"sequence": "ATCGATCG"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "gc_percent" in data
        assert abs(data["gc_percent"] - 50.0) < 0.01
        assert data["sequence_length"] == 8

    def test_gc_content_all_gc(self):
        """100% GC sequence."""
        resp = client.post(
            "/api/v1/sequence/gc-content",
            json={"sequence": "GCGCGCGC"},
        )
        assert resp.status_code == 200
        assert resp.json()["gc_percent"] == 100.0

    def test_gc_content_empty(self):
        """Empty sequence should still work."""
        resp = client.post(
            "/api/v1/sequence/gc-content",
            json={"sequence": ""},
        )
        assert resp.status_code == 422  # Pydantic validation: min_length=1

    def test_reverse_complement(self):
        """Reverse complement of ATCG should be CGAT."""
        resp = client.post(
            "/api/v1/sequence/reverse-complement",
            json={"sequence": "ATCG"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["reverse_complement"] == "CGAT"
        assert data["original"] == "ATCG"

    def test_translate_frame1(self):
        """ATGAAATTTTAA translated in frame 1."""
        resp = client.post(
            "/api/v1/sequence/translate",
            json={"sequence": "ATGAAATTTTAA", "frame": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["protein"] == "MKF*"  # ATG→M, AAA→K, TTT→F, TAA→*

    def test_translate_frame2(self):
        """Translation in frame 2 should differ from frame 1."""
        resp = client.post(
            "/api/v1/sequence/translate",
            json={"sequence": "ATGAAATTTTAA", "frame": 2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["protein"]) >= 1
        assert data["frame"] == 2

    def test_sequence_stats(self):
        """Stats for ATCGATCG should report correct base counts."""
        resp = client.post(
            "/api/v1/sequence/stats",
            json={"sequence": "ATCGATCG"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["length"] == 8
        assert data["A"] == 2
        assert data["T"] == 2
        assert data["G"] == 2
        assert data["C"] == 2


# ── Alignment ────────────────────────────────────────────────────────────────


class TestAlignmentEndpoints:
    """POST endpoints under /api/v1/alignment/."""

    def test_needleman_wunsch_basic(self):
        """NW alignment should return two aligned sequences and a score."""
        resp = client.post(
            "/api/v1/alignment/needleman-wunsch",
            json={"seq1": "AGTACGCA", "seq2": "TATGC"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "aligned_seq1" in data
        assert "aligned_seq2" in data
        assert "score" in data
        assert isinstance(data["score"], (int, float))

    def test_needleman_wunsch_identical(self):
        """Identical sequences should produce perfect match."""
        resp = client.post(
            "/api/v1/alignment/needleman-wunsch",
            json={"seq1": "ATCG", "seq2": "ATCG"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] > 0

    def test_smith_waterman_basic(self):
        """SW alignment should return two aligned sequences and a score."""
        resp = client.post(
            "/api/v1/alignment/smith-waterman",
            json={"seq1": "AGTACGCA", "seq2": "TATGC"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "aligned_seq1" in data
        assert "aligned_seq2" in data
        assert data["score"] > 0

    def test_smith_waterman_local(self):
        """SW on partially matching sequences should find local match."""
        resp = client.post(
            "/api/v1/alignment/smith-waterman",
            json={"seq1": "XXXATCGYYY", "seq2": "ATCG"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] > 0


# ── CRISPR ──────────────────────────────────────────────────────────────────


class TestCRISPREndpoint:
    """POST /api/v1/crispr/design."""

    def test_crispr_design(self):
        """Design guide RNAs for a valid target sequence."""
        resp = client.post(
            "/api/v1/crispr/design",
            json={
                "sequence": "A" * 30 + "ATCGATCGATCGATCGATCG" + "NGG" + "A" * 30,
                "pam_type": "SpCas9",
                "guide_length": 20,
                "max_guides": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "guides" in data
        assert isinstance(data["guides"], list)
        assert data["engine"] == "builtin"

    def test_crispr_design_with_guides(self):
        """Ensure guides have expected structure."""
        seq_with_pam = "GCTAGCTAGCTAGCTAGCTAG" + "AGG" + "GCTAGCTAGCTAGCTAGCTAG" + "TGG" + "A" * 20
        resp = client.post(
            "/api/v1/crispr/design",
            json={
                "sequence": seq_with_pam,
                "pam_type": "SpCas9",
                "guide_length": 20,
                "max_guides": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        if len(data["guides"]) > 0:
            guide = data["guides"][0]
            assert "sequence" in guide
            assert "pam" in guide
            assert "score" in guide


# ── Population Genetics ──────────────────────────────────────────────────────


class TestPopGenEndpoint:
    """POST /api/v1/popgen/hwe."""

    def test_hwe_balanced(self):
        """A population in exact Hardy-Weinberg equilibrium.

        AA=25 Aa=50 aa=25 gives p = q = 0.5 and expected counts identical to
        the observed ones, so chi2 must be exactly 0 and the p-value 1.

        (This test previously accepted `status_code in (200, 500)` and
        documented a numpy-serialisation 500 as a "known limitation".)
        """
        resp = client.post(
            "/api/v1/popgen/hwe",
            json={"AA": 25, "Aa": 50, "aa": 25},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["chi2"] == pytest.approx(0.0, abs=1e-9)
        assert data["p_value"] == pytest.approx(1.0, abs=1e-9)
        assert data["allele_freq_p"] == pytest.approx(0.5)
        assert data["allele_freq_q"] == pytest.approx(0.5)
        assert data["in_hwe"] is True

    def test_hwe_unbalanced(self):
        """Unbalanced counts, checked against an independent chi-square."""
        from scipy import stats

        observed = {"AA": 90, "Aa": 10, "aa": 0}
        resp = client.post("/api/v1/popgen/hwe", json=observed)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        n = sum(observed.values())
        p = (2 * observed["AA"] + observed["Aa"]) / (2 * n)
        q = 1 - p
        expected = {"AA": p * p * n, "Aa": 2 * p * q * n, "aa": q * q * n}
        chi2 = sum((observed[k] - expected[k]) ** 2 / expected[k]
                   for k in observed)
        p_value = float(stats.chi2.sf(chi2, df=1))

        assert data["allele_freq_p"] == pytest.approx(p)
        assert data["allele_freq_q"] == pytest.approx(q)
        assert data["chi2"] == pytest.approx(chi2, rel=1e-2)
        assert data["p_value"] == pytest.approx(p_value, rel=1e-2)

    def test_hwe_rejects_negative_counts(self):
        resp = client.post("/api/v1/popgen/hwe",
                           json={"AA": -1, "Aa": 10, "aa": 5})
        assert resp.status_code in (400, 422), resp.text

    def test_hwe_rejects_an_empty_population(self):
        resp = client.post("/api/v1/popgen/hwe",
                           json={"AA": 0, "Aa": 0, "aa": 0})
        assert resp.status_code in (400, 422), resp.text


# ── Metagenomics ─────────────────────────────────────────────────────────────


class TestMetagenomicsEndpoint:
    """POST /api/v1/metagenomics/diversity."""

    def test_diversity_metrics(self):
        """Alpha diversity should return shannon, simpson, chao1."""
        resp = client.post(
            "/api/v1/metagenomics/diversity",
            json=[10, 20, 30, 40, 50],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "shannon" in data
        assert "simpson" in data
        assert "chao1" in data
        assert data["shannon"] > 0
        assert 0 < data["simpson"] <= 1

    def test_diversity_single_taxon(self):
        """Single taxon should have zero shannon entropy."""
        resp = client.post(
            "/api/v1/metagenomics/diversity",
            json=[100],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["shannon"] == 0.0

    def test_diversity_observed_taxa(self):
        """Observed taxa should count non-zero entries."""
        resp = client.post(
            "/api/v1/metagenomics/diversity",
            json=[10, 0, 30, 0, 50],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["observed_taxa"] == 3


# ── Epitope Prediction ──────────────────────────────────────────────────────


class TestEpitopeEndpoint:
    """POST /api/v1/epitope/predict."""

    def test_epitope_predict(self):
        """Predict epitopes for a protein sequence."""
        protein = "MKLLILTCLVAVALARPKEVKLFDKAVDKIEKMGVHMDQFIADPP"
        resp = client.post(
            "/api/v1/epitope/predict",
            json={"sequence": protein, "mhc_type": "A0201"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "t_cell_epitopes" in data
        assert "b_cell_epitopes" in data
        assert isinstance(data["t_cell_epitopes"], list)
        assert isinstance(data["b_cell_epitopes"], list)

    def test_epitope_has_peptide_info(self):
        """Each epitope should have peptide and score."""
        protein = "MKLLILTCLVAVALARPKEVKLFDKAVDKIEKMGVHMDQFIADPP"
        resp = client.post(
            "/api/v1/epitope/predict",
            json={"sequence": protein, "mhc_type": "A0201"},
        )
        data = resp.json()
        if len(data["t_cell_epitopes"]) > 0:
            ep = data["t_cell_epitopes"][0]
            assert "peptide" in ep
            assert "score" in ep


# ── GWAS Demo ────────────────────────────────────────────────────────────────


class TestGWASDemoEndpoint:
    """GET /api/v1/gwas/demo."""

    def test_gwas_demo(self):
        """Demo GWAS data should contain SNP records."""
        resp = client.get("/api/v1/gwas/demo")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert "total" in data
        assert data["total"] > 0

    def test_gwas_demo_custom_size(self):
        """Demo GWAS with custom n_snps.

        generate_gwas_data may filter some SNPs (e.g. non-autosomal),
        so total may differ from the requested count.
        """
        resp = client.get("/api/v1/gwas/demo?n_snps=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 50  # At least some SNPs returned
        assert len(data["data"]) > 0


# ── Differential Expression ──────────────────────────────────────────────────


class TestExpressionEndpoint:
    """POST /api/v1/expression/differential."""

    def test_differential_expression(self):
        """Basic differential expression with two conditions.

        Regression guard: the endpoint used to transpose the counts matrix and
        always fail with "positional indexers are out-of-bounds"; a 500 is now
        a test failure, not an accepted outcome.
        """
        gene_counts = {
            "gene1": [10, 12, 11, 100, 95, 105],
            "gene2": [50, 55, 48, 52, 50, 51],
            "gene3": [200, 210, 190, 205, 195, 200],
        }
        resp = client.post(
            "/api/v1/expression/differential",
            json={
                "counts": gene_counts,
                "conditions": ["ctrl", "ctrl", "ctrl", "treat", "treat", "treat"],
                "method": "ttest",
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["num_genes"] == 3
        genes = {row["gene"] for row in data["results"]}
        assert genes == {"gene1", "gene2", "gene3"}
        by_gene = {row["gene"]: row for row in data["results"]}
        # gene1 goes 10/12/11 -> 100/95/105, i.e. an ~3-log2 up-shift.
        assert by_gene["gene1"]["log2FC"] > 2.5
        # gene2/gene3 are flat: |log2FC| must stay near zero.
        assert abs(by_gene["gene2"]["log2FC"]) < 0.2
        assert abs(by_gene["gene3"]["log2FC"]) < 0.2

    def test_differential_expression_with_upregulation(self):
        """A strongly up-regulated gene must be called significant.

        Replicates carry realistic variation; a group with *zero* variance has
        no estimable standard error and is covered separately by
        ``tests/core/test_expression.py`` (the p-value is NaN, not a fabricated
        1.0 or 0.0).
        """
        gene_counts = {
            "gene1": [10, 12, 9, 500, 520, 480],
        }
        resp = client.post(
            "/api/v1/expression/differential",
            json={
                "counts": gene_counts,
                "conditions": ["ctrl", "ctrl", "ctrl", "treat", "treat", "treat"],
                "method": "ttest",
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["num_upregulated"] >= 1
        assert data["num_downregulated"] == 0


# ── Error Handling ───────────────────────────────────────────────────────────


class TestErrorHandling:
    """Invalid requests should return proper error responses."""

    def test_gc_content_missing_sequence(self):
        """Missing 'sequence' field should return 422."""
        resp = client.post("/api/v1/sequence/gc-content", json={})
        assert resp.status_code == 422

    def test_gc_content_wrong_type(self):
        """Wrong type should return 422."""
        resp = client.post(
            "/api/v1/sequence/gc-content",
            json={"sequence": 123},
        )
        assert resp.status_code == 422

    def test_alignment_missing_fields(self):
        """Alignment request missing both sequences should return 422."""
        resp = client.post("/api/v1/alignment/needleman-wunsch", json={})
        assert resp.status_code == 422

    def test_nonexistent_endpoint(self):
        """Hitting a non-existent route should return 404."""
        resp = client.get("/api/v1/nonexistent")
        assert resp.status_code == 404

    def test_post_to_get_endpoint(self):
        """Wrong HTTP method should return 405."""
        resp = client.post("/health")
        assert resp.status_code in (404, 405)


# ── CORS Headers ─────────────────────────────────────────────────────────────


class TestCORS:
    """CORS middleware should add appropriate headers."""

    def test_cors_headers_present(self):
        """OPTIONS preflight should include CORS headers."""
        resp = client.options(
            "/api/v1/sequence/gc-content",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # FastAPI CORSMiddleware returns 200 for OPTIONS with correct headers
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    def test_cors_unknown_origin_is_not_reflected(self):
        """Security regression guard for BSU-012.

        The middleware used to echo *any* Origin back together with
        ``allow-credentials: true``, which let a hostile page drive a
        researcher's local API.  Unknown origins must now get no
        access-control-allow-origin header at all.
        """
        resp = client.options(
            "/api/v1/sequence/gc-content",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        acao = resp.headers.get("access-control-allow-origin")
        assert acao != "*"
        assert acao != "https://evil.example"
        assert acao is None

    def test_cors_never_pairs_wildcard_with_credentials(self):
        """`*` and allow-credentials must never be sent together."""
        resp = client.options(
            "/api/v1/sequence/gc-content",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        if resp.headers.get("access-control-allow-origin") == "*":
            assert "access-control-allow-credentials" not in resp.headers
