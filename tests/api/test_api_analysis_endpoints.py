"""The remaining REST endpoints: what they return has to be what the core computed.

Each of these endpoints is a thin marshalling layer — build the input the core
function wants, call it, reshape the answer into JSON.  That seam is exactly where the
expression endpoints were wrong (see ``test_api_expression_endpoints.py``), so each
test here pins a property of the *response*: the counts agree with the records sent
back, the shapes are what the docstrings advertise, and the numbers are the core's
own.  The database-search endpoints are stubbed rather than called, because hitting
NCBI from a test suite is neither reproducible nor free.
"""
import os

import pytest
from fastapi.testclient import TestClient

from biosuite.api import app


@pytest.fixture(scope="module")
def client():
    c = TestClient(app, raise_server_exceptions=False)
    c.headers["X-API-Key"] = os.environ.get("BIOSUITE_API_KEY", "test-api-key-not-a-real-secret")
    return c


def _post(client, path, payload=None, **params):
    response = client.post(path, json=payload, params=params)
    assert response.status_code == 200, response.text
    return response.json()


# ── BLAST ────────────────────────────────────────────────────────────────────

def test_blast_search_without_a_database_matches_the_query_against_itself(client):
    """The built-in k-mer engine must find the query in itself, at 100 % identity."""
    body = _post(client, "/api/v1/blast/search",
                 {"query": "ATCGATCGATCGATCGATCGATCG"})

    assert body["num_hits"] >= 1
    assert body["engine"]
    top = body["hits"][0]
    assert top["identity"] == 100.0
    assert set(top) == {"subject_id", "identity", "e_value", "score", "alignment_length"}


def test_blast_search_reports_a_missing_database_as_404(client):
    response = client.post("/api/v1/blast/search",
                           json={"query": "ATCGATCG", "database": "nowhere.fasta"})

    assert response.status_code == 404
    assert response.json()["detail"]


# ── CRISPR ───────────────────────────────────────────────────────────────────

def test_crispr_design_returns_guides_of_the_requested_length(client):
    target = ("ATGCAAAGCTAGCTAGCAAGTTGACCGGTATCGGATCCAAGCTCGGATCC"
              "AGCTAGCTAGCATCGATTCAAGCTCGATCGATCGATCGATCGATCGA")
    body = _post(client, "/api/v1/crispr/design",
                 {"sequence": target, "guide_length": 20, "max_guides": 5})

    assert body["num_guides"] == len(body["guides"]) <= 5
    for guide in body["guides"]:
        assert len(guide["sequence"]) == 20
        assert guide["pam"]
        assert 0 <= guide["gc_content"] <= 100


# ── Epitopes ─────────────────────────────────────────────────────────────────

def test_epitope_prediction_caps_the_records_but_reports_the_full_count(client):
    protein = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ" * 3
    body = _post(client, "/api/v1/epitope/predict", {"sequence": protein})

    assert body["t_cell_count"] >= len(body["t_cell_epitopes"])
    assert len(body["t_cell_epitopes"]) <= 20
    assert len(body["b_cell_epitopes"]) <= 20
    for epitope in body["t_cell_epitopes"]:
        assert {"peptide", "start", "end", "score"} <= set(epitope)


# ── Metagenomics ─────────────────────────────────────────────────────────────

def test_16s_classification_counts_what_it_returns(client):
    sequences = [{"name": "a", "sequence": "AGAGTTTGATCCTGGCTCAG"},
                 {"name": "b", "sequence": "AGAGTTTGATCCTGGCTCAG"}]
    body = _post(client, "/api/v1/metagenomics/classify-16s", sequences)

    assert body["num_classified"] == len(body["classifications"])
    assert body["engine"]
    assert isinstance(body["abundance"], list)


def test_diversity_metrics_are_rounded_core_values(client):
    from biosuite.core.metagenomics import chao1_estimator, shannon_entropy, simpson_index

    counts = [10, 5, 0, 1]
    body = _post(client, "/api/v1/metagenomics/diversity", counts)

    assert body["observed_taxa"] == 3
    assert body["shannon"] == round(shannon_entropy(counts), 4)
    assert body["simpson"] == round(simpson_index(counts), 4)
    assert body["chao1"] == round(chao1_estimator(counts), 2)


# ── GWAS ─────────────────────────────────────────────────────────────────────

ASSOCIATED = {"chrom": "1", "pos": 1000, "snp_id": "rsTight",
              "case_alt": 90, "case_ref": 10, "ctrl_alt": 10, "ctrl_ref": 90}
NULL = {"chrom": "1", "pos": 2_000_000, "snp_id": "rsNull",
        "case_alt": 50, "case_ref": 50, "ctrl_alt": 50, "ctrl_ref": 50}


def test_gwas_run_agrees_with_the_records_it_returns(client):
    body = _post(client, "/api/v1/gwas/run", [ASSOCIATED, NULL])

    assert body["num_snps"] == 2 == len(body["results"])
    by_snp = {r["snp_id"]: r for r in body["results"]}
    assert by_snp["rsTight"]["p_value"] < by_snp["rsNull"]["p_value"]
    assert body["num_significant"] == sum(1 for r in body["results"] if r["p_value"] < 5e-8)
    assert [s["snp_id"] for s in body["lead_snps"]] == ["rsTight"]


def test_gwas_demo_returns_as_many_snps_as_asked(client):
    response = client.get("/api/v1/gwas/demo", params={"n_snps": 250})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 250
    assert len(body["data"]) == 100          # the endpoint's documented preview size


# ── Phylogeny ────────────────────────────────────────────────────────────────

SEQUENCES = ["ACGTACGT", "ACGAACGT", "ACGTACGA", "TTTTGGGG"]


def test_distance_matrix_is_square_symmetric_and_zero_on_the_diagonal(client):
    from biosuite.core.phylogeny import distance_matrix

    body = _post(client, "/api/v1/phylogeny/distance-matrix", SEQUENCES)
    matrix = body["matrix"]

    n = len(SEQUENCES)
    assert len(matrix) == n and all(len(row) == n for row in matrix)
    assert [matrix[i][i] for i in range(n)] == [0.0] * n
    assert matrix == [list(row) for row in zip(*matrix)]      # symmetric
    assert body["labels"] == [f"seq_{i}" for i in range(n)]
    expected = distance_matrix(SEQUENCES).tolist()
    for row, wanted in zip(matrix, expected):
        assert row == pytest.approx(wanted)


def test_upgma_returns_one_merge_per_pair_of_lineages(client):
    body = _post(client, "/api/v1/phylogeny/upgma", SEQUENCES)

    assert body["num_sequences"] == len(SEQUENCES)
    assert len(body["linkage_matrix"]) == len(SEQUENCES) - 1
    assert len(body["labels"]) == len(SEQUENCES)


# ── Population genetics ──────────────────────────────────────────────────────

def test_fst_pairs_are_named_after_the_populations_compared(client):
    from biosuite.core.popgen import calculate_fst

    pops = [[[0, 1, 2], [0, 0, 1]], [[2, 2, 1], [2, 2, 2]], [[0, 0, 0], [1, 0, 0]]]
    body = _post(client, "/api/v1/popgen/fst", pops)

    keys = sorted(body["fst_pairs"])
    assert keys == sorted(f"{i}-{j}" for i in range(3) for j in range(i + 1, 3))
    assert all(0.0 <= v <= 1.0 for v in body["fst_pairs"].values())
    import numpy as np
    expected = {f"{k[0]}-{k[1]}": v for k, v in
                calculate_fst([np.array(p) for p in pops]).items()}
    assert body["fst_pairs"] == pytest.approx(expected)


def test_tajimas_d_returns_a_single_number(client):
    from biosuite.core.popgen import tajimas_d

    import numpy as np
    matrix = [[0, 1, 2, 0], [0, 1, 1, 0], [2, 2, 1, 0], [1, 0, 2, 1]]
    body = _post(client, "/api/v1/popgen/tajimas-d", matrix)

    assert isinstance(body["tajimas_d"], float)
    assert body["tajimas_d"] == pytest.approx(float(tajimas_d(np.array(matrix))))


# ── Workflow ─────────────────────────────────────────────────────────────────

def test_pipeline_endpoint_refuses_instead_of_pretending(client):
    """A stub that answers 200 tells every generated client the pipeline ran."""
    response = client.post("/api/v1/workflow/pipeline",
                           json=[{"function": "gc_content", "args": {"seq": "ATCG"}}])

    assert response.status_code == 501
    detail = response.json()["detail"]
    assert "1 step(s) were not run" in detail


# ── Database search (no network in tests) ────────────────────────────────────

@pytest.mark.parametrize(("path", "func"), [
    ("/api/v1/database/ncbi", "search_ncbi"),
    ("/api/v1/database/uniprot", "search_uniprot"),
    ("/api/v1/database/pdb", "search_pdb"),
    ("/api/v1/database/kegg", "search_kegg"),
])
def test_database_endpoints_pass_the_query_through_and_return_the_records(
        client, monkeypatch, path, func):
    """The endpoints only marshal: same records, same count, nothing invented."""
    from biosuite.core import databases

    def fake(query, max_results=10):
        calls.append((query, max_results))
        return databases.DBResult(source=func, query=query,
                                  data={"count": 2},
                                  records=[{"id": "a"}, {"id": "b"}])

    calls = []
    monkeypatch.setattr(databases, func, fake)
    body = client.get(path, params={"query": "BRCA1", "max_results": 2}).json()

    assert calls == [("BRCA1", 2)]
    assert body["results"] == [{"id": "a"}, {"id": "b"}]
    if path.endswith("ncbi"):
        assert body["count"] == 2


def test_database_endpoint_reports_a_failed_query_as_an_empty_result(client, monkeypatch):
    """A source that errors must not become a 500 with no explanation."""
    from biosuite.core import databases

    def offline(query, max_results=10):
        return databases.DBResult(source="ncbi", query=query, data={},
                                  records=[], error="network disabled")

    monkeypatch.setattr(databases, "search_ncbi", offline)
    response = client.get("/api/v1/database/ncbi", params={"query": "anything"})

    assert response.status_code == 200
    assert response.json() == {"results": [], "count": 0}


# ── Files ────────────────────────────────────────────────────────────────────

@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    (root / "regions.bed").write_text("chr1\t100\t200\tgeneA\nchr1\t300\t400\tgeneB\n",
                                      encoding="utf-8")
    monkeypatch.setenv("BIOSUITE_DATA_DIR", str(root))
    return root


def test_detect_format_reads_inside_the_data_directory(client, data_dir):
    body = _post(client, "/api/v1/file/detect-format", None, file_path="regions.bed")

    assert body["format"] == "bed"
    assert body["file"] == "regions.bed"


def test_read_file_summarises_the_records_it_found(client, data_dir):
    body = _post(client, "/api/v1/file/read", None, file_path="regions.bed")

    assert body["format"] == "bed"
    assert "regions.bed" in body["summary"] or "2" in body["summary"]


def test_blast_search_against_a_database_inside_the_data_directory(client, data_dir,
                                                                   monkeypatch):
    """The database argument is resolved, checked, and then handed to the engine."""
    from biosuite import api as api_module

    (data_dir / "db.fasta").write_text(">subject\nATCGATCGATCGATCGATCGATCG\n", encoding="utf-8")
    monkeypatch.setenv("BIOSUITE_DATA_DIR", str(data_dir))

    body = _post(client, "/api/v1/blast/search",
                 {"query": "ATCGATCGATCGATCGATCGATCG", "database": "db.fasta"})

    assert body["num_hits"] == len(body["hits"]) >= 1
    assert body["hits"][0]["subject_id"] == "subject"
    assert body["hits"][0]["identity"] == 100.0
    assert isinstance(body["hits"][0]["alignment_length"], int)


def test_blast_search_rejects_a_database_that_is_not_a_file(client, data_dir, monkeypatch):
    """A directory inside the data root resolves, but it is not a FASTA database."""
    from biosuite import api as api_module

    (data_dir / "genomes").mkdir()
    monkeypatch.setenv("BIOSUITE_DATA_DIR", str(data_dir))

    response = client.post("/api/v1/blast/search",
                           json={"query": "ATCGATCG", "database": "genomes"})

    assert response.status_code == 404
    assert "BLAST database not found" in response.json()["detail"]


def test_blast_search_leaves_no_query_file_behind(client):
    """The temp FASTA is created per request, so a request must clean it up."""
    import glob
    import tempfile

    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "biosuite_*")))
    _post(client, "/api/v1/blast/search", {"query": "ATCGATCGATCG"})
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "biosuite_*")))

    assert not after - before


def test_docs_routes_stay_behind_the_api_key_when_not_public(client):
    """Interactive documentation publishes the whole endpoint surface.

    The app registers ``Depends(verify_api_key)`` at construction, so the only way to
    swap the authenticator is FastAPI's own ``dependency_overrides`` — patching the
    module attribute would leave the real dependency in place.
    """
    from biosuite import api as api_module
    from biosuite.api.auth import verify_api_key

    if api_module._DOCS_PUBLIC:
        pytest.skip("documentation is public in this environment (dev mode)")

    async def reject(key: str = ""):
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="nope")

    try:
        client.app.dependency_overrides[verify_api_key] = reject
        assert client.get("/openapi.json").status_code == 401
        assert client.get("/docs").status_code == 401

        client.app.dependency_overrides[verify_api_key] = _allow
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        assert schema.json()["info"]["title"] == "BioSuite Ultra API"
        assert "swagger" in client.get("/docs").text.lower()
    finally:
        client.app.dependency_overrides.pop(verify_api_key, None)


async def _allow(key: str = ""):
    return "allowed"


def test_read_file_of_a_missing_path_is_404_not_a_stack_trace(client, data_dir):
    response = client.post("/api/v1/file/read", params={"file_path": "absent.bed"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Requested file was not found"}


# ── Provenance ───────────────────────────────────────────────────────────────

def _summary_steps(client):
    """The step count and session id the server's own summary reports."""
    import re

    text = client.get("/api/v1/provenance/summary").json()["summary"]
    return int(re.search(r"Total steps:\s*(\d+)", text).group(1)), text


def test_provenance_record_and_summary_share_one_tracker(client):
    """A tracker created per request made /provenance/summary report zero forever.

    The recorded step has to show up in the very next summary, in the same session.
    """
    before, _ = _summary_steps(client)
    recorded = _post(client, "/api/v1/provenance/record", None,
                     module="sequence", function="gc_content",
                     params={"sequence": "ATCG"}, result_summary="50% GC")

    assert isinstance(recorded["step_id"], int)
    assert recorded["session_id"].startswith("session_")

    after, text = _summary_steps(client)
    assert after == before + 1
    assert recorded["session_id"] in text
    assert "gc_content" in text
