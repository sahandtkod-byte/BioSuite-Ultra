"""The expression endpoints must normalise the library, not the gene.

``pd.DataFrame(counts)`` turns ``{gene: [counts]}`` inside out — the genes become
columns — so CPM scaled every *gene* to one million counts and TPM divided the samples
by the gene lengths.  The numbers looked plausible and no test called these three
endpoints, which is how the transpose that FIX-4 removed from
``/expression/differential`` stayed alive in the normalisation endpoints next to it.
Every expectation below is derived from the definition of the measure, not from the
implementation.
"""
import os

import pytest
from fastapi.testclient import TestClient

from biosuite.api import app

# 2 genes x 2 samples.  Sample 1 holds 40 counts, sample 2 holds 60.
COUNTS = {"G1": [10, 20], "G2": [30, 40]}


@pytest.fixture(scope="module")
def client():
    c = TestClient(app, raise_server_exceptions=False)
    c.headers["X-API-Key"] = os.environ.get("BIOSUITE_API_KEY", "test-api-key-not-a-real-secret")
    return c


def _normalised(client, path, payload):
    response = client.post(path, json=payload)
    assert response.status_code == 200, response.text
    return response.json()["normalized"]


def _assert_close_per_gene(actual, expected):
    """Compare {gene: [values]} numerically — pytest.approx cannot take a dict of lists."""
    assert set(actual) == set(expected)
    for gene, values in expected.items():
        assert actual[gene] == pytest.approx(values, rel=1e-6)


def _error_text(response):
    """The message of a 422, whichever of the two shapes FastAPI used to answer with.

    A validator rejects with a list of ``{"loc": ..., "msg": ...}`` entries, while an
    ``HTTPException`` carries one plain string; both say the same thing to a client.
    """
    detail = response.json()["detail"]
    if isinstance(detail, list):
        return " ".join(str(part.get("msg", part)) for part in detail)
    return str(detail)


# ── CPM ──────────────────────────────────────────────────────────────────────

def test_cpm_scales_each_sample_to_one_million(client):
    """G1 is 10 of sample 1's 40 counts → 0.25 × 1e6; of sample 2's 60 → 1/3 × 1e6."""
    normalized = _normalised(client, "/api/v1/expression/normalize/cpm", COUNTS)

    _assert_close_per_gene(normalized, {"G1": [250_000.0, 333_333.333333],
                                       "G2": [750_000.0, 666_666.666667]})


def test_cpm_column_sums_are_the_library_size(client):
    """The invariant CPM exists for: each sample, not each gene, sums to 1e6."""
    normalized = _normalised(client, "/api/v1/expression/normalize/cpm", COUNTS)
    genes = sorted(normalized)

    assert sum(normalized[g][0] for g in genes) == pytest.approx(1e6)
    assert sum(normalized[g][1] for g in genes) == pytest.approx(1e6)


def test_cpm_response_is_keyed_by_gene_name(client):
    """A {0: [...], 1: [...]} answer would mean the orientation drifted back."""
    normalized = _normalised(client, "/api/v1/expression/normalize/cpm", COUNTS)

    assert set(normalized) == set(COUNTS)
    assert all(len(v) == 2 for v in normalized.values())


# ── TPM ──────────────────────────────────────────────────────────────────────

def test_tpm_divides_by_gene_length_before_scaling(client):
    """G2 is twice as long as G1, so its 30/40 counts land at 600k/500k TPM.

    RPK: G1 [10 000, 20 000], G2 [15 000, 20 000]; the columns then sum to 25 000 and
    40 000, so the scaling factors are 0.025 and 0.04.
    """
    normalized = _normalised(client, "/api/v1/expression/normalize/tpm",
                             {"counts": COUNTS, "gene_lengths": [1.0, 2.0]})

    _assert_close_per_gene(normalized, {"G1": [400_000.0, 500_000.0],
                                       "G2": [600_000.0, 500_000.0]})
    assert normalized["G1"][0] + normalized["G2"][0] == pytest.approx(1e6)


def test_tpm_rejects_gene_lengths_that_do_not_match_the_genes(client):
    response = client.post("/api/v1/expression/normalize/tpm",
                           json={"counts": COUNTS, "gene_lengths": [1.0]})

    assert response.status_code == 422
    assert "one entry per gene" in response.json()["detail"]


# ── DESeq2 median-of-ratios ──────────────────────────────────────────────────

def test_deseq2_removes_a_uniform_difference_in_depth(client):
    """Sample 2 is exactly twice sample 1, so normalised counts must be identical.

    A size factor that does its job is invisible in this test otherwise: raw counts
    differ by a factor of two and the normalised ones must not.
    """
    doubled = {"G1": [10, 20], "G2": [30, 60]}
    normalized = _normalised(client, "/api/v1/expression/normalize/deseq2", doubled)

    _assert_close_per_gene(normalized, {"G1": [15.0, 15.0], "G2": [45.0, 45.0]})


@pytest.mark.parametrize("bad", [
    {"G1": [1, 2], "G2": [3]},      # ragged rows
    {},                              # no genes at all
    {"G1": []},                      # no samples
])
def test_normalizers_reject_matrices_they_cannot_build(client, bad):
    for path in ("/api/v1/expression/normalize/cpm", "/api/v1/expression/normalize/deseq2"):
        response = client.post(path, json=bad)
        assert response.status_code == 422, (path, bad, response.text)
        assert "detail" in response.json()


# ── Differential expression ──────────────────────────────────────────────────

DE_COUNTS = {"UP": [10, 12, 100, 120], "DOWN": [100, 120, 10, 12], "FLAT": [50, 50, 50, 50]}


def test_differential_expression_returns_one_record_per_gene(client):
    """The endpoint's job is the marshalling: genes in, genes out, tenfold up/down.

    Before FIX-4 the DataFrame was transposed and every request died with
    "positional indexers are out-of-bounds" inside pandas.
    """
    response = client.post("/api/v1/expression/differential",
                           json={"counts": DE_COUNTS, "conditions": ["A", "A", "B", "B"]})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["num_genes"] == 3
    assert body["num_upregulated"] == 1
    assert body["num_downregulated"] == 1
    by_gene = {r["gene"]: r for r in body["results"]}
    assert set(by_gene) == set(DE_COUNTS)
    assert by_gene["UP"]["log2FC"] > 1
    assert by_gene["DOWN"]["log2FC"] < -1
    assert by_gene["UP"]["log2FC"] == pytest.approx(-by_gene["DOWN"]["log2FC"])
    assert by_gene["FLAT"]["log2FC"] == 0.0


@pytest.mark.parametrize("method", ["ttest", "nb"])
def test_differential_expression_accepts_both_methods(client, method):
    response = client.post("/api/v1/expression/differential",
                           json={"counts": DE_COUNTS, "conditions": ["A", "A", "B", "B"],
                                 "method": method})

    assert response.status_code == 200, response.text
    assert response.json()["num_genes"] == 3


def test_differential_expression_demands_replicates_instead_of_returning_nan(client):
    """One sample per group has no within-group variance.

    The request used to be accepted, produce NaN, and then surface as
    ``Invalid input: Out of range float values are not JSON compliant: nan`` — a
    422 that blamed the client's formatting for what is a study-design problem.
    """
    response = client.post("/api/v1/expression/differential",
                           json={"counts": COUNTS, "conditions": ["A", "B"]})

    assert response.status_code == 422
    assert "at least two samples" in _error_text(response)
    assert "Out of range float values" not in _error_text(response)


@pytest.mark.parametrize(("counts", "conditions"), [
    (COUNTS, ["A", "A", "B"]),               # conditions do not match the width
    (COUNTS, ["A", "A"]),                     # one group is not a comparison
    ({"G1": [1, 2, 3, 4], "G2": [5]}, ["A", "A", "B", "B"]),   # ragged
    ({}, ["A", "B"]),                         # empty
])
def test_differential_expression_rejects_malformed_matrices(client, counts, conditions):
    response = client.post("/api/v1/expression/differential",
                           json={"counts": counts, "conditions": conditions})

    assert response.status_code == 422
