"""Every pydantic validator on the API has to reject with a message that says why.

The request models are the boundary between an HTTP client and code that indexes
arrays by position, so each rule below exists to stop a 500 — or, worse, an answer
computed from the wrong shape.  These tests pin the status code *and* the wording,
because a 422 that says "1 validation error" and nothing else is not much help to
whoever sent the request.
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


@pytest.mark.parametrize(("payload", "message"), [
    ({"sequence": "ATCG", "frame": 4}, "frame must be one of"),
    ({"sequence": "ATCG", "frame": 0}, "frame must be one of"),
    ({"sequence": "ATCG", "frame": -4}, "frame must be one of"),
])
def test_translation_frame_is_validated(client, payload, message):
    response = client.post("/api/v1/sequence/translate", json=payload)

    assert response.status_code == 422
    assert message in response.text


def test_an_empty_sequence_is_rejected(client):
    response = client.post("/api/v1/sequence/gc-content", json={"sequence": ""})

    assert response.status_code == 422
    assert "at least 1 character" in response.text


VOLCANO_OK = {"log2fc": [1.0, -1.0], "pvalues": [0.01, 0.02]}


@pytest.mark.parametrize(("payload", "message"), [
    ({**VOLCANO_OK, "pvalues": [0.01]}, "must have the same length"),
    ({**VOLCANO_OK, "gene_names": ["A"]}, "must match log2fc"),
])
def test_volcano_lengths_are_validated(client, payload, message):
    response = client.post("/api/v1/plotting/volcano", json=payload)

    assert response.status_code == 422
    assert message in response.text


@pytest.mark.parametrize(("payload", "message"), [
    ({}, "Field required"),
    ({"data": []}, "data must contain at least one sample"),
    ({"data": [[1.0, 2.0], [3.0]]}, "same number of features"),
    ({"data": [[]]}, "at least one feature"),
    ({"data": [[1.0, 2.0], [3.0, 4.0]], "n_components": 3}, "cannot exceed min(n_samples"),
    ({"data": [[1.0, 2.0], [3.0, 4.0]], "labels": ["only-one"]}, "one entry per sample"),
])
def test_pca_matrix_shape_is_validated(client, payload, message):
    response = client.post("/api/v1/plotting/pca", json=payload)

    assert response.status_code == 422
    assert message in response.text


@pytest.mark.parametrize(("payload", "message"), [
    ({"chromosomes": ["1", "2"], "positions": [10], "pvalues": [0.1, 0.2]},
     "must all have the same length"),
    ({"chromosomes": ["1"], "positions": [10, 20], "pvalues": [0.1]},
     "(got 1, 2, 1)"),
])
def test_manhattan_lengths_are_validated(client, payload, message):
    response = client.post("/api/v1/plotting/manhattan", json=payload)

    assert response.status_code == 422
    assert message in response.text


@pytest.mark.parametrize(("payload", "message"), [
    ({"query": ""}, "at least 1 character"),
    ({"query": "ATCG", "evalue": 0}, "greater_than"),
    ({"query": "ATCG", "evalue": -1}, "greater_than"),
])
def test_blast_request_fields_are_validated(client, payload, message):
    response = client.post("/api/v1/blast/search", json=payload)

    assert response.status_code == 422
    assert message in response.text


@pytest.mark.parametrize(("payload", "message"), [
    ({"sequence": "ATCG", "guide_length": 0}, "greater_than_equal"),
    ({"sequence": "ATCG", "guide_length": 51}, "less_than_equal"),
    ({"sequence": "ATCG", "max_guides": 0}, "greater_than_equal"),
])
def test_crispr_request_fields_are_validated(client, payload, message):
    response = client.post("/api/v1/crispr/design", json=payload)

    assert response.status_code == 422
    assert message in response.text


@pytest.mark.parametrize(("payload", "message"), [
    ({}, "Field required"),
    ({"counts": {"G1": [1]}, "conditions": ["A", "B"]}, "must match the number of"),
    ({"counts": {"G1": [1, 2]}, "conditions": ["A", "A"]}, "exactly two distinct groups"),
])
def test_differential_expression_request_is_validated(client, payload, message):
    response = client.post("/api/v1/expression/differential", json=payload)

    assert response.status_code == 422
    assert message in response.text


def test_admin_login_needs_both_credentials(client):
    for payload in ({}, {"username": "admin"}, {"password": "pw"}):
        response = client.post("/api/v1/admin/login", json=payload)
        assert response.status_code == 422, payload
