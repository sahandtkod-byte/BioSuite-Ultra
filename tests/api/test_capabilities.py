"""WS2: capability discovery endpoints (manifest, detail, well-known).

Covers key gating, the BIOSUITE_PUBLIC_MANIFEST opt-in, the 404 contract and
the manifest's shape against the WS1 catalog (341 capabilities, 47 modules).
"""
import os

import pytest
from fastapi.testclient import TestClient

from biosuite.api import app
from biosuite.api.capabilities import router as capabilities_router

API_KEY = os.environ.get("BIOSUITE_API_KEY", "changeme-dev-key")
client = TestClient(app, raise_server_exceptions=False)

import biosuite  # noqa: E402  (after app import keeps env setup intact)
from biosuite.capabilities import list_capabilities  # noqa: E402

EXPECTED_MODULES = {capability.module for capability in list_capabilities()}

@pytest.fixture
def auth_headers():
    return {"X-API-Key": API_KEY}


# ── GET /api/v1/capabilities ────────────────────────────────────────────────


class TestCapabilitiesManifest:
    def test_requires_key_by_default(self, monkeypatch):
        monkeypatch.delenv("BIOSUITE_PUBLIC_MANIFEST", raising=False)
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 401

    def test_public_flag_unlocks_discovery(self, monkeypatch):
        monkeypatch.setenv("BIOSUITE_PUBLIC_MANIFEST", "1")
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 200

    def test_public_flag_does_not_unlock_execution(self, monkeypatch):
        monkeypatch.setenv("BIOSUITE_PUBLIC_MANIFEST", "1")
        response = client.post("/api/v1/sequence/gc-content", json={})
        assert response.status_code in (401, 422)

    def test_key_gated_manifest_served_with_key(self, monkeypatch, auth_headers):
        monkeypatch.delenv("BIOSUITE_PUBLIC_MANIFEST", raising=False)
        response = client.get("/api/v1/capabilities", headers=auth_headers)
        assert response.status_code == 200

    def test_manifest_shape(self, auth_headers):
        response = client.get("/api/v1/capabilities", headers=auth_headers)
        body = response.json()
        assert body["total"] == 341
        assert body["mcp_exposed_total"] == 4
        assert len(body["capabilities"]) == 341
        assert body["biosuite_version"] == biosuite.__version__
        assert {entry["module"] for entry in body["capabilities"]} == EXPECTED_MODULES

    def test_manifest_is_deterministic(self, auth_headers):
        first = client.get("/api/v1/capabilities", headers=auth_headers).content
        second = client.get("/api/v1/capabilities", headers=auth_headers).content
        assert first == second


# ── GET /api/v1/capabilities/{id} ───────────────────────────────────────────


class TestCapabilityDetail:
    def test_requires_key_by_default(self, monkeypatch):
        monkeypatch.delenv("BIOSUITE_PUBLIC_MANIFEST", raising=False)
        response = client.get("/api/v1/capabilities/core.sequence.gc_content")
        assert response.status_code == 401

    def test_detail_with_key(self, auth_headers):
        response = client.get("/api/v1/capabilities/sequence.gc_content",
                              headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["capability"]["id"] == "sequence.gc_content"
        assert "availability" in body

    def test_detail_for_each_mcp_tool(self, auth_headers):
        from biosuite.capabilities import list_capabilities
        mcp_ids = [c.id for c in list_capabilities() if c.mcp_exposed]
        for capability_id in mcp_ids:
            response = client.get(f"/api/v1/capabilities/{capability_id}",
                                  headers=auth_headers)
            assert response.status_code == 200, capability_id
            assert response.json()["capability"]["id"] == capability_id

    def test_unknown_id_is_404(self, auth_headers):
        response = client.get("/api/v1/capabilities/no.such.capability",
                              headers=auth_headers)
        assert response.status_code == 404
        assert "no.such.capability" in response.json()["detail"]

    def test_detail_is_consistent_with_manifest(self, auth_headers):
        manifest = client.get("/api/v1/capabilities", headers=auth_headers).json()
        one = client.get(
            f"/api/v1/capabilities/{manifest['capabilities'][0]['id']}",
            headers=auth_headers)
        assert one.status_code == 200
        assert one.json()["capability"]["id"] == manifest['capabilities'][0]['id']


# ── GET /.well-known/biosuite.json ──────────────────────────────────────────


class TestWellKnown:
    def test_same_manifest_at_discovery_path(self, auth_headers):
        well_known = client.get("/.well-known/biosuite.json",
                                headers=auth_headers).json()
        manifest = client.get("/api/v1/capabilities", headers=auth_headers).json()
        assert well_known == manifest

    def test_key_gated_by_default(self, monkeypatch):
        monkeypatch.delenv("BIOSUITE_PUBLIC_MANIFEST", raising=False)
        response = client.get("/.well-known/biosuite.json")
        assert response.status_code == 401

    def test_public_flag_applies(self, monkeypatch):
        monkeypatch.setenv("BIOSUITE_PUBLIC_MANIFEST", "1")
        response = client.get("/.well-known/biosuite.json")
        assert response.status_code == 200
        assert response.json()["total"] == 341

    def test_well_known_not_counted_in_versioned_pin(self):
        versioned = {route.path for route in capabilities_router.routes
                     if route.path.startswith("/api/v1")}
        assert "/.well-known/biosuite.json" not in versioned
        assert "/api/v1/capabilities" in versioned


@pytest.mark.parametrize("value", ["0", "false", "true", "yes", "", " 1 "])
def test_other_public_manifest_values_remain_private(monkeypatch, value):
    monkeypatch.setenv("BIOSUITE_PUBLIC_MANIFEST", value)
    assert client.get("/api/v1/capabilities").status_code == 401


def test_openapi_enrichment_is_idempotent_and_preserves_path_metadata():
    from biosuite.api.capabilities import enrich_openapi

    schema = {"paths": {
        "/sample": {"parameters": [], "summary": "Path summary", "x-metadata": {},
                    "get": {"responses": {"200": {"description": "OK"}}},
                    "post": {"responses": {"201": {"description": "Created"}}},
                    "put": None},
        "/reference": "external reference",
        "/api/v1/modules": {"get": {"responses": {"200": {"description": "OK"}}}},
    }}
    assert enrich_openapi(schema) is schema
    assert enrich_openapi(schema) is schema
    path = schema["paths"]["/sample"]
    assert path["summary"] == "Path summary"
    assert path["x-metadata"] == {}
    for method, status in (("get", "200"), ("post", "201")):
        assert len(path[method]["parameters"]) == 2
        assert "X-Request-ID" in path[method]["responses"][status]["headers"]
    path["get"]["parameters"][0]["schema"]["type"] = "mutated"
    assert path["post"]["parameters"][0]["schema"]["type"] == "string"
    assert schema["paths"]["/api/v1/modules"]["get"]["deprecated"] is True


def test_live_openapi_documents_headers_for_all_operations(auth_headers):
    response = client.get("/openapi.json", headers=auth_headers)
    assert response.status_code == 200
    for path in response.json()["paths"].values():
        for operation in path.values():
            parameters = {p["name"] for p in operation["parameters"]}
            assert {"X-Request-ID", "X-BioSuite-Error-Envelope"} <= parameters
            for result in operation["responses"].values():
                assert "X-Request-ID" in result["headers"]


def test_app_openapi_hook_enriches_public_docs():
    schema = app.openapi()
    assert schema["paths"]["/api/v1/modules"]["get"]["deprecated"] is True
    for path in schema["paths"].values():
        for operation in path.values():
            names = {p["name"] for p in operation["parameters"]}
            assert {"X-Request-ID", "X-BioSuite-Error-Envelope"} <= names
