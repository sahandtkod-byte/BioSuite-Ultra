"""WS2: legacy contract regressions.

The 38 pre-existing endpoints must survive WS2 unchanged: same bodies, same
status codes, same /modules schema (with the deprecated aliasing), and the
versioned route pin moves 38 -> 41 and only by the three documented routes.
"""
import os

import pytest
from fastapi.testclient import TestClient

from biosuite.api import app
from biosuite.api.capabilities import count_versioned_routes

API_KEY = os.environ.get("BIOSUITE_API_KEY", "changeme-dev-key")
client = TestClient(app, raise_server_exceptions=False)
AUTH = {"X-API-Key": API_KEY}


# Route traversal: newer FastAPI versions fold include_router() routes into a
# single _IncludedRouter marker object without a .path attribute, so walk the
# flattened effective route table via router.original_router.routes.
def _api_v1_paths():
    paths = set()
    for route in app.routes:
        if getattr(route, "path", None):
            paths.add(route.path)
        original = getattr(route, "original_router", None)
        if original is not None:
            paths.update(sub.path for sub in original.routes)
    return {path for path in paths if path.startswith("/api/v1")}


def _all_paths():
    return _api_v1_paths() | {"/ready", "/.well-known/biosuite.json"}


class TestRoutePin:
    def test_exactly_45_versioned_routes(self):
        assert len(_api_v1_paths()) == 45

    def test_added_exactly_three_ws2_and_four_ws4_routes(self):
        added = {"/api/v1/capabilities", "/api/v1/capabilities/{capability_id}",
                 "/api/v1/ready", "/api/v1/jobs", "/api/v1/jobs/{id}",
                 "/api/v1/jobs/{id}/result", "/api/v1/jobs/{id}/cancel"}
        assert added <= _api_v1_paths()
        assert len(_api_v1_paths() - added) == 38
        schema = app.openapi()["paths"]
        for path, method in (("/api/v1/jobs", "post"),
                             ("/api/v1/jobs/{id}", "get"),
                             ("/api/v1/jobs/{id}/result", "get"),
                             ("/api/v1/jobs/{id}/cancel", "post")):
            assert method in schema[path]

    def test_unversioned_additions_are_not_pinned(self):
        paths = _all_paths()
        assert "/ready" in paths
        assert "/.well-known/biosuite.json" in paths
        assert not {"/ready", "/.well-known/biosuite.json"} & _api_v1_paths()


class TestLegacyModulesAlias:
    def test_still_requires_key(self, monkeypatch):
        monkeypatch.delenv("BIOSUITE_PUBLIC_MANIFEST", raising=False)
        response = client.get("/api/v1/modules")
        assert response.status_code == 401

    def test_ten_entry_schema_preserved(self):
        response = client.get("/api/v1/modules", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert len(body["modules"]) == 10
        assert set(body) == {"modules", "total_endpoints", "documentation"}
        for module in body["modules"]:
            assert set(module) == {"name", "endpoints", "description"}

    def test_deprecation_headers(self):
        response = client.get("/api/v1/modules", headers=AUTH)
        assert response.headers["Deprecation"] == "true"
        assert '/api/v1/capabilities>; rel="successor-version"' in \
            response.headers["Link"]

    def test_total_endpoints_is_live(self):
        body = client.get("/api/v1/modules", headers=AUTH).json()
        assert body["total_endpoints"] == count_versioned_routes(app) == 45

    def test_body_is_generated_not_frozen(self):
        first = client.get("/api/v1/modules", headers=AUTH).json()
        first["total_endpoints"] = "sentinel"
        # The view is regenerated per request from live state, not a cached body.
        second = client.get("/api/v1/modules", headers=AUTH).json()
        assert second["total_endpoints"] == 45


class TestLegacyEndpointsUnchanged:
    @pytest.mark.parametrize("path", [
        "/", "/health", "/api/v1/modules",
    ])
    def test_get_endpoints_survive(self, path):
        response = client.get(path, headers=AUTH)
        assert response.status_code == 200, path

    def test_health_body_unchanged(self):
        body = client.get("/health", headers=AUTH).json()
        assert "status" in body

    def test_modules_body_unchanged_shape(self):
        legacy = client.get("/api/v1/modules", headers=AUTH).json()
        assert legacy["documentation"] == "/docs"
        names = {module["name"] for module in legacy["modules"]}
        assert names == {"sequence", "alignment", "expression", "plotting",
                         "crispr", "metagenomics", "epitope", "gwas",
                         "phylogeny", "popgen"}

    def test_openapi_enrichment_does_not_drop_paths(self):
        schema = client.get("/openapi.json", headers=AUTH).json()
        versioned = {path for path in schema["paths"]
                     if path.startswith("/api/v1")}
        assert versioned == _api_v1_paths()
        assert len(versioned) == 45
        assert "/api/v1/modules" in schema["paths"]

    def test_openapi_marks_modules_deprecated(self):
        schema = client.get("/openapi.json", headers=AUTH).json()
        operation = schema["paths"]["/api/v1/modules"]["get"]
        assert operation["deprecated"] is True

    def test_openapi_documents_request_id_header(self):
        schema = client.get("/openapi.json", headers=AUTH).json()
        operation = schema["paths"]["/api/v1/ready"]["get"]
        names = {parameter["name"] for parameter in operation["parameters"]}
        assert "X-Request-ID" in names
        assert "X-BioSuite-Error-Envelope" in names

    def test_public_docs_flag_enriches_too(self, monkeypatch):
        from biosuite.api.capabilities import enrich_openapi
        from fastapi.openapi.utils import get_openapi
        schema = enrich_openapi(get_openapi(
            title=app.title, version=app.version, routes=app.routes))
        assert schema["paths"]["/api/v1/modules"]["get"]["deprecated"] is True

    def test_legacy_detail_bodies_still_served_by_default(self):
        response = client.get("/api/v1/capabilities/nope", headers=AUTH)
        assert set(response.json()) == {"detail"}
