"""WS2: readiness endpoint contract.

Missing optional engines must degrade individual capabilities without making
the service unready, so the probe always answers 200 with a structured body.
"""
import os

from fastapi.testclient import TestClient

from biosuite.api import app
from biosuite.api.capabilities import ready_payload
from biosuite.capabilities import discover_availability

client = TestClient(app, raise_server_exceptions=False)


def test_ready_is_anonymous():
    response = client.get("/api/v1/ready")
    assert response.status_code == 200


def test_ready_body_shape():
    body = client.get("/api/v1/ready").json()
    assert body["status"] == "ready"
    caps = body["capabilities"]
    assert caps["total"] == 341
    assert caps["ready"] + caps["degraded"] == 341
    assert isinstance(body["missing_engines"], list)


def test_ready_alias_is_unversioned_and_anonymous():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == client.get("/api/v1/ready").json()


def test_alias_not_in_versioned_route_pin():
    from biosuite.api.capabilities import router as capabilities_router
    versioned = {route.path for route in app.routes
                 if getattr(route, "path", "").startswith("/api/v1")}
    versioned |= {route.path for route in capabilities_router.routes
                  if route.path.startswith("/api/v1")}
    assert "/ready" not in versioned
    assert "/api/v1/ready" in versioned


def test_ready_carries_request_id():
    response = client.get("/api/v1/ready")
    assert response.headers.get("X-Request-ID")


def test_missing_engines_match_discovery():
    payload = ready_payload()
    discovery = discover_availability()
    expected = sorted(
        {probe.name
         for entry in discovery.values()
         for probe in entry.tools + entry.packages
         if not probe.available})
    assert payload["missing_engines"] == expected


def test_ready_counts_match_discovery():
    payload = ready_payload()
    discovery = discover_availability()
    assert payload["capabilities"]["ready"] == sum(
        1 for entry in discovery.values() if entry.ready)
