"""WS3: opt-in scoped API keys — full scope/operation matrix and regressions.

Real API requests exercise the scope/operation matrix, alongside direct
verification and configuration unit contracts. Legacy master-key access and
independent JWT admin authentication are preserved.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from biosuite.api import auth
from biosuite.api.security import verify_admin_token
from biosuite.api.scopes import (
    ENV_VAR,
    ScopesConfigurationError,
    expand_scopes,
    key_scope,
    load_scoped_keys,
    parse_scoped_keys,
    scope_satisfies,
)


class _FakeRoute:
    def __init__(self, endpoint):
        self.endpoint = endpoint


class _FakeRequest:
    def __init__(self, endpoint=None):
        self.scope = {"route": _FakeRoute(endpoint)} if endpoint else {}


def _scoped_endpoint():
    return lambda: None


def _reload_scopes(raw):
    auth._pending_scope_env = raw
    import os

    os.environ[ENV_VAR] = raw
    return auth.reload_scoped_keys()


# ── configuration parsing ────────────────────────────────────────────────────


def test_absent_configuration_means_no_scoped_keys(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert load_scoped_keys() == {}


def test_blank_configuration_means_no_scoped_keys():
    assert parse_scoped_keys("   ") == {}
    assert parse_scoped_keys(None) == {}


def test_parse_binds_effective_scopes():
    parsed = parse_scoped_keys("readkey:read;exec:execute;adm:admin")
    assert parsed["readkey"] == frozenset({"read"})
    assert parsed["exec"] == frozenset({"execute", "read"})
    assert parsed["adm"] == frozenset({"admin", "execute", "read"})


def test_inheritance_admin_includes_execute_includes_read():
    assert expand_scopes(["admin"]) >= {"admin", "execute", "read"}
    assert expand_scopes(["execute"]) >= {"execute", "read"}
    assert "read" in expand_scopes(["read"])


@pytest.mark.parametrize("raw", [
    "key-without-separator",
    ":read",
    "key:",
    "key:read;;key2:admin",
    "key:read,key:root,key2:admin",
    "key:write",
    "key: read , , execute",
])
def test_malformed_configuration_raises(raw):
    with pytest.raises(ScopesConfigurationError):
        parse_scoped_keys(raw)


def test_malformed_configuration_fails_closed_at_load(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "key:not-a-scope")
    with pytest.raises(ScopesConfigurationError):
        load_scoped_keys()


def test_malformed_configuration_fails_closed_at_reload(monkeypatch):
    monkeypatch.setenv(ENV_VAR, ";")
    with pytest.raises(ScopesConfigurationError):
        auth.reload_scoped_keys()
    # The previously loaded table stays in force — no half-understood state.
    assert auth.SCOPED_KEYS == {}


def test_duplicate_key_is_rejected():
    with pytest.raises(ScopesConfigurationError):
        parse_scoped_keys("k:read;k:admin")


# ── scope lookup and comparison ──────────────────────────────────────────────


def test_key_scope_uses_effective_grants():
    configured = {"key-one": frozenset({"read"})}
    assert key_scope("key-one", configured) == "read"
    assert key_scope(None, configured) is None
    assert key_scope("", configured) is None
    assert key_scope("wrong", configured) is None


def test_key_scope_continues_after_first_match():
    # The loop must keep scanning after a match: a later, weaker match must not
    # overwrite the strongest grant found so far (branch strongest-update false).
    class _Multi(dict):
        def items(self):
            return iter([("k", frozenset({"admin"})), ("k", frozenset({"read"}))])

    assert key_scope("k", _Multi()) == "admin"


def test_scope_satisfies_matrix():
    assert scope_satisfies("read", "read")
    assert not scope_satisfies("read", "execute")
    assert scope_satisfies("execute", "read")
    assert scope_satisfies("execute", "execute")
    assert not scope_satisfies("execute", "admin")
    assert scope_satisfies("admin", "read")
    assert scope_satisfies("admin", "execute")
    assert scope_satisfies("admin", "admin")
    assert not scope_satisfies(None, "read")
    assert not scope_satisfies("root", "read")
    assert not scope_satisfies("admin", "root")


# ── dependency matrix through verify_api_key ─────────────────────────────────


@pytest.fixture()
def scoped(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "ro:read;rw:execute;root:admin")
    auth.reload_scoped_keys()
    monkeypatch.setattr(auth, "API_KEY", "legacy-master-key")
    yield auth
    monkeypatch.delenv(ENV_VAR, raising=False)
    auth.reload_scoped_keys()


@pytest.mark.asyncio
async def test_legacy_key_keeps_full_access(scoped):
    assert await scoped.verify_api_key("legacy-master-key") == "legacy-master-key"
    assert await scoped.verify_api_key(
        "legacy-master-key", _FakeRequest(_scoped_endpoint())) == "legacy-master-key"


def test_scoped_read_key_denied_on_computation(scoped):
    from fastapi.testclient import TestClient
    from biosuite.api import app

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sequence/gc-content",
            headers={"X-API-Key": "ro"}, json={"sequence": "ATGC"},
        )
    assert response.status_code == 403, response.text


@pytest.mark.asyncio
async def test_legacy_master_key_wins_over_scoped_table(scoped, monkeypatch):
    monkeypatch.setattr(scoped, "API_KEY", "ro")
    assert await scoped.verify_api_key("ro") == "ro"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [None, "", "nope"])
async def test_invalid_keys_rejected_with_401(scoped, bad):
    with pytest.raises(HTTPException) as exc:
        await scoped.verify_api_key(bad, _FakeRequest(_scoped_endpoint()))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_public_endpoints_bypass_scopes(scoped):
    # The fake request carries a route endpoint registered as public.
    from biosuite.api.auth import public_endpoint

    public = public_endpoint(lambda: None)
    assert await scoped.verify_api_key(None, _FakeRequest(public)) is None


@pytest.mark.asyncio
async def test_no_credential_values_in_401_detail(scoped):
    from starlette.requests import Request
    from starlette.routing import Route

    request = Request({"type": "http", "method": "GET",
                       "route": Route("/api/v1/modules", _scoped_endpoint())})
    assert await scoped.verify_api_key("root", request) == "root"
    with pytest.raises(HTTPException) as exc:
        await scoped.verify_api_key("unknown-secret-key")
    assert exc.value.status_code == 401
    assert "unknown-secret-key" not in str(exc.value.detail)


# ── legacy fallback and JWT untouched ────────────────────────────────────────


@pytest.mark.asyncio
async def test_legacy_full_access_without_scopes(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    auth.reload_scoped_keys()
    monkeypatch.setattr(auth, "API_KEY", "only-key")
    assert await auth.verify_api_key("only-key") == "only-key"
    with pytest.raises(HTTPException):
        await auth.verify_api_key("other")


@pytest.mark.asyncio
async def test_scoped_key_cannot_replace_master_key_semantics(scoped, monkeypatch):
    # The master key remains accepted even when a scoped key shares its value:
    # the legacy comparison runs first and grants full access.
    monkeypatch.setattr(scoped, "API_KEY", "root")
    assert await scoped.verify_api_key("root") == "root"


@pytest.mark.asyncio
async def test_jwt_admin_requirement_unchanged(scoped):
    # Scoped keys must not grant admin JWT abilities.
    with pytest.raises(HTTPException) as exc:
        await verify_admin_token(None)
    assert exc.value.status_code == 401
    assert "root" not in str(exc.value.detail)


@pytest.fixture()
def scope_client(scoped):
    from fastapi.testclient import TestClient
    from biosuite.api import app

    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("key", ["ro", "rw", "root", "legacy-master-key"])
@pytest.mark.parametrize("path", [
    "/api/v1/modules", "/api/v1/capabilities",
    "/api/v1/capabilities/sequence.gc_content", "/.well-known/biosuite.json",
    "/api/v1/ready", "/ready", "/api/v1/provenance/summary",
])
def test_scope_read_operation_matrix(scope_client, key, path):
    response = scope_client.get(path, headers={"X-API-Key": key})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("key,status", [
    ("ro", 403), ("rw", 200), ("root", 200), ("legacy-master-key", 200),
    (None, 401), ("invalid", 401),
])
def test_scope_computation_matrix(scope_client, key, status):
    headers = {"X-API-Key": key} if key else {}
    response = scope_client.post(
        "/api/v1/sequence/gc-content", headers=headers, json={"sequence": "ATGC"},
    )
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["gc_percent"] == 50.0


@pytest.mark.parametrize("key,status", [
    ("ro", 403), ("rw", 403), ("root", 200), ("legacy-master-key", 200),
    (None, 200), ("invalid", 401),
])
def test_scope_admin_matrix_with_valid_jwt(scope_client, key, status):
    from biosuite.api.security import create_access_token

    headers = {"Authorization": f"Bearer {create_access_token('scope-test-admin')}"}
    if key:
        headers["X-API-Key"] = key
    response = scope_client.get("/api/v1/admin/status", headers=headers)
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json() == {"admin": "scope-test-admin", "status": "ok"}


@pytest.mark.parametrize("key,status", [
    ("ro", 403), ("rw", 403), ("root", 401), ("legacy-master-key", 401), (None, 401),
])
@pytest.mark.parametrize("bearer", [None, "invalid-jwt"])
def test_no_api_key_bypasses_jwt(scope_client, key, status, bearer):
    headers = {"X-API-Key": key} if key else {}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    assert scope_client.get("/api/v1/admin/status", headers=headers).status_code == status


@pytest.mark.parametrize("path", ["/health", "/ready", "/api/v1/ready"])
def test_public_routes_remain_key_free(scope_client, path):
    assert scope_client.get(path).status_code == 200


@pytest.mark.parametrize("path", [
    "/api/v1/capabilities", "/api/v1/capabilities/sequence.gc_content",
    "/.well-known/biosuite.json",
])
@pytest.mark.parametrize("key", [None, "invalid"])
def test_discovery_invalid_key_and_public_opt_in(scope_client, monkeypatch, path, key):
    headers = {"X-API-Key": key} if key else {}
    monkeypatch.delenv("BIOSUITE_PUBLIC_MANIFEST", raising=False)
    assert scope_client.get(path, headers=headers).status_code == 401
    monkeypatch.setenv("BIOSUITE_PUBLIC_MANIFEST", "1")
    assert scope_client.get(path, headers=headers).status_code == 200


@pytest.mark.parametrize("key,status", [
    ("ro", 403), ("rw", 403), ("root", 403), ("legacy-master-key", 200),
    (None, 401), ("invalid", 401),
])
@pytest.mark.parametrize("method,path", [
    ("GET", "/api/v1/modules/unclassified"),
    ("POST", "/api/v1/sequence/unclassified"),
    ("POST", "/api/v1/modules"),
])
def test_unclassified_routes_default_deny(scoped, key, status, method, path):
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    application = FastAPI(dependencies=[Depends(auth.verify_api_key)])

    @application.api_route(path, methods=[method])
    def unclassified():
        return {"reached": True}

    with TestClient(application) as client:
        headers = {"X-API-Key": key} if key else {}
        assert client.request(method, path, headers=headers).status_code == status


@pytest.mark.parametrize("envelope", [False, True])
def test_scope_denial_request_id_and_error_envelope(scope_client, caplog, envelope):
    headers = {"X-API-Key": "ro", "X-Request-ID": "scope-denial-1"}
    if envelope:
        headers["X-BioSuite-Error-Envelope"] = "1"
    response = scope_client.post(
        "/api/v1/sequence/gc-content", headers=headers, json={"sequence": "ATGC"},
    )
    assert response.status_code == 403
    assert response.headers["X-Request-ID"] == "scope-denial-1"
    if envelope:
        assert response.json() == {"error": {
            "code": "forbidden", "status": 403, "detail": "Insufficient API key scope",
            "request_id": "scope-denial-1",
        }}
    else:
        assert response.json() == {"detail": "Insufficient API key scope"}
    for credential in ("legacy-master-key", "unknown-secret-key"):
        assert credential not in response.text + caplog.text


def test_malformed_config_fails_closed_at_startup():
    import os
    import subprocess
    import sys

    environment = dict(os.environ, BIOSUITE_SCOPED_KEYS="startup-secret:unknown")
    result = subprocess.run(
        [sys.executable, "-c", "import biosuite.api.auth"],
        env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "ScopesConfigurationError" in result.stderr
    assert "startup-secret" not in result.stdout + result.stderr


def test_every_published_operation_has_explicit_scope():
    from biosuite.api import app

    operations = {(method.upper(), path) for path, item in app.openapi()["paths"].items()
                  for method in item if method in {"get", "post", "put", "patch", "delete"}}
    assert operations <= auth.ROUTE_SCOPES.keys()
    assert len({path for _, path in operations if path.startswith("/api/")}) == 41


@pytest.mark.asyncio
@pytest.mark.parametrize("key,grants", [
    ("ro", {"read"}), ("rw", {"read", "execute"}),
    ("root", {"read", "execute", "admin"}),
])
async def test_scope_matrix_covers_every_classified_operation(scoped, key, grants):
    from starlette.requests import Request
    from starlette.routing import Route

    for (method, path), required in scoped.ROUTE_SCOPES.items():
        request = Request({"type": "http", "method": method,
                           "route": Route(path, _scoped_endpoint())})
        if required in grants:
            assert await scoped.verify_api_key(key, request) == key
        else:
            with pytest.raises(HTTPException) as exc:
                await scoped.verify_api_key(key, request)
            assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_scoped_key_without_resolved_route_is_denied(scoped):
    with pytest.raises(HTTPException) as exc:
        await scoped.verify_api_key("root")
    assert exc.value.status_code == 403


def test_discovery_does_not_swallow_scope_denial(scope_client, monkeypatch):
    monkeypatch.setattr(auth, "ROUTE_SCOPES", {})
    response = scope_client.get("/api/v1/capabilities", headers={"X-API-Key": "ro"})
    assert response.status_code == 403


def test_scope_denial_does_not_log_credentials(scope_client, monkeypatch, caplog):
    credential = "scope-denial-test-credential"
    monkeypatch.setattr(auth, "SCOPED_KEYS", {credential: frozenset({"read"})})
    response = scope_client.post(
        "/api/v1/sequence/gc-content", headers={"X-API-Key": credential},
        json={"sequence": "ATGC"},
    )
    assert response.status_code == 403
    assert credential not in response.text + caplog.text


def test_auth_exports_exist():
    assert all(hasattr(auth, name) for name in auth.__all__)
