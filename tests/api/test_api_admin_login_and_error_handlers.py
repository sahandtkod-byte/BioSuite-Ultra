"""Admin login, the login lockout, and the JSON error handlers.

The login route carries the API's security decisions — credentials never in a URL, no
more than a handful of failures per address, and no login at all when no admin password
exists — and each of those is a branch that only fires on one specific client
behaviour.  slowapi's own rate limiting is switched off here so the tests exercise
*this* module's lockout counter; the limiter is covered by
``test_api_security_regression.py``.
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from biosuite import api as api_module

#: The address TestClient reports for itself; every lockout in here is for this key.
CLIENT_KEY = "testclient"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_module.limiter, "enabled", False)
    monkeypatch.setattr(api_module, "_login_failures", {})
    c = TestClient(api_module.app, raise_server_exceptions=False)
    c.headers["X-API-Key"] = os.environ.get("BIOSUITE_API_KEY", "test-api-key-not-a-real-secret")
    return c


def _request(path):
    """Just enough of a Request for the handlers, which only read ``url.path``."""
    return SimpleNamespace(url=SimpleNamespace(path=path))


def _detail(response):
    return json.loads(response.body)["detail"]


# ── Login ────────────────────────────────────────────────────────────────────

def test_login_is_refused_until_an_admin_password_exists(client, monkeypatch):
    monkeypatch.setattr(api_module, "admin_login_enabled", lambda: False)

    response = client.post("/api/v1/admin/login",
                           json={"username": "admin", "password": "whatever"})

    assert response.status_code == 503
    assert "BIOSUITE_ADMIN_PASSWORD" in response.json()["detail"]


def test_wrong_credentials_are_rejected_and_counted(client, monkeypatch):
    monkeypatch.setattr(api_module, "admin_login_enabled", lambda: True)
    monkeypatch.setattr(api_module, "authenticate_admin", lambda u, p: False)

    response = client.post("/api/v1/admin/login",
                           json={"username": "admin", "password": "wrong"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid admin credentials"}
    assert len(api_module._login_failures[CLIENT_KEY]) == 1


def test_repeated_failures_lock_the_address_out(client, monkeypatch):
    monkeypatch.setattr(api_module, "admin_login_enabled", lambda: True)
    monkeypatch.setattr(api_module, "authenticate_admin", lambda u, p: False)

    for _ in range(api_module._LOGIN_MAX_FAILURES):
        assert client.post("/api/v1/admin/login",
                           json={"username": "admin", "password": "wrong"}).status_code == 401

    locked = client.post("/api/v1/admin/login",
                         json={"username": "admin", "password": "still-wrong"})
    assert locked.status_code == 429
    assert "Too many failed login attempts" in locked.json()["detail"]


def test_a_good_login_clears_the_failure_history(client, monkeypatch):
    monkeypatch.setattr(api_module, "admin_login_enabled", lambda: True)
    monkeypatch.setattr(api_module, "authenticate_admin", lambda u, p: p == "s3cret")
    monkeypatch.setattr(api_module, "create_access_token", lambda user: f"token-for-{user}")

    assert client.post("/api/v1/admin/login",
                       json={"username": "admin", "password": "wrong"}).status_code == 401
    ok = client.post("/api/v1/admin/login",
                     json={"username": "admin", "password": "s3cret"})

    assert ok.status_code == 200
    assert ok.json() == {"access_token": "token-for-admin", "token_type": "bearer"}
    assert CLIENT_KEY not in api_module._login_failures


def test_credentials_in_the_query_string_still_work_but_are_warned_about(
        client, monkeypatch, caplog):
    """URLs end up in proxy and browser logs, so the JSON body is the supported way."""
    monkeypatch.setattr(api_module, "admin_login_enabled", lambda: True)
    monkeypatch.setattr(api_module, "authenticate_admin", lambda u, p: (u, p) == ("admin", "pw"))

    with caplog.at_level("WARNING", logger="biosuite.api"):
        response = client.post("/api/v1/admin/login",
                               params={"username": "admin", "password": "pw"})

    assert response.status_code == 200
    assert any("deprecated query-parameter" in r.message for r in caplog.records)


def test_an_empty_login_attempt_is_rejected_without_touching_the_password(
        client, monkeypatch):
    """No body and no query parameters: the credentials are simply absent."""
    checks = []
    monkeypatch.setattr(api_module, "admin_login_enabled", lambda: True)
    monkeypatch.setattr(api_module, "authenticate_admin",
                        lambda u, p: checks.append((u, p)) or False)

    response = client.post("/api/v1/admin/login")

    assert response.status_code == 401
    assert checks == [("", "")]


def test_admin_status_accepts_the_token_login_issued(client):
    from biosuite.api.security import create_access_token

    response = client.get("/api/v1/admin/status",
                          headers={"Authorization": f"Bearer {create_access_token('admin')}"})

    assert response.status_code == 200
    assert response.json() == {"admin": "admin", "status": "ok"}


def test_admin_status_rejects_a_request_without_a_token(client):
    response = client.get("/api/v1/admin/status")

    assert response.status_code == 401


# ── Error handlers ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_value_error_handler_reports_invalid_input():
    response = await api_module._value_error_handler(
        _request("/api/v1/plotting/volcano"), ValueError("lengths differ"))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 422
    assert _detail(response) == "Invalid input: lengths differ"


@pytest.mark.asyncio
async def test_not_found_handler_says_nothing_about_the_path():
    """The message must not echo a client-supplied path back at them."""
    response = await api_module._not_found_handler(
        _request("/api/v1/file/read"), FileNotFoundError("/etc/shadow"))

    assert response.status_code == 404
    assert _detail(response) == "Requested file was not found"


@pytest.mark.asyncio
async def test_key_error_handler_names_the_missing_field():
    response = await api_module._key_error_handler(
        _request("/api/v1/metagenomics/classify-16s"), KeyError("sequence"))

    assert response.status_code == 422
    assert _detail(response) == "Missing required field: 'sequence'"


@pytest.mark.asyncio
async def test_index_error_handler_reports_array_mismatch():
    response = await api_module._index_error_handler(
        _request("/api/v1/gwas/run"), IndexError("out of bounds"))

    assert response.status_code == 422
    assert _detail(response) == "Input arrays have inconsistent lengths"


# ── Server entry point ───────────────────────────────────────────────────────

@pytest.fixture
def uvicorn_run(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    return calls


def test_server_main_binds_loopback_unless_told_otherwise(uvicorn_run, monkeypatch):
    from biosuite.api import server

    for variable in ("BIOSUITE_API_HOST", "BIOSUITE_API_PORT",
                     "BIOSUITE_API_RELOAD", "BIOSUITE_DEV_MODE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(server, "ensure_safe_to_serve", lambda: None)

    assert server.main() == 0

    (target, kwargs), = uvicorn_run
    assert target == ("biosuite.api:app",)
    assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 8000
    assert kwargs["reload"] is False, "auto-reload must not follow a production default"


def test_server_main_honours_the_host_port_and_reload_configuration(
        uvicorn_run, monkeypatch):
    from biosuite.api import server

    monkeypatch.setenv("BIOSUITE_API_HOST", "0.0.0.0")
    monkeypatch.setenv("BIOSUITE_API_PORT", "9000")
    monkeypatch.setenv("BIOSUITE_DEV_MODE", "1")
    monkeypatch.setenv("BIOSUITE_API_RELOAD", "1")
    monkeypatch.setattr(server, "ensure_safe_to_serve", lambda: None)

    server.main()

    (_, kwargs), = uvicorn_run
    assert kwargs["host"] == "0.0.0.0" and kwargs["port"] == 9000
    assert kwargs["reload"] is True


def test_server_main_never_starts_the_server_after_a_failed_safety_check(
        uvicorn_run, monkeypatch):
    """``ensure_safe_to_serve`` exits the process; main must not reach uvicorn."""
    from biosuite.api import server

    def refuse():
        raise SystemExit(2)

    monkeypatch.setattr(server, "ensure_safe_to_serve", refuse)

    with pytest.raises(SystemExit):
        server.main()

    assert not uvicorn_run


# ── The startup safety gate itself ───────────────────────────────────────────

def test_the_gate_lets_a_complete_configuration_through(monkeypatch, caplog):
    from biosuite.api import server

    monkeypatch.setattr(server, "validate_runtime_config", lambda: [])
    monkeypatch.delenv("BIOSUITE_DEV_MODE", raising=False)

    with caplog.at_level("WARNING", logger="biosuite.api.server"):
        server.ensure_safe_to_serve()          # must not raise

    assert not caplog.records


def test_the_gate_refuses_an_unsafe_configuration_outside_dev_mode(monkeypatch):
    from biosuite.api import server

    monkeypatch.setattr(server, "validate_runtime_config",
                        lambda: ["BIOSUITE_API_KEY is not set"])
    monkeypatch.delenv("BIOSUITE_DEV_MODE", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        server.ensure_safe_to_serve()

    assert "Refusing to start" in str(exit_info.value)
    assert "BIOSUITE_API_KEY is not set" in str(exit_info.value)


def test_the_gate_warns_but_starts_when_dev_mode_is_explicit(monkeypatch, caplog):
    from biosuite.api import server

    monkeypatch.setattr(server, "validate_runtime_config",
                        lambda: ["BIOSUITE_JWT_SECRET is not set"])
    monkeypatch.setattr(server, "dev_mode", lambda: True)

    with caplog.at_level("WARNING", logger="biosuite.api.server"):
        server.ensure_safe_to_serve()          # warned about, not fatal

    assert "insecure configuration" in caplog.text
    assert "BIOSUITE_JWT_SECRET is not set" in caplog.text


# ── HWE result contract ──────────────────────────────────────────────────────

def test_hwe_answers_in_json_native_types(client):
    """The endpoint hands the core's dict straight back; this is what keeps that safe."""
    body = client.post("/api/v1/popgen/hwe", json={"AA": 30, "Aa": 50, "aa": 20}).json()

    assert type(body["p_value"]) is float
    assert type(body["in_hwe"]) is bool
    assert all(type(v) is float for v in body["expected"].values())
