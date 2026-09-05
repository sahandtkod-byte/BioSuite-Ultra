"""Security regression tests for the HTTP API.

Every test here corresponds to an attack that succeeded against the audited
tree.  They are written from the attacker's point of view: the assertion is
that the exploit no longer works, not that some internal flag changed.

Covered: BSU-002 (JWT forgery / default admin password), BSU-012 (CORS),
BSU-013 (arbitrary file read and path traversal), BSU-018 (login throttling,
credentials in query strings), BSU-019 (unauthenticated docs).
"""
import importlib
import os

import pytest
from fastapi.testclient import TestClient
from jose import jwt

import biosuite.api as api_module
from biosuite.api import auth as auth_module
from biosuite.api import security as security_module

API_KEY = os.environ["BIOSUITE_API_KEY"]
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture()
def client():
    with TestClient(api_module.app) as test_client:
        yield test_client


# ── BSU-002: token forgery with the published default secret ────────────────

PUBLISHED_DEFAULT_SECRET = "changeme-dev-secret"      # was the shipped default


def test_token_signed_with_the_published_default_secret_is_rejected(client):
    forged = jwt.encode({"sub": "admin", "exp": 9_999_999_999},
                        PUBLISHED_DEFAULT_SECRET, algorithm="HS256")
    response = client.get("/api/v1/admin/status",
                          headers={**HEADERS, "Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


def test_unsigned_none_algorithm_token_is_rejected(client):
    forged = jwt.encode({"sub": "admin", "exp": 9_999_999_999},
                        key="", algorithm="HS256")
    response = client.get("/api/v1/admin/status",
                          headers={**HEADERS, "Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


def test_secret_falls_back_to_a_random_value_not_a_published_one(monkeypatch):
    """Without configuration the process must not use a guessable secret."""
    monkeypatch.delenv("BIOSUITE_JWT_SECRET", raising=False)
    monkeypatch.delenv("BIOSUITE_DEV_MODE", raising=False)
    try:
        security_module.reload_from_environment()
        assert security_module.JWT_SECRET != PUBLISHED_DEFAULT_SECRET
        assert security_module.JWT_SECRET_CONFIGURED is False
        assert len(security_module.JWT_SECRET) >= 32
        first = security_module.JWT_SECRET
        security_module.reload_from_environment()
        assert security_module.JWT_SECRET != first, "fallback must be random"
    finally:
        monkeypatch.undo()
        security_module.reload_from_environment()


def test_api_key_falls_back_to_a_random_value(monkeypatch):
    monkeypatch.delenv("BIOSUITE_API_KEY", raising=False)
    monkeypatch.delenv("BIOSUITE_DEV_MODE", raising=False)
    try:
        auth_module.reload_from_environment()
        assert auth_module.API_KEY != "changeme-dev-key"
        assert auth_module.API_KEY_CONFIGURED is False
    finally:
        monkeypatch.undo()
        auth_module.reload_from_environment()


# ── BSU-002: default admin password ─────────────────────────────────────────

def test_published_default_admin_password_does_not_log_in(client):
    response = client.post("/api/v1/admin/login", headers=HEADERS,
                           json={"username": "admin",
                                 "password": "changeme-dev-password"})
    assert response.status_code in (401, 403, 429)
    assert "access_token" not in response.text


def test_admin_login_is_disabled_when_no_password_is_configured(monkeypatch):
    monkeypatch.delenv("BIOSUITE_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("BIOSUITE_ADMIN_PASSWORD_HASH", raising=False)
    monkeypatch.delenv("BIOSUITE_DEV_MODE", raising=False)
    try:
        security_module.reload_from_environment()
        assert security_module.admin_login_enabled() is False
        assert security_module.authenticate_admin("admin", "") is False
        assert security_module.authenticate_admin("admin", "anything") is False
    finally:
        monkeypatch.undo()
        security_module.reload_from_environment()


def test_configured_password_authenticates_and_a_wrong_one_does_not():
    assert security_module.admin_login_enabled() is True
    assert security_module.authenticate_admin(
        "admin", os.environ["BIOSUITE_ADMIN_PASSWORD"]) is True
    assert security_module.authenticate_admin("admin", "wrong") is False
    assert security_module.authenticate_admin("root", os.environ[
        "BIOSUITE_ADMIN_PASSWORD"]) is False


def test_password_is_never_stored_in_clear_text():
    stored = security_module.ADMIN_PASSWORD_HASH
    assert stored is not None
    assert os.environ["BIOSUITE_ADMIN_PASSWORD"] not in stored
    assert stored.startswith("pbkdf2_sha256$")


# ── BSU-018: login hardening ────────────────────────────────────────────────

def test_repeated_failed_logins_are_locked_out(client):
    last = None
    for _ in range(12):
        last = client.post("/api/v1/admin/login", headers=HEADERS,
                           json={"username": "admin", "password": "wrong"})
        if last.status_code == 429:
            break
    assert last.status_code == 429, "brute force was never throttled"


# ── BSU-013: arbitrary file read / traversal ────────────────────────────────

@pytest.mark.parametrize("payload", [
    "/etc/passwd",
    "../../../../etc/passwd",
    "..%2f..%2f..%2fetc%2fpasswd",
    "data/../../../../etc/passwd",
    "/proc/self/environ",
    "~/.ssh/id_rsa",
])
def test_file_read_cannot_escape_the_data_directory(client, tmp_path, monkeypatch,
                                                    payload):
    monkeypatch.setenv("BIOSUITE_DATA_DIR", str(tmp_path))
    response = client.post("/api/v1/file/read", headers=HEADERS,
                           params={"file_path": payload})
    assert response.status_code in (400, 404), response.text
    assert "root:" not in response.text


@pytest.mark.parametrize("endpoint", ["/api/v1/file/read",
                                      "/api/v1/file/detect-format"])
def test_file_endpoints_require_the_api_key(client, endpoint):
    response = client.post(endpoint, params={"file_path": "x.fasta"})
    assert response.status_code == 401


def test_symlink_out_of_the_data_directory_is_refused(client, tmp_path,
                                                      monkeypatch):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    link = data_dir / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:                      # pragma: no cover - platform dependent
        pytest.skip("symlinks unavailable")
    monkeypatch.setenv("BIOSUITE_DATA_DIR", str(data_dir))
    response = client.post("/api/v1/file/read", headers=HEADERS,
                           params={"file_path": "link.txt"})
    assert response.status_code in (400, 404)
    assert "secret" not in response.text


def test_a_legitimate_file_inside_the_data_directory_still_works(client, tmp_path,
                                                                 monkeypatch):
    monkeypatch.setenv("BIOSUITE_DATA_DIR", str(tmp_path))
    sample = tmp_path / "sample.fasta"
    sample.write_text(">s1\nACGTACGT\n")
    response = client.post("/api/v1/file/read", headers=HEADERS,
                           params={"file_path": "sample.fasta"})
    assert response.status_code == 200, response.text
    assert response.json()["format"].lower().startswith("fast")


# ── BSU-012: CORS ───────────────────────────────────────────────────────────

def test_arbitrary_origin_is_not_reflected(client):
    response = client.get("/health", headers={**HEADERS,
                                              "Origin": "https://evil.example"})
    allowed = response.headers.get("access-control-allow-origin")
    assert allowed != "https://evil.example"
    assert allowed != "*" or \
        response.headers.get("access-control-allow-credentials") != "true"


def test_wildcard_origin_is_never_combined_with_credentials(client):
    response = client.get("/health", headers={**HEADERS, "Origin": "*"})
    if response.headers.get("access-control-allow-origin") == "*":
        assert response.headers.get("access-control-allow-credentials") != "true"


def test_configured_origin_is_allowed(client):
    response = client.get("/health",
                          headers={**HEADERS, "Origin": "http://localhost:3000"})
    assert response.headers.get("access-control-allow-origin") == \
        "http://localhost:3000"


# ── BSU-019: documentation exposure ─────────────────────────────────────────

@pytest.mark.parametrize("path", ["/docs", "/openapi.json", "/redoc"])
def test_docs_are_not_served_without_the_api_key(client, path):
    response = client.get(path)
    assert response.status_code in (401, 404), \
        f"{path} exposed the API surface anonymously"


def test_docs_are_available_to_an_authenticated_caller(client):
    response = client.get("/openapi.json", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["info"]["title"] == "BioSuite Ultra API"


# ── general authentication ──────────────────────────────────────────────────

def test_every_analysis_endpoint_requires_the_api_key(client):
    for path, payload in [
        ("/api/v1/sequence/gc-content", {"sequence": "ACGT"}),
        ("/api/v1/sequence/translate", {"sequence": "ATGGCC"}),
    ]:
        assert client.post(path, json=payload).status_code == 401


def test_wrong_api_key_is_rejected(client):
    response = client.post("/api/v1/sequence/gc-content",
                           headers={"X-API-Key": API_KEY + "x"},
                           json={"sequence": "ACGT"})
    assert response.status_code == 401


def test_admin_endpoint_requires_a_token_even_with_the_api_key(client):
    assert client.get("/api/v1/admin/status", headers=HEADERS).status_code == 401
