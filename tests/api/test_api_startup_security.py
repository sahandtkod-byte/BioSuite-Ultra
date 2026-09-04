"""Regression tests for secure-by-default startup (BSU-002, BSU-006).

The audited tree shipped working credentials: ``changeme-dev-secret`` signed
admin tokens and ``changeme-dev-password`` logged in, on any deployment that
forgot to set the environment.  The fix is that the process refuses to serve
traffic at all in that state, so "forgot to configure" fails loudly instead of
failing open.

BSU-006 is covered here too: the documented ASGI target
``biosuite.api.server:app`` did not exist, so the container command and the
README instructions both failed.
"""
import subprocess
import sys

import pytest

from biosuite.api import config as api_config
from biosuite.api.server import ensure_safe_to_serve


# ── BSU-006: the documented entry point exists ──────────────────────────────

def test_documented_asgi_target_resolves():
    from biosuite.api import app as package_app
    from biosuite.api.server import app as server_app
    assert server_app is package_app


def test_uvicorn_import_string_resolves():
    """`uvicorn biosuite.api.server:app` must be importable exactly as written."""
    import importlib
    module = importlib.import_module("biosuite.api.server")
    assert callable(getattr(module.app, "__call__", None))


# ── BSU-002: refusal to serve without configuration ─────────────────────────

def _run_server_startup(env_overrides):
    """Start `python -m biosuite.api.server` and capture the refusal."""
    import os
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("BIOSUITE_")}
    env["MPLBACKEND"] = "Agg"
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "biosuite.api.server"],
        capture_output=True, text=True, timeout=120, env=env)


def test_server_refuses_to_start_with_no_configuration():
    result = _run_server_startup({})
    assert result.returncode != 0
    assert "Refusing to start" in result.stderr + result.stdout
    assert "BIOSUITE_API_KEY" in result.stderr + result.stdout


@pytest.mark.parametrize("placeholder", [
    {"BIOSUITE_API_KEY": "changeme-dev-key",
     "BIOSUITE_JWT_SECRET": "strong-enough-secret-value-for-testing-123456",
     "BIOSUITE_ADMIN_PASSWORD": "strong-admin-password"},
    {"BIOSUITE_API_KEY": "strong-enough-api-key-value-for-testing-123456",
     "BIOSUITE_JWT_SECRET": "changeme-dev-secret",
     "BIOSUITE_ADMIN_PASSWORD": "strong-admin-password"},
    {"BIOSUITE_API_KEY": "strong-enough-api-key-value-for-testing-123456",
     "BIOSUITE_JWT_SECRET": "strong-enough-secret-value-for-testing-123456",
     "BIOSUITE_ADMIN_PASSWORD": "changeme-dev-password"},
])
def test_server_refuses_well_known_placeholder_credentials(placeholder):
    result = _run_server_startup(placeholder)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "well-known placeholder" in combined


def test_validate_runtime_config_reports_each_missing_secret(monkeypatch):
    for name in ("BIOSUITE_API_KEY", "BIOSUITE_JWT_SECRET",
                 "BIOSUITE_ADMIN_PASSWORD", "BIOSUITE_ADMIN_PASSWORD_HASH",
                 "BIOSUITE_DEV_MODE"):
        monkeypatch.delenv(name, raising=False)
    problems = api_config.validate_runtime_config()
    text = " ".join(problems)
    assert "BIOSUITE_API_KEY" in text
    assert "BIOSUITE_JWT_SECRET" in text


def test_validate_runtime_config_is_silent_when_configured(monkeypatch):
    monkeypatch.setenv("BIOSUITE_API_KEY", "a-strong-api-key-value-0123456789")
    monkeypatch.setenv("BIOSUITE_JWT_SECRET", "a-strong-jwt-secret-value-0123456789")
    monkeypatch.setenv("BIOSUITE_ADMIN_PASSWORD", "a-strong-admin-password")
    monkeypatch.delenv("BIOSUITE_DEV_MODE", raising=False)
    assert api_config.validate_runtime_config() == []
    ensure_safe_to_serve()          # must not raise


def test_dev_mode_allows_an_insecure_start_but_says_so(monkeypatch, caplog):
    for name in ("BIOSUITE_API_KEY", "BIOSUITE_JWT_SECRET",
                 "BIOSUITE_ADMIN_PASSWORD", "BIOSUITE_ADMIN_PASSWORD_HASH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BIOSUITE_DEV_MODE", "1")
    with caplog.at_level("WARNING"):
        ensure_safe_to_serve()
    assert "BIOSUITE_DEV_MODE" in caplog.text
    assert "insecure" in caplog.text.lower()


# ── password hashing ────────────────────────────────────────────────────────

def test_password_hash_is_salted_and_verifiable():
    first = api_config.hash_password("correct horse battery staple")
    second = api_config.hash_password("correct horse battery staple")
    assert first != second, "hashes must be salted"
    assert api_config.verify_password("correct horse battery staple", first)
    assert api_config.verify_password("correct horse battery staple", second)
    assert not api_config.verify_password("wrong", first)


def test_verify_password_handles_missing_or_malformed_hashes():
    assert api_config.verify_password("x", None) is False
    assert api_config.verify_password("x", "") is False
    assert api_config.verify_password("x", "not-a-hash") is False
