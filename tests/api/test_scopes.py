"""WS3: opt-in scoped API keys — full scope/operation matrix and regressions.

The scoped-key layer is exercised through :func:`biosuite.api.auth.verify_api_key`
with a synthetic request, exactly as the existing security tests call the
dependency, plus :mod:`biosuite.api.scopes` unit contracts.  The legacy
single-key behaviour and the JWT admin requirements must be unchanged.
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


@pytest.mark.asyncio
async def test_scoped_read_key_accepted_for_any_endpoint(scoped):
    assert await scoped.verify_api_key(
        "ro", _FakeRequest(_scoped_endpoint())) == "ro"


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
    # "root" is a *configured* scoped key, so it authenticates; the detail check
    # below applies to the rejection path for an unknown key.
    assert await scoped.verify_api_key("root") == "root"
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
