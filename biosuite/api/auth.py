"""API key authentication for BioSuite Ultra API.

The key is read from ``BIOSUITE_API_KEY``.  When it is absent the process falls
back to an unpredictable random key (or, in explicit dev mode, to the
documented placeholder) — never to a value an attacker can look up in the
source tree.  See :mod:`biosuite.api.config`.
"""
import hmac
from typing import Any, Callable, Set

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from ..core.log import get_logger
from .config import DEV_API_KEY, resolve_secret
from .scopes import ENV_VAR, load_scoped_keys, scope_satisfies

logger = get_logger(__name__)

API_KEY, API_KEY_CONFIGURED = resolve_secret(
    "BIOSUITE_API_KEY", DEV_API_KEY, "API key")

#: Effective ``{key: scopes}`` table.  Empty unless ``BIOSUITE_SCOPED_KEYS`` is
#: configured; an empty table preserves the legacy single-key full access.
SCOPED_KEYS = load_scoped_keys()


def reload_scoped_keys():
    """Re-read ``BIOSUITE_SCOPED_KEYS`` (used by tests and config reloads)."""
    global SCOPED_KEYS
    SCOPED_KEYS = load_scoped_keys()
    return SCOPED_KEYS

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

#: Routes that need no key (FIX-23).  Matched on the resolved endpoint object, not
#: on the request path, and everything not registered here stays behind the key.
PUBLIC_ENDPOINTS: Set[Callable[..., Any]] = set()


def public_endpoint(func):
    """Mark an endpoint as key-free and return it unchanged."""
    PUBLIC_ENDPOINTS.add(func)
    return func


def key_matches(key) -> bool:
    """True when *key* is the configured API key.  Constant-time comparison."""
    return bool(key) and hmac.compare_digest(str(key), API_KEY)


async def verify_api_key(key: str = Security(api_key_header),
                         request: Request = None):
    """Reject the request unless the ``X-API-Key`` header matches exactly.

    Key-free routes are exempted here rather than by removing the global
    dependency, so a new route is authenticated by default.  ``request`` is
    optional, and last, so the dependency keeps its original positional
    contract when it is called as a plain function.
    """
    scope = getattr(request, "scope", None) or {}
    if getattr(scope.get("route"), "endpoint", None) in PUBLIC_ENDPOINTS:
        return key
    if not key_matches(key):
        if SCOPED_KEYS:
            for configured in SCOPED_KEYS:
                if hmac.compare_digest(str(key or ""), configured):
                    return key
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


def reload_from_environment() -> str:
    """Re-read ``BIOSUITE_API_KEY`` (used by tests and by config reloads)."""
    global API_KEY, API_KEY_CONFIGURED
    API_KEY, API_KEY_CONFIGURED = resolve_secret(
        "BIOSUITE_API_KEY", DEV_API_KEY, "API key")
    reload_scoped_keys()
    return API_KEY


__all__ = ["API_KEY", "API_KEY_CONFIGURED", "PUBLIC_ENDPOINTS", "SCOPED_KEYS",
           "api_key_header", "key_matches", "public_endpoint",
           "reload_from_environment", "reload_scoped_keys", "scope_satisfies",
           "verify_api_key"]
