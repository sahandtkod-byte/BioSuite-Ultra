"""API key authentication for BioSuite Ultra API.

The key is read from ``BIOSUITE_API_KEY``.  When it is absent the process falls
back to an unpredictable random key (or, in explicit dev mode, to the
documented placeholder) — never to a value an attacker can look up in the
source tree.  See :mod:`biosuite.api.config`.
"""
import hmac
import logging

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from .config import DEV_API_KEY, resolve_secret

logger = logging.getLogger(__name__)

API_KEY, API_KEY_CONFIGURED = resolve_secret(
    "BIOSUITE_API_KEY", DEV_API_KEY, "API key")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Security(api_key_header)):
    """Reject the request unless the ``X-API-Key`` header matches exactly.

    Comparison is constant-time so the endpoint cannot be used as a timing
    oracle for the configured key.
    """
    if not key or not hmac.compare_digest(str(key), API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


def reload_from_environment() -> str:
    """Re-read ``BIOSUITE_API_KEY`` (used by tests and by config reloads)."""
    global API_KEY, API_KEY_CONFIGURED
    API_KEY, API_KEY_CONFIGURED = resolve_secret(
        "BIOSUITE_API_KEY", DEV_API_KEY, "API key")
    return API_KEY


__all__ = ["API_KEY", "API_KEY_CONFIGURED", "verify_api_key", "api_key_header",
           "reload_from_environment"]
