"""API key authentication for BioSuite Ultra API.

NO hardcoded keys. API key is REQUIRED via env var BIOSUITE_API_KEY.
"""
import os
import hmac
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader


def _get_api_key() -> str:
    """Get API key from environment. Raises if not set."""
    key = os.environ.get("BIOSUITE_API_KEY", "")
    if not key:
        raise RuntimeError(
            "BIOSUITE_API_KEY environment variable not set. "
            "Generate one: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
    return key


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Security(api_key_header)):
    """Verify API key using constant-time comparison."""
    if key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    
    expected = _get_api_key()
    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key
