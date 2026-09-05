"""Regression tests for api/auth.py + api/security.py hardening."""
import time

import pytest
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_api_key_rejects_wrong_key():
    from biosuite.api.auth import verify_api_key
    with pytest.raises(HTTPException) as exc:
        await verify_api_key("definitely-not-the-key")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_api_key_accepts_configured_key():
    from biosuite.api import auth
    assert await auth.verify_api_key(auth.API_KEY) == auth.API_KEY


def test_jwt_roundtrip():
    from biosuite.api.security import create_access_token
    from jose import jwt
    from biosuite.api import security
    token = create_access_token("admin")
    payload = jwt.decode(token, security.JWT_SECRET, algorithms=[security.JWT_ALGORITHM])
    assert payload["sub"] == "admin"
    assert payload["exp"] > time.time()


@pytest.mark.asyncio
async def test_verify_admin_token_missing():
    from biosuite.api.security import verify_admin_token
    with pytest.raises(HTTPException) as exc:
        await verify_admin_token(None)
    assert exc.value.status_code == 401
    assert "Missing" in exc.value.detail


@pytest.mark.asyncio
async def test_verify_admin_token_expired():
    from jose import jwt
    from fastapi.security import HTTPAuthorizationCredentials
    from biosuite.api import security
    token = jwt.encode({"sub": "admin", "exp": time.time() - 10},
                       security.JWT_SECRET, algorithm=security.JWT_ALGORITHM)
    with pytest.raises(HTTPException) as exc:
        await security.verify_admin_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_admin_token_missing_sub():
    from jose import jwt
    from fastapi.security import HTTPAuthorizationCredentials
    from biosuite.api import security
    token = jwt.encode({"exp": time.time() + 300},
                       security.JWT_SECRET, algorithm=security.JWT_ALGORITHM)
    with pytest.raises(HTTPException) as exc:      # old code -> KeyError -> 500
        await security.verify_admin_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_admin_token_valid():
    from fastapi.security import HTTPAuthorizationCredentials
    from biosuite.api import security
    token = security.create_access_token("alice")
    assert await security.verify_admin_token(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)) == "alice"
