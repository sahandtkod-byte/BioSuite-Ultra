"""JWT authentication for BioSuite Ultra admin routes.

The signing secret comes from ``BIOSUITE_JWT_SECRET``.  When it is absent the
process signs with an unpredictable random secret (or, in explicit dev mode,
with the documented placeholder), so tokens cannot be forged offline from the
published source.  The admin password is never stored or compared in plain
text — only a PBKDF2-SHA256 hash is kept in memory, and login is disabled
outright when no password is configured.  See :mod:`biosuite.api.config`.
"""
import hmac
import logging
import os
import time
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from .config import (
    DEV_JWT_SECRET,
    resolve_admin_password_hash,
    resolve_secret,
    verify_password,
)

logger = logging.getLogger(__name__)

JWT_SECRET, JWT_SECRET_CONFIGURED = resolve_secret(
    "BIOSUITE_JWT_SECRET", DEV_JWT_SECRET, "JWT signing secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 3600

ADMIN_USERNAME = os.environ.get("BIOSUITE_ADMIN_USER", "admin")
ADMIN_PASSWORD_HASH, ADMIN_PASSWORD_CONFIGURED = resolve_admin_password_hash()

bearer_scheme = HTTPBearer(auto_error=False)


def admin_login_enabled() -> bool:
    """True when an admin password (or hash) has been configured."""
    return ADMIN_PASSWORD_HASH is not None


def authenticate_admin(username: str, password: str) -> bool:
    """Constant-time credential check for the admin account."""
    if not admin_login_enabled():
        return False
    user_ok = hmac.compare_digest(str(username or ""), ADMIN_USERNAME)
    pass_ok = verify_password(str(password or ""), ADMIN_PASSWORD_HASH)
    return user_ok and pass_ok


def create_access_token(username: str) -> str:
    payload = {"sub": username, "exp": time.time() + JWT_EXPIRE_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def verify_admin_token(
    creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),  # noqa: B008 — FastAPI dependency idiom
):
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing admin token")
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token missing subject")
    return sub


def reload_from_environment() -> None:
    """Re-read every credential from the environment (tests / config reload)."""
    global JWT_SECRET, JWT_SECRET_CONFIGURED, ADMIN_USERNAME
    global ADMIN_PASSWORD_HASH, ADMIN_PASSWORD_CONFIGURED
    JWT_SECRET, JWT_SECRET_CONFIGURED = resolve_secret(
        "BIOSUITE_JWT_SECRET", DEV_JWT_SECRET, "JWT signing secret")
    ADMIN_USERNAME = os.environ.get("BIOSUITE_ADMIN_USER", "admin")
    ADMIN_PASSWORD_HASH, ADMIN_PASSWORD_CONFIGURED = resolve_admin_password_hash()
