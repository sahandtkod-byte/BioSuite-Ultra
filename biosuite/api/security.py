"""JWT authentication for BioSuite Ultra admin routes.

NO hardcoded secrets. All secrets loaded from environment variables.
Password hashing via bcrypt or stdlib hashlib fallback.
"""
import os
import time
import secrets
import hashlib
import hmac
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# ── JWT Configuration (env-only, no defaults for secrets) ─────────────

def _get_jwt_secret() -> str:
    """Get JWT secret from environment. Auto-generates on first run."""
    secret = os.environ.get("BIOSUITE_JWT_SECRET", "")
    if not secret:
        secret_file = os.path.expanduser("~/.biosuite/.jwt_secret")
        os.makedirs(os.path.dirname(secret_file), exist_ok=True)
        if os.path.exists(secret_file):
            with open(secret_file) as f:
                secret = f.read().strip()
        else:
            secret = secrets.token_urlsafe(64)
            with open(secret_file, "w") as f:
                f.write(secret)
            try:
                os.chmod(secret_file, 0o600)
            except OSError:
                pass
    return secret


JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = int(os.environ.get("BIOSUITE_JWT_EXPIRE_SECONDS", "3600"))


# ── Password Hashing ──────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-SHA256 (stdlib, no extra deps).
    
    Format: pbkdf2:sha256:iterations:salt_hex:hash_hex
    """
    salt = secrets.token_hex(16)
    # Truncate to 72 bytes (SHA-256 block size)
    pwd_bytes = password.encode('utf-8')[:72]
    h = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt.encode(), iterations=100_000)
    return f"pbkdf2:sha256:100000:{salt}:{h.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    if hashed_password.startswith("pbkdf2:sha256:"):
        parts = hashed_password.split(":")
        if len(parts) == 5:
            iterations, salt, stored_hash = int(parts[2]), parts[3], parts[4]
            pwd_bytes = plain_password.encode('utf-8')[:72]
            h = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt.encode(), iterations)
            return hmac.compare_digest(h.hex(), stored_hash)
    # Legacy bcrypt hashes (passlib)
    try:
        from passlib.context import CryptContext
        ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return ctx.verify(plain_password, hashed_password)
    except Exception:
        pass
    return False


# ── Admin User Management ─────────────────────────────────────────────

def get_admin_credentials() -> tuple[str, str]:
    """Get admin username/password from environment. Raises if not set."""
    username = os.environ.get("BIOSUITE_ADMIN_USER", "")
    password = os.environ.get("BIOSUITE_ADMIN_PASSWORD", "")
    if not username or not password:
        raise RuntimeError(
            "BIOSUITE_ADMIN_USER and BIOSUITE_ADMIN_PASSWORD must be set. "
            "Example:\n"
            "  export BIOSUITE_ADMIN_USER=admin\n"
            "  export BIOSUITE_ADMIN_PASSWORD=$(python -c \"import secrets; print(secrets.token_urlsafe(24))\")"
        )
    return username, password


# ── JWT Token Operations ──────────────────────────────────────────────

bearer_scheme = HTTPBearer(auto_error=False)


def create_access_token(username: str) -> str:
    """Create a JWT access token."""
    try:
        from jose import jwt
    except ImportError:
        raise RuntimeError("python-jose not installed. Install with: pip install python-jose[cryptography]")
    secret = _get_jwt_secret()
    payload = {"sub": username, "exp": time.time() + JWT_EXPIRE_SECONDS}
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


async def verify_admin_token(creds: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    """Verify admin JWT token."""
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing admin token")
    try:
        from jose import jwt
    except ImportError:
        raise RuntimeError("python-jose not installed")
    secret = _get_jwt_secret()
    try:
        payload = jwt.decode(creds.credentials, secret, algorithms=[JWT_ALGORITHM])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload["sub"]
