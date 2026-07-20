"""JWT authentication for BioSuite Ultra admin routes."""
import os
import time
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

JWT_SECRET = os.environ.get("BIOSUITE_JWT_SECRET", "changeme-dev-secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 3600

ADMIN_USERNAME = os.environ.get("BIOSUITE_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("BIOSUITE_ADMIN_PASSWORD", "changeme-dev-password")

bearer_scheme = HTTPBearer(auto_error=False)

def create_access_token(username: str) -> str:
    payload = {"sub": username, "exp": time.time() + JWT_EXPIRE_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def verify_admin_token(creds: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing admin token")
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload["sub"]