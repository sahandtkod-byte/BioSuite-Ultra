"""API key authentication for BioSuite Ultra API."""
import os
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY = os.environ.get("BIOSUITE_API_KEY", "changeme-dev-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key