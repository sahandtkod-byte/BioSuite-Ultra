"""API key authentication for BioSuite Ultra API."""
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

import logging

API_KEY = os.environ.get("BIOSUITE_API_KEY", "changeme-dev-key")

if API_KEY == "changeme-dev-key":
    logging.getLogger(__name__).warning(
        "BioSuite API key is the DEVELOPMENT default — set BIOSUITE_API_KEY "
        "before serving traffic.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key