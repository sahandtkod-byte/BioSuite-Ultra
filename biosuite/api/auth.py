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
from .scopes import load_scoped_keys, scope_satisfies

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


ROUTE_SCOPES = {
    ("GET", "/"): "read",
    ("GET", "/health"): "read",
    ("GET", "/openapi.json"): "read",
    ("GET", "/docs"): "read",
    ("GET", "/api/v1/modules"): "read",
    ("GET", "/api/v1/capabilities"): "read",
    ("GET", "/api/v1/capabilities/{capability_id}"): "read",
    ("GET", "/.well-known/biosuite.json"): "read",
    ("GET", "/api/v1/ready"): "read",
    ("GET", "/ready"): "read",
    ("GET", "/api/v1/gwas/demo"): "read",
    ("GET", "/api/v1/database/ncbi"): "read",
    ("GET", "/api/v1/database/uniprot"): "read",
    ("GET", "/api/v1/database/pdb"): "read",
    ("GET", "/api/v1/database/kegg"): "read",
    ("GET", "/api/v1/provenance/summary"): "read",
    ("GET", "/api/v1/admin/status"): "admin",
    ("POST", "/api/v1/admin/login"): "admin",
    ("POST", "/api/v1/sequence/gc-content"): "execute",
    ("POST", "/api/v1/sequence/reverse-complement"): "execute",
    ("POST", "/api/v1/sequence/translate"): "execute",
    ("POST", "/api/v1/sequence/stats"): "execute",
    ("POST", "/api/v1/alignment/needleman-wunsch"): "execute",
    ("POST", "/api/v1/alignment/smith-waterman"): "execute",
    ("POST", "/api/v1/blast/search"): "execute",
    ("POST", "/api/v1/expression/differential"): "execute",
    ("POST", "/api/v1/expression/normalize/cpm"): "execute",
    ("POST", "/api/v1/expression/normalize/tpm"): "execute",
    ("POST", "/api/v1/expression/normalize/deseq2"): "execute",
    ("POST", "/api/v1/crispr/design"): "execute",
    ("POST", "/api/v1/metagenomics/classify-16s"): "execute",
    ("POST", "/api/v1/metagenomics/diversity"): "execute",
    ("POST", "/api/v1/epitope/predict"): "execute",
    ("POST", "/api/v1/gwas/run"): "execute",
    ("POST", "/api/v1/phylogeny/distance-matrix"): "execute",
    ("POST", "/api/v1/phylogeny/upgma"): "execute",
    ("POST", "/api/v1/popgen/hwe"): "execute",
    ("POST", "/api/v1/popgen/fst"): "execute",
    ("POST", "/api/v1/popgen/tajimas-d"): "execute",
    ("POST", "/api/v1/plotting/volcano"): "execute",
    ("POST", "/api/v1/plotting/pca"): "execute",
    ("POST", "/api/v1/plotting/manhattan"): "execute",
    ("POST", "/api/v1/plotting/heatmap"): "execute",
    ("POST", "/api/v1/workflow/pipeline"): "execute",
    ("POST", "/api/v1/file/detect-format"): "execute",
    ("POST", "/api/v1/file/read"): "execute",
    ("POST", "/api/v1/provenance/record"): "execute",
}


def authorize_key(key, request):
    """Authenticate a key and enforce its explicit method/route scope.

    The legacy master key has full API-key access, including unclassified
    routes, but never replaces endpoint-specific authentication such as JWT.
    Scoped keys fail closed without a classified resolved route.
    """
    if key_matches(key):
        return key
    scope = getattr(request, "scope", None) or {}
    route = scope.get("route")
    required = ROUTE_SCOPES.get((scope.get("method"), getattr(route, "path", None)))
    for configured, grants in SCOPED_KEYS.items():
        if hmac.compare_digest(str(key or ""), configured):
            if required not in grants:
                raise HTTPException(status_code=403, detail="Insufficient API key scope")
            return key
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def verify_api_key(key: str = Security(api_key_header),
                         request: Request = None):
    """Enforce scoped access; public routes and JWT-only admin access remain.

    Admin status still validates its JWT in the endpoint dependency. An API
    key supplied alongside it must independently authorize admin access.
    """
    scope = getattr(request, "scope", None) or {}
    route = scope.get("route")
    if getattr(route, "endpoint", None) in PUBLIC_ENDPOINTS:
        return key
    if not key and (scope.get("method"), getattr(route, "path", None)) == (
        "GET", "/api/v1/admin/status"
    ):
        return key
    return authorize_key(key, request)


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
