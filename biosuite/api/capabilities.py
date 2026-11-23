"""REST exposure of the capability catalog (WS2).

Serves the deterministic WS1 manifest and availability discovery:

* ``GET /api/v1/capabilities`` — the full manifest.
* ``GET /api/v1/capabilities/{capability_id}`` — one entry with live availability.
* ``GET /api/v1/ready`` (+ unversioned ``/ready`` alias) — readiness probe.
* ``GET /.well-known/biosuite.json`` — the same manifest at a discovery path.

Discovery is key-gated by default: like every other route it needs the
``X-API-Key`` header.  Setting ``BIOSUITE_PUBLIC_MANIFEST=1`` makes the
manifest and per-capability views readable anonymously — discovery only, it
never unlocks execution or administration, because only these read-only
handlers consult the flag.
"""
from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Security

from biosuite.capabilities import (
    build_manifest,
    discover_availability,
    get_capability,
    list_capabilities,
)

from .auth import api_key_header, key_matches, public_endpoint
from .errors import ENVELOPE_HEADER
from .middleware import REQUEST_ID_HEADER

__all__ = ["MANIFEST_PUBLIC_ENV", "MANIFEST_PUBLIC_VALUES", "router",
           "manifest_is_public", "ready_payload"]

MANIFEST_PUBLIC_ENV = "BIOSUITE_PUBLIC_MANIFEST"
MANIFEST_PUBLIC_VALUES = {"1"}

router = APIRouter()


def manifest_is_public() -> bool:
    """True when public manifest discovery is explicitly enabled."""
    return os.environ.get(MANIFEST_PUBLIC_ENV, "") in MANIFEST_PUBLIC_VALUES


def _require_discovery_access(key: Optional[str],
                              capability_id: Optional[str] = None) -> None:
    """Key-gate discovery unless ``BIOSUITE_PUBLIC_MANIFEST=1``.

    Mirrors :func:`biosuite.api.auth.verify_api_key` so anonymous callers get
    the same 401 they would get anywhere else in the API.
    """
    if manifest_is_public() or key_matches(key):
        return
    target = f" '{capability_id}'" if capability_id else ""
    raise HTTPException(
        status_code=401,
        detail=f"Invalid or missing API key (capability discovery{target})")


@public_endpoint
@router.get("/api/v1/capabilities")
async def capabilities_manifest(key: Optional[str] = Security(api_key_header)):
    """Full capability manifest (key-gated unless BIOSUITE_PUBLIC_MANIFEST=1)."""
    _require_discovery_access(key)
    return build_manifest()


@public_endpoint
@router.get("/api/v1/capabilities/{capability_id}")
async def capability_detail(capability_id: str,
                            key: Optional[str] = Security(api_key_header)):
    """One capability entry with its live availability (same gating)."""
    _require_discovery_access(key, capability_id)
    try:
        capability = get_capability(capability_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Capability '{capability_id}' not found") from None
    availability = discover_availability()[capability_id]
    return {
        "capability": asdict(capability),
        "availability": asdict(availability),
    }


@public_endpoint
@router.get("/api/v1/ready")
async def ready():
    """Readiness probe: 200 while the service can serve, engine issues reported.

    Missing optional engines degrade individual capabilities but never make
    the service unready.
    """
    return ready_payload()


@public_endpoint
@router.get("/ready", include_in_schema=False)
async def ready_alias():
    """Unversioned alias of ``/api/v1/ready`` (not part of the 41 pinned routes)."""
    return ready_payload()


@public_endpoint
@router.get("/.well-known/biosuite.json")
async def well_known_manifest(key: Optional[str] = Security(api_key_header)):
    """Discovery-path alias of the manifest (same gating, unversioned)."""
    _require_discovery_access(key)
    return build_manifest()


def count_versioned_routes(application: Any) -> int:
    """Count registered ``/api/v1`` routes for *application*.

    Newer FastAPI folds ``include_router`` routes into a single
    ``_IncludedRouter`` marker without a ``.path`` attribute, so the walk
    descends into ``original_router.routes`` to keep the count complete.
    """
    paths = set()
    for route in application.routes:
        path = getattr(route, "path", None)
        if path:
            paths.add(path)
        original = getattr(route, "original_router", None)
        if original is not None:
            paths.update(sub.path for sub in original.routes
                         if getattr(sub, "path", None))
    return sum(1 for path in paths if path.startswith("/api/v1"))


def ready_payload() -> Dict[str, Any]:
    """Aggregate readiness over the catalogued capabilities."""
    availability = discover_availability()
    total = len(list_capabilities())
    ready_count = sum(1 for entry in availability.values() if entry.ready)
    missing = sorted(
        {probe.name
         for entry in availability.values()
         for probe in entry.tools + entry.packages
         if not probe.available})
    return {
        "status": "ready",
        "capabilities": {
            "total": total,
            "ready": ready_count,
            "degraded": total - ready_count,
        },
        "missing_engines": missing,
    }


_REQUEST_ID_PARAM = {
    "name": "X-Request-ID",
    "in": "header",
    "required": False,
    "schema": {"type": "string", "maxLength": 128,
               "pattern": "^[A-Za-z0-9][A-Za-z0-9._\\-]*$"},
    "description": ("Caller-supplied correlation id; echoed when well-formed, "
                    "otherwise a fresh id is generated and returned in the "
                    "response header of the same name."),
}

_ENVELOPE_PARAM = {
    "name": ENVELOPE_HEADER,
    "in": "header",
    "required": False,
    "schema": {"type": "string", "enum": ["1"]},
    "description": ("Opt in to structured error envelopes "
                    "({error: {code, status, detail, request_id}}). "
                    "Without it, existing error bodies are preserved."),
}


def enrich_openapi(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Document the WS2 cross-cutting behaviour in the OpenAPI schema.

    Adds the request-id and error-envelope headers to every operation, marks
    ``/api/v1/modules`` deprecated with its successor link, and records the
    discovery gating under ``info``.  Mutates and returns *schema*.
    """
    methods = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
    for path, path_item in schema.get("paths", {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method not in methods or not isinstance(operation, dict):
                continue
            parameters = operation.setdefault("parameters", [])
            for parameter in (_REQUEST_ID_PARAM, _ENVELOPE_PARAM):
                if not any(p.get("name") == parameter["name"] and p.get("in") == "header"
                           for p in parameters):
                    parameters.append(deepcopy(parameter))
            responses = operation.setdefault("responses", {})
            for response in responses.values():
                response.setdefault("headers", {})[REQUEST_ID_HEADER] = {
                    "schema": deepcopy(_REQUEST_ID_PARAM["schema"]),
                    "description": "Validated incoming request ID or a generated UUID4.",
                }
            if path == "/api/v1/modules":
                operation["deprecated"] = True
                operation["summary"] = ("Deprecated module discovery alias; "
                                        "use /api/v1/capabilities")
                responses.setdefault("200", {"description": "Legacy module discovery"}).setdefault(
                    "headers", {}).update({
                        "Deprecation": {"schema": {"type": "string"}, "description": "true"},
                        "Link": {"schema": {"type": "string"},
                                 "description": '</api/v1/capabilities>; rel="successor-version"'},
                    })
    schema.setdefault("info", {})["x-biosuite-capability-discovery"] = {
        "manifest": "/api/v1/capabilities",
        "well_known": "/.well-known/biosuite.json",
        "key_gated_by_default": True,
        "public_flag": "BIOSUITE_PUBLIC_MANIFEST=1",
    }
    return schema
