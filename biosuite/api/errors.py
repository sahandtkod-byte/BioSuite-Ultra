"""Opt-in structured error envelopes for the BioSuite API.

Legacy error bodies are preserved by default, including plain-text server
errors and SlowAPI rate-limit errors. With ``X-BioSuite-Error-Envelope: 1``,
a client receives a structured envelope instead::

    {"error": {"code": ..., "status": ..., "detail": ..., "request_id": ...}}

The envelope adds fields; it never changes status codes, and the legacy body
shape is byte-identical to what the handlers produced before WS2.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from slowapi import _rate_limit_exceeded_handler

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from .middleware import REQUEST_ID_HEADER, is_valid_request_id

__all__ = [
    "ENVELOPE_HEADER",
    "ERROR_CODE_BY_HANDLER",
    "envelope_requested",
    "error_response",
    "http_exception_response",
    "request_id_of",
    "validation_error_response",
]

ENVELOPE_HEADER = "X-BioSuite-Error-Envelope"

#: Structured codes for the exception handlers that previously built their own
#: ``JSONResponse`` inline; keyed by handler origin so each stays stable.
ERROR_CODE_BY_HANDLER = {
    "value_error": "invalid_input",
    "file_not_found": "file_not_found",
    "missing_field": "missing_field",
    "inconsistent_lengths": "inconsistent_lengths",
}

#: Fallback codes by status for ``HTTPException`` raised without a specific code.
_ERROR_CODE_BY_STATUS = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    429: "rate_limited",
}


def request_id_of(request: Request) -> Optional[str]:
    """The request id assigned by the middleware, when it ran.

    Falls back to a well-formed incoming header, and to ``None`` for
    synthetic request doubles that carry neither (legacy handler tests).
    """
    state = getattr(request, "state", None)
    request_id = getattr(state, "request_id", None)
    if request_id is not None:
        return request_id
    headers = getattr(request, "headers", None)
    if headers is not None:
        incoming = headers.get(REQUEST_ID_HEADER)
        if is_valid_request_id(incoming):
            return incoming
    return None


def envelope_requested(request: Request) -> bool:
    """True when the client explicitly asked for the structured envelope."""
    headers = getattr(request, "headers", None)
    if headers is None:
        return False
    return headers.get(ENVELOPE_HEADER, "").strip() == "1"


def _code_for_status(status_code: int) -> str:
    return _ERROR_CODE_BY_STATUS.get(status_code, f"http_{status_code}")


def _envelope(request: Request, *, status_code: int, code: str,
              detail: Any) -> Dict[str, Any]:
    return {
        "error": {
            "code": code,
            "status": status_code,
            "detail": detail,
            "request_id": request_id_of(request),
        }
    }


def _json(status_code: int, content: Dict[str, Any],
          headers: Optional[Dict[str, str]] = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=content,
                        headers=headers, background=None)


def error_response(request: Request, *, status_code: int, detail: Any,
                   code: str, headers: Optional[Dict[str, str]] = None) -> JSONResponse:
    """Legacy ``{"detail": ...}`` body, or the envelope when opted in."""
    if envelope_requested(request):
        return _json(status_code,
                     _envelope(request, status_code=status_code, code=code,
                               detail=detail),
                     headers)
    return _json(status_code, {"detail": detail}, headers)


def http_exception_response(request: Request, exc) -> JSONResponse:
    """Reproduces Starlette's default ``HTTPException`` body, envelope-aware."""
    headers = getattr(exc, "headers", None)
    code = _code_for_status(exc.status_code)
    if envelope_requested(request):
        return _json(exc.status_code,
                     _envelope(request, status_code=exc.status_code, code=code,
                               detail=exc.detail),
                     headers)
    return _json(exc.status_code, {"detail": exc.detail}, headers)


def validation_error_response(request: Request,
                              exc: RequestValidationError) -> JSONResponse:
    """Reproduces FastAPI's default 422 body, envelope-aware."""
    detail = jsonable_encoder(exc.errors())
    if envelope_requested(request):
        return _json(422,
                     _envelope(request, status_code=422, code="validation_error",
                               detail=detail),
                     None)
    return _json(422, {"detail": detail}, None)


def server_error_response(request: Request):
    headers = {REQUEST_ID_HEADER: request_id_of(request)}
    if envelope_requested(request):
        return error_response(request, status_code=500, detail="Internal Server Error",
                              code="internal_server_error", headers=headers)
    return PlainTextResponse("Internal Server Error", status_code=500, headers=headers)


def rate_limit_error_response(request: Request, exc):
    response = _rate_limit_exceeded_handler(request, exc)
    if envelope_requested(request):
        headers = dict(response.headers)
        headers.pop("content-length", None)
        return error_response(request, status_code=429,
                              detail=json.loads(response.body)["error"],
                              code="rate_limited", headers=headers)
    return response
