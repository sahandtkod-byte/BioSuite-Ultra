"""Request-ID middleware for the BioSuite API.

Every response carries an ``X-Request-ID`` header.  A caller may supply its own
correlation id through the same request header; well-formed values are echoed
back so distributed traces can be correlated end to end, and anything
malformed (empty, over-long, or containing characters that would poison logs
or downstream systems) is silently replaced by a fresh generated id rather
than rejected.
"""
from __future__ import annotations

import re
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

from biosuite.core.provenance import bind_correlation, reset_correlation

__all__ = [
    "REQUEST_ID_HEADER",
    "REQUEST_ID_PATTERN",
    "RequestIDMiddleware",
    "is_valid_request_id",
    "new_request_id",
    "resolve_request_id",
]

REQUEST_ID_HEADER = "X-Request-ID"

# One token of printable, log-safe ASCII: it starts alphanumerically and may
# then contain alphanumerics, dots, underscores and hyphens.  This accepts the
# common formats in the wild (UUIDs, ULIDs, opaque base62 tokens, traces like
# ``span-1.2.3``) while rejecting control characters, whitespace and header
# injection.  128 characters bounds log lines; anything longer is treated as
# malformed rather than truncated, so a caller never receives an id it did
# not send.
REQUEST_ID_PATTERN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._\-]{0,127}\Z")


def is_valid_request_id(value: object) -> bool:
    """True when *value* is a well-formed request id."""
    return isinstance(value, str) and REQUEST_ID_PATTERN.match(value) is not None


def new_request_id() -> str:
    """Generate a fresh request id (UUID4, the de-facto correlation format)."""
    return str(uuid.uuid4())


def resolve_request_id(incoming: object) -> str:
    """Echo a valid incoming id, otherwise generate one."""
    if is_valid_request_id(incoming):
        return incoming
    return new_request_id()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign an id to every request and stamp it on every response.

    The resolved id is also stored on ``request.state.request_id`` so error
    envelopes (WS2) and provenance records (WS5) can reference it.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        request_id = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        # WS5: bind the correlation for this execution context (one context per
        # request, isolated from concurrent requests) so provenance steps
        # recorded while the request is handled carry its ids.
        token = bind_correlation(request_id)
        try:
            response = await call_next(request)
        finally:
            reset_correlation(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
