"""WS2: opt-in structured error envelope (X-BioSuite-Error-Envelope: 1).

The default preserves existing JSON and plain-text error bodies.
Only the explicit header switches to the structured envelope, and no status
code changes in either mode.
"""
import os

import pytest
from fastapi.testclient import TestClient

from biosuite.api import app

API_KEY = os.environ.get("BIOSUITE_API_KEY", "changeme-dev-key")
client = TestClient(app, raise_server_exceptions=False)
ENVELOPE = {"X-BioSuite-Error-Envelope": "1", "X-API-Key": API_KEY}


def _legacy(path, **kwargs):
    return client.get(path, headers=kwargs.pop("headers", {}), **kwargs)


class TestLegacyDefault:
    def test_unknown_route_body_unchanged(self):
        response = client.get("/api/v1/definitely-not-a-route")
        assert response.status_code == 404
        assert response.json() == {"detail": "Not Found"}

    def test_unknown_capability_body_unchanged(self):
        response = client.get("/api/v1/capabilities/nope",
                              headers={"X-API-Key": API_KEY})
        assert response.status_code == 404
        assert set(response.json()) == {"detail"}

    def test_validation_error_body_unchanged(self):
        response = client.post("/api/v1/sequence/gc-content", json={},
                               headers={"X-API-Key": API_KEY})
        assert response.status_code == 422
        assert set(response.json()) == {"detail"}

    def test_no_envelope_without_header(self):
        response = client.get("/api/v1/definitely-not-a-route")
        assert "error" not in response.json()


class TestEnvelopeOptIn:
    def test_unknown_route_envelope(self):
        response = client.get("/api/v1/definitely-not-a-route", headers=ENVELOPE)
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["status"] == 404
        assert error["code"] == "not_found"
        assert error["detail"] == "Not Found"
        assert error["request_id"]

    def test_unknown_capability_envelope(self):
        response = client.get("/api/v1/capabilities/nope", headers=ENVELOPE)
        error = response.json()["error"]
        assert error["status"] == 404
        assert error["code"] == "not_found"

    def test_validation_error_envelope(self):
        response = client.post("/api/v1/sequence/gc-content", json={},
                               headers=ENVELOPE)
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "validation_error"
        assert isinstance(error["detail"], list)

    def test_other_envelope_values_are_ignored(self):
        for value in ("0", "", "true", "yes", "on"):
            response = client.get(
                "/api/v1/definitely-not-a-route",
                headers={"X-BioSuite-Error-Envelope": value})
            assert response.json() == {"detail": "Not Found"}, value

    def test_status_codes_never_change(self):
        legacy = client.get("/api/v1/capabilities/nope",
                            headers={"X-API-Key": API_KEY})
        enveloped = client.get("/api/v1/capabilities/nope", headers=ENVELOPE)
        assert legacy.status_code == enveloped.status_code


class TestEnvelopeUnit:
    def test_error_codes_are_registered(self):
        from biosuite.api.errors import ERROR_CODE_BY_HANDLER
        assert ERROR_CODE_BY_HANDLER["value_error"] == "invalid_input"
        assert ERROR_CODE_BY_HANDLER["file_not_found"] == "file_not_found"


@pytest.mark.parametrize("incoming, expected", [("trace-123", "trace-123"),
                                               ("not valid", None), (None, None)])
def test_request_id_fallback_without_middleware(incoming, expected):
    from types import SimpleNamespace
    from biosuite.api.errors import request_id_of

    headers = {"X-Request-ID": incoming} if incoming is not None else {}
    assert request_id_of(SimpleNamespace(headers=headers)) == expected
    assert request_id_of(SimpleNamespace()) is None


def test_request_id_prefers_middleware_state():
    from types import SimpleNamespace
    from biosuite.api.errors import request_id_of

    request = SimpleNamespace(state=SimpleNamespace(request_id="assigned"),
                              headers={"X-Request-ID": "incoming"})
    assert request_id_of(request) == "assigned"


def test_error_response_preserves_legacy_without_request():
    import json
    from biosuite.api.errors import error_response

    response = error_response(None, status_code=409, detail="conflict", code="conflict",
                              headers={"Retry-After": "5"})
    assert response.status_code == 409
    assert json.loads(response.body) == {"detail": "conflict"}
    assert response.headers["Retry-After"] == "5"


@pytest.mark.parametrize("status, expected", [(400, "bad_request"), (401, "unauthorized"),
    (403, "forbidden"), (404, "not_found"), (429, "rate_limited"), (503, "http_503")])
def test_default_error_codes(status, expected):
    import json
    from types import SimpleNamespace
    from fastapi import HTTPException
    from biosuite.api.errors import http_exception_response

    response = http_exception_response(SimpleNamespace(headers=ENVELOPE),
                                       HTTPException(status, "detail"))
    assert json.loads(response.body)["error"]["code"] == expected


@pytest.mark.parametrize("exception, code, status", [
    (ValueError("invalid"), "invalid_input", 422),
    (FileNotFoundError("missing"), "file_not_found", 404),
    (KeyError("sample"), "missing_field", 422),
])
def test_scientific_errors_preserve_detail_and_correlate(monkeypatch, exception, code, status):
    import biosuite.core.sequence as sequence

    def fail(*args, **kwargs):
        raise exception

    monkeypatch.setattr(sequence, "gc_content", fail)
    path = "/api/v1/sequence/gc-content"
    legacy = client.post(path, headers={"X-API-Key": API_KEY}, json={"sequence": "ATGC"})
    enriched = client.post(path, headers={**ENVELOPE, "X-Request-ID": "error-trace"},
                           json={"sequence": "ATGC"})
    assert legacy.status_code == enriched.status_code == status
    assert enriched.json()["error"] == {
        "code": code, "status": status, "detail": legacy.json()["detail"],
        "request_id": "error-trace",
    }
    assert enriched.headers["X-Request-ID"] == "error-trace"


@pytest.mark.parametrize("envelope", [False, True])
def test_unhandled_server_errors_keep_legacy_body_or_opt_in(monkeypatch, envelope):
    import biosuite.core.sequence as sequence

    def fail(*args, **kwargs):
        raise RuntimeError("private diagnostic must not be exposed")

    monkeypatch.setattr(sequence, "gc_content", fail)
    headers = {"X-API-Key": API_KEY, "X-Request-ID": "server-error-trace"}
    if envelope:
        headers["X-BioSuite-Error-Envelope"] = "1"
    response = client.post("/api/v1/sequence/gc-content", headers=headers,
                           json={"sequence": "ATGC"})
    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "server-error-trace"
    if envelope:
        assert response.json()["error"] == {
            "code": "internal_server_error", "status": 500,
            "detail": "Internal Server Error", "request_id": "server-error-trace",
        }
    else:
        assert response.content == b"Internal Server Error"
        assert response.headers["content-type"] == "text/plain; charset=utf-8"


@pytest.mark.parametrize("envelope", [False, True])
def test_rate_limited_responses_have_request_ids_and_optional_envelopes(envelope):
    app.state.limiter.reset()
    headers = {"X-API-Key": API_KEY, "X-Request-ID": "limited-trace"}
    if envelope:
        headers["X-BioSuite-Error-Envelope"] = "1"
    try:
        responses = [client.get("/api/v1/modules", headers=headers) for _ in range(110)]
        limited = [response for response in responses if response.status_code == 429]
        assert limited
        for response in limited:
            assert response.headers["X-Request-ID"] == "limited-trace"
            if envelope:
                assert response.json()["error"]["code"] == "rate_limited"
                assert response.json()["error"]["request_id"] == "limited-trace"
            else:
                assert set(response.json()) == {"error"}
    finally:
        app.state.limiter.reset()
