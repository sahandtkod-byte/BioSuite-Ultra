"""WS2: X-Request-ID propagation on every response."""
import re
import uuid

import pytest
from fastapi.testclient import TestClient

from biosuite.api import app
from biosuite.api.middleware import (
    REQUEST_ID_HEADER,
    REQUEST_ID_PATTERN,
    is_valid_request_id,
    new_request_id,
    resolve_request_id,
)

client = TestClient(app, raise_server_exceptions=False)


class TestUnit:
    def test_generated_id_is_uuid4(self):
        value = new_request_id()
        assert uuid.UUID(value).version == 4

    def test_valid_ids(self):
        assert is_valid_request_id("abc-123")
        assert is_valid_request_id(str(uuid.uuid4()))
        assert is_valid_request_id("a" * 128)

    def test_invalid_ids(self):
        assert not is_valid_request_id("")
        assert not is_valid_request_id("-leading-dash")
        assert not is_valid_request_id("a" * 129)
        assert not is_valid_request_id("has space")
        assert not is_valid_request_id("line\ninjection")
        assert not is_valid_request_id("héllo")
        assert not is_valid_request_id(42)
        assert not is_valid_request_id(None)

    def test_resolve_echoes_valid(self):
        assert resolve_request_id("abc-123") == "abc-123"

    def test_resolve_replaces_invalid(self):
        fresh = resolve_request_id("bad id!\n")
        assert fresh != "bad id!\n"
        assert is_valid_request_id(fresh)

    def test_resolve_generates_for_missing(self):
        assert is_valid_request_id(resolve_request_id(None))


class TestPropagation:
    def test_every_response_has_header(self):
        for path in ("/ready", "/api/v1/ready", "/api/v1/healthz"):
            response = client.get(path)
            assert response.headers.get(REQUEST_ID_HEADER), path

    def test_incoming_valid_id_is_echoed(self):
        response = client.get("/ready", headers={REQUEST_ID_HEADER: "my-trace-42"})
        assert response.headers[REQUEST_ID_HEADER] == "my-trace-42"

    def test_incoming_invalid_id_is_replaced(self):
        response = client.get("/ready", headers={REQUEST_ID_HEADER: "bad id\n"})
        value = response.headers[REQUEST_ID_HEADER]
        assert value != "bad id\n"
        assert REQUEST_ID_PATTERN.match(value)

    def test_absent_id_gets_generated_uuid(self):
        response = client.get("/ready")
        assert uuid.UUID(response.headers[REQUEST_ID_HEADER]).version == 4

    def test_header_on_error_responses(self):
        response = client.get("/api/v1/definitely-not-a-route")
        assert response.status_code == 404
        assert response.headers.get(REQUEST_ID_HEADER)

    def test_ids_differ_across_requests(self):
        first = client.get("/ready").headers[REQUEST_ID_HEADER]
        second = client.get("/ready").headers[REQUEST_ID_HEADER]
        assert first != second

    def test_envelope_embeds_request_id(self):
        response = client.get(
            "/api/v1/definitely-not-a-route",
            headers={"X-BioSuite-Error-Envelope": "1",
                     REQUEST_ID_HEADER: "corr-7"})
        assert response.json()["error"]["request_id"] == "corr-7"
