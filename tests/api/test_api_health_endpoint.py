"""FIX-23 — /health must work for a monitor that holds no credential.

Covered: anonymous reachability, what an anonymous caller is allowed to learn,
the keyed view that ``test_api.py`` already pins, the rate-limit budget the probe
used to share with ordinary traffic, and the README claim that describes it (P6-7).
"""
import os
import pathlib

import pytest
from fastapi.testclient import TestClient

from biosuite.api import app

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

API_KEY = os.environ["BIOSUITE_API_KEY"]
ANON = TestClient(app)


@pytest.fixture(autouse=True)
def _isolated_rate_limits():
    """The limiter keys on client address, which TestClient shares globally."""
    limiter = app.state.limiter
    limiter.reset()
    yield
    limiter.reset()


def test_health_is_reachable_without_a_credential():
    assert ANON.get("/health").status_code == 200


def test_anonymous_health_reports_only_liveness():
    """Version and module inventory stay behind the key: a public probe must not
    become a free software fingerprint."""
    assert set(ANON.get("/health").json()) == {"status", "timestamp"}


def test_keyed_health_still_reports_version_and_modules():
    from biosuite import __version__

    body = TestClient(app, headers={"X-API-Key": API_KEY}).get("/health").json()
    assert body["version"] == __version__
    assert isinstance(body["modules"], int) and body["modules"] > 0


def test_health_does_not_consume_the_shared_rate_limit_budget():
    """120 pings in a row: a busy API must not make a healthy server look dead."""
    codes = {ANON.get("/health").status_code for _ in range(120)}
    assert codes == {200}


def test_ordinary_routes_are_still_limited_and_still_keyed():
    codes = [ANON.get("/api/v1/modules").status_code for _ in range(110)]
    assert codes[0] == 401, "a protected route answered without a key"
    assert 429 in codes, "the limiter no longer applies to ordinary traffic"


def test_readme_describes_the_endpoint_as_it_behaves():
    """P6-7: the security section used to claim every endpoint needs a key."""
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "unauthenticated `/health`" in readme
    assert "Every endpoint requires an `X-API-Key` header except the" in readme
    assert ANON.get("/health").status_code == 200
