"""`Retry-After` and the backoff curve that backs it up.

The header was read with `float()` alone, so its other legal form — an HTTP-date,
which is what a proxy or a CDN answers — raised out of the retry loop and reached
the user as ``Invalid response: could not convert string to float: 'Wed, 21 Oct
2015 07:28:00 GMT'``: a rate-limit answer described as a malformed body.  A
numeric value was also taken at face size, so one header could park the caller for
100000 seconds, and the 429 arm slept after the final attempt, buying nothing.
"""
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from email.message import Message

import pytest

from biosuite.core import databases as db


class _Resp:
    def __init__(self, body=b''):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Clock:
    def __init__(self):
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 4))

    @staticmethod
    def time():
        import time as _t
        return _t.time()

    @staticmethod
    def monotonic():
        import time as _t
        return _t.monotonic()


def _http_error(code, reason='error', retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers['Retry-After'] = retry_after
    return urllib.error.HTTPError('https://svc.test/x', code, reason, headers, None)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    db.invalidate_cache()
    monkeypatch.setattr(db, 'get_api_key', lambda name: '')
    monkeypatch.setattr(db, 'prompt_api_key', lambda *a, **k: '')
    monkeypatch.setattr(db, '_rate_limiter',
                        type('R', (), {'wait': lambda self, s: None})())
    clock = _Clock()
    monkeypatch.setattr(db, 'time', clock)
    return clock


def _serve(monkeypatch, script):
    calls = []

    def _urlopen(req, timeout=None):
        calls.append(req)
        outcome = script(len(calls))
        if isinstance(outcome, BaseException):
            raise outcome
        return _Resp(outcome)

    monkeypatch.setattr(urllib.request, 'urlopen', _urlopen)
    return calls


# ── the header parser ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (None, None),
    ("", None),
    ("   ", None),
    ("0.25", 0.25),
    ("2", 2.0),
    ("  7 ", 7.0),
    (3.7, 3.7),
    ("-5", 0.0),                    # a negative delta means "do not wait"
    ("100000", db.MAX_RETRY_WAIT),   # capped, not obeyed
    ("1e400", db.MAX_RETRY_WAIT),
    ("not a date", None),            # unreadable: the backoff curve decides
    ("Wed, 21 Oct 2015 07:28:00 GMT", 0.0),   # long past -> wait nothing
])
def test_a_retry_after_header_becomes_a_bounded_wait_or_nothing_at_all(value, expected):
    assert db._retry_after_seconds(value) == expected


def test_a_future_http_date_becomes_the_remaining_seconds():
    when = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=30)
    wait = db._retry_after_seconds(format_datetime(when, usegmt=True))
    assert 27.0 <= wait <= 30.0


def test_a_zone_less_date_is_read_as_utc_rather_than_as_local_time():
    """The same instant, with and without the zone, must agree.

    ``parsedate_to_datetime`` returns a naive datetime for the zone-less form;
    comparing that against an aware "now" would raise, so the naive reading is
    pinned to UTC.
    """
    with_zone = db._retry_after_seconds('Wed, 21 Oct 2015 07:28:00 GMT')
    without_zone = db._retry_after_seconds('Wed, 21 Oct 2015 07:28:00')

    assert (with_zone, without_zone) == (0.0, 0.0)


# ── what the retry loop does with it ─────────────────────────────────────────

def test_a_date_valued_429_is_retried_instead_of_aborting_the_request(isolated, monkeypatch):
    """The HTTP-date form is legal; it used to escape as a ValueError."""
    body = json.dumps({"esearchresult": {"idlist": []}}).encode()
    calls = _serve(monkeypatch,
                   lambda n: _http_error(429, 'Too Many Requests',
                                         'Wed, 21 Oct 2015 07:28:00 GMT')
                   if n == 1 else body)

    res = db.search_ncbi('brca')

    assert res.error == ''                      # was "Invalid response: could not convert…"
    assert res.data == {"count": 0}
    assert len(calls) == 2
    assert isolated.sleeps == [0.0]             # a date long past means: retry now


def test_an_absurd_retry_after_is_capped_at_the_module_maximum(isolated, monkeypatch):
    calls = _serve(monkeypatch,
                   lambda n: _http_error(429, 'Too Many Requests', '100000')
                   if n == 1 else b'{"ok": 1}')

    assert db._http_get('https://svc.test/x', service='uniprot') == b'{"ok": 1}'
    assert isolated.sleeps == [db.MAX_RETRY_WAIT]
    assert len(calls) == 2


def test_an_unreadable_retry_after_falls_back_to_the_backoff_curve(isolated, monkeypatch):
    calls = _serve(monkeypatch,
                   lambda n: _http_error(429, 'Too Many Requests', 'soon')
                   if n == 1 else b'{}')

    assert db._http_get('https://svc.test/y', service='ncbi') == b'{}'
    assert isolated.sleeps == [1.5]
    assert len(calls) == 2


def test_a_503_may_use_retry_after_too(isolated, monkeypatch):
    """RFC 9110 sends Retry-After with 503 as well as 429.

    Only the 429 arm read the header before, so a server saying "come back in two
    seconds" behind a 503 was answered with the 1.5 s backoff curve instead.
    """
    calls = _serve(monkeypatch,
                   lambda n: _http_error(503, 'Service Unavailable', '2')
                   if n == 1 else b'{"ok": true}')

    assert db._http_get('https://svc.test/503', service='pdb') == b'{"ok": true}'
    assert isolated.sleeps == [2.0]
    assert len(calls) == 2


def test_the_backoff_curve_still_governs_every_status_that_says_nothing(isolated, monkeypatch):
    calls = _serve(monkeypatch, lambda n: _http_error(503, 'Service Unavailable'))

    with pytest.raises(urllib.error.HTTPError):
        db._http_get('https://svc.test/z', service='pdb')

    assert len(calls) == db.DEFAULT_MAX_RETRIES
    assert isolated.sleeps == [1.5, 2.25]


def test_no_wait_is_taken_when_no_attempt_is_left(isolated, monkeypatch):
    """A sleep the code cannot act on is a sleep the user still pays for."""
    calls = _serve(monkeypatch, lambda n: _http_error(429, 'Too Many Requests'))

    with pytest.raises(urllib.error.HTTPError):
        db._http_get('https://svc.test/w', service='ncbi')

    assert len(calls) == db.DEFAULT_MAX_RETRIES
    assert isolated.sleeps == [1.5, 2.25]      # two waits for three requests


def test_without_a_retry_budget_no_request_is_made_and_that_is_named(monkeypatch):
    monkeypatch.setattr(db, 'DEFAULT_MAX_RETRIES', 0)
    asked = []
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=None: asked.append(req) or _Resp(b'x'))

    with pytest.raises(RuntimeError) as excinfo:
        db._http_get('https://svc.test/v', service='ncbi')

    assert asked == []
    assert str(excinfo.value) == "[ncbi] no HTTP attempt was made for https://svc.test/v"
