"""HTTP retry, rate-limit and error-reporting contract of ``core.databases``.

Every endpoint here talks to a public REST service, so the mocked payload is the
only honest input: these tests pin the URL/params/command each function builds,
how it parses the canned body, and — the part that bites users — exactly what an
HTTP error, a timeout, a rate limit, an empty body or a malformed body turns
into.  Nothing in this file touches a socket.

The companion ``test_databases_mocked.py`` covers happy-path parsing; this file
covers ``_http_get`` itself plus every ``except`` arm of every endpoint.
"""
import contextlib
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from email.message import Message

import pytest

from biosuite.core import databases as db


# ── fakes ────────────────────────────────────────────────────────────────────

class _Resp:
    """Stand-in for the object ``urlopen`` returns as a context manager."""

    def __init__(self, body=b''):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, reason, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers['Retry-After'] = retry_after
    return urllib.error.HTTPError('https://svc.test/x', code, reason, headers, None)


class _Clock:
    """``time`` replacement that records sleeps instead of performing them."""

    def __init__(self):
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 4))

    @staticmethod
    def time():
        return time.time()

    @staticmethod
    def monotonic():
        return time.monotonic()


class _Limiter:
    """``_RateLimiter`` replacement that records which service asked to wait."""

    def __init__(self):
        self.services = []

    def wait(self, service):
        self.services.append(service)


@pytest.fixture
def clock(monkeypatch):
    monkeypatch.setattr(db, 'time', _Clock())
    return db.time


@pytest.fixture
def limiter(monkeypatch):
    monkeypatch.setattr(db, '_rate_limiter', _Limiter())
    return db._rate_limiter


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    """Empty cache, no waiting, no credential prompting, no real HTTP."""
    db.invalidate_cache()
    monkeypatch.setattr(db, 'get_api_key', lambda name: '')
    monkeypatch.setattr(db, 'prompt_api_key', lambda *a, **k: '')
    monkeypatch.setattr(db, '_rate_limiter', _Limiter())
    monkeypatch.setattr(db, 'time', _Clock())
    yield
    db.invalidate_cache()


def fake_http(monkeypatch, payload):
    """Replace ``_http_get``; return the (url, kwargs) list it collected.

    ``payload`` is bytes or a callable taking the url and returning bytes.
    """
    asked = []

    def _get(url, **kwargs):
        asked.append((url, kwargs))
        return payload(url, kwargs) if callable(payload) else payload

    monkeypatch.setattr(db, '_http_get', _get)
    return asked


def fake_urlopen(monkeypatch, script):
    """Replace ``urllib.request.urlopen``; ``script(n)`` returns bytes or raises."""
    calls = []

    def _urlopen(req, timeout=None):
        calls.append(req)
        outcome = script(len(calls))
        if isinstance(outcome, BaseException):
            raise outcome
        return _Resp(outcome)

    monkeypatch.setattr(urllib.request, 'urlopen', _urlopen)
    return calls


# ── _http_get: request building ──────────────────────────────────────────────

def test_http_get_sends_declared_headers_and_the_named_service(monkeypatch, limiter, clock):
    calls = fake_urlopen(monkeypatch, lambda n: b'payload')

    body = db._http_get('https://svc.test/a', timeout=7,
                        headers={'X-Api-Key': 'secret'}, service='kegg')

    assert body == b'payload'
    assert len(calls) == 1
    assert calls[0].full_url == 'https://svc.test/a'
    # urllib capitalises the header name it stores
    assert calls[0].get_header('X-api-key') == 'secret'
    assert limiter.services == ['kegg']
    assert clock.sleeps == []


def test_http_get_adds_no_headers_when_none_are_declared(monkeypatch, limiter):
    calls = fake_urlopen(monkeypatch, lambda n: b'raw')

    assert db._http_get('https://svc.test/b', service='general') == b'raw'
    assert calls[0].header_items() == []
    assert calls[0].get_method() == 'GET'
    assert limiter.services == ['general']   # an unnamed service still rate limits


# ── _http_get: retry / backoff / give-up ─────────────────────────────────────

def test_http_get_honours_retry_after_on_429(monkeypatch, clock):
    calls = fake_urlopen(monkeypatch,
                         lambda n: _http_error(429, 'Too Many Requests', '0.25')
                         if n == 1 else b'second try works')

    assert db._http_get('https://svc.test/c', service='ncbi') == b'second try works'
    assert clock.sleeps == [0.25]          # Retry-After, not the backoff curve
    assert len(calls) == 2


def test_http_get_backs_off_exponentially_when_429_has_no_retry_after(monkeypatch, clock):
    calls = fake_urlopen(monkeypatch,
                         lambda n: _http_error(429, 'Too Many Requests') if n <= 2 else b'ok')

    assert db._http_get('https://svc.test/d', service='uniprot') == b'ok'
    assert clock.sleeps == [1.5, 2.25]     # DEFAULT_RETRY_BACKOFF ** attempt
    assert len(calls) == 3


@pytest.mark.parametrize('status', sorted(db.RETRYABLE_STATUS - {429}))
def test_http_get_retries_every_retryable_status_then_returns(status, monkeypatch, clock):
    calls = fake_urlopen(monkeypatch,
                         lambda n: _http_error(status, 'flaky') if n == 1 else b'back')

    assert db._http_get('https://svc.test/e', service='pdb') == b'back'
    assert clock.sleeps == [1.5]
    assert len(calls) == 2


def test_http_get_raises_a_non_retryable_status_without_retrying(monkeypatch, clock):
    calls = fake_urlopen(monkeypatch, lambda n: _http_error(404, 'Not Found'))

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        db._http_get('https://svc.test/f', service='ncbi')

    assert excinfo.value.code == 404
    assert len(calls) == 1                 # a 404 is not worth three requests
    assert clock.sleeps == []


def test_http_get_raises_the_server_error_after_using_all_retries(monkeypatch, clock):
    calls = fake_urlopen(monkeypatch, lambda n: _http_error(503, 'Service Unavailable'))

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        db._http_get('https://svc.test/g', service='ncbi')

    assert excinfo.value.code == 503
    assert len(calls) == db.DEFAULT_MAX_RETRIES
    assert clock.sleeps == [1.5, 2.25]     # no sleep after the final attempt


def test_http_get_raises_the_stored_429_when_the_retry_budget_runs_out(monkeypatch, clock):
    """A 429 is raised, not slept through, once the last request has been made.

    The 429 arm used to differ from every other retryable status here: it slept
    and ``continue``d even on the final attempt, so three requests cost three
    waits and the third one bought nothing — and the old pin named that
    asymmetry in its own docstring.
    """
    calls = fake_urlopen(monkeypatch, lambda n: _http_error(429, 'Too Many Requests'))

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        db._http_get('https://svc.test/h', service='ncbi')

    assert excinfo.value.code == 429
    assert len(calls) == db.DEFAULT_MAX_RETRIES
    assert clock.sleeps == [1.5, 2.25]     # identical to the 5xx arm


@pytest.mark.parametrize('factory', [
    lambda n: urllib.error.URLError('name resolution failed'),
    lambda n: OSError('network down'),
    lambda n: TimeoutError('read timed out'),
])
def test_http_get_raises_the_last_connection_error_after_all_retries(factory, monkeypatch, clock):
    calls = fake_urlopen(monkeypatch, factory)

    with pytest.raises((urllib.error.URLError, OSError, TimeoutError)):
        db._http_get('https://svc.test/i', service='kegg')

    assert len(calls) == db.DEFAULT_MAX_RETRIES
    assert clock.sleeps == [1.5, 2.25]


def test_http_get_recovers_from_a_transient_connection_error(monkeypatch, clock):
    calls = fake_urlopen(monkeypatch,
                         lambda n: urllib.error.URLError('reset') if n == 1 else b'recovered')

    assert db._http_get('https://svc.test/j', service='ensembl') == b'recovered'
    assert clock.sleeps == [1.5]
    assert len(calls) == 2


def test_http_get_retry_log_redacts_the_api_key(monkeypatch, clock, caplog):
    """A retry must never write the caller's api_key into the log.

    The credential genuinely rides in the request URL — the positive control
    pins that, so the test cannot pass vacuously on a URL that never carried a
    key — but the warning that names the URL has to mask it.  A secret in a log
    file is a secret leaked to whoever can read that file.
    """
    import logging

    secret = 'SECRET_API_KEY_TOKEN_99'
    url = f'https://svc.test/x?db=protein&api_key={secret}&retmode=json'
    calls = fake_urlopen(monkeypatch,
                         lambda n: _http_error(429, 'Too Many Requests', '0.25')
                         if n == 1 else b'ok')

    with caplog.at_level(logging.WARNING, logger='biosuite.core.databases'):
        assert db._http_get(url, service='ncbi') == b'ok'

    assert calls[0].full_url == url                 # the key really went out
    assert f'api_key={secret}' in calls[0].full_url
    assert secret not in caplog.text, caplog.text   # but never into the log
    assert 'api_key=REDACTED' in caplog.text, caplog.text


@pytest.mark.parametrize('raw,expected', [
    ('https://x/y?api_key=ABC&db=p', 'https://x/y?api_key=REDACTED&db=p'),
    ('https://x/y?db=p&api_key=ABC', 'https://x/y?db=p&api_key=REDACTED'),
    ('https://x/y?db=p&api_key=ABC&retmode=json',
     'https://x/y?db=p&api_key=REDACTED&retmode=json'),
    ('https://x/y?db=p', 'https://x/y?db=p'),
])
def test_redact_url_masks_only_the_api_key_value(raw, expected):
    """The redaction keeps the URL debuggable but never carries the credential."""
    assert db._redact_url(raw) == expected


# ── per-service rate limiting ────────────────────────────────────────────────

class _SteppedClock:
    """``time`` replacement whose monotonic() walks a fixed list of readings."""

    def __init__(self, readings):
        self._readings = list(readings)
        self.sleeps = []

    def monotonic(self):
        return self._readings.pop(0)

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 4))

    @staticmethod
    def time():
        return 0.0


def test_rate_limiter_lets_the_first_call_through_then_spaces_the_rest(monkeypatch):
    clock = _SteppedClock([10.0, 10.0, 10.2, 10.2])
    monkeypatch.setattr(db, 'time', clock)
    limiter = db._RateLimiter()
    limiter.wait('uniprot')
    limiter.wait('uniprot')

    assert clock.sleeps == [0.8]                 # 1.0 s interval, 0.2 s elapsed
    assert limiter._last_call == {'uniprot': 10.2}
    assert list(limiter._locks) == ['uniprot']   # one lock per service, created once


def test_rate_limiter_gives_each_service_its_own_bucket(monkeypatch):
    """Two different services must not push each other's schedule back."""
    clock = _SteppedClock([10.0, 10.0, 10.1, 10.1])
    monkeypatch.setattr(db, 'time', clock)
    limiter = db._RateLimiter()
    limiter.wait('ensembl')     # 0.2 s interval, nothing elapsed yet
    limiter.wait('kegg')        # separate bucket, first call

    assert clock.sleeps == []
    assert limiter._last_call == {'ensembl': 10.0, 'kegg': 10.1}


def test_rate_limiter_falls_back_to_a_half_second_interval_for_unknown_services(monkeypatch):
    clock = _SteppedClock([5.0, 5.0, 5.1, 5.1])
    monkeypatch.setattr(db, 'time', clock)
    limiter = db._RateLimiter()
    limiter.wait('somewhere-new')
    limiter.wait('somewhere-new')

    assert clock.sleeps == [0.4]


# ── _json_response ───────────────────────────────────────────────────────────

def test_json_response_rejects_an_empty_body_by_name():
    assert db._json_response(b'{"a": 1}') == {'a': 1}

    with pytest.raises(ValueError) as excinfo:
        db._json_response(b'')

    assert str(excinfo.value) == 'Empty response body (expected JSON)'


def test_json_response_previews_a_long_malformed_body():
    body = b'{' + b'x' * 300

    with pytest.raises(ValueError) as excinfo:
        db._json_response(body)

    assert str(excinfo.value).startswith('Malformed JSON response: ')
    assert repr(('{' + 'x' * 199)[:200]) in str(excinfo.value)


# ── NCBI ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('body,expected', [
    (b'{"results": []}', dict),
    (b'[1, 2]', 'list'),
    (b'"just a string"', 'str'),
    (b'42', 'int'),
    (b'null', 'NoneType'),
])
def test_json_response_accepts_an_object_and_names_any_other_shape(body, expected):
    """One rule for every endpoint, instead of each one discovering its own.

    Only ``search_ensembl`` used to check this, so a UniProt or PDB array came
    back as ``error="'list' object has no attribute 'get'"`` — an internal
    exception standing in for a statement about the response.
    """
    if expected is dict:
        assert db._json_response(body) == json.loads(body)
        return
    with pytest.raises(ValueError) as excinfo:
        db._json_response(body)
    assert str(excinfo.value).startswith(
        f'Expected a JSON object at the top level, got {expected}. Body preview:')


def test_search_ncbi_builds_esearch_and_esummary_urls_with_credentials(monkeypatch):
    urls = []

    def payload(url, kwargs):
        urls.append((url, kwargs))
        if 'esearch' in url:
            return json.dumps({'esearchresult': {'idlist': ['1', '2']}}).encode()
        return json.dumps({'result': {
            '1': {'title': 'A', 'organism': 'Homo sapiens', 'accessionversion': 'NM_1'},
            '2': {'title': 'B', 'organism': 'Mus musculus', 'accessionversion': 'NM_2'},
        }}).encode()

    asked = fake_http(monkeypatch, payload)
    monkeypatch.setattr(db, 'get_api_key',
                        lambda name: {'ncbi_email': 'me@example.com',
                                      'ncbi_api_key': 'KEY'}[name])

    res = db.search_ncbi('p53 human', database='protein', max_results=3)

    assert not res.error
    assert asked[0][0].startswith('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?')
    assert 'db=protein' in asked[0][0]
    assert 'term=p53+human' in asked[0][0]
    assert 'retmax=3' in asked[0][0] and 'retmode=json' in asked[0][0]
    assert 'email=me%40example.com' in asked[0][0]
    assert 'api_key=KEY' in asked[0][0]
    # the summary call must carry the key too — it used to be dropped
    assert 'esummary.fcgi' in asked[1][0]
    assert 'id=1%2C2' in asked[1][0]
    assert 'api_key=KEY' in asked[1][0] and 'email=me%40example.com' in asked[1][0]
    assert [kw['service'] for _, kw in asked] == ['ncbi', 'ncbi']
    assert asked[0][1]['timeout'] == db.DEFAULT_TIMEOUT
    assert res.records == [
        {'id': '1', 'title': 'A', 'organism': 'Homo sapiens', 'accession': 'NM_1'},
        {'id': '2', 'title': 'B', 'organism': 'Mus musculus', 'accession': 'NM_2'},
    ]
    assert res.data == {'count': 2}


def test_search_ncbi_skips_ids_the_summary_did_not_return(monkeypatch):
    """An id in the idlist with no esummary entry yields no record, not a crash."""
    def payload(url, kwargs):
        if 'esearch' in url:
            return json.dumps({'esearchresult': {'idlist': ['1', '2', '3']}}).encode()
        return json.dumps({'result': {
            '1': {'title': 'A', 'organism': 'Homo sapiens', 'accessionversion': 'NM_1'},
            '3': {'title': 'C', 'organism': 'Mus musculus', 'accessionversion': 'NM_3'},
        }}).encode()

    fake_http(monkeypatch, payload)

    res = db.search_ncbi('q')

    assert [r['id'] for r in res.records] == ['1', '3']
    assert res.data == {'count': 3}         # count is the idlist, not the records


def test_search_ncbi_email_is_prompted_only_when_not_configured(monkeypatch):
    asked = fake_http(monkeypatch, json.dumps({'esearchresult': {'idlist': []}}).encode())
    prompts = []
    monkeypatch.setattr(db, 'get_api_key', lambda name: '')
    monkeypatch.setattr(db, 'prompt_api_key',
                        lambda service, description='': prompts.append(service) or 'p@x.org')

    res = db.search_ncbi('nothing matches')

    assert prompts == ['ncbi_email']
    assert 'email=p%40x.org' in asked[0][0]
    assert res.records == [] and res.data == {'count': 0}


@pytest.mark.parametrize('exc,expected', [
    (lambda: _http_error(403, 'Forbidden'), 'HTTP 403: Forbidden'),
    (lambda: urllib.error.URLError('refused'), 'Connection failed: <urlopen error ref'),
    (lambda: ValueError('bad json'), 'Invalid response: bad json'),
    (lambda: RuntimeError('boom'), 'boom'),
])
def test_search_ncbi_turns_every_failure_into_an_error_result(exc, expected, monkeypatch):
    def raiser(url, **kwargs):
        raise exc()

    monkeypatch.setattr(db, '_http_get', raiser)

    res = db.search_ncbi('q')

    assert res.source == 'ncbi' and res.query == 'q'
    assert res.records == [] and res.data == {}
    assert res.error.startswith(expected)


def test_search_ncbi_uses_an_explicit_email_without_prompting(monkeypatch):
    """``email=None`` means "look it up"; a passed address means "use as-is"."""
    asked = fake_http(monkeypatch, json.dumps({'esearchresult': {'idlist': []}}).encode())
    prompts = []
    monkeypatch.setattr(db, 'prompt_api_key',
                        lambda service, description='': prompts.append(service) or 'p@x.org')

    res = db.search_ncbi('q', email='curator@example.org')

    assert prompts == []
    assert 'email=curator%40example.org' in asked[0][0]
    assert res.data == {'count': 0} and res.cached is False


def test_identical_search_ncbi_call_is_served_from_cache_and_flagged(monkeypatch):
    asked = fake_http(monkeypatch, json.dumps({'esearchresult': {'idlist': []}}).encode())

    first = db.search_ncbi('tp53')
    second = db.search_ncbi('tp53')

    assert len(asked) == 1                      # the hit never reached the network
    assert (first.cached, second.cached) == (False, True)
    assert second.data == first.data == {'count': 0}


def test_search_ncbi_cache_key_includes_the_email_and_page_size(monkeypatch):
    """Same term, different request: each must go out separately."""
    asked = fake_http(monkeypatch, json.dumps({'esearchresult': {'idlist': []}}).encode())

    db.search_ncbi('q', email='a@example.org', max_results=5)
    db.search_ncbi('q', email='b@example.org', max_results=5)
    db.search_ncbi('q', email='b@example.org', max_results=6)

    assert len(asked) == 3


def test_fetch_ncbi_sequence_returns_fasta_and_caches_the_record(monkeypatch):
    fasta = b'>NM_1.1 Homo sapiens example\nACGTACGTACGT\n'
    asked = fake_http(monkeypatch, fasta)
    monkeypatch.setattr(db, 'get_api_key',
                        lambda name: 'KEY' if name == 'ncbi_api_key'
                        else 'me@lab.edu' if name == 'ncbi_email' else '')

    first = db.fetch_ncbi_sequence('NM_1.1', database='nucleotide')
    second = db.fetch_ncbi_sequence('NM_1.1', database='nucleotide')

    assert asked[0][0].startswith('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?')
    assert 'db=nucleotide&id=NM_1.1&rettype=fasta&retmode=text' in asked[0][0]
    assert 'email=me%40lab.edu' in asked[0][0]
    assert 'api_key=KEY' in asked[0][0]
    assert asked[0][1] == {'timeout': 60, 'service': 'ncbi'}
    assert first.records == [{'fasta': fasta.decode()}]
    assert first.data == {'format': 'fasta'}
    assert first.cached is False
    assert second.cached is True            # served from the in-memory TTL cache
    assert second.records == first.records
    assert len(asked) == 1                  # the hit did not go back to the network


def test_fetch_ncbi_sequence_invents_no_contact_address_when_none_is_configured(monkeypatch):
    """The Entrez endpoints ask the user for an email; they may not invent one.

    ``fetch_ncbi_sequence`` used to send a hardcoded ``biosuite@example.com``,
    while ``search_ncbi`` beside it prompted for the real address — so the one
    endpoint that never asked was the one that answered on the user's behalf, with
    an address that exists for nobody.
    """
    asked = fake_http(monkeypatch, b'>NM_1.1 ACGT')

    db.fetch_ncbi_sequence('NM_1.1')

    assert 'email=' not in asked[0][0]
    assert 'api_key=' not in asked[0][0]


def test_fetch_ncbi_sequence_reports_an_empty_body_as_an_error(monkeypatch):
    fake_http(monkeypatch, b'   \n')

    res = db.fetch_ncbi_sequence('NM_1.1')

    assert res.error == 'Empty sequence response'
    assert res.records == [] and res.data == {}


@pytest.mark.parametrize('exc,expected', [
    (lambda: _http_error(404, 'Not Found'), 'HTTP 404: Not Found'),
    (lambda: OSError('dns failed'), 'Connection failed: dns failed'),
    (lambda: TypeError('bad thing'), 'bad thing'),
])
def test_fetch_ncbi_sequence_reports_transport_failures(exc, expected, monkeypatch):
    def raiser(url, **kwargs):
        raise exc()

    monkeypatch.setattr(db, '_http_get', raiser)

    res = db.fetch_ncbi_sequence('NM_9.9')

    assert res.source == 'ncbi' and res.query == 'NM_9.9'
    assert res.error.startswith(expected)


# ── UniProt ──────────────────────────────────────────────────────────────────

UNIPROT_SEARCH_BODY = json.dumps({'results': [{
    'primaryAccession': 'P04637',
    'proteinDescription': {'recommendedName': {'fullName': {'value': 'Cellular tumor antigen p53'}}},
    'organism': {'scientificName': 'Homo sapiens'},
    'sequence': {'length': 393},
    'genes': [{'geneName': {'value': 'TP53'}}],
}, {
    'primaryAccession': 'Q99999',
    'sequence': {'length': 10},
}]}).encode()


def test_search_uniprot_quotes_the_query_and_limits_the_page_size(monkeypatch):
    asked = fake_http(monkeypatch, UNIPROT_SEARCH_BODY)

    res = db.search_uniprot('p53 gene:9606', max_results=7)

    assert asked[0][0] == ('https://rest.uniprot.org/uniprotkb/search'
                           '?query=p53%20gene%3A9606&format=json&size=7')
    assert asked[0][1] == {'timeout': db.DEFAULT_TIMEOUT, 'service': 'uniprot'}
    assert res.records == [
        {'accession': 'P04637', 'protein': 'Cellular tumor antigen p53',
         'organism': 'Homo sapiens', 'length': 393, 'gene': 'TP53'},
        {'accession': 'Q99999', 'protein': '', 'organism': '', 'length': 10, 'gene': ''},
    ]


@pytest.mark.parametrize('exc,expected', [
    (lambda: _http_error(500, 'Internal Server Error'), 'HTTP 500: Internal Server Error'),
    (lambda: urllib.error.URLError('reset by peer'), 'Connection failed: <urlopen error'),
    (lambda: ValueError('Empty response body (expected JSON)'),
     'Invalid response: Empty response body (expected JSON)'),
])
def test_search_uniprot_reports_http_connection_and_parse_failures(exc, expected, monkeypatch):
    def raiser(url, **kwargs):
        raise exc()

    monkeypatch.setattr(db, '_http_get', raiser)

    res = db.search_uniprot('p53')

    assert res.source == 'uniprot' and res.records == []
    assert res.error.startswith(expected)


def test_search_uniprot_reports_a_failure_that_is_not_an_http_problem(caplog, monkeypatch):
    """The last arm: anything that is not HTTPError, URLError or ValueError.

    It returns the exception's own text as the result's error and logs the
    traceback, so an internal defect in this module cannot arrive at the user's
    screen as an empty UniProt answer.  This arm was reached only by a
    grab-bag test until cluster 26 dissolved that file, and the coverage diff of
    databases.py:606-608 is what brought it home.
    """
    import logging

    def raiser(url, **kwargs):
        raise RuntimeError("no such method 'post'")

    monkeypatch.setattr(db, '_http_get', raiser)

    with caplog.at_level(logging.ERROR, logger="biosuite.core.databases"):
        res = db.search_uniprot('p53')

    assert res.error == "no such method 'post'"
    assert res.source == 'uniprot' and res.records == [] and res.data == {}
    assert "[uniprot] Unexpected error" in caplog.text
    assert "RuntimeError: no such method 'post'" in caplog.text, \
        "logger.exception put the traceback in the log, not in the user's answer"


def test_search_uniprot_second_request_is_answered_from_the_cache(monkeypatch):
    asked = fake_http(monkeypatch, UNIPROT_SEARCH_BODY)

    first = db.search_uniprot('p53', max_results=2)
    second = db.search_uniprot('p53', max_results=2)
    third = db.search_uniprot('p53', max_results=3)   # different page size, different key

    assert len(asked) == 2                     # only the two distinct keys went out
    assert (first.cached, second.cached, third.cached) == (False, True, False)
    assert second.records == first.records


def test_search_uniprot_reports_a_body_of_the_wrong_shape(monkeypatch):
    """A JSON array is not a UniProt page: the entry is named, not invented."""
    fake_http(monkeypatch, b'[{"primaryAccession": "P04637"}]')

    res = db.search_uniprot('p53')

    assert res.records == []
    assert res.error.startswith(
        "Invalid response: Expected a JSON object at the top level, got list")


def test_fetch_uniprot_collects_go_terms_and_caches_the_entry(monkeypatch):
    body = json.dumps({
        'primaryAccession': 'P04637',
        'proteinDescription': {'recommendedName': {'fullName': {'value': 'p53'}}},
        'organism': {'scientificName': 'Homo sapiens'},
        'sequence': {'value': 'MEEPQSD', 'length': 7},
        'uniProtKBCrossReferences': [
            {'database': 'GO', 'properties': [{'value': 'GO:0006915'}]},
            {'database': 'GO', 'properties': [{'value': 'GO:0000956'}]},
            {'database': 'Pfam', 'properties': [{'value': 'P53'}]},
            {'database': 'GO'},
        ],
    }).encode()
    asked = fake_http(monkeypatch, body)

    first = db.fetch_uniprot('P04637')
    second = db.fetch_uniprot('P04637')

    assert asked[0][0] == 'https://rest.uniprot.org/uniprotkb/P04637.json'
    assert first.records == [{'accession': 'P04637', 'protein': 'p53',
                              'organism': 'Homo sapiens', 'sequence': 'MEEPQSD',
                              'length': 7, 'go_terms': ['GO:0006915', 'GO:0000956', '']}]
    assert second.cached is True and second.records == first.records
    assert len(asked) == 1


@pytest.mark.parametrize('exc,expected', [
    (lambda: _http_error(404, 'Not Found'), 'HTTP 404: Not Found'),
    (lambda: urllib.error.URLError('closed'), 'Connection failed: <urlopen error closed'),
])
def test_fetch_uniprot_reports_http_and_connection_failures(exc, expected, monkeypatch):
    def raiser(url, **kwargs):
        raise exc()

    monkeypatch.setattr(db, '_http_get', raiser)

    res = db.fetch_uniprot('P04637')

    assert res.source == 'uniprot' and res.query == 'P04637'
    assert res.error.startswith(expected)


@pytest.mark.parametrize('body,expected', [
    (b'', 'Invalid response: Empty response body (expected JSON)'),
    (b'[{"primaryAccession": "P04637"}]',
     'Invalid response: Expected a JSON object at the top level, got list'),
])
def test_fetch_uniprot_reports_an_unusable_body(body, expected, monkeypatch):
    fake_http(monkeypatch, body)

    res = db.fetch_uniprot('P04637')

    assert res.records == [] and res.data == {}
    assert res.error.startswith(expected)


def test_fetch_uniprot_hit_is_flagged_and_skips_the_network(monkeypatch):
    asked = fake_http(monkeypatch, b'{"primaryAccession": "P04637"}')

    db.fetch_uniprot('P04637')
    second = db.fetch_uniprot('P04637')

    assert len(asked) == 1
    assert second.cached is True


# ── PDB (POST through the shared transport) ──────────────────────────────────────────

PDB_BODY = json.dumps({'result_set': [
    {'identifier': '1TUP', 'title': 'Structure of glutathione synthetase'},
    {'identifier': '2XWR'},
]}).encode()


def test_search_pdb_posts_a_json_query_to_the_rcsb_search_endpoint(monkeypatch, limiter):
    calls = fake_urlopen(monkeypatch, lambda n: PDB_BODY)

    res = db.search_pdb('glutathione synthetase', max_results=5)

    assert limiter.services == ['pdb']
    assert len(calls) == 1
    req = calls[0]
    assert req.full_url == 'https://search.rcsb.org/rcsbsearch/v2/query'
    assert req.get_method() == 'POST'
    assert req.get_header('Content-type') == 'application/json'
    assert json.loads(req.data) == {
        'query': {'type': 'terminal', 'service': 'full_text',
                  'parameters': {'value': 'glutathione synthetase'}},
        'return_type': 'entry',
        'request_options': {'results_content_type': ['experimental'],
                            'pager': {'start': 0, 'rows': 5}},
    }
    assert res.records == [
        {'pdb_id': '1TUP', 'title': 'Structure of glutathione synthetase',
         'url': 'https://www.rcsb.org/structure/1TUP'},
        {'pdb_id': '2XWR', 'title': '', 'url': 'https://www.rcsb.org/structure/2XWR'},
    ]


def test_search_pdb_caches_the_entry_and_marks_the_hit(monkeypatch):
    calls = fake_urlopen(monkeypatch, lambda n: PDB_BODY)

    first = db.search_pdb('p53')
    second = db.search_pdb('p53')

    assert len(calls) == 1
    assert first.cached is False and second.cached is True
    assert second.records == first.records


def test_search_pdb_retries_429_honoring_retry_after_then_succeeds(monkeypatch, clock):
    calls = fake_urlopen(monkeypatch,
                         lambda n: _http_error(429, 'Too Many Requests', '0.5')
                         if n == 1 else PDB_BODY)

    res = db.search_pdb('kinase')

    assert clock.sleeps == [0.5]
    assert len(calls) == 2
    assert [r['pdb_id'] for r in res.records] == ['1TUP', '2XWR']


def test_search_pdb_retries_a_transient_connection_error_then_succeeds(monkeypatch, clock):
    calls = fake_urlopen(monkeypatch,
                         lambda n: urllib.error.URLError('reset') if n == 1 else PDB_BODY)

    res = db.search_pdb('kinase')

    assert clock.sleeps == [1.5]
    assert len(calls) == 2
    assert not res.error


def test_search_pdb_raises_a_non_retryable_status_on_the_first_attempt(monkeypatch, clock):
    calls = fake_urlopen(monkeypatch, lambda n: _http_error(400, 'Bad Request'))

    res = db.search_pdb('bad query syntax')

    assert res.error == 'HTTP 400: Bad Request'
    assert len(calls) == 1 and clock.sleeps == []


def test_search_pdb_gives_up_after_retrying_a_persistent_429(monkeypatch, clock):
    """The POST search shares ``_http_get``, so it stops waiting once it will not retry."""
    calls = fake_urlopen(monkeypatch, lambda n: _http_error(429, 'Too Many Requests'))

    res = db.search_pdb('kinase')

    assert res.error == 'HTTP 429: Too Many Requests'
    assert len(calls) == db.DEFAULT_MAX_RETRIES
    assert clock.sleeps == [1.5, 2.25]      # no sleep after the final 429


def test_search_pdb_posts_the_query_and_honours_a_429_date_through_the_shared_path(monkeypatch, clock):
    """RCSB wants a POST, and its rate-limit header may be an HTTP-date.

    Both used to be handled by a copy of the retry loop inside ``search_pdb``,
    which is why the two paths could drift: stubbing ``_http_get`` in a probe
    left PDB still driving ``urlopen`` itself.
    """
    sent = []

    def spy(req, timeout=None):
        sent.append(req)
        raise _http_error(429, 'Too Many Requests', '2')

    monkeypatch.setattr(urllib.request, 'urlopen', spy)
    res = db.search_pdb('kinase')

    assert len(sent) == db.DEFAULT_MAX_RETRIES
    req = sent[0]
    assert req.get_method() == 'POST'
    assert json.loads(req.data)['query']['service'] == 'full_text'
    assert req.get_header('Content-type') == 'application/json'
    assert clock.sleeps == [2.0, 2.0]       # Retry-After honoured, not the curve
    assert res.error == 'HTTP 429: Too Many Requests'


def test_search_pdb_names_a_result_entry_that_is_not_an_object(monkeypatch):
    """A shape the parser cannot map is reported, and the log keeps the traceback.

    ``search_pdb`` deliberately keeps a last-resort arm: these endpoints are
    contracted never to raise at the API boundary (see test_search_all_*), so the
    contract is that the *message* names what went wrong rather than that the
    arm does not exist.
    """
    fake_urlopen(monkeypatch,
                 lambda n: json.dumps({'result_set': ['1TUP', {'identifier': '2XWR'}]}).encode())

    res = db.search_pdb('p53')

    assert res.records == []
    assert res.error == "'str' object has no attribute 'get'"


def test_search_pdb_raises_the_last_connection_error_after_all_retries(monkeypatch, clock):
    calls = fake_urlopen(monkeypatch, lambda n: TimeoutError('timed out'))

    res = db.search_pdb('kinase')

    assert res.error == 'Connection failed: timed out'
    assert len(calls) == db.DEFAULT_MAX_RETRIES
    assert clock.sleeps == [1.5, 2.25]


@pytest.mark.parametrize('body,expected', [
    (b'{oops', "Invalid response: Malformed JSON response"),
    (b'[1, 2]', "Invalid response: Expected a JSON object at the top level, got list"),
])
def test_search_pdb_names_a_body_it_cannot_use(body, expected, monkeypatch):
    """The shape is named by ``_json_response``, not discovered as an AttributeError."""
    fake_urlopen(monkeypatch, lambda n: body)

    res = db.search_pdb('whatever')

    assert res.records == []
    assert res.error.startswith(expected)


# ── KEGG ─────────────────────────────────────────────────────────────────────

def test_search_kegg_appends_the_configured_email(monkeypatch):
    asked = fake_http(monkeypatch, b'path:hsa04010\tMAPK signaling\n')
    monkeypatch.setattr(db, 'get_api_key', lambda name: 'me@kegg.org' if name == 'kegg_email' else '')

    res = db.search_kegg('mapk', database='pathway')

    assert asked[0][0] == 'https://rest.kegg.jp/find/pathway/mapk?email=me%40kegg.org'
    assert asked[0][1] == {'timeout': db.DEFAULT_TIMEOUT, 'service': 'kegg'}
    assert res.records == [{'id': 'path:hsa04010', 'description': 'MAPK signaling'}]


def test_search_kegg_splits_on_the_tab_and_skips_unusable_rows(monkeypatch):
    """KEGG ids contain their own colon, and blank/one-field rows carry no entry."""
    body = ("path:hsa04614\tRenal cell carcinoma\n"
            "\n"
            "hsa04010   MAPK signaling pathway\n"
            "orphan_row_without_a_description\n"
            "path:hsa00010\tGlycolysis\n").encode()
    fake_http(monkeypatch, body)

    res = db.search_kegg('mapk', max_results=5)

    assert res.records == [
        {'id': 'path:hsa04614', 'description': 'Renal cell carcinoma'},
        {'id': 'hsa04010', 'description': 'MAPK signaling pathway'},
        {'id': 'path:hsa00010', 'description': 'Glycolysis'},
    ]


def test_search_kegg_limits_rows_and_caches_the_page(monkeypatch):
    asked = fake_http(monkeypatch,
                      b'path:a\tone\npath:b\ttwo\npath:c\tthree\n')

    first = db.search_kegg('x', max_results=2)
    second = db.search_kegg('x', max_results=2)

    assert [r['id'] for r in first.records] == ['path:a', 'path:b']
    assert second.cached is True and len(asked) == 1


@pytest.mark.parametrize('exc,expected', [
    (lambda: _http_error(502, 'Bad Gateway'), 'HTTP 502: Bad Gateway'),
    (lambda: urllib.error.URLError('no route to host'), 'Connection failed: <urlopen error'),
    (lambda: KeyError('missing'), "'missing'"),
])
def test_search_kegg_reports_http_connection_and_unexpected_failures(exc, expected, monkeypatch):
    def raiser(url, **kwargs):
        raise exc()

    monkeypatch.setattr(db, '_http_get', raiser)

    res = db.search_kegg('mapk')

    assert res.source == 'kegg' and res.records == []
    assert res.error.startswith(expected)


def test_fetch_kegg_pathway_returns_the_raw_record_and_caches_it(monkeypatch):
    kgml = b'ENTRY       hsa04610 Pathway\nNAME          Complement and coagulation\n'
    asked = fake_http(monkeypatch, kgml)
    monkeypatch.setattr(db, 'get_api_key', lambda name: 'me@kegg.org')

    first = db.fetch_kegg_pathway('hsa04610')
    second = db.fetch_kegg_pathway('hsa04610')

    assert asked[0][0] == 'https://rest.kegg.jp/get/hsa04610?email=me%40kegg.org'
    assert first.records == [{'text': kgml.decode()}]
    assert first.data == {'format': 'kgml'}
    assert second.cached is True and len(asked) == 1


def test_fetch_kegg_pathway_without_an_email_uses_a_bare_url(monkeypatch):
    asked = fake_http(monkeypatch, b'ENTRY  hsa00001\n')

    db.fetch_kegg_pathway('hsa00001')

    assert asked[0][0] == 'https://rest.kegg.jp/get/hsa00001'


@pytest.mark.parametrize('exc,expected', [
    (lambda: _http_error(404, 'Not Found'), 'HTTP 404: Not Found'),
    (lambda: OSError('connection reset'), 'Connection failed: connection reset'),
    (lambda: ValueError('nope'), 'nope'),
])
def test_fetch_kegg_pathway_reports_failures(exc, expected, monkeypatch):
    def raiser(url, **kwargs):
        raise exc()

    monkeypatch.setattr(db, '_http_get', raiser)

    res = db.fetch_kegg_pathway('hsa99999')

    assert res.source == 'kegg' and res.query == 'hsa99999'
    assert res.error.startswith(expected)


# ── Ensembl ──────────────────────────────────────────────────────────────────

ENSEMBL_GENE = {'id': 'ENSG00000012345', 'description': 'BRCA2 [Source:HGNC]',
                'species': 'sapiens', 'biotype': 'protein_coding',
                'start': 55915239, 'end': 56799391}


def test_search_ensembl_hit_is_flagged_and_needs_no_second_request(monkeypatch):
    asked = fake_http(monkeypatch, json.dumps(ENSEMBL_GENE).encode())

    first = db.search_ensembl('BRCA2', species='sapiens')
    second = db.search_ensembl('BRCA2', species='sapiens')

    assert asked[0][0] == ('https://rest.ensembl.org/lookup/symbol/sapiens/BRCA2'
                           '?content-type=application/json')
    assert first.records == [{'id': 'ENSG00000012345',
                              'description': 'BRCA2 [Source:HGNC]',
                              'species': 'sapiens', 'biotype': 'protein_coding',
                              'start': 55915239, 'end': 56799391}]
    assert len(asked) == 1 and second.cached is True


def test_search_ensembl_names_a_json_array_instead_of_a_crash(monkeypatch):
    """/xref endpoints answer with a list; the gene lookup contract is one object.

    The check used to live in ``search_ensembl`` alone, so a UniProt or PDB
    array came back as ``'list' object has no attribute 'get'``; ``_json_response``
    now names the shape for every endpoint and this pins the shared wording.
    """
    fake_http(monkeypatch, b'[{"id": "ENSG00000012345"}]')

    res = db.search_ensembl('BRCA2')

    assert res.records == []
    assert res.error.startswith(
        'Invalid response: Expected a JSON object at the top level, got list')


@pytest.mark.parametrize('exc,expected', [
    (lambda: _http_error(500, 'Internal Server Error'), 'HTTP 500: Internal Server Error'),
    (lambda: urllib.error.URLError('temporarily unavailable'),
     'Connection failed: <urlopen error temporarily'),
    (lambda: ValueError('Ensembl returned a str, expected one JSON object'),
     'Invalid response: Ensembl returned a str'),
    (lambda: RuntimeError('unexpected'), 'unexpected'),
])
def test_search_ensembl_reports_every_failure_arm(exc, expected, monkeypatch):
    def raiser(url, **kwargs):
        raise exc()

    monkeypatch.setattr(db, '_http_get', raiser)

    res = db.search_ensembl('BRCA2', species='mouse')

    assert res.source == 'ensembl' and res.records == []
    assert res.error.startswith(expected)


# ── cache management ─────────────────────────────────────────────────────────

def test_search_all_returns_one_result_per_known_database(monkeypatch):
    def payload(url, kwargs):
        if 'esearch' in url:
            return json.dumps({'esearchresult': {'idlist': []}}).encode()
        if 'uniprot' in url:
            return json.dumps({'results': []}).encode()
        if 'kegg' in url:
            return b''
        return json.dumps({'id': 'ENSG00000141510', 'species': 'homo_sapiens'}).encode()

    asked = fake_http(monkeypatch, payload)

    results = db.search_all('tp53', databases=['ncbi', 'uniprot', 'kegg', 'ensembl'])

    assert sorted(results) == ['ensembl', 'kegg', 'ncbi', 'uniprot']
    assert all(r.error == '' for r in results.values())
    assert [url for url, _ in asked] == [
        'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=nucleotide&term=tp53'
        '&retmax=10&retmode=json',
        'https://rest.uniprot.org/uniprotkb/search?query=tp53&format=json&size=10',
        'https://rest.kegg.jp/find/pathway/tp53',
        'https://rest.ensembl.org/lookup/symbol/human/tp53?content-type=application/json',
    ]


def test_search_all_answers_an_unknown_database_name_with_an_error_result(monkeypatch, caplog):
    """A name the module does not know is a failure, not an absent key.

    It used to be logged and dropped, so ``search_all(q, ['ncbi', 'clinklink'])``
    returned one entry and the caller could not tell a typo from a database that
    was never asked — while a database whose *function* raised did get an error
    entry.  Both paths now answer the same way.
    """
    asked = fake_http(monkeypatch, b'{"esearchresult": {"idlist": []}}')

    results = db.search_all('tp53', databases=['ncbi', 'clinklink'])

    assert list(results) == ['ncbi', 'clinklink']
    assert len(asked) == 1                       # nothing was fetched for the bad name
    assert results['clinklink'].source == 'clinklink'
    assert results['clinklink'].query == 'tp53'
    assert results['clinklink'].records == []
    assert results['clinklink'].error == (
        "Unknown database: clinklink. Known: ensembl, kegg, ncbi, pdb, uniprot.")
    assert 'Unknown database: clinklink' in caplog.text


def test_search_all_defaults_to_the_four_sequence_databases(monkeypatch):
    """No ``databases`` argument means ncbi + uniprot + pdb + kegg, Ensembl left out.

    The name list is a constant the test can point at, because the docstring used
    to promise "default: all" while four of the five services were searched.
    """
    asked = fake_http(monkeypatch, lambda url, kwargs: (
        json.dumps({'esearchresult': {'idlist': []}}).encode() if 'ncbi' in url
        else json.dumps({'results': []}).encode() if 'uniprot' in url
        else json.dumps({'result_set': []}).encode() if 'rcsb' in url
        else b''))
    posted = fake_urlopen(monkeypatch, lambda n: b'{"must not be used": 1}')

    results = db.search_all('tp53')

    assert sorted(results) == sorted(db.DEFAULT_SEARCH_DATABASES)
    assert all(r.source == name for name, r in results.items())
    assert all(not r.error for r in results.values())
    assert len(asked) == 4                       # every service went through _http_get
    assert posted == []                          # PDB no longer bypasses it


def test_search_all_keeps_going_when_one_database_function_explodes(monkeypatch, caplog):
    fake_http(monkeypatch, b'{"esearchresult": {"idlist": []}}')

    def exploded(*args, **kwargs):
        raise RuntimeError('uniprot exploded')

    monkeypatch.setattr(db, 'search_uniprot', exploded)

    results = db.search_all('tp53', databases=['ncbi', 'uniprot'])

    assert results['ncbi'].error == ''
    assert results['uniprot'].source == 'uniprot'
    assert results['uniprot'].query == 'tp53'
    assert results['uniprot'].error == 'uniprot exploded'
    assert results['uniprot'].records == []
    assert 'uniprot failed: uniprot exploded' in caplog.text


# ── the urllib import guard ──────────────────────────────────────────────────

@contextlib.contextmanager
def _executed_without(module, *names):
    """Re-run ``module`` with ``names`` hidden from the import system.

    The guard decides ``HAS_URLLIB`` while the module body executes, so the
    other side can only be reached by executing the file again — into a fresh
    module object, leaving the shared ``biosuite.core.databases`` intact.
    """
    class HiddenFinder:
        def find_spec(self, name, path=None, target=None):
            if name in names:
                raise ImportError(f"{name} is hidden by this test")
            return None

    finder = HiddenFinder()
    parked = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    sys.meta_path.insert(0, finder)
    alias = f"{module.__name__}_without_{names[0].replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(alias, module.__file__)
    fresh = importlib.util.module_from_spec(spec)
    sys.modules[alias] = fresh
    try:
        spec.loader.exec_module(fresh)
        yield fresh
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(parked)
        sys.modules.pop(alias, None)


ENTRIES = [
    ('search_ncbi', 'tp53', 'ncbi'),
    ('fetch_ncbi_sequence', 'NM_1.1', 'ncbi'),
    ('search_uniprot', 'tp53', 'uniprot'),
    ('fetch_uniprot', 'P04637', 'uniprot'),
    ('search_pdb', 'p53', 'pdb'),
    ('search_kegg', 'mapk', 'kegg'),
    ('fetch_kegg_pathway', 'hsa04610', 'kegg'),
    ('search_ensembl', 'TP53', 'ensembl'),
]


@pytest.mark.parametrize('func_name,argument,source', ENTRIES)
def test_every_endpoint_reports_instead_of_raising_without_urllib(func_name, argument, source):
    with _executed_without(db, 'urllib.request') as fresh:
        assert fresh.HAS_URLLIB is False
        res = getattr(fresh, func_name)(argument)

    assert res.source == source
    assert res.query == argument
    assert res.error == 'urllib not available'
    assert res.records == [] and res.data == {}
