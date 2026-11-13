"""
Database API integrations: NCBI Entrez, UniProt, Ensembl, PDB, KEGG.

All APIs are free for academic use.  Users can optionally provide API keys
for higher rate limits via the config system.

Features:
  - Retry logic with exponential backoff for transient failures, honouring a
    capped Retry-After (both of its RFC 9110 forms)
  - In-memory TTL cache to avoid redundant API calls, bounded at
    MAX_CACHE_ENTRIES with expired-first eviction
  - Per-service rate limiting
  - Proper HTTP error handling (4xx/5xx/status codes)
  - Malformed JSON safe-parsing, including the top-level shape
  - Identifiers checked before they are placed in a request path
  - Configurable timeouts

The contract every search_*() / fetch_*() function keeps: it returns a DBResult
and never raises.  An unusable argument (a page size below one, an identifier
that would leave its path segment) comes back the same way a refused connection
does — in ``DBResult.error``, with the detail in the log.  Nothing here opens a
socket on import or on ``import biosuite``.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

try:
    import urllib.error
    import urllib.parse
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

from .log import get_logger
from .utils import get_api_key, prompt_api_key


@dataclass
class DBResult:
    """Result from a biological database query.

    Attributes:
        source: Database name (ncbi, uniprot, pdb, etc.).
        query: The original query string or identifier.
        data: Parsed response data as dictionary.
        records: List of result records.
        error: Error message if query failed.
        cached: Whether the result was served from cache.
    """
    source: str
    query: str
    data: Dict[str, Any] = field(default_factory=dict)
    records: List[Any] = field(default_factory=list)
    error: str = ""
    cached: bool = False


logger = get_logger(__name__)

# ── Retry / timeout / rate-limit config ─────────────────────────────────────

DEFAULT_TIMEOUT = 30          # seconds per HTTP request
DEFAULT_MAX_RETRIES = 3       # max retry attempts for transient errors
DEFAULT_RETRY_BACKOFF = 1.5   # exponential backoff base (seconds)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Longest wait a server may ask for.  Retry-After is a suggestion, and a caller
# stuck in the GUI thread should not be held for hours by one header: waits above
# this are shortened to it, and the retry budget then decides whether to try again.
MAX_RETRY_WAIT = 60.0

# NCBI asks every tool for a contact address.  Both Entrez entry points ask the
# user for the same value instead of one of them inventing one.
NCBI_EMAIL_PROMPT = ("Free registration at: "
                     "https://www.ncbi.nlm.nih.gov/account/")

# Largest number of cached responses kept.  Entries are dropped when they expire
# or, if the cache is full, oldest first -- see _Cache.set.
MAX_CACHE_ENTRIES = 512

# Per-service rate limits: (min_interval_seconds, label)
RATE_LIMITS = {
    'ncbi':    (0.34, 'NCBI'),   # ≤ 3 req/s without key, 10 with key
    'uniprot': (1.0, 'UniProt'),
    'pdb':     (0.5, 'PDB'),
    'kegg':    (1.0, 'KEGG'),
    'ensembl': (0.2, 'Ensembl'),
}

# Cache: {key: (timestamp, value)} with per-service TTLs
_DEFAULT_TTL = 300            # a service outside this table is cached 5 minutes
_CACHE_TTL = {
    'ncbi':    600,    # 10 min
    'uniprot': 3600,   # 1 hour
    'pdb':     1800,   # 30 min
    'kegg':    3600,   # 1 hour
    'ensembl': 1800,   # 30 min
}


# ── Internal helpers ─────────────────────────────────────────────────────────

class _RateLimiter:
    """Thread-safe per-service rate limiter using a simple token-bucket approach."""

    def __init__(self):
        """Initialize rate limiter with empty per-service state."""
        self._locks = {}
        self._last_call = {}
        self._global_lock = threading.Lock()

    def wait(self, service: str) -> None:
        """Block until the minimum interval for the given service has elapsed.

        Args:
            service: Service identifier (e.g. ncbi, uniprot).
        """
        interval, _ = RATE_LIMITS.get(service, (0.5, service))
        with self._global_lock:
            if service not in self._locks:
                self._locks[service] = threading.Lock()
                self._last_call[service] = 0.0

        lock = self._locks[service]
        with lock:
            now = time.monotonic()
            elapsed = now - self._last_call.get(service, 0.0)
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last_call[service] = time.monotonic()


_rate_limiter = _RateLimiter()


class _Cache:
    """Simple in-memory TTL cache keyed by (service, args_hash)."""

    def __init__(self):
        """Initialize empty in-memory TTL cache."""
        self._store = {}
        self._lock = threading.Lock()

    def get(self, service: str, key: str) -> Optional[Any]:
        """Retrieve a cached value if it exists and has not expired.

        Args:
            service: Service identifier for TTL lookup.
            key: Cache key string.

        Returns:
            Cached value or None if missing/expired.
        """
        ttl = _CACHE_TTL.get(service, _DEFAULT_TTL)
        full_key = f"{service}:{key}"
        with self._lock:
            entry = self._store.get(full_key)
            if entry and (time.time() - entry[0]) < ttl:
                return entry[1]
            # expired or missing
            self._store.pop(full_key, None)
        return None

    def _drop_expired_locked(self) -> None:
        """Remove every entry whose TTL has run out.  The lock must be held."""
        now = time.time()
        for full_key in list(self._store):
            service = full_key.split(':', 1)[0]
            ttl = _CACHE_TTL.get(service, _DEFAULT_TTL)
            if (now - self._store[full_key][0]) >= ttl:
                del self._store[full_key]

    def set(self, service: str, key: str, value: Any) -> None:
        """Store a value in the cache with current timestamp.

        Args:
            service: Service identifier for TTL tracking.
            key: Cache key string.
            value: Value to cache.

        An expired entry is otherwise only shed when the *same* key is asked for
        again, so a session that keeps searching new terms would hold every
        response it ever fetched — records and all — for the life of the process.
        Each store that fills the cache first drops what has expired, then the
        oldest entries if it is still over the limit.
        """
        full_key = f"{service}:{key}"
        with self._lock:
            self._store[full_key] = (time.time(), value)
            if len(self._store) <= MAX_CACHE_ENTRIES:
                return
            self._drop_expired_locked()
            overflow = len(self._store) - MAX_CACHE_ENTRIES
            if overflow > 0:
                oldest = sorted(self._store.items(), key=lambda item: item[1][0])
                for stale, _entry in oldest[:overflow]:
                    del self._store[stale]

    def invalidate(self, service: Optional[str] = None) -> None:
        """Remove cached entries for a specific service or all services.

        Args:
            service: Service to invalidate. If None, clears entire cache.
        """
        with self._lock:
            if service:
                to_remove = [k for k in self._store if k.startswith(f"{service}:")]
                for k in to_remove:
                    del self._store[k]
            else:
                self._store.clear()


_cache = _Cache()


def _cache_key(*args) -> str:
    """Derive a deterministic cache key from positional args."""
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_hit(result: "DBResult") -> "DBResult":
    """Hand a stored result back marked as served from cache.

    ``DBResult.cached`` documents "Whether the result was served from cache",
    but the object in the store is the original live response with cached=False
    and every endpoint returned it unchanged, so no caller could ever tell a
    hit from a fetch.  A copy keeps the stored entry's own flag untouched.
    """
    return replace(result, cached=True)


def _retry_after_seconds(value: Any) -> Optional[float]:
    """Convert a Retry-After header into seconds, or None if it says nothing usable.

    RFC 9110 allows two forms: a number of seconds *or* an HTTP-date.  Reading
    only the first meant a date-valued header raised out of the retry loop — and
    the endpoint reported it as ``Invalid response: could not convert string to
    float: 'Wed, 21 Oct 2015 07:28:00 GMT'``, blaming the database for our own
    header parsing.  Whatever the form, the wait is capped at MAX_RETRY_WAIT: one
    header must not park a GUI thread for hours.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            # On this interpreter parsedate_to_datetime raises ValueError for
            # every unreadable value ('', 'garbage', 'xxx' were checked) rather
            # than returning None, and it is handed a str here, so the only
            # failure to expect is the one it raises.
            when = parsedate_to_datetime(text)
        except ValueError:
            return None
        if when.tzinfo is None:
            # A date without a zone is read as UTC, the way RFC 9110 writes it.
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, min(seconds, MAX_RETRY_WAIT))


def _backoff_wait(attempt: int, headers: Any) -> float:
    """Seconds to sleep before the next attempt, honouring the server if it spoke."""
    retry_after = _retry_after_seconds(headers.get('Retry-After') if headers else None)
    if retry_after is not None:
        return retry_after
    return DEFAULT_RETRY_BACKOFF ** attempt


def _redact_url(url: str) -> str:
    """Mask the api_key credential so a retry log never carries a secret."""
    return re.sub(r'(api_key=)[^&]*', r'\1REDACTED', url)


def _http_get(url: str, *, timeout: int = DEFAULT_TIMEOUT,
              headers: Optional[Dict[str, str]] = None, service: str = 'general',
              data: Optional[bytes] = None) -> bytes:
    """
    Perform an HTTP GET (or a POST, when *data* is given) with retry,
    rate-limiting, and proper error handling.

    Returns raw response bytes.
    Raises urllib.error.HTTPError on non-retryable HTTP failures.
    Raises urllib.error.URLError / OSError / TimeoutError on connection issues.
    """
    _rate_limiter.wait(service)

    for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
        last = attempt == DEFAULT_MAX_RETRIES
        try:
            req = urllib.request.Request(url, data=data)
            if headers:
                for k, v in headers.items():
                    req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()

        except urllib.error.HTTPError as exc:
            status = exc.code
            retryable = status == 429 or status in RETRYABLE_STATUS
            if not retryable or last:
                raise
            wait = _backoff_wait(attempt, exc.headers)
            logger.warning(f"[{service}] HTTP {status} from {_redact_url(url)}, retrying in {wait:.1f}s "
                           f"(attempt {attempt}/{DEFAULT_MAX_RETRIES})")
            time.sleep(wait)

        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if last:
                raise
            wait = _backoff_wait(attempt, None)
            logger.warning(f"[{service}] Connection error ({exc}), retrying in {wait:.1f}s "
                           f"(attempt {attempt}/{DEFAULT_MAX_RETRIES})")
            time.sleep(wait)

    # Every attempt either returned or raised; the loop cannot fall through.
    raise RuntimeError(f"[{service}] no HTTP attempt was made for {_redact_url(url)}")


def _json_response(raw_bytes: bytes) -> dict:
    """
    Safely parse a JSON object from raw bytes.

    Returns the parsed dict.  Raises ValueError with a descriptive message on a
    malformed body, an empty body, or a body whose top level is not an object —
    every endpoint here is asked for one object, and a JSON array used to reach
    the caller as ``error="'list' object has no attribute 'get'"``: the shape
    problem was real, the sentence describing it was not.
    """
    text = raw_bytes.decode('utf-8', errors='replace')
    if not text.strip():
        raise ValueError("Empty response body (expected JSON)")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:200] if len(text) > 200 else text
        raise ValueError(f"Malformed JSON response: {exc}. Body preview: {preview!r}") from exc
    if not isinstance(parsed, dict):
        preview = text[:200] if len(text) > 200 else text
        raise ValueError(f"Expected a JSON object at the top level, got "
                         f"{type(parsed).__name__}. Body preview: {preview!r}")
    return parsed


# Characters that would move the request somewhere else when a value is placed
# into the path: a separator, a query or fragment start, or an escape sign.
_PATH_BREAKS = re.compile(r"[\s/?#%]")


def _id_segment(value: str, name: str) -> str:
    """Return *value* as exactly one URL path segment, or refuse it.

    Identifiers reach these URLs by path, not by query, so a value carrying a
    '/' or a '?' is not an odd accession — it is a different endpoint.  Measured
    before the guard: ``fetch_kegg_pathway('hsa00010/list/compound')`` asked
    ``rest.kegg.jp/get/hsa00010/list/compound`` and
    ``fetch_uniprot('P12345/../../admin')`` asked
    ``/uniprotkb/P12345/../../admin.json``.
    """
    return _checked_segment(value, name, _PATH_BREAKS)


def _query_segment(value: str, name: str) -> str:
    """Return a free-text search term as one path segment (spaces are encoded).

    KEGG's ``find`` takes the term in the path, where ``glycolysis ATP`` is a
    legitimate search — so only the characters that would leave the segment are
    refused, and the rest is percent-encoded.
    """
    return _checked_segment(value, name, re.compile(r"[/?#%]"))


def _checked_segment(value: str, name: str, breaks: "re.Pattern[str]") -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} is empty")
    found = ''.join(sorted(set(breaks.findall(text))))
    if found:
        raise ValueError(f"{name} cannot be placed in a request path: it contains "
                         f"{found!r} (got {value!r})")
    return urllib.parse.quote(text, safe='-_.:')


def _search_limit(max_results: int, name: str = "max_results") -> int:
    """Refuse a page size that cannot mean what the caller meant.

    ``max_results=0`` asked the services for zero records, and a negative value
    was handed straight to Entrez as ``retmax=-2`` (and, in the KEGG parser, used
    as a Python slice that silently dropped the *last* rows instead of the first).
    """
    if max_results < 1:
        raise ValueError(f"{name} must be at least 1, got {max_results}.")
    return max_results


# ── NCBI Entrez ─────────────────────────────────────────────────────────────

def search_ncbi(query: str, database: str = 'nucleotide', max_results: int = 10, email: Optional[str] = None) -> DBResult:
    """Search NCBI databases via Entrez API.

    Free API: https://www.ncbi.nlm.nih.gov/account/settings/
    Includes retry logic, caching, and rate limiting.  Entrez is asked for the
    contact address it requests: *email*, else the configured one, else the prompt.
    An empty or blank query is refused by name instead of being sent to Entrez.
    """
    if not HAS_URLLIB:
        return DBResult(source='ncbi', query=query, error="urllib not available")

    if not query or not query.strip():
        return DBResult(source='ncbi', query=query,
                        error="Empty query: search_ncbi needs a non-empty search term")

    try:
        max_results = _search_limit(max_results)
    except ValueError as exc:
        return DBResult(source='ncbi', query=query, error=str(exc))

    if email is None:
        email = get_api_key('ncbi_email') or prompt_api_key('ncbi_email',
                                                            NCBI_EMAIL_PROMPT)

    api_key = get_api_key('ncbi_api_key')

    # Check cache
    ck = _cache_key(query, database, max_results, email)
    cached = _cache.get('ncbi', ck)
    if cached is not None:
        return _cache_hit(cached)

    params = {
        'db': database,
        'term': query,
        'retmax': max_results,
        'retmode': 'json',
    }
    if email:
        params['email'] = email
    if api_key:
        params['api_key'] = api_key

    try:
        # Search request
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{urllib.parse.urlencode(params)}"
        raw = _http_get(url, timeout=DEFAULT_TIMEOUT, service='ncbi')
        result = _json_response(raw)
        ids = result.get('esearchresult', {}).get('idlist', [])

        if not ids:
            out = DBResult(source='ncbi', query=query, data={'count': 0})
            _cache.set('ncbi', ck, out)
            return out

        # Summary request — include api_key here too
        summary_params = {
            'db': database,
            'id': ','.join(ids),
            'retmode': 'json',
        }
        if email:
            summary_params['email'] = email
        if api_key:
            summary_params['api_key'] = api_key

        sum_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{urllib.parse.urlencode(summary_params)}"
        raw = _http_get(sum_url, timeout=DEFAULT_TIMEOUT, service='ncbi')
        summary = _json_response(raw)

        records = []
        for uid in ids:
            if uid in summary.get('result', {}):
                rec = summary['result'][uid]
                records.append({
                    'id': uid,
                    'title': rec.get('title', ''),
                    'organism': rec.get('organism', ''),
                    'accession': rec.get('accessionversion', ''),
                })

        out = DBResult(source='ncbi', query=query, records=records,
                       data={'count': len(ids)})
        _cache.set('ncbi', ck, out)
        return out

    except urllib.error.HTTPError as exc:
        return DBResult(source='ncbi', query=query,
                        error=f"HTTP {exc.code}: {exc.reason}")
    except (urllib.error.URLError, OSError) as exc:
        return DBResult(source='ncbi', query=query,
                        error=f"Connection failed: {exc}")
    except ValueError as exc:
        return DBResult(source='ncbi', query=query,
                        error=f"Invalid response: {exc}")
    except Exception as exc:
        logger.exception("[ncbi] Unexpected error")
        return DBResult(source='ncbi', query=query, error=str(exc))


def fetch_ncbi_sequence(accession: str, database: str = 'nucleotide') -> DBResult:
    """Fetch a sequence record from NCBI by accession number.

    Includes retry, rate limiting, and timeout handling.
    """
    if not HAS_URLLIB:
        return DBResult(source='ncbi', query=accession, error="urllib not available")

    email = get_api_key('ncbi_email') or prompt_api_key('ncbi_email', NCBI_EMAIL_PROMPT)
    api_key = get_api_key('ncbi_api_key')

    # Check cache
    ck = _cache_key(accession, database)
    cached = _cache.get('ncbi', ck)
    if cached is not None:
        return _cache_hit(cached)

    params = {
        'db': database,
        'id': accession,
        'rettype': 'fasta',
        'retmode': 'text',
    }
    if email:
        params['email'] = email
    if api_key:
        params['api_key'] = api_key

    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{urllib.parse.urlencode(params)}"
        raw = _http_get(url, timeout=60, service='ncbi')
        fasta = raw.decode('utf-8', errors='replace')

        if not fasta.strip():
            return DBResult(source='ncbi', query=accession,
                            error="Empty sequence response")

        out = DBResult(source='ncbi', query=accession,
                       records=[{'fasta': fasta}], data={'format': 'fasta'})
        _cache.set('ncbi', ck, out)
        return out

    except urllib.error.HTTPError as exc:
        return DBResult(source='ncbi', query=accession,
                        error=f"HTTP {exc.code}: {exc.reason}")
    except (urllib.error.URLError, OSError) as exc:
        return DBResult(source='ncbi', query=accession,
                        error=f"Connection failed: {exc}")
    except Exception as exc:
        logger.exception("[ncbi] Unexpected error in fetch_ncbi_sequence")
        return DBResult(source='ncbi', query=accession, error=str(exc))


# ── UniProt ──────────────────────────────────────────────────────────────────

def search_uniprot(query: str, max_results: int = 10) -> DBResult:
    """Search UniProt protein database.

    Free API: https://www.uniprot.org/help/api
    No API key required. Includes retry, caching, and rate limiting.
    """
    if not HAS_URLLIB:
        return DBResult(source='uniprot', query=query, error="urllib not available")

    try:
        max_results = _search_limit(max_results)
    except ValueError as exc:
        return DBResult(source='uniprot', query=query, error=str(exc))

    ck = _cache_key(query, max_results)
    cached = _cache.get('uniprot', ck)
    if cached is not None:
        return _cache_hit(cached)

    try:
        url = f"https://rest.uniprot.org/uniprotkb/search?query={urllib.parse.quote(query)}&format=json&size={max_results}"
        raw = _http_get(url, timeout=DEFAULT_TIMEOUT, service='uniprot')
        data = _json_response(raw)

        records = []
        for entry in data.get('results', []):
            genes = entry.get('genes') or []
            records.append({
                'accession': entry.get('primaryAccession', ''),
                'protein': entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
                'organism': entry.get('organism', {}).get('scientificName', ''),
                'length': entry.get('sequence', {}).get('length', 0),
                'gene': genes[0].get('geneName', {}).get('value', '') if genes else '',
            })

        out = DBResult(source='uniprot', query=query, records=records)
        _cache.set('uniprot', ck, out)
        return out

    except urllib.error.HTTPError as exc:
        return DBResult(source='uniprot', query=query,
                        error=f"HTTP {exc.code}: {exc.reason}")
    except (urllib.error.URLError, OSError) as exc:
        return DBResult(source='uniprot', query=query,
                        error=f"Connection failed: {exc}")
    except ValueError as exc:
        return DBResult(source='uniprot', query=query,
                        error=f"Invalid response: {exc}")
    except Exception as exc:
        logger.exception("[uniprot] Unexpected error")
        return DBResult(source='uniprot', query=query, error=str(exc))


def fetch_uniprot(accession: str) -> DBResult:
    """Fetch a UniProt record by accession.

    Includes retry, caching, and rate limiting.  *accession* is one path segment
    (`P04637`, `P04637-2`); a value carrying a '/' or a '?' is refused.
    """
    if not HAS_URLLIB:
        return DBResult(source='uniprot', query=accession, error="urllib not available")

    try:
        segment = _id_segment(accession, "UniProt accession")
    except ValueError as exc:
        return DBResult(source='uniprot', query=accession, error=str(exc))

    ck = _cache_key(accession)
    cached = _cache.get('uniprot', ck)
    if cached is not None:
        return _cache_hit(cached)

    try:
        url = f"https://rest.uniprot.org/uniprotkb/{segment}.json"
        raw = _http_get(url, timeout=DEFAULT_TIMEOUT, service='uniprot')
        entry = _json_response(raw)

        record = {
            'accession': entry.get('primaryAccession', ''),
            'protein': entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
            'organism': entry.get('organism', {}).get('scientificName', ''),
            'sequence': entry.get('sequence', {}).get('value', ''),
            'length': entry.get('sequence', {}).get('length', 0),
            'go_terms': [(t.get('properties') or [{}])[0].get('value', '')
                        for t in entry.get('uniProtKBCrossReferences', [])
                        if t.get('database') == 'GO'],
        }
        out = DBResult(source='uniprot', query=accession, records=[record])
        _cache.set('uniprot', ck, out)
        return out

    except urllib.error.HTTPError as exc:
        return DBResult(source='uniprot', query=accession,
                        error=f"HTTP {exc.code}: {exc.reason}")
    except (urllib.error.URLError, OSError) as exc:
        return DBResult(source='uniprot', query=accession,
                        error=f"Connection failed: {exc}")
    except ValueError as exc:
        return DBResult(source='uniprot', query=accession,
                        error=f"Invalid response: {exc}")
    except Exception as exc:
        logger.exception("[uniprot] Unexpected error")
        return DBResult(source='uniprot', query=accession, error=str(exc))


# ── PDB / RCSB ──────────────────────────────────────────────────────────────

def search_pdb(query: str, max_results: int = 10) -> DBResult:
    """Search RCSB PDB for protein structures.

    Free API: https://search.rcsb.org/
    No API key required. Includes retry, caching, and rate limiting.
    """
    if not HAS_URLLIB:
        return DBResult(source='pdb', query=query, error="urllib not available")

    try:
        max_results = _search_limit(max_results)
    except ValueError as exc:
        return DBResult(source='pdb', query=query, error=str(exc))

    ck = _cache_key(query, max_results)
    cached = _cache.get('pdb', ck)
    if cached is not None:
        return _cache_hit(cached)

    try:
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        search_body = json.dumps({
            "query": {"type": "terminal", "service": "full_text", "parameters": {"value": query}},
            "return_type": "entry",
            "request_options": {"results_content_type": ["experimental"], "pager": {"start": 0, "rows": max_results}}
        }).encode()

        # The search is a POST; _http_get drives it so the retry, the rate limit
        # and the Retry-After rule stay in one place instead of three copies.
        raw = _http_get(search_url, timeout=DEFAULT_TIMEOUT, service='pdb',
                        headers={'Content-Type': 'application/json'},
                        data=search_body)

        result = _json_response(raw)

        records = []
        for hit in result.get('result_set', []):
            pdb_id = hit.get('identifier', '')
            records.append({
                'pdb_id': pdb_id,
                'title': hit.get('title', ''),
                'url': f"https://www.rcsb.org/structure/{pdb_id}",
            })

        out = DBResult(source='pdb', query=query, records=records)
        _cache.set('pdb', ck, out)
        return out

    except urllib.error.HTTPError as exc:
        return DBResult(source='pdb', query=query,
                        error=f"HTTP {exc.code}: {exc.reason}")
    except (urllib.error.URLError, OSError) as exc:
        return DBResult(source='pdb', query=query,
                        error=f"Connection failed: {exc}")
    except ValueError as exc:
        return DBResult(source='pdb', query=query,
                        error=f"Invalid response: {exc}")
    except Exception as exc:
        logger.exception("[pdb] Unexpected error")
        return DBResult(source='pdb', query=query, error=str(exc))


# ── KEGG ─────────────────────────────────────────────────────────────────────

def search_kegg(query: str, database: str = 'pathway', max_results: int = 10) -> DBResult:
    """Search KEGG database.

    Free for academics: https://www.genome.jp/kegg/
    Includes retry, caching, and rate limiting.

    *database* names the collection (`pathway`, `compound`, `gene`, …) and is one
    path segment; *query* is the search term, spaces included, and reaches the URL
    percent-encoded.  Both are checked, and `max_results` must be at least 1.
    """
    if not HAS_URLLIB:
        return DBResult(source='kegg', query=query, error="urllib not available")

    try:
        segment = _id_segment(database, "KEGG collection")
        term = _query_segment(query, "KEGG search term")
        max_results = _search_limit(max_results)
    except ValueError as exc:
        return DBResult(source='kegg', query=query, error=str(exc))

    ck = _cache_key(query, database, max_results)
    cached = _cache.get('kegg', ck)
    if cached is not None:
        return _cache_hit(cached)

    email = get_api_key('kegg_email')
    url = f"https://rest.kegg.jp/find/{segment}/{term}"
    if email:
        url += f"?email={urllib.parse.quote(email)}"

    try:
        raw = _http_get(url, timeout=DEFAULT_TIMEOUT, service='kegg')
        text = raw.decode('utf-8', errors='replace')

        # Blank and separator lines carry no entry, so they must not spend the
        # caller's page budget: slicing first made max_results=4 return 3 rows
        # for a body that begins with an empty line.
        rows = [line for line in text.split('\n') if line.strip()]
        records = []
        for line in rows[:max_results]:
            # KEGG rows are "<db>:<entry>\t<description>" — the id contains
            # its OWN colon, so split on the tab (not the first colon).
            if '\t' in line:
                kegg_id, desc = line.split('\t', 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                kegg_id, desc = parts
            records.append({
                'id': kegg_id.strip(),
                'description': desc.strip(),
            })

        out = DBResult(source='kegg', query=query, records=records)
        _cache.set('kegg', ck, out)
        return out

    except urllib.error.HTTPError as exc:
        return DBResult(source='kegg', query=query,
                        error=f"HTTP {exc.code}: {exc.reason}")
    except (urllib.error.URLError, OSError) as exc:
        return DBResult(source='kegg', query=query,
                        error=f"Connection failed: {exc}")
    except Exception as exc:
        logger.exception("[kegg] Unexpected error")
        return DBResult(source='kegg', query=query, error=str(exc))


def fetch_kegg_pathway(pathway_id: str) -> DBResult:
    """Fetch a KEGG pathway record.

    Includes retry, caching, and rate limiting.  *pathway_id* is one path segment
    — `hsa00010` or `path:hsa00010` — so it cannot name a second segment.
    """
    if not HAS_URLLIB:
        return DBResult(source='kegg', query=pathway_id, error="urllib not available")

    try:
        segment = _id_segment(pathway_id, "KEGG pathway id")
    except ValueError as exc:
        return DBResult(source='kegg', query=pathway_id, error=str(exc))

    ck = _cache_key(pathway_id)
    cached = _cache.get('kegg', ck)
    if cached is not None:
        return _cache_hit(cached)

    email = get_api_key('kegg_email')
    url = f"https://rest.kegg.jp/get/{segment}"
    if email:
        url += f"?email={urllib.parse.quote(email)}"

    try:
        raw = _http_get(url, timeout=DEFAULT_TIMEOUT, service='kegg')
        text = raw.decode('utf-8', errors='replace')

        out = DBResult(source='kegg', query=pathway_id,
                       records=[{'text': text}], data={'format': 'kgml'})
        _cache.set('kegg', ck, out)
        return out

    except urllib.error.HTTPError as exc:
        return DBResult(source='kegg', query=pathway_id,
                        error=f"HTTP {exc.code}: {exc.reason}")
    except (urllib.error.URLError, OSError) as exc:
        return DBResult(source='kegg', query=pathway_id,
                        error=f"Connection failed: {exc}")
    except Exception as exc:
        logger.exception("[kegg] Unexpected error")
        return DBResult(source='kegg', query=pathway_id, error=str(exc))


# ── Ensembl ──────────────────────────────────────────────────────────────────

def search_ensembl(query: str, species: str = 'human') -> DBResult:
    """Look up one Ensembl gene symbol.

    Free API: https://rest.ensembl.org/
    No API key required. Includes retry, caching, and rate limiting.  Both halves
    of ``/lookup/symbol/<species>/<symbol>`` are checked as single path segments,
    so ``species='mus musculus'`` is refused instead of asking for a 404.
    """
    if not HAS_URLLIB:
        return DBResult(source='ensembl', query=query, error="urllib not available")

    try:
        species_segment = _id_segment(species, "Ensembl species")
        symbol_segment = _id_segment(query, "Ensembl gene symbol")
    except ValueError as exc:
        return DBResult(source='ensembl', query=query, error=str(exc))

    ck = _cache_key(query, species)
    cached = _cache.get('ensembl', ck)
    if cached is not None:
        return _cache_hit(cached)

    try:
        # The gene-symbol lookup is /lookup/symbol/:species/:symbol; the shorter
        # /symbol/… path this used to call answers HTTP 404 for every gene.
        url = (f"https://rest.ensembl.org/lookup/symbol/{species_segment}/{symbol_segment}"
               "?content-type=application/json")
        raw = _http_get(url, timeout=DEFAULT_TIMEOUT, service='ensembl')
        data = _json_response(raw)

        record = {
            'id': data.get('id', ''),
            'description': data.get('description', ''),
            'species': data.get('species', ''),
            'biotype': data.get('biotype', ''),
            'start': data.get('start', 0),
            'end': data.get('end', 0),
        }
        out = DBResult(source='ensembl', query=query, records=[record])
        _cache.set('ensembl', ck, out)
        return out

    except urllib.error.HTTPError as exc:
        return DBResult(source='ensembl', query=query,
                        error=f"HTTP {exc.code}: {exc.reason}")
    except (urllib.error.URLError, OSError) as exc:
        return DBResult(source='ensembl', query=query,
                        error=f"Connection failed: {exc}")
    except ValueError as exc:
        return DBResult(source='ensembl', query=query,
                        error=f"Invalid response: {exc}")
    except Exception as exc:
        logger.exception("[ensembl] Unexpected error")
        return DBResult(source='ensembl', query=query, error=str(exc))


# ── Cache management ─────────────────────────────────────────────────────────

def invalidate_cache(service: Optional[str] = None) -> None:
    """Invalidate cached results.

    Args:
        service: Specific service ('ncbi', 'uniprot', etc.) or None for all.
    """
    _cache.invalidate(service)
    logger.info(f"Cache invalidated for service={service or 'ALL'}")


def cache_info() -> Dict[str, int]:
    """Return cache statistics: number of entries per service."""
    info = {}
    with _cache._lock:
        for key in list(_cache._store.keys()):
            service = key.split(':')[0]
            info[service] = info.get(service, 0) + 1
    return info


# ── Universal Search ─────────────────────────────────────────────────────────

# Every service this module can search, and the four that a search_all() call
# uses when the caller names none.  Ensembl is left out of the default because
# its endpoint is a gene-*symbol* lookup: a free-text term such as "p53" is not a
# symbol and answers HTTP 404 there, which would put a failure line in every
# general search result.
SEARCHABLE_SERVICES = ('ncbi', 'uniprot', 'pdb', 'kegg', 'ensembl')
DEFAULT_SEARCH_DATABASES = ('ncbi', 'uniprot', 'pdb', 'kegg')


def search_all(query: str, databases: Optional[List[str]] = None) -> Dict[str, DBResult]:
    """Search multiple databases sequentially.

    Args:
        query: Search term.
        databases: List of database names.  The default is
            ``DEFAULT_SEARCH_DATABASES`` — ncbi, uniprot, pdb and kegg — not
            every service in ``SEARCHABLE_SERVICES``; name 'ensembl' to search it too.

    Returns:
        Dict mapping database name to DBResult.  Every name asked for is a key,
        including one the module does not know: it gets an error result like any
        other failure, instead of silently missing from the answer.
    """
    if databases is None:
        databases = list(DEFAULT_SEARCH_DATABASES)

    # Built here, not at import time: the mapping has to resolve the module
    # attribute on every call, or patching search_ncbi() would not reach this.
    search_funcs = {
        'ncbi': search_ncbi,
        'uniprot': search_uniprot,
        'pdb': search_pdb,
        'kegg': search_kegg,
        'ensembl': search_ensembl,
    }

    results = {}
    for service in databases:
        func = search_funcs.get(service)
        if func is None:
            logger.warning("[search_all] Unknown database: %s", service)
            results[service] = DBResult(
                source=service, query=query,
                error=f"Unknown database: {service}. "
                      f"Known: {', '.join(sorted(search_funcs))}.")
            continue
        try:
            results[service] = func(query)
        except Exception as exc:
            logger.warning("[search_all] %s failed: %s", service, exc)
            results[service] = DBResult(
                source=service,
                query=query,
                error=str(exc),
            )

    return results


def format_search_results(results: Any, source: Optional[str] = None) -> str:
    """Format search results as readable string.

    *results* may be one DBResult or a dict of them (what search_all returns).
    Handing over a single result *and* a source used to crash on the membership
    test — ``TypeError: argument of type 'DBResult' is not iterable`` — because
    the source branch was tried first whatever the object was.
    """
    if isinstance(results, DBResult):
        return _format_single(results)
    if isinstance(results, dict):
        if source and source in results:
            return _format_single(results[source])
        lines = []
        for db, result in results.items():
            lines.append(f"\n=== {db.upper()} Results ===")
            lines.append(_format_single(result))
        return '\n'.join(lines)
    return "No results"


MAX_DISPLAYED_RECORDS = 10


def _format_single(result: DBResult) -> str:
    """Render one database record for single-result display mode."""
    if result.error:
        return f"Error: {result.error}"
    if not result.records:
        return "No records found."
    lines = []
    for i, rec in enumerate(result.records[:MAX_DISPLAYED_RECORDS]):
        parts = [f"{k}: {v}" for k, v in rec.items() if v]
        lines.append(f"  {i+1}. {' | '.join(parts[:3])}")
    remaining = len(result.records) - MAX_DISPLAYED_RECORDS
    if remaining > 0:
        lines.append(f"  … {remaining} more record(s) not shown.")
    return '\n'.join(lines)
