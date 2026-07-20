"""
Database API integrations: NCBI Entrez, UniProt, Ensembl, PDB, KEGG.

All APIs are free for academic use. Users can optionally provide API keys
for higher rate limits via the config system.

Features:
  - Retry logic with exponential backoff for transient failures
  - In-memory TTL cache to avoid redundant API calls
  - Per-service rate limiting
  - Proper HTTP error handling (4xx/5xx/status codes)
  - Malformed JSON safe-parsing
  - Configurable timeouts
"""
import os
import json
import time
import hashlib
import logging
import tempfile
import threading
from dataclasses import dataclass, field
from functools import wraps

try:
    import urllib.request
    import urllib.parse
    import urllib.error
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

from .utils import get_api_key, set_api_key, prompt_api_key


@dataclass
class DBResult:
    source: str
    query: str
    data: dict = field(default_factory=dict)
    records: list = field(default_factory=list)
    error: str = ""
    cached: bool = False


logger = logging.getLogger(__name__)

# ── Retry / timeout / rate-limit config ─────────────────────────────────────

DEFAULT_TIMEOUT = 30          # seconds per HTTP request
DEFAULT_MAX_RETRIES = 3       # max retry attempts for transient errors
DEFAULT_RETRY_BACKOFF = 1.5   # exponential backoff base (seconds)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Per-service rate limits: (min_interval_seconds, label)
RATE_LIMITS = {
    'ncbi':    (0.34, 'NCBI'),   # ≤ 3 req/s without key, 10 with key
    'uniprot': (1.0, 'UniProt'),
    'pdb':     (0.5, 'PDB'),
    'kegg':    (1.0, 'KEGG'),
    'ensembl': (0.2, 'Ensembl'),
}

# Cache: {key: (timestamp, value)} with per-service TTLs
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
        self._locks = {}
        self._last_call = {}
        self._global_lock = threading.Lock()

    def wait(self, service: str) -> None:
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
        self._store = {}
        self._lock = threading.Lock()

    def get(self, service: str, key: str):
        ttl = _CACHE_TTL.get(service, 300)
        full_key = f"{service}:{key}"
        with self._lock:
            entry = self._store.get(full_key)
            if entry and (time.time() - entry[0]) < ttl:
                return entry[1]
            # expired or missing
            self._store.pop(full_key, None)
        return None

    def set(self, service: str, key: str, value) -> None:
        full_key = f"{service}:{key}"
        with self._lock:
            self._store[full_key] = (time.time(), value)

    def invalidate(self, service: str = None) -> None:
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


def _http_get(url: str, *, timeout: int = DEFAULT_TIMEOUT,
              headers: dict = None, service: str = 'general') -> bytes:
    """
    Perform an HTTP GET with retry, rate-limiting, and proper error handling.

    Returns raw response bytes.
    Raises urllib.error.HTTPError on non-retryable HTTP failures.
    Raises urllib.error.URLError / OSError / TimeoutError on connection issues.
    """
    _rate_limiter.wait(service)

    last_exc = None
    for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url)
            if headers:
                for k, v in headers.items():
                    req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()

        except urllib.error.HTTPError as exc:
            last_exc = exc
            status = exc.code
            if status == 429:
                # Rate limited — honour Retry-After if present, else backoff
                retry_after = exc.headers.get('Retry-After')
                if retry_after:
                    wait = float(retry_after)
                else:
                    wait = DEFAULT_RETRY_BACKOFF ** attempt
                logger.warning(f"[{service}] HTTP 429 from {url}, retrying in {wait:.1f}s (attempt {attempt}/{DEFAULT_MAX_RETRIES})")
                time.sleep(wait)
                continue
            if status in RETRYABLE_STATUS and attempt < DEFAULT_MAX_RETRIES:
                wait = DEFAULT_RETRY_BACKOFF ** attempt
                logger.warning(f"[{service}] HTTP {status} from {url}, retrying in {wait:.1f}s (attempt {attempt}/{DEFAULT_MAX_RETRIES})")
                time.sleep(wait)
                continue
            # Non-retryable (4xx other than 429, or last attempt)
            raise

        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_exc = exc
            if attempt < DEFAULT_MAX_RETRIES:
                wait = DEFAULT_RETRY_BACKOFF ** attempt
                logger.warning(f"[{service}] Connection error ({exc}), retrying in {wait:.1f}s (attempt {attempt}/{DEFAULT_MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise

    # Should not reach here, but just in case
    raise last_exc


def _json_response(raw_bytes: bytes) -> dict:
    """
    Safely parse JSON from raw bytes.
    Returns parsed dict on success.
    Raises ValueError with a descriptive message on malformed JSON.
    """
    text = raw_bytes.decode('utf-8', errors='replace')
    if not text.strip():
        raise ValueError("Empty response body (expected JSON)")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:200] if len(text) > 200 else text
        raise ValueError(f"Malformed JSON response: {exc}. Body preview: {preview!r}") from exc


# ── NCBI Entrez ─────────────────────────────────────────────────────────────

def search_ncbi(query, database='nucleotide', max_results=10, email=None):
    """Search NCBI databases via Entrez API.

    Free API: https://www.ncbi.nlm.nih.gov/account/settings/
    Includes retry logic, caching, and rate limiting.
    """
    if not HAS_URLLIB:
        return DBResult(source='ncbi', query=query, error="urllib not available")

    if email is None:
        email = get_api_key('ncbi_email') or prompt_api_key('ncbi_email',
            "Free registration at: https://www.ncbi.nlm.nih.gov/account/")

    api_key = get_api_key('ncbi_api_key')

    # Check cache
    ck = _cache_key(query, database, max_results, email)
    cached = _cache.get('ncbi', ck)
    if cached is not None:
        return cached

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


def fetch_ncbi_sequence(accession, database='nucleotide'):
    """Fetch a sequence record from NCBI by accession number.

    Includes retry, rate limiting, and timeout handling.
    """
    if not HAS_URLLIB:
        return DBResult(source='ncbi', query=accession, error="urllib not available")

    email = get_api_key('ncbi_email') or 'biosuite@example.com'
    api_key = get_api_key('ncbi_api_key')

    # Check cache
    ck = _cache_key(accession, database)
    cached = _cache.get('ncbi', ck)
    if cached is not None:
        return cached

    params = {
        'db': database,
        'id': accession,
        'rettype': 'fasta',
        'retmode': 'text',
        'email': email,
    }
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

def search_uniprot(query, max_results=10):
    """Search UniProt protein database.

    Free API: https://www.uniprot.org/help/api
    No API key required. Includes retry, caching, and rate limiting.
    """
    if not HAS_URLLIB:
        return DBResult(source='uniprot', query=query, error="urllib not available")

    ck = _cache_key(query, max_results)
    cached = _cache.get('uniprot', ck)
    if cached is not None:
        return cached

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


def fetch_uniprot(accession):
    """Fetch a UniProt record by accession.

    Includes retry, caching, and rate limiting.
    """
    if not HAS_URLLIB:
        return DBResult(source='uniprot', query=accession, error="urllib not available")

    ck = _cache_key(accession)
    cached = _cache.get('uniprot', ck)
    if cached is not None:
        return cached

    try:
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        raw = _http_get(url, timeout=DEFAULT_TIMEOUT, service='uniprot')
        entry = _json_response(raw)

        record = {
            'accession': entry.get('primaryAccession', ''),
            'protein': entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', ''),
            'organism': entry.get('organism', {}).get('scientificName', ''),
            'sequence': entry.get('sequence', {}).get('value', ''),
            'length': entry.get('sequence', {}).get('length', 0),
            'go_terms': [t.get('properties', [{}])[0].get('value', '')
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

def search_pdb(query, max_results=10):
    """Search RCSB PDB for protein structures.

    Free API: https://search.rcsb.org/
    No API key required. Includes retry, caching, and rate limiting.
    """
    if not HAS_URLLIB:
        return DBResult(source='pdb', query=query, error="urllib not available")

    ck = _cache_key(query, max_results)
    cached = _cache.get('pdb', ck)
    if cached is not None:
        return cached

    try:
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        search_body = json.dumps({
            "query": {"type": "terminal", "service": "full_text", "parameters": {"value": query}},
            "return_type": "entry",
            "request_options": {"results_content_type": ["experimental"], "pager": {"start": 0, "rows": max_results}}
        }).encode()

        # _http_get uses GET; for POST we need a custom path
        _rate_limiter.wait('pdb')
        last_exc = None
        raw = None
        for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(search_url, data=search_body,
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
                    raw = resp.read()
                break
            except urllib.error.HTTPError as exc:
                last_exc = exc
                status = exc.code
                if status == 429 or (status in RETRYABLE_STATUS and attempt < DEFAULT_MAX_RETRIES):
                    retry_after = exc.headers.get('Retry-After')
                    wait = float(retry_after) if retry_after else DEFAULT_RETRY_BACKOFF ** attempt
                    logger.warning(f"[pdb] HTTP {status}, retrying in {wait:.1f}s (attempt {attempt}/{DEFAULT_MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                raise
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_exc = exc
                if attempt < DEFAULT_MAX_RETRIES:
                    wait = DEFAULT_RETRY_BACKOFF ** attempt
                    logger.warning(f"[pdb] Connection error ({exc}), retrying in {wait:.1f}s (attempt {attempt}/{DEFAULT_MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                raise

        if raw is None:
            raise last_exc

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

def search_kegg(query, database='pathway', max_results=10):
    """Search KEGG database.

    Free for academics: https://www.genome.jp/kegg/
    Includes retry, caching, and rate limiting.
    """
    if not HAS_URLLIB:
        return DBResult(source='kegg', query=query, error="urllib not available")

    ck = _cache_key(query, database, max_results)
    cached = _cache.get('kegg', ck)
    if cached is not None:
        return cached

    email = get_api_key('kegg_email')
    url = f"https://rest.kegg.jp/find/{database}/{urllib.parse.quote(query)}"
    if email:
        url += f"?email={urllib.parse.quote(email)}"

    try:
        raw = _http_get(url, timeout=DEFAULT_TIMEOUT, service='kegg')
        text = raw.decode('utf-8', errors='replace')

        records = []
        for line in text.strip().split('\n')[:max_results]:
            if ':' in line:
                kegg_id, desc = line.split(':', 1)
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


def fetch_kegg_pathway(pathway_id):
    """Fetch a KEGG pathway record.

    Includes retry, caching, and rate limiting.
    """
    if not HAS_URLLIB:
        return DBResult(source='kegg', query=pathway_id, error="urllib not available")

    ck = _cache_key(pathway_id)
    cached = _cache.get('kegg', ck)
    if cached is not None:
        return cached

    email = get_api_key('kegg_email')
    url = f"https://rest.kegg.jp/get/{pathway_id}"
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

def search_ensembl(query, species='human'):
    """Search Ensembl REST API.

    Free API: https://rest.ensembl.org/
    No API key required. Includes retry, caching, and rate limiting.
    """
    if not HAS_URLLIB:
        return DBResult(source='ensembl', query=query, error="urllib not available")

    ck = _cache_key(query, species)
    cached = _cache.get('ensembl', ck)
    if cached is not None:
        return cached

    try:
        url = f"https://rest.ensembl.org/symbol/{species}/{query}?content-type=application/json"
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

def invalidate_cache(service: str = None) -> None:
    """Invalidate cached results.

    Args:
        service: Specific service ('ncbi', 'uniprot', etc.) or None for all.
    """
    _cache.invalidate(service)
    logger.info(f"Cache invalidated for service={service or 'ALL'}")


def cache_info() -> dict:
    """Return cache statistics: number of entries per service."""
    info = {}
    with _cache._lock:
        for key in list(_cache._store.keys()):
            service = key.split(':')[0]
            info[service] = info.get(service, 0) + 1
    return info


# ── Universal Search ─────────────────────────────────────────────────────────

def search_all(query, databases=None):
    """Search multiple databases simultaneously.

    Args:
        query: Search term.
        databases: List of database names (default: all).

    Returns:
        Dict mapping database name to DBResult.
    """
    if databases is None:
        databases = ['ncbi', 'uniprot', 'pdb', 'kegg']

    results = {}
    for db in databases:
        if db == 'ncbi':
            results['ncbi'] = search_ncbi(query)
        elif db == 'uniprot':
            results['uniprot'] = search_uniprot(query)
        elif db == 'pdb':
            results['pdb'] = search_pdb(query)
        elif db == 'kegg':
            results['kegg'] = search_kegg(query)
        elif db == 'ensembl':
            results['ensembl'] = search_ensembl(query)

    return results


def format_search_results(results, source=None):
    """Format search results as readable string."""
    if source and source in results:
        result = results[source]
        return _format_single(result)
    elif isinstance(results, dict):
        lines = []
        for db, result in results.items():
            lines.append(f"\n=== {db.upper()} Results ===")
            lines.append(_format_single(result))
        return '\n'.join(lines)
    elif isinstance(results, DBResult):
        return _format_single(results)
    return "No results"


def _format_single(result):
    if result.error:
        return f"Error: {result.error}"
    if not result.records:
        return "No records found."
    lines = []
    for i, rec in enumerate(result.records[:10]):
        parts = [f"{k}: {v}" for k, v in rec.items() if v]
        lines.append(f"  {i+1}. {' | '.join(parts[:3])}")
    return '\n'.join(lines)
