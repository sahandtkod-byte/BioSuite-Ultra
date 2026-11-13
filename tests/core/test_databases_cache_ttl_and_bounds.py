"""The TTL cache: what it keeps, for how long, and how much of it.

An expired entry was only removed when the *same key* was asked for again, so a
long GUI session that kept searching new terms retained every response it had
ever fetched — records and all — with no bound and nothing to shed them.  The
cache is now bounded, each store first drops what has expired, and only then the
oldest entries.
"""
import pytest

from biosuite.core import databases as db


class _Step:
    """A clock the test walks forward; sleeps advance it instead of blocking."""

    def __init__(self, start=1000.0):
        self.now = start
        self.sleeps = []

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    step = _Step()
    monkeypatch.setattr(db, 'time', step)
    return step


@pytest.fixture(autouse=True)
def clean_cache():
    db.invalidate_cache()
    yield
    db.invalidate_cache()


def test_cache_info_counts_entries_per_service(monkeypatch):
    monkeypatch.setattr(db, 'get_api_key', lambda name: '')
    monkeypatch.setattr(db, 'prompt_api_key', lambda *a, **k: '')
    monkeypatch.setattr(db, '_http_get', lambda url, **kw: b'{"esearchresult": {"idlist": []}}')
    db.search_ncbi('q1')
    db.search_ncbi('q2')

    monkeypatch.setattr(db, '_http_get', lambda url, **kw: b'{"results": []}')
    db.search_uniprot('q1')

    assert db.cache_info() == {'ncbi': 2, 'uniprot': 1}


def test_invalidate_cache_drops_only_the_named_service(monkeypatch):
    monkeypatch.setattr(db, 'get_api_key', lambda name: '')
    monkeypatch.setattr(db, '_http_get', lambda url, **kw: b'{"esearchresult": {"idlist": []}}')
    db.search_ncbi('q1')
    monkeypatch.setattr(db, '_http_get', lambda url, **kw: b'{"results": []}')
    db.search_uniprot('q1')

    db.invalidate_cache('ncbi')

    assert db.cache_info() == {'uniprot': 1}
    db.invalidate_cache()
    assert db.cache_info() == {}


def test_a_second_identical_search_is_served_from_cache_without_a_second_request(monkeypatch):
    """The hit is the point of the cache, and it has to be visible on the result."""
    asked = []

    def capture(url, **kwargs):
        asked.append(url)
        return b'{"id": "ENSG00000141510", "species": "homo_sapiens"}'

    db.invalidate_cache()
    monkeypatch.setattr(db, 'get_api_key', lambda name: '')
    monkeypatch.setattr(db, '_http_get', capture)

    first = db.search_ensembl("BRCA1")
    second = db.search_ensembl("BRCA1")

    assert len(asked) == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.records == first.records
    assert db.cache_info() == {'ensembl': 1}


def test_a_service_outside_the_ttl_table_is_cached_for_the_default_period(clock, monkeypatch):
    monkeypatch.setattr(db, '_CACHE_TTL', {'ncbi': 600})
    cache = db._Cache()
    cache.set('orbit', 'k', 'value')

    clock.now += db._DEFAULT_TTL - 1
    assert cache.get('orbit', 'k') == 'value'
    clock.now += 2
    assert cache.get('orbit', 'k') is None


def test_an_expired_entry_is_dropped_when_the_key_is_asked_again(clock):
    cache = db._Cache()
    cache.set('ncbi', 'k', 'value')

    clock.now += 601                      # past ncbi's ten-minute TTL
    assert cache.get('ncbi', 'k') is None
    assert cache._store == {}


def test_filling_the_cache_prunes_the_expired_before_evicting_the_live(clock, monkeypatch):
    monkeypatch.setattr(db, 'MAX_CACHE_ENTRIES', 3)
    monkeypatch.setattr(db, '_CACHE_TTL', {'ncbi': 10})
    cache = db._Cache()

    cache.set('ncbi', 'a', 1)             # a: age 12 when d arrives -> expired
    clock.now += 4
    cache.set('ncbi', 'b', 2)             # b: age 8  -> still live
    clock.now += 4
    cache.set('ncbi', 'c', 3)             # c: age 4  -> still live
    clock.now += 4
    cache.set('ncbi', 'd', 4)             # the store would hold four

    assert len(cache._store) == 3
    assert cache.get('ncbi', 'a') is None
    assert (cache.get('ncbi', 'b'), cache.get('ncbi', 'c'), cache.get('ncbi', 'd')) == (2, 3, 4)


def test_when_nothing_has_expired_the_oldest_entry_is_evicted(clock, monkeypatch):
    monkeypatch.setattr(db, 'MAX_CACHE_ENTRIES', 3)
    monkeypatch.setattr(db, '_CACHE_TTL', {'ncbi': 10_000})
    cache = db._Cache()

    for i in range(4):
        clock.now += 1
        cache.set('ncbi', f"k{i}", i)

    assert len(cache._store) == 3
    assert cache.get('ncbi', 'k0') is None            # the oldest went first
    assert [cache.get('ncbi', f"k{i}") for i in (1, 2, 3)] == [1, 2, 3]


def test_a_long_session_cannot_grow_the_cache_without_bound(clock, monkeypatch):
    monkeypatch.setattr(db, 'MAX_CACHE_ENTRIES', 8)
    monkeypatch.setattr(db, '_CACHE_TTL', {'ncbi': 10_000})
    cache = db._Cache()

    for i in range(500):
        cache.set('ncbi', f"query-{i}", i)

    assert len(cache._store) == 8
    assert cache.get('ncbi', 'query-499') == 499
    assert cache.get('ncbi', 'query-0') is None
    assert db.cache_info() == {}                        # the module cache is untouched
