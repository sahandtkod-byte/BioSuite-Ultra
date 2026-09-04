"""Offline-auditable tests for core/databases.py (cache, formats, error paths)."""
import pytest

from biosuite.core import databases as db


def test_cache_key_stability():
    k = db._cache_key("a", "b", 1)
    assert isinstance(k, str) and len(k) > 4
    assert db._cache_key("a", "b", 1) == k
    assert db._cache_key("a", "b", 2) != k


def test_cache_info_and_invalidate():
    assert isinstance(db.cache_info(), dict)
    db.invalidate_cache("pytest_scoped")      # scoped invalidate must not raise
    db.invalidate_cache()                      # full invalidate must not raise


def test_json_response_parsing():
    assert db._json_response(b'{"ok": true}') == {"ok": True}
    with pytest.raises(Exception):
        db._json_response(b'not json!!!')


def test_dbresult_fields():
    r = db.DBResult(source='ncbi', query='tp53', data=[], records=[], error=None, cached=False)
    assert r.source == 'ncbi' and r.records == []


def test_format_search_results_dict():
    rec = {'id': 'X1', 'title': 'demo title', 'description': 'demo field'}
    r = db.DBResult(source='ncbi', query='q', data=[], records=[rec],
                    error=None, cached=False)
    text = db.format_search_results(r)
    assert isinstance(text, str) and 'demo' in text


def test_fetch_uniprot_http_failure(monkeypatch):
    monkeypatch.setattr(db, '_http_get', lambda *a, **k: None)
    res = db.fetch_uniprot('BADACC')
    assert isinstance(res, db.DBResult)
    assert (res.error is not None) or (res.records == [])


def test_search_all_with_stubs(monkeypatch):
    def fake(query, *a, **k):
        return db.DBResult(source='fake', query=query, data=[],
                           records=[], error=None, cached=False)
    for name in ('search_ncbi', 'search_uniprot', 'search_pdb',
                 'search_kegg', 'search_ensembl'):
        monkeypatch.setattr(db, name, fake)
    out = db.search_all('tp53', databases=['ncbi', 'uniprot'])
    assert set(out.keys()) == {'ncbi', 'uniprot'}


def test_search_all_tolerates_failures(monkeypatch):
    def boom(query, *a, **k):
        raise ConnectionError("offline")
    for name in ('search_ncbi', 'search_uniprot', 'search_pdb',
                 'search_kegg', 'search_ensembl'):
        monkeypatch.setattr(db, name, boom)
    out = db.search_all('nothing', databases=['ncbi'])
    assert 'ncbi' in out  # failure folded into DBResult, not an exception
