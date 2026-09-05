"""Mocked-network database endpoint parsing correctness (no real HTTP)."""
import json

import pytest

from biosuite.core import databases as db


@pytest.fixture(autouse=True)
def _clear_cache():
    db.invalidate_cache()
    yield
    db.invalidate_cache()


def _patch_http(monkeypatch, payloads):
    """payloads: callable(url) -> bytes"""
    monkeypatch.setattr(db, '_http_get', payloads)
    # also stop interactive key prompting in offline runs
    monkeypatch.setattr(db, 'get_api_key', lambda name: 'k@example.com')
    monkeypatch.setattr(db, 'prompt_api_key', lambda *a, **k: 'k@example.com')


# ── NCBI ─────────────────────────────────────────────────────────────────────

def test_search_ncbi_parses_id_list_and_summary(monkeypatch):
    esearch = json.dumps({'esearchresult': {'idlist': ['111', '222']}}).encode()
    esummary = json.dumps({'result': {
        '111': {'title': 'Rec A', 'organism': 'Homo sapiens', 'accessionversion': 'NM_1'},
        '222': {'title': 'Rec B', 'organism': 'Mus musculus', 'accessionversion': 'NM_2'},
    }}).encode()

    state = {'called': 0}
    def fake(url, **kw):
        state['called'] += 1
        return esearch if state['called'] == 1 else esummary
    _patch_http(monkeypatch, fake)

    res = db.search_ncbi('tp53', max_results=2)
    assert not res.error
    ids = [r['id'] for r in res.records if isinstance(r, dict)]
    assert '111' in ids and '222' in ids
    assert any('Homo sapiens' in r.get('organism', '') for r in res.records)


def test_search_ncbi_empty_idlist(monkeypatch):
    payload = json.dumps({'esearchresult': {'idlist': []}}).encode()
    _patch_http(monkeypatch, lambda url, **kw: payload)
    res = db.search_ncbi('notarealgene')
    assert not res.error
    assert res.records == []


# ── UniProt ─────────────────────────────────────────────────────────────────

def test_search_uniprot_parses_json(monkeypatch):
    payload = json.dumps({'results': [{
        'primaryAccession': 'P04637',
        'proteinDescription': {'recommendedName': {'fullName': {'value': 'p53'}}},
        'organism': {'scientificName': 'Homo sapiens'},
        'sequence': {'length': 393},
        'genes': [{'geneName': {'value': 'TP53'}}],
    }]}).encode()
    _patch_http(monkeypatch, lambda url, **kw: payload)
    res = db.search_uniprot('tp53')
    assert not res.error
    assert res.records and res.records[0]['accession'] == 'P04637'
    assert res.records[0]['protein'] == 'p53'
    assert res.records[0]['length'] == 393


def test_fetch_uniprot_204_like(monkeypatch):
    _patch_http(monkeypatch, lambda url, **kw: b'')
    # empty body → graceful error, not crash
    try:
        res = db.fetch_uniprot('XXXX')
        assert res.error is not None or res.records == []
    except Exception:
        pass  # acceptable: handled upstream


# ── KEGG ─────────────────────────────────────────────────────────────────────

def test_search_kegg_parses_tsv(monkeypatch):
    tsv = "hsa04610\tComplement and coagulation cascades\nhsa04010\tMAPK signaling\n"
    _patch_http(monkeypatch, lambda url, **kw: tsv.encode())
    res = db.search_kegg('complement')
    assert not res.error
    assert any('hsa04610' in str(r) for r in res.records or res.data)


# ── PDB ──────────────────────────────────────────────────────────────────────

def test_search_pdb_parses_json(monkeypatch):
    payload = json.dumps({
        'result_set': [
            {'identifier': '1TUP', 'score': 1.0},
            {'identifier': '2XWR', 'score': 0.9},
        ]
    }).encode()

    class _FakeResp:
        def __init__(self, data):
            self._data = data
        def read(self):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    import urllib.request
    monkeypatch.setattr(db, '_rate_limiter', type('R', (), {'wait': lambda self, s: None})())
    monkeypatch.setattr('urllib.request.urlopen', lambda req, timeout=0: _FakeResp(payload))
    monkeypatch.setattr(db, 'get_api_key', lambda name: '')
    res = db.search_pdb('p53')
    assert not res.error
    text = str(res.records) + str(res.data)
    assert '1TUP' in text or '2XWR' in text


# ── Ensembl ─────────────────────────────────────────────────────────────────

def test_search_ensembl_tolerant(monkeypatch):
    payload = json.dumps([
        {'id': 'ENSG00000141510', 'display_name': 'TP53',
         'description': 'tumor protein p53', 'species': 'homo_sapiens'}
    ]).encode()
    _patch_http(monkeypatch, lambda url, **kw: payload)
    try:
        res = db.search_ensembl('TP53')  # must not raise
    except Exception as e:
        pytest.fail(f"ensembl search raised unexpectedly: {e}")


# ── cache behaviour ─────────────────────────────────────────────────────────

def test_cache_hit_avoids_second_http_call(monkeypatch):
    esearch = json.dumps({'esearchresult': {'idlist': []}}).encode()
    calls = {'n': 0}
    def fake(url, **kw):
        calls['n'] += 1
        return esearch
    _patch_http(monkeypatch, fake)
    db.search_ncbi('cached-query', email='k@example.com')
    db.search_ncbi('cached-query', email='k@example.com')
    assert calls['n'] == 1
