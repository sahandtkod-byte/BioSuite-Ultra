"""Regression tests for databases.py (+ utils prompt guard) review fixes."""
import json

import pytest

from biosuite.core import databases as db
from biosuite.core.utils import prompt_api_key, config


@pytest.fixture(autouse=True)
def _no_cache():
    db.invalidate_cache()
    yield
    db.invalidate_cache()


def test_prompt_api_key_no_tty_no_block(monkeypatch):
    # Ensure the interactive input() is never attempted without a TTY —
    # the previous version died with EOFError (or hung) inside the GUI.
    monkeypatch.setattr('sys.stdin', type('S', (), {'isatty': lambda self: False})())
    monkeypatch.setattr('biosuite.core.utils.get_api_key', lambda s: None)
    assert prompt_api_key('ncbi_email') == ''


def test_kegg_find_parses_tab_rows(monkeypatch):
    payload = ("path:map00010\tGlycolysis / Gluconeogenesis\n"
               "path:map00020\tCitrate cycle (TCA cycle)\n").encode()
    monkeypatch.setattr('biosuite.core.databases.get_api_key', lambda s: None)
    monkeypatch.setattr('biosuite.core.databases._http_get', lambda url, **kw: payload)
    res = db.search_kegg("glycolysis")
    assert res.error == ''
    assert res.records[0] == {'id': 'path:map00010', 'description': 'Glycolysis / Gluconeogenesis'}
    assert res.records[1]['id'] == 'path:map00020'


def test_kegg_find_handles_whitespace_fallback(monkeypatch):
    payload = b"path:map00010 Glycolysis\n"
    monkeypatch.setattr('biosuite.core.databases.get_api_key', lambda s: None)
    monkeypatch.setattr('biosuite.core.databases._http_get', lambda url, **kw: payload)
    res = db.search_kegg("glycolysis")
    assert res.records == [{'id': 'path:map00010', 'description': 'Glycolysis'}]


def test_uniprot_fetch_go_terms_empty_properties(monkeypatch):
    entry = {
        "primaryAccession": "P12345",
        "uniProtKBCrossReferences": [
            {"database": "GO", "properties": []},
            {"database": "GO", "properties": [{"value": "GO:0008150"}]},
        ],
        "sequence": {"value": "MKT", "length": 3},
    }
    monkeypatch.setattr('biosuite.core.databases._http_get',
                        lambda url, **kw: json.dumps(entry).encode())
    res = db.fetch_uniprot("P12345")
    assert res.error == ''
    assert res.records[0]['go_terms'] == ['', 'GO:0008150']


def test_ncbi_search_empty_idlist_no_summary_call(monkeypatch):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return json.dumps({"esearchresult": {"idlist": []}}).encode()

    monkeypatch.setattr('biosuite.core.databases.get_api_key', lambda s: None)
    monkeypatch.setattr('biosuite.core.databases._http_get', fake_get)
    res = db.search_ncbi("nothingmatchesxyz", email="a@b.c")
    assert res.data == {'count': 0} and len(calls) == 1


def test_ncbi_search_parses_records(monkeypatch):
    def fake_get(url, **kw):
        if 'esearch' in url:
            return json.dumps({"esearchresult": {"idlist": ["1", "2"]}}).encode()
        return json.dumps({"result": {
            "1": {"title": "T1", "organism": "E. coli", "accessionversion": "NC_1"},
            "2": {"title": "T2", "organism": "S. cerevisiae", "accessionversion": "NC_2"},
        }}).encode()

    monkeypatch.setattr('biosuite.core.databases.get_api_key', lambda s: None)
    monkeypatch.setattr('biosuite.core.databases._http_get', fake_get)
    res = db.search_ncbi("recA", email="a@b.c")
    assert len(res.records) == 2 and res.records[0]['accession'] == 'NC_1'


def test_search_all_collects_errors_not_raises(monkeypatch):
    monkeypatch.setattr(db, 'search_uniprot', lambda q: (_ for _ in ()).throw(RuntimeError('boom')))
    out = db.search_all("x", databases=['uniprot'])
    assert 'boom' in out['uniprot'].error


def test_cache_hit_skips_http(monkeypatch):
    called = {'n': 0}
    monkeypatch.setattr('biosuite.core.databases._http_get', lambda url, **kw: called.__setitem__('n', called['n'] + 1) or b"{}")
    monkeypatch.setattr('biosuite.core.databases.get_api_key', lambda s: None)
    db.search_ensembl("BRCA1")
    db.search_ensembl("BRCA1")
    assert called['n'] == 1
