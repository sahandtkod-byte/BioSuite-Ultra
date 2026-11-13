"""How each service's body becomes records.

The rows a user sees are made here: KEGG's tab-separated lines (and the
space-separated fallback some collections answer with), UniProt's cross-references
whose `properties` list can be empty, and Entrez's two-call search/summary pair
whose second request must not be made when the first found nothing.
"""
import json

import pytest

from biosuite.core import databases as db


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch):
    db.invalidate_cache()
    monkeypatch.setattr(db, 'get_api_key', lambda name: '')
    monkeypatch.setattr(db, 'prompt_api_key', lambda *a, **k: '')
    yield
    db.invalidate_cache()


def serve(monkeypatch, payload):
    """Answer every request with `payload`; return the URL list it collected."""
    asked = []

    def fake(url, **kwargs):
        asked.append(url)
        return payload(url) if callable(payload) else payload

    monkeypatch.setattr(db, '_http_get', fake)
    return asked


def test_kegg_rows_are_split_on_the_tab_because_the_id_contains_a_colon(monkeypatch):
    asked = serve(monkeypatch,
                  ("path:map00010\tGlycolysis / Gluconeogenesis\n"
                   "path:map00020\tCitrate cycle (TCA cycle)\n").encode())

    res = db.search_kegg('glycolysis')

    assert asked == ['https://rest.kegg.jp/find/pathway/glycolysis']
    assert res.error == ''
    assert res.records == [
        {'id': 'path:map00010', 'description': 'Glycolysis / Gluconeogenesis'},
        {'id': 'path:map00020', 'description': 'Citrate cycle (TCA cycle)'}]


def test_a_kegg_row_without_a_tab_falls_back_to_the_first_space(monkeypatch):
    serve(monkeypatch, b"path:map00010 Glycolysis\n")

    assert db.search_kegg('glycolysis').records == [
        {'id': 'path:map00010', 'description': 'Glycolysis'}]


@pytest.mark.parametrize("body", [b"", b"   \n", b"\n\n"])
def test_an_empty_kegg_answer_is_no_records_and_no_error(monkeypatch, body):
    serve(monkeypatch, body)

    res = db.search_kegg('nothingmatches')

    assert res.records == [] and res.error == ''


def test_a_kegg_row_that_is_only_an_id_is_dropped_rather_than_half_parsed(monkeypatch):
    serve(monkeypatch, b"path:map00010\npath:map00020\tCitrate cycle\n")

    assert db.search_kegg('glycolysis').records == [
        {'id': 'path:map00020', 'description': 'Citrate cycle'}]


def test_uniprot_go_terms_keep_their_place_when_properties_are_empty(monkeypatch):
    serve(monkeypatch, json.dumps({
        "primaryAccession": "P12345",
        "uniProtKBCrossReferences": [
            {"database": "GO", "properties": []},
            {"database": "GO", "properties": [{"value": "GO:0008150"}]},
            {"database": "Pfam", "properties": [{"value": "P53"}]},
        ],
        "sequence": {"value": "MKT", "length": 3},
    }).encode())

    res = db.fetch_uniprot("P12345")

    assert res.error == ''
    assert res.records[0]['go_terms'] == ['', 'GO:0008150']     # Pfam is not a GO term
    assert res.records[0]['accession'] == 'P12345'
    assert res.records[0]['length'] == 3


def test_an_entrez_search_that_found_no_id_never_asks_for_summaries(monkeypatch):
    asked = serve(monkeypatch, json.dumps({"esearchresult": {"idlist": []}}).encode())

    res = db.search_ncbi("nothingmatchesxyz")

    assert len(asked) == 1
    assert 'esummary' not in asked[0]
    assert res.data == {'count': 0} and res.records == []


def test_entrez_records_are_built_from_the_summary_of_each_id(monkeypatch):
    def payload(url):
        if 'esearch' in url:
            return json.dumps({"esearchresult": {"idlist": ["1", "2", "3"]}}).encode()
        return json.dumps({"result": {
            "1": {"title": "T1", "organism": "E. coli", "accessionversion": "NC_1"},
            "2": {"uids": "2", "version": "2"},              # a hit with no title/organism
            "3": {"title": "T3", "organism": "S. cerevisiae", "accessionversion": "NC_3"},
        }}).encode()

    asked = serve(monkeypatch, payload)

    res = db.search_ncbi("recA")

    assert len(asked) == 2 and 'esearch' in asked[0] and 'esummary' in asked[1]
    assert res.data == {'count': 3}
    assert res.records == [
        {'id': '1', 'title': 'T1', 'organism': 'E. coli', 'accession': 'NC_1'},
        {'id': '2', 'title': '', 'organism': '', 'accession': ''},
        {'id': '3', 'title': 'T3', 'organism': 'S. cerevisiae', 'accession': 'NC_3'}]


def test_an_id_that_the_summary_did_not_return_is_skipped(monkeypatch):
    def payload(url):
        if 'esearch' in url:
            return json.dumps({"esearchresult": {"idlist": ["7", "8"]}}).encode()
        return json.dumps({"result": {"8": {"title": "only the second"}}}).encode()

    serve(monkeypatch, payload)

    res = db.search_ncbi("gap")

    assert res.data == {'count': 2}                  # two ids were found...
    assert res.records == [{'id': '8', 'title': 'only the second',
                            'organism': '', 'accession': ''}]     # ...one had no summary
