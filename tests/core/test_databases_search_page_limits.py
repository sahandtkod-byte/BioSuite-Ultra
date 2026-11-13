"""How many records a search asks for, and how many it returns.

`max_results` reached the services unchecked: Entrez got `retmax=-2`, UniProt got
`size=-2`, and the KEGG parser used the number as a Python slice — so
`max_results=-2` answered with the first two of four pathways while *silently
dropping the last two*, the opposite of what a smaller page means.  KEGG also
spent the budget before skipping blank rows, so a body that began with an empty
line returned three records for a page of four.
"""
import json

import pytest

from biosuite.core import databases as db

FOUR_PATHWAYS = ("\n"
                 "path:hsa00010\tGlycolysis / Gluconeogenesis\n"
                 "\npath:hsa00020\tCitrate cycle\n"
                 "path:hsa00030\tPentose phosphate pathway\n"
                 "path:hsa00040\tPentose and glucuronate interconversions\n")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    asked = []
    db.invalidate_cache()
    monkeypatch.setattr(db, 'get_api_key', lambda name: '')
    monkeypatch.setattr(db, 'prompt_api_key', lambda *a, **k: '')

    def capture(url, **kwargs):
        asked.append((url, kwargs))
        if 'kegg' in url:
            return FOUR_PATHWAYS.encode()
        if 'esearch' in url:
            return json.dumps({'esearchresult': {'idlist': []}}).encode()
        return json.dumps({'results': [], 'result_set': []}).encode()

    monkeypatch.setattr(db, '_http_get', capture)
    return asked


@pytest.mark.parametrize("call,source", [
    (lambda limit: db.search_ncbi('brca', max_results=limit), 'ncbi'),
    (lambda limit: db.search_uniprot('p53', max_results=limit), 'uniprot'),
    (lambda limit: db.search_pdb('kinase', max_results=limit), 'pdb'),
    (lambda limit: db.search_kegg('glycolysis', max_results=limit), 'kegg'),
])
@pytest.mark.parametrize("limit", [0, -1, -2])
def test_a_page_size_below_one_is_refused_naming_its_value(call, source, limit):
    res = call(limit)

    assert res.source == source
    assert res.error == f"max_results must be at least 1, got {limit}."
    assert res.records == []


def test_a_refused_page_size_never_reaches_a_service(isolated):
    db.search_ncbi('brca', max_results=0)
    db.search_kegg('glycolysis', max_results=-5)
    db.search_uniprot('p53', max_results=0)
    db.search_pdb('kinase', max_results=0)

    assert isolated == []


@pytest.mark.parametrize("limit,expected", [
    (1, ['path:hsa00010']),
    (2, ['path:hsa00010', 'path:hsa00020']),
    (4, ['path:hsa00010', 'path:hsa00020', 'path:hsa00030', 'path:hsa00040']),
    (10, ['path:hsa00010', 'path:hsa00020', 'path:hsa00030', 'path:hsa00040']),
])
def test_blank_rows_do_not_spend_the_page_budget(isolated, limit, expected):
    db.invalidate_cache()
    res = db.search_kegg('glycolysis', max_results=limit)

    assert [r['id'] for r in res.records] == expected
    assert len(res.records) == min(limit, 4)


def test_the_page_size_reaches_the_service_that_uses_one(isolated):
    db.search_ncbi('brca', max_results=7)
    db.invalidate_cache()
    db.search_uniprot('p53', max_results=3)
    db.invalidate_cache()
    db.search_pdb('kinase', max_results=5)

    assert 'retmax=7' in isolated[0][0]
    assert 'size=3' in isolated[1][0]
    assert json.loads(isolated[2][1]['data'])['request_options']['pager']['rows'] == 5


def test_kegg_reports_the_description_it_splits_on_a_tab(isolated):
    res = db.search_kegg('glycolysis', max_results=1)

    assert res.records == [{'id': 'path:hsa00010',
                            'description': 'Glycolysis / Gluconeogenesis'}]
