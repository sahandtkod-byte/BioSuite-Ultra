"""Identifiers are placed in the *path* of these URLs, so they must be one segment.

Five endpoints interpolated a caller's value straight after the last slash.  The
host is fixed, so this is not open redirection — but a value carrying a '/' or a
'?' silently changes which endpoint is asked:
``fetch_kegg_pathway('hsa00010/list/compound')`` went to
``/get/hsa00010/list/compound``, ``fetch_uniprot('P12345/../../admin')`` to
``/uniprotkb/P12345/../../admin.json``, and ``search_kegg(q, database='pathway?x=1')``
turned the rest of the path into a query string, dropping the search term on the
floor.  A refusal that names the character beats a response from somewhere else.
"""
import pytest

from biosuite.core import databases as db

KEGG_ROW = b'path:hsa00010\tGlycolysis / Gluconeogenesis\n'


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """One captured URL per call, a canned body, and no waiting or credentials."""
    asked = []

    def capture(url, **kwargs):
        asked.append(url)
        return KEGG_ROW if 'kegg' in url else b'{}'

    db.invalidate_cache()
    monkeypatch.setattr(db, 'get_api_key', lambda name: '')
    monkeypatch.setattr(db, 'prompt_api_key', lambda *a, **k: '')
    monkeypatch.setattr(db, '_http_get', capture)
    return asked


# ── the helpers themselves ────────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["P04637", "P04637-2", "path:hsa00010", "hsa00010",
                                   "BRCA1-AS1", "mus_musculus", "a.b_c-d:e"])
def test_a_real_identifier_survives_the_check_unchanged(value):
    assert db._id_segment(value, "id") == value


@pytest.mark.parametrize("value,bad", [
    ("P12345/../admin", "/"),
    ("P12345?format=tsv", "?"),
    ("P12345#frag", "#"),
    ("P12%3345", "%"),
    ("BR CA2", " "),
])
def test_a_value_that_would_leave_the_path_segment_is_refused(value, bad):
    with pytest.raises(ValueError) as excinfo:
        db._id_segment(value, "UniProt accession")
    assert str(excinfo.value) == (
        f"UniProt accession cannot be placed in a request path: it contains "
        f"{bad!r} (got {value!r})")


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_a_blank_identifier_is_refused_as_empty_by_name(value):
    with pytest.raises(ValueError) as excinfo:
        db._id_segment(value, "KEGG pathway id")
    assert str(excinfo.value) == "KEGG pathway id is empty"


def test_a_free_text_term_keeps_its_spaces_and_encodes_them():
    assert db._query_segment("glycolysis ATP", "term") == "glycolysis%20ATP"
    assert db._query_segment("p53", "term") == "p53"


def test_a_free_text_term_still_cannot_escape_its_segment():
    with pytest.raises(ValueError) as excinfo:
        db._query_segment("pathway/genes", "KEGG search term")
    assert str(excinfo.value) == (
        "KEGG search term cannot be placed in a request path: it contains "
        "'/' (got 'pathway/genes')")


# ── what each endpoint does with them ────────────────────────────────────────

def test_the_kegg_search_puts_the_collection_and_the_term_in_the_path(isolated):
    res = db.search_kegg('mapk', database='pathway')

    assert isolated[-1] == 'https://rest.kegg.jp/find/pathway/mapk'
    assert res.error == ''


def test_a_two_word_kegg_query_is_encoded_not_refused(isolated):
    db.search_kegg('glycolysis ATP')
    db.invalidate_cache()
    db.search_kegg('hexose', database='compound')

    assert isolated == ['https://rest.kegg.jp/find/pathway/glycolysis%20ATP',
                        'https://rest.kegg.jp/find/compound/hexose']


@pytest.mark.parametrize("database,expected", [
    ('pathway?email=a@b', '?'),
    ('pathway/genes', '/'),
    ('', None),
])
def test_a_collection_that_is_not_one_segment_never_reaches_the_network(isolated,
                                                                       database, expected):
    res = db.search_kegg('hexose', database=database)

    assert isolated == []                       # nothing was fetched
    assert res.records == []
    if expected is None:
        assert res.error == "KEGG collection is empty"
    else:
        assert res.error == (
            f"KEGG collection cannot be placed in a request path: it contains "
            f"{expected!r} (got {database!r})")


def test_a_pathway_id_with_a_second_segment_is_refused_by_name(isolated):
    res = db.fetch_kegg_pathway('hsa00010/list/compound')

    assert isolated == []
    assert res.error == (
        "KEGG pathway id cannot be placed in a request path: it contains '/' "
        "(got 'hsa00010/list/compound')")


def test_a_colon_prefixed_pathway_id_is_still_fetched(isolated):
    res = db.fetch_kegg_pathway('path:hsa00010')

    assert isolated[-1] == 'https://rest.kegg.jp/get/path:hsa00010'
    assert res.records == [{'text': KEGG_ROW.decode()}]


def test_an_accession_that_would_change_the_endpoint_is_refused(isolated):
    res = db.fetch_uniprot('P12345/../../admin')

    assert isolated == []
    assert res.source == 'uniprot' and res.query == 'P12345/../../admin'
    assert res.error == (
        "UniProt accession cannot be placed in a request path: it contains "
        "'/' (got 'P12345/../../admin')")


def test_an_isoform_accession_is_a_legitimate_segment(isolated):
    db.fetch_uniprot('P04637-2')

    assert isolated[-1] == 'https://rest.uniprot.org/uniprotkb/P04637-2.json'


def test_both_ensembl_path_values_are_checked(isolated):
    species = db.search_ensembl('BRCA2', species='mus musculus')
    symbol = db.search_ensembl('BR CA2')
    db.invalidate_cache()
    ok = db.search_ensembl('BRCA1-AS1', species='mus_musculus')

    assert species.error == (
        "Ensembl species cannot be placed in a request path: it contains ' ' "
        "(got 'mus musculus')")
    assert symbol.error == (
        "Ensembl gene symbol cannot be placed in a request path: it contains ' ' "
        "(got 'BR CA2')")
    assert isolated == [
        'https://rest.ensembl.org/lookup/symbol/mus_musculus/BRCA1-AS1'
        '?content-type=application/json']
    assert ok.error == ''


def test_a_blank_query_is_refused_before_anything_is_fetched(isolated):
    assert db.search_ensembl('  ').error == "Ensembl gene symbol is empty"
    assert db.search_kegg('  ').error == "KEGG search term is empty"
    assert isolated == []
