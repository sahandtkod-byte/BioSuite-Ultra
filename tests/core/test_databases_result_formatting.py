"""How a `DBResult` becomes the text a user reads.

The formatters used to test ``source in results`` before they tested what
*results* was, so handing over one result plus its name — the natural pairing
after ``search_all(query, ['ncbi'])`` — died as
``TypeError: argument of type 'DBResult' is not iterable``.  And a twelve-record
answer printed ten lines and never said that two were missing.
"""
from biosuite.core import databases as db

ONE = db.DBResult(source='uniprot', query='p53',
                  records=[{'accession': 'P04637', 'organism': 'Homo sapiens'}])
TWO = db.DBResult(source='pdb', query='q', records=[{'pdb_id': '1TUP'}])


def test_a_single_result_and_a_single_result_with_its_name_both_format():
    assert db.format_search_results(ONE) == '  1. accession: P04637 | organism: Homo sapiens'
    assert db.format_search_results(ONE, source='uniprot') == \
        '  1. accession: P04637 | organism: Homo sapiens'


def test_a_dict_shows_only_the_requested_source():
    results = {'ncbi': db.DBResult(source='ncbi', query='q', records=[{'id': 'NM_1'}]),
               'pdb': TWO}

    assert db.format_search_results(results, source='ncbi') == '  1. id: NM_1'


def test_a_dict_without_a_named_source_labels_each_database_section():
    results = {
        'ncbi': db.DBResult(source='ncbi', query='q', records=[{'id': 'NM_1'}]),
        'kegg': db.DBResult(source='kegg', query='q', error='HTTP 500: Bad Gateway'),
    }

    assert db.format_search_results(results) == (
        '\n=== NCBI Results ===\n  1. id: NM_1\n'
        '\n=== KEGG Results ===\nError: HTTP 500: Bad Gateway')


def test_a_source_that_is_not_a_key_shows_everything_rather_than_nothing():
    results = {'ncbi': db.DBResult(source='ncbi', query='q', records=[{'id': 'NM_1'}]),
               'pdb': TWO}

    assert db.format_search_results(results, source='nope').startswith('\n=== NCBI Results ===')
    assert '1TUP' in db.format_search_results(results, source='nope')


def test_unknown_input_is_reported_as_no_results():
    assert db.format_search_results('not a result') == 'No results'
    assert db.format_search_results(None) == 'No results'
    assert db.format_search_results({}) == ''
    assert db.format_search_results([], source='ncbi') == 'No results'


def test_a_record_line_skips_empty_fields_and_shows_three_of_them():
    result = db.DBResult(source='ncbi', query='q', records=[
        {'id': 'NM_1', 'title': 'a title', 'organism': '', 'extra': None,
         'fourth': 'ignored', 'fifth': 'ignored'}])

    assert db._format_single(result) == '  1. id: NM_1 | title: a title | fourth: ignored'


def test_more_records_than_the_cap_are_announced_with_their_count():
    result = db.DBResult(source='ncbi', query='q', records=[
        {'id': str(i), 'title': f'Record {i}', 'organism': '', 'extra': None}
        for i in range(1, 13)
    ])

    lines = db._format_single(result).splitlines()

    assert len(lines) == db.MAX_DISPLAYED_RECORDS + 1     # 12 in, 10 shown + 1 note
    assert lines[0] == '  1. id: 1 | title: Record 1'
    assert lines[9] == '  10. id: 10 | title: Record 10'
    assert lines[10] == '  … 2 more record(s) not shown.'
    assert 'organism' not in '\n'.join(lines) and 'None' not in '\n'.join(lines)


def test_a_result_at_or_below_the_display_cap_is_not_followed_by_a_note():
    ten = db.DBResult(source='ncbi', query='q',
                      records=[{'id': str(i)} for i in range(1, 11)])
    assert db._format_single(ten).splitlines()[-1] == '  10. id: 10'
    assert 'more record' not in db._format_single(ten)


def test_an_error_result_is_printed_as_the_error_and_not_as_empty_output():
    assert db._format_single(db.DBResult(source='pdb', query='q', error='no such entry')) \
        == 'Error: no such entry'
    assert db._format_single(db.DBResult(source='pdb', query='q')) == 'No records found.'
