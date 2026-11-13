"""
Unit tests for biosuite.core.databases module.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from biosuite.core.databases import (
    DBResult,
    format_search_results,
    search_kegg,
    search_ncbi,
    search_pdb,
    search_uniprot,
)
from biosuite.core.utils import config


@pytest.fixture(autouse=True)
def _no_api_key_prompts(monkeypatch):
    """Quiet for this module only, restored afterwards.

    This used to be a module-level ``config['quiet'] = True`` that the shared config
    dict carried into every later test file — so a test anywhere after this one that
    expected a prompt found none instead of failing.
    """
    monkeypatch.setitem(config, 'quiet', True)


class TestDBResult:
    def test_creation(self):
        result = DBResult(source='test', query='test query')
        assert result.source == 'test'
        assert result.query == 'test query'
        assert result.records == []
        assert result.error == ''


class TestFormatResults:
    def test_empty_result(self):
        result = DBResult(source='test', query='test')
        formatted = format_search_results(result)
        assert 'No records' in formatted

    def test_error_result(self):
        result = DBResult(source='test', query='test', error='Connection failed')
        formatted = format_search_results(result)
        assert 'Connection failed' in formatted

    def test_with_records(self):
        result = DBResult(source='test', query='test',
                         records=[{'id': '123', 'title': 'Test Record'}])
        formatted = format_search_results(result)
        assert '123' in formatted


class TestSearchNCBI:
    def test_returns_result(self):
        result = search_ncbi('BRCA1')
        assert isinstance(result, DBResult)
        assert result.source == 'ncbi'

    def test_an_empty_query_is_refused_by_name(self):
        result = search_ncbi('')
        assert result.source == 'ncbi'
        assert result.records == []
        assert "Empty query" in result.error, result.error

    def test_a_blank_query_is_refused_by_name(self):
        result = search_ncbi('   ')
        assert result.source == 'ncbi'
        assert result.records == []
        assert "Empty query" in result.error, result.error


class TestSearchUniProt:
    def test_returns_result(self):
        result = search_uniprot('insulin')
        assert isinstance(result, DBResult)
        assert result.source == 'uniprot'


class TestSearchPDB:
    def test_returns_result(self):
        result = search_pdb('hemoglobin')
        assert isinstance(result, DBResult)
        assert result.source == 'pdb'


class TestSearchKEGG:
    def test_returns_result(self):
        result = search_kegg('glycolysis')
        assert isinstance(result, DBResult)
        assert result.source == 'kegg'
