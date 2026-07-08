"""
Unit tests for biosuite.core.databases module.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biosuite.core.databases import (
    search_ncbi, search_uniprot, search_pdb, search_kegg,
    format_search_results, DBResult
)
from biosuite.core.utils import config

# Prevent API key prompts during testing
config['quiet'] = True


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

    def test_error_handling(self):
        # Test with invalid query
        result = search_ncbi('')
        assert isinstance(result, DBResult)


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
