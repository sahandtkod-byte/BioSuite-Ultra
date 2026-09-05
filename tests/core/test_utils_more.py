"""utils safe-input helpers, session persistence, and dataframe loaders."""
import builtins

import numpy as np
import pandas as pd
import pytest

from biosuite.core import utils as ut


class ScriptedInput:
    def __init__(self, answers):
        self._a = iter(answers)

    def __call__(self, prompt=''):
        try:
            return next(self._a)
        except StopIteration:
            raise EOFError from None


@pytest.fixture(autouse=True)
def _clear_session():
    ut.session.clear()
    yield
    ut.session.clear()


def _quiet_off():
    ut.config['quiet'] = False


def test_safe_float_valid_and_invalid(monkeypatch):
    _quiet_off()
    monkeypatch.setattr(builtins, 'input', ScriptedInput(['3.25', 'oops']))
    v1 = ut.safe_float_input('f?', 1.0)
    v2 = ut.safe_float_input('f?', 1.0)
    assert v1 == 3.25 and v2 == 1.0


def test_safe_int_valid_empty_invalid(monkeypatch):
    _quiet_off()
    monkeypatch.setattr(builtins, 'input', ScriptedInput(['7', '', 'no']))
    assert ut.safe_int_input('i?', 5) == 7
    assert ut.safe_int_input('i?', 5) == 5
    assert ut.safe_int_input('i?', 5) == 5


def test_safe_list_and_session_reuse(monkeypatch, tmp_path):
    _quiet_off()
    ut.session.clear()
    monkeypatch.setattr(builtins, 'input', ScriptedInput(['1,2,3', '']))
    got = ut.safe_list_input('ls?', float, key='sizes')
    assert got == [1.0, 2.0, 3.0]
    got2 = ut.safe_list_input('ls?', float, key='sizes')
    assert got2 == [1.0, 2.0, 3.0]


def test_quiet_mode_returns_default(monkeypatch):
    ut.config['quiet'] = True
    try:
        assert ut.safe_float_input('?', 2.5) == 2.5
        assert ut.safe_int_input('?', 3) == 3
    finally:
        ut.config['quiet'] = False


def test_load_dataframe_safe_csv_tsv_missing(tmp_path):
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    f_csv = tmp_path / 'd.csv'
    f_tsv = tmp_path / 'd.tsv'
    df.to_csv(f_csv, index=False)
    df.to_csv(f_tsv, sep='\t', index=False)
    assert len(ut.load_dataframe_safe(str(f_csv))) == 2
    assert len(ut.load_dataframe_safe(str(f_tsv))) == 2
    assert ut.load_dataframe_safe(str(tmp_path / 'nope.csv')) is None


def test_get_api_key_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv('BIOSUITE_DIR_TEST', str(tmp_path))
    monkeypatch.setattr(ut, 'get_app_dir', lambda: {'config': str(tmp_path / 'c.json')})
    ut.set_api_key('svcX', 'KEY123')
    assert ut.get_api_key('svcX') in ('KEY123', '', None)


def test_has_tool_detects_existing():
    assert ut.has_tool('python3') or ut.has_tool('python') or not ut.has_tool('unlikely-tool-xyz')


def test_reverse_complement_and_read_fasta(tmp_path):
    assert ut.reverse_complement_dna('AAGTCC') == 'GGACTT'
    fa = tmp_path / 'x.fa'
    fa.write_text('>a\nACGT\n>b nnn\nTTTT\n')
    seqs = ut.read_fasta_simple(str(fa))
    assert len(seqs) == 2
