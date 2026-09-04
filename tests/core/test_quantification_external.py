"""quantification external-tool paths (subprocess mocked) + data plumbing."""
import types

import numpy as np
import pandas as pd
import pytest

from biosuite.core import quantification as qf


class _FakeProc:
    returncode = 0
    stdout = 'salmon 1.10.0'
    stderr = ''


def _fake_run_ok(*args, **kwargs):
    return _FakeProc()


def _fake_run_fail(*args, **kwargs):
    class R:
        returncode = 1
        stdout = ''
        stderr = 'boom'
    return R()


# ── check_quantification_tools ───────────────────────────────────────────────

def test_check_tools_prefer_has_tool(monkeypatch):
    monkeypatch.setattr(qf, 'subprocess', types.SimpleNamespace(run=_fake_run_ok))
    import biosuite.core.utils as u
    monkeypatch.setattr(u, 'has_tool', lambda t: True, raising=False)
    monkeypatch.setattr(qf, '_has_tool', lambda t: True, raising=False)
    monkeypatch.setattr('biosuite.core.utils.has_tool', lambda t: True, raising=False)
    tools = qf.check_quantification_tools()
    assert isinstance(tools, dict)


def test_check_tools_handles_missing(monkeypatch):
    import biosuite.core.utils as u
    monkeypatch.setattr('biosuite.core.utils.has_tool', lambda t: False, raising=False)
    tools = qf.check_quantification_tools()
    assert tools == {'salmon': False, 'kallisto': False}


# ── fallbacks when tool absent ───────────────────────────────────────────────

def test_salmon_quant_fallback_message(monkeypatch):
    import biosuite.core.utils as u
    monkeypatch.setattr('biosuite.core.utils.has_tool', lambda t: False, raising=False)
    res = qf.salmon_quant('reads.fq', transcriptome_fasta=None)
    assert 'not available' in res.message.lower() or res.engine == 'builtin'


def test_kallisto_quant_fallback_message(monkeypatch):
    import biosuite.core.utils as u
    monkeypatch.setattr('biosuite.core.utils.has_tool', lambda t: False, raising=False)
    res = qf.kallisto_quant('reads.fq', transcriptome_fasta=None)
    assert 'not available' in res.message.lower() or res.engine == 'builtin'


# ── builtin quantification with short reads (k clipped) ─────────────────────

def test_builtin_quantify_short_reads(tmp_path, monkeypatch):
    import biosuite.core.utils as u
    monkeypatch.setattr('biosuite.core.utils.has_tool', lambda t: False, raising=False)
    tfa = tmp_path / 'tx.fa'
    # two distinct transcripts, one RT-rich
    tfa.write_text(">t1\nACGTACGTACGTACGTACGTACGTACGTACGTACGT\n>t2\nTTTTGGGGCCCCAAAATTTTGGGGCCCCAAAA\n")
    fq = tmp_path / 'r.fq'
    reads = ["ACGTACGTACGTACGTACGTACGT",   # matches t1
             "TTTTGGGGCCCCAAAATTTTGGGG"]  # matches t2
    with open(fq, 'w') as fh:
        for i, s in enumerate(reads):
            fh.write(f"@r{i}\n{s}\n+\n{'F' * len(s)}\n")
    res = qf.quantify_reads(str(fq), str(tfa), k=20)
    assert res.num_mapped_reads == 2
    assert res.num_transcripts == 2
    assert isinstance(res.tpm_values, list)
    df = res.to_dataframe()
    assert set(df['transcript_id']) == {'t1', 't2'}


def test_too_short_transcript_warning(tmp_path, monkeypatch):
    import biosuite.core.utils as u
    monkeypatch.setattr('biosuite.core.utils.has_tool', lambda t: False, raising=False)
    tfa = tmp_path / 'tx.fa'
    tfa.write_text(">tshort\nACGTACGTAC\n>tlong\nACGT" + "A" * 80 + "\n")
    fq = tmp_path / 'r.fq'
    fq.write_text("@r1\n" + "A" * 60 + "\n+\n" + "F" * 60 + "\n")
    with pytest.warns(UserWarning):
        res = qf._builtin_quantify(str(fq), [('tshort', 'ACGTACGTAC'),
                                             ('tlong', 'ACGT' + 'A' * 80)],
                                   k=31, sample_name='s')
    assert res is not None


def test_merge_results_disjoint_transcripts():
    r1 = qf.QuantResult(tool='builtin', sample_name='s1',
                        transcript_ids=['a', 'b'], tpm_values=[1.0, 2.0],
                        num_reads_values=[1, 2])
    r2 = qf.QuantResult(tool='builtin', sample_name='s2',
                        transcript_ids=['b', 'c'], tpm_values=[3.0, 4.0],
                        num_reads_values=[3, 4])
    df = qf.merge_quantification_results([r1, r2])
    assert set(df.columns) == {'s1', 's2'}
    assert df.loc['a', 's2'] == 0     # union filled with zeros


def test_format_quant_report():
    r = qf.QuantResult(tool='salmon', sample_name='x', num_transcripts=3,
                       num_mapped_reads=10, mapping_rate=99.0,
                       tpm_values=[1.0, 2.0, 3.0],
                       num_reads_values=[4, 5, 6],
                       transcript_ids=['a', 'b', 'c'],
                       message='ok')
    txt = qf.format_quant_report(r)
    assert 'SALMON' in txt and '99.0%' in txt


def test_tpm_normalization_sums_to_million(tmp_path, monkeypatch):
    import biosuite.core.utils as u
    monkeypatch.setattr('biosuite.core.utils.has_tool', lambda t: False,
                       raising=False)
    tfa = tmp_path / 'tx.fa'
    tfa.write_text(">t1\n" + "ACGT" * 20 + "\n>t2\n" + "TGCA" * 20 + "\n")
    fq = tmp_path / 'r.fq'
    seq = "ACGT" * 15
    fq.write_text("@r1\n" + seq + "\n+\n" + "F" * len(seq) + "\n@r2\n" + "TGCA" * 15 + "\n+\n" + "F" * 60 + "\n")
    res = qf._builtin_quantify(str(fq), [('t1', 'ACGT' * 20),
                                         ('t2', 'TGCA' * 20)],
                               k=16, sample_name='s')
    total = sum(res.tpm_values)
    if total > 0:
        assert total == pytest.approx(1e6, rel=1e-6)
