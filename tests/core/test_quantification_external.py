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

def _tools_available(monkeypatch, value):
    """Report the external tools as installed (or not), at every probe site.

    ``quantification`` imports ``has_tool`` at module level, so patching only
    ``biosuite.core.utils`` would let the version probe see the fake while the index
    builders still saw the real machine.
    """
    monkeypatch.setattr(qf, 'has_tool', lambda name: value, raising=False)
    monkeypatch.setattr('biosuite.core.utils.has_tool', lambda name: value,
                        raising=False)


def test_check_tools_prefer_has_tool(monkeypatch):
    monkeypatch.setattr(qf, 'subprocess', types.SimpleNamespace(run=_fake_run_ok))
    _tools_available(monkeypatch, True)
    assert qf.check_quantification_tools() == {'salmon': True, 'kallisto': True}


def test_check_tools_handles_missing(monkeypatch):
    _tools_available(monkeypatch, False)
    tools = qf.check_quantification_tools()
    assert tools == {'salmon': False, 'kallisto': False}


# ── fallbacks when tool absent ───────────────────────────────────────────────

def test_salmon_quant_fallback_message(monkeypatch):
    _tools_available(monkeypatch, False)
    res = qf.salmon_quant('reads.fq', transcriptome_fasta='tx.fa')
    assert res.engine == 'builtin' and res.tool == 'salmon'
    assert res.message == ("Salmon not available. Use quantify_reads() for built-in.")
    assert res.mapping_rate == 0.0 and res.num_mapped_reads == 0


def test_kallisto_quant_fallback_message(monkeypatch):
    _tools_available(monkeypatch, False)
    res = qf.kallisto_quant('reads.fq', transcriptome_fasta='tx.fa')
    assert res.engine == 'builtin' and res.tool == 'kallisto'
    assert res.message == ("Kallisto not available. Use quantify_reads() for built-in.")


def test_salmon_quant_without_any_index_is_refused(monkeypatch):
    """The missing-input case used to answer the same 'Salmon not available' message
    an uninstalled tool produces, which sent the caller looking at PATH instead of at
    their own arguments."""
    _tools_available(monkeypatch, True)
    with pytest.raises(ValueError, match=(
            r"^salmon_quant needs a Salmon index: pass index_dir=\.\.\., or "
            r"transcriptome_fasta=\.\.\. so one can be built\.$")):
        qf.salmon_quant('reads.fq')


def test_kallisto_quant_without_any_index_is_refused(monkeypatch):
    _tools_available(monkeypatch, True)
    with pytest.raises(ValueError, match=(
            r"^kallisto_quant needs a Kallisto index: pass index_file=\.\.\., or "
            r"transcriptome_fasta=\.\.\. so one can be built\.$")):
        qf.kallisto_quant('reads.fq')


def _fake_ok(monkeypatch):
    """Pretend the external command ran successfully; the tables are on disk."""
    monkeypatch.setattr(qf, 'subprocess', types.SimpleNamespace(run=_fake_run_ok))


def test_an_external_run_reports_its_reads_and_an_unknown_rate(tmp_path, monkeypatch):
    """Both wrappers left num_mapped_reads=0 and mapping_rate=0.0 after a successful
    external run, so the CLI/GUI report read 'Mapped reads: 0 / Mapping rate: 0.0%'."""
    _fake_ok(monkeypatch)
    out = tmp_path / 'o' / 'sample'
    out.mkdir(parents=True)
    (out / 'quant.sf').write_text(
        "Name\tLength\tEffectiveLength\tTPM\tNumReads\nt1\t100\t70\t6e5\t18.4\n"
        "t2\t100\t70\t4e5\t11.6\n", encoding='utf-8')
    res = qf._salmon_quant('reads.fq', None, str(tmp_path / 'idx'), str(tmp_path / 'o'),
                           'sample', 1)
    assert res.num_mapped_reads == 30, "sum of the assigned reads, rounded"
    assert res.mapping_rate is None, "no total read count was reported"
    text = qf.format_quant_report(res)
    assert "Mapped reads: 30" in text
    assert "Mapping rate: not assessed (the engine reported no total read count)" in text
    assert "0.0%" not in text


def test_salmon_meta_info_gives_the_real_mapping_numbers(tmp_path, monkeypatch):
    import json
    _fake_ok(monkeypatch)
    out = tmp_path / 'o' / 'sample'
    out.mkdir(parents=True)
    (out / 'quant.sf').write_text(
        "Name\tLength\tEffectiveLength\tTPM\tNumReads\nt1\t100\t70\t1e6\t25.0\n",
        encoding='utf-8')
    (out / 'meta_info.json').write_text(json.dumps(
        {"num_mapped_reads": 4096.0, "percent_mapped": 81.25}), encoding='utf-8')
    res = qf._salmon_quant('reads.fq', None, str(tmp_path / 'idx'), str(tmp_path / 'o'),
                           'sample', 1)
    assert (res.num_mapped_reads, res.mapping_rate) == (4096, 81.25)
    assert "Mapping rate: 81.2%" in qf.format_quant_report(res) or \
           "Mapping rate: 81.3%" in qf.format_quant_report(res)


def test_a_broken_meta_info_falls_back_to_the_abundance_sum(tmp_path, monkeypatch):
    _fake_ok(monkeypatch)
    out = tmp_path / 'o' / 'sample'
    out.mkdir(parents=True)
    (out / 'quant.sf').write_text(
        "Name\tLength\tEffectiveLength\tTPM\tNumReads\nt1\t100\t70\t1e6\t12.4\n",
        encoding='utf-8')
    (out / 'meta_info.json').write_text('{"num_mapped_reads": "n/a"}', encoding='utf-8')
    res = qf._salmon_quant('reads.fq', None, str(tmp_path / 'idx'), str(tmp_path / 'o'),
                           'sample', 1)
    assert res.num_mapped_reads == 12
    assert res.mapping_rate is None


# ── builtin quantification with short reads (k clipped) ─────────────────────

def test_builtin_quantify_short_reads(tmp_path, monkeypatch):
    _tools_available(monkeypatch, False)
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
    _tools_available(monkeypatch, False)
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
    _tools_available(monkeypatch, False)
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
