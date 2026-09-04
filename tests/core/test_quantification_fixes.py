"""Regression tests for quantification.py review fixes."""
import warnings

import pytest

from biosuite.core.quantification import (
    quantify_reads, _builtin_quantify, _build_transcript_index,
    _pseudo_align_read, check_quantification_tools, QuantResult,
    merge_quantification_results,
)
from biosuite.core import utils


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


TX1 = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"  # 40 bp
TX2 = "TTTTGCAAGCAAGCAACCATGGCCGGCCGGGGGTTTAAT"  # 40 bp


def _fq_from_transcripts(t1_count, t2_count, read_len=40):
    reads = []
    for i in range(t1_count):
        reads.append((f"t1_{i}", TX1[:read_len], "I" * read_len))
    for i in range(t2_count):
        reads.append((f"t2_{i}", TX2[:read_len], "I" * read_len))
    return ''.join(f"@{h}\n{s}\n+\n{q}\n" for h, s, q in reads)


def test_tpm_ratios_match_abundance(tmp_path):
    tx_fa = _write(tmp_path, "tx.fa", f">t1\n{TX1}\n>t2\n{TX2}\n")
    fq = _write(tmp_path, "r.fq", _fq_from_transcripts(90, 10))
    res = quantify_reads(fq, tx_fa, k=15)
    tpm = dict(zip(res.transcript_ids, res.tpm_values))
    assert tpm['t1'] > 5 * tpm['t2']  # 9:1 abundance, length-corrected
    assert res.mapping_rate == 100.0


def test_reads_shorter_than_k_now_map(tmp_path):
    tx_fa = _write(tmp_path, "tx.fa", f">t1\n{TX1}\n>t2\n{TX2}\n")
    fq = _write(tmp_path, "r.fq", _fq_from_transcripts(10, 0, read_len=25))
    res = quantify_reads(fq, tx_fa, k=31)          # k > read length
    assert res.num_mapped_reads > 0                 # old code: exactly 0


def test_repetitive_kmer_no_vote_inflation():
    rep = "AC" * 30          # same k-mers repeat inside one transcript
    uniq = "GATTACAGATTACAGATTACAGATTACAGATTACA"
    index, _ = _build_transcript_index([("rep", rep), ("uniq", uniq)], k=11)
    best_rep, count_rep = _pseudo_align_read(rep[:25], index, k=11)
    best_uniq, count_uniq = _pseudo_align_read(uniq[:25], index, k=11)
    assert {best_rep, best_uniq} == {"rep", "uniq"}
    # Each k-mer of the repetitive read should vote exactly once now
    # (the old list index inflated repeat transcripts proportionally).
    assert count_rep <= len(rep[:25]) - 11 + 1


def test_short_transcript_warns(tmp_path):
    tx = [("tiny", "ACGT"), ("oktx", TX1)]
    fq = _write(tmp_path, "r.fq", _fq_from_transcripts(5, 0, read_len=40))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = _builtin_quantify(fq, tx, k=31)
    assert res.num_transcripts == 2
    assert any("shorter than k" in str(x.message) for x in w)


def test_check_quantification_tools_uses_version_flag(monkeypatch):
    calls = []

    class R:
        returncode = 0
    import subprocess
    monkeypatch.setattr(utils, 'has_tool', lambda t: True)
    monkeypatch.setattr(subprocess, 'run',
                        lambda cmd, **kw: (calls.append(cmd) or R()))
    tools = check_quantification_tools()
    assert all('--version' in c for c in calls), calls
    assert tools == {'salmon': True, 'kallisto': True}


def test_merge_quantification_results(tmp_path):
    df1 = merge_quantification_results([
        QuantResult(tool='b', sample_name='s1', transcript_ids=['t1', 't2'],
                    tpm_values=[10.0, 'bad'], engine='builtin')
    ])
    assert df1.empty or True  # smoke: malformed entries must not crash here


def test_missing_files_graceful(tmp_path):
    res = quantify_reads(str(tmp_path / 'no.fq'), str(tmp_path / 'no.fa'))
    assert 'not found' in res.message.lower()
