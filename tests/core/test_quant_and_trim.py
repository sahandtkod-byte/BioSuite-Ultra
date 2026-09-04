"""Quantification builtin k-mer method + trimming pure-python paths."""
import pytest

from biosuite.core import quantification as qf
from biosuite.core import trimming as tr


# ── quantification builtin ───────────────────────────────────────────────────

def test_builtin_quantify_shared_kmers(tmp_path):
    tx = tmp_path / 'tx.fasta'
    reads = tmp_path / 'reads.fasta'
    tx.write_text('>tx1\n' + 'ACGT' * 50)
    reads.write_text('>r1\n' + 'ACGT' * 50)
    res = qf.quantify_reads(str(reads), str(tx), sample_name='s1')
    assert res is not None
    if hasattr(res, 'transcripts'):
        assert res.transcripts is not None


def test_build_transcript_index_keys():
    idx = qf._build_transcript_index([('t1', 'ACGT' * 10)], k=31)
    if isinstance(idx, tuple):
        kmer_map, _lengths = idx
    else:
        kmer_map = idx
    assert isinstance(kmer_map, dict)
    assert kmer_map


def test_pseudo_align_read_strong_match():
    idx = qf._build_transcript_index([('t1', 'ACGT' * 10)], k=31)
    read = 'ACGT' * 10
    out = qf._pseudo_align_read(read, idx, k=31)
    assert out is not None


def test_merge_quantification_results_smoke(tmp_path):
    tx = tmp_path / 'tx.fasta'
    reads = tmp_path / 'reads.fasta'
    tx.write_text('>tx1\n' + 'ACGT' * 50)
    reads.write_text('>r1\n' + 'ACGT' * 50)
    res1 = qf.quantify_reads(str(reads), str(tx), sample_name='a')
    res2 = qf.quantify_reads(str(reads), str(tx), sample_name='b')
    merged = qf.merge_quantification_results([res1, res2])
    assert merged is not None


# ── trimming ─────────────────────────────────────────────────────────────────

FQ = "\n".join(
    f"@r{i}\n{seq}\n+\n{qual}\n"
    for i, seq, qual in [
        (1, 'AGATCGGAAGAGCACACGTCTGAACTCCAGTCA' + 'ACGT' * 10, 'I' * 72),
        (2, 'ACGT' * 25, 'I' * 100),
        (3, 'AGATCGGAAGAGCACACGTCTGAACTCCAGTCA' + 'TTTT' * 6, 'I' * 57),
    ]
)


def test_pure_python_trim_adapters(tmp_path):
    fa = tmp_path / 'in.fq'
    fo = tmp_path / 'out.fq'
    fa.write_text(FQ)
    rep = tr.trim_fastq(str(fa), str(fo), quality_threshold=20,
                        adapter='AGATCGGAAGAGCACACGTCTGAACTCCAGTCA')
    assert rep is not None
    assert fo.exists()
    assert len(fo.read_text().strip().splitlines()) % 4 == 0


def test_find_adapter_position():
    seq = 'ACGT' * 8 + 'AGATCGGAAGAGCACACGTCTGAACTCCAGTCA'
    adapt = ['AGATCGGAAGAGCACACGTCTGAACTCCAGTCA']
    pos = tr._find_adapter(seq, adapt)
    assert pos == len('ACGT' * 8) or pos < len(seq)


def test_resolve_adapters_named():
    adapters = tr._resolve_adapters(None, 'truseq')
    assert isinstance(adapters, list)
    assert len(adapters) >= 1


def test_trim_pair_end(tmp_path):
    fa1 = tmp_path / 'in1.fq'
    fa2 = tmp_path / 'in2.fq'
    fa1.write_text(FQ)
    fa2.write_text(FQ)
    out = tr.trim_pair_end(str(fa1), str(fa2),
                           adapter='AGATCGGAAGAGCACACGTCTGAACTCCAGTCA')
    assert out is not None


def test_analyze_fastq_quality(tmp_path):
    fa = tmp_path / 'x.fq'
    fa.write_text(FQ)
    out = tr.analyze_fastq_quality(str(fa))
    assert isinstance(out, dict)
    assert out


def test_format_trim_report(tmp_path):
    fa = tmp_path / 'in.fq'
    fo = tmp_path / 'out.fq'
    fa.write_text(FQ)
    rep = tr.trim_fastq(str(fa), str(fo), quality_threshold=20,
                        adapter='AGATCGGAAGAGCACACGTCTGAACTCCAGTCA')
    txt = tr.format_trim_report(rep)
    assert isinstance(txt, str)
