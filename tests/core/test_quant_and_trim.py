"""Quantification builtin k-mer method + trimming pure-python paths."""
import pytest
import re

from biosuite.core import quantification as qf
from biosuite.core import trimming as tr


# ── quantification builtin ───────────────────────────────────────────────────
#
# These four used to be smoke tests (`assert res is not None`, `isinstance`, and an
# `if hasattr(...)` that never held), so none of them could fail.  The k-mer engine
# is now pinned by value; the disjunctive assertions were replaced because the old
# third test handed `_pseudo_align_read` the (index, lengths) tuple it got back from
# `_build_transcript_index`, so the lookup always missed and the function answered
# (None, 0) — and `assert out is not None` still passed.

TX1 = 'ACGT' * 50      # 200 nt
TX2 = 'ACAA' * 50      # 200 nt, no 31-mer shared with TX1

FQ_SHARED = "".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(
    ['ACGT' * 10, 'ACGT' * 10, 'ACAA' * 10, 'TTTT' * 10]))


def _files(tmp_path):
    tx = tmp_path / 'tx.fasta'
    tx.write_text(f">tx1\n{TX1}\n>tx2\n{TX2}\n", encoding='utf-8')
    reads = tmp_path / 'reads.fq'
    reads.write_text(FQ_SHARED, encoding='utf-8')
    return str(reads), str(tx)


def test_builtin_quantify_counts_reads_per_transcript_and_flags_the_unmapped(tmp_path):
    reads, tx = _files(tmp_path)
    res = qf.quantify_reads(reads, tx, sample_name='s1', k=31)
    assert res.transcript_ids == ['tx1', 'tx2']
    assert res.num_reads_values == [2, 1]
    assert res.num_mapped_reads == 3
    assert res.mapping_rate == 75.0
    assert res.engine == 'builtin' and res.tool == 'builtin_kmer'
    assert res.message == ("Using built-in k-mer quantifier (3/4 reads mapped)")
    # Length-normalised per million: both transcripts are 200 nt, so the TPM ratio
    # is the count ratio and the column still sums to one million.
    tpm = res.tpm_values
    assert len(tpm) == 2 and sum(tpm) == pytest.approx(1e6)
    assert tpm[0] == pytest.approx(2 / 3 * 1e6)
    assert tpm[1] == pytest.approx(1 / 3 * 1e6)


def test_builtin_quantify_refuses_a_fasta_input_instead_of_guessing(tmp_path):
    """`_builtin_quantify` used to read any four-line block as FASTQ, so a FASTA
    file answered '1 reads mapped, 100.0%' — measured on a two-record FASTA."""
    reads = tmp_path / 'reads.fasta'
    reads.write_text(f">r0\n{TX1[:40]}\n>r1\n{'TTTT' * 10}\n", encoding='utf-8')
    _, tx = _files(tmp_path)
    with pytest.warns(UserWarning, match="skipped 1 malformed FASTQ record"):
        res = qf.quantify_reads(str(reads), tx, k=31)
    assert res.num_mapped_reads == 0 and res.num_transcripts == 2
    assert res.mapping_rate == 0
    assert "1 record(s) skipped as malformed FASTQ" in res.message


def test_builtin_quantify_reads_a_gzipped_fastq(tmp_path):
    """A plain `open(encoding='utf-8')` died on the gzip magic byte (0x8b) with
    UnicodeDecodeError, although .fastq.gz is how sequencer output arrives."""
    import gzip
    reads = tmp_path / 'reads.fq.gz'
    with gzip.open(str(reads), 'wt', encoding='utf-8') as fh:
        fh.write(FQ_SHARED)
    _, tx = _files(tmp_path)
    res = qf.quantify_reads(str(reads), tx, sample_name='gz', k=31)
    assert res.num_mapped_reads == 3
    assert res.mapping_rate == 75.0
    assert res.num_reads_values == [2, 1]


def test_builtin_quantify_tolerates_blank_lines_between_records(tmp_path):
    """Real FASTQ dumps carry stray blank lines between records; they are skipped
    without being counted as a read, exactly as `sequence.read_fastq` treats them."""
    reads = tmp_path / 'spaced.fq'
    reads.write_text(FQ_SHARED.replace("@r", "\n@r"), encoding='utf-8')
    _, tx = _files(tmp_path)
    res = qf.quantify_reads(str(reads), tx, sample_name='spaced', k=31)
    assert res.num_mapped_reads == 3
    assert res.mapping_rate == 75.0
    assert res.num_reads_values == [2, 1]
    assert "malformed" not in res.message


def test_build_transcript_index_returns_the_map_and_the_lengths():
    kmer_map, lengths = qf._build_transcript_index([('t1', 'ACGT' * 10)], k=31)
    assert lengths == {'t1': 40}
    # 40 - 31 + 1 = 10 windows, but the sequence has period 4, so only 4 distinct
    # 31-mers exist — and a k-mer repeated inside one transcript must vote once.
    assert len(kmer_map) == 4
    assert all(v == ['t1'] for v in kmer_map.values())
    assert 'ACGT' * 7 + 'ACG' in kmer_map


def test_pseudo_align_read_votes_for_the_matching_transcript():
    kmer_map, _lengths = qf._build_transcript_index([('t1', 'ACGT' * 10)], k=31)
    assert qf._pseudo_align_read('ACGT' * 10, kmer_map, k=31) == ('t1', 10)
    assert qf._pseudo_align_read('TTTT' * 10, kmer_map, k=31) == (None, 0)
    # below min_hits the read is unassigned rather than guessed
    assert qf._pseudo_align_read('ACGT' * 3, kmer_map, k=31) == (None, 0)


def test_merge_quantification_results_lays_samples_side_by_side(tmp_path):
    reads, tx = _files(tmp_path)
    res1 = qf.quantify_reads(reads, tx, sample_name='a', k=31)
    res2 = qf.quantify_reads(reads, tx, sample_name='b', k=31)
    merged = qf.merge_quantification_results([res1, res2])
    assert merged.shape == (2, 2)
    assert list(merged.columns) == ['a', 'b']
    assert merged.index.tolist() == ['tx1', 'tx2']
    assert merged['a'].tolist() == merged['b'].tolist()
    assert merged.loc['tx1', 'a'] == pytest.approx(2 / 3 * 1e6)
    assert qf.merge_quantification_results([]).empty


# ── trimming ─────────────────────────────────────────────────────────────────

FQ = "\n".join(
    f"@r{i}\n{seq}\n+\n{qual}\n"
    for i, seq, qual in [
        (1, 'AGATCGGAAGAGCACACGTCTGAACTCCAGTCA' + 'ACGT' * 10, 'I' * 73),
        (2, 'ACGT' * 25, 'I' * 100),
        (3, 'AGATCGGAAGAGCACACGTCTGAACTCCAGTCA' + 'TTTT' * 6, 'I' * 57),
    ]
)
# Record 1 used to carry 73 bases over a 72-character quality line.  The trimmer
# accepted it and wrote it back unchanged, so this fixture - and the four tests
# below - were built on a file no FASTQ reader downstream would take.

ADAPTER = 'AGATCGGAAGAGCACACGTCTGAACTCCAGTCA'


def test_pure_python_trim_adapters(tmp_path):
    fa = tmp_path / 'in.fq'
    fo = tmp_path / 'out.fq'
    fa.write_text(FQ, encoding="utf-8")
    rep = tr.trim_fastq(str(fa), str(fo), quality_threshold=20, adapter=ADAPTER)
    assert (rep.total_reads, rep.reads_trimmed, rep.reads_removed,
            rep.adapter_trimmed) == (3, 2, 2, 2)
    assert rep.engine == 'builtin'
    # Both adapter-bearing reads are cut at position 0 and so vanish; only r2 survives.
    assert fo.read_text(encoding="utf-8").count('@') == 1
    assert fo.read_text(encoding="utf-8").splitlines()[0] == '@r2'
    assert rep.avg_quality_before == pytest.approx(40.0)
    assert rep.avg_quality_after == pytest.approx(40.0)


def test_find_adapter_position():
    seq = 'ACGT' * 8 + ADAPTER
    assert tr._find_adapter(seq, [ADAPTER]) == 32
    assert tr._find_adapter('ACGT' * 8, [ADAPTER]) == -1
    earlier = 'AC' + ADAPTER + 'TTTT' + ADAPTER
    assert tr._find_adapter(earlier, [ADAPTER]) == 2


def test_resolve_adapters_named():
    """An unknown name is refused: silently falling back to 'auto' would trim with
    every known adapter, including ones the caller never asked for."""
    assert tr._resolve_adapters('auto', 'illumina_nextera') == ['CTGTCTCTTATACACATCT']
    assert tr._resolve_adapters('auto', 'polya') == ['AAAAAAAAAA']
    assert len(tr._resolve_adapters('auto', None)) == len(tr.ADAPTERS)
    assert tr._resolve_adapters('ACGT', None) == ['ACGT'], "a literal sequence passes through"
    with pytest.raises(ValueError, match=re.escape(
            "unknown adapter_name 'truseq'; known names are illumina_nextera, "
            "illumina_truseq_dna, illumina_truseq_rna, polya, polyg, or pass the "
            "adapter sequence in adapter=")):
        tr._resolve_adapters('auto', 'truseq')


def test_trim_pair_end(tmp_path):
    fa1 = tmp_path / 'in1.fq'
    fa2 = tmp_path / 'in2.fq'
    fa1.write_text(FQ, encoding="utf-8")
    fa2.write_text(FQ, encoding="utf-8")
    out = tr.trim_pair_end(str(fa1), str(fa2),
                           output_r1=str(tmp_path / 'p1.fq'), output_r2=str(tmp_path / 'p2.fq'),
                           adapter=ADAPTER)
    assert (out.total_reads, out.reads_trimmed, out.reads_removed,
            out.adapter_trimmed) == (6, 4, 4, 4)
    assert out.engine == 'builtin'
    assert '1/3 pairs kept' in out.message
    for mate in (tmp_path / 'p1.fq', tmp_path / 'p2.fq'):
        assert mate.read_text(encoding="utf-8").count('@') == 1, "mates stay in lockstep"
    assert out.avg_quality_before == pytest.approx(40.0)
    assert out.avg_quality_after == pytest.approx(40.0)


def test_analyze_fastq_quality(tmp_path):
    fa = tmp_path / 'x.fq'
    fa.write_text(FQ, encoding="utf-8")
    assert tr.analyze_fastq_quality(str(fa)) == {
        'total_reads': 3,
        'read_length_mean': pytest.approx(76.66666666666667),
        'quality_mean': pytest.approx(40.0),
        'quality_median': pytest.approx(40.0),
        'percent_above_q20': pytest.approx(100.0),
        'percent_above_q30': pytest.approx(100.0),
    }


def test_format_trim_report(tmp_path):
    fa = tmp_path / 'in.fq'
    fo = tmp_path / 'out.fq'
    fa.write_text(FQ, encoding="utf-8")
    rep = tr.trim_fastq(str(fa), str(fo), quality_threshold=20, adapter=ADAPTER)
    text = tr.format_trim_report(rep)
    assert text.splitlines()[0] == "=== FASTQ Trimming Report ==="
    assert "Engine: builtin" in text
    assert "Total reads: 3" in text
    assert "Reads trimmed: 2" in text and "Reads removed (too short): 2" in text
    assert "Adapters trimmed: 2" in text
    assert "Mean quality before: 40.0" in text and "Mean quality after: 40.0" in text
