"""Regression tests for assembly.py review fixes."""
import random

import pytest

from biosuite.core.assembly import (
    assemble, _builtin_assembly, _compute_assembly_stats, _compute_n50,
    _compute_l50, _parse_assembly_fasta, AssemblyResult,
    _detect_unitigs, _load_reads_fasta,
)


def _fake_reads(genome_len=300, read_len=50, step=20):
    random.seed(7)
    genome = ''.join(random.choice('ACGT') for _ in range(genome_len))
    reads = [genome[i:i + read_len] for i in range(0, genome_len - read_len, step)]
    return genome, reads


def _write_fasta(tmp_path, reads):
    p = tmp_path / "reads.fa"
    p.write_text("".join(f">r{i}\n{s}\n" for i, s in enumerate(reads)))
    return str(p)


def test_overlap_graph_reconstructs_region(tmp_path):
    genome, reads = _fake_reads()
    res = _builtin_assembly(_write_fasta(tmp_path, reads))
    assert res.num_contigs == 1, res.message
    contig = ''.join(
        line.strip() for line in open(res.output_fasta)
        if line.strip() and not line.startswith('>')
    )
    # contig must reassemble the covered region exactly
    covered = genome[:len(genome) - (len(genome) - 50) % 20]
    assert contig == covered
    assert res.n50 == len(covered)


def test_unitig_pass_through_is_identity():
    contigs = ["ACGT" * 10, "TTGGCCAA"]
    assert _detect_unitigs([], contigs, {}, 15) is contigs


def test_n50_l50_known_answers():
    lengths = [8, 7, 6, 5, 4]  # total 30; half = 15 -> N50=7, L50=2
    assert _compute_n50(lengths) == 7
    assert _compute_l50(lengths) == 2


def test_stats_returns_result_not_dict():
    res = _compute_assembly_stats(["ACGTACGT"])
    assert isinstance(res, AssemblyResult)
    assert res.gc_content == 50.0
    assert res.min_contig == 8


def test_stats_empty_contigs():
    res = _compute_assembly_stats([])
    assert res.num_contigs == 0 and "No contigs" in res.message


def test_parse_assembly_fasta_returns_assembly_result(tmp_path):
    p = tmp_path / "contigs.fasta"
    p.write_text(">c1\nACGT\n>c2\nACGTACGT\n")
    res = _parse_assembly_fasta(str(p), "test")
    assert isinstance(res, AssemblyResult)
    assert res.engine == "test" and res.num_contigs == 2


def test_fastq_and_fasta_loading(tmp_path):
    fq = tmp_path / "r.fq"
    fq.write_text("@r1\nACGT\n+\nIIII\n@r2\nTTGG\n+\nIIII\n")
    reads = _load_reads_fasta(str(fq))
    assert [r[0] for r in reads] == ["r1", "r2"]
    fa = tmp_path / "r.fa"
    fa.write_text(">r1\nAC\nGT\n>r2\nTTGG\n")
    reads = _load_reads_fasta(str(fa))
    assert reads[0] == ("r1", "ACGT")


def test_assemble_missing_file():
    res = assemble("/nonexistent.fa")
    assert res.engine == 'none'


def test_builtin_assembly_performance_smoke(tmp_path):
    # ~600 reads: the dead unitig scan made this quadratic; keep it snappy.
    genome, reads = _fake_reads(600, 60, 10)
    import time
    t0 = time.time()
    res = _builtin_assembly(_write_fasta(tmp_path, reads))
    assert time.time() - t0 < 20
    assert res.message.startswith("Built-in")
