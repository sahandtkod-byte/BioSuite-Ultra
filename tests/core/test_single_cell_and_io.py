"""scanpy-free-auditable single_cell paths + core/sequence.py FASTA/FASTQ IO."""
import os

import pytest

from biosuite.core import single_cell as scmod
from biosuite.core import sequence as sq


# ── single_cell graceful fallbacks (no scanpy in CI) ────────────────────────

def test_check_single_cell_tools_dict():
    tools = scmod.check_single_cell_tools()
    assert isinstance(tools, dict) and 'scanpy' in tools


def test_load_count_matrix_no_scanpy(tmp_path):
    if scmod.HAS_SCANPY:
        pytest.skip("scanpy installed")
    csv = tmp_path / 'm.csv'
    csv.write_text("gene,c1\nG1,10\n")
    adata, msg = scmod.load_count_matrix(str(csv))
    assert adata is None and 'scanpy' in msg.lower()


def test_load_count_matrix_unknown_extension_returns_msg(tmp_path, monkeypatch):
    if not scmod.HAS_SCANPY:
        pytest.skip("needs scanpy")
    bad = tmp_path / 'm.xyz'
    bad.write_bytes(b"not a matrix")
    adata, msg = scmod.load_count_matrix(str(bad))
    assert adata is None and 'Unrecognized' in msg


def test_qc_pipeline_no_scanpy_guard(monkeypatch):
    if scmod.HAS_SCANPY:
        pytest.skip("scanpy installed")
    adata, report = scmod.run_full_pipeline(object())
    assert 'scanpy' in report.message.lower()


def test_format_sc_report():
    rep = scmod.SingleCellReport(
        num_cells=3, num_genes=4, num_clusters=2,
        cluster_counts={'0': 2, '1': 1},
        top_markers={}, qc_stats={}, message='demo')
    txt = scmod.format_sc_report(rep)
    assert '3' in txt and '2' in txt and 'demo' in txt


# ── sequence read/write ──────────────────────────────────────────────────────

def test_read_fasta_basic(tmp_path):
    p = tmp_path / 'a.fa'
    p.write_text(">s1\nACGT\nTTGG\n>s2 desc here\nCCGA\n")
    records = sq.read_fasta(str(p))
    assert records == [('s1', 'ACGTTTGG'), ('s2 desc here', 'CCGA')]


def test_read_fasta_missing_file():
    assert sq.read_fasta('/definitely/missing/x.fa') is None


def test_read_fastq_basic(tmp_path):
    p = tmp_path / 'r.fq'
    p.write_text("@r1\nACGT\n+\nFFFF\n@r2 description\nTTGA\n+\n####\n")
    records = sq.read_fastq(str(p))
    assert len(records) == 2
    name, seq, qual = records[0]
    assert name == 'r1' and seq == 'ACGT' and qual == 'FFFF'


def test_read_fastq_missing_file():
    assert sq.read_fastq('/definitely/missing/x.fq') is None
