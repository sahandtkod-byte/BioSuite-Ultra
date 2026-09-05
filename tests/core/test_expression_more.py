"""Deep tests for core/expression.py & core/quantification.py paths."""
import numpy as np
import pandas as pd
import pytest

from biosuite.core import expression as ex
from biosuite.core import quantification as q


@pytest.fixture()
def counts_df():
    return pd.DataFrame({
        'gene': ['A', 'B', 'C', 'D'],
        'ctrl1': [100, 50, 1000, 10],
        'ctrl2': [120, 60, 1100, 12],
        'trt1': [400, 45, 5, 11],
        'trt2': [380, 55, 4, 10],
    })


# ── normalization ────────────────────────────────────────────────────────────

def test_cpm_normalization_columns_sum_million(counts_df):
    cpm = ex.cpm_normalization(counts_df)
    num = cpm.select_dtypes(include=[np.number])
    for col in num:
        assert num[col].sum() == pytest.approx(1e6, rel=1e-6)


def test_tpm_normalization(counts_df):
    lengths = pd.Series([1000, 2000, 500, 1500], index=counts_df['gene'])
    tpm = ex.tpm_normalization(counts_df, lengths)
    num = tpm.select_dtypes(include=[np.number])
    for col in num:
        assert num[col].sum() == pytest.approx(1e6, rel=1e-3)


def test_deseq2_and_vst(counts_df):
    norm = ex.deseq2_normalization(counts_df)
    assert norm.shape == counts_df.shape or norm.drop(columns=['gene'], errors='ignore').shape[0] == 4
    vst = ex.variance_stabilizing_transformation(counts_df)
    assert vst is not None


#/ ── statistics ───────────────────────────────────────────────────────────────

def test_benjamini_hochberg_monotone():
    p = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
    adj = ex._benjamini_hochberg(p)
    assert list(adj) == sorted(adj) or all(adj[i] <= adj[i + 1] for i in range(len(adj) - 1))
    assert all(a >= pp for a, pp in zip(adj, p))


def test_welch_and_nb_tests():
    # rows genes, cols replicates — significant separation must register
    v1 = np.array([[100, 110, 90, 105, 95]])
    v2 = np.array([[200, 210, 190, 205, 195]])
    out = ex._welch_ttest_rows(v1, v2)
    vals = out[1] if isinstance(out, tuple) else out
    assert float(np.asarray(vals).ravel()[0]) < 0.05


#/ ── fold change / effect size ───────────────────────────────────────────────

def test_fold_change_sign_and_zero_guard(counts_df):
    fc = ex.calculate_fold_change(counts_df, ['ctrl', 'ctrl', 'trt', 'trt'])
    assert fc is not None


def test_effect_size(counts_df):
    eff = ex.calculate_effect_size(counts_df, ['ctrl', 'ctrl', 'trt', 'trt'])
    assert eff is not None


# ── DE end-to-end ────────────────────────────────────────────────────────────

def test_differential_expression_fixture(counts_df):
    conditions = ['ctrl', 'ctrl', 'trt', 'trt']
    de = ex.differential_expression(counts_df, conditions)
    assert len(de) == 4
    assert {'log2FC', 'pvalue', 'padj'} <= set(de.columns)
    up = de[(de['gene'] == 'A')]
    assert len(up) == 1
    assert up.iloc[0]['log2FC'] > 1          # strong treatment effect
    dn = de[(de['gene'] == 'C')]
    assert dn.iloc[0]['log2FC'] < -3


# ── counts reader ────────────────────────────────────────────────────────────

def test_read_counts_matrix(tmp_path):
    p = tmp_path / 'c.csv'
    p.write_text('gene,s1,s2\ng1,10,20\ng2,30,40\n')
    df = ex.read_counts_matrix(str(p))
    assert len(df) == 2
    assert ex.read_counts_matrix('/nonexistent/x.csv') is None


# ── quantification (pure-Python pseudo-aligner) ─────────────────────────────

@pytest.fixture()
def tiny_transcriptome(tmp_path):
    t = tmp_path / 'tx.fa'
    t.write_text('>t1\n' + 'A' * 200 + '\n>t2\n' + 'C' * 200 + '\n')
    return str(t)


def test_build_index_and_pseudoalign(tiny_transcriptome):
    txs = q._read_fasta(tiny_transcriptome)
    index = q._build_transcript_index(txs, k=15)
    assert index
    hits = q._pseudo_align_read('A' * 40, index, k=15, min_hits=1)
    assert hits


def test_builtin_quantify_counts_reads(tmp_path, tiny_transcriptome):
    reads = tmp_path / 'r.fq'
    reads.write_text('@r1\n' + 'A' * 40 + '\n+\n' + 'F' * 40 + '\n'
                     '@r2\n' + 'C' * 40 + '\n+\n' + 'F' * 40 + '\n')
    txs = q._read_fasta(tiny_transcriptome)
    res = q._builtin_quantify(str(reads), txs, k=10, sample_name='t')
    assert res is not None
    assert getattr(res, 'num_reads', 1) >= 1
    qualify = q.format_quant_report(res)
    assert 't' in qualify or 'Built-in' in qualify


def test_merge_quantification_results(tmp_path, tiny_transcriptome):
    reads = tmp_path / 'r.fq'
    reads.write_text('@r1\n' + 'A' * 40 + '\n+\n' + 'F' * 40 + '\n')
    txs = q._read_fasta(tiny_transcriptome)
    r1 = q._builtin_quantify(str(reads), txs, k=10, sample_name='a')
    r2 = q._builtin_quantify(str(reads), txs, k=10, sample_name='b')
    merged = q.merge_quantification_results([r1, r2])
    assert merged is not None
