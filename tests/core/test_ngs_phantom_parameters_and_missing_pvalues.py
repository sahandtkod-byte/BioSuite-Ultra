"""Phantom parameters in ngs.py and the p-value a broken cell becomes.

`read_vcf` advertised ``chunk_size`` and `manhattan_from_vcf` advertised
``threshold``: neither name appears in its function body, so both were refusals
waiting to be made explicit.  Measured before the change — a 50-record VCF read
with ``chunk_size=1`` returned the same 50 rows as the default, and passing
``threshold=0.05`` returned a frame equal to the one built without it.
"""
import numpy as np
import pandas as pd
import pytest

from biosuite.core import ngs as ng

VCF = "\n".join([
    "##fileformat=VCFv4.2",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    "chr1\t1\trs1\tA\tG\t60\tPASS\tDP=5",
    "chr1\t5\trs5\tC\tT\t30\t.\tDP=2",
    "chr2\t2\trs2\tG\tA\t99\tPASS\tDP=9",
    "chr2\t8\t.\tAT\tA\t12\tLOWQUAL\tDP=1",
]) + "\n"


def test_read_vcf_reads_every_record_and_no_longer_offers_chunking(tmp_path):
    vcf = tmp_path / "v.vcf"
    vcf.write_text(VCF)
    assert len(ng.read_vcf(str(vcf))) == 4
    with pytest.raises(TypeError, match="unexpected keyword argument 'chunk_size'"):
        ng.read_vcf(str(vcf), chunk_size=2)


def test_max_variants_is_the_parameter_that_actually_limits_the_read(tmp_path):
    """Positive control: the sibling option the phantom replaced does work — and
    0 now means zero records instead of 'no limit' (the falsy check used to read
    the whole file)."""
    vcf = tmp_path / "v.vcf"
    vcf.write_text(VCF)
    assert [int(p) for p in ng.read_vcf(str(vcf), max_variants=2)['POS']] == [1, 5]
    assert ng.read_vcf(str(vcf), max_variants=0) is None
    assert len(ng.read_vcf(str(vcf), max_variants=4)) == 4
    assert len(ng.read_vcf(str(vcf), max_variants=99)) == 4


def test_a_missing_p_value_is_no_bar_instead_of_the_tallest_one():
    """fillna(0) made an empty or unparsable p-value -log10(1e-300) = 300, so a
    broken cell towered over a genuine hit at 9.0.  It is NaN now and leaves the
    plot."""
    df = pd.DataFrame({'CHROM': ['chr1', 'chr1', 'chr2', 'chr3'],
                       'POS': [1, 5, 2, 7],
                       'QUAL': [1e-9, None, 'abc', 0.4]})
    out = ng.manhattan_from_vcf(df)
    assert out['neg_log10'].to_numpy() == pytest.approx(
        [9.0, np.nan, np.nan, 0.39794], abs=1e-5, nan_ok=True)
    assert np.isfinite(out['neg_log10']).sum() == 2
    assert out['neg_log10'].max() == pytest.approx(9.0)


def test_a_real_zero_p_value_still_reads_as_the_floor():
    """A genuine p = 0 is not a broken cell: the 1e-300 floor keeps it finite."""
    df = pd.DataFrame({'CHROM': ['chr1'], 'POS': [1], 'QUAL': [0.0]})
    assert ng.manhattan_from_vcf(df)['neg_log10'].iloc[0] == pytest.approx(300.0)


def test_a_negative_p_value_cannot_produce_a_complex_logarithm():
    df = pd.DataFrame({'CHROM': ['chr1', 'chr2'], 'POS': [1, 2], 'QUAL': [-0.5, 1e-9]})
    out = ng.manhattan_from_vcf(df)
    assert out['neg_log10'].iloc[0] == pytest.approx(300.0), "clipped to zero, then floored"
    assert out['neg_log10'].iloc[1] == pytest.approx(9.0)


def test_pvalue_col_selects_the_column_it_names():
    df = pd.DataFrame({'CHROM': ['chr1', 'chr2'], 'POS': [1, 2],
                       'QUAL': [60.0, 60.0], 'PVALUE': [1e-12, 0.5]})
    by_p = ng.manhattan_from_vcf(df, pvalue_col='PVALUE')
    assert by_p['neg_log10'].tolist() == pytest.approx([12.0, 0.30103], abs=1e-5)
    # QUAL is a Phred quality, not a probability: -log10(60) is negative, which is
    # exactly why pointing the plot at a real p-value column matters.
    assert ng.manhattan_from_vcf(df)['neg_log10'].tolist() == pytest.approx(
        [-1.778151, -1.778151], abs=1e-5)
    assert list(ng.manhattan_from_vcf(df).columns) == ['CHROM', 'POS', 'neg_log10']


def test_rows_are_ordered_by_chromosome_number_then_position():
    """Sorting on the label string put chr10 between chr1 and chr2, so the x-axis
    jumped back and forth across the genome.  Numbered chromosomes come first in
    numeric order, then X, Y and the mitochondrion, then unplaced names
    alphabetically."""
    df = pd.DataFrame({'CHROM': ['chr10', 'chr2', 'chr2', 'chr1', 'chrX', 'chrMT',
                                 'chrY', 'unplaced_1', '2L'],
                       'POS': [5, 9, 1, 3, 7, 4, 2, 6, 8],
                       'QUAL': [1e-9] * 9})
    out = ng.manhattan_from_vcf(df)
    assert list(zip(out['CHROM'], out['POS'])) == [
        ('chr1', 3), ('chr2', 1), ('chr2', 9), ('chr10', 5),
        ('chrX', 7), ('chrY', 2), ('chrMT', 4), ('2L', 8), ('unplaced_1', 6)]


def test_manhattan_no_longer_accepts_a_threshold_it_cannot_use():
    df = pd.DataFrame({'CHROM': ['chr1'], 'POS': [1], 'QUAL': [1e-9]})
    with pytest.raises(TypeError, match="unexpected keyword argument 'threshold'"):
        ng.manhattan_from_vcf(df, threshold=0.05)
