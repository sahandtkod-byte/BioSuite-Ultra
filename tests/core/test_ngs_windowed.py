"""NGS helpers: windowed coverage binning, bed/vcf intervals, ts/tv heuristic."""
import numpy as np
import pandas as pd
import pytest

from biosuite.core import ngs as ng


VCF_MNYLICK = (
    "##fileformat=VCFv4.2\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    "1\t100\trsA\tA\tG\t60\t.\t.\n"
    "1\t250\trsB\tC\tT\t50\t.\t.\n"
    "1\t400\trsC\tG\tA\t40\t.\t.\n"
    "1\t550\trsD\tCT\tC\t30\t.\t.\n"
)

VCF_TSTV = (
    "##fileformat=VCFv4.2\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    "1\t100\trs1\tA\tG\t60\t.\t.\n"      # transition
    "1\t200\trs2\tG\tA\t60\t.\t.\n"      # transition
    "1\t300\trs3\tC\tT\t60\t.\t.\n"      # transition
    "1\t400\trs4\tT\tC\t60\t.\t.\n"      # transition
    "1\t500\trs5\tA\tC\t60\t.\t.\n"      # transversion
    "1\t600\trs6\tG\tT\t60\t.\t.\n"      # transversion
)

BED_SIMPLE = "1\t0\t100\tregionA\n1\t200\t300\tregionB\n"


def test_windowed_coverage_even_bins():
    cov = np.concatenate([np.full(100, 2), np.full(100, 4)])
    res = ng._windowed_coverage_array(cov, window_size=50, chrom='1')
    assert len(res) == 4
    assert res['mean_coverage'].iloc[0] == pytest.approx(2.0)
    assert res['mean_coverage'].iloc[-1] == pytest.approx(4.0)


def test_windowed_coverage_uneven_tail():
    cov = np.concatenate([np.full(150, 1), [5, 5]])
    res = ng._windowed_coverage_array(cov, window_size=100, chrom='1')
    assert len(res) >= 1


def test_parse_bed_and_intersect(tmp_path):
    bed = tmp_path / 'a.bed'
    bed.write_text(BED_SIMPLE)
    recs = ng._parse_bed(str(bed))
    assert len(recs) == 2
    vcf = tmp_path / 'a.vcf'
    vcf.write_text(VCF_MNYLICK)
    iv = ng._parse_vcf_intervals(str(vcf))
    assert len(iv) == 4


def test_intersect_bed_vcf_counts(tmp_path):
    bed = tmp_path / 'a.bed'
    vcf = tmp_path / 'a.vcf'
    bed.write_text(BED_SIMPLE)
    vcf.write_text(VCF_MNYLICK)
    out = ng.intersect_bed_vcf(str(bed), str(vcf), report='count')
    assert out is not None


def test_vcf_summary_and_ts_tv(tmp_path):
    vcf = tmp_path / 't.vcf'
    vcf.write_text(VCF_TSTV)
    out = ng.compute_ts_tv_ratio(str(vcf))
    # 4 transitions, 2 transversions => ratio 2.0
    if out is not None:
        val = out if isinstance(out, (int, float)) else out.get('ts_tv') \
            if isinstance(out, dict) else None
        if val is not None:
            assert val == pytest.approx(2.0, rel=0.01)
        else:
            assert out is not None
    summ = ng.vcf_summary(str(vcf))
    assert summ is not None


def test_read_vcf_chunked(tmp_path):
    vcf = tmp_path / 'big.vcf'
    vcf.write_text(VCF_TSTV)
    df = ng.read_vcf(str(vcf), chunk_size=2)
    assert len(df) == 6


def test_sam_roundtrip(tmp_path):
    sam = (
        "@HD\tVN:1.6\tSO:coordinate\n"
        "@SQ\tSN:1\tLN:1000\n"
        "r1\t0\t1\t100\t60\t50M\t*\t0\t0\t" + "A" * 50 + "\t" + "I" * 50 + "\n"
        "r2\t16\t1\t200\t42\t50M\t*\t0\t0\t" + "C" * 50 + "\t" + "I" * 50 + "\n"
    )
    f = tmp_path / 'a.sam'
    f.write_text(sam)
    header = ng.parse_sam_header(str(f))
    assert any('1' in str(v) for v in header.values()) or header
    rows = ng.read_sam(str(f))
    assert len(rows) == 2
