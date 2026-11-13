"""Pins the failure contracts and pysam-gated arithmetic of core/ngs.py.

Every NGS helper here is expected to degrade to ``None`` (with a log line)
rather than raise, and the coverage arithmetic is the only thing a genome
browser trusts, so each window slice, per-base sum, interval intersection and
Ts/Tv count is asserted against an independently computed value instead of a
shape check.
"""
import importlib
import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest

from biosuite.core import ngs

# ── fixtures ─────────────────────────────────────────────────────────────────

SAM_MIXED = (
    "@HD\tVN:1.6\tSO:coordinate\n"
    "@SQ\tSN:chr1\tLN:1000\n"
    "@SQ\tSN:chr2\tLN:500\n"
    "@RG\tID:rg1\tSM:sampleA\tLB:lib1\n"
    "@RG\tID:rg2\tSM:sampleB\n"
    "@PG\tID:bwa\tPN:bwa mem\tVN:0.7.17\n"
    "@CO\tcomment lines are kept raw and classified nowhere\n"
    "r1\t0\tchr1\t1\t60\t10M\t*\t0\t0\tACGTACGTAC\tFFFFJJJJJJ\tNM:i:0\tXX:zy:hello:world\tBROKEN\n"
    "r2\t16\tchr1\t10\t30\t10M\t*\t0\t0\tACGTACGTAC\tFFFFJ\n"
    "truncated\t0\tchr1\n"
)

SAM_HEADER_ONLY = (
    "@HD\tVN:1.6\tSO:unsorted\n"
    "@SQ\tSN:chrM\tLN:16569\n"
)


@pytest.fixture()
def sam_file(tmp_path):
    p = tmp_path / "mixed.sam"
    p.write_text(SAM_MIXED, encoding="utf-8")
    return str(p)


@pytest.fixture()
def header_only_sam(tmp_path):
    p = tmp_path / "header.sam"
    p.write_text(SAM_HEADER_ONLY, encoding="utf-8")
    return str(p)


@pytest.fixture()
def unreadable_path(tmp_path):
    """A path open() refuses to read as text: a directory."""
    d = tmp_path / "a_directory"
    d.mkdir()
    return str(d)


# ── a pysam stand-in ─────────────────────────────────────────────────────────

class _FakeBam:
    """Mimics the slice of pysam.AlignmentFile that ngs.py touches.

    ``per_base`` maps reference -> 1-D array of A allele counts; pysam returns
    a 4-tuple (A,C,G,T) so the fake spreads the same counts over A and zeroes
    the other three, which makes ``np.sum(count_coverage(...), axis=0)``
    directly comparable with ``per_base``.
    """

    per_base: dict = {}
    lengths: dict = {}
    header_value: dict = {}
    open_error: Exception | None = None
    count_error: Exception | None = None
    bad_reference: str | None = None

    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        if type(self).open_error is not None:
            raise type(self).open_error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    @property
    def header(self):
        return type(self).header_value

    @property
    def references(self):
        return list(type(self).lengths)

    def get_reference_length(self, name):
        return type(self).lengths[name]

    def _slice(self, chrom, start=None, end=None):
        if type(self).count_error is not None:
            raise type(self).count_error
        if chrom == type(self).bad_reference:
            raise ValueError(f"region {chrom} is not in the BAM header")
        full = type(self).per_base[chrom]
        part = full if start is None else full[start - 1:end]
        empty = np.zeros_like(part)
        return [part.astype(float), empty, empty.copy(), empty.copy()]

    def count_coverage(self, region=None, reference=None):
        if region is not None:
            chrom, _, span = region.partition(':')
            first, _, last = span.partition('-')
            return self._slice(chrom, int(first), int(last))
        return self._slice(reference)


def _install_pysam(monkeypatch, *, per_base, lengths, header=None,
                   open_error=None, count_error=None, bad_reference=None):
    fake = type("_Pysam", (), {"AlignmentFile": _FakeBam})()
    _FakeBam.per_base = per_base
    _FakeBam.lengths = lengths
    _FakeBam.header_value = header or {'HD': {'VN': '1.6', 'SO': 'coordinate'}}
    _FakeBam.open_error = open_error
    _FakeBam.count_error = count_error
    _FakeBam.bad_reference = bad_reference
    monkeypatch.setattr(ngs, "pysam", fake, raising=False)
    monkeypatch.setattr(ngs, "HAS_PYSAM", True)
    return fake


# ── read_bam_header ──────────────────────────────────────────────────────────

def test_read_bam_header_refuses_without_pysam(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(ngs, "HAS_PYSAM", False)
    monkeypatch.delattr(ngs, "pysam", raising=False)
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"BAM\x01")
    with caplog.at_level(logging.WARNING, logger="biosuite.core.ngs"):
        assert ngs.read_bam_header(str(bam)) is None, \
            "a build without pysam must report unavailable, not pretend the BAM is empty"
    assert "pysam not installed" in caplog.text


def test_read_bam_header_returns_the_bam_header_dict(monkeypatch, tmp_path):
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"BAM\x01")
    _install_pysam(monkeypatch, per_base={}, lengths={},
                   header={'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chr1', 'LN': 8}]})
    assert ngs.read_bam_header(str(bam)) == {'HD': {'VN': '1.6'},
                                             'SQ': [{'SN': 'chr1', 'LN': 8}]}


def test_read_bam_header_missing_file_is_none(monkeypatch, tmp_path):
    _install_pysam(monkeypatch, per_base={}, lengths={})
    assert ngs.read_bam_header(str(tmp_path / "nope.bam")) is None


def test_read_bam_header_reports_an_unreadable_bam_as_none(monkeypatch, tmp_path):
    bam = tmp_path / "broken.bam"
    bam.write_bytes(b"BAM\x01")
    _install_pysam(monkeypatch, per_base={}, lengths={},
                   open_error=OSError("truncated BGZF block"))
    assert ngs.read_bam_header(str(bam)) is None, \
        "AlignmentFile failures have to degrade to None, not escape to the GUI"


def test_import_probe_marks_pysam_present(monkeypatch):
    """The HAS_PYSAM flag is decided by the import, both of its arms."""
    fake = types.ModuleType("pysam")
    fake.AlignmentFile = _FakeBam
    monkeypatch.setitem(sys.modules, "pysam", fake)
    try:
        importlib.reload(ngs)
        assert ngs.HAS_PYSAM is True, "a working pysam import must set HAS_PYSAM"
        assert ngs.pysam is fake
    finally:
        sys.modules.pop("pysam", None)
        importlib.reload(ngs)
        assert ngs.HAS_PYSAM is False
        monkeypatch.delattr(ngs, "pysam", raising=False)


# ── read_vcf ─────────────────────────────────────────────────────────────────

def test_read_vcf_ignores_records_that_do_not_reach_the_info_column(tmp_path):
    p = tmp_path / "short.vcf"
    p.write_text("##fileformat=VCFv4.2\n"
                 "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                 "chr1\t100\trs1\tA\tG\t99\tPASS\tAF=0.2\n"
                 "chr1\t101\trs2\tA\tG\n"
                 "chr1\t102\trs3\tA\tG\t50\tPASS\tAF=0.1\n",
                 encoding="utf-8")
    df = ngs.read_vcf(str(p))
    assert list(df["POS"]) == [100, 102], "7-field records are not VCF data"
    assert df["POS"].dtype.kind == "i", "POS must stay an integer column"
    assert df["QUAL"].tolist() == [99.0, 50.0]


def test_read_vcf_warns_once_the_file_reaches_100k_variants(tmp_path_factory, caplog):
    root = tmp_path_factory.mktemp("huge_vcf")
    p = root / "big.vcf"
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("##fileformat=VCFv4.2\n"
                 "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for i in range(1, 100_002):
            fh.write(f"chr1\t{i}\trs{i}\tA\tG\t50\tPASS\tAF=0.1\n")
    with caplog.at_level(logging.WARNING, logger="biosuite.core.ngs"):
        df = ngs.read_vcf(str(p))
    assert len(df) == 100_001
    fired = [m for m in caplog.messages if "100K+" in m]
    assert fired == ["Warning: VCF has 100K+ variants. Consider using max_variants parameter."], \
        "the 100k advisory must fire exactly once, not per record"


def test_read_vcf_returns_none_when_no_record_lines_are_present(tmp_path):
    p = tmp_path / "meta.vcf"
    p.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n",
                 encoding="utf-8")
    assert ngs.read_vcf(str(p)) is None


def test_read_vcf_returns_none_when_the_path_cannot_be_opened(unreadable_path):
    assert ngs.read_vcf(unreadable_path) is None, \
        "an unreadable path must not become an empty DataFrame"


def test_read_vcf_missing_file_is_none(tmp_path):
    missing = str(tmp_path / "ghost.vcf")
    assert ngs.read_vcf(missing) is None, \
        "an absent VCF is reported as unavailable, not as zero variants"


def test_read_vcf_max_variants_stops_at_the_requested_count(tmp_path):
    p = tmp_path / "n.vcf"
    p.write_text("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" +
                 "".join(f"chr1\t{i}\trs{i}\tA\tG\t50\tPASS\t.\n" for i in range(1, 11)),
                 encoding="utf-8")
    assert len(ngs.read_vcf(str(p), max_variants=4)) == 4
    assert len(ngs.read_vcf(str(p), max_variants=10)) == 10
    assert len(ngs.read_vcf(str(p), max_variants=11)) == 10, "more than exist is all of them"
    assert ngs.read_vcf(str(p), max_variants=0) is None, \
        "`if max_variants` made 0 falsy and read the whole file; it now means zero"


# ── manhattan_from_vcf ───────────────────────────────────────────────────────

def test_manhattan_from_vcf_uses_the_column_it_was_given():
    frame = pd.DataFrame({'CHROM': ['chr1', 'chr1'], 'POS': [1, 2],
                          'QUAL': [50.0, 60.0], 'PVALUE': [1e-9, 1e-3]})
    got = ngs.manhattan_from_vcf(frame, pvalue_col='PVALUE')['neg_log10'].tolist()
    expected = list(-np.log10(np.array([1e-9, 1e-3]) + 1e-300))
    assert got == pytest.approx(expected), \
        "pvalue_col used to be ignored, so QUAL was plotted as if it were a p-value"
    assert got == pytest.approx([9.0, 3.0])


def test_manhattan_from_vcf_defaults_to_qual_and_leaves_a_missing_value_unplotted():
    """A genuine p = 0 lands on the 1e-300 floor (-log10 = 300); a missing value is
    NaN and drops out of the plot.  fillna(0) used to hand the missing one the same
    300, so a broken cell became the tallest bar on the chart."""
    frame = pd.DataFrame({'CHROM': ['chr1', 'chr2'], 'POS': [5, 6],
                          'QUAL': [0.0, np.nan]})
    out = ngs.manhattan_from_vcf(frame)
    assert out['neg_log10'].iloc[0] == pytest.approx(300.0)
    assert np.isnan(out['neg_log10'].iloc[1])
    assert list(out.columns) == ['CHROM', 'POS', 'neg_log10']


def test_manhattan_from_vcf_orders_chromosomes_numerically_then_x_y_mt():
    """'chr10' sorts before 'chr2' as a string, which made the x-axis jump back
    across the genome; numbered chromosomes now come first in numeric order, then
    X, Y and MT."""
    frame = pd.DataFrame({'CHROM': ['chr10', 'chr2', 'chr1', 'chrX', 'chrMT', 'chrY'],
                          'POS': [4, 3, 2, 1, 1, 1], 'QUAL': [1e-4] * 6})
    assert list(ngs.manhattan_from_vcf(frame)['CHROM']) == \
        ['chr1', 'chr2', 'chr10', 'chrX', 'chrY', 'chrMT']


# ── SAM parsing ──────────────────────────────────────────────────────────────

def test_read_sam_keeps_valid_records_and_three_part_tags(sam_file):
    reads = ngs.read_sam(sam_file)
    assert [r['QNAME'] for r in reads] == ['r1', 'r2'], \
        "a 3-field line is not an alignment record and must be skipped, not fatal"
    assert reads[0]['FLAG'] == 0 and reads[1]['FLAG'] == 16
    assert reads[0]['POS'] == 1 and reads[0]['TLEN'] == 0
    assert reads[0]['NM'] == '0'
    assert reads[0]['XX'] == 'hello:world', "tag values may themselves contain ':'"
    assert 'BROKEN' not in reads[0], "a 2-part tag field carries no value to store"


def test_read_sam_missing_file_is_none(tmp_path):
    assert ngs.read_sam(str(tmp_path / "ghost.sam")) is None


def test_read_sam_returns_none_when_a_numeric_field_is_not_numeric(tmp_path):
    p = tmp_path / "bad.sam"
    p.write_text("@HD\tVN:1.6\nr1\t0\tchr1\t-\t60\t10M\t*\t0\t0\tACGT\tIIII\n",
                 encoding="utf-8")
    assert ngs.read_sam(str(p)) is None, \
        "one unparsable POS used to be reported as 'no reads at all'"


def test_read_sam_returns_none_when_the_path_cannot_be_opened(unreadable_path):
    assert ngs.read_sam(unreadable_path) is None


def test_parse_sam_header_groups_every_header_record_type(sam_file):
    header = ngs.parse_sam_header(sam_file)
    assert header['@HD'] == {'VN': '1.6', 'SO': 'coordinate'}
    assert header['@SQ'] == [{'SN': 'chr1', 'LN': '1000'}, {'SN': 'chr2', 'LN': '500'}]
    assert header['@RG'] == [{'ID': 'rg1', 'SM': 'sampleA', 'LB': 'lib1'},
                             {'ID': 'rg2', 'SM': 'sampleB'}]
    assert header['@PG'] == [{'ID': 'bwa', 'PN': 'bwa mem', 'VN': '0.7.17'}]
    assert len(header['raw']) == 7, "raw must hold the header lines verbatim"
    assert header['raw'][0] == "@HD\tVN:1.6\tSO:coordinate"
    assert set(header) == {'@HD', '@SQ', '@RG', '@PG', 'raw'}, \
        "an unclassified @CO line belongs to raw only"


def test_parse_sam_header_stops_at_the_first_alignment_record(sam_file):
    header = ngs.parse_sam_header(sam_file)
    assert not any(line.startswith('r1\t') for line in header['raw']), \
        "alignment lines must never leak into the header"


def test_parse_sam_header_reads_a_header_only_file_to_end_of_file(header_only_sam):
    header = ngs.parse_sam_header(header_only_sam)
    assert header['@HD'] == {'VN': '1.6', 'SO': 'unsorted'}
    assert header['@SQ'] == [{'SN': 'chrM', 'LN': '16569'}]
    assert header['@RG'] == [] and header['@PG'] == []


def test_parse_sam_header_missing_file_is_none(tmp_path):
    assert ngs.parse_sam_header(str(tmp_path / "ghost.sam")) is None


def test_parse_sam_header_returns_none_when_the_path_cannot_be_opened(unreadable_path):
    assert ngs.parse_sam_header(unreadable_path) is None


def test_parse_sam_header_returns_none_when_a_header_field_has_no_value(tmp_path):
    p = tmp_path / "malformed.sam"
    p.write_text("@HD\tVN1.6\n@SQ\tSN:chr1\tLN:100\n", encoding="utf-8")
    assert ngs.parse_sam_header(str(p)) is None, \
        "a '@HD' tag without a colon separator cannot be classified"


def test_parse_sam_read_returns_none_for_short_and_unparsable_lines():
    assert ngs.parse_sam_read("r1\t0\tchr1\t10") is None, \
        "fewer than 11 mandatory fields is not a SAM record"
    assert ngs.parse_sam_read("r1\tnotanint\tchr1\t10\t60\t10M\t*\t0\t0\tACGT\tIIII") is None
    rec = ngs.parse_sam_read("r1\t0\tchr1\t10\t60\t10M\t*\t0\t0\tACGT\tIIII\tNM:i:0\tJUNK\n")
    assert rec['QNAME'] == 'r1' and rec['POS'] == 10 and rec['NM'] == '0'
    assert 'JUNK' not in rec


# ── compute_coverage ─────────────────────────────────────────────────────────

def test_compute_coverage_region_is_the_one_based_inclusive_slice(monkeypatch, tmp_path):
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"BAM\x01")
    per_base = np.arange(1, 9, dtype=float)
    _install_pysam(monkeypatch, per_base={'chr1': per_base}, lengths={'chr1': 8})
    cov = ngs.compute_coverage(str(bam), region='chr1:3-6')
    assert cov.tolist() == per_base[2:6].tolist(), \
        "region chr1:3-6 must map to zero-based slice [2:6]"


def test_compute_coverage_region_windowed_uses_exact_windows(monkeypatch, tmp_path):
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"BAM\x01")
    per_base = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
    _install_pysam(monkeypatch, per_base={'chr1': per_base}, lengths={'chr1': 8})
    df = ngs.compute_coverage(str(bam), region='chr1:1-8', window_size=4)
    assert list(df.columns) == ['start', 'end', 'mean_coverage'], \
        "a region query has no chrom column - only whole-genome queries carry one"
    assert df['mean_coverage'].tolist() == [5.0, 13.0], \
        "window means must come from non-overlapping windows of 4"
    assert df['start'].tolist() == [0, 4] and df['end'].tolist() == [4, 8]


def test_compute_coverage_whole_genome_keys_by_reference(monkeypatch, tmp_path):
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"BAM\x01")
    per_base = {'chr1': np.array([1, 2, 3, 4], dtype=float),
                'chrZero': np.array([], dtype=float),
                'chr2': np.array([5, 5, 5, 5], dtype=float)}
    _install_pysam(monkeypatch, per_base=per_base,
                   lengths={'chr1': 4, 'chrZero': 0, 'chr2': 4})
    result = ngs.compute_coverage(str(bam))
    assert sorted(result) == ['chr1', 'chr2'], \
        "a zero-length reference has no bases to cover and is skipped"
    assert result['chr1'].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert result['chr2'].tolist() == [5.0, 5.0, 5.0, 5.0]


def test_compute_coverage_whole_genome_windowed_carries_chrom(monkeypatch, tmp_path):
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"BAM\x01")
    per_base = {'chr1': np.array([1, 1, 1, 1, 9, 9, 9, 9], dtype=float),
                'chr2': np.array([2, 2], dtype=float)}
    _install_pysam(monkeypatch, per_base=per_base, lengths={'chr1': 8, 'chr2': 2})
    result = ngs.compute_coverage(str(bam), window_size=4)
    assert result['chr1']['mean_coverage'].tolist() == [1.0, 9.0]
    assert result['chr1']['chrom'].tolist() == ['chr1', 'chr1']
    assert list(result['chr2'].columns) == ['chrom', 'start', 'end', 'mean_coverage'], \
        "chr2 is shorter than the window; its frame must still carry 'chrom'"
    assert result['chr2']['mean_coverage'].tolist() == [2.0]


def test_compute_coverage_refuses_without_pysam(monkeypatch, tmp_path):
    monkeypatch.setattr(ngs, "HAS_PYSAM", False)
    monkeypatch.delattr(ngs, "pysam", raising=False)
    assert ngs.compute_coverage(str(tmp_path / "x.bam")) is None


def test_compute_coverage_missing_file_is_none(monkeypatch, tmp_path):
    _install_pysam(monkeypatch, per_base={}, lengths={})
    assert ngs.compute_coverage(str(tmp_path / "ghost.bam")) is None


def test_compute_coverage_reports_a_failure_as_none(monkeypatch, tmp_path):
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"BAM\x01")
    _install_pysam(monkeypatch, per_base={'chr1': np.ones(4)}, lengths={'chr1': 4},
                   count_error=ValueError("no index file"))
    assert ngs.compute_coverage(str(bam), region='chr1:1-4') is None


def test_compute_coverage_window_size_one_returns_per_base(monkeypatch, tmp_path):
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"BAM\x01")
    per_base = np.array([3, 3, 4, 4], dtype=float)
    _install_pysam(monkeypatch, per_base={'chr1': per_base}, lengths={'chr1': 4})
    assert ngs.compute_coverage(str(bam), region='chr1:1-4',
                                window_size=1).tolist() == [3.0, 3.0, 4.0, 4.0]


# ── _windowed_coverage_array ─────────────────────────────────────────────────

def test_windowed_coverage_without_a_chrom_omits_the_column():
    cov = np.array([2, 4, 6, 8], dtype=float)
    df = ngs._windowed_coverage_array(cov, 2)
    assert list(df.columns) == ['start', 'end', 'mean_coverage']
    assert df['mean_coverage'].tolist() == [3.0, 7.0]
    assert df['end'].tolist() == [2, 4]


def test_windowed_coverage_of_a_sub_window_span_without_a_chrom():
    cov = np.array([2.0, 4.0, 6.0])
    df = ngs._windowed_coverage_array(cov, 10)
    assert list(df.columns) == ['start', 'end', 'mean_coverage'], \
        "no chrom was asked for, so no chrom column appears - even for one span"
    assert df['start'].tolist() == [0] and df['end'].tolist() == [3]
    assert df['mean_coverage'].tolist() == [pytest.approx(4.0)]


def test_windowed_coverage_drops_the_unfilled_tail():
    cov = np.full(5, 4.0)
    df = ngs._windowed_coverage_array(cov, 2, chrom='chr1')
    assert len(df) == 2, "3 bases do not make a window of 2 - the tail is dropped"
    assert df['end'].iloc[-1] == 4
    assert df['mean_coverage'].tolist() == [4.0, 4.0]


# ── BED / VCF interval helpers ───────────────────────────────────────────────

def test_parse_bed_keeps_three_columns_and_skips_noise(tmp_path):
    p = tmp_path / "a.bed"
    p.write_text("# a comment\n"
                 "\n"
                 "track name=x description=y\n"
                 "browser url=http://x\n"
                 "chr1\t0\t100\tA\t99\t+\n"
                 "chr2\t50\t60\n"
                 "onlytwo\tfields\n", encoding="utf-8")
    assert ngs._parse_bed(str(p)) == [('chr1', 0, 100), ('chr2', 50, 60)], \
        "BED3-BED12 all share the first three columns; anything else is noise"


def test_parse_vcf_intervals_is_half_open_one_base(tmp_path):
    p = tmp_path / "a.vcf"
    p.write_text("##hdr\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                 "\n"
                 "chr1\t10\trs1\tA\tG\t99\tPASS\t.\n"
                 "singlefieldline\n"
                 "chr2\t55\trs2\tC\tT\t50\tPASS\t.\n", encoding="utf-8")
    assert ngs._parse_vcf_intervals(str(p)) == [('chr1', 9, 10), ('chr2', 54, 55)], \
        "POS is 1-based; the interval must be [POS-1, POS) to be half-open"


def test_intersect_bed_vcf_counts_and_detail_agree(tmp_path):
    bed = tmp_path / "a.bed"
    bed.write_text("chr1\t0\t100\nchr2\t50\t60\nchr1\t300\t400\n", encoding="utf-8")
    vcf = tmp_path / "a.vcf"
    vcf.write_text("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                   "chr1\t10\trs1\tA\tG\t99\tPASS\t.\n"
                   "chr1\t350\trs3\tA\tC\t1\tPASS\t.\n"
                   "chr2\t55\trs2\tC\tT\t50\tPASS\t.\n", encoding="utf-8")
    counts = ngs.intersect_bed_vcf(str(bed), str(vcf), report='count')
    assert counts.tolist() == [1, 1, 1]
    assert counts.name == 'overlap_count'
    detail = ngs.intersect_bed_vcf(str(bed), str(vcf), report='detail')
    assert sorted(detail['vcf_pos'].tolist()) == [10, 55, 350], \
        "detail positions are 1-based again, matching the VCF"
    assert len(detail) == int(counts.sum())
    assert list(detail.columns) == ['bed_chrom', 'bed_start', 'bed_end',
                                    'vcf_chrom', 'vcf_pos']


def test_intersect_bed_vcf_detail_is_empty_when_nothing_overlaps(tmp_path):
    bed = tmp_path / "e.bed"
    bed.write_text("chrZ\t0\t1\n", encoding="utf-8")
    vcf = tmp_path / "e.vcf"
    vcf.write_text("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\t10\t.\tA\tG\t1\t.\t.\n",
                   encoding="utf-8")
    detail = ngs.intersect_bed_vcf(str(bed), str(vcf), report='detail')
    assert len(detail) == 0
    assert ngs.intersect_bed_vcf(str(bed), str(vcf)).tolist() == [0]


def test_intersect_bed_vcf_missing_inputs_are_none(tmp_path, bed_file):
    missing_vcf = str(tmp_path / "no.vcf")
    assert ngs.intersect_bed_vcf(bed_file, missing_vcf) is None
    assert ngs.intersect_bed_vcf(str(tmp_path / "no.bed"), missing_vcf) is None, \
        "the BED check runs first, so a missing BED is reported before the VCF"


@pytest.fixture()
def bed_file(tmp_path):
    p = tmp_path / "r.bed"
    p.write_text("chr1\t0\t100\n", encoding="utf-8")
    return str(p)


def test_intersect_bed_vcf_propagates_a_malformed_coordinate(tmp_path, bed_file):
    """Pinned measured behaviour: parsing errors raise rather than returning None."""
    vcf = tmp_path / "bad.vcf"
    vcf.write_text("chr1\tNA\trs\tA\tG\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid literal"):
        ngs.intersect_bed_vcf(bed_file, str(vcf))
    bad_bed = tmp_path / "bad.bed"
    bad_bed.write_text("chr1\tnope\t300\n", encoding="utf-8")
    good_vcf = tmp_path / "ok.vcf"
    good_vcf.write_text("chr1\t10\t.\tA\tG\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid literal"):
        ngs.intersect_bed_vcf(str(bad_bed), str(good_vcf))


# ── coverage_from_bed ────────────────────────────────────────────────────────

def test_coverage_from_bed_means_every_interval(monkeypatch, tmp_path):
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"BAM\x01")
    per_base = np.array([2, 2, 2, 2, 8, 8, 8, 8], dtype=float)
    _install_pysam(monkeypatch, per_base={'chr1': per_base}, lengths={'chr1': 8})
    bed = tmp_path / "a.bed"
    bed.write_text("chr1\t0\t4\tfirst\nchr1\t4\t8\tsecond\n", encoding="utf-8")
    df = ngs.coverage_from_bed(str(bam), str(bed))
    assert df['mean_coverage'].tolist() == [2.0, 8.0]
    assert df['total_bases'].tolist() == [4, 4]
    assert df['start'].tolist() == [0, 4] and df['end'].tolist() == [4, 8]
    assert df['chrom'].tolist() == ['chr1', 'chr1']


def test_coverage_from_bed_zero_length_interval_averages_nothing(monkeypatch, tmp_path):
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"BAM\x01")
    _install_pysam(monkeypatch, per_base={'chr1': np.array([5.0, 5.0])},
                   lengths={'chr1': 2})
    bed = tmp_path / "z.bed"
    bed.write_text("chr1\t4\t4\tempty\n", encoding="utf-8")
    df = ngs.coverage_from_bed(str(bam), str(bed))
    assert df['mean_coverage'].tolist() == [0.0], \
        "an empty slice reports 0.0 rather than NaN"
    assert df['total_bases'].tolist() == [0]


def test_coverage_from_bed_logs_an_unmappable_interval(monkeypatch, tmp_path):
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"BAM\x01")
    _install_pysam(monkeypatch,
                   per_base={'chr1': np.array([1.0, 2.0]), 'chrZ': np.array([0.0])},
                   lengths={'chr1': 2, 'chrZ': 1}, bad_reference='chrZ')
    bed = tmp_path / "m.bed"
    bed.write_text("chr1\t0\t2\tknown\nchrZ\t0\t1\tunknown_contig\n", encoding="utf-8")
    captured = []

    class _Grab(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    logger = logging.getLogger("biosuite.core.ngs")
    handler = _Grab()
    logger.addHandler(handler)
    try:
        df = ngs.coverage_from_bed(str(bam), str(bed))
    finally:
        logger.removeHandler(handler)
    assert df['mean_coverage'].tolist() == [1.5, 0.0], \
        "a failure still reports 0.0 so the BED row count is preserved"
    assert len(captured) == 1 and "chrZ:1-1" in captured[0], \
        "the interval that could not be covered has to be named in the log"


def test_coverage_from_bed_refuses_without_pysam(monkeypatch, tmp_path):
    monkeypatch.setattr(ngs, "HAS_PYSAM", False)
    monkeypatch.delattr(ngs, "pysam", raising=False)
    assert ngs.coverage_from_bed("x.bam", "y.bed") is None


def test_coverage_from_bed_checks_its_inputs(monkeypatch, tmp_path):
    _install_pysam(monkeypatch, per_base={}, lengths={})
    bed = tmp_path / "a.bed"
    bed.write_text("chr1\t0\t2\n", encoding="utf-8")
    missing_bam = str(tmp_path / "ghost.bam")
    assert ngs.coverage_from_bed(missing_bam, str(bed)) is None
    assert ngs.coverage_from_bed(str(bed), str(tmp_path / "ghost.bed")) is None, \
        "the BAM check passes first only when the BAM really exists"


def test_coverage_from_bed_returns_none_when_the_bam_will_not_open(monkeypatch, tmp_path):
    bam = tmp_path / "x.bam"
    bam.write_bytes(b"BAM\x01")
    _install_pysam(monkeypatch, per_base={'chr1': np.ones(4)}, lengths={'chr1': 4},
                   open_error=OSError("not a BAM file"))
    bed = tmp_path / "a.bed"
    bed.write_text("chr1\t0\t4\n", encoding="utf-8")
    assert ngs.coverage_from_bed(str(bam), str(bed)) is None


# ── VCF statistics ───────────────────────────────────────────────────────────

VCF_MIXED = (
    "##fileformat=VCFv4.2\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    "chr1\t100\trs1\tA\tG\t99\tPASS\t.\n"        # Ti
    "chr1\t150\trs2\tC\tT\t50\tPASS\t.\n"        # Ti
    "chr1\t200\trs3\tA\tT\t10\tq10\t.\n"         # Tv
    "chr2\t250\t.\tAT\tA\t40\tPASS\t.\n"         # deletion
    "chr2\t300\trs5\tA\tC,G\t.\tPASS\tMULTI\n"   # multi-allelic, QUAL '.'
    "truncated\trecord\n"
)


def _independent_vcf_stats(text):
    """Recount Ts/Tv and quality stats straight from the text, without ngs.py."""
    purines, pyrimidines = set('AG'), set('CT')
    ti = tv = snp = indel = multi = 0
    quals = []
    per_chrom = {}
    for line in text.splitlines():
        if line.startswith('#'):
            continue
        fields = line.split('\t')
        if len(fields) < 6:
            continue
        per_chrom[fields[0]] = per_chrom.get(fields[0], 0) + 1
        try:
            quals.append(float(fields[5]))
        except ValueError:
            pass
        alts = fields[4].split(',')
        if len(alts) > 1:
            multi += 1
        for alt in alts:
            if len(fields[3]) == 1 and len(alt) == 1:
                snp += 1
                if (fields[3] in purines and alt in purines) or \
                   (fields[3] in pyrimidines and alt in pyrimidines):
                    ti += 1
                else:
                    tv += 1
            else:
                indel += 1
    return {'snp': snp, 'indel': indel, 'multi': multi, 'ti': ti, 'tv': tv,
            'per_chrom': per_chrom, 'quals': quals}


def test_vcf_summary_matches_an_independent_recount(tmp_path):
    p = tmp_path / "mixed.vcf"
    p.write_text(VCF_MIXED, encoding="utf-8")
    summ = ngs.vcf_summary(str(p))
    truth = _independent_vcf_stats(VCF_MIXED)
    assert truth['quals'] == [99.0, 50.0, 10.0, 40.0], \
        "the '.' QUAL must be unreadable-as-float, not silently zero"
    assert summ['snp_count'] == truth['snp'] == 5
    assert summ['indel_count'] == truth['indel'] == 1
    assert summ['multi_allelic_count'] == truth['multi'] == 1
    assert summ['ti_count'] == truth['ti'] == 3, "A>G, C>T and the A>G of the multi-allelic"
    assert summ['tv_count'] == truth['tv'] == 2, "A>T and A>C"
    assert summ['per_chrom'] == truth['per_chrom'] == {'chr1': 3, 'chr2': 2}
    assert summ['total_variants'] == sum(truth['per_chrom'].values()) == 5
    assert summ['qual_mean'] == pytest.approx(np.mean(truth['quals']))
    assert summ['qual_median'] == pytest.approx(np.median(truth['quals']))
    assert summ['qual_gt_20_count'] == 3 == sum(q > 20 for q in truth['quals'])
    assert summ['ti_tv_ratio'] == pytest.approx(3 / 2, rel=1e-9)


def test_vcf_summary_skips_a_record_that_carries_no_quality_field(tmp_path):
    """Regression: a 5-field line used to raise IndexError and discard the file."""
    p = tmp_path / "five.vcf"
    p.write_text("##hdr\n#CHROM\tPOS\tID\tREF\tALT\n"
                 "chr1\t100\trs1\tA\tG\n"
                 "chr1\t200\trs2\tC\tT\t60\tPASS\t.\n", encoding="utf-8")
    summ = ngs.vcf_summary(str(p))
    assert summ['total_variants'] == 1, "only the complete record is classifiable"
    assert summ['per_chrom'] == {'chr1': 1}
    assert summ['qual_mean'] == pytest.approx(60.0)
    assert summ['ti_tv_ratio'] == float('inf'), "0 transversions means an infinite ratio"


def test_vcf_summary_on_an_empty_but_readable_vcf(tmp_path):
    p = tmp_path / "none.vcf"
    p.write_text("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n", encoding="utf-8")
    summ = ngs.vcf_summary(str(p))
    assert summ['total_variants'] == 0
    assert summ['per_chrom'] == {}
    assert summ['qual_mean'] == 0.0 and summ['qual_median'] == 0.0, \
        "with no QUAL column the reported mean is the 0.0 sentinel, not NaN"
    assert summ['ti_tv_ratio'] == float('inf')
    assert summ['qual_gt_20_count'] == 0


def test_vcf_summary_missing_file_is_none(tmp_path):
    assert ngs.vcf_summary(str(tmp_path / "ghost.vcf")) is None


def test_vcf_summary_returns_none_for_an_unreadable_path(unreadable_path):
    assert ngs.vcf_summary(unreadable_path) is None


def test_compute_ts_tv_ratio_matches_an_independent_recount(tmp_path):
    p = tmp_path / "tstv.vcf"
    p.write_text(VCF_MIXED, encoding="utf-8")
    got = ngs.compute_ts_tv_ratio(str(p))
    truth = _independent_vcf_stats(VCF_MIXED)
    assert got == {'transitions': truth['ti'], 'transversions': truth['tv'],
                   'ratio': pytest.approx(truth['ti'] / truth['tv'])}
    assert got['transitions'] == 3 and got['transversions'] == 2
    assert got['ratio'] == pytest.approx(1.5, abs=1e-4)


def test_compute_ts_tv_ratio_is_inf_without_transversions(tmp_path):
    p = tmp_path / "ti.vcf"
    p.write_text("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                 "chr1\t100\t.\tA\tG\t9\tPASS\t.\n", encoding="utf-8")
    assert ngs.compute_ts_tv_ratio(str(p)) == {'transitions': 1,
                                               'transversions': 0,
                                               'ratio': float('inf')}


def test_compute_ts_tv_ratio_skips_records_without_alt_and_ref(tmp_path):
    p = tmp_path / "short.vcf"
    p.write_text("#CHROM\tPOS\tID\tREF\nchr1\t100\t.\tA\n"
                 "chr1\t200\t.\tG\tA\t9\tPASS\t.\n", encoding="utf-8")
    got = ngs.compute_ts_tv_ratio(str(p))
    assert got['transitions'] == 1 and got['transversions'] == 0, \
        "a 4-field line has no ALT to classify and must not be counted"


def test_compute_ts_tv_ratio_missing_file_is_none(tmp_path):
    assert ngs.compute_ts_tv_ratio(str(tmp_path / "ghost.vcf")) is None


def test_compute_ts_tv_ratio_returns_none_for_an_unreadable_path(unreadable_path):
    assert ngs.compute_ts_tv_ratio(unreadable_path) is None
