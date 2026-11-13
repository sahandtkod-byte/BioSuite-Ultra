"""Genome browser tracks: the parsers, both coverage paths, and what each track draws.

Everything here reads real files written into ``tmp_path`` (BED, VCF, SAM) and checks
the drawn result — patches, collections, markers, axis limits — instead of only
checking that a figure came back.  ``compute_coverage`` has two paths: ``pysam`` if
it is importable and a text SAM fallback if it is not.  pysam is not installed on
this machine, so the fallback runs for real and the pysam branch is driven with an
injected stand-in module, which is the only way to test this project's own binning
arithmetic against a BAM.
"""
import sys
import types

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import genome_browser as gb


@pytest.fixture(autouse=True)
def _clean_figures():
    plt.close('all')
    yield
    plt.close('all')


def _text_ax():
    fig, ax = plt.subplots()
    return fig, ax


# ─── BED and VCF parsing ─────────────────────────────────────────────────────

def test_parse_bed_reads_name_and_score_when_they_are_there(tmp_path):
    bed = tmp_path / "genes.bed"
    bed.write_text("chr1\t1000\t2000\tgeneA\t55\n", encoding="utf-8")

    assert gb.parse_bed(str(bed)) == [("chr1", 1000, 2000, "geneA", 55.0)]


def test_parse_bed_defaults_a_missing_name_and_score(tmp_path):
    bed = tmp_path / "bare.bed"
    bed.write_text("chr1\t10\t20\nchr2\t30\t40\tname_only\n", encoding="utf-8")

    assert gb.parse_bed(str(bed)) == [
        ("chr1", 10, 20, "", 0.0),
        ("chr2", 30, 40, "name_only", 0.0),
    ]


def test_parse_bed_skips_comments_track_lines_and_blanks(tmp_path):
    bed = tmp_path / "c.bed"
    bed.write_text("# a comment\ntrack type=bed\n\nbrowser line\nchr1\t1\t2\tx\t5\n",
                   encoding="utf-8")

    regions = gb.parse_bed(str(bed))

    assert regions == [("chr1", 1, 2, "x", 5.0)], "the browser line is a viewer directive"


def test_parse_bed_stops_at_max_regions(tmp_path):
    bed = tmp_path / "many.bed"
    bed.write_text("".join(f"chr1\t{i}\t{i + 1}\tg{i}\n" for i in range(5)),
                   encoding="utf-8")

    assert len(gb.parse_bed(str(bed), max_regions=3)) == 3
    assert len(gb.parse_bed(str(bed))) == 5


def test_parse_vcf_reads_qual_filter_and_info(tmp_path):
    vcf = tmp_path / "calls.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n"
                   "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                   "chr1\t1200\t.\tA\tG\t99.5\tPASS\tDP=12\n"
                   "chr1\t3000\trs1\tC\tT\t.\tq0\tDP=3\n", encoding="utf-8")

    variants = gb.parse_vcf(str(vcf))

    assert [v['pos'] for v in variants] == [1200, 3000]
    assert variants[0]['qual'] == 99.5 and variants[0]['info'] == "DP=12"
    assert variants[1]['qual'] == 0.0, "a '.' quality means unreported, not negative"
    assert variants[1]['filter'] == "q0"


def test_parse_vcf_skips_rows_without_ref_and_alt(tmp_path):
    vcf = tmp_path / "short.vcf"
    vcf.write_text("#CHROM\tPOS\tID\tREF\nchr1\t10\t.\tA\nchr1\t20\t.\tA\tT\t1\t.\t\n",
                   encoding="utf-8")

    variants = gb.parse_vcf(str(vcf))

    assert len(variants) == 1 and variants[0]['pos'] == 20


def test_parse_vcf_fills_in_the_columns_a_truncated_row_lacks(tmp_path):
    """QUAL, FILTER and INFO are the tail of the record; a half-written line that
    stops before them is still a call, and must not raise."""
    vcf = tmp_path / "cut.vcf"
    vcf.write_text("chr1\t100\t.\tA\tG\n"
                   "chr1\t200\t.\tA\tG\t99\n"
                   "chr1\t300\t.\tA\tG\t99\tPASS\n", encoding="utf-8")

    variants = gb.parse_vcf(str(vcf))

    assert [v['pos'] for v in variants] == [100, 200, 300]
    assert [v['qual'] for v in variants] == [0.0, 99.0, 99.0]
    assert [v['filter'] for v in variants] == ["", "", "PASS"], "row two stops before FILTER"
    assert [v['info'] for v in variants] == ["", "", ""]


def test_parse_vcf_stops_at_max_variants(tmp_path):
    body = "".join(f"chr1\t{i}\t.\tA\tG\t1\t.\t\n" for i in range(1, 6))
    vcf = tmp_path / "many.vcf"
    vcf.write_text("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + body,
                   encoding="utf-8")

    assert len(gb.parse_vcf(str(vcf), max_variants=2)) == 2


# ─── CIGAR arithmetic ────────────────────────────────────────────────────────

@pytest.mark.parametrize("cigar, expected", [
    ("10M", 10),
    ("20S80M", 80),
    ("5M100N5M", 110),
    ("10M5D10M", 25),
    ("30=", 30),
    ("8M1I8M", 16),
    ("*", 0),
    ("30H", 0),
])
def test_cigar_reference_length_counts_only_reference_consuming_ops(cigar, expected):
    assert gb._parse_cigar_length(cigar) == expected


# ─── coverage: the text SAM fallback ─────────────────────────────────────────

SAM_LINES = [
    "@HD\tVN:1.6",
    "@SQ\tSN:chr1\tLN:1000",
    "",
    "r1\t0\tchr1\t101\t60\t25M\t*\t0\t0\t" + "A" * 25,
    "r_unmapped\t4\tchr1\t300\t60\t20M\t*\t0\t0\t" + "A" * 20,
    "too_short\t0\tchr1",
    "r_star\t0\tchr1\t51\t60\t*\t*\t0\t0\t" + "A" * 10,
    "r_other_chr\t0\tchr9\t600\t60\t50M\t*\t0\t0\t" + "A" * 50,
]


@pytest.fixture
def sam_file(tmp_path):
    path = tmp_path / "reads.sam"
    path.write_text("\n".join(SAM_LINES) + "\n", encoding="utf-8")
    return path


def test_sam_coverage_counts_every_bin_a_read_touches(sam_file):
    """Reads start mid-bin, which is what the old stepping loop got wrong: a 25 bp
    read from position 100 spans bins 10, 11 and 12."""
    positions, coverage = gb._compute_coverage_sam(str(sam_file), None, 10)

    assert list(positions[:13]) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    assert [coverage[i] for i in (10, 11, 12)] == [1, 1, 1], "r1 = 25M from 101"
    assert coverage[5] == 1, "r_star has no CIGAR, so its SEQ length (10) spans bin 5"
    assert coverage[6] == 0, "the line break is not part of the read"
    assert [coverage[i] for i in range(59, 65)] == [1] * 6, "the chr9 read is kept unfiltered"
    assert sum(coverage) == 10, coverage
    assert coverage[0] == 0 and coverage[7] == 0, "bins no read covers stay empty"


def test_sam_coverage_can_be_limited_to_one_chromosome(sam_file):
    positions, coverage = gb._compute_coverage_sam(str(sam_file), ("chr1", 0, 1000), 10)

    assert sum(coverage) == 4, "the chr9 read is filtered out by name"
    # The window is the axis: the pysam path spans the region it was given, and
    # this fallback used to stop at the last read instead (positions[-1] == 120),
    # so the same call drew two different axes depending on whether pysam was
    # installed.
    assert positions[-1] == 1000, "the track spans the region it was asked for"


def test_compute_coverage_without_pysam_reads_the_sam_as_text(sam_file, monkeypatch):
    """This fallback is what runs when pysam is missing, so the module is hidden
    rather than trusting whatever happens to be installed."""
    monkeypatch.setitem(sys.modules, "pysam", None)

    positions, coverage = gb.compute_coverage(str(sam_file), None, 10)

    assert len(positions) == len(coverage)
    assert sum(coverage) == 10, "the same bins the SAM fallback reports"


# ─── coverage: the pysam path, driven with a stand-in module ────────────────

class _Read:
    def __init__(self, start, end):
        self.reference_start = start
        self.reference_end = end


class _BamFile:
    def __init__(self, lengths, reads):
        self._lengths = lengths
        self._reads = reads
        self.references = list(lengths)
        self.opened_with = None

    def get_reference_length(self, chrom):
        return self._lengths[chrom]

    def fetch(self, chrom=None, start=None, end=None):
        self.opened_with = (chrom, start, end)
        return iter(self._reads)


@pytest.fixture
def fake_pysam(monkeypatch):
    """Install a stand-in pysam; the binning arithmetic under test is ours."""
    state = {"lengths": {"chr1": 250, "chr2": 100}, "reads": [_Read(0, 20), _Read(50, 60)]}
    opened = []

    def alignment_file(path, mode):
        opened.append((path, mode))
        return _BamFile(state["lengths"], state["reads"])

    monkeypatch.setitem(sys.modules, "pysam",
                        types.SimpleNamespace(AlignmentFile=alignment_file))
    return {"state": state, "opened": opened}


def test_pysam_coverage_uses_the_region_it_was_given(tmp_path, fake_pysam):
    positions, coverage = gb.compute_coverage(str(tmp_path / "x.bam"),
                                              ("chr1", 0, 100), 10)

    assert fake_pysam["opened"] == [(str(tmp_path / "x.bam"), "rb")]
    assert list(positions) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert list(coverage) == [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],             "a read ending at base 20 covers bins 0 and 1, not 2"
    assert fake_pysam["state"] and fake_pysam["opened"]


def test_pysam_coverage_fills_in_an_open_region_from_the_header(tmp_path, fake_pysam):
    fake_pysam["state"]["reads"] = [_Read(200, 240)]

    positions, coverage = gb.compute_coverage(str(tmp_path / "x.bam"),
                                              ("chr1", 150, None), 10)

    assert positions[-1] == 250, "the chromosome length from the header bounds the track"
    assert list(coverage) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],             "bases 200-239 cover bins 5 to 8 of the window"


def test_pysam_coverage_without_a_region_spans_the_longest_reference(tmp_path, fake_pysam):
    fake_pysam["state"]["reads"] = [_Read(0, 10)]

    positions, coverage = gb.compute_coverage(str(tmp_path / "x.bam"), None, 50)

    assert positions[-1] == 250
    assert fake_pysam["opened"][0][0].endswith("x.bam")
    assert coverage[0] == 1


def test_a_pysam_read_with_no_reference_end_is_skipped(tmp_path, fake_pysam):
    """A read with no CIGAR reports reference_end None; it cannot be binned, and
    guessing an end is how a phantom coverage bump appears."""
    fake_pysam["state"]["reads"] = [_Read(0, None), _Read(50, 60)]

    positions, coverage = gb.compute_coverage(str(tmp_path / "x.bam"),
                                              ("chr1", 0, 100), 10)

    assert coverage[5] == 1, "the read that does have an end is still counted"
    assert coverage[0] == 0, "the one without an end is not placed"


def test_pysam_coverage_of_an_empty_reference_returns_nothing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pysam", types.SimpleNamespace(
        AlignmentFile=lambda path, mode: _BamFile({"chrE": 0}, [])))

    positions, coverage = gb.compute_coverage(str(tmp_path / "e.bam"), ("chrE", 0, 0), 10)

    assert positions.size == 0 and coverage.size == 0


def test_coverage_track_carries_positions_and_coverage(tmp_path, fake_pysam):
    track = gb.create_coverage_from_bam(str(tmp_path / "x.bam"), ("chr1", 0, 100), 10)

    assert track['type'] == 'coverage' and track['name'] == 'Coverage'
    assert list(track['data']['coverage']) == [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]


# ─── track dictionaries handed to the browser ────────────────────────────────

def test_bed_track_carries_the_regions_it_parsed(tmp_path):
    bed = tmp_path / "genes.bed"
    bed.write_text("chr1\t1000\t2000\tgeneA\t55\nchr1\t3000\t4000\n", encoding="utf-8")

    track = gb.create_bed_track(str(bed))

    assert (track['type'], track['name'], track['color']) == ('bed', 'BED', '#4ecdc4')
    assert track['data']['regions'] == [
        ("chr1", 1000, 2000, "geneA", 55.0), ("chr1", 3000, 4000, "", 0.0)]


def test_variant_track_carries_the_variants_it_parsed(tmp_path):
    vcf = tmp_path / "calls.vcf"
    vcf.write_text("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                   "chr1\t1500\t.\tA\tG\t99\tPASS\t.\n", encoding="utf-8")

    track = gb.create_variant_track(str(vcf))

    assert (track['type'], track['name'], track['color']) == ('variant', 'Variants', '#ff6b6b')
    assert track['data']['variants'] == [
        {'chrom': 'chr1', 'pos': 1500, 'ref': 'A', 'alt': 'G',
         'qual': 99.0, 'filter': 'PASS', 'info': '.'}]


def test_both_loaders_report_a_file_that_is_not_there(tmp_path):
    """The browser checks the path and says so; the parsers themselves just raise."""
    missing = str(tmp_path / "absent")

    with pytest.raises(FileNotFoundError):
        gb.create_bed_track(missing)
    with pytest.raises(FileNotFoundError):
        gb.create_variant_track(missing)


# ─── track drawing ───────────────────────────────────────────────────────────

def test_plot_genome_tracks_draws_one_row_per_track():
    tracks = [
        {'type': 'coverage', 'name': 'cov', 'data': {
            'positions': np.arange(0, 50, 10), 'coverage': np.array([1, 2, 3, 4, 5])}},
        {'type': 'bed', 'name': 'bed', 'data': {'regions': [('chr1', 10, 20, 'a', 1.0)]}},
        {'type': 'variant', 'name': 'var', 'data': {'variants': [
            {'pos': 15, 'ref': 'A', 'alt': 'G', 'qual': 1.0, 'filter': 'PASS', 'info': ''}]}},
        {'type': 'signal', 'name': 'sig', 'data': {
            'positions': np.arange(0, 50, 10), 'values': np.array([1, 2, 3, 4, 5])}},
    ]

    fig = gb.plot_genome_tracks(tracks, region=('chr1', 0, 100), title="Browser")

    assert [ax.get_ylabel() for ax in fig.axes[:4]] == ['cov', 'bed', 'var', 'sig']
    assert any(t.get_text() == "Browser" for t in fig.texts), "the title argument is used"
    assert [len(ax.get_xticklabels()) for ax in fig.axes] == [0, 0, 0, 6], "only the last row shows ticks"
    assert [ax.get_xlabel() for ax in fig.axes[:3]] == ['', '', '']
    assert fig.axes[3].get_xlabel() == "Position (bp)", "only the last row labels x"


def test_a_track_type_the_browser_does_not_know_names_itself_in_the_row():
    """A type outside the four drawers still takes a row and its name — and says
    so, because an empty row with no word reads as "this region has no features"."""
    fig = gb.plot_genome_tracks([{'type': 'gene', 'name': 'genes'}])

    assert len(fig.axes) == 1
    assert fig.axes[0].get_ylabel() == "genes"
    assert [t.get_text() for t in fig.axes[0].texts] == ["unknown track type 'gene'"]
    assert (len(fig.axes[0].patches), len(fig.axes[0].collections)) == (0, 0)


def test_plot_genome_tracks_wraps_a_single_axes():
    track = {'type': 'bed', 'name': 'one', 'data': {'regions': [('chr1', 1, 5, 'x', 0.0)]}}

    fig = gb.plot_genome_tracks([track])

    assert len(fig.axes) == 1
    assert fig.axes[0].get_ylabel() == 'one'
    assert fig.axes[0].get_xlabel() == "Position (bp)"


def test_plot_genome_tracks_without_tracks_draws_nothing():
    assert gb.plot_genome_tracks([]) is None


def test_plot_genome_tracks_labels_an_unnamed_track_by_position():
    fig = gb.plot_genome_tracks([{'data': {'regions': []}}])

    assert fig.axes[0].get_ylabel() == "Track 1", "an unknown type still gets a row label"


def test_coverage_track_says_when_it_has_no_data():
    fig, ax = _text_ax()

    gb._draw_coverage_track(ax, {'positions': np.array([]), 'coverage': np.array([])},
                            None, 'blue')

    assert [t.get_text() for t in ax.texts] == ["No coverage data"]
    assert len(ax.collections) == 0 and len(ax.lines) == 0


def test_coverage_track_frames_the_data_it_drew():
    fig, ax = _text_ax()
    data = {'positions': np.array([0, 10, 20]), 'coverage': np.array([1, 5, 3])}

    gb._draw_coverage_track(ax, data, ('chr1', 0, 40), 'green')

    assert ax.get_xlim() == (0, 40)
    assert ax.get_ylim() == pytest.approx((0, 5.5))
    assert len(ax.get_lines()) == 1


def test_coverage_track_without_a_region_uses_its_own_span():
    fig, ax = _text_ax()
    data = {'positions': np.array([100, 200, 300]), 'coverage': np.array([1, 2, 3])}

    gb._draw_coverage_track(ax, data, None, 'green')

    assert ax.get_xlim() == (100, 300)


def test_a_single_point_coverage_track_still_sets_limits():
    fig, ax = _text_ax()

    gb._draw_coverage_track(ax, {'positions': np.array([5]), 'coverage': np.array([2])},
                            None, 'green')

    assert ax.get_xlim() == (5, 1) or ax.get_xlim() == (5.0, 1.0)


def test_bed_track_names_a_file_without_intervals():
    fig, ax = _text_ax()

    gb._draw_bed_track(ax, {'regions': []}, None, 'cyan')

    assert [t.get_text() for t in ax.texts] == ["No BED data"]


def test_bed_track_stacks_overlapping_intervals_into_lanes():
    fig, ax = _text_ax()
    regions = [('chr1', 100, 300, 'a', 0.0), ('chr1', 200, 400, 'b', 0.0),
               ('chr1', 400, 500, 'c', 0.0)]

    gb._draw_bed_track(ax, {'regions': regions}, None, 'cyan')

    assert len(ax.patches) == 3, "one rectangle per interval"
    assert ax.get_ylim() == (-0.5, 2), "two lanes: the second interval overlaps the first"
    assert sorted(t.get_text() for t in ax.texts) == ['a', 'b', 'c']


def test_a_zero_width_interval_is_drawn_without_a_label():
    fig, ax = _text_ax()

    gb._draw_bed_track(ax, {'regions': [('chr1', 100, 100, 'point', 0.0)]}, None, 'cyan')

    assert len(ax.patches) == 1
    assert len(ax.texts) == 0


def test_bed_track_labels_are_cut_to_fifteen_characters():
    fig, ax = _text_ax()

    gb._draw_bed_track(ax, {'regions': [('chr1', 0, 100, "x" * 30, 0.0)]}, None, 'cyan')

    assert [t.get_text() for t in ax.texts] == ["x" * 15]


def test_variant_track_classifies_by_ref_and_alt_length():
    fig, ax = _text_ax()
    variants = [
        {'pos': 10, 'ref': 'A', 'alt': 'G'},
        {'pos': 20, 'ref': 'ACG', 'alt': 'A'},
        {'pos': 30, 'ref': 'A', 'alt': 'ACGT'},
    ]

    gb._draw_variant_track(ax, {'variants': variants}, ('chr1', 0, 50), 'grey')

    colors = [line.get_markerfacecolor() for line in ax.get_lines()]
    assert colors == ['#ff6b6b', '#ffd93d', '#6bcb77'], "SNP, deletion, insertion"
    assert ax.get_xlim() == (0, 50)
    assert sorted(t.get_text() for t in ax.texts) == ['A>ACGT', 'A>G', 'ACG>A']


def test_variant_track_names_an_empty_file():
    fig, ax = _text_ax()

    gb._draw_variant_track(ax, {'variants': []}, None, 'grey')

    assert [t.get_text() for t in ax.texts] == ["No variant data"]


def test_signal_track_draws_its_values():
    fig, ax = _text_ax()

    gb._draw_signal_track(ax, {'positions': np.array([0, 10, 20]),
                               'values': np.array([2, 6, 4])}, ('chr1', 0, 20), 'red')

    assert ax.get_ylim() == pytest.approx((0, 6.6))
    assert ax.get_xlim() == (0, 20)
    assert ax.get_ylabel() == "Signal"


def test_signal_track_says_when_it_has_no_positions():
    fig, ax = _text_ax()

    gb._draw_signal_track(ax, {'positions': np.array([]), 'values': np.array([])},
                          None, 'red')

    assert [t.get_text() for t in ax.texts] == ["No signal data"]


def test_signal_track_without_a_region_keeps_the_span_of_its_own_data():
    fig, ax = _text_ax()

    gb._draw_signal_track(ax, {'positions': np.array([0, 10, 20, 30, 40]),
                               'values': np.array([1, 2, 3, 4, 5])}, None, 'red')

    assert ax.get_xlim() == pytest.approx((-2.0, 42.0)), "the view is left to the data"
    assert ax.get_ylim() == pytest.approx((0, 5.5))
