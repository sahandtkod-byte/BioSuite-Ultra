"""Peak calling's merge, its BED output, and every way MACS2 can fail to be there.

The merge is what stops one broad enrichment being reported as several independent
peaks, so it is tested on summits closer and farther apart than ``min_distance``.
The MACS2 wrapper must return False for a non-zero exit, for a missing binary and
for a timeout, and ``call_peaks`` must then still produce a real result from the
built-in caller rather than an empty report.  Annotation is tested on the three
cases that matter: a peak inside a gene, a peak between two genes on the same
chromosome, and a peak on a chromosome with no genes at all.
"""
import subprocess

import pytest

from biosuite.core import peak_calling as pc


def peak(start, end, summit=None, score=10.0, chrom="chr1", name="", p_value=1e-9,
          fold=4.0):
    return pc.Peak(chrom=chrom, start=start, end=end,
                   summit=summit if summit is not None else (start + end) // 2,
                   score=score, p_value=p_value, fold_enrichment=fold,
                   length=end - start, name=name)


def sam_with_reads(tmp_path, *positions, mapq=60):
    path = tmp_path / "reads.sam"
    lines = ["@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:1000000\n"]
    for i, pos in enumerate(positions):
        lines.append("\t".join(str(field) for field in
                               (f"r{i}", 0, "chr1", pos, mapq, "4M", "*", 0, 0,
                                "ACGT", "IIII")) + "\n")
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


class FakeRun:
    def __init__(self, returncode=0, raises=None):
        self.returncode = returncode
        self.raises = raises
        self.commands = []

    def __call__(self, cmd, **kwargs):
        self.commands.append((list(cmd), kwargs))
        if self.raises:
            raise self.raises
        return subprocess.CompletedProcess(cmd, self.returncode, "", "")


# ─── merging ─────────────────────────────────────────────────────────────────

def test_two_summits_inside_min_distance_become_the_stronger_one():
    near = [peak(0, 100, score=8.0), peak(50, 120, score=14.0)]

    merged = pc._merge_close_peaks(near, min_distance=200)

    assert len(merged) == 1, merged
    assert merged[0].score == 14.0 and merged[0].p_value == 1e-9
    assert (merged[0].start, merged[0].end) == (0, 120), "the span still covers both"
    assert merged[0].length == 120
    assert merged[0].summit == 85, "the summit moved to the stronger peak's middle"


def test_summits_exactly_min_distance_apart_stay_two_peaks():
    apart = [peak(0, 100), peak(400, 500)]

    merged = pc._merge_close_peaks(apart, min_distance=200)

    assert [p.start for p in merged] == [0, 400], merged


def test_peaks_on_different_chromosomes_are_never_merged():
    other = [peak(0, 100, chrom="chr1"), peak(10, 110, chrom="chr2")]

    assert [p.chrom for p in pc._merge_close_peaks(other, 200)] == ["chr1", "chr2"]


@pytest.mark.parametrize("peaks, min_distance", [
    ([peak(0, 100)], 200),
    ([], 200),
    ([peak(0, 100), peak(50, 120)], 0),
])
def test_a_merge_that_has_nothing_to_do_returns_the_same_peaks(peaks, min_distance):
    merged = pc._merge_close_peaks(peaks, min_distance)

    assert merged == peaks
    assert merged is not peaks, "the caller keeps its own list either way"


# ─── the built-in caller and its BED ────────────────────────────────────────

def test_a_sam_of_only_unmapped_reads_says_there_is_nothing_to_call(tmp_path):
    sam = sam_with_reads(tmp_path, 100, mapq=5)

    report = pc._builtin_call_peaks(sam)

    assert report.engine == 'builtin'
    assert report.message == "No mapped reads found.", report.message
    assert report.total_peaks == 0


def test_a_chromosome_that_holds_no_reads_is_skipped_not_divided_by_zero():
    """The map can carry an empty list — from a SAM whose only reads were filtered
    out — and the mean of nothing is not a background rate."""
    peaks = pc._find_peaks_from_coverage({"chr1": [], "chr2": [100]})

    assert all(p.chrom == "chr2" for p in peaks) or peaks == []


def test_the_bed_rows_are_the_peaks_the_report_counted(tmp_path):
    """Sixty reads stacked on one spot, with one distant read holding the spread
    open: the pile is far above the Poisson cut, so at least one peak must come
    out — a BED that agreed with the report on zero peaks would prove nothing."""
    sam = sam_with_reads(tmp_path, *([1000] * 60), 400_000)
    bed = tmp_path / "peaks.bed"

    report = pc._builtin_call_peaks(sam, output_bed=str(bed), min_score=1)

    assert report.total_peaks > 0, "the fixture no longer piles up into a peak"
    assert report.output_bed == str(bed)
    rows = [line.split("\t") for line in
            bed.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == report.total_peaks, (rows, report.total_peaks)
    for row, called in zip(rows, report.peaks):
        assert row[0] == called.chrom and int(row[1]) == called.start
        assert int(row[2]) == called.end and float(row[4]) == called.score
        assert row[5] == "."


def test_an_explicit_threshold_below_the_score_floor_still_rejects_the_run():
    """The threshold decides what counts as enriched; ``min_score`` decides what is
    worth reporting, and a caller can lower the first without lowering the second."""
    peaks = pc._find_peaks_from_coverage({"chr1": [1000] * 8},
                                         min_score=50, threshold=0.0)

    assert peaks == [], peaks


def test_a_single_locus_pile_is_refused_with_a_named_message(tmp_path):
    """C-727: the pile's Poisson background equals the pile itself, so the
    strongest possible signal used to answer "0 peaks found" with no reason."""
    sam = sam_with_reads(tmp_path, *([100] * 100))

    report = pc._builtin_call_peaks(sam)

    assert report.engine == 'builtin'
    assert report.total_peaks == 0
    assert "single-locus pile" in report.message, report.message
    assert "position 100" in report.message, report.message


def test_a_pile_on_one_chromosome_does_not_hide_peaks_on_another(tmp_path):
    """The pile is named in the message; the callable chromosome still reports."""
    sam = tmp_path / "mixed.sam"
    lines = ["@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:1000000\n@SQ\tSN:chr2\tLN:1000000\n"]
    for i in range(100):
        lines.append(f"r{i}\t0\tchr1\t100\t60\t4M\t*\t0\t0\tACGT\tIIII\n")
    for i in range(60):
        lines.append(f"s{i}\t0\tchr2\t1000\t60\t4M\t*\t0\t0\tACGT\tIIII\n")
    # One distant read keeps chr2's spread open so its pile is callable.
    lines.append("far\t0\tchr2\t400000\t60\t4M\t*\t0\t0\tACGT\tIIII\n")
    sam.write_text("".join(lines), encoding="utf-8")

    report = pc._builtin_call_peaks(str(sam), min_score=1)

    assert report.total_peaks > 0, report.message
    assert all(p.chrom == "chr2" for p in report.peaks), report.peaks
    assert "Skipped chr1" in report.message, report.message
    assert "single-locus pile" in report.message, report.message


def test_a_run_still_open_at_the_last_base_is_reported_rather_than_dropped():
    peaks = pc._find_peaks_from_coverage({"chr1": [1000] * 8},
                                         min_score=1, threshold=0.0)

    assert len(peaks) == 1, peaks
    assert peaks[0].end == 1000 + 200, "the window the read covers is the end of the span"
    assert peaks[0].name == "" and peaks[0].score > 1


# ─── MACS2, external ────────────────────────────────────────────────────────

@pytest.mark.parametrize("outcome, expected", [
    ({"returncode": 0}, True),
    ({"returncode": 1}, False),
    ({"raises": FileNotFoundError("macs2")}, False),
    ({"raises": subprocess.TimeoutExpired(cmd="macs2", timeout=7200)}, False),
])
def test_the_macs2_wrapper_answers_yes_only_for_a_clean_run(monkeypatch, outcome, expected):
    fake = FakeRun(**outcome)
    monkeypatch.setattr(pc.subprocess, "run", fake)

    assert pc._macs2_call_peaks("t.bam", "/tmp/out") is expected

    cmd, kwargs = fake.commands[0]
    assert cmd[:2] == ['macs2', 'callpeak'] and kwargs['timeout'] == 7200
    assert cmd[cmd.index('-g') + 1] == 'hs'


def test_macs2_is_used_when_it_is_present_and_not_ignored_when_it_fails(
        tmp_path, monkeypatch):
    sam = sam_with_reads(tmp_path, 100)
    monkeypatch.setattr(pc, "check_peak_tools", lambda: {'macs2': True})

    seen = []
    monkeypatch.setattr(pc, "_macs2_call_peaks",
                        lambda *a, **k: seen.append((a, k)) or True)
    used = pc.call_peaks(sam, genome='mm')
    assert used.engine == 'macs2', used
    assert len(seen) == 1 and seen[0][0][0] == sam and seen[0][1] == {'genome': 'mm'}, \
        "the genome the caller named must reach MACS2, not stay at 'hs'"
    assert used.message == "Using MACS2 (external, genome=mm)"

    monkeypatch.setattr(pc, "_macs2_call_peaks", lambda *a, **k: False)
    fell_back = pc.call_peaks(sam)
    assert fell_back.engine == 'builtin', \
        "under tool='auto' a failed external run must still answer with the built-in caller"


def test_a_missing_input_file_is_answered_before_any_tool_is_asked(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(pc, "check_peak_tools", lambda: asked.append(1) or {'macs2': False})
    absent = tmp_path / "nope.bam"

    report = pc.call_peaks(str(absent))

    assert report.engine == 'none'
    assert report.message == f"File not found: {absent}", report.message
    assert asked == [], "the file is checked first"


# ─── annotation and the report ───────────────────────────────────────────────

def test_a_peak_is_named_for_the_gene_it_overlaps_and_an_empty_region_is_intergenic(tmp_path):
    bed = tmp_path / "genes.bed"
    bed.write_text("chr1\t1000\t2000\tBRCA1\nchr1\t5000\t6000\tTP53\n"
                   "junk line without fields\n", encoding="utf-8")
    peaks = [peak(1500, 1600), peak(4990, 5010), peak(30000, 30100, chrom="chr9")]

    annotated = pc.annotate_peaks_with_genes(peaks, str(bed))

    assert [p.name for p in annotated] == ["BRCA1", "TP53", "intergenic"], annotated


def test_the_distance_to_a_gene_is_the_gap_between_the_two_intervals(tmp_path):
    """200-210 is 100 bases from the gene ending at 100 and 190 from the one
    starting at 400, so the nearer flank decides the name — not the summit."""
    bed = tmp_path / "genes.bed"
    bed.write_text("chr1\t0\t100\tleft\nchr1\t400\t500\tright\n", encoding="utf-8")
    lone = peak(200, 210)

    pc.annotate_peaks_with_genes([lone], str(bed))

    assert lone.name == "left", lone.name

    nearer = peak(250, 260)          # 150 from left, 140 from right
    pc.annotate_peaks_with_genes([nearer], str(bed))
    assert nearer.name == "right", nearer.name


def test_a_weaker_neighbour_still_widens_the_span_of_the_peak_it_falls_into():
    strong = peak(0, 100, score=20.0, p_value=1e-12, fold=9.0)
    weak = peak(50, 200, score=3.0, p_value=0.5, fold=1.5)

    merged = pc._merge_close_peaks([strong, weak], min_distance=200)

    assert len(merged) == 1
    assert merged[0].score == 20.0 and merged[0].summit == 50, "the stronger peak wins"
    assert (merged[0].start, merged[0].end, merged[0].length) == (0, 200, 200)


def test_a_window_of_zero_bases_leaves_the_reads_stacked_on_one_spot():
    """The span cannot be zero: the count itself becomes the depth."""
    assert pc._background_lambda([7, 7, 7], window=0) == pytest.approx(3.0)


def test_a_bed_path_the_caller_gave_is_the_path_that_comes_back(tmp_path, monkeypatch):
    sam = sam_with_reads(tmp_path, *([1000] * 60), 400_000)
    bed = tmp_path / "named.bed"
    monkeypatch.setattr(pc, "check_peak_tools", lambda: {'macs2': False})

    report = pc.call_peaks(sam, output_bed=str(bed))

    assert report.output_bed == str(bed), report
    assert bed.exists(), "the caller's path was written, not a temporary one"


def test_macs2_being_absent_skips_the_external_branch_entirely(tmp_path, monkeypatch):
    sam = sam_with_reads(tmp_path, 100)
    monkeypatch.setattr(pc, "check_peak_tools", lambda: {'macs2': False})
    monkeypatch.setattr(pc, "_macs2_call_peaks",
                        lambda *a: pytest.fail("macs2 must not run when it is absent"))

    report = pc.call_peaks(sam)

    assert report.engine == 'builtin', report


def test_a_report_with_no_note_prints_no_note():
    lines = pc.format_peak_report(pc.PeakReport(engine='macs2', total_peaks=0)).splitlines()

    assert lines == ["=== Peak Calling Report ===", "Engine: macs2", "Total peaks: 0"]


def test_the_report_shows_the_score_range_only_when_there_are_scores_to_show():
    empty = pc.format_peak_report(pc.PeakReport(engine='builtin', message="nothing found"))
    filled = pc.format_peak_report(pc.PeakReport(
        engine='builtin', total_peaks=2, peaks=[peak(0, 10, score=6.5),
                                                peak(20, 30, score=21.25)],
        message="two peaks"))

    assert empty.splitlines() == ["=== Peak Calling Report ===", "Engine: builtin",
                                  "Total peaks: 0", "Note: nothing found"]
    lines = filled.splitlines()
    assert "Total peaks: 2" in lines
    assert "Score range: 6.5 — 21.2" in "\n".join(lines) or "21.3" in "\n".join(lines), lines
    assert lines[-1] == "Note: two peaks"
