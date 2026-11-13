"""The external aligners and the SAM this module reads back from them.

``_bwa_align`` and ``_bowtie2_align`` answer ``None`` whenever their tool did not
run, and ``align_reads`` must then try the next engine rather than report an empty
alignment.  The counters of an external run come from ``_summarize_external_sam``,
which is tested on a SAM with header lines, unmapped flags and a truncated record —
the three things that used to make an external report claim zero reads.  Scratch
files are checked for the one property that matters: they are gone afterwards.
"""
import os
import subprocess

import pytest

from biosuite.core import read_aligner as ra

SAM_BODY = """\
@HD\tVN:1.6\tSO:coordinate
@SQ\tSN:chr1\tLN:1000
r1\t0\tchr1\t11\t60\t4M\t*\t0\t0\tACGT\tIIII
r2\t4\t*\t0\t0\t*\t*\t0\t0\tACGT\tIIII
r3\t0\tchr1\t21\t30\t4M\t*\t0\t0\tACGT\tIIII
r4\t16\tchr1\t31\t20\t4M\t*\t0\t0\tACGT\tIIII
r5\t4\t*\t0\t0\t*\t*\t0\t0\tNNNN\t!!!!
short\t0\tchr1
"""


@pytest.fixture
def sam_file(tmp_path):
    path = tmp_path / "external.sam"
    path.write_text(SAM_BODY, encoding="utf-8")
    return str(path)


class ToolRun:
    """A stand-in ``subprocess.run`` that answers each command by name."""

    def __init__(self, sam_text="", returncode=0, raises=None, writes=None):
        self.sam_text = sam_text
        self.returncode = returncode
        self.raises = raises
        self.writes = writes          # path -> text, for tools writing files
        self.commands = []

    def __call__(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        if self.raises:
            raise self.raises
        if self.writes:
            for path, text in self.writes.items():
                if path in cmd:
                    with open(path, "w", encoding="utf-8") as handle:
                        handle.write(text)
        return subprocess.CompletedProcess(cmd, self.returncode, self.sam_text, "")


# ─── reading an external SAM back ────────────────────────────────────────────

def test_the_report_counts_come_from_the_sam_the_tool_wrote(sam_file):
    summary = ra._summarize_external_sam(sam_file)

    assert summary['total_reads'] == 5, "the header and the truncated row are not reads"
    assert summary['mapped_reads'] == 3
    assert summary['unmapped_reads'] == 2
    assert summary['mapping_rate'] == pytest.approx(60.0)
    assert summary['avg_mapping_quality'] == pytest.approx((60 + 30 + 20) / 3)


def test_an_empty_sam_reports_a_zero_rate_instead_of_dividing_by_zero(tmp_path):
    path = tmp_path / "empty.sam"
    path.write_text("@HD\tVN:1.6\n", encoding="utf-8")

    summary = ra._summarize_external_sam(str(path))

    assert summary == {'total_reads': 0, 'mapped_reads': 0, 'unmapped_reads': 0,
                       'mapping_rate': 0, 'avg_mapping_quality': 0}


# ─── scratch files ───────────────────────────────────────────────────────────

def test_a_scratch_file_is_removed_and_its_absence_is_not_an_error(tmp_path):
    gone = tmp_path / "never-existed.sam"
    left = tmp_path / "scratch.sam"
    left.write_text("@HD\n", encoding="utf-8")

    ra._drop_scratch(None)
    ra._drop_scratch(str(gone))
    ra._drop_scratch(str(left))

    assert not left.exists()


def test_a_scratch_file_that_refuses_to_be_deleted_is_not_fatal(tmp_path, monkeypatch):
    left = tmp_path / "scratch.sam"
    left.write_text("@HD\n", encoding="utf-8")
    attempted = []

    def refuse(path):
        attempted.append(path)
        raise OSError("device busy")

    monkeypatch.setattr(os, "unlink", refuse)

    ra._drop_scratch(str(left))

    assert attempted == [str(left)], "nothing was attempted, so the guard proved nothing"
    assert left.exists()


# ─── BWA ─────────────────────────────────────────────────────────────────────

def test_bwa_that_cannot_be_started_is_reported_as_not_run(monkeypatch, tmp_path):
    monkeypatch.setattr(ra.subprocess, "run", ToolRun(raises=FileNotFoundError("bwa")))

    assert ra._bwa_align("ref.fa", "reads.fa", str(tmp_path / "out.sam")) is None


def test_bwa_answering_nothing_writes_nothing(monkeypatch, tmp_path):
    """Exit 0 with empty stdout is a failed alignment, not an empty result."""
    out = tmp_path / "out.sam"
    fake = ToolRun(sam_text="", returncode=0)
    monkeypatch.setattr(ra.subprocess, "run", fake)

    assert ra._bwa_align("ref.fa", "reads.fa", str(out)) is None
    assert not out.exists(), "an empty SAM must not be handed to the report"


def test_bwa_writes_the_sam_to_the_path_the_caller_named(monkeypatch, tmp_path):
    out = tmp_path / "out.sam"
    fake = ToolRun(sam_text=SAM_BODY, returncode=0)
    monkeypatch.setattr(ra.subprocess, "run", fake)

    sam = ra._bwa_align("ref.fa", "reads.fa", str(out), threads=4)

    assert sam == str(out) and out.read_text(encoding="utf-8") == SAM_BODY
    assert fake.commands[0] == ['bwa', 'index', 'ref.fa'], "the reference is indexed first"
    assert fake.commands[1] == ['bwa', 'mem', '-t', '4', 'ref.fa', 'reads.fa']


def test_a_bwa_run_that_fails_leaves_no_temporary_file_behind(monkeypatch):
    fake = ToolRun(sam_text="", returncode=1)
    monkeypatch.setattr(ra.subprocess, "run", fake)
    made = []
    original = ra.tempfile.NamedTemporaryFile

    def spy(**kwargs):
        handle = original(**kwargs)
        made.append(handle.name)
        return handle

    monkeypatch.setattr(ra.tempfile, "NamedTemporaryFile", spy)

    assert ra._bwa_align("ref.fa", "reads.fa", "") is None
    assert made and all(not os.path.exists(path) for path in made), \
        "the scratch SAM outlived a failed run"


def test_a_bwa_mem_that_dies_after_indexing_leaves_no_scratch(monkeypatch):
    """Indexing can succeed while the alignment itself dies mid-run."""
    commands = []

    def run(cmd, **kwargs):
        commands.append(list(cmd))
        if cmd[:2] == ["bwa", "mem"]:
            raise OSError("bwa mem died")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    made = []
    original = ra.tempfile.NamedTemporaryFile

    def spy(**kwargs):
        handle = original(**kwargs)
        made.append(handle.name)
        return handle

    monkeypatch.setattr(ra.tempfile, "NamedTemporaryFile", spy)
    monkeypatch.setattr(ra.subprocess, "run", run)

    assert ra._bwa_align("ref.fa", "reads.fa", "") is None
    assert [c[:2] for c in commands] == [["bwa", "index"], ["bwa", "mem"]]
    assert made and all(not os.path.exists(path) for path in made)


# ─── bowtie2 ─────────────────────────────────────────────────────────────────

def test_bowtie2_returns_the_sam_only_once_it_is_on_disk(monkeypatch, tmp_path):
    out = tmp_path / "bt.sam"
    fake = ToolRun(returncode=0, writes={str(out): SAM_BODY})
    monkeypatch.setattr(ra.subprocess, "run", fake)

    sam = ra._bowtie2_align("ref.fa", "reads.fa", str(out), threads=2)

    assert sam == str(out)
    build, align = fake.commands
    assert build[0] == 'bowtie2-build' and align[:2] == ['bowtie2', '-x']
    assert align[align.index('-U') + 1] == "reads.fa"
    assert align[align.index('--threads') + 1] == "2"
    index_dir = os.path.dirname(build[2])
    assert not os.path.exists(index_dir), \
        "the bowtie2-build sidecars were left behind in the index directory"


def test_bowtie2_that_writes_no_output_answers_none(monkeypatch, tmp_path):
    fake = ToolRun(returncode=0, writes={})
    monkeypatch.setattr(ra.subprocess, "run", fake)

    assert ra._bowtie2_align("ref.fa", "reads.fa", str(tmp_path / "missing.sam")) is None


def test_bowtie2_that_cannot_be_run_leaves_no_scratch(monkeypatch):
    made = []
    original = ra.tempfile.NamedTemporaryFile

    def spy(**kwargs):
        handle = original(**kwargs)
        made.append(handle.name)
        return handle

    monkeypatch.setattr(ra.tempfile, "NamedTemporaryFile", spy)
    monkeypatch.setattr(ra.subprocess, "run", ToolRun(raises=OSError("no bowtie2")))

    assert ra._bowtie2_align("ref.fa", "reads.fa", "") is None
    assert made and all(not os.path.exists(path) for path in made)


# ─── the engine choice ───────────────────────────────────────────────────────

@pytest.fixture
def inputs(tmp_path):
    ref = tmp_path / "ref.fa"
    ref.write_text(">chr1\n" + "ACGT" * 250 + "\n", encoding="utf-8")
    reads = tmp_path / "reads.fa"
    reads.write_text(">r1\nACGTACGTAC\n", encoding="utf-8")
    return str(ref), str(reads)


def test_bwa_result_carries_the_counts_from_its_own_sam(inputs, monkeypatch, tmp_path):
    reference, reads = inputs
    out = tmp_path / "chosen.sam"
    monkeypatch.setattr(ra, "check_aligner_tools", lambda: {'bwa': True, 'bowtie2': False})
    monkeypatch.setattr(ra.subprocess, "run", ToolRun(sam_text=SAM_BODY))

    report = ra.align_reads(reference, reads, output_file=str(out), tool='bwa')

    assert (report.tool, report.engine) == ('bwa', 'bwa')
    assert report.output_file == str(out)
    assert report.total_reads == 5 and report.mapped_reads == 3
    assert report.message == "Using BWA-MEM (external)"


def test_bowtie2_is_tried_when_bwa_declines(inputs, monkeypatch, tmp_path):
    reference, reads = inputs
    out = tmp_path / "bt2.sam"
    monkeypatch.setattr(ra, "check_aligner_tools", lambda: {'bwa': True, 'bowtie2': True})
    monkeypatch.setattr(ra, "_bwa_align", lambda *a, **k: None)
    monkeypatch.setattr(ra.subprocess, "run", ToolRun(returncode=0,
                                                      writes={str(out): SAM_BODY}))

    report = ra.align_reads(reference, reads, output_file=str(out), tool='auto')

    assert (report.tool, report.engine) == ('bowtie2', 'bowtie2')
    assert report.mapped_reads == 3


def test_a_bowtie2_that_produces_no_sam_hands_the_work_to_the_built_in(
        inputs, monkeypatch, tmp_path):
    """``None`` from the tool means "did not run", not "aligned nothing"."""
    reference, reads = inputs
    monkeypatch.setattr(ra, "check_aligner_tools", lambda: {'bwa': False, 'bowtie2': True})
    monkeypatch.setattr(ra, "_bowtie2_align", lambda *a, **k: None)
    reserved = []
    original_mkstemp = ra.tempfile.mkstemp

    def fake_mkstemp(suffix="", dir=None):
        # captured before patching: ra.tempfile *is* the stdlib module
        handle, path = original_mkstemp(suffix=suffix, dir=str(tmp_path))
        reserved.append(path)
        return handle, path

    monkeypatch.setattr(ra.tempfile, "mkstemp", fake_mkstemp)

    report = ra.align_reads(reference, reads, tool='auto')

    assert (report.tool, report.engine) == ('builtin_seed_extend', 'builtin')
    assert reserved and report.output_file == reserved[0]
    assert os.path.exists(report.output_file)


def test_no_engine_at_all_still_produces_a_sam_from_the_built_in(inputs, monkeypatch, tmp_path):
    reference, reads = inputs
    out = tmp_path / "builtin.sam"
    monkeypatch.setattr(ra, "check_aligner_tools", lambda: {'bwa': False, 'bowtie2': False})

    report = ra.align_reads(reference, reads, output_file=str(out))

    assert report.engine == 'builtin', report
    assert out.exists() and report.output_file == str(out)
