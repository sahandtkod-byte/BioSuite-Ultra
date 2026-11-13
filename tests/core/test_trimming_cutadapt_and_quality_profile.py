"""The cutadapt wrapper, the numbers it reads out of cutadapt's stderr, and the
quality profile that stands in when no external tool is installed.

The wrapper answers ``None`` when cutadapt exits non-zero or cannot be started, so
``trim_fastq`` can fall through to the built-in engine instead of reporting an empty
run - and it warns when it does, because a silent engine switch reads as a clean run.
The stderr parser is tested against a real cutadapt summary block, including the
thousands separators, and the paired-end path is checked for the property that
matters most: both mates of a read survive or neither does.
"""
import re
import subprocess

import pytest

from biosuite.core import trimming as tr


def write_fastq(tmp_path, name, records):
    """records: (name, sequence, quality) triples."""
    path = tmp_path / name
    path.write_text("".join(f"@{n}\n{s}\n+\n{q}\n" for n, s, q in records),
                    encoding="utf-8")
    return str(path)


CUTADAPT_SUMMARY = """\
Finished in 0.1 seconds, the processing speed was 20,000 reads/s.
Total processed reads: 1,234
Total reads processed:                  1,234
Reads with adapters:                     234 (19.0%)
Reads written (passing filters):       1,000 (81.0%)
Total basepairs processed: 123,400 bp
Quality-trimmed:              1,234 bp (1.0%)
Total written (filtered):   100,000 bp (81.0%)
Reads that were too short:                   234 (19.0%)
"""


class FakeRun:
    def __init__(self, returncode=0, stderr="", raises=None):
        self.returncode = returncode
        self.stderr = stderr
        self.raises = raises
        self.commands = []

    def __call__(self, cmd, **kwargs):
        self.commands.append((list(cmd), kwargs))
        if self.raises:
            raise self.raises
        return subprocess.CompletedProcess(cmd, self.returncode, "", self.stderr)


# ─── the tool probe ──────────────────────────────────────────────────────────

def test_the_tool_probe_asks_cutadapt_its_own_version(monkeypatch):
    fake = FakeRun(returncode=0)
    monkeypatch.setattr(tr.subprocess, "run", fake)

    assert tr.check_trimming_tools() == {'cutadapt': True}
    assert fake.commands[0][0] == ['cutadapt', '--version']


def test_a_cutadapt_that_cannot_be_asked_is_reported_as_absent(monkeypatch):
    monkeypatch.setattr(tr.subprocess, "run",
                        FakeRun(raises=FileNotFoundError("cutadapt")))

    assert tr.check_trimming_tools() == {'cutadapt': False}


def test_a_cutadapt_that_cannot_be_executed_is_also_absent(monkeypatch):
    """Measured before: the probe caught only ``FileNotFoundError`` and
    ``TimeoutExpired``, so a cutadapt on PATH without the execute bit raised
    PermissionError out of ``check_trimming_tools()`` - and out of every entry point
    that asks it which engine to use."""
    monkeypatch.setattr(tr.subprocess, "run",
                        FakeRun(raises=PermissionError("Permission denied: 'cutadapt'")))

    assert tr.check_trimming_tools() == {'cutadapt': False}


# ─── the built-in engine's legacy argument ───────────────────────────────────

def test_a_single_adapter_sequence_still_trims(tmp_path):
    """``adapter_seq=`` predates the adapter list and has to keep working."""
    src = write_fastq(tmp_path, "in.fastq", [("r1", "AACCCAGATCGGAAGAGCGG", "IIIIIIIIIIIIIIIIIIII")])
    out = tmp_path / "out.fastq"

    report = tr._pure_python_trim(src, str(out), quality_threshold=20, min_length=5,
                                 adapter_seq="AGATCGGAAGAGC")

    assert report.adapter_trimmed == 1, report
    assert out.read_text(encoding="utf-8").splitlines()[1] == "AACCC"


# ─── running cutadapt ────────────────────────────────────────────────────────

@pytest.mark.parametrize("outcome, expected, adapter", [
    ({"returncode": 0, "stderr": CUTADAPT_SUMMARY}, "report", None),
    ({"returncode": 0, "stderr": CUTADAPT_SUMMARY}, "report", "AGATCGGA"),
    ({"returncode": 2, "stderr": "cutadapt: error"}, None, None),
    ({"raises": FileNotFoundError("cutadapt")}, None, None),
    ({"raises": subprocess.TimeoutExpired(cmd="cutadapt", timeout=3600)}, None, None),
])
def test_cutadapt_is_only_claimed_when_it_ran_cleanly(monkeypatch, outcome, expected, adapter):
    fake = FakeRun(**outcome)
    monkeypatch.setattr(tr.subprocess, "run", fake)

    report = tr._cutadapt_trim("in.fastq", "out.fastq", 20, 36, adapter)

    if expected is None:
        assert report is None, "a failed external run must fall back, not report zero reads"
    else:
        assert report.engine == 'cutadapt' and report.total_reads == 1234

    cmd, kwargs = fake.commands[0]
    assert cmd[:3] == ['cutadapt', '-q', '20'] and kwargs['timeout'] == 3600
    assert cmd[cmd.index('--minimum-length') + 1] == '36'
    if adapter:
        assert cmd[cmd.index('-a') + 1] == adapter
    else:
        assert '-a' not in cmd, "'auto' must not hand cutadapt a guess it cannot make"


def test_the_summary_block_is_read_for_every_count_it_reports():
    report = tr._parse_cutadapt_stderr(CUTADAPT_SUMMARY, "in.fastq", "out.fastq")

    assert (report.total_reads, report.adapter_trimmed, report.reads_removed) == (1234, 234, 234)
    assert report.reads_trimmed == 234
    assert report.message == "Using Cutadapt (external)"
    assert (report.input_file, report.output_file) == ("in.fastq", "out.fastq")


def test_reads_trimmed_is_the_adapter_count_and_not_the_drop_count():
    """'Total processed minus reads written' is the number of reads that did NOT
    reach the output - the same reads ``reads_removed`` already reports for the
    too-short case.  On this summary the two readings differ, so the pin below
    cannot be satisfied by the old arithmetic."""
    summary = """\
Total reads processed:                 1,000
Reads with adapters:                     400 (40.0%)
Reads written (passing filters):         700 (70.0%)
Reads that were too short:               100 (10.0%)
"""
    report = tr._parse_cutadapt_stderr(summary, "in.fastq", "out.fastq")

    assert (report.total_reads, report.adapter_trimmed) == (1000, 400)
    assert report.reads_removed == 100, "too short, as cutadapt labels it"
    assert report.reads_trimmed == 400, "cut reads, not the 300 that were dropped"
    assert report.reads_trimmed != report.total_reads - 700


def test_a_summary_line_without_a_count_is_reported_not_swallowed():
    with pytest.raises(ValueError, match=re.escape(
            "unexpected cutadapt summary line: 'Total reads processed: n/a' (no count "
            "after 'Total reads processed')")):
        tr._parse_cutadapt_stderr("Total reads processed: n/a\n", "a", "b")


def test_a_summary_that_names_nothing_leaves_the_counts_at_zero():
    report = tr._parse_cutadapt_stderr("nothing familiar here\n", "a", "b")

    assert (report.total_reads, report.adapter_trimmed, report.reads_removed,
            report.reads_trimmed) == (0, 0, 0, 0)


# ─── the public entry points ─────────────────────────────────────────────────

def test_no_output_name_means_the_input_name_with_a_suffix(tmp_path, monkeypatch):
    src = write_fastq(tmp_path, "sample.fastq", [("r1", "ACGT" * 12, "IIII" * 12)])
    monkeypatch.setattr(tr, "check_trimming_tools", lambda: {'cutadapt': False})

    report = tr.trim_fastq(src)

    assert report.output_file.endswith("sample_trimmed.fastq"), report.output_file
    assert report.engine == 'builtin', report.engine


def test_a_missing_input_is_reported_before_any_engine_is_chosen(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "check_trimming_tools",
                        lambda: pytest.fail("no engine may run for a file that is not there"))

    report = tr.trim_fastq(str(tmp_path / "absent.fastq"))

    assert report.message == f"File not found: {tmp_path / 'absent.fastq'}", report
    assert report.output_file == ""


def test_cutadapt_result_is_used_when_it_is_there(tmp_path, monkeypatch):
    src = write_fastq(tmp_path, "s.fastq", [("r1", "ACGT" * 12, "IIII" * 12)])
    fake = FakeRun(returncode=0, stderr=CUTADAPT_SUMMARY)
    monkeypatch.setattr(tr, "check_trimming_tools", lambda: {'cutadapt': True})
    monkeypatch.setattr(tr.subprocess, "run", fake)

    report = tr.trim_fastq(src, str(tmp_path / "out.fastq"))

    assert report.engine == 'cutadapt' and report.total_reads == 1234


def test_a_failed_cutadapt_run_still_produces_a_real_trim(tmp_path, monkeypatch):
    """The fallback runs, and it says so: silently switching engines made a
    cutadapt that refused an argument look like a clean built-in run."""
    r1 = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"
    src = write_fastq(tmp_path, "s.fastq", [
        ("r1", r1, "I" * len(r1)),
        ("r2", "C" * 14, "!" * 14),
    ])
    out = tmp_path / "out.fastq"
    monkeypatch.setattr(tr, "check_trimming_tools", lambda: {'cutadapt': True})
    monkeypatch.setattr(tr.subprocess, "run", FakeRun(returncode=4, stderr="nope"))

    with pytest.warns(RuntimeWarning, match=r"cutadapt did not finish.*exit code 4: nope"):
        report = tr.trim_fastq(src, str(out), min_length=5)

    assert report.engine == 'builtin', "the fallback ran, so the answer is the built-in's"
    assert out.exists() and report.total_reads == 2
    assert (report.reads_trimmed, report.reads_removed) == (2, 2)
    assert out.read_text(encoding="utf-8") == "", "both reads were cut to nothing"


def test_paired_end_cutadapt_passes_both_inputs_and_both_outputs(tmp_path, monkeypatch):
    r1 = write_fastq(tmp_path, "a_1.fastq", [("r1", "ACGT" * 12, "IIII" * 12)])
    r2 = write_fastq(tmp_path, "a_2.fastq", [("r1", "TTTT" * 12, "IIII" * 12)])
    fake = FakeRun(returncode=0, stderr=CUTADAPT_SUMMARY)
    monkeypatch.setattr(tr, "check_trimming_tools", lambda: {'cutadapt': True})
    monkeypatch.setattr(tr.subprocess, "run", fake)

    report = tr.trim_pair_end(r1, r2, adapter='AGATCGGA')

    cmd, _kwargs = fake.commands[0]
    assert cmd[cmd.index('-o') + 1].endswith("a_1_trimmed.fastq")
    assert cmd[cmd.index('-p') + 1].endswith("a_2_trimmed.fastq")
    assert cmd[cmd.index('-a') + 1] == cmd[cmd.index('-A') + 1] == 'AGATCGGA'
    assert report.input_file == f"{r1} + {r2}", report.input_file


@pytest.mark.parametrize("outcome", [
    {"returncode": 7, "stderr": "cutadapt: paired input rejected"},
    {"raises": FileNotFoundError("cutadapt")},
])
def test_a_paired_run_falls_back_whenever_cutadapt_does_not_finish(tmp_path, monkeypatch,
                                                                   outcome):
    r1 = write_fastq(tmp_path, "p_1.fastq", [("r1", "ACGT" * 12, "IIII" * 12)])
    r2 = write_fastq(tmp_path, "p_2.fastq", [("r1", "TTTT" * 12, "IIII" * 12)])
    o1, o2 = tmp_path / "o1.fastq", tmp_path / "o2.fastq"
    monkeypatch.setattr(tr, "check_trimming_tools", lambda: {'cutadapt': True})
    monkeypatch.setattr(tr.subprocess, "run", FakeRun(**outcome))

    with pytest.warns(RuntimeWarning, match="cutadapt did not finish"):
        report = tr.trim_pair_end(r1, r2, str(o1), str(o2), min_length=5)

    assert report.engine == 'builtin', report
    assert o1.exists() and o2.exists(), "the built-in pair trim still wrote both mates"


def test_a_paired_auto_adapter_runs_cutadapt_without_a_guess(tmp_path, monkeypatch):
    r1 = write_fastq(tmp_path, "a_1.fastq", [("r1", "ACGT" * 12, "IIII" * 12)])
    r2 = write_fastq(tmp_path, "a_2.fastq", [("r1", "TTTT" * 12, "IIII" * 12)])
    fake = FakeRun(returncode=0, stderr=CUTADAPT_SUMMARY)
    monkeypatch.setattr(tr, "check_trimming_tools", lambda: {'cutadapt': True})
    monkeypatch.setattr(tr.subprocess, "run", fake)

    report = tr.trim_pair_end(r1, r2, adapter='auto')

    cmd, _kwargs = fake.commands[0]
    assert '-a' not in cmd and '-A' not in cmd, "cutadapt cannot guess, so nothing is passed"
    assert report.engine == 'cutadapt'


def test_a_paired_run_without_cutadapt_keeps_the_mates_synchronised(tmp_path, monkeypatch):
    records = [("r%d" % i, "AGATCGGAAGAGC" + "C" * 20, "I" * 33) for i in range(3)]
    records += [("s%d" % i, "T" * 8, "!" * 8) for i in range(2)]
    r1 = write_fastq(tmp_path, "p_1.fastq", records)
    r2 = write_fastq(tmp_path, "p_2.fastq", records)
    o1, o2 = tmp_path / "o1.fastq", tmp_path / "o2.fastq"
    monkeypatch.setattr(tr, "check_trimming_tools", lambda: {'cutadapt': False})

    report = tr.trim_pair_end(r1, r2, str(o1), str(o2), min_length=5)

    first = [line for line in o1.read_text(encoding="utf-8").splitlines() if line.startswith("@")]
    second = [line for line in o2.read_text(encoding="utf-8").splitlines() if line.startswith("@")]
    assert first == second, "a mate pair must never be split across the two files"
    assert report.engine == 'builtin', report.engine


# ─── the quality profile ────────────────────────────────────────────────────

def test_a_missing_file_is_an_error_string_not_a_crash(tmp_path):
    assert tr.analyze_fastq_quality(str(tmp_path / "gone.fastq")) == {
        "error": f"File not found: {tmp_path / 'gone.fastq'}"}


def test_a_fastq_with_empty_quality_lines_says_which_record_is_broken(tmp_path):
    path = tmp_path / "blank.fastq"
    path.write_text("@r1\nACGT\n+\n\n", encoding="utf-8")

    assert tr.analyze_fastq_quality(str(path)) == {
        "error": "malformed FASTQ in " + str(path) +
                 ": record '@r1' has an empty sequence or quality line."}


def test_the_profile_counts_every_cycle_of_every_read_it_was_given(tmp_path):
    src = write_fastq(tmp_path, "q.fastq", [
        ("r1", "ACGT", "IIII"),      # Q40
        ("r2", "AC", "!!"),          # Q0
    ])

    profile = tr.analyze_fastq_quality(src)

    assert profile['total_reads'] == 2
    assert profile['read_length_mean'] == pytest.approx(3.0), "four bases then two"
    assert profile['quality_mean'] == pytest.approx(160 / 6), "four Q40 cycles, two Q0"
    assert profile['percent_above_q20'] == pytest.approx(100 * 4 / 6)
    assert profile['percent_above_q30'] == pytest.approx(100 * 4 / 6)


def test_the_profile_stops_at_the_read_limit_it_was_given(tmp_path):
    src = write_fastq(tmp_path, "many.fastq",
                      [("r%d" % i, "ACGT", "IIII") for i in range(10)])

    assert tr.analyze_fastq_quality(src, max_reads=4)['total_reads'] == 4


# ─── the report ──────────────────────────────────────────────────────────────

def test_a_report_with_no_reads_prints_no_statistics_block():
    empty = tr.format_trim_report(tr.TrimReport(input_file="a.fastq", output_file="b.fastq"))

    assert empty.splitlines() == ["=== FASTQ Trimming Report ===", "Engine: builtin",
                                  "Input: a.fastq", "Output: b.fastq"]


def test_a_report_with_reads_prints_the_counts_and_the_quality_change():
    text = tr.format_trim_report(tr.TrimReport(
        input_file="a.fastq", output_file="b.fastq", engine="python",
        total_reads=1234, reads_trimmed=20, reads_removed=14, adapter_trimmed=6,
        avg_quality_before=28.46, avg_quality_after=33.0, message="done"))

    assert "Status: done" in text
    assert "Total reads: 1,234" in text
    assert "Mean quality before: 28.5" in text and "after: 33.0" in text
