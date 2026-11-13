"""``trimming.py`` refuses a FASTQ file it does not understand, and says which record
is wrong.

The trimmer is a file-rewriting tool, so the failures that matter are the ones that put a
broken record back on disk or quietly stop reading the rest of the file.  Measured on the
pre-cluster code first: a record of 4 bases over a 6-character quality line was written
out **unchanged** because nothing compared the two lengths; a quality line running into
the next header yielded a report of "1 read processed" beside one mangled record while two
more reads vanished; a stray byte was scored as Q-32; and a lone missing mate raised
``FileNotFoundError`` where the single-end entry point returns a report.
"""
import re

import pytest

from biosuite.core import trimming as tr


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_a_record_whose_quality_line_is_the_wrong_length_is_refused_not_rewritten(tmp_path):
    src = _write(tmp_path, "in.fastq", "@r1\nACGT\n+\nIIIIII\n@r2\nACGTACGT\n+\nIIIIIIII\n")
    out = tmp_path / "out.fastq"

    with pytest.raises(ValueError, match=re.escape(
            f"malformed FASTQ in {src}: record '@r1' has 4 bases but 6 quality scores.")):
        tr.trim_fastq(src, str(out), quality_threshold=10, min_length=2)
    assert out.stat().st_size == 0, "nothing may be written from the refused record"


def test_a_record_that_does_not_start_with_an_at_sign_is_refused(tmp_path):
    """A file handed to the trimmer without '@' headers - a FASTA, or a quality file -
    used to be read as a sequence line and quietly produced nothing at all."""
    src = _write(tmp_path, "in.fastq", "ACGTACGT\n+\nIIIIIIII\n")

    with pytest.raises(ValueError, match=re.escape(
            f"malformed FASTQ in {src}: expected '@' at the start of a record, "
            "found 'ACGTACGT'.")):
        tr.trim_fastq(src, str(tmp_path / "out.fastq"), quality_threshold=10, min_length=2)


def test_a_missing_separator_line_stops_the_run_with_a_message(tmp_path):
    src = _write(tmp_path, "in.fastq", "@r1\nACGT\nIIII\nACGT\n")

    with pytest.raises(ValueError, match=re.escape(
            f"malformed FASTQ in {src}: expected a '+' separator after '@r1', "
            "found 'IIII'.")):
        tr.trim_fastq(src, str(tmp_path / "out.fastq"), quality_threshold=10, min_length=2)


def test_a_quality_line_running_into_the_next_header_is_caught_by_the_length_check(tmp_path):
    src = _write(tmp_path, "in.fastq", "@r1\nACGT\n+\nIIII@r2\nACGT\n+\nIIII\n")

    with pytest.raises(ValueError, match=re.escape(
            f"malformed FASTQ in {src}: record '@r1' has 4 bases but 7 quality scores.")):
        tr.trim_fastq(src, str(tmp_path / "out.fastq"), quality_threshold=10, min_length=2)


def test_a_quality_character_below_the_phred_33_range_is_refused(tmp_path):
    """The same class of nonsense ``sequence.quality_stats`` refuses: measured before,
    the byte was scored as a negative quality and averaged into the profile."""
    src = _write(tmp_path, "in.fastq", "@r1\nACGT\n+\nII\x01I\n")

    with pytest.raises(ValueError, match=re.escape(
            "record '@r1' carries 1 character(s) outside the Phred+33 range, starting "
            "with '\\x01' (Q-32).")):
        tr.trim_fastq(src, str(tmp_path / "out.fastq"), quality_threshold=10, min_length=2)


def test_an_empty_sequence_or_quality_line_is_named_not_skipped(tmp_path):
    """``analyze_fastq_quality`` keeps its own convention - a dict with an ``error``
    key - but the message now names the record instead of saying "no data"."""
    src = _write(tmp_path, "in.fastq", "@r1\nACGT\n+\n\n")

    assert tr.analyze_fastq_quality(src) == {
        "error": f"malformed FASTQ in {src}: record '@r1' has an empty sequence or "
                 "quality line."}


def test_trim_pair_end_reports_a_missing_mate_instead_of_raising(tmp_path):
    present = _write(tmp_path, "r1.fastq",
                     "".join(f"@r{i}\nACGTACGT\n+\nIIIIIIII\n" for i in range(3)))
    missing = str(tmp_path / "r2.fastq")

    for report in (tr.trim_pair_end(present, missing), tr.trim_pair_end(missing, present)):
        assert report.message == f"File not found: {missing}"
        assert report.output_file == ""
        assert report.total_reads == 0


def test_paired_files_that_disagree_in_length_are_refused(tmp_path):
    """Dropping one mate of a pair corrupts every downstream tool, and files with
    different record counts cannot be trimmed in lockstep at all."""
    good = "".join(f"@r{i}\nACGTACGT\n+\nIIIIIIII\n" for i in range(3))
    r1 = _write(tmp_path, "a.fastq", good)
    r2 = _write(tmp_path, "b.fastq", "@r0\nACGTACGT\n+\nIIIIIIII\n")

    with pytest.raises(ValueError, match=re.escape("paired FASTQ files disagree")):
        tr.trim_pair_end(r1, r2, str(tmp_path / "o1.fastq"), str(tmp_path / "o2.fastq"))


def test_the_quality_profile_refuses_a_record_it_cannot_read(tmp_path):
    """Same validation, same message - reported through this function's own
    ``{"error": ...}`` convention instead of raising."""
    src = _write(tmp_path, "bad.fastq", "@r1\nACGT\n+\nIII\n")
    assert tr.analyze_fastq_quality(src) == {
        "error": f"malformed FASTQ in {src}: record '@r1' has 4 bases but 3 quality scores."}


def test_a_short_read_that_is_valid_is_removed_rather_than_reported_as_malformed(tmp_path):
    """Positive control: the validator must not refuse legitimate short records, which
    are the normal case for a read that quality trimming cut."""
    src = _write(tmp_path, "in.fastq", "@r1\nAC\n+\nII\n@r2\nACGTACGT\n+\nIIIIIIII\n")
    out = tmp_path / "out.fastq"
    report = tr.trim_fastq(src, str(out), quality_threshold=10, min_length=4,
                           adapter='polyg')

    assert (report.total_reads, report.reads_removed) == (2, 1)
    assert out.read_text(encoding="utf-8").count('@') == 1
