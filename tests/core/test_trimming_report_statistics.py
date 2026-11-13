"""The numbers ``trimming.py`` puts in its report: what each average is taken over, and
whether a field its own docstring advertises exists at all.

Measured on the pre-cluster code: ``avg_quality_after`` averaged the reads the length
filter had just thrown away; the paired engine left both quality means at their 0.0
default, so ``format_trim_report`` printed "Mean quality before: 0.0" over a file of Q40
reads; and the ``TrimReport`` docstring described ``reads_in``, ``reads_out`` and
``bases_trimmed`` - "Total bases removed by quality/adapter cuts" - three fields the
dataclass never had, in a module that never counted bases.
"""
import re

import pytest

from biosuite.core import trimming as tr

FASTQ8 = "".join(f"@r{i}\nACGTACGT\n+\n{'I' * 8}\n" for i in range(3))


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_the_after_quality_average_describes_the_file_that_was_written(tmp_path):
    """The dropped read is Q40 and every kept base is Q20, so an average that includes
    it is measurably not the output file's mean."""
    src = _write(tmp_path, "in.fastq",
                 "@r1\nACGTACGTAC\n+\n" + "5" * 10 + "\n"      # Q20, kept (10 >= 8)
                 "@r2\nCACTG\n+\nIIIII\n")                     # Q40, dropped: 5 < 8
    out = tmp_path / "out.fastq"
    report = tr.trim_fastq(src, str(out), quality_threshold=20, min_length=8)

    assert (report.total_reads, report.reads_removed, report.reads_trimmed) == (2, 1, 0)
    assert report.avg_quality_before == pytest.approx(400 / 15)
    assert report.avg_quality_after == pytest.approx(20.0), "the kept reads only"
    assert out.read_text(encoding="utf-8").count('@') == 1


def test_the_paired_report_prints_the_means_it_measured(tmp_path):
    src = _write(tmp_path, "p.fastq", FASTQ8)
    report = tr.trim_pair_end(src, src, str(tmp_path / "o1.fastq"), str(tmp_path / "o2.fastq"),
                              quality_threshold=20, min_length=4)

    assert report.total_reads == 6
    assert '3/3 pairs kept' in report.message
    assert report.avg_quality_before == pytest.approx(40.0)
    assert report.avg_quality_after == pytest.approx(40.0)

    text = tr.format_trim_report(report)
    assert "Mean quality before: 40.0" in text and "Mean quality after: 40.0" in text
    assert "before: 0.0" not in text


def test_the_trim_report_docstring_names_only_fields_that_exist():
    fields = set(tr.TrimReport.__dataclass_fields__)
    documented = re.findall(r"^\s{4}([\w/]+):", tr.TrimReport.__doc__, re.MULTILINE)
    names = {part for entry in documented for part in entry.split('/')}
    assert names, "the docstring stopped describing the fields"
    assert names <= fields, f"documented but absent: {sorted(names - fields)}"
    assert 'bases_trimmed' not in tr.TrimReport.__doc__
    assert 'reads_in' not in tr.TrimReport.__doc__


def test_the_quality_profile_counts_the_records_it_actually_profiled(tmp_path):
    """Measured before: a record with an empty quality line still incremented
    ``total_reads`` without contributing a length, so a three-record file reported
    ``total_reads == 3`` over a ``read_length_mean`` taken from two reads."""
    src = _write(tmp_path, "q.fastq",
                 "@r1\nACGT\n+\nIIII\n"
                 "@r2\nAC\n+\n!!\n"
                 "@r3\nGGGGGG\n+\nIIIIII\n")
    profile = tr.analyze_fastq_quality(src)
    assert profile['total_reads'] == 3
    assert profile['read_length_mean'] == pytest.approx((4 + 2 + 6) / 3)
    assert profile['quality_mean'] == pytest.approx((4 * 40 + 2 * 0 + 6 * 40) / 12)
    assert profile['percent_above_q20'] == pytest.approx(100 * 10 / 12)

    limited = tr.analyze_fastq_quality(src, max_reads=2)
    assert limited['total_reads'] == 2
    assert limited['read_length_mean'] == pytest.approx(3.0)


def test_the_quality_profile_of_an_empty_file_says_there_is_no_data(tmp_path):
    src = _write(tmp_path, "empty.fastq", "")
    assert tr.analyze_fastq_quality(src) == {"error": "No quality data extracted."}
