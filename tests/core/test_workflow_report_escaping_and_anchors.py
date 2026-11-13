"""biosuite.core.workflow.report: one escape per value, one id per section.

``ReportSection`` escapes its title; the report generators escaped it too, so a
step named ``alpha & co`` came out as ``alpha &amp;amp; co`` on the page.  The
execution log was escaped and then had ``<br>`` added, which the escaping turned
back into text, and every section with the same title — or none, as for a plot
without a caption — shared one ``id``, which is invalid HTML and made the table
of contents point every link at the same place.
"""
import re

import pytest

from biosuite.core.workflow.pipeline import Pipeline
from biosuite.core.workflow.report import (
    HTMLReport,
    ReportSection,
    generate_pipeline_report,
)


@pytest.fixture
def report():
    return HTMLReport("T")


def test_a_title_is_escaped_exactly_once(report):
    report.add_section("alpha & co", "body")
    text = report.to_html()
    assert "<h2>alpha &amp; co</h2>" in text
    assert "&amp;amp;" not in text


def test_markup_in_a_title_stays_text(report):
    report.add_section('A <script>alert(1)</script>', "body")
    text = report.to_html()
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text


def test_two_sections_with_one_title_get_two_ids(report):
    report.add_section("Results", "one")
    report.add_section("Results", "two")
    text = report.to_html()
    assert text.count('id="results"') == 1
    assert 'id="results-2"' in text
    hrefs = re.findall(r'<a href="([^"]+)"', text)
    assert hrefs == ["#results", "#results-2"], "every link reaches its own section"


def test_sections_without_a_title_do_not_share_the_empty_slug(report):
    report.add_section("", "one")
    report.add_section("", "two")
    text = report.to_html()
    assert 'id="section"' in text
    assert 'id="section-2"' in text
    assert text.count('id="section"') == 1


def test_the_heading_level_is_a_number_between_one_and_six():
    for level in (0, 1.5, "2", True, None, 7, -1):
        with pytest.raises(ValueError, match="level must be an integer from 1 to 6"):
            ReportSection("t", "c", level)


@pytest.mark.parametrize("level", [1, 3, 6])
def test_a_valid_level_renders_the_heading_it_names(level):
    section = ReportSection("t", "c", level)
    assert section.to_html().startswith(f"<h{level}>t</h{level}>")


def test_a_string_level_can_no_longer_break_out_of_the_heading_tag():
    """Measured before the fix: '<h' + the string + '>' emitted two live
    ``<script>`` tags, and building the whole report died in the TOC instead."""
    with pytest.raises(ValueError, match="level must be an integer"):
        HTMLReport("T").add_section("t", "c", level='2><script>alert(1)</script>')


def test_a_report_with_no_sections_has_no_contents_block(report):
    assert "Contents" not in report.to_html()


def test_the_pipeline_report_writes_the_log_with_real_line_breaks(tmp_path):
    p = Pipeline("rep")
    p.add_step("ok", lambda: 1)
    p.add_step("bad", lambda: 1 / 0)
    p.run(stop_on_error=False)
    path = str(tmp_path / "r.html")
    generate_pipeline_report(p, output_path=path)
    text = open(path, encoding="utf-8").read()
    log_block = text.split('id="execution-log"')[1]
    assert "<br>" in log_block
    assert "&lt;br&gt;" not in log_block
    assert "finished in" in log_block, "the run's own closing line is in the log"
    assert "&#x27;rep&#x27;" in log_block, "quotes in log text are escaped, not raw"


def test_the_step_title_in_a_generated_report_is_not_double_escaped(tmp_path):
    p = Pipeline("rep")
    p.add_step("alpha & co", lambda: 1)
    p.run()
    path = str(tmp_path / "r.html")
    generate_pipeline_report(p, output_path=path)
    text = open(path, encoding="utf-8").read()
    assert "<h2>alpha &amp; co</h2>" in text
    assert "&amp;amp;" not in text


def test_a_step_that_never_ran_is_not_coloured_as_an_error(tmp_path):
    p = Pipeline("halt")
    p.add_step("a", lambda: 1)
    p.add_step("bad", lambda: 1 / 0)
    p.add_step("later", lambda: 3)
    p.run(stop_on_error=True)
    path = str(tmp_path / "r.html")
    generate_pipeline_report(p, output_path=path)
    text = open(path, encoding="utf-8").read()
    block = text.split('id="later"')[1].split("</div>")[0]
    assert 'class="error"' not in block
    assert "Status: pending" in block
    assert 'class="success"' in text.split('id="a"')[1].split("</div>")[0]


def test_the_error_text_of_a_failed_step_still_reaches_the_report(tmp_path):
    p = Pipeline("err")
    p.add_step("bad", lambda: 1 / 0)
    p.run()
    path = str(tmp_path / "r.html")
    generate_pipeline_report(p, output_path=path)
    text = open(path, encoding="utf-8").read()
    assert "division by zero" in text
