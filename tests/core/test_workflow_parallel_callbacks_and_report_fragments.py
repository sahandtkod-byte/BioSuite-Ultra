"""Parallel bookkeeping and report fragments in ``biosuite.core.workflow``.

``BatchProcessor`` and ``Pipeline`` switch to a thread pool when given more than
one worker, and that path keeps its own counters: the progress callback, the
failure lines and the per-step context hand-off.  The report helpers are the
other half — what an empty table, an untitled paragraph, a missing table of
contents or an errored job turn into.  Every string below was read off the
running code.
"""
import pandas as pd

from biosuite.core.workflow.batch import BatchProcessor
from biosuite.core.workflow.pipeline import Pipeline
from biosuite.core.workflow.report import (
    HTMLReport, _df_to_html_table, generate_batch_report,
)


def _mixed_batch(name="b"):
    """Two samples, one of which always fails."""
    def work(sample_id):
        if sample_id == "bad":
            raise ValueError("<boom> failed")
        return sample_id.upper()

    processor = BatchProcessor(name)
    processor.add_samples(["good", "bad"], work)
    return processor


# ─── the parallel path keeps its own books ───────────────────────────────────

def test_a_parallel_batch_reports_progress_once_per_sample():
    seen = []
    processor = BatchProcessor("p")
    processor.add_samples([f"s{i}" for i in range(6)], lambda sid: sid.upper())

    processor.run(max_workers=3,
                  progress_callback=lambda done, total, sid: seen.append((done, total, sid)))

    assert sorted(done for done, _t, _s in seen) == [1, 2, 3, 4, 5, 6]
    assert {sid for _d, _t, sid in seen} == {f"s{i}" for i in range(6)}
    assert {total for _d, total, _s in seen} == {6}


def test_the_sequential_path_reports_progress_too():
    """The callback is part of the contract of run(), not of one of its modes."""
    seen = []
    processor = BatchProcessor("s")
    processor.add_samples(["a", "b", "c"], lambda sid: sid.upper())
    processor.run(max_workers=1,
                  progress_callback=lambda done, total, sid: seen.append((done, total, sid)))
    assert seen == [(1, 3, "a"), (2, 3, "b"), (3, 3, "c")]


def test_a_parallel_failure_is_logged_the_way_a_sequential_one_is():
    processor = _mixed_batch()

    processor.run(max_workers=2)

    assert [sid for sid, _err in processor.get_failures()] == ["bad"]
    assert "  bad FAILED: <boom> failed" in processor._log
    done = [line for line in processor._log if " done in " in line]
    assert len(done) == 1 and done[0].startswith("  good done in ")


def test_summary_prints_the_error_beneath_the_failed_sample():
    processor = _mixed_batch()
    processor.run(max_workers=1)

    lines = processor.summary().splitlines()

    assert lines[0] == "Batch: b"
    assert "Samples: 2" in lines
    assert "Completed: 1" in lines
    assert "Failed: 1" in lines
    failed = [line for line in lines if line.startswith("  [X] bad (")]
    succeeded = [line for line in lines if line.startswith("  [+] good (")]
    assert len(failed) == 1 and len(succeeded) == 1
    assert "      <boom> failed" in lines


def test_a_parallel_step_that_returns_nothing_is_recorded_but_not_chained():
    pipeline = Pipeline("none")
    pipeline.add_step("silent", lambda: None)
    pipeline.add_step("loud", lambda: 5)

    pipeline.run(max_workers=2, stop_on_error=False)

    assert pipeline.results == {"silent": None, "loud": 5}
    # a None result is not a value the next step may read from the context
    assert "silent" not in pipeline.context
    assert pipeline.context["loud"] == 5


# ─── report fragments ────────────────────────────────────────────────────────

def test_a_table_without_rows_says_so():
    assert _df_to_html_table(None) == "<p>No data</p>"
    assert _df_to_html_table(pd.DataFrame()) == "<p>No data</p>"


def test_a_table_only_object_is_emitted_as_is():
    class TableOnly:
        """Has ``to_html`` but no ``head`` — the guard must not assume pandas."""

        def __init__(self):
            self.kwargs = None

        def to_html(self, **kwargs):
            self.kwargs = kwargs
            return "<table>duck</table>"

    table = TableOnly()
    assert _df_to_html_table(table) == "<table>duck</table>"
    assert table.kwargs == {"classes": "data-table", "index": False, "border": 0}


def test_long_tables_are_cut_to_the_documented_row_limit():
    frame = pd.DataFrame({"gene": [f"g{i}" for i in range(60)]})

    capped = _df_to_html_table(frame)         # max_rows defaults to 50
    assert capped.count("<tr>") == 50
    assert "g0" in capped and "g50" not in capped

    trimmed = _df_to_html_table(frame, max_rows=5)
    assert trimmed.count("<tr>") == 5
    assert "g4" in trimmed and "g5" not in trimmed


def test_untitled_text_continues_the_section_it_follows():
    report = HTMLReport("T")
    report.add_section("Results", "<p>one</p>")
    report.add_text("two")

    assert len(report.sections) == 1
    assert report.sections[0].content == "<p>one</p>\n<p>two</p>"

    alone = HTMLReport("T")
    alone.add_text("orphan")
    assert [(s.title, s.content) for s in alone.sections] == [("Report", "<p>orphan</p>")]


def test_a_report_without_sections_carries_no_contents_block():
    assert HTMLReport("T")._build_toc() == ""
    assert '<div class="toc">' not in HTMLReport("T").to_html()

    nested = HTMLReport("T")
    nested.add_section("Top", "<p>a</p>")
    nested.add_section("Deep", "<p>b</p>", level=3)
    toc = nested._build_toc()
    assert toc.startswith('<div class="toc"><h3>Contents</h3><ul>')
    assert '<a href="#top">Top</a>' in toc
    assert toc.index("Top") < toc.index("Deep")


def test_the_batch_report_shows_and_escapes_a_failed_sample(tmp_path):
    processor = _mixed_batch()
    processor.run(max_workers=1)

    path = generate_batch_report(processor, str(tmp_path / "batch.html"))
    document = open(path, encoding="utf-8").read()

    assert "<boom> failed" not in document
    assert "&lt;boom&gt; failed" in document
    assert 'class="error"' in document
    assert 'class="success"' in document
    assert ('<div class="stat-value">1</div>'
            '<div class="stat-label">Failed</div>') in document
    assert ('<div class="stat-value">2</div>'
            '<div class="stat-label">Total</div>') in document


def test_a_plot_without_a_caption_adds_no_empty_paragraph():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    report = HTMLReport("plots")
    report.add_plot(fig, title="Curve")
    block = report.sections[0].content
    assert "<em>" not in block
    assert block.endswith("</div>")

    fig2, ax2 = plt.subplots()
    ax2.plot([1], [1])
    captioned = HTMLReport("plots")
    captioned.add_plot(fig2, title="Curve", caption="Fig. 1")
    assert "<p><em>Fig. 1</em></p>" in captioned.sections[0].content


def test_titled_text_opens_its_own_section_rather_than_joining_the_last():
    report = HTMLReport("T")
    report.add_section("Existing", "body")
    report.add_text("appended", title="Own")
    assert [s.title for s in report.sections] == ["Existing", "Own"]
    assert report.sections[-1].content == "<p>appended</p>"
    assert "<p>appended</p>" not in report.sections[0].content
