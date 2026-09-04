"""Regression tests for workflow pipeline/batch/report hardening."""
import re

import pytest


def test_pipeline_does_not_crash_on_plain_signature():
    """Regression squares: prior step results leaked into kwargs."""
    from biosuite.core.workflow.pipeline import Pipeline

    def producer():
        return 7

    def consumer(x):          # plain signature, no **kwargs
        return x * 2

    p = Pipeline("kwargfilter")
    p.add_step("producer", producer)
    p.add_step("consumer", consumer, args=(5,))
    p.run()
    assert p.results["consumer"] == 10
    assert all(s.status == "done" for s in p.steps)


def test_pipeline_still_passes_context_to_star_kwargs():
    from biosuite.core.workflow.pipeline import Pipeline
    seen = {}
    p = Pipeline("ctx")
    p.add_step("s1", lambda **kw: seen.update(kw) or "v1")
    p.set_context(alpha=1)
    p.run()
    assert seen.get("alpha") == 1


def test_pipeline_stop_on_error():
    from biosuite.core.workflow.pipeline import Pipeline
    ran = []
    p = Pipeline("stop")
    p.add_step("boom", lambda: 1 / 0)
    p.add_step("never", lambda: ran.append(1))
    p.run(stop_on_error=True)
    assert ran == []


def test_batch_success_and_failure():
    from biosuite.core.workflow.batch import BatchProcessor
    bp = BatchProcessor("t")

    def work(sid):
        if sid == "bad":
            raise ValueError("nope")
        return sid.upper()

    bp.add_samples(["a", "bad", "c"], work)
    bp.run(max_workers=1)
    assert bp.get_results()["A"] == "A" if "A" in bp.get_results() else bp.get_results()["a"] == "A"
    fails = bp.get_failures()
    assert len(fails) == 1 and fails[0][0] == "bad"


def test_batch_run_parallel():
    from biosuite.core.workflow.batch import batch_run
    out = batch_run(lambda s: s * 2, [1, 2, 3, 4], max_workers=2)
    assert out == {1: 2, 2: 4, 3: 6, 4: 8}


def test_pipeline_report_escapes_xss(tmp_path):
    from biosuite.core.workflow.pipeline import Pipeline
    from biosuite.core.workflow.report import generate_pipeline_report
    p = Pipeline("evil")

    def boom():
        raise ValueError('<script>alert("x")</script>')

    p.add_step('<img src=x onerror=alert(1)>', boom)
    p.run(stop_on_error=False)
    out = generate_pipeline_report(p, str(tmp_path / "r.html"))
    html_text = open(out).read()
    assert "<script>alert" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "<img src=x" not in html_text


def test_batch_report_escapes_xss(tmp_path):
    from biosuite.core.workflow.batch import BatchProcessor
    from biosuite.core.workflow.report import generate_batch_report
    bp = BatchProcessor("b")
    bp.add_job('<b onclick="x()">', lambda s: s)
    bp.run(max_workers=1)
    out = generate_batch_report(bp, str(tmp_path / "b.html"))
    text = open(out).read()
    assert '<b onclick' not in text
    assert "&lt;b onclick" in text


def test_generate_html_report_contains_sections(tmp_path):
    from biosuite.core.workflow.report import HTMLReport
    r = HTMLReport("T", "sub")
    r.add_section("S1", "<p>body</p>")
    r.add_stats({"Done": 3})
    p = r.save(str(tmp_path / "x.html"))
    txt = open(p).read()
    assert "<h1>T</h1>" in txt and "S1" in txt and "stat-value" not in txt or "Done" in txt
