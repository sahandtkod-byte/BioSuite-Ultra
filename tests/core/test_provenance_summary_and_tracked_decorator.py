"""Provenance: the text summary, the error branch of the HTML export, the
`@tracked` decorator and the process-wide tracker.

`test_provenance_hardening.py` already covers the thread-safety and exotic
parameter work these paths grew out of.  What had never executed was the human
report (`summary()`), the ``step-error`` div, the decorator that most callers
actually use to record steps, and the global tracker that `record_step()` writes
into.
"""
import json

import pytest

from biosuite.core import provenance as pv
from biosuite.core.provenance import ProvenanceTracker, _json_default, tracked


@pytest.fixture()
def tracker(tmp_path):
    t = ProvenanceTracker(db_path=str(tmp_path / "prov.db"), session_id="sess_x")
    yield t
    t.close()


@pytest.fixture()
def isolated_global_tracker():
    """Give the module-level singleton a fresh slot and put it back afterwards."""
    saved = pv._default_tracker
    pv._default_tracker = None
    yield
    pv._default_tracker = saved


# ── parameter coercion ───────────────────────────────────────────────────────

def test_sets_and_tuples_become_lists_so_the_json_is_strict():
    assert sorted(_json_default({3, 1, 2})) == [1, 2, 3]
    assert _json_default((1, 2)) == [1, 2]
    assert json.loads(pv.dumps_params({"genes": frozenset(["a"])})) == {"genes": ["a"]}


def test_an_object_with_no_numeric_or_sequence_form_is_named_not_crashed():
    class Opaque:
        pass

    assert _json_default(Opaque()) == "<Opaque>"
    assert json.loads(pv.dumps_params({"x": Opaque()})) == {"x": "<Opaque>"}


def test_a_numpy_shaped_parameter_round_trips_through_the_database(tracker):
    import numpy as np
    tracker.record("core", "align", {"threshold": np.float64(0.75),
                                     "positions": np.array([1, 2, 3])})
    step = tracker.get_steps()[0]
    assert step.params == {"threshold": 0.75, "positions": [1, 2, 3]}


# ── summary() ────────────────────────────────────────────────────────────────

def test_summary_of_a_session_that_recorded_nothing(tracker):
    text = tracker.summary()
    assert "=== Provenance Summary ===" in text
    assert "Session: sess_x" in text
    assert "Total steps: 0" in text
    assert "Total execution time: 0ms" in text
    assert "Success: 0" in text
    assert "Errors: 0" in text
    assert "Skipped: 0" in text
    assert "Other:" not in text, "no legacy rows, so no Other counter"
    assert "-" * 80 in text
    assert text.count("\n") == 9, "the table header, no rows"


def test_summary_prints_one_row_per_step_in_order(tracker):
    tracker.record("sequence", "gc_content", {"seq": "ATCG"}, "50.0%",
                   execution_time_ms=12, engine="builtin")
    tracker.record("alignment", "pairwise", {"k": 5}, "score 42",
                   execution_time_ms=340, engine="needleman", status="error",
                   error_message="diverged")
    text = tracker.summary()
    lines = text.splitlines()
    assert lines[2] == "Total steps: 2"
    assert lines[3] == "Total execution time: 352ms"
    assert lines[4] == "Success: 1"
    assert lines[5] == "Errors: 1"
    assert lines[6] == "Skipped: 0"
    assert len(lines) == 12, "blank line + table header + rule + 2 rows"
    row1, row2 = lines[-2], lines[-1]
    assert row1.startswith("1    sequence")
    assert "gc_content" in row1 and "12ms" in row1 and row1.endswith("success")
    assert row2.startswith("2    alignment")
    assert "needleman" in row2 and "340ms" in row2 and row2.endswith("error")


def test_summary_is_plain_text_and_does_not_print_parameters(tracker):
    tracker.record("seq", "parse", {"name": "<b>note</b>"}, "ok")
    text = tracker.summary()
    assert "parse" in text
    assert "<b>note</b>" not in text, "the summary is a table, not a parameter dump"


# ── the error branch of the HTML export ──────────────────────────────────────

def test_html_export_shows_the_failure_and_escapes_it(tracker, tmp_path):
    tracker.record("alignment", "pairwise", {"k": 5}, "",
                   status="error", error_message="<script>bad & ugly</script>")
    out = tracker.export_html(str(tmp_path / "p.html"))
    with open(out, encoding="utf-8") as fh:
        html = fh.read()
    assert '<div class="step-error">Error: ' in html
    assert "<script>bad" not in html
    assert "&lt;script&gt;bad &amp; ugly&lt;/script&gt;" in html
    assert "status-error" in html and "✗" in html


def test_html_export_omits_the_error_div_for_a_successful_step(tracker, tmp_path):
    """The stylesheet always mentions step-error, so only the div itself proves
    whether a failure block was emitted."""
    tracker.record("sequence", "gc_content", {"seq": "AT"}, "50%")
    html = open(tracker.export_html(str(tmp_path / "ok.html")),
                encoding="utf-8").read()
    assert '<div class="step-error">' not in html
    assert "status-ok" in html and "✓" in html


def test_html_export_writes_nothing_for_a_step_without_params(tracker, tmp_path):
    tracker.record("sequence", "reverse_complement", None, "GTCA")
    html = open(tracker.export_html(str(tmp_path / "e.html")),
                encoding="utf-8").read()
    assert "Params: <code>{}</code>" in html


# ── the tracked() decorator ──────────────────────────────────────────────────

def test_tracked_records_a_successful_call(tracker):
    @tracked(tracker)
    def add(a, b, scale=1):
        return (a + b) * scale

    assert add(2, 3, scale=2) == 10
    step = tracker.get_steps()[0]
    assert step.function == "add"
    assert step.module == "test_provenance_summary_and_tracked_decorator"
    assert step.params == {"arg0": "2", "arg1": "3", "scale": "2"}
    assert step.result_summary == "10"
    assert step.status == "success"
    assert step.error_message == ""
    assert isinstance(step.execution_time_ms, int)


def test_tracked_records_a_failing_call_and_still_raises(tracker):
    @tracked(tracker)
    def boom():
        raise ValueError("no good")

    with pytest.raises(ValueError, match="no good"):
        boom()
    step = tracker.get_steps()[0]
    assert step.status == "error"
    assert step.error_message == "no good"
    assert step.result_summary == ""
    assert step.params == {}


def test_tracked_marks_the_values_it_truncates(tracker):
    """The stored text is capped at 100 characters and a cut is now visible: the
    HTML report tells the reader to replay these parameters, so a 100-character
    prefix of a 500-character sequence must not look like the whole value."""
    @tracked(tracker)
    def takes(blob):
        return len(blob)

    takes("A" * 500)
    takes("short")
    short, long_value = tracker.get_steps()[1], tracker.get_steps()[0]
    assert long_value.params["arg0"] == "A" * 97 + "..."
    assert len(long_value.params["arg0"]) == 100
    assert short.params["arg0"] == "short", "a value under the cap is stored verbatim"


def test_a_falsy_result_is_recorded_as_the_value_it_returned(tracker):
    """`str(result) if result else "None"` used to collapse 0, '' and [] to "None",
    so the audit line could not tell a function that answered zero from one that
    answered nothing. Only an actual None is written as "None"."""

    @tracked(tracker)
    def zero():
        return 0

    @tracked(tracker)
    def empty():
        return []

    @tracked(tracker)
    def nothing():
        return None

    assert zero() == 0
    assert empty() == []
    assert nothing() is None
    summaries = [s.result_summary for s in tracker.get_steps()]
    assert summaries == ["0", "[]", "None"]


def test_tracked_without_a_tracker_runs_the_function_and_records_nothing():
    @tracked()
    def multiply(a, b):
        return a * b

    assert multiply(6, 7) == 42


def test_tracked_without_a_tracker_still_propagates_a_failure():
    """The decorator is used bare in example code, so the no-tracker path has to
    survive the error branch too: nothing is recorded, and the exception is not
    swallowed."""
    @tracked()
    def boom():
        raise KeyError("missing column")

    with pytest.raises(KeyError, match="missing column"):
        boom()


def test_tracked_keeps_the_wrapped_function_metadata(tracker):
    @tracked(tracker)
    def documented(x):
        """Docstring survives."""
        return x

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Docstring survives."
    assert documented.__wrapped__ is not None


def test_each_tracked_call_gets_its_own_step_id(tracker):
    @tracked(tracker)
    def noop():
        return "x"

    noop(); noop(); noop()
    assert [s.step_id for s in tracker.get_steps()] == [1, 2, 3]


# ── the process-wide tracker ────────────────────────────────────────────────

def test_the_global_tracker_is_created_on_first_use(isolated_global_tracker):
    t = pv.get_global_tracker()
    assert isinstance(t, ProvenanceTracker)
    assert t.db_path is None, "the default tracker keeps its steps in memory"
    assert pv.get_global_tracker() is t


def test_setting_the_global_tracker_redirects_record_step(isolated_global_tracker):
    replacement = ProvenanceTracker(session_id="sess_global")
    pv.set_global_tracker(replacement)
    assert pv.get_global_tracker() is replacement
    step = pv.record_step("sequence", "gc_content", {"seq": "GCGC"}, "50%",
                          execution_time_ms=7, engine="builtin")
    assert step.module == "sequence"
    assert replacement.get_steps() == [step]
    assert pv.record_step("alignment", "msa").status == "success"
    assert len(replacement.get_steps()) == 2


def test_record_step_defaults_match_the_tracker_signature(isolated_global_tracker):
    replacement = ProvenanceTracker(session_id="sess_defaults")
    pv.set_global_tracker(replacement)
    step = pv.record_step("mod", "func")
    assert step.params == {}
    assert step.result_summary == ""
    assert step.execution_time_ms == 0
    assert step.engine == "builtin"
    assert step.status == "success"
    assert step.session_id == "sess_defaults"


def test_a_reopened_database_keeps_its_history(tmp_path):
    path = str(tmp_path / "audit.db")
    first = ProvenanceTracker(db_path=path, session_id="sess_reopen")
    first.record("sequence", "gc_content", {"seq": "AT"}, "50%")
    first.close()

    second = ProvenanceTracker(db_path=path, session_id="sess_reopen")
    steps = second.get_steps()
    assert len(steps) == 1
    assert steps[0].params == {"seq": "AT"}
    # A fresh session sees none of it: steps are filtered by session_id.
    other = ProvenanceTracker(db_path=path, session_id="sess_other")
    assert other.get_steps() == []
    second.close()
    other.close()


def test_json_export_round_trips_through_the_file(tracker, tmp_path):
    tracker.record("sequence", "gc_content", {"seq": "AT", "set": {1}}, "50%",
                   execution_time_ms=3)
    path = tracker.export_json(str(tmp_path / "p.json"))
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["session_id"] == "sess_x"
    assert data["total_steps"] == 1
    assert data["steps"][0]["module"] == "sequence"
    assert data["steps"][0]["params"] == {"seq": "AT", "set": [1]}
    assert "export_time" in data
