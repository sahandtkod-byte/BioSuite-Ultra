"""Parameter refusals, session reopening and the audit fidelity of ``core/provenance.py``.

The "before" figures quoted in the docstrings were measured against the untouched HEAD
copy of the module (executed in memory from `git show HEAD:biosuite/core/provenance.py`),
and the race counts by running the real eight-thread probe three times.
"""
import json
import threading
from dataclasses import dataclass

import pytest

from biosuite.core import provenance as pv
from biosuite.core.provenance import (
    ProvenanceTracker, _brief, _json_default, tracked,
)


@pytest.fixture()
def tracker(tmp_path):
    t = ProvenanceTracker(db_path=str(tmp_path / "prov.db"), session_id="sess_pin")
    yield t
    t.close()


@pytest.fixture()
def isolated_global_tracker():
    """Give the module-level singleton a fresh slot and put it back afterwards."""
    saved = pv._default_tracker
    pv.set_global_tracker(None)
    yield
    pv._default_tracker = saved


# ── status validation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["sucess", "Success", "Skipped", "", None, "warning"])
def test_record_refuses_a_status_that_is_not_one_of_the_three_documented(tracker, bad):
    """HEAD stored any string.  The HTML timeline marks every status that is not
    'success' with a cross while the counters count only 'error', so a typo was shown
    as a failure and counted as neither: `n_error` stayed 0 for status='sucess'."""
    with pytest.raises(ValueError, match=rf"^status must be one of success, error, "
                                         rf"skipped, got {repr(bad)}\.$"):
        tracker.record("m", "f", {}, "r", 0, "builtin", status=bad)


def test_the_three_documented_statuses_are_all_still_accepted(tracker):
    for status in ("success", "error", "skipped"):
        tracker.record("m", status, {}, "r", 1, "builtin", status=status)
    assert [s.status for s in tracker.get_steps()] == ["success", "error", "skipped"]
    assert "Errors: 1" in tracker.summary()


def test_a_refused_record_does_not_burn_a_step_id(tracker):
    for _ in range(3):
        with pytest.raises(ValueError, match="^status must be one of"):
            tracker.record("m", "f", {}, "r", 0, "builtin", status="failing")
    step = tracker.record("m", "f", {}, "r", 0, "builtin")
    assert step.step_id == 1
    assert [s.step_id for s in tracker.get_steps()] == [1]


# ── execution_time_ms validation ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "bad,reason",
    [("'abc'", "must be a number"), ("None", "must be a number"),
     ("True", "must be a number"), ("-40", "cannot be negative"),
     ("float('nan')", "must be finite"), ("float('inf')", "must be finite")],
)
def test_record_refuses_a_duration_it_cannot_add_up(tracker, bad, reason):
    """HEAD accepted them all.  The column is declared INTEGER, and SQLite stored a
    value it cannot convert verbatim: after `record(..., execution_time_ms='abc')`
    the summary raised `TypeError: unsupported operand type(s) for +: 'int' and
    'str'`, so one bad row made the whole audit report unreadable.  -40 was accepted
    and reported as `Total execution time: -40ms`."""
    with pytest.raises(ValueError, match=rf"execution_time_ms {reason}"):
        tracker.record("m", "f", {}, "r", eval(bad), "builtin")


def test_integral_and_fractional_durations_are_stored_and_summed(tracker):
    tracker.record("m", "a", {}, "r", 7, "builtin")
    tracker.record("m", "b", {}, "r", 12.5, "builtin")
    tracker.record("m", "c", {}, "r", 0, "builtin")
    assert [s.execution_time_ms for s in tracker.get_steps()] == [7, 12.5, 0]
    assert "Total execution time: 19.5ms" in tracker.summary()


# ── reopening a session ──────────────────────────────────────────────────────

def test_a_reopened_session_appends_instead_of_colliding(tmp_path):
    """HEAD took step ids from a per-instance counter that restarted at 0, so the
    first new step of the reopened session raised `sqlite3.IntegrityError: UNIQUE
    constraint failed: steps.session_id, steps.step_id`."""
    path = str(tmp_path / "audit.db")
    first = ProvenanceTracker(db_path=path, session_id="sess_reopen")
    first.record("m", "a", {}, "r")
    first.record("m", "b", {}, "r")
    first.close()

    second = ProvenanceTracker(db_path=path, session_id="sess_reopen")
    assert [s.step_id for s in second.get_steps()] == [1, 2]
    assert second.record("m", "c", {}, "r").step_id == 3
    assert [s.step_id for s in second.get_steps()] == [1, 2, 3]
    second.close()


def test_a_different_session_in_the_same_file_still_starts_at_one(tmp_path):
    path = str(tmp_path / "audit.db")
    first = ProvenanceTracker(db_path=path, session_id="sess_a")
    first.record("m", "a", {}, "r")
    first.record("m", "b", {}, "r")
    other = ProvenanceTracker(db_path=path, session_id="sess_b")
    assert other.record("m", "c", {}, "r").step_id == 1
    assert [s.step_id for s in other.get_steps()] == [1]
    assert [s.step_id for s in first.get_steps()] == [1, 2]
    first.close()
    other.close()


# ── what the recorded parameters keep ────────────────────────────────────────

@dataclass
class StepConfig:
    window: int
    genes: list


def test_a_dataclass_parameter_is_recorded_as_its_fields():
    """The helper's own docstring names dataclasses as a case it handles, but HEAD
    had no branch for one: `_json_default(StepConfig(5, ['a']))` answered the string
    '<Cfg>' and a recorded step kept `{"cfg": "<Cfg>"}` — the parameters themselves
    were gone from the file that exists to make the run reproducible."""
    assert _json_default(StepConfig(5, ['a'])) == {'window': 5, 'genes': ['a']}
    assert json.loads(pv.dumps_params({"cfg": StepConfig(5, ['a'])})) == \
           {"cfg": {"window": 5, "genes": ["a"]}}


def test_an_object_whose_conversion_raises_falls_back_to_its_type_name():
    """The branch that used to sit behind `except Exception` and a
    `# pragma: no cover`, i.e. outside the coverage the package claims."""

    class ClosedArray:
        def tolist(self):
            raise ValueError("dataset is closed")

    assert _json_default(ClosedArray()) == "<ClosedArray>"


# ── how the reports render each status ───────────────────────────────────────

def test_a_skipped_step_is_not_reported_as_a_failure(tmp_path):
    """HEAD: `status_class = "status-ok" if ... else "status-error"` gave a skipped
    step the red cross of a crash, and the header counted only success and error —
    measured on 1 success + 2 skipped: `Steps: 3, Success: 1, Errors: 0`, two steps
    unaccounted for and both drawn as failures."""
    t = ProvenanceTracker(db_path=str(tmp_path / "sk.db"), session_id="s")
    t.record("m", "ok", {}, "r", 5)
    t.record("m", "skip", {}, "r", 1, status="skipped")
    t.record("m", "bad", {}, "r", 1, status="error", error_message="engine died")
    path = t.export_html(str(tmp_path / "sk.html"))
    html = open(path, encoding="utf-8").read()

    assert '<span class="status-skipped">⊘</span>' in html
    assert ".status-skipped { color: #ffcc44; }" in html
    assert "<strong>Skipped:</strong> 1" in html
    assert "<strong>Success:</strong> 1" in html
    assert "<strong>Errors:</strong> 1" in html
    text = t.summary()
    assert [line for line in text.splitlines()
            if line.startswith(("Total steps", "Success", "Errors", "Skipped"))] == \
           ["Total steps: 3", "Success: 1", "Errors: 1", "Skipped: 1"]
    t.close()


def test_the_status_counters_account_for_every_step(tmp_path):
    t = ProvenanceTracker(db_path=str(tmp_path / "c.db"), session_id="s")
    for status in ("success", "success", "error", "skipped", "skipped"):
        t.record("m", status, {}, "r", 1, status=status)
    text = t.summary()
    counted = {line.split(":")[0]: int(line.split(":")[1])
               for line in text.splitlines()
               if line.split(":")[0] in ("Total steps", "Success", "Errors", "Skipped")}
    assert counted == {"Total steps": 5, "Success": 2, "Errors": 1, "Skipped": 2}
    assert counted["Success"] + counted["Errors"] + counted["Skipped"] == \
           counted["Total steps"]
    t.close()


def test_a_legacy_row_with_an_unknown_status_still_renders(tmp_path):
    """Databases written before record() validated can hold any status; reading one
    must not raise out of the report."""
    t = ProvenanceTracker(db_path=str(tmp_path / "l.db"), session_id="s")
    t.conn.execute("INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (99, "s", "2026-01-01T00:00:00", "m", "old", "{}", "r", 1,
                    "builtin", "warning", ""))
    t.conn.commit()
    assert [s.status for s in t.get_steps()] == ["warning"]
    path = t.export_html(str(tmp_path / "l.html"))
    html = open(path, encoding="utf-8").read()
    assert '<span class="status-error">✗</span>' in html
    assert "<strong>Skipped:</strong> 0" in html
    assert "<strong>Other:</strong> 1" in html, "the odd row is still counted"
    assert "Other: 1" in t.summary()
    t.close()


def test_brief_marks_a_cut_and_leaves_a_short_value_alone():
    """The decorator caps a recorded value at 100 characters; HEAD cut silently, so a
    500-character sequence was stored as 100 characters that looked complete."""
    assert _brief("short", 100) == "short"
    assert _brief("A" * 100, 100) == "A" * 100
    assert _brief("A" * 500, 100) == "A" * 97 + "..."
    assert len(_brief("A" * 500, 100)) == 100, "the marker sits inside the budget"
    assert _brief(0, 10) == "0"


# ── the decorator's misuse guard ─────────────────────────────────────────────

def test_bare_tracked_is_refused_rather_than_returning_a_wrapper():
    """`@tracked` (without the call) bound the name to the inner factory, so
    `f(21)` answered a function object — measured — instead of running f and
    recording nothing."""
    with pytest.raises(TypeError, match=r"^tracked\(\) needs a ProvenanceTracker: "
                                        r"write @tracked\(tracker\), not a bare "
                                        r"@tracked \(got <function .* at "
                                        r"0x[0-9a-fA-F]+>\)\.$"):
        @tracked
        def f(x):
            return x * 2
def test_tracked_with_no_tracker_is_still_a_valid_empty_call(isolated_global_tracker):
    """The guard above must not swallow the documented no-recording usage."""
    @tracked()
    def multiply(a, b):
        return a * b

    assert multiply(6, 7) == 42


def test_tracked_records_a_zero_result_as_zero_and_only_none_as_none(tmp_path):
    t = ProvenanceTracker(db_path=str(tmp_path / "z.db"), session_id="sess_zero")
    for name, value in (("zero", 0), ("empty", []), ("none", None)):
        @tracked(t)
        def producing(v=value):
            return v
        producing()
    assert [s.result_summary for s in t.get_steps()] == ["0", "[]", "None"]
    t.close()


# ── the process-wide singleton ───────────────────────────────────────────────
def test_eight_threads_on_a_cold_global_tracker_get_one_tracker(
    isolated_global_tracker
):
    """Measured on the unguarded version: three runs of this probe produced 6, 8 and
    7 distinct trackers, and every step recorded through a loser of that race landed
    in a tracker no caller could reach again."""
    seen = []
    start = threading.Barrier(8)

    def worker():
        start.wait()
        seen.append(pv.get_global_tracker())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len({id(t) for t in seen}) == 1
    tracker = seen[0]
    assert tracker.get_steps() == []
    for i in range(8):
        tracker.record("m", f"f{i}", {}, "r")
    assert len(tracker.get_steps()) == 8
    tracker.close()
