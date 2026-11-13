"""biosuite.core.workflow.pipeline: the parallel run's two hard promises.

Steps in a parallel run must see only the context as it stood before the run —
the old loop kept writing results into that same dict, so a step that started
late read another's output — and a run stopped on error must neither lose a step
that had already finished nor leave a queued step looking like one that ran.
"""
import time

import pytest

from biosuite.core.workflow.pipeline import Pipeline


def recorder(**_ignored):
    return None


def make_halt(slow=0.05):
    p = Pipeline("halt")
    p.add_step("bad", lambda: 1 / 0)
    for i in range(6):
        p.add_step(f"s{i}", lambda: (time.sleep(slow), 1)[1])
    return p


def test_parallel_steps_read_only_the_context_that_existed_before_the_run():
    p = Pipeline("iso")
    p.add_step("alpha", lambda: "V")
    p.add_step("beta", lambda alpha=None: sorted(k for k in (alpha,) if k))
    p.add_step("gamma", lambda alpha=None: alpha)
    p.run(max_workers=3)
    assert p.get_result("beta") == [], "beta must not see alpha's result"
    assert p.get_result("gamma") is None
    assert p.context["alpha"] == "V", "the finished run still records its output"


def test_sequential_steps_do_see_the_previous_ones():
    p = Pipeline("chain")
    p.add_step("alpha", lambda: "V")
    p.add_step("beta", lambda alpha=None: sorted(k for k in (alpha,) if k))
    p.run()
    assert p.get_result("beta") == ["V"]


def test_parallel_results_are_ordered_by_step_not_by_completion():
    p = Pipeline("order")
    p.add_step("slow", lambda: (time.sleep(0.1), "S")[1])
    p.add_step("fast", lambda: "F")
    p.add_step("middle", lambda: (time.sleep(0.02), "M")[1])
    p.run(max_workers=3, stop_on_error=False)
    assert list(p.results) == ["slow", "fast", "middle"]


def test_a_stop_on_error_run_never_hangs_and_never_loses_a_finished_step():
    """Measured: two workers, a failing first step, six slow ones. The result is
    the steps that had already started, and the rest are named as cancelled."""
    p = make_halt()
    p.run(stop_on_error=True, max_workers=2)
    assert p.steps[0].status == "failed"
    statuses = [s.status for s in p.steps]
    assert "cancelled" in statuses
    assert "pending" not in statuses and "running" not in statuses
    for name, value in p.results.items():
        assert p.get_result(name) == value == 1


def test_the_stop_line_says_which_steps_did_not_run():
    log = "\n".join(make_halt().run(stop_on_error=True, max_workers=2)._log)
    assert "Stopped on bad" in log
    assert "step(s) did not run" in log
    assert "s5 cancelled: not started" in log or "s4 cancelled: not started" in log


def test_a_step_cancelled_before_it_ran_is_marked_and_not_counted_as_failed():
    p = make_halt()
    p.run(stop_on_error=True, max_workers=2)
    cancelled = [s for s in p.steps if s.status == "cancelled"]
    assert cancelled
    assert all(s.error is None and s.elapsed == 0.0 for s in cancelled)
    assert all(s.name not in p.results for s in cancelled)


def test_parallel_without_stop_on_error_runs_every_step():
    p = make_halt()
    p.run(stop_on_error=False, max_workers=2)
    assert [s.status for s in p.steps] == ["failed"] + ["done"] * 6
    assert len(p.results) == 6
    assert "Stopped on" not in "\n".join(p._log)


def test_a_sequential_stop_names_the_steps_it_skipped():
    p = Pipeline("seq")
    p.add_step("a", lambda: 1)
    p.add_step("bad", lambda: 1 / 0)
    p.add_step("c", lambda: 3)
    p.add_step("d", lambda: 4)
    p.run(stop_on_error=True)
    log = "\n".join(p._log)
    assert "Stopped after bad: 2 step(s) (c, d) were not run" in log
    assert [s.status for s in p.steps] == ["done", "failed", "pending", "pending"]


def test_summary_marks_a_cancelled_step_differently_from_a_pending_one():
    p = make_halt()
    p.run(stop_on_error=True, max_workers=2)
    icons = {s.name: line.split("]")[0] for s, line
             in zip(p.steps, p.summary().splitlines()[3:3 + len(p.steps)])}
    assert icons["bad"] == "  [X"
    assert icons["s0"] == "  [+" or icons["s1"] == "  [+"
    assert any(v == "  [o" for v in icons.values()), "cancelled has its own mark"


def test_the_snapshot_survives_a_step_that_mutates_its_own_kwargs():
    """A step cannot reach back into the pipeline's context through its args."""
    seen = []

    def grab(**kwargs):
        seen.append(dict(kwargs))
        kwargs["injected"] = 1
        return None

    p = Pipeline("snap")
    p.set_context(seed=[1, 2])
    p.add_step("one", grab)
    p.add_step("two", grab)
    p.run(max_workers=2)
    assert p.context["seed"] == [1, 2]
    assert all("injected" not in s for s in seen)
