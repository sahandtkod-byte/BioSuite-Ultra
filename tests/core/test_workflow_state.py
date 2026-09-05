"""Regression tests for the workflow engine (BSU-009).

Each test corresponds to a defect that made pipeline/batch results depend on
hidden state rather than on the declared inputs.
"""
import threading

import pytest

from biosuite.core.workflow.batch import BatchProcessor
from biosuite.core.workflow.pipeline import Pipeline


# ── argument precedence ─────────────────────────────────────────────────────

def test_explicit_step_kwargs_beat_the_shared_context():
    """`{**self.kwargs, **ctx}` gave the context priority - backwards."""
    pipeline = Pipeline("precedence")
    pipeline.add_step("step", lambda val: val, kwargs={"val": "EXPLICIT"})
    pipeline.set_context(val="FROM_CONTEXT")
    pipeline.run()
    assert pipeline.results["step"] == "EXPLICIT"


def test_context_still_fills_in_unspecified_arguments():
    pipeline = Pipeline("fill-in")
    pipeline.add_step("step", lambda val: val)
    pipeline.set_context(val="FROM_CONTEXT")
    pipeline.run()
    assert pipeline.results["step"] == "FROM_CONTEXT"


def test_positional_arguments_are_not_duplicated_by_the_context():
    pipeline = Pipeline("positional")
    pipeline.add_step("step", lambda val: val, args=["POSITIONAL"])
    pipeline.set_context(val="FROM_CONTEXT")
    pipeline.run()
    assert pipeline.steps[0].status == "done", pipeline.steps[0].error
    assert pipeline.results["step"] == "POSITIONAL"


# ── state isolation ─────────────────────────────────────────────────────────

def test_results_do_not_leak_into_the_next_run():
    seen = []
    pipeline = Pipeline("leak")
    pipeline.add_step("first", lambda: "A")
    pipeline.add_step("second", lambda **kw: seen.append(sorted(kw)) or "B")
    pipeline.run()
    pipeline.run()
    # Both runs must see exactly the same context; the second must not also
    # observe the previous run's "second" result.
    assert seen[0] == seen[1] == ["first"]


def test_a_step_that_recovers_clears_its_previous_error():
    state = {"fail": True}

    def flaky():
        if state["fail"]:
            raise RuntimeError("boom")
        return "ok"

    pipeline = Pipeline("flaky").add_step("f", flaky)
    pipeline.run()
    assert pipeline.steps[0].status == "failed"
    assert pipeline.steps[0].error == "boom"

    state["fail"] = False
    pipeline.run()
    assert pipeline.steps[0].status == "done"
    assert pipeline.steps[0].error is None
    assert pipeline.results["f"] == "ok"


def test_pipeline_run_is_repeatable():
    pipeline = Pipeline("repeat")
    pipeline.add_step("double", lambda x: x * 2)
    pipeline.set_context(x=3)
    first = pipeline.run().results["double"]
    second = pipeline.run().results["double"]
    assert first == second == 6


# ── identity uniqueness ─────────────────────────────────────────────────────

def test_duplicate_step_names_are_rejected():
    pipeline = Pipeline("dupes").add_step("x", str)
    with pytest.raises(ValueError, match="duplicate"):
        pipeline.add_step("x", str)


def test_duplicate_batch_sample_ids_are_rejected():
    """Three jobs silently produced two results."""
    batch = BatchProcessor().add_job("s1", str)
    with pytest.raises(ValueError, match="duplicate"):
        batch.add_job("s1", str)


def test_batch_keeps_every_sample():
    batch = BatchProcessor()
    batch.add_samples(["a", "b", "c"], lambda sid: sid.upper())
    batch.run(max_workers=1)
    assert batch.results == {"a": "A", "b": "B", "c": "C"}


def test_batch_job_clears_stale_error_on_rerun():
    state = {"fail": True}

    def flaky(_sample_id):
        if state["fail"]:
            raise RuntimeError("boom")
        return "ok"

    batch = BatchProcessor().add_job("s1", flaky)
    batch.run(max_workers=1)
    assert batch.jobs[0].status == "failed"
    state["fail"] = False
    batch.run(max_workers=1)
    assert batch.jobs[0].status == "done"
    assert batch.jobs[0].error is None


# ── concurrency ─────────────────────────────────────────────────────────────

def test_parallel_batch_records_every_result_exactly_once():
    """Results/progress were mutated from several worker threads unguarded."""
    n = 120
    batch = BatchProcessor()
    started = threading.Event()

    def work(sample_id):
        # Encourage interleaving without depending on timing for correctness.
        started.set()
        started.wait(0.01)
        return int(sample_id)

    batch.add_samples([str(i) for i in range(n)], work)
    batch.run(max_workers=8)
    assert len(batch.results) == n
    assert sorted(batch.results.values()) == list(range(n))
    assert batch._progress == n
