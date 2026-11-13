"""Pipeline step dispatch, bulk step registration and the parallel executor.

Pins how ``PipelineStep.run`` merges the shared context for a callable whose
signature cannot be introspected, that ``add_steps`` applies the same
duplicate-name rule as ``add_step`` and refuses anything that is not a step,
which step results reach the shared context, and what the thread-pool executor
records and logs when a step fails.  No test here sleeps or runs a long step.
"""
import inspect

import pytest

from biosuite.core.workflow.pipeline import (Pipeline, PipelineStep,
                                             build_pipeline_from_steps,
                                             format_pipeline_report,
                                             run_quick_pipeline)


def produce():
    return 7


def consume(x):
    return x * 2


def test_a_step_that_never_ran_still_exposes_every_run_attribute():
    step = PipelineStep("fresh", max, args=([3, 1, 2],))

    assert (step.result, step.error, step.traceback, step.elapsed, step.status) \
        == (None, None, None, 0.0, "pending")


def test_signatureless_builtin_step_runs_with_the_context_it_was_given():
    # inspect.signature() refuses to answer for max(), so the step is called
    # with the untouched merged context instead of a filtered copy.
    with pytest.raises(ValueError, match="no signature found for builtin"):
        inspect.signature(max)

    step = PipelineStep("biggest", max, args=([3, 1, 2],))
    assert step.run() == 3
    assert step.status == "done"
    assert step.error is None


def test_signatureless_builtin_step_fails_visibly_on_context_keywords():
    step = PipelineStep("biggest", max, args=([3, 1, 2],))
    assert step.run({"seed": 1}) is None

    assert step.status == "failed"
    assert step.error == "max() got an unexpected keyword argument 'seed'"
    assert "TypeError" in step.traceback


def test_add_steps_accepts_dicts_tuples_and_prebuilt_steps():
    p = Pipeline("bulk")
    returned = p.add_steps([
        {"name": "from_dict", "func": produce, "description": "a dict step"},
        ["from_list", consume, [5], {}, "a list step"],
        ("from_tuple", lambda: 9),
        PipelineStep("prebuilt", lambda: 11),
    ])

    assert returned is p
    assert [(s.name, s.func.__name__ if hasattr(s.func, '__name__') else 'x')
            for s in p.steps] == [("from_dict", "produce"), ("from_list", "consume"),
                                  ("from_tuple", "<lambda>"), ("prebuilt", "<lambda>")]
    assert p.steps[0].description == "a dict step"
    p.run()
    assert list(p.get_result().values()) == [7, 10, 9, 11]


def test_add_steps_rejects_a_prebuilt_step_that_duplicates_a_name():
    p = Pipeline("dup")
    p.add_step("one", produce)

    with pytest.raises(ValueError, match="duplicate pipeline step name 'one'"):
        p.add_steps([PipelineStep("one", consume, args=[1])])
    assert [s.name for s in p.steps] == ["one"]


@pytest.mark.parametrize("junk", ["gc=seq", 42, None],
                         ids=["string", "int", "none"])
def test_add_steps_refuses_items_that_are_not_steps(junk):
    p = Pipeline("junk")

    with pytest.raises(TypeError, match="pipeline step must be a dict"):
        p.add_steps([junk])
    assert p.steps == []


def test_add_steps_rejects_a_duplicate_reaching_add_step_through_a_dict():
    p = Pipeline("dup-dict")
    p.add_steps([{"name": "a", "func": produce}])

    with pytest.raises(ValueError, match="duplicate pipeline step name 'a'"):
        p.add_steps([("a", consume)])


def test_a_step_returning_none_is_recorded_but_never_enters_the_context():
    p = Pipeline("silence")
    p.add_step("quiet", lambda: None)
    p.add_step("peek", lambda **kw: sorted(kw))
    p.run()

    assert p.get_result("quiet") is None
    assert "quiet" not in p.context
    assert p.get_result("peek") == []


def test_a_producing_step_lands_in_the_context_for_the_next_step():
    def doubled(producer):
        return producer * 2

    p = Pipeline("chain")
    p.add_step("producer", produce)
    p.add_step("consumer", doubled)
    p.run()

    assert p.context["producer"] == 7
    assert p.get_result("consumer") == 14


def test_a_step_whose_parameter_matches_no_context_key_fails_loudly():
    p = Pipeline("missing")
    p.add_step("producer", produce)
    p.add_step("consumer", consume)
    p.run(stop_on_error=False)

    assert p.get_result("producer") == 7
    assert p.steps[1].status == "failed"
    assert p.steps[1].error == "consume() missing 1 required positional argument: 'x'"


def test_parallel_run_records_independent_steps_and_their_failure():
    p = Pipeline("par")
    p.add_step("a", lambda: 1)
    p.add_step("b", lambda: 2)
    p.add_step("bad", lambda: 1 / 0)

    assert p.run(stop_on_error=False, max_workers=3) is p
    assert dict(p.results) == {"a": 1, "b": 2}
    assert [s.status for s in p.steps] == ["done", "done", "failed"]
    assert p.context["a"] == 1 and p.context["b"] == 2
    assert "bad" not in p.context

    log = "\n".join(p._log)
    for expected in ("[1] Submitting: a", "[2] Submitting: b", "[3] Submitting: bad",
                     "a done in ", "b done in ",
                     "bad FAILED: division by zero"):
        assert expected in log
    assert "Pipeline 'par' finished in " in log


def test_parallel_run_honours_stop_on_error_before_recording_later_steps():
    p = Pipeline("halt-par")
    p.add_step("bad", lambda: 1 / 0)
    for i in range(6):
        p.add_step(f"s{i}", lambda: 1)

    p.run(stop_on_error=True, max_workers=2)

    assert "bad" not in p.results
    assert p.steps[0].status == "failed"
    assert p.get_result("bad") is None
    assert "  bad FAILED: division by zero" in "\n".join(p._log)


def test_summary_marks_done_failed_and_pending_steps():
    p = Pipeline("icons")
    p.add_step("ok", produce)
    p.add_step("bad", lambda: 1 / 0)
    p.add_step("later", produce)
    p.run(stop_on_error=True)

    lines = p.summary().splitlines()
    assert lines[0] == "Pipeline: icons"
    assert lines[1] == "Steps: 3"
    assert lines[3].startswith("  [+] 1. ok (")
    assert lines[4] == "  [X] 2. bad (0.00s)" or "  [X] 2. bad (" in lines[4]
    assert "      Error: division by zero" in lines[5]
    assert "  [-] 3. later (" in "\n".join(lines)


def test_build_pipeline_from_steps_carries_args_kwargs_and_descriptions():
    p = build_pipeline_from_steps([
        {"name": "first", "func": produce, "description": "produces 7"},
        {"name": "second", "func": consume, "args": [10], "kwargs": None},
        {"name": "third", "func": lambda x=0: x + 1, "kwargs": {"x": 100}},
    ])

    assert isinstance(p, Pipeline) and p.name == "pipeline"
    assert [s.name for s in p.steps] == ["first", "second", "third"]
    assert p.steps[1].args == [10] and p.steps[2].kwargs == {"x": 100}

    results = p.run().get_result()
    assert list(results.values()) == [7, 20, 101]


def test_run_quick_pipeline_seeds_the_context_and_returns_results_in_order():
    results = run_quick_pipeline(
        [("doubled", lambda x: x * 2, {}),
         ("labelled", lambda doubled: f"n={doubled}", {}),
         ("explicit", lambda x: x - 1, {"x": 50})],
        x=21)

    assert list(results.items()) == [("doubled", 42), ("labelled", "n=42"),
                                     ("explicit", 49)]


def test_run_quick_pipeline_without_context_gives_the_step_its_own_kwargs():
    results = run_quick_pipeline([("squared", lambda n: n ** 2, {"n": 9})])
    assert dict(results) == {"squared": 81}


def test_format_pipeline_report_is_the_summary_of_the_finished_run():
    p = Pipeline("report")
    p.add_step("only", produce)
    p.run()

    text = format_pipeline_report(p)
    assert text == p.summary()
    assert text.startswith("Pipeline: report\nSteps: 1\n\n")
    assert "[+] 1. only (" in text
    assert "Pipeline 'report' finished in " in text
