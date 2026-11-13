"""biosuite.core.workflow.batch: what a finished batch hands back.

A batch of samples has three outputs — the result dict, the log and the summary —
and they used to disagree: a failed job appeared in the results as ``None``,
indistinguishable from a function that legitimately returned nothing; the
sequential path logged no per-job outcome while the parallel one did; and the
parallel path filled the results in completion order, so one batch could hand
back two different sequences.
"""
import pytest

from biosuite.core.workflow.batch import BatchJob, BatchProcessor, batch_run


def mixed(sid):
    if sid == "bad":
        raise ValueError("nope")
    if sid == "silent":
        return None
    return sid.upper()


def processor(*ids):
    bp = BatchProcessor("t")
    bp.add_samples(list(ids), mixed)
    return bp


def test_only_successful_jobs_appear_in_the_results():
    bp = processor("a", "bad", "c")
    bp.run(max_workers=1)
    assert dict(bp.get_results()) == {"a": "A", "c": "C"}
    assert bp.get_failures() == [("bad", "nope")]


def test_a_job_returning_none_is_indistinguishable_no_longer_the_contract():
    """``silent`` succeeded with nothing to say; ``bad`` failed. Only the first
    is in the results, and the failure is named in the log."""
    bp = processor("silent", "bad")
    bp.run(max_workers=1)
    assert dict(bp.get_results()) == {"silent": None}
    assert "  bad FAILED: nope" in "\n".join(bp._log)
    assert "bad" not in bp.get_results()


@pytest.mark.parametrize("workers", [1, 4])
def test_both_paths_log_one_outcome_line_per_job(workers):
    log = "\n".join(processor("a", "bb", "bad").run(max_workers=workers)._log)
    assert "a done in " in log and "bb done in " in log
    assert "bad FAILED: nope" in log


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_the_results_keep_the_order_the_samples_were_added_in(workers):
    """The dict is an OrderedDict, so its order is part of what a caller sees."""
    bp = processor(*["s1", "s2", "s3", "s4", "s5", "s6"])
    bp.run(max_workers=workers)
    assert list(bp.get_results()) == ["s1", "s2", "s3", "s4", "s5", "s6"]


def test_rerunning_a_batch_forgets_the_previous_failure():
    bp = processor("a", "bad")
    bp.run(max_workers=1)
    assert bp.get_failures() == [("bad", "nope")]
    bp.jobs[1].func = lambda sid: sid.lower()
    bp.run(max_workers=1)
    assert bp.get_failures() == []
    assert dict(bp.get_results()) == {"a": "A", "bad": "bad"}


def test_a_job_that_has_never_run_answers_for_every_attribute():
    """``traceback`` used to exist only on a job that had run, so inspecting a
    queued job raised AttributeError while a finished one never did."""
    job = BatchJob("s1", mixed)
    assert job.status == "pending"
    assert job.result is None and job.error is None
    assert job.traceback is None
    assert job.elapsed == 0.0
    assert job.to_dict() == {"sample_id": "s1", "status": "pending",
                             "elapsed": 0.0, "error": None}


def test_a_failed_job_carries_a_traceback_a_report_cannot_show():
    job = BatchJob("bad", mixed)
    job.run()
    assert job.status == "failed"
    assert "ValueError: nope" in job.traceback
    assert "Traceback" in job.traceback
    assert job.to_dict()["error"] == "nope"


def test_batch_run_returns_only_the_successes_and_says_so_in_its_docstring():
    results = batch_run(mixed, ["a", "bad", "c"], max_workers=2)
    assert results == {"a": "A", "c": "C"}
    assert "failed" in batch_run.__doc__


def test_summary_counts_and_per_job_lines_agree_with_the_results():
    bp = processor("a", "bad", "c")
    bp.run(max_workers=1)
    lines = bp.summary().splitlines()
    assert lines[1] == "Samples: 3"
    assert lines[2] == "Completed: 2"
    assert lines[3] == "Failed: 1"
    assert "[X] bad" in "\n".join(lines)
    assert "      nope" in lines


def test_format_batch_report_is_the_summary_the_cli_prints():
    from biosuite.core.workflow.batch import format_batch_report
    bp = processor("a", "bad")
    bp.run(max_workers=1)
    assert format_batch_report(bp) == bp.summary()


def test_to_dict_is_the_machine_readable_shape_of_the_batch():
    bp = processor("a", "bad")
    bp.run(max_workers=1)
    assert bp.to_dict() == {
        "name": "t",
        "total": 2,
        "done": 1,
        "failed": 1,
        "jobs": [
            {"sample_id": "a", "status": "done", "elapsed": 0.0, "error": None},
            {"sample_id": "bad", "status": "failed", "elapsed": 0.0,
             "error": "nope"},
        ],
    }


def test_a_batch_of_nothing_reports_zeroes():
    bp = BatchProcessor("empty")
    bp.run(max_workers=1)
    assert bp.to_dict()["total"] == 0
    assert dict(bp.get_results()) == {}
    assert "Batch 'empty' finished: 0 done, 0 failed" in "\n".join(bp._log)
