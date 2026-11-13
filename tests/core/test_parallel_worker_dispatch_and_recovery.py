"""Dispatch, bookkeeping and failure recovery of ``core/parallel.py``.

Pins what the two pool helpers promise: results in input order, a failed item
surfaced as a logged ``None`` (never a silent success and never a size-dependent
crash), the requested worker count actually handed to the executor, chunked
submissions scattered back to their real offsets, and progress that stops at
exactly the number of items.  Executors are swapped at the module's own
reference so no process is spawned and no timing is asserted.
"""
import functools
import logging
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from biosuite.core import parallel
from biosuite.core.alignment import needleman_wunsch, smith_waterman
from biosuite.core.parallel import (
    ParallelBatchProcessor, _apply_chunk, _is_picklable, _threads_preferred,
    get_default_pool, get_optimal_workers, parallel_align_pairs,
    parallel_gc_content, parallel_map, parallel_reverse_complement,
    parallel_submit, parallel_translate, shutdown_default_pool,
)
from biosuite.core.sequence import gc_content, reverse_complement, translate


# ── module-level callables (picklable, as a process pool requires) ───────────

def double(value):
    return value * 2


def add_offset(value, offset=0):
    return value + offset


def boom_on_three(value):
    """Fails for exactly one item, so a single slot must come back as None."""
    if value == 3:
        raise ValueError("no three")
    return value


class Recorder:
    def shout(self, value):
        return f"{value}!"


def _make_local_function():
    def locally_defined(value):
        return value * value

    return locally_defined


SEQUENCES = ['ACGT' * 3, 'AAAA' * 3, 'ACAC' * 3, 'GCGC' * 3,
             'TTTTTT', 'ACGTAC', 'GGGGGG', 'AAAAAA']
PAIRS = [('ACGTACGT', 'ACGTACGA'), ('AAAGGG', 'AAAGGG'), ('ACGT', 'TGCA'),
         ('ACGTTT', 'ACGTTT'), ('GGGAAA', 'GGGTTT'), ('ACACAC', 'ACGTAC')]


# ── test doubles ─────────────────────────────────────────────────────────────

class RecordingPool:
    """Stand-in for a real executor: records the worker count and runs the work
    on threads inside this process, so dispatch bookkeeping is observable."""

    created = []

    def __init__(self, max_workers=None):
        RecordingPool.created.append(max_workers)
        self._inner = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn, *args, **kwargs):
        return self._inner.submit(fn, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._inner.shutdown(wait=True)
        return False


class UnavailablePool:
    """An executor that cannot be created at all (the real failure mode of a
    saturated machine: ``OSError`` / ``ValueError`` from the constructor)."""

    def __init__(self, max_workers=None):
        raise RuntimeError("cannot start a worker pool")


@pytest.fixture
def posix(monkeypatch):
    """Look like a fork-capable platform so a process pool can be chosen."""
    monkeypatch.setattr(parallel.sys, "platform", "linux")


@pytest.fixture
def process_pool(monkeypatch, posix):
    RecordingPool.created.clear()
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", RecordingPool)
    return RecordingPool.created


@pytest.fixture
def thread_pool(monkeypatch):
    RecordingPool.created.clear()
    monkeypatch.setattr(parallel, "ThreadPoolExecutor", RecordingPool)
    return RecordingPool.created


@pytest.fixture
def forbid_pools(monkeypatch):
    """Anything that reaches for an executor is a test failure, not a fallback."""
    def refuse(*args, **kwargs):
        raise AssertionError("no executor may be created for this call")

    monkeypatch.setattr(parallel, "ThreadPoolExecutor", refuse)
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", refuse)


def _error_messages(caplog):
    """The ERROR texts of ``biosuite.core.parallel``, and of nothing else.

    pytest hands records up through the root handler, so naming a logger in
    ``caplog.at_level`` filters nothing; these pins once named 'biosuite.parallel'
    and still passed.  Requiring the emitting logger keeps the identity check real.
    """
    assert {r.name for r in caplog.records} == {"biosuite.core.parallel"}
    return [r.getMessage() for r in caplog.records]


# ── worker auto-detection ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "cpu_count,io_bound,expected",
    [(1, False, 1), (8, False, 8), (40, False, 16), (4, True, 8),
     (40, True, 32), (20, True, 32)],
)
def test_get_optimal_workers_scales_with_the_task_type_and_caps_itself(
    monkeypatch, cpu_count, io_bound, expected
):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: cpu_count)
    assert get_optimal_workers(io_bound=io_bound) == expected


def test_get_optimal_workers_falls_back_to_four_cores_when_the_count_is_unknown(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: None)
    assert get_optimal_workers() == 4
    assert get_optimal_workers(io_bound=True) == 8


# ── picklability and executor selection ──────────────────────────────────────

def test_is_picklable_accepts_anything_a_worker_can_be_handed():
    assert _is_picklable(double) is True
    assert _is_picklable(add_offset) is True
    assert _is_picklable(Recorder().shout) is True
    assert _is_picklable(len) is True


UNPICKLABLE = [
    ("lambda", lambda value: value, pickle.PicklingError),
    ("nested function", _make_local_function(), AttributeError),
    ("partial of a lambda", functools.partial(lambda value: value, 1), pickle.PicklingError),
    ("thread lock", threading.Lock(), TypeError),
]


@pytest.mark.parametrize(
    "label,value,error",
    UNPICKLABLE,
    ids=[case[0] for case in UNPICKLABLE],
)
def test_is_picklable_catches_every_failure_mode_pickle_can_raise(label, value, error):
    """The helper promises to swallow PicklingError, AttributeError and TypeError."""
    with pytest.raises(error):
        pickle.dumps(value)
    assert _is_picklable(value) is False


def test_threads_are_preferred_on_windows_even_for_a_picklable_function():
    assert _threads_preferred([double], io_bound=False) is True


def test_threads_are_preferred_for_io_bound_work_on_any_platform(posix):
    assert _threads_preferred([double], io_bound=True) is True


def test_process_pool_is_chosen_off_windows_for_a_picklable_function(posix):
    assert _threads_preferred([double], io_bound=False) is False
    assert _threads_preferred([double, add_offset], io_bound=False) is False


def test_threads_are_chosen_off_windows_when_any_task_is_unpicklable(posix):
    assert _threads_preferred([lambda value: value], io_bound=False) is True
    assert _threads_preferred([double, lambda value: value], io_bound=False) is True


# ── parallel_map: dispatch and ordering ──────────────────────────────────────

def test_parallel_map_returns_nothing_for_no_items(forbid_pools):
    assert parallel_map(double, []) == []


@pytest.mark.parametrize("items", [[5], [5, 6], [5, 6, 7], [5, 6, 7, 8]])
def test_parallel_map_runs_four_items_or_fewer_in_this_process(forbid_pools, items):
    assert parallel_map(double, items, workers=4) == [2 * value for value in items]


def test_parallel_map_runs_a_single_worker_batch_in_this_process(forbid_pools):
    assert parallel_map(double, list(range(10)), workers=1) == [2 * i for i in range(10)]


def test_parallel_map_honours_the_worker_count_it_was_given(thread_pool):
    got = parallel_map(double, list(range(12)), workers=3, io_bound=True)
    assert got == [2 * i for i in range(12)]
    assert thread_pool == [3]


def test_parallel_map_auto_detects_the_worker_count(thread_pool):
    expected_workers = get_optimal_workers(io_bound=True)
    got = parallel_map(double, list(range(12)), io_bound=True)
    assert got == [2 * i for i in range(12)]
    assert thread_pool == [expected_workers]


def test_parallel_map_forwards_extra_keyword_arguments_to_every_item(thread_pool):
    got = parallel_map(add_offset, list(range(8)), workers=2, io_bound=True, offset=10)
    assert got == [10, 11, 12, 13, 14, 15, 16, 17]


def test_parallel_map_keeps_results_in_input_order(thread_pool):
    items = list(range(24, 54))
    assert parallel_map(double, items, workers=4, io_bound=True) == [2 * i for i in items]


def test_parallel_map_selects_a_process_pool_for_a_picklable_function(process_pool):
    got = parallel_map(double, list(range(10)), workers=2)
    assert got == [2 * i for i in range(10)]
    assert process_pool == [2]


# ── parallel_map: chunking ───────────────────────────────────────────────────

def test_chunked_submission_scatters_each_chunk_back_to_its_own_slice(process_pool):
    got = parallel_map(double, list(range(10)), workers=3, chunk_size=4)
    assert got == [2 * i for i in range(10)]
    assert process_pool == [3]


def test_chunked_submission_with_a_short_final_chunk_is_still_exact(thread_pool):
    for chunk_size in (1, 2, 3, 5, 7, 12):
        got = parallel_map(double, list(range(12)), workers=2, io_bound=True,
                           chunk_size=chunk_size)
        assert got == [2 * i for i in range(12)], f"chunk_size={chunk_size}"


def test_chunk_size_of_one_returns_values_not_one_item_lists(thread_pool):
    """_apply_chunk always returns a list, so a 1-item chunk is still a chunk."""
    got = parallel_map(double, list(range(6)), workers=2, io_bound=True, chunk_size=1)
    assert got == [0, 2, 4, 6, 8, 10]


def test_apply_chunk_maps_the_function_over_the_whole_chunk():
    assert _apply_chunk(double, [1, 2, 3], {}) == [2, 4, 6]
    assert _apply_chunk(add_offset, [1, 2], {"offset": 5}) == [6, 7]
    assert _apply_chunk(double, [], {}) == []


@pytest.mark.parametrize("chunk_size", [2, 3, 5])
def test_progress_callback_never_reports_more_items_than_exist(thread_pool, chunk_size):
    seen = []
    items = list(range(10))
    parallel_map(double, items, workers=2, io_bound=True, chunk_size=chunk_size,
                 progress_callback=lambda done, total: seen.append((done, total)))
    assert all(total == 10 for _, total in seen)
    assert len(seen) == -(-len(items) // chunk_size)      # one call per chunk
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)
    assert seen[-1] == (10, 10)          # exact bookkeeping, no over-count


def test_progress_callback_is_skipped_when_the_caller_did_not_ask_for_one(thread_pool):
    assert parallel_map(double, list(range(9)), workers=2, io_bound=True) == \
           [2 * i for i in range(9)]


# ── parallel_map: failure recovery ───────────────────────────────────────────

def test_parallel_map_turns_a_failed_item_into_a_logged_none(thread_pool, caplog):
    with caplog.at_level(logging.ERROR, logger="biosuite.core.parallel"):
        got = parallel_map(boom_on_three, [1, 2, 3, 4, 5, 6], workers=2, io_bound=True)
    assert got == [1, 2, None, 4, 5, 6]
    assert _error_messages(caplog) == [
        "Parallel task failed at index 2: no three"
    ]


def test_a_failed_chunk_poisons_only_that_chunk(thread_pool):
    got = parallel_map(boom_on_three, [1, 2, 3, 4, 5], workers=2, io_bound=True,
                       chunk_size=2)
    assert got == [1, 2, None, None, 5]  # item 3 sits in the chunk [2,3]


def test_a_chunk_keeps_its_own_results_and_only_nones_the_items_that_failed(thread_pool):
    """One chunk of five was asked for and all three items fit inside it, so the
    failure of item 3 must not be allowed to erase the values already produced."""
    got = parallel_map(boom_on_three, [1, 2, 3], workers=2, io_bound=True,
                       chunk_size=5)

    assert got == [1, 2, None], got


def test_progress_counts_failed_items_so_it_still_ends_at_the_total(thread_pool):
    seen = []
    got = parallel_map(boom_on_three, list(range(1, 9)), workers=2, io_bound=True,
                       progress_callback=lambda done, total: seen.append(done))
    assert got == [1, 2, None, 4, 5, 6, 7, 8]
    assert seen[-1] == 8


def test_parallel_map_falls_back_to_sequential_when_no_pool_can_start(
    monkeypatch, caplog
):
    monkeypatch.setattr(parallel, "ThreadPoolExecutor", UnavailablePool)
    with caplog.at_level(logging.ERROR, logger="biosuite.core.parallel"):
        got = parallel_map(double, list(range(8)), workers=2, io_bound=True)
    assert got == [2 * i for i in range(8)]
    assert any("Falling back to sequential" in m for m in _error_messages(caplog))


def test_the_sequential_fallback_reports_failures_exactly_like_the_pool(
    monkeypatch, caplog
):
    """Same work, same contract: a failed item is None, whatever path ran it."""
    monkeypatch.setattr(parallel, "ThreadPoolExecutor", UnavailablePool)
    with caplog.at_level(logging.ERROR, logger="biosuite.core.parallel"):
        pooled = parallel_map(boom_on_three, [1, 2, 3, 4, 5, 6], workers=2, io_bound=True)
    messages = _error_messages(caplog)
    assert pooled == [1, 2, None, 4, 5, 6]
    assert messages[-1] == "Parallel task failed at index 2: no three"
    assert "Parallel executor failed: cannot start a worker pool. " \
           "Falling back to sequential." in messages


def test_in_process_execution_does_not_raise_where_the_pool_would_not(
    thread_pool, caplog
):
    """3 items stay in-process while 6 items use a pool; both must return None."""
    with caplog.at_level(logging.ERROR, logger="biosuite.core.parallel"):
        small = parallel_map(boom_on_three, [1, 2, 3])
        large = parallel_map(boom_on_three, [1, 2, 3, 4, 5, 6], workers=2, io_bound=True)
    assert small == [1, 2, None]
    assert large == [1, 2, None, 4, 5, 6]
    assert thread_pool == [2]            # only the 6-item call reached a pool
    assert _error_messages(caplog) == [
        "Parallel task failed at index 2: no three",
        "Parallel task failed at index 2: no three",
    ]


# ── parallel_submit ──────────────────────────────────────────────────────────

def test_parallel_submit_returns_nothing_for_no_tasks(forbid_pools):
    assert parallel_submit([]) == []


def test_parallel_submit_runs_a_single_task_in_this_process(forbid_pools):
    assert parallel_submit([(double, (21,), {})], workers=4) == [42]


def test_parallel_submit_runs_a_single_worker_batch_in_this_process(forbid_pools):
    tasks = [(double, (1,), {}), (boom_on_three, (3,), {}), (add_offset, (1,), {"offset": 4})]
    assert parallel_submit(tasks, workers=1) == [2, None, 5]


def test_parallel_submit_logs_a_sequential_task_failure(forbid_pools, caplog):
    with caplog.at_level(logging.ERROR, logger="biosuite.core.parallel"):
        got = parallel_submit([(boom_on_three, (3,), {})], workers=1)
    assert got == [None]
    assert _error_messages(caplog) == ["Task failed: no three"]


def test_parallel_submit_auto_detects_its_worker_count(thread_pool):
    tasks = [(double, (i,), {}) for i in range(8)]
    assert parallel_submit(tasks, io_bound=True) == [2 * i for i in range(8)]
    assert thread_pool == [get_optimal_workers(io_bound=True)]


def test_parallel_submit_keeps_heterogeneous_results_in_task_order(thread_pool):
    tasks = [(double, (5,), {}), (str, (42,), {}), (len, ((1, 2, 3),), {}),
             (add_offset, (1,), {"offset": 9})]
    assert parallel_submit(tasks, workers=2, io_bound=True) == [10, '42', 3, 10]


def test_parallel_submit_reports_a_failing_task_as_none(thread_pool, caplog):
    tasks = [(double, (1,), {}), (boom_on_three, (3,), {}), (double, (2,), {}),
             (double, (3,), {})]
    with caplog.at_level(logging.ERROR, logger="biosuite.core.parallel"):
        got = parallel_submit(tasks, workers=2, io_bound=True)

    assert got == [2, None, 4, 6], got
    assert _error_messages(caplog) == ["Task 1 failed: no three"], \
        "the fourth task is double(3), which succeeds; only the boom is reported"


def test_parallel_submit_selects_a_process_pool_for_picklable_tasks(process_pool):
    tasks = [(double, (i,), {}) for i in range(6)] + [(add_offset, (1,), {"offset": 1})]
    assert parallel_submit(tasks, workers=2) == [2 * i for i in range(6)] + [2]
    assert process_pool == [2]


def test_parallel_submit_falls_back_when_the_pool_cannot_start(monkeypatch, caplog):
    monkeypatch.setattr(parallel, "ThreadPoolExecutor", UnavailablePool)
    tasks = [(double, (1,), {}), (boom_on_three, (3,), {}), (double, (2,), {})]
    with caplog.at_level(logging.ERROR, logger="biosuite.core.parallel"):
        got = parallel_submit(tasks, workers=2, io_bound=True)
    assert got == [2, None, 4]
    assert any("Parallel executor failed" in m for m in _error_messages(caplog))


# ── ParallelBatchProcessor ───────────────────────────────────────────────────

def test_batch_processor_accumulates_total_completed_and_failed_counts():
    processor = ParallelBatchProcessor(workers=2, io_bound=True)
    got = processor.process(double, list(range(10)), batch_size=3)
    assert got == [2 * i for i in range(10)]
    stats = processor.get_stats()
    assert stats['total'] == 10 and stats['completed'] == 10 and stats['failed'] == 0


def test_batch_processor_counts_none_results_as_failures():
    processor = ParallelBatchProcessor(workers=2, io_bound=True)
    got = processor.process(boom_on_three, list(range(1, 9)), batch_size=4)
    assert got == [1, 2, None, 4, 5, 6, 7, 8]
    assert processor.get_stats()['failed'] == 1
    assert processor.get_stats()['completed'] == 8


def test_batch_processor_progress_callback_sees_every_batch():
    processor = ParallelBatchProcessor(workers=2, io_bound=True)
    seen = []
    got = processor.process(double, list(range(7)), batch_size=3,
                            progress_callback=lambda done, total: seen.append((done, total)))
    assert got == [2 * i for i in range(7)]
    assert seen == [(3, 7), (6, 7), (7, 7)]


def test_batch_processor_defaults_its_worker_count_and_keeps_stats_a_copy(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 2)
    processor = ParallelBatchProcessor()
    assert processor.workers == 2 and processor.io_bound is False
    assert ParallelBatchProcessor(io_bound=True).workers == 4
    assert ParallelBatchProcessor(workers=5, io_bound=True).workers == 5
    snapshot = processor.get_stats()
    snapshot['total'] = 999
    assert processor.get_stats()['total'] == 0


# ── convenience helpers ──────────────────────────────────────────────────────

def test_parallel_gc_content_matches_the_sequential_reference(thread_pool):
    assert parallel_gc_content(SEQUENCES, workers=2) == [gc_content(s) for s in SEQUENCES]
    assert parallel_gc_content([], workers=2) == []


def test_parallel_reverse_complement_matches_the_sequential_reference(thread_pool):
    assert parallel_reverse_complement(SEQUENCES, workers=2) == \
           [reverse_complement(s) for s in SEQUENCES]


def test_parallel_translate_carries_the_reading_frame_through_the_pool(thread_pool):
    assert parallel_translate(SEQUENCES, frame=1, workers=2) == \
           [translate(s, frame=1) for s in SEQUENCES]


def test_parallel_align_pairs_uses_the_requested_algorithm(thread_pool):
    assert parallel_align_pairs(PAIRS, workers=2) == \
           [needleman_wunsch(a, b) for a, b in PAIRS]
    assert parallel_align_pairs(PAIRS, algorithm='smith_waterman', workers=2) == \
           [smith_waterman(a, b) for a, b in PAIRS]


def test_parallel_align_pairs_never_sends_its_closure_to_a_process_pool(
    monkeypatch, posix
):
    def refuse(*args, **kwargs):
        raise AssertionError("a process pool was chosen for a closure")

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", refuse)
    assert parallel_align_pairs(PAIRS, workers=2) == \
           [needleman_wunsch(a, b) for a, b in PAIRS]


# ── default pool singleton ───────────────────────────────────────────────────

@pytest.fixture
def fresh_default_pool():
    shutdown_default_pool()
    yield
    shutdown_default_pool()


def test_default_pool_is_created_once_and_reused(fresh_default_pool, monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 3)
    first = get_default_pool(workers=2)
    assert isinstance(first, ParallelBatchProcessor)
    assert first.workers == 2
    assert get_default_pool(workers=99) is first      # the second call ignores args
    shutdown_default_pool()
    assert parallel._default_pool is None


def test_default_pool_auto_detects_workers_on_the_first_call_only(
    fresh_default_pool, monkeypatch
):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 3)
    pool = get_default_pool(io_bound=True)
    assert pool.workers == 6 and pool.io_bound is True
    assert get_default_pool() is pool
