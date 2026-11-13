"""Parameter refusal, progress bookkeeping and logger identity of ``core/parallel.py``.

Every "before" figure quoted below was measured by executing the untouched HEAD copy
of the module in memory (`git show HEAD:biosuite/core/parallel.py`) before the fix;
the "after" figures were measured against the edited module the same way.
"""
import ast
import logging
import re
from pathlib import Path

import pytest

from biosuite.core import parallel
from biosuite.core.parallel import parallel_map


def double(value):
    return value * 2


def boom_on_three(value):
    """Fails for exactly one item of the batches used below."""
    if value == 3:
        raise ValueError("no three")
    return value * 2


class RecordingPool:
    """Runs the work on threads in this process and records nothing but its use."""

    created = []

    def __init__(self, max_workers=None):
        RecordingPool.created.append(max_workers)
        from concurrent.futures import ThreadPoolExecutor
        self._inner = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn, *args, **kwargs):
        return self._inner.submit(fn, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._inner.shutdown(wait=True)
        return False


@pytest.fixture
def thread_pool(monkeypatch):
    RecordingPool.created.clear()
    monkeypatch.setattr(parallel, "ThreadPoolExecutor", RecordingPool)
    return RecordingPool.created


@pytest.fixture
def forbid_pools(monkeypatch):
    """Reaching for an executor at all is a failure for these calls."""
    def refuse(*args, **kwargs):
        raise AssertionError("no executor may be created for this call")

    monkeypatch.setattr(parallel, "ThreadPoolExecutor", refuse)
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", refuse)


# ── chunk_size validation (C-75) ─────────────────────────────────────────────

BAD_CHUNK_SIZES = [0, -1, -5, 1.5, 2.0, True, False, "2"]


@pytest.mark.parametrize("bad", BAD_CHUNK_SIZES)
def test_parallel_map_refuses_a_chunk_size_that_is_not_a_positive_integer(bad):
    """HEAD accepted all of these: `chunk_size=-5` answered [None] * 8 with no error
    and no log line, `chunk_size=True` returned [[2], [4], None, ...] (a one-element
    list per item), and 1.5 / '2' escaped as the *item's* own ValueError."""
    with pytest.raises(ValueError, match=rf"^chunk_size must be a positive integer, "
                                       rf"got {re.escape(repr(bad))}\.$"):
        parallel_map(double, list(range(8)), workers=2, io_bound=True, chunk_size=bad)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5, 7, 12])
def test_every_real_chunk_size_still_produces_the_ordered_results(thread_pool, chunk_size):
    """Positive control for the guard above: it rejects the eight bad values and
    still accepts the useful ones, with the flat, correctly ordered result."""
    got = parallel_map(double, list(range(12)), workers=2, io_bound=True,
                       chunk_size=chunk_size)
    assert got == [2 * i for i in range(12)]
    assert all(not isinstance(r, list) for r in got), got


def test_chunk_size_none_and_an_absurdly_large_one_both_run_the_whole_batch(thread_pool):
    assert parallel_map(double, list(range(7)), workers=2, io_bound=True,
                        chunk_size=None) == [2 * i for i in range(7)]
    assert parallel_map(double, list(range(7)), workers=2, io_bound=True,
                        chunk_size=10 ** 9) == [2 * i for i in range(7)]


def test_a_refused_chunk_size_never_reaches_an_executor(forbid_pools):
    """The check is up front: the old code only discovered the bad value inside the
    pool, and the executor failure was then papered over by the sequential fallback."""
    with pytest.raises(ValueError, match="^chunk_size must be a positive integer"):
        parallel_map(double, list(range(8)), workers=2, io_bound=True, chunk_size=-5)


def test_the_empty_batch_still_short_circuits_before_any_validation():
    """Documented precedence: no items means no work, whatever chunk_size says."""
    assert parallel_map(double, [], workers=2, io_bound=True, chunk_size=-5) == []


# ── progress reporting across a failure (C-76) ───────────────────────────────

def test_progress_callback_reports_the_unit_that_failed(thread_pool):
    """HEAD reported 6 calls for 7 items — [(1,7), (2,7), (4,7), (5,7), (6,7),
    (7,7)] — because the callback sat inside the success branch.  A progress bar
    wired to it therefore never reached its total on any batch with a bad item."""
    seen = []
    got = parallel_map(boom_on_three, [1, 2, 3, 4, 5, 6, 7], workers=2, io_bound=True,
                       progress_callback=lambda done, total: seen.append((done, total)))
    assert got == [2, 4, None, 8, 10, 12, 14]
    assert seen == [(1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7)]


def test_a_failed_chunk_reports_its_full_item_count(thread_pool):
    """The chunk [2, 3] is two items of work, so its failure has to advance the
    counter by two or the callback would never report all five items.

    Threads finish in unpredictable order, so what is pinned is the sequence of
    *increments* (the real chunk sizes 2, 2, 1) and the final total, not the order."""
    seen = []
    got = parallel_map(boom_on_three, [1, 2, 3, 4, 5], workers=2, io_bound=True,
                       chunk_size=2,
                       progress_callback=lambda done, total: seen.append((done, total)))
    assert got == [2, 4, None, None, 10]
    assert [total for _, total in seen] == [5, 5, 5]
    increments = [seen[0][0]] + [b - a for (a, _), (b, _) in zip(seen, seen[1:])]
    assert sorted(increments) == [1, 2, 2]
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)
    assert seen[-1] == (5, 5)


def test_a_failed_chunk_of_the_whole_batch_still_reports_the_total(thread_pool):
    seen = []
    got = parallel_map(boom_on_three, [1, 2, 3, 4, 5], workers=2, io_bound=True,
                       chunk_size=5,
                       progress_callback=lambda done, total: seen.append((done, total)))
    assert got == [None, None, None, None, None]
    assert seen == [(5, 5)]


# ── failure logging (C-74, C-78) ─────────────────────────────────────────────

def test_the_module_logs_under_its_own_package_name():
    """HEAD used a hand-written 'biosuite.parallel', which sits outside the
    biosuite.core tree that the rest of the package logs under."""
    assert parallel.logger.name == "biosuite.core.parallel"


def test_failure_records_are_emitted_by_the_module_logger(thread_pool, caplog):
    with caplog.at_level(logging.ERROR, logger="biosuite.core.parallel"):
        parallel_map(boom_on_three, [1, 2, 3, 4, 5, 6], workers=2, io_bound=True)
    assert [(r.name, r.getMessage()) for r in caplog.records] == [
        ("biosuite.core.parallel", "Parallel task failed at index 2: no three")
    ]


def test_a_failed_chunk_is_logged_as_the_span_it_covered(thread_pool, caplog):
    """The chunk starting at 0 covers items 0-2, so naming index 0 pointed the reader
    at an item that had run perfectly before the third one raised."""
    with caplog.at_level(logging.ERROR, logger="biosuite.core.parallel"):
        got = parallel_map(boom_on_three, [1, 2, 3, 4, 5], workers=2, io_bound=True,
                           chunk_size=3)
    assert got == [None, None, None, 8, 10]
    assert [r.getMessage() for r in caplog.records] == [
        "Parallel task failed for items 0-2: no three"
    ]


def test_a_single_item_chunk_keeps_the_per_item_wording(thread_pool, caplog):
    with caplog.at_level(logging.ERROR, logger="biosuite.core.parallel"):
        parallel_map(boom_on_three, [1, 2, 3, 4, 5], workers=2, io_bound=True,
                     chunk_size=1)
    assert [r.getMessage() for r in caplog.records] == [
        "Parallel task failed at index 2: no three"
    ]


def test_no_core_module_names_a_logger_with_a_hand_written_string():
    """The class of defect C-74 belongs to, closed statically over the package.

    Every module-level ``logger = ...`` must be derived from ``__name__``; a literal
    ('biosuite.parallel') detaches the records from the package tree that handlers and
    filters address.  The number of assignments the scan found is asserted as a
    **floor**, not an equality: the guard exists so that a scan which silently matched
    nothing cannot pass, and a module gaining a proper logger (``blast.py`` did in
    cluster 16) is progress that an equality would punish.
    """
    offenders, found = [], 0
    core_root = Path(__file__).resolve().parents[2] / "biosuite" / "core"
    scanned = 0
    for path in sorted(core_root.rglob("*.py")):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not (isinstance(node, ast.Assign)
                    and any(getattr(t, "id", "") == "logger" for t in node.targets)):
                continue
            found += 1
            value = node.value
            if not (isinstance(value, ast.Call) and len(value.args) == 1
                    and isinstance(value.args[0], ast.Name)
                    and value.args[0].id == "__name__"):
                offenders.append(str(path))
    assert scanned >= 49, f"the scan read {scanned} core files; the package has 49"
    assert found >= 17, (f"the scan saw {found} module loggers; the package had 17 "
                         f"when this guard was written")
    assert offenders == [], f"loggers not derived from __name__: {offenders}"
