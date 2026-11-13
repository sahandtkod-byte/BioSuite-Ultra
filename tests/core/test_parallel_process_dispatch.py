"""Tests for parallel.py proactive picklability dispatch."""
from biosuite.core.parallel import (
    parallel_map, parallel_submit, ParallelBatchProcessor,
    get_default_pool, shutdown_default_pool,
)


def double(x):
    return x * 2


def test_order_and_correctness_processpath():
    assert parallel_map(double, list(range(30)), workers=2) == [2 * i for i in range(30)]


def test_unpicklable_closure_still_parallel_no_none():
    def local_sq(x):
        return x * x
    got = parallel_map(local_sq, list(range(20)), workers=3)
    assert got == [i * i for i in range(20)]
    assert all(v is not None for v in got)


def test_progress_callback_counts():
    calls = []
    parallel_map(double, list(range(12)), workers=2,
                 progress_callback=lambda c, t: calls.append((c, t)))
    assert calls and calls[-1][1] == 12


def test_parallel_submit_heterogeneous():
    tasks = [(double, (5,), {}), (str, (42,), {}), (len, ((1, 2),), {})]
    assert parallel_submit(tasks, workers=2) == [10, '42', 2]


def test_error_item_yields_none_not_crash():
    def boom(x):
        if x == 2:
            raise ValueError('x')
        return x
    got = parallel_map(boom, [1, 2, 3, 4, 5, 6], workers=2, io_bound=True)
    assert got[1] is None and got[0] == 1


def test_batch_processor_stats():
    p = ParallelBatchProcessor(workers=2)
    got = p.process(double, list(range(10)), batch_size=3)
    assert got == [2 * i for i in range(10)]
    assert p.stats['completed'] == 10
    shutdown_default_pool()
    assert get_default_pool() is not None
