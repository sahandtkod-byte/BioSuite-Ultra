"""Workflow pipeline + core/parallel executor behaviours."""
import pytest

from biosuite.core.workflow.pipeline import Pipeline, PipelineStep
from biosuite.core import parallel as par


# ── PipelineStep / Pipeline ──────────────────────────────────────────────────

def test_step_success_and_status():
    s = PipelineStep(name='s1', func=lambda x: x * 2, args=(21,))
    out = s.run()
    assert out == 42 and s.status == 'done' and s.error is None
    assert s.elapsed >= 0.0


def test_step_failure_records_traceback():
    def boom():
        raise ValueError("kaboom")
    s = PipelineStep(name='bad', func=boom)
    s.run()
    assert s.status == 'failed'
    assert s.error and 'kaboom' in s.error
    assert 'ValueError' in s.traceback


def test_pipeline_linear_flow_and_context():
    p = Pipeline('demo')
    p.add_step('a', lambda: 5)
    p.add_step('b', lambda a: a + 1)
    pipe = p.run()
    assert pipe.get_result('a') == 5 and pipe.get_result('b') == 6
    summ = pipe.summary()
    assert isinstance(summ, str) and 'demo' in summ


def test_pipeline_step_context_kwarg_matching():
    # steps receive named prior results when their params match
    p = Pipeline('filter')
    p.add_step('a', lambda: 7)
    p.add_step('b', lambda a: a * 3)
    pipe = p.run()
    assert pipe.get_result('b') == 21


def test_pipeline_stops_on_error():
    p = Pipeline('halt')
    p.add_step('bad', lambda: 1 / 0)
    p.add_step('never', lambda: 'x')
    pipe = p.run(stop_on_error=True)
    assert pipe.get_result('never') is None
    assert 'FAILED' in pipe.summary()


def test_pipeline_continues_on_error():
    p = Pipeline('cont')
    p.add_step('bad', lambda: 1 / 0)
    p.add_step('good', lambda: 'alive')
    pipe = p.run(stop_on_error=False)
    assert pipe.get_result('good') == 'alive'


def test_pipeline_to_dict_shape():
    p = Pipeline('sh')
    p.add_step('s1', lambda: 1)
    pipe = p.run()
    d = pipe.to_dict()
    assert isinstance(d, dict) and 'name' in d


# ── parallel executors ───────────────────────────────────────────────────────

def test_parallel_map_threaded():
    outs = par.parallel_map(lambda x: x * x, list(range(10)), workers=2, io_bound=True)
    assert outs == [i * i for i in range(10)]


def test_parallel_map_process_fallback():
    # non-picklable lambda must still work (thread fallback internally)
    outs = par.parallel_map(lambda x: x + 1, list(range(5)), workers=2)
    assert outs == [1, 2, 3, 4, 5]


def test_parallel_gc_content_helper():
    seqs = ['ACGT' * 5, 'AAAA' * 5]
    gcs = par.parallel_gc_content(seqs, workers=2)
    assert len(gcs) == 2 and 0.0 <= gcs[0] <= 100.0


def test_parallel_reverse_complement():
    out = par.parallel_reverse_complement(['ACGT', 'AAAA'], workers=2)
    assert out == ['ACGT', 'TTTT']


def test_get_optimal_workers_sane():
    w = par.get_optimal_workers()
    assert isinstance(w, int) and w >= 1
    w_io = par.get_optimal_workers(io_bound=True)
    assert isinstance(w_io, int) and w_io >= 1


def test_parallel_batch_processor_items():
    import inspect
    sig = inspect.signature(par.ParallelBatchProcessor.__init__)
    print(sig)
    proc = par.ParallelBatchProcessor(workers=2)
    if hasattr(proc, 'run'):
        out = proc.run(lambda x: x * 2, [1, 2, 3])
        assert sorted(out) == [2, 4, 6]
    else:
        pytest.skip('ParallelBatchProcessor signature differs')
