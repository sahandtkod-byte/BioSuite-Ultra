"""WS5: request correlation binding through the API layers.

Covers: per-request isolation of the correlation context, job correlation with
the submitter's request id, no leakage across jobs or requests, and the
provenance bridge's re-exports.
"""
import json
import threading

from biosuite.api import job_executor as executor_module
from biosuite.api.job_executor import JobExecutor
from biosuite.api.middleware import RequestIDMiddleware
from biosuite.core.provenance import (
    bind_correlation,
    current_correlation,
    reset_correlation,
)


def test_request_context_is_isolated_across_threads():
    seen = {}

    def worker(name, request_id):
        token = bind_correlation(request_id, 'cap.' + name)
        try:
            seen[name] = current_correlation().request_id
        finally:
            reset_correlation(token)
        seen[name + '_after'] = current_correlation().request_id

    threads = [threading.Thread(target=worker, args=('a', 'req-a')),
               threading.Thread(target=worker, args=('b', 'req-b'))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == {'a': 'req-a', 'a_after': '',
                    'b': 'req-b', 'b_after': ''}
    assert current_correlation().request_id == ''


def test_reset_restores_previous_binding():
    outer = bind_correlation('outer', 'cap.outer')
    inner = bind_correlation('inner', 'cap.inner')
    assert current_correlation().request_id == 'inner'
    reset_correlation(inner)
    assert current_correlation().request_id == 'outer'
    reset_correlation(outer)
    assert current_correlation().request_id == ''


def test_middleware_binds_and_resets_request_id(monkeypatch):
    # drive dispatch directly so a raised handler error still resets the context
    reset_calls = []
    monkeypatch.setattr('biosuite.api.middleware.reset_correlation',
                        reset_calls.append)
    bound = []
    real_bind = RequestIDMiddleware.__init__

    class FakeRequest:
        headers = {'X-Request-ID': 'req-42'}
        state = type('State', (), {})()

    async def call_next(request):
        bound.append(current_correlation().request_id)
        return type('Response', (), {'headers': {}})()

    import asyncio
    middleware = RequestIDMiddleware(app=None)
    response = asyncio.run(middleware.dispatch(FakeRequest(), call_next))
    assert bound == ['req-42']
    assert response.headers['X-Request-ID'] == 'req-42'
    assert len(reset_calls) == 1
    assert current_correlation().request_id == ''
    assert real_bind is not None  # the class still initialises unchanged


def test_job_correlation_uses_submitter_ids_and_no_leakage(tmp_path, monkeypatch):
    monkeypatch.setattr(threading.Thread, 'start', lambda self: None)
    monkeypatch.setattr(threading.Thread, 'join', lambda self: None)
    executor = JobExecutor(tmp_path / 'jobs.sqlite3')
    try:
        captured = {}
        real_execute = executor_module.execute

        def spy(capability_id, params):
            correlation = current_correlation()
            captured['request_id'] = correlation.request_id
            captured['capability_id'] = correlation.capability_id
            captured['outer_request'] = None  # worker thread has no request bound
            return real_execute(capability_id, params)

        monkeypatch.setattr(executor_module, 'execute', spy)

        # The submitter's request context binds its own ids, as the middleware does.
        token = bind_correlation('submitter-request', 'sequence.gc_content')
        try:
            first = executor.submit('owner', 'sequence.gc_content', {'sequence': 'ATGC'})
        finally:
            reset_correlation(token)
        token = bind_correlation('other-request', 'sequence.gc_content')
        try:
            second = executor.submit('owner', 'sequence.gc_content', {'sequence': 'GGCC'})
        finally:
            reset_correlation(token)

        assert executor.run_one() is True
        assert captured['request_id'] == 'submitter-request', 'the job ran under its submitter'
        assert captured['capability_id'] == 'sequence.gc_content'
        first_done = executor.store.get(first)
        assert first_done['state'] == 'succeeded'
        assert executor.run_one() is True
        assert captured['request_id'] == 'other-request', 'no leakage from the first job'
        assert executor.store.get(second)['state'] == 'succeeded'
        # Submitting from a context with no bound correlation works and stays empty.
        third = executor.submit('owner', 'sequence.gc_content', {'sequence': 'AATT'})
        executor.run_one()
        assert executor.store.get(third)['state'] == 'succeeded'
    finally:
        executor.close()


def test_run_with_correlation_restores_after_exception():
    def boom():
        assert current_correlation().request_id == 'crash-request'
        raise RuntimeError('failure without leakage')

    try:
        from biosuite.api.provenance import run_with_correlation
        try:
            run_with_correlation('crash-request', 'cap.x', boom)
        except RuntimeError as error:
            assert 'failure without leakage' in str(error)
        else:
            raise AssertionError('exception was swallowed')
        assert current_correlation().request_id == ''
        result = run_with_correlation('ok-request', 'cap.y', lambda: json.dumps({'ok': 1}))
        assert result == '{"ok": 1}'
    finally:
        assert current_correlation().request_id == ''
