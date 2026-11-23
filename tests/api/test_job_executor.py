import json
import threading
import time

import pytest

from biosuite.api import job_executor as module
from biosuite.api.job_executor import JobExecutor, execute, validate_submission
from biosuite.api.job_store import JobStore


@pytest.mark.parametrize('sequence,expected', [('ATGC', 50.0), ('GGCC', 100.0), ('AATT', 0.0)])
def test_real_core_execution(sequence, expected):
    assert execute('sequence.gc_content', {'sequence': sequence}) == {
        'gc_percent': expected, 'sequence_length': len(sequence)}


@pytest.mark.parametrize('capability,params', [
    (None, {}), ('unknown', {}), ('sequence.reverse_complement', {'sequence': 'ATGC'}),
    ('sequence.gc_content', []), ('sequence.gc_content', {}),
    ('sequence.gc_content', {'sequence': 'ATGC', 'path': '../private'}),
    ('sequence.gc_content', {'sequence': None}),
    ('sequence.gc_content', {'sequence': ''}),
    ('sequence.gc_content', {'sequence': 'A' * 100001}),
    ('sequence.gc_content', {'sequence': '../private'}),
])
def test_validation(capability, params):
    with pytest.raises(ValueError):
        validate_submission(capability, params)


def test_boundary_input():
    assert len(validate_submission('sequence.gc_content', {'sequence': 'A' * 100000})['sequence']) == 100000
    assert validate_submission('sequence.gc_content', {'sequence': 'a'}) == {'sequence': 'a'}


@pytest.fixture
def controlled(tmp_path, monkeypatch):
    monkeypatch.setattr(threading.Thread, 'start', lambda self: None)
    monkeypatch.setattr(threading.Thread, 'join', lambda self: None)
    executor = JobExecutor(tmp_path / 'jobs.sqlite3')
    yield executor
    executor.close()


def test_real_worker_lifecycle_and_exclusive_owner(tmp_path):
    path = tmp_path / 'jobs.sqlite3'
    executor = JobExecutor(path)
    thread = executor.thread
    try:
        with pytest.raises(RuntimeError, match='owning worker'):
            JobExecutor(path)
        job_id = executor.submit('owner-hash', 'sequence.gc_content', {'sequence': 'ATGC'})
        deadline = time.monotonic() + 5
        while executor.store.get(job_id)['state'] in ('queued', 'running'):
            assert time.monotonic() < deadline
            threading.Event().wait(0.01)
        assert json.loads(executor.store.get(job_id)['result'])['gc_percent'] == 50.0
    finally:
        executor.close()
    assert not thread.is_alive()
    reopened = JobExecutor(path)
    reopened.close()


def test_store_initialization_failure_releases_lease(tmp_path, monkeypatch):
    path = tmp_path / 'jobs.sqlite3'
    def broken(*args, **kwargs):
        raise ValueError('schema failure')
    monkeypatch.setattr(module, 'JobStore', broken)
    with pytest.raises(ValueError, match='schema failure'):
        JobExecutor(path)
    monkeypatch.setattr(module, 'JobStore', JobStore)
    worker = JobExecutor(path)
    worker.close()


def test_run_one_empty_success_timeout_and_failure(controlled, monkeypatch, caplog):
    assert controlled.run_one() is False
    success = controlled.submit('owner', 'sequence.gc_content', {'sequence': 'GGCC'})
    assert controlled.run_one() is True
    assert controlled.store.get(success)['state'] == 'succeeded'
    timeout = controlled.submit('owner', 'sequence.gc_content', {'sequence': 'AATT'})
    ticks = iter((0, 6))
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks))
    controlled.run_one()
    assert controlled.store.get(timeout)['error'] == 'execution_timeout'
    monkeypatch.setattr(module.time, 'monotonic', lambda: 0)
    failure = controlled.submit('owner', 'sequence.gc_content', {'sequence': 'ATGC'})
    def broken(*args):
        raise RuntimeError('private-credential')
    monkeypatch.setattr(module, 'execute', broken)
    controlled.run_one()
    job = controlled.store.get(failure)
    assert job['state'] == 'failed'
    assert job['error'] == 'execution_failed'
    assert 'private-credential' not in str(job) + caplog.text


def test_cancel_during_real_execution(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real_execute = execute
    def waiting(*args):
        entered.set()
        assert release.wait(5)
        return real_execute(*args)
    monkeypatch.setattr(module, 'execute', waiting)
    executor = JobExecutor(tmp_path / 'cancel.sqlite3')
    try:
        job_id = executor.submit('owner', 'sequence.gc_content', {'sequence': 'ATGC'})
        assert entered.wait(5)
        assert executor.store.cancel(job_id)['state'] == 'cancellation_requested'
        release.set()
        deadline = time.monotonic() + 5
        while executor.store.get(job_id)['state'] != 'cancelled':
            assert time.monotonic() < deadline
            threading.Event().wait(0.01)
        assert executor.store.get(job_id)['result'] is None
    finally:
        release.set()
        executor.close()
