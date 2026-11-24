import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from biosuite.api.job_store import JobStore, QueueFullError


@pytest.fixture
def store(tmp_path):
    instance = JobStore(tmp_path / 'jobs.sqlite3')
    yield instance
    instance.close()


def submit(store, owner='hashed-owner'):
    return store.submit(owner, 'sequence.gc_content', {'sequence': 'ATGC'})


def test_schema_persistence_and_recovery(tmp_path):
    path = tmp_path / 'jobs.sqlite3'
    store = JobStore(path)
    running = submit(store)
    assert store.claim()['id'] == running
    cancelling = submit(store)
    assert store.claim()['id'] == cancelling
    store.cancel(cancelling)
    queued = submit(store)
    store.close()
    reopened = JobStore(path)
    try:
        assert reopened.db.execute('PRAGMA user_version').fetchone()[0] == 1
        if os.name == 'posix':
            assert path.stat().st_mode & 0o777 == 0o600
        reopened.recover()
        assert reopened.get(running)['state'] == 'failed'
        assert reopened.get(running)['error'] == 'worker_restarted'
        assert reopened.get(cancelling)['state'] == 'cancelled'
        assert reopened.get(queued)['state'] == 'queued'
        assert reopened.claim()['id'] == queued
    finally:
        reopened.close()


def test_reject_unknown_schema(tmp_path):
    path = tmp_path / 'future.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA user_version = 2')
    with pytest.raises(ValueError, match='Unsupported job schema'):
        JobStore(path)


def test_success_result_and_missing(store):
    assert store.get('absent') is None
    assert store.claim() is None
    job_id = submit(store)
    assert store.get(job_id)['state'] == 'queued'
    store.claim()
    store.finish(job_id, result={'gc_percent': 50.0})
    job = store.get(job_id)
    assert job['state'] == 'succeeded'
    assert json.loads(job['result']) == {'gc_percent': 50.0}
    assert job['error'] is None
    assert store.cancel(job_id) == job
    store.finish(job_id, error='execution_failed')
    assert store.get(job_id) == job
    assert store.cancel('absent') is None


@pytest.mark.parametrize('result,error', [
    ('x' * 20000, 'result_too_large'),
    (float('nan'), 'invalid_result'),
    ({'not_json': {1}}, 'invalid_result'),
])
def test_result_bounds(store, result, error):
    job_id = submit(store)
    store.claim()
    store.finish(job_id, result=result)
    job = store.get(job_id)
    assert job['state'] == 'failed'
    assert job['error'] == error
    assert job['result'] is None


def test_cancellation_races(store):
    queued = submit(store)
    assert store.cancel(queued)['state'] == 'cancelled'
    assert store.claim() is None
    for error in (None, 'execution_failed'):
        running = submit(store)
        store.claim()
        assert store.cancel(running)['state'] == 'cancellation_requested'
        assert store.cancel(running)['state'] == 'cancellation_requested'
        store.finish(running, result={'value': 1}, error=error)
        job = store.get(running)
        assert job['state'] == 'cancelled'
        assert job['result'] is None
        assert job['error'] is None


def test_retention_only_terminal_and_capacity(store, monkeypatch):
    monkeypatch.setattr('biosuite.api.job_store.time.time', lambda: 100000)
    completed = submit(store)
    store.cancel(completed)
    queued = submit(store)
    running = submit(store)
    store.claim()
    monkeypatch.setattr('biosuite.api.job_store.time.time', lambda: 200000)
    assert store.cleanup() == 1
    assert store.get(completed) is None
    assert store.get(queued) is not None
    assert store.get(running) is not None
    store.capacity = 2
    with pytest.raises(QueueFullError):
        submit(store)
    store.cancel(queued)
    store.cancel(running)
    store.capacity = 32
    store.max_jobs = 2
    with pytest.raises(QueueFullError):
        submit(store)


def test_atomic_admission_across_connections(tmp_path):
    path = tmp_path / 'atomic.sqlite3'
    first, second = JobStore(path, capacity=1), JobStore(path, capacity=1)
    def attempt(store):
        try:
            return submit(store)
        except QueueFullError:
            return None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            admitted = list(pool.map(attempt, (first, second)))
        assert sum(job_id is not None for job_id in admitted) == 1
        assert first.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 1
    finally:
        first.close()
        second.close()
