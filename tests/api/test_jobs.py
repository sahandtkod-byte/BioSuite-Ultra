import asyncio
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from biosuite.api import app, auth, jobs
from biosuite.api.job_executor import execute


@pytest.fixture
def client(tmp_path, monkeypatch):
    original = auth.SCOPED_KEYS
    monkeypatch.setenv('BIOSUITE_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('BIOSUITE_SCOPED_KEYS', 'owner-a:execute;owner-b:execute;reader:read;admin:admin')
    monkeypatch.setattr(auth, 'API_KEY', 'master-job-key')
    auth.reload_scoped_keys()
    with TestClient(app) as instance:
        yield instance
    auth.SCOPED_KEYS = original


def headers(key='owner-a'):
    return {'X-API-Key': key}


def submit(client, sequence='ATGC', key='owner-a'):
    response = client.post('/api/v1/jobs', headers=headers(key), json={
        'capability_id': 'sequence.gc_content', 'params': {'sequence': sequence}})
    assert response.status_code == 202, response.text
    return response.json()['id']


def wait(client, job_id, key='owner-a'):
    deadline = time.monotonic() + 5
    while True:
        response = client.get(f'/api/v1/jobs/{job_id}', headers=headers(key))
        assert response.status_code == 200
        if response.json()['state'] not in ('queued', 'running'):
            return response.json()
        assert time.monotonic() < deadline
        threading.Event().wait(0.01)


@pytest.mark.parametrize('sequence,expected', [('ATGC', 50.0), ('GGCC', 100.0), ('AATT', 0.0)])
def test_real_job_end_to_end_and_restart(client, sequence, expected):
    job_id = submit(client, sequence)
    assert wait(client, job_id)['state'] == 'succeeded'
    response = client.get(f'/api/v1/jobs/{job_id}/result', headers=headers())
    assert response.status_code == 200
    assert response.json()['result'] == {'gc_percent': expected, 'sequence_length': 4}
    app.state.job_executor.close()
    app.state.job_executor = None
    assert client.get(f'/api/v1/jobs/{job_id}/result', headers=headers()).json() == response.json()


@pytest.mark.parametrize('payload', [
    {}, {'capability_id': 'sequence.gc_content', 'params': {}, 'extra': 'secret'},
    {'capability_id': 'unknown', 'params': {}},
    {'capability_id': 'sequence.reverse_complement', 'params': {'sequence': 'ATGC'}},
    {'capability_id': 'sequence.gc_content', 'params': {'sequence': '../private'}},
    {'capability_id': 'sequence.gc_content', 'params': {'sequence': 'A' * 100001}},
    {'capability_id': 'sequence.gc_content', 'params': []},
])
def test_invalid_submissions_are_sanitized(client, payload, caplog):
    response = client.post('/api/v1/jobs', headers=headers(), json=payload)
    assert response.status_code == 422
    assert response.json() == {'detail': 'Invalid job submission'}
    assert 'secret' not in response.text + caplog.text
    assert app.state.job_executor is None


def test_malformed_and_oversized_body(client):
    response = client.post('/api/v1/jobs', headers=headers(), content='not-json')
    assert response.status_code == 422
    response = client.post('/api/v1/jobs', headers=headers(), content=b'x' * (jobs.MAX_BODY + 1))
    assert response.status_code == 413
    assert app.state.job_executor is None


def test_missing_invalid_and_read_key(client):
    for key in (None, 'wrong'):
        response = client.post('/api/v1/jobs', headers={} if key is None else headers(key), json={})
        assert response.status_code == 401
    response = client.post('/api/v1/jobs', headers={**headers('reader'),
        'X-Request-ID': 'job-scope-check', 'X-BioSuite-Error-Envelope': '1'}, json={})
    assert response.status_code == 403
    assert response.headers['X-Request-ID'] == 'job-scope-check'
    assert response.json()['error']['status'] == 403
    assert app.state.job_executor is None


def test_ownership_admin_master_and_no_keys_stored(client, tmp_path, caplog):
    job_id = submit(client)
    assert wait(client, job_id)['state'] == 'succeeded'
    for suffix, method in (('', 'get'), ('/result', 'get'), ('/cancel', 'post')):
        response = getattr(client, method)(f'/api/v1/jobs/{job_id}{suffix}', headers=headers('owner-b'))
        assert response.status_code == 404
        for administrator in ('admin', 'master-job-key'):
            assert getattr(client, method)(f'/api/v1/jobs/{job_id}{suffix}',
                                           headers=headers(administrator)).status_code == 200
    row = app.state.job_executor.store.get(job_id)
    assert row['owner'] == hashlib.sha256(b'owner-a').hexdigest()
    db_bytes = (tmp_path / '.biosuite-jobs' / 'jobs.sqlite3').read_bytes()
    for credential in ('owner-a', 'owner-b', 'master-job-key'):
        assert credential.encode() not in db_bytes
        assert credential not in caplog.text
    assert client.get('/api/v1/jobs/missing', headers=headers()).status_code == 404


def test_queue_saturation_and_result_conflict(client, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return execute(*args)
    monkeypatch.setattr('biosuite.api.job_executor.execute', blocked)
    job_id = submit(client)
    try:
        assert entered.wait(5)
        app.state.job_executor.store.capacity = 1
        assert client.get(f'/api/v1/jobs/{job_id}/result', headers=headers()).status_code == 409
        response = client.post('/api/v1/jobs', headers=headers(), json={
            'capability_id': 'sequence.gc_content', 'params': {'sequence': 'ATGC'}})
        assert response.status_code == 429
        assert response.headers['Retry-After'] == '1'
        response = client.post(f'/api/v1/jobs/{job_id}/cancel', headers=headers())
        assert response.json()['state'] == 'cancellation_requested'
    finally:
        release.set()
    deadline = time.monotonic() + 5
    while client.get(f'/api/v1/jobs/{job_id}', headers=headers()).json()['state'] != 'cancelled':
        assert time.monotonic() < deadline
        threading.Event().wait(0.01)
    assert client.get(f'/api/v1/jobs/{job_id}/result', headers=headers()).status_code == 409


def test_identity_missing_and_service_lifespan_guard():
    request = Request({'type': 'http', 'headers': [], 'app': FastAPI()})
    with pytest.raises(HTTPException) as exc:
        jobs.identity(request)
    assert exc.value.status_code == 401
    with pytest.raises(HTTPException) as exc:
        jobs.service(request)
    assert exc.value.status_code == 503


def test_service_failure_is_sanitized(client, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('private-credential')
    monkeypatch.setattr(jobs, 'JobExecutor', broken)
    response = client.get('/api/v1/jobs/missing', headers=headers())
    assert response.status_code == 503
    assert response.json() == {'detail': 'Job service unavailable'}


def test_stream_body_limit_before_full_consumption():
    consumed = []
    async def chunks():
        consumed.append(1)
        yield b'x' * jobs.MAX_BODY
        consumed.append(2)
        yield b'x'
        consumed.append(3)
        yield b'unread'
    request = SimpleNamespace(stream=chunks)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(jobs.submit_job(request))
    assert exc.value.status_code == 413
    assert consumed == [1, 2]


def test_submit_job_success_path_in_the_calling_thread(monkeypatch):
    """Exercise the async route body without the TestClient portal thread.

    TestClient runs async endpoints on its portal thread, where coverage on
    Python 3.11 can lose the line data. Calling the route directly keeps the
    success path measured on every supported interpreter.
    """
    class Store:
        def get(self, job_id):
            return {
                'id': job_id,
                'capability_id': 'sequence.gc_content',
                'state': 'queued',
                'created': 1.0,
                'updated': 1.0,
                'error': None,
            }

    class Executor:
        store = Store()

        def submit(self, owner, capability_id, params):
            assert owner
            assert capability_id == 'sequence.gc_content'
            assert params == {'sequence': 'ATGC'}
            return 'direct-job'

    monkeypatch.setattr(auth, 'API_KEY', 'master-job-key')
    app = SimpleNamespace(state=SimpleNamespace(job_executor=Executor()))
    async def chunks():
        yield b'{"capability_id":"sequence.gc_content","params":{"sequence":"ATGC"}}'
    request = SimpleNamespace(
        stream=chunks,
        headers={'X-API-Key': 'master-job-key'},
        app=app,
    )
    result = asyncio.run(jobs.submit_job(request))
    assert result['id'] == 'direct-job'
    assert result['state'] == 'queued'


def test_submit_job_error_paths_in_the_calling_thread(monkeypatch):
    """Measure the 422 and 429 branches without the TestClient portal thread."""
    class FullExecutor:
        def submit(self, owner, capability_id, params):
            raise jobs.QueueFullError('full')

    monkeypatch.setattr(auth, 'API_KEY', 'master-job-key')
    app = SimpleNamespace(state=SimpleNamespace(job_executor=FullExecutor()))

    def request(body):
        async def chunks():
            yield body
        return SimpleNamespace(
            stream=chunks,
            headers={'X-API-Key': 'master-job-key'},
            app=app,
        )

    with pytest.raises(HTTPException) as invalid:
        asyncio.run(jobs.submit_job(
            request(b'{"capability_id":"unknown","params":{}}')))
    assert invalid.value.status_code == 422

    with pytest.raises(HTTPException) as full:
        asyncio.run(jobs.submit_job(request(
            b'{"capability_id":"sequence.gc_content","params":{"sequence":"ATGC"}}')))
    assert full.value.status_code == 429
    assert full.value.headers == {'Retry-After': '1'}
