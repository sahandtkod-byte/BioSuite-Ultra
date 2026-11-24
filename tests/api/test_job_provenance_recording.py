"""WS8: job outcomes land in the provenance audit trail (API-level e2e).

The WS5 bridge proved request correlation; these tests prove the missing half:
each terminal job outcome records exactly one bounded provenance step carrying
the submitter's request id, the capability's module/function, a params summary
that never contains the full submitted sequence, the result summary, timing,
engine, status, error and ids.
"""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from biosuite.api import app, auth, job_executor, jobs
from biosuite.core.provenance import ProvenanceTracker, set_global_tracker


@pytest.fixture
def tracker_client(tmp_path, monkeypatch):
    tracker = ProvenanceTracker()  # in-memory, own session
    set_global_tracker(tracker)
    monkeypatch.setenv('BIOSUITE_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(auth, 'API_KEY', 'master-job-key')
    try:
        with TestClient(app) as instance:
            yield instance, tracker
    finally:
        set_global_tracker(None)


def _submit_and_wait(client, request_id, sequence='ATGCATGC', key='master-job-key'):
    response = client.post('/api/v1/jobs', headers={
        'X-API-Key': key, 'X-Request-ID': request_id}, json={
        'capability_id': 'sequence.gc_content', 'params': {'sequence': sequence}})
    assert response.status_code == 202, response.text
    job_id = response.json()['id']
    deadline = time.monotonic() + 5
    while True:
        state = client.get(f'/api/v1/jobs/{job_id}',
                           headers={'X-API-Key': key}).json()['state']
        if state in ('succeeded', 'failed', 'cancelled'):
            return job_id, state
        assert time.monotonic() < deadline
        threading.Event().wait(0.01)


def _steps_for(tracker, request_id):
    return [step for step in tracker.get_steps()
            if step.request_id == request_id]


def test_successful_job_records_one_provenance_step(tracker_client):
    client, tracker = tracker_client
    job_id, state = _submit_and_wait(client, 'prov-e2e-ok')
    assert state == 'succeeded'
    steps = _steps_for(tracker, 'prov-e2e-ok')
    assert len(steps) == 1
    step = steps[0]
    assert step.capability_id == 'sequence.gc_content'
    assert step.module == 'sequence'
    assert step.function == 'gc_content'
    assert step.status == 'success'
    assert step.engine == 'builtin'
    assert step.execution_time_ms >= 0
    assert step.params == {'sequence': '<string length=8>'}
    assert json.loads(step.result_summary) == {'gc_percent': 50.0,
                                               'sequence_length': 8}
    assert step.error_message == ''


def test_failed_job_records_error_status(tracker_client, monkeypatch):
    client, tracker = tracker_client
    monkeypatch.setattr(job_executor, 'execute', _failing_execute)
    job_id, state = _submit_and_wait(client, 'prov-e2e-fail')
    assert state == 'failed'
    steps = _steps_for(tracker, 'prov-e2e-fail')
    assert len(steps) == 1
    step = steps[0]
    assert step.status == 'error'
    assert step.error_message == 'execution_failed'
    assert step.module == 'sequence'
    assert step.function == 'gc_content'


def test_params_summary_never_keeps_a_string_verbatim():
    assert job_executor._params_summary({
        'n': 5, 'flag': True, 'seq': 'ACGT'}) == {
        'n': 5, 'flag': True, 'seq': '<string length=4>'}


def _failing_execute(capability_id, params):
    raise RuntimeError('engine exploded')


def test_large_sequence_is_never_recorded_verbatim(tracker_client):
    client, tracker = tracker_client
    payload = 'ACGT' * 25000  # 100k characters
    job_id, state = _submit_and_wait(client, 'prov-e2e-big', sequence=payload)
    assert state == 'succeeded'
    steps = _steps_for(tracker, 'prov-e2e-big')
    assert len(steps) == 1
    step = steps[0]
    assert step.params == {'sequence': '<string length=100000>'}
    assert payload not in json.dumps(step.params) + step.result_summary
    assert len(step.result_summary) <= 200


def test_recorder_failure_never_fails_the_job(tracker_client, monkeypatch):
    client, tracker = tracker_client

    def broken(**step):
        raise RuntimeError('audit sink down')

    monkeypatch.setattr(jobs, '_record_job_provenance', broken)
    job_id, state = _submit_and_wait(client, 'prov-e2e-broken-recorder')
    assert state == 'succeeded'
    assert _steps_for(tracker, 'prov-e2e-broken-recorder') == []
    row = client.app.state.job_executor.store.get(job_id)
    assert row['state'] == 'succeeded'


def test_broken_global_tracker_never_fails_the_job(tracker_client, monkeypatch):
    client, _ = tracker_client

    class BrokenTracker:
        def record(self, **step):
            raise RuntimeError('audit sink down')

    monkeypatch.setattr(jobs, 'get_global_tracker', BrokenTracker)
    job_id, state = _submit_and_wait(client, 'prov-e2e-broken-tracker')
    assert state == 'succeeded'
    row = client.app.state.job_executor.store.get(job_id)
    assert row['state'] == 'succeeded'
    assert row['result'] == json.dumps({'gc_percent': 50.0, 'sequence_length': 8})
