"""Authenticated, owner-isolated asynchronous jobs; no arbitrary execution."""

import hashlib
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from . import auth
from .job_executor import JobExecutor, validate_submission
from .job_store import QueueFullError
from .scopes import key_scope

router = APIRouter(prefix='/api/v1/jobs', tags=['jobs'])
MAX_BODY = 101024
_service_lock = threading.Lock()


class JobSubmission(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    capability_id: str
    params: dict


@asynccontextmanager
async def jobs_lifespan(app):
    app.state.job_executor = None
    try:
        yield
    finally:
        if app.state.job_executor is not None:
            app.state.job_executor.close()
            app.state.job_executor = None


def service(request):
    with _service_lock:
        if not hasattr(request.app.state, 'job_executor'):
            raise HTTPException(503, 'Job service requires application lifespan')
        if request.app.state.job_executor is None:
            root = Path(os.environ.get('BIOSUITE_DATA_DIR', '.')).resolve()
            path = root / '.biosuite-jobs' / 'jobs.sqlite3'
            try:
                request.app.state.job_executor = JobExecutor(path)
            except (OSError, RuntimeError):
                raise HTTPException(503, 'Job service unavailable') from None
        return request.app.state.job_executor


def identity(request):
    key = request.headers.get('X-API-Key')
    if not key:
        raise HTTPException(401, 'Invalid or missing API key')
    administrator = auth.key_matches(key) or key_scope(key, auth.SCOPED_KEYS) == 'admin'
    return hashlib.sha256(key.encode('utf-8')).hexdigest(), administrator


def owned_job(request, job_id):
    owner, administrator = identity(request)
    store = service(request).store
    store.cleanup()
    job = store.get(job_id)
    if job is None or (job['owner'] != owner and not administrator):
        raise HTTPException(404, 'Job not found')
    return job


def status(job):
    return {name: job[name] for name in
            ('id', 'capability_id', 'state', 'created', 'updated', 'error')}


@router.post('', status_code=202, openapi_extra={
    'requestBody': {'required': True, 'content': {
        'application/json': {'schema': JobSubmission.model_json_schema()}}}})
async def submit_job(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY:
            raise HTTPException(413, 'Job request too large')
    try:
        submission = JobSubmission.model_validate_json(body)
        params = validate_submission(submission.capability_id, submission.params)
    except (ValidationError, ValueError):
        raise HTTPException(422, 'Invalid job submission') from None
    owner, _ = identity(request)
    executor = service(request)
    try:
        job_id = executor.submit(owner, submission.capability_id, params)
    except QueueFullError:
        raise HTTPException(429, 'Job capacity reached', headers={'Retry-After': '1'}) from None
    return status(executor.store.get(job_id))


@router.get('/{id}')
def get_job(id: str, request: Request):
    return status(owned_job(request, id))


@router.get('/{id}/result')
def get_result(id: str, request: Request):
    job = owned_job(request, id)
    if job['state'] != 'succeeded':
        raise HTTPException(409, 'Job has no successful result')
    return {'id': id, 'result': json.loads(job['result'])}


@router.post('/{id}/cancel')
def cancel_job(id: str, request: Request):
    owned_job(request, id)
    return status(service(request).store.cancel(id))
