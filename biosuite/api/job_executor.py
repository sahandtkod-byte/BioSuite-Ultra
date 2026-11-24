"""Single-process worker for an explicit, bounded capability allowlist."""

import json
import re
import sqlite3
import threading
import time

from biosuite.capabilities.catalog import get_capability
from biosuite.core.sequence import gc_content

from .job_store import JobStore
from .provenance import current_correlation, run_with_correlation

MAX_SEQUENCE = 100000
EXECUTION_SECONDS = 5.0
ALLOWLIST = frozenset({'sequence.gc_content'})

# Bounded provenance payloads: a submitted sequence can be 100k characters and
# is never stored verbatim in the audit trail, only as a length placeholder.
RESULT_SUMMARY_LIMIT = 200


def _params_summary(params):
    return {key: (f'<string length={len(value)}>'
                  if isinstance(value, str) else value)
            for key, value in params.items()}


def _bounded_json(payload, limit):
    return json.dumps(payload, sort_keys=True, default=str)[:limit]


def validate_submission(capability_id, params):
    if not isinstance(capability_id, str):
        raise ValueError('Invalid capability')
    try:
        get_capability(capability_id)
    except KeyError:
        raise ValueError('Unknown capability') from None
    if capability_id not in ALLOWLIST:
        raise ValueError('Capability not enabled for jobs')
    if not isinstance(params, dict) or set(params) != {'sequence'}:
        raise ValueError('Expected only sequence parameter')
    sequence = params['sequence']
    if not isinstance(sequence, str) or not 1 <= len(sequence) <= MAX_SEQUENCE:
        raise ValueError('Sequence must contain 1 to 100000 characters')
    if re.fullmatch(r'[ACGTURYSWKMBDHVNacgturyswkmbdhvn.*-]+', sequence) is None:
        raise ValueError('Invalid nucleotide sequence')
    return {'sequence': sequence}


def execute(capability_id, params):
    params = validate_submission(capability_id, params)
    return {'gc_percent': gc_content(params['sequence']),
            'sequence_length': len(params['sequence'])}


class JobExecutor:
    def __init__(self, path, *, provenance_recorder=None, **store_options):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lease = sqlite3.connect(str(path) + '.owner', timeout=0, check_same_thread=False)
        try:
            self.lease.execute('BEGIN EXCLUSIVE')
        except sqlite3.OperationalError:
            self.lease.close()
            raise RuntimeError('Job store already has an owning worker') from None
        try:
            self.store = JobStore(path, **store_options)
        except Exception:
            self.lease.close()
            raise
        self.store.recover()
        # API-level provenance hook: jobs.service passes the process-wide
        # tracker's record method; core never imports this layer.  None keeps
        # the legacy behaviour of recording nothing.
        self.provenance_recorder = provenance_recorder
        self.stopping = threading.Event()
        self.wake = threading.Event()
        # Submitter correlation (WS5), keyed by job id.  Captured in the
        # request context at submit time and consumed by the worker at claim
        # time; entries never outlive their job's claim or the executor.
        self._correlations = {}
        self._correlations_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name='biosuite-jobs', daemon=True)
        self.thread.start()

    def submit(self, owner, capability_id, params):
        params = validate_submission(capability_id, params)
        correlation = current_correlation()
        job_id = self.store.submit(owner, capability_id, params)
        with self._correlations_lock:
            self._correlations[job_id] = correlation
        self.wake.set()
        return job_id

    def run_one(self):
        job = self.store.claim()
        if job is None:
            return False
        with self._correlations_lock:
            correlation = self._correlations.pop(job['id'], None)
        request_id = correlation.request_id if correlation else ''
        start = time.monotonic()
        result = None
        error = None
        try:
            result = run_with_correlation(
                request_id, job['capability_id'], execute,
                job['capability_id'], json.loads(job['params']))
        except Exception:
            result = None
            error = 'execution_failed'
        elapsed = time.monotonic() - start
        elapsed_ms = elapsed * 1000
        if error is None and elapsed > EXECUTION_SECONDS:
            result = None
            error = 'execution_timeout'
        # WS10: the outcome is fully computed before anything is published.
        # The row is still 'running' (or 'cancellation_requested'), never
        # terminal, so a pre-finish read decides cancellation; finish() maps
        # 'cancellation_requested' to 'cancelled' atomically.  One bounded
        # provenance step is recorded synchronously BEFORE the terminal state
        # is published, so any client that observes a terminal state finds its
        # provenance step already in the trail — there is no observe-then-wait
        # race, and exactly one step is written because only this worker
        # records jobs it claimed (queued cancellations are recorded by the
        # cancel route via record_queued_cancellation, which cannot race a
        # claim).  A recorder failure leaves the job's own state untouched.
        state = self.store.get(job['id'])['state']
        if state == 'cancellation_requested':
            self._record_outcome(job, request_id, elapsed_ms, status='skipped',
                                 result_summary='cancelled', error_message='')
        elif error is None:
            self._record_outcome(job, request_id, elapsed_ms, status='success',
                                 result_summary=_bounded_json(
                                     result, RESULT_SUMMARY_LIMIT),
                                 error_message='')
        else:
            self._record_outcome(job, request_id, elapsed_ms, status='error',
                                 result_summary=str(error)[:RESULT_SUMMARY_LIMIT],
                                 error_message=error)
        if state == 'cancellation_requested':
            self.store.finish(job['id'])
        elif error is None:
            self.store.finish(job['id'], result=result)
        else:
            self.store.finish(job['id'], error=error)
        return True

    def _record_outcome(self, job, request_id, elapsed_ms, *, status,
                        result_summary, error_message):
        recorder = self.provenance_recorder
        if recorder is None:
            return
        try:
            capability = get_capability(job['capability_id'])
            recorder(
                module=capability.module,
                function=capability.function,
                params=_params_summary(json.loads(job['params'])),
                result_summary=result_summary,
                execution_time_ms=elapsed_ms,
                engine='builtin',
                status=status,
                error_message=error_message,
                request_id=request_id,
                capability_id=job['capability_id'])
        except Exception:
            # Provenance is an audit trail, never a job failure: a broken or
            # raising recorder leaves the job's own state untouched.
            pass

    def _run(self):
        while not self.stopping.is_set():
            self.wake.clear()
            self.store.cleanup()
            while not self.stopping.is_set() and self.run_one():
                pass
            self.wake.wait(timeout=1.0)

    def close(self):
        self.stopping.set()
        self.wake.set()
        self.thread.join()
        self.store.close()
        self.lease.close()
