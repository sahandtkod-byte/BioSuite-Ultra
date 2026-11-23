"""Single-process worker for an explicit, bounded capability allowlist."""

import json
import re
import sqlite3
import threading
import time

from biosuite.capabilities.catalog import get_capability
from biosuite.core.sequence import gc_content

from .job_store import JobStore

MAX_SEQUENCE = 100000
EXECUTION_SECONDS = 5.0
ALLOWLIST = frozenset({'sequence.gc_content'})


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
    def __init__(self, path, **store_options):
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
        self.stopping = threading.Event()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._run, name='biosuite-jobs', daemon=True)
        self.thread.start()

    def submit(self, owner, capability_id, params):
        params = validate_submission(capability_id, params)
        job_id = self.store.submit(owner, capability_id, params)
        self.wake.set()
        return job_id

    def run_one(self):
        job = self.store.claim()
        if job is None:
            return False
        start = time.monotonic()
        try:
            result = execute(job['capability_id'], json.loads(job['params']))
            if time.monotonic() - start > EXECUTION_SECONDS:
                self.store.finish(job['id'], error='execution_timeout')
            else:
                self.store.finish(job['id'], result=result)
        except Exception:
            self.store.finish(job['id'], error='execution_failed')
        return True

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
