"""SQLite job metadata with atomic transitions and terminal-only retention."""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class QueueFullError(Exception):
    pass


class JobStore:
    def __init__(self, path, *, capacity=32, max_jobs=1024, retention=86400,
                 result_limit=16384):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.capacity = capacity
        self.max_jobs = max_jobs
        self.retention = retention
        self.result_limit = result_limit
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self.db.close()
            raise ValueError("Unsupported job schema version")
        with self.db:
            self.db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                capability_id TEXT NOT NULL, params TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN
                    ('queued','running','cancellation_requested',
                     'succeeded','failed','cancelled')),
                created REAL NOT NULL, updated REAL NOT NULL,
                result TEXT, error TEXT)""")
            self.db.execute("CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, created)")
            self.db.execute("CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner)")
            self.db.execute("CREATE INDEX IF NOT EXISTS jobs_updated ON jobs(updated)")
            self.db.execute("PRAGMA user_version = 1")
        self.path.chmod(0o600)

    def close(self):
        self.db.close()

    def cleanup(self):
        with self.lock, self.db:
            return self.db.execute(
                "DELETE FROM jobs WHERE state IN ('succeeded','failed','cancelled') "
                "AND updated < ?", (time.time() - self.retention,)
            ).rowcount

    def recover(self):
        with self.lock, self.db:
            self.db.execute(
                "UPDATE jobs SET state = CASE WHEN state = 'cancellation_requested' "
                "THEN 'cancelled' ELSE 'failed' END, error = 'worker_restarted', "
                "updated = ? WHERE state IN ('running','cancellation_requested')",
                (time.time(),),
            )

    def submit(self, owner, capability_id, params):
        self.cleanup()
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            active, total = self.db.execute(
                "SELECT COALESCE(SUM(state IN ('queued','running','cancellation_requested')), 0), "
                "COUNT(*) FROM jobs"
            ).fetchone()
            if active >= self.capacity or total >= self.max_jobs:
                raise QueueFullError("Job capacity reached")
            job_id = uuid.uuid4().hex
            now = time.time()
            self.db.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, 'queued', ?, ?, NULL, NULL)",
                (job_id, owner, capability_id, json.dumps(params, allow_nan=False), now, now),
            )
            return job_id

    def get(self, job_id):
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row is not None else None

    def claim(self):
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT * FROM jobs WHERE state = 'queued' ORDER BY created, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            self.db.execute("UPDATE jobs SET state = 'running', updated = ? WHERE id = ?",
                            (time.time(), row['id']))
            return dict(row)

    def finish(self, job_id, *, result=None, error=None):
        encoded = None
        if error is None:
            try:
                encoded = json.dumps(result, sort_keys=True, allow_nan=False)
                if len(encoded.encode('utf-8')) > self.result_limit:
                    error = 'result_too_large'
                    encoded = None
            except (TypeError, ValueError):
                error = 'invalid_result'
        with self.lock, self.db:
            self.db.execute(
                "UPDATE jobs SET state = CASE WHEN state = 'cancellation_requested' "
                "THEN 'cancelled' ELSE ? END, "
                "result = CASE WHEN state = 'cancellation_requested' THEN NULL ELSE ? END, "
                "error = CASE WHEN state = 'cancellation_requested' THEN NULL ELSE ? END, "
                "updated = ? WHERE id = ? AND state IN ('running','cancellation_requested')",
                ('failed' if error else 'succeeded', encoded, error, time.time(), job_id),
            )

    def cancel(self, job_id):
        with self.lock, self.db:
            self.db.execute(
                "UPDATE jobs SET state = CASE WHEN state = 'queued' THEN 'cancelled' "
                "ELSE 'cancellation_requested' END, updated = ? "
                "WHERE id = ? AND state IN ('queued','running')",
                (time.time(), job_id),
            )
        return self.get(job_id)
