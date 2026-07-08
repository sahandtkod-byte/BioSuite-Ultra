"""
Batch processor — run the same analysis on hundreds of samples in parallel.
Tracks progress, collects results, handles failures gracefully.
"""
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict


class BatchJob:
    """Single batch job for one sample."""

    def __init__(self, sample_id, func, args=None, kwargs=None):
        self.sample_id = sample_id
        self.func = func
        self.args = args or []
        self.kwargs = kwargs or {}
        self.result = None
        self.error = None
        self.elapsed = 0.0
        self.status = "pending"

    def run(self):
        self.status = "running"
        start = time.time()
        try:
            self.result = self.func(self.sample_id, *self.args, **self.kwargs)
            self.status = "done"
        except Exception as e:
            self.error = str(e)
            self.traceback = traceback.format_exc()
            self.status = "failed"
        self.elapsed = time.time() - start
        return self.result

    def to_dict(self):
        return {
            "sample_id": self.sample_id,
            "status": self.status,
            "elapsed": round(self.elapsed, 2),
            "error": self.error,
        }


class BatchProcessor:
    """Process multiple samples with the same function in parallel."""

    def __init__(self, name="batch"):
        self.name = name
        self.jobs = []
        self.results = OrderedDict()
        self._log = []
        self._progress = 0
        self._total = 0

    def add_job(self, sample_id, func, args=None, kwargs=None):
        self.jobs.append(BatchJob(sample_id, func, args, kwargs))
        return self

    def add_samples(self, sample_ids, func, args=None, kwargs=None):
        for sid in sample_ids:
            self.add_job(sid, func, args, kwargs)
        return self

    def run(self, max_workers=4, progress_callback=None):
        self._log = []
        self.results = OrderedDict()
        self._progress = 0
        self._total = len(self.jobs)
        start = time.time()

        if max_workers <= 1:
            self._run_sequential(progress_callback)
        else:
            self._run_parallel(max_workers, progress_callback)

        total = time.time() - start
        n_done = sum(1 for j in self.jobs if j.status == "done")
        n_fail = sum(1 for j in self.jobs if j.status == "failed")
        self._log.append(f"Batch '{self.name}' finished: {n_done} done, {n_fail} failed in {total:.2f}s")
        return self

    def _run_sequential(self, progress_callback):
        for job in self.jobs:
            self._log.append(f"Processing: {job.sample_id}")
            job.run()
            self.results[job.sample_id] = job.result
            self._progress += 1
            if progress_callback:
                progress_callback(self._progress, self._total, job.sample_id)
            if job.status == "failed":
                self._log.append(f"  FAILED: {job.error}")

    def _run_parallel(self, max_workers, progress_callback):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(job.run): job for job in self.jobs}
            for future in as_completed(futures):
                job = futures[future]
                self.results[job.sample_id] = job.result
                self._progress += 1
                if progress_callback:
                    progress_callback(self._progress, self._total, job.sample_id)
                if job.status == "failed":
                    self._log.append(f"  {job.sample_id} FAILED: {job.error}")
                else:
                    self._log.append(f"  {job.sample_id} done in {job.elapsed:.2f}s")

    def summary(self):
        n_done = sum(1 for j in self.jobs if j.status == "done")
        n_fail = sum(1 for j in self.jobs if j.status == "failed")
        lines = [
            f"Batch: {self.name}",
            f"Samples: {len(self.jobs)}",
            f"Completed: {n_done}",
            f"Failed: {n_fail}",
            "",
        ]
        for job in self.jobs:
            icon = "+" if job.status == "done" else "X" if job.status == "failed" else "-"
            lines.append(f"  [{icon}] {job.sample_id} ({job.elapsed:.2f}s)")
            if job.error:
                lines.append(f"      {job.error}")
        lines.append("")
        lines.extend(self._log)
        return "\n".join(lines)

    def get_results(self):
        return self.results

    def get_failures(self):
        return [(j.sample_id, j.error) for j in self.jobs if j.status == "failed"]

    def to_dict(self):
        return {
            "name": self.name,
            "total": len(self.jobs),
            "done": sum(1 for j in self.jobs if j.status == "done"),
            "failed": sum(1 for j in self.jobs if j.status == "failed"),
            "jobs": [j.to_dict() for j in self.jobs],
        }


def batch_run(func, sample_ids, *args, max_workers=4, **kwargs):
    """Quick batch run — apply func to each sample_id.

    Args:
        func: callable(sample_id, *args, **kwargs)
        sample_ids: list of sample identifiers
        *args: positional args passed to func
        max_workers: parallel workers
        **kwargs: keyword args passed to func

    Returns:
        dict mapping sample_id -> result
    """
    bp = BatchProcessor()
    for sid in sample_ids:
        bp.add_job(sid, func, args=args, kwargs=kwargs)
    bp.run(max_workers=max_workers)
    return bp.get_results()


def format_batch_report(processor):
    """Format batch results as text report."""
    return processor.summary()
