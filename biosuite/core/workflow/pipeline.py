"""
Pipeline builder — chain bioinformatics steps into automated workflows.
Each step is a function + kwargs. Pipelines are serial by default,
with optional parallel branches for independent steps.
"""
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed


class PipelineStep:
    """Single step in a pipeline."""

    def __init__(self, name, func, args=None, kwargs=None, description=""):
        self.name = name
        self.func = func
        self.args = args or []
        self.kwargs = kwargs or {}
        self.description = description
        self.result = None
        self.error = None
        self.elapsed = 0.0
        self.status = "pending"

    def run(self, context=None):
        self.status = "running"
        ctx = dict(context) if context else {}
        start = time.time()
        try:
            merged_kwargs = {**self.kwargs, **ctx}
            self.result = self.func(*self.args, **merged_kwargs)
            self.status = "done"
        except Exception as e:
            self.error = str(e)
            self.traceback = traceback.format_exc()
            self.status = "failed"
        self.elapsed = time.time() - start
        return self.result

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "elapsed": round(self.elapsed, 2),
            "error": self.error,
        }


class Pipeline:
    """Bioinformatics pipeline — ordered steps with context passing."""

    def __init__(self, name="pipeline"):
        self.name = name
        self.steps = []
        self.context = {}
        self.results = OrderedDict()
        self._log = []

    def add_step(self, name, func, args=None, kwargs=None, description=""):
        step = PipelineStep(name, func, args, kwargs, description)
        self.steps.append(step)
        return self

    def add_steps(self, step_list):
        for s in step_list:
            if isinstance(s, dict):
                self.add_step(**s)
            elif isinstance(s, (list, tuple)):
                self.add_step(*s)
            elif isinstance(s, PipelineStep):
                self.steps.append(s)
        return self

    def set_context(self, **kwargs):
        self.context.update(kwargs)
        return self

    def run(self, stop_on_error=True, max_workers=1):
        self._log = []
        self.results = OrderedDict()
        start = time.time()

        if max_workers > 1:
            self._run_parallel(max_workers, stop_on_error)
        else:
            self._run_sequential(stop_on_error)

        total = time.time() - start
        self._log.append(f"Pipeline '{self.name}' finished in {total:.2f}s")
        return self

    def _run_sequential(self, stop_on_error):
        for i, step in enumerate(self.steps):
            self._log.append(f"[{i+1}/{len(self.steps)}] Running: {step.name}")
            step.run(self.context)
            if step.status == "done":
                self.results[step.name] = step.result
                if step.result is not None:
                    self.context[step.name] = step.result
                self._log.append(f"  Done in {step.elapsed:.2f}s")
            else:
                self._log.append(f"  FAILED: {step.error}")
                if stop_on_error:
                    break

    def _run_parallel(self, max_workers, stop_on_error):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, step in enumerate(self.steps):
                self._log.append(f"[{i+1}] Submitting: {step.name}")
                futures[executor.submit(step.run, self.context)] = step

            for future in as_completed(futures):
                step = futures[future]
                if step.status == "done":
                    self.results[step.name] = step.result
                    if step.result is not None:
                        self.context[step.name] = step.result
                    self._log.append(f"  {step.name} done in {step.elapsed:.2f}s")
                else:
                    self._log.append(f"  {step.name} FAILED: {step.error}")
                    if stop_on_error:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

    def summary(self):
        lines = [f"Pipeline: {self.name}", f"Steps: {len(self.steps)}", ""]
        for i, step in enumerate(self.steps):
            status_icon = {"done": "+", "failed": "X", "pending": "-", "running": "~"}.get(step.status, "?")
            lines.append(f"  [{status_icon}] {i+1}. {step.name} ({step.elapsed:.2f}s)")
            if step.error:
                lines.append(f"      Error: {step.error}")
        lines.append("")
        lines.extend(self._log)
        return "\n".join(lines)

    def to_dict(self):
        return {
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
            "context_keys": list(self.context.keys()),
            "result_keys": list(self.results.keys()),
        }

    def get_result(self, step_name=None):
        if step_name:
            return self.results.get(step_name)
        return self.results


def build_pipeline_from_steps(step_configs):
    """Build a pipeline from a list of step config dicts.

    Args:
        step_configs: list of dicts with keys: name, func, args, kwargs, description

    Returns:
        Pipeline instance
    """
    p = Pipeline()
    for cfg in step_configs:
        p.add_step(**cfg)
    return p


def run_quick_pipeline(steps, **context):
    """Run a simple pipeline and return the final result.

    Args:
        steps: list of (name, func, args_dict) tuples
        **context: initial context variables

    Returns:
        dict with results per step
    """
    p = Pipeline()
    for name, func, kwargs in steps:
        p.add_step(name, func, kwargs=kwargs)
    p.set_context(**context)
    p.run()
    return p.results


def format_pipeline_report(pipeline):
    """Format a pipeline run as a text report."""
    return pipeline.summary()
