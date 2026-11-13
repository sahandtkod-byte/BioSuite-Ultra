"""
Pipeline builder — chain bioinformatics steps into automated workflows.
Each step is a function + kwargs. Pipelines are serial by default,
with optional parallel branches for independent steps.
"""
import copy
import inspect
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
        # Every other piece of run state is initialised here; traceback was
        # only created by run(), so reading it from a step that never ran
        # raised AttributeError while a failed step exposed it.
        self.traceback = None
        self.elapsed = 0.0
        self.status = "pending"

    def run(self, context=None):
        # Reset any state left over from a previous run of this step,
        # otherwise a step that failed once keeps reporting that error even
        # after a later successful run.
        self.status = "running"
        self.result = None
        self.error = None
        self.traceback = None
        ctx = dict(context) if context else {}
        start = time.time()
        try:
            # Precedence: arguments declared on the step ALWAYS win over
            # values picked up from the shared pipeline context.  The context
            # only fills in what the caller did not specify.
            merged_kwargs = {**ctx, **self.kwargs}
            # Drop context keys the function cannot accept: intermediate
            # results are added to the context under each step's name, so
            # a plain `def step2(x)` would die with "unexpected keyword
            # 'step1'" once step 1 succeeded.  (Functions with **kwargs
            # and builtins keep the full context.)
            try:
                sig = inspect.signature(self.func)
            except (TypeError, ValueError):
                # builtin / C function with no introspectable signature:
                # pass everything and let the call itself decide.
                sig = None
            if sig is not None:
                params = sig.parameters
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in params.values())
                if not accepts_var_kw:
                    # Positional arguments already supplied by the step must
                    # not be duplicated by a same-named context entry.
                    positional = [
                        n for n, p in params.items()
                        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    ][:len(self.args)]
                    merged_kwargs = {
                        k: v for k, v in merged_kwargs.items()
                        if k in params and k not in positional
                        and params[k].kind is not inspect.Parameter.POSITIONAL_ONLY
                    }
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
        self._initial_context = {}
        self.context = {}
        self.results = OrderedDict()
        self._log = []

    def add_step(self, name, func, args=None, kwargs=None, description=""):
        """Append a step.

        Raises:
            ValueError: if *name* duplicates an existing step, which would
                make the two steps overwrite each other in ``results`` and in
                the shared context.
        """
        self._require_unique_name(name)
        step = PipelineStep(name, func, args, kwargs, description)
        self.steps.append(step)
        return self

    def add_steps(self, step_list):
        """Append several steps from dicts, tuples/lists or PipelineStep objects.

        Raises:
            TypeError: for an item that is none of those — silently dropping a
                misspelled step turned a pipeline into a shorter one.
            ValueError: on a duplicate step name, including for pre-built
                ``PipelineStep`` objects (see :meth:`add_step`).
        """
        for s in step_list:
            if isinstance(s, dict):
                self.add_step(**s)
            elif isinstance(s, (list, tuple)):
                self.add_step(*s)
            elif isinstance(s, PipelineStep):
                # Appending directly used to bypass the duplicate-name guard,
                # so two pre-built steps with one name silently overwrote each
                # other's entry in results and in the shared context.
                self._require_unique_name(s.name)
                self.steps.append(s)
            else:
                raise TypeError(
                    f"pipeline step must be a dict, a list/tuple, or a "
                    f"PipelineStep, not {type(s).__name__}: {s!r}")
        return self

    def _require_unique_name(self, name):
        """Reject a step name that is already taken (results key on it)."""
        if any(existing.name == name for existing in self.steps):
            raise ValueError(
                f"duplicate pipeline step name {name!r}; step names must be "
                "unique because results and context entries are keyed by name")

    def set_context(self, **kwargs):
        """Set values available to every step of the *next* run."""
        self._initial_context.update(kwargs)
        self.context.update(kwargs)
        return self

    def run(self, stop_on_error=True, max_workers=1):
        """Execute the pipeline.

        Each call starts from the context supplied via :meth:`set_context`
        (or the constructor); results produced during a run are visible to
        later steps but are discarded afterwards, so two runs of the same
        pipeline object are independent.
        """
        self._log = []
        self.results = OrderedDict()
        # Work on a copy: intermediate results must not leak into the next
        # run of the same pipeline object.
        self.context = copy.deepcopy(self._initial_context)
        for step in self.steps:
            step.status = "pending"
            step.result = None
            step.error = None
            step.traceback = None
            step.elapsed = 0.0
        start = time.time()

        if max_workers > 1:
            self._run_parallel(max_workers, stop_on_error)
        else:
            self._run_sequential(stop_on_error)

        # Step order, not completion order: the results are an OrderedDict and a
        # parallel run used to fill it as threads finished, so two runs of the
        # same pipeline produced two different sequences.
        self.results = OrderedDict(
            (step.name, step.result) for step in self.steps
            if step.name in self.results)

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
                    skipped = len(self.steps) - i - 1
                    if skipped:
                        names = ", ".join(s.name for s in self.steps[i + 1:])
                        self._log.append(
                            f"  Stopped after {step.name}: {skipped} step(s) "
                            f"({names}) were not run")
                    break

    def _run_parallel(self, max_workers, stop_on_error):
        """Run the steps concurrently, each with the context as it stood before.

        Every step is submitted with its own snapshot of the context, so in
        parallel mode steps cannot see each other's output and a pipeline whose
        steps chain must run with the default ``max_workers=1``.  Completed
        results are still recorded on ``self.context`` for whoever reads the
        finished pipeline.  (The previous note said the steps saw the pre-run
        context while the loop kept inserting results into that very dict, so a
        step that happened to start late read another's output — the isolation
        was a timing accident rather than a property.)
        """
        snapshot = dict(self.context)
        stopped_at = None
        collected = set()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, step in enumerate(self.steps):
                self._log.append(f"[{i+1}] Submitting: {step.name}")
                futures[executor.submit(step.run, snapshot)] = step

            for future in as_completed(futures):
                step = futures[future]
                collected.add(step.name)
                self._record(step)
                if stopped_at is None and stop_on_error and step.status == "failed":
                    stopped_at = step.name
                    break

            if stopped_at is not None:
                # Cancelling the queue here rather than while as_completed was
                # still live: on CPython 3.13 a future cancelled mid-iteration
                # never wakes the waiter, and the run hangs (measured: 4 of 8
                # quick steps yielded, then nothing).
                executor.shutdown(wait=False, cancel_futures=True)

        if stopped_at is not None:
            # The with block has waited for every step that had already started.
            # Those are collected now — a step that finished after the break used
            # to be left 'done' with no result and no log line — and the ones
            # that never started are named instead of silently kept 'pending'.
            for step in self.steps:
                if step.name in collected:
                    continue
                if step.status == "pending":
                    step.status = "cancelled"
                    self._log.append(f"  {step.name} cancelled: not started")
                else:
                    self._record(step)
            not_run = [s.name for s in self.steps if s.status == "cancelled"]
            self._log.append(
                f"  Stopped on {stopped_at}"
                + (f": {len(not_run)} step(s) did not run ({', '.join(not_run)})"
                   if not_run else ""))

    def _record(self, step):
        """File one finished step under its own name, in the context too.

        The result is never handed back to a running step: those read only the
        snapshot taken before the run, so the isolation the parallel mode
        promises is a property of this method's timing, not an accident of it.
        """
        if step.status == "done":
            self.results[step.name] = step.result
            if step.result is not None:
                self.context[step.name] = step.result
            self._log.append(f"  {step.name} done in {step.elapsed:.2f}s")
        else:
            self._log.append(f"  {step.name} FAILED: {step.error}")

    def summary(self):
        lines = [f"Pipeline: {self.name}", f"Steps: {len(self.steps)}", ""]
        for i, step in enumerate(self.steps):
            status_icon = {"done": "+", "failed": "X", "pending": "-", "running": "~",
                           "cancelled": "o"}.get(step.status, "?")
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
