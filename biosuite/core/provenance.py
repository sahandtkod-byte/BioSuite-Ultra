"""
Provenance tracking for reproducible bioinformatics analyses.

Records every analysis step with parameters, timestamps, and results.
Exports as JSON, HTML timeline, or SQLite for sharing and auditing.

Thread safety
-------------
A tracker is routinely shared by worker threads (``core.parallel``,
``workflow.batch``, and FastAPI's synchronous endpoint threadpool), so the
SQLite connection is opened with ``check_same_thread=False`` and every
statement is serialised through an instance lock.  Without this, recording from
any thread other than the creating one raised ``sqlite3.ProgrammingError`` and
the step was silently lost.
"""
import html as _html
import json
import sqlite3
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from functools import wraps


def _json_default(obj):
    """Best-effort JSON coercion for scientific parameter values.

    numpy arrays/scalars, sets and dataclasses are common step parameters and
    used to raise ``TypeError: Object of type ndarray is not JSON
    serializable`` from inside :meth:`ProvenanceTracker.record`, aborting the
    analysis that was merely being *recorded*.
    """
    for attr in ("tolist", "item"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except (TypeError, ValueError):
                # A closed or half-initialised array-like can expose `tolist` and
                # still fail on it; fall through to the type-name placeholder
                # rather than let a recording helper break the analysis.
                break
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return f"<{type(obj).__name__}>"


def dumps_params(params) -> str:
    """Serialise step parameters, never raising on exotic scientific objects.

    ``allow_nan=False`` because Python's default output for a NaN or infinite
    parameter is ``NaN``/``Infinity``, which is not JSON: ``json.loads`` reads it
    back, but every strict consumer of a provenance file (jq, JS, another
    language) rejects it.  Such a parameter is recorded as a string instead.
    """
    try:
        return json.dumps(params, default=_json_default, allow_nan=False)
    except (TypeError, ValueError):
        return json.dumps({k: str(v) for k, v in dict(params).items()})


@dataclass
class AnalysisStep:
    """A single analysis step in the provenance chain."""
    step_id: int = 0
    session_id: str = ""
    timestamp: str = ""
    module: str = ""
    function: str = ""
    params: dict = field(default_factory=dict)
    result_summary: str = ""
    execution_time_ms: int = 0
    engine: str = "builtin"
    status: str = "success"  # success, error, skipped
    error_message: str = ""
    # WS5 correlation ids.  Appended with defaults so existing positional callers
    # are untouched; empty means "not correlated" and is omitted from exports.
    request_id: str = ""
    capability_id: str = ""


@dataclass(frozen=True)
class Correlation:
    """The request a piece of provenance belongs to.

    Bound per execution context (HTTP request or job run) through
    :func:`bind_correlation`; ``record`` reads it so callers that know nothing
    about HTTP still produce correlated audit rows.
    """
    request_id: str = ""
    capability_id: str = ""


_correlation_var: ContextVar = ContextVar("biosuite_provenance_correlation",
                                          default=None)


def _bind_correlation(request_id="", capability_id=""):
    """Bind the current request/job correlation for this execution context.

    Returns a token to hand back to :func:`_reset_correlation`.  Contexts are
    isolated (one per thread/task), so concurrent requests and jobs never see
    each other's ids.
    """
    return _correlation_var.set(
        Correlation(request_id=str(request_id or ""),
                    capability_id=str(capability_id or "")))


def _reset_correlation(token):
    """Restore the correlation that was active before :func:`_bind_correlation`."""
    _correlation_var.reset(token)


def _current_correlation():
    """The correlation active in this context, or an empty one."""
    correlation = _correlation_var.get()
    return correlation if correlation is not None else Correlation()


# Public bridge for the API layer; these live in biosuite/api/provenance.py,
# which re-exports them.  Core-side names stay private because every public
# top-level function of biosuite/core must be a registered catalog capability.
bind_correlation = _bind_correlation
reset_correlation = _reset_correlation
current_correlation = _current_correlation


def _step_payload(step):
    """Serialise a step for JSON export, omitting unset correlation fields.

    Steps recorded before WS5 (or with no correlation bound) exported eleven
    keys; adding two always-null keys would change every legacy export.
    """
    payload = asdict(step)
    if not step.request_id:
        del payload["request_id"]
    if not step.capability_id:
        del payload["capability_id"]
    return payload


# SQLite columns the WS5 migration adds to a database written before correlation
# existed.  Declared nullable TEXT: legacy rows read back as empty strings.
_MIGRATION_COLUMNS = (("request_id", "TEXT"), ("capability_id", "TEXT"))


# `record` accepts only these.  Anything else reaches the reports, where the HTML
# timeline already marks every non-'success' status with a cross while the counters
# count only 'error' — a typo'd status then looks like a failure and is never
# counted as one.
VALID_STATUSES = ('success', 'error', 'skipped')


def _add_column(conn, name, decl):
    """Add one nullable column inside a transaction; roll back on failure."""
    try:
        conn.execute(f"ALTER TABLE steps ADD COLUMN {name} {decl}")
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise


def _check_status(status):
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status must be one of {', '.join(VALID_STATUSES)}, got {status!r}.")
    return status


def _check_execution_time(execution_time_ms):
    """Return *execution_time_ms* as a non-negative number, or refuse.

    The steps table declares the column INTEGER, and SQLite stores a value it
    cannot convert (``'abc'``) as TEXT.  One such row then makes
    ``sum(step.execution_time_ms ...)`` in :meth:`ProvenanceTracker.summary` and
    :meth:`ProvenanceTracker.export_html` raise ``TypeError`` — a single bad
    record is enough to make the whole audit report unreadable.
    """
    if isinstance(execution_time_ms, bool) or not isinstance(
            execution_time_ms, (int, float)):
        raise ValueError(
            f"execution_time_ms must be a number, got "
            f"{type(execution_time_ms).__name__}: {execution_time_ms!r}.")
    value = float(execution_time_ms)
    if value != value or value in (float('inf'), float('-inf')):
        raise ValueError(f"execution_time_ms must be finite, got {execution_time_ms!r}.")
    if value < 0:
        raise ValueError(
            f"execution_time_ms cannot be negative, got {execution_time_ms!r}.")
    return execution_time_ms


class ProvenanceTracker:
    """Track analysis provenance for reproducibility.

    Usage:
        tracker = ProvenanceTracker("my_analysis.db")
        tracker.record("sequence", "gc_content", {"seq": "ATCGATCG"}, "50.0%")
        tracker.export_html("provenance.html")
        tracker.export_json("provenance.json")
    """

    def __init__(self, db_path=None, session_id=None):
        """Initialize provenance tracker.

        Args:
            db_path: path to SQLite database. If None, uses in-memory DB.
            session_id: unique session identifier. Auto-generated if None — a
                random one, because a timestamp to the second made two analyses
                started in the same second share a session, and their steps then
                merged into one indistinguishable audit history.
        """
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        self.db_path = db_path
        self.step_counter = 0
        self._lock = threading.RLock()

        # check_same_thread=False + the instance lock make the tracker usable
        # from worker threads; serialisation is our responsibility.
        self.conn = sqlite3.connect(db_path or ":memory:", check_same_thread=False)

        self._create_tables()
        # The primary key is (session_id, step_id) and step ids restart at 1 for
        # every instance, so a tracker reopened on the same database with the same
        # session id collided with its own history: the first new step raised
        # sqlite3.IntegrityError: UNIQUE constraint failed.  Seeding the counter
        # from the stored maximum makes an append to an existing session work.
        self.step_counter = self._last_step_id()

    def _last_step_id(self):
        with self._lock:
            row = self.conn.execute(
                "SELECT MAX(step_id) FROM steps WHERE session_id=?",
                (self.session_id,)
            ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def _create_tables(self):
        """Create provenance tables."""
        with self._lock:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                step_id INTEGER,
                session_id TEXT,
                timestamp TEXT,
                module TEXT,
                function TEXT,
                params TEXT,
                result_summary TEXT,
                execution_time_ms INTEGER,
                engine TEXT,
                status TEXT,
                error_message TEXT,
                request_id TEXT,
                capability_id TEXT,
                PRIMARY KEY (session_id, step_id)
            )
            """)
            self.conn.commit()
            self._migrate_steps()

    def _migrate_steps(self):
        """Add the correlation columns to a pre-WS5 database, idempotently.

        Guarded by ``PRAGMA table_info`` so repeated opens (and a fresh database,
        which already has the columns) are no-ops.  Each ``ALTER TABLE`` runs in
        its own transaction: a failure rolls back that statement, leaving the
        database readable and migratable on the next open.
        """
        with self._lock:
            existing = {row[1] for row in self.conn.execute(
                "PRAGMA table_info(steps)")}
            for name, decl in _MIGRATION_COLUMNS:
                if name not in existing:
                    _add_column(self.conn, name, decl)

    def record(self, module, function, params=None, result_summary="",
               execution_time_ms=0, engine="builtin", status="success",
               error_message="", request_id=None, capability_id=None):
        """Record an analysis step.

        Args:
            module: module name (e.g., 'sequence', 'alignment').
            function: function name (e.g., 'gc_content').
            params: dict of function parameters.
            result_summary: brief description of result.
            execution_time_ms: execution time in milliseconds.
            engine: 'builtin' or name of external tool.
            status: 'success', 'error', or 'skipped'; anything else raises ValueError.
            error_message: error details if status is 'error'.
            request_id: correlation id for the originating request/job; appended
                after ``error_message`` (existing positional callers are
                unaffected).  When left as None, the id bound by
                :func:`bind_correlation` in the current context is used.
            capability_id: capability the step belongs to; falls back to the
                bound correlation the same way.

        Raises:
            ValueError: if ``status`` is not one of the three documented values, or
                if ``execution_time_ms`` is not a finite, non-negative number.  Both
                are checked before any step id is taken, so a refused call leaves the
                numbering untouched.
        """
        _check_status(status)
        execution_time_ms = _check_execution_time(execution_time_ms)
        correlation = current_correlation()
        if request_id is None:
            request_id = correlation.request_id
        if capability_id is None:
            capability_id = correlation.capability_id

        with self._lock:
            self.step_counter += 1
            step_id = self.step_counter
        step = AnalysisStep(
            step_id=step_id,
            session_id=self.session_id,
            timestamp=datetime.now().isoformat(),
            module=module,
            function=function,
            params=params or {},
            result_summary=str(result_summary),
            execution_time_ms=execution_time_ms,
            engine=engine,
            status=status,
            error_message=error_message,
            request_id=str(request_id),
            capability_id=str(capability_id),
        )

        with self._lock:
            self.conn.execute(
                "INSERT INTO steps (step_id, session_id, timestamp, module,"
                " function, params, result_summary, execution_time_ms, engine,"
                " status, error_message, request_id, capability_id)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (step.step_id, step.session_id, step.timestamp, step.module,
                 step.function, dumps_params(step.params), step.result_summary,
                 step.execution_time_ms, step.engine, step.status,
                 step.error_message, step.request_id, step.capability_id)
            )
            self.conn.commit()
        return step

    def get_steps(self):
        """Retrieve all steps for this session."""
        with self._lock:
            cursor = self.conn.execute(
                "SELECT step_id, session_id, timestamp, module, function,"
                " params, result_summary, execution_time_ms, engine, status,"
                " error_message, request_id, capability_id"
                " FROM steps WHERE session_id=? ORDER BY step_id",
                (self.session_id,)
            )
            rows = cursor.fetchall()
        steps = []
        for row in rows:
            steps.append(AnalysisStep(
                step_id=row[0], session_id=row[1], timestamp=row[2],
                module=row[3], function=row[4],
                params=json.loads(row[5]) if row[5] else {},
                result_summary=row[6], execution_time_ms=row[7],
                engine=row[8], status=row[9], error_message=row[10],
                # Pre-migration rows hold NULL in the new columns.
                request_id=row[11] or "", capability_id=row[12] or ""
            ))
        return steps

    def export_json(self, output_path):
        """Export provenance as machine-readable JSON.

        Args:
            output_path: path to output JSON file.
        """
        steps = self.get_steps()
        data = {
            "session_id": self.session_id,
            "export_time": datetime.now().isoformat(),
            "total_steps": len(steps),
            "steps": [_step_payload(s) for s in steps],
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path

    def export_html(self, output_path):
        """Export provenance as interactive HTML timeline.

        Args:
            output_path: path to output HTML file.
        """
        steps = self.get_steps()
        total_time = sum(s.execution_time_ms for s in steps)
        counts = {status: sum(1 for s in steps if s.status == status)
                  for status in VALID_STATUSES}
        # Reachable only for rows written before record() validated; counting them
        # keeps the header numbers adding up to the total instead of dropping
        # steps that the timeline still marks with a cross.
        n_other = len(steps) - sum(counts.values())
        other_span = (f'    <span><strong>Other:</strong> {n_other}</span>\n'
                      if n_other else '')

        html = f"""<!DOCTYPE html>
<html>
<head>
<title>BioSuite Provenance Report</title>
<style>
body {{ background: #0a0f0a; color: #e0ffe8; font-family: 'Segoe UI', sans-serif; padding: 30px; }}
h1 {{ color: #00ff88; border-bottom: 2px solid #00ff88; padding-bottom: 10px; }}
h2 {{ color: #00cc66; margin-top: 30px; }}
.summary {{ background: #111c11; border: 1px solid #1a3a1a; border-radius: 8px; padding: 20px; margin: 20px 0; }}
.summary span {{ margin-right: 30px; font-size: 18px; }}
.step {{ background: #111c11; border: 1px solid #1a3a1a; border-radius: 8px; padding: 15px; margin: 10px 0; }}
.step-header {{ display: flex; justify-content: space-between; align-items: center; }}
.step-id {{ color: #00ff88; font-weight: bold; font-size: 14px; }}
.step-time {{ color: #6b9b7a; font-size: 12px; }}
.step-func {{ color: #e0ffe8; font-size: 16px; font-weight: bold; margin: 8px 0; }}
.step-params {{ color: #8a7a6a; font-family: Consolas, monospace; font-size: 12px; }}
.step-result {{ color: #00cc6a; margin-top: 5px; }}
.step-error {{ color: #ff4444; margin-top: 5px; }}
.engine {{ display: inline-block; background: #1a3a1a; color: #00ff88; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 10px; }}
.status-ok {{ color: #00ff88; }}
.status-error {{ color: #ff4444; }}
.status-skipped {{ color: #ffcc44; }}
code {{ background: #0d170d; padding: 2px 6px; border-radius: 3px; font-family: Consolas, monospace; }}
</style>
</head>
<body>
<h1>BioSuite Provenance Report</h1>
<div class="summary">
    <span><strong>Session:</strong> {_html.escape(self.session_id)}</span>
    <span><strong>Steps:</strong> {len(steps)}</span>
    <span><strong>Total time:</strong> {total_time}ms</span>
    <span><strong>Success:</strong> {counts['success']}</span>
    <span><strong>Errors:</strong> {counts['error']}</span>
    <span><strong>Skipped:</strong> {counts['skipped']}</span>
{other_span}</div>
<h2>Analysis Timeline</h2>
"""
        for step in steps:
            # A 'skipped' step used to render exactly like a failure (cross, red)
            # while appearing in neither counter, so the header numbers did not add
            # up and a deliberate skip read as a crash.
            status_class, status_icon = {
                "success": ("status-ok", "✓"),
                "error": ("status-error", "✗"),
                "skipped": ("status-skipped", "⊘"),
                # Rows written before record() validated may hold anything, so an
                # unknown status keeps rendering as a failure instead of raising.
            }.get(step.status, ("status-error", "✗"))
            params_str = _html.escape(
                dumps_params(step.params) if step.params else "{}")

            html += f"""
<div class="step">
    <div class="step-header">
        <span class="step-id">#{step.step_id} <span class="{status_class}">{status_icon}</span></span>
        <span class="step-time">{step.timestamp} ({step.execution_time_ms}ms)</span>
    </div>
    <div class="step-func">{_html.escape(step.module)}.{_html.escape(step.function)} <span class="engine">{_html.escape(step.engine)}</span></div>
    <div class="step-params">Params: <code>{params_str}</code></div>
    <div class="step-result">Result: {_html.escape(step.result_summary)}</div>
"""
            if step.error_message:
                html += f'    <div class="step-error">Error: {_html.escape(step.error_message)}</div>\n'
            html += "</div>\n"

        html += """
<h2>How to Reproduce</h2>
<div class="step">
    <div class="step-params">
To reproduce this analysis, install BioSuite and replay these steps:<br><br>
<code>pip install biosuite</code><br><br>
Then call each function with the parameters shown above.
    </div>
</div>
</body>
</html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return output_path

    def summary(self):
        """Print a text summary of the provenance."""
        steps = self.get_steps()
        total_time = sum(s.execution_time_ms for s in steps)
        counts = {status: sum(1 for s in steps if s.status == status)
                  for status in VALID_STATUSES}
        lines = [
            "=== Provenance Summary ===",
            f"Session: {self.session_id}",
            f"Total steps: {len(steps)}",
            f"Total execution time: {total_time}ms",
            f"Success: {counts['success']}",
            f"Errors: {counts['error']}",
            f"Skipped: {counts['skipped']}",
        ]
        n_other = len(steps) - sum(counts.values())
        if n_other:
            # Legacy rows (recorded before status was validated) would otherwise
            # leave the counters short of the total.
            lines.append(f"Other: {n_other}")
        lines += [
            "",
            f"{'#':<4} {'Module':<15} {'Function':<25} {'Engine':<10} {'Time':>8} {'Status'}",
            "-" * 80,
        ]
        for step in steps:
            lines.append(
                f"{step.step_id:<4} {step.module:<15} {step.function:<25} "
                f"{step.engine:<10} {step.execution_time_ms:>6}ms  {step.status}"
            )
        return "\n".join(lines)

    def close(self):
        """Close the database connection."""
        with self._lock:
            self.conn.close()


# ── Decorator for automatic recording ────────────────────────────────────────

def _brief(value, limit):
    """``str(value)`` cut to *limit* characters, with the cut made visible.

    The audit file advertises "To reproduce this analysis, replay these steps", so a
    silently truncated parameter is worse than a marked one: the reader cannot tell a
    100-character sequence from a 500-character one.
    """
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."


def tracked(tracker=None):
    """Decorator that automatically records function calls to a ProvenanceTracker.

    Usage:
        tracker = ProvenanceTracker("analysis.db")

        @tracked(tracker)          # called with the tracker
        def my_analysis(seq):
            return gc_content(seq)

        result = my_analysis("ATCGATCG")  # Automatically recorded

    Raises:
        TypeError: if a non-tracker (typically the decorated function itself, from a
            bare ``@tracked``) is passed.  Bare use used to "work" by accident: the
            name was bound to the inner ``decorator``, so calling it returned a
            function object instead of running the analysis.
    """
    if tracker is not None and not hasattr(tracker, "record"):
        raise TypeError(
            "tracked() needs a ProvenanceTracker: write @tracked(tracker), not a bare "
            f"@tracked (got {tracker!r}).")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            module_name = func.__module__.split('.')[-1] if func.__module__ else "unknown"
            func_name = func.__name__

            params = {}
            for i, arg in enumerate(args):
                params[f"arg{i}"] = _brief(arg, 100)
            for k, v in kwargs.items():
                params[k] = _brief(v, 100)

            try:
                result = func(*args, **kwargs)
                elapsed_ms = int((time.time() - start_time) * 1000)
                # `if result` made 0, '' and [] indistinguishable from a missing
                # return: the audit line claimed "None" for a function that had
                # deliberately answered with zero.
                result_summary = "None" if result is None else _brief(result, 200)

                if tracker:
                    tracker.record(
                        module=module_name,
                        function=func_name,
                        params=params,
                        result_summary=result_summary,
                        execution_time_ms=elapsed_ms,
                        status="success"
                    )
                return result
            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                if tracker:
                    tracker.record(
                        module=module_name,
                        function=func_name,
                        params=params,
                        result_summary="",
                        execution_time_ms=elapsed_ms,
                        status="error",
                        error_message=str(e)
                    )
                raise
        return wrapper
    return decorator


# ── Global tracker (optional) ────────────────────────────────────────────────

_default_tracker = None
# The lazy singleton is created from FastAPI's endpoint threads and from
# core.parallel workers, so it needs its own lock: measured with 8 threads racing on
# a cold import, the unguarded version produced 6, 7 and 8 distinct trackers over
# three attempts, and every step recorded through a loser of that race went into a
# tracker no one could reach again.
_tracker_lock = threading.Lock()


def get_global_tracker():
    """Get or create the global provenance tracker (thread-safe)."""
    global _default_tracker
    with _tracker_lock:
        if _default_tracker is None:
            _default_tracker = ProvenanceTracker()
        return _default_tracker


def set_global_tracker(tracker):
    """Set the global provenance tracker (pass None to drop it)."""
    global _default_tracker
    with _tracker_lock:
        _default_tracker = tracker

def record_step(module, function, params=None, result_summary="",
                execution_time_ms=0, engine="builtin"):
    """Record a step to the global tracker."""
    tracker = get_global_tracker()
    return tracker.record(module, function, params, result_summary,
                         execution_time_ms, engine)
