"""
Provenance tracking for reproducible bioinformatics analyses.

Records every analysis step with parameters, timestamps, and results.
Exports as JSON, HTML timeline, or SQLite for sharing and auditing.
"""
import os
import json
import sqlite3
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from functools import wraps


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
            session_id: unique session identifier. Auto-generated if None.
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        self.db_path = db_path
        self.step_counter = 0

        if db_path:
            self.conn = sqlite3.connect(db_path)
        else:
            self.conn = sqlite3.connect(":memory:")

        self._create_tables()

    def _create_tables(self):
        """Create provenance tables."""
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
                PRIMARY KEY (session_id, step_id)
            )
        """)
        self.conn.commit()

    def record(self, module, function, params=None, result_summary="",
               execution_time_ms=0, engine="builtin", status="success",
               error_message=""):
        """Record an analysis step.

        Args:
            module: module name (e.g., 'sequence', 'alignment').
            function: function name (e.g., 'gc_content').
            params: dict of function parameters.
            result_summary: brief description of result.
            execution_time_ms: execution time in milliseconds.
            engine: 'builtin' or name of external tool.
            status: 'success', 'error', or 'skipped'.
            error_message: error details if status is 'error'.
        """
        self.step_counter += 1
        step = AnalysisStep(
            step_id=self.step_counter,
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
        )

        self.conn.execute(
            "INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (step.step_id, step.session_id, step.timestamp, step.module,
             step.function, json.dumps(step.params), step.result_summary,
             step.execution_time_ms, step.engine, step.status, step.error_message)
        )
        self.conn.commit()
        return step

    def get_steps(self):
        """Retrieve all steps for this session."""
        cursor = self.conn.execute(
            "SELECT * FROM steps WHERE session_id=? ORDER BY step_id",
            (self.session_id,)
        )
        steps = []
        for row in cursor.fetchall():
            steps.append(AnalysisStep(
                step_id=row[0], session_id=row[1], timestamp=row[2],
                module=row[3], function=row[4],
                params=json.loads(row[5]) if row[5] else {},
                result_summary=row[6], execution_time_ms=row[7],
                engine=row[8], status=row[9], error_message=row[10]
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
            "steps": [asdict(s) for s in steps],
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
        n_success = sum(1 for s in steps if s.status == 'success')
        n_error = sum(1 for s in steps if s.status == 'error')

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
code {{ background: #0d170d; padding: 2px 6px; border-radius: 3px; font-family: Consolas, monospace; }}
</style>
</head>
<body>
<h1>BioSuite Provenance Report</h1>
<div class="summary">
    <span><strong>Session:</strong> {self.session_id}</span>
    <span><strong>Steps:</strong> {len(steps)}</span>
    <span><strong>Total time:</strong> {total_time}ms</span>
    <span><strong>Success:</strong> {n_success}</span>
    <span><strong>Errors:</strong> {n_error}</span>
</div>
<h2>Analysis Timeline</h2>
"""
        for step in steps:
            status_class = "status-ok" if step.status == "success" else "status-error"
            status_icon = "✓" if step.status == "success" else "✗"
            params_str = json.dumps(step.params, indent=None) if step.params else "{}"

            html += f"""
<div class="step">
    <div class="step-header">
        <span class="step-id">#{step.step_id} <span class="{status_class}">{status_icon}</span></span>
        <span class="step-time">{step.timestamp} ({step.execution_time_ms}ms)</span>
    </div>
    <div class="step-func">{step.module}.{step.function} <span class="engine">{step.engine}</span></div>
    <div class="step-params">Params: <code>{params_str}</code></div>
    <div class="step-result">Result: {step.result_summary}</div>
"""
            if step.error_message:
                html += f'    <div class="step-error">Error: {step.error_message}</div>\n'
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
        lines = [
            f"=== Provenance Summary ===",
            f"Session: {self.session_id}",
            f"Total steps: {len(steps)}",
            f"Total execution time: {total_time}ms",
            f"Success: {sum(1 for s in steps if s.status == 'success')}",
            f"Errors: {sum(1 for s in steps if s.status == 'error')}",
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
        self.conn.close()


# ── Decorator for automatic recording ────────────────────────────────────────

def tracked(tracker=None):
    """Decorator that automatically records function calls to a ProvenanceTracker.

    Usage:
        tracker = ProvenanceTracker("analysis.db")

        @tracked(tracker)
        def my_analysis(seq):
            return gc_content(seq)

        result = my_analysis("ATCGATCG")  # Automatically recorded
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            module_name = func.__module__.split('.')[-1] if func.__module__ else "unknown"
            func_name = func.__name__

            params = {}
            for i, arg in enumerate(args):
                params[f"arg{i}"] = str(arg)[:100]  # Truncate long values
            for k, v in kwargs.items():
                params[k] = str(v)[:100]

            try:
                result = func(*args, **kwargs)
                elapsed_ms = int((time.time() - start_time) * 1000)
                result_summary = str(result)[:200] if result else "None"

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

def get_global_tracker():
    """Get or create the global provenance tracker."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = ProvenanceTracker()
    return _default_tracker

def set_global_tracker(tracker):
    """Set the global provenance tracker."""
    global _default_tracker
    _default_tracker = tracker

def record_step(module, function, params=None, result_summary="",
                execution_time_ms=0, engine="builtin"):
    """Record a step to the global tracker."""
    tracker = get_global_tracker()
    return tracker.record(module, function, params, result_summary,
                         execution_time_ms, engine)
