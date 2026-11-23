"""WS5: SQLite migration of the provenance steps table.

Covers: legacy databases gain nullable ``request_id``/``capability_id`` columns
without losing rows; reopening is idempotent; a failing migration rolls back
and leaves the database usable; legacy JSON/HTML exports are unchanged while
the fields are unset.
"""
import json
import sqlite3

import pytest

from biosuite.core.provenance import ProvenanceTracker

LEGACY_COLUMNS = ('step_id', 'session_id', 'timestamp', 'module', 'function',
                  'params', 'result_summary', 'execution_time_ms', 'engine',
                  'status', 'error_message')

LEGACY_ROW = (7, 'legacy-session', '2026-01-01T00:00:00', 'sequence',
              'gc_content', '{"sequence": "ATGC"}', 'gc_percent=50.0', 3,
              'builtin', 'success', '')


def make_legacy_db(path):
    """Create a pre-WS5 database with one recorded step, exactly as WS4 wrote it."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE steps (
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
    conn.execute("INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?)", LEGACY_ROW)
    conn.commit()
    conn.close()


def column_names(path):
    conn = sqlite3.connect(str(path))
    names = [row[1] for row in conn.execute("PRAGMA table_info(steps)")]
    conn.close()
    return names


def test_legacy_database_migrates_and_rows_survive(tmp_path):
    path = tmp_path / 'legacy.db'
    make_legacy_db(path)
    tracker = ProvenanceTracker(db_path=str(path), session_id='legacy-session')
    try:
        assert set(column_names(path)) >= set(LEGACY_COLUMNS) | {'request_id', 'capability_id'}
        steps = tracker.get_steps()
        assert len(steps) == 1
        step = steps[0]
        assert (step.step_id, step.module, step.function) == (7, 'sequence', 'gc_content')
        assert step.status == 'success'
        assert step.params == {'sequence': 'ATGC'}
        # A migrated row has no correlation: the fields read back empty, never None.
        assert step.request_id == '' and step.capability_id == ''
    finally:
        tracker.close()


def test_repeated_migration_and_reopen_is_idempotent(tmp_path):
    path = tmp_path / 'legacy.db'
    make_legacy_db(path)
    for _ in range(3):
        tracker = ProvenanceTracker(db_path=str(path), session_id='legacy-session')
        tracker.close()
    conn = sqlite3.connect(str(path))
    names = [row[1] for row in conn.execute("PRAGMA table_info(steps)")]
    count = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
    conn.close()
    assert names.count('request_id') == 1 and names.count('capability_id') == 1
    assert count == 1
    tracker = ProvenanceTracker(db_path=str(path), session_id='legacy-session')
    try:
        assert [s.step_id for s in tracker.get_steps()] == [7]
    finally:
        tracker.close()


def test_failed_migration_rolls_back_and_leaves_db_usable(tmp_path):
    path = tmp_path / 'legacy.db'
    make_legacy_db(path)
    conn = sqlite3.connect(str(path))
    try:
        from biosuite.core import provenance as provenance_module
        # First column migrates for real; a duplicate add fails inside SQLite
        # and must roll back without corrupting the connection.
        provenance_module._add_column(conn, 'request_id', 'TEXT')
        with pytest.raises(sqlite3.OperationalError, match='duplicate column'):
            provenance_module._add_column(conn, 'request_id', 'TEXT')
        names = [row[1] for row in conn.execute("PRAGMA table_info(steps)")]
        assert names.count('request_id') == 1
        assert 'capability_id' not in names
        count = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
        assert count == 1
    finally:
        conn.close()
    # A later open migrates the remaining column cleanly.
    tracker = ProvenanceTracker(db_path=str(path), session_id='legacy-session')
    try:
        names = [row[1] for row in tracker.conn.execute("PRAGMA table_info(steps)")]
        assert 'capability_id' in names and names.count('request_id') == 1
        assert [s.step_id for s in tracker.get_steps()] == [7]
    finally:
        tracker.close()


def test_legacy_exports_omit_unset_fields(tmp_path):
    tracker = ProvenanceTracker(db_path=str(tmp_path / 'fresh.db'),
                                session_id='legacy-session')
    try:
        tracker.record('sequence', 'gc_content', {'sequence': 'ATGC'}, 'ok', 1)
        json_path = tracker.export_json(str(tmp_path / 'p.json'))
        payload = json.loads(open(json_path, encoding='utf-8').read())
        [step] = payload['steps']
        assert 'request_id' not in step and 'capability_id' not in step
        assert set(step) == {'step_id', 'session_id', 'timestamp', 'module',
                             'function', 'params', 'result_summary',
                             'execution_time_ms', 'engine', 'status',
                             'error_message'}
        html = open(tracker.export_html(str(tmp_path / 'p.html')),
                    encoding='utf-8').read()
        assert 'request_id' not in html and 'capability_id' not in html
    finally:
        tracker.close()


def test_new_fields_exported_when_set(tmp_path):
    tracker = ProvenanceTracker(db_path=str(tmp_path / 'fresh.db'),
                                session_id='s')
    try:
        tracker.record('sequence', 'gc_content', {'sequence': 'ATGC'}, 'ok', 1,
                       request_id='req-abc', capability_id='sequence.gc_content')
        json_path = tracker.export_json(str(tmp_path / 'p.json'))
        payload = json.loads(open(json_path, encoding='utf-8').read())
        [step] = payload['steps']
        assert step['request_id'] == 'req-abc'
        assert step['capability_id'] == 'sequence.gc_content'
    finally:
        tracker.close()


def test_explicit_fields_override_the_bound_context(tmp_path):
    from biosuite.core.provenance import bind_correlation, current_correlation, reset_correlation
    tracker = ProvenanceTracker(db_path=str(tmp_path / 'fresh.db'),
                                session_id='s')
    try:
        token = bind_correlation('bound-request', 'bound.capability')
        try:
            tracker.record('m', 'f')
            assert tracker.get_steps()[0].request_id == 'bound-request'
            tracker.record('m', 'g', request_id='explicit', capability_id='explicit.cap')
            explicit = tracker.get_steps()[1]
            assert (explicit.request_id, explicit.capability_id) == ('explicit', 'explicit.cap')
            assert current_correlation().request_id == 'bound-request'
        finally:
            reset_correlation(token)
    finally:
        tracker.close()
