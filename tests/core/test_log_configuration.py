"""biosuite.core.log: console-level resolution, optional file handler, helpers.

``_setup_root()`` reads three environment variables and installs a *console*
handler plus an optional *file* handler under ``~/.biosuite/logs``.  Because it
is guarded by the module-global ``_configured`` flag it only ever runs once per
process, so the level-resolution branches need it re-armed with the handler
list restored afterwards.
"""
import logging
import os

import pytest

from biosuite.core import log as logmod


def _console_handlers():
    return [h for h in logging.getLogger('biosuite').handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]


def _file_handlers():
    return [h for h in logging.getLogger('biosuite').handlers
            if isinstance(h, logging.FileHandler)]


@pytest.fixture()
def fresh_setup(tmp_path, monkeypatch):
    """Re-arm ``_setup_root`` and redirect its optional log file into tmp_path."""
    root = logging.getLogger('biosuite')
    saved_handlers, saved_level = list(root.handlers), root.level
    saved_configured = logmod._configured
    root.handlers[:] = []
    logmod._configured = False
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.delenv('BIOSUITE_DEBUG', raising=False)
    monkeypatch.delenv('BIOSUITE_LOG_LEVEL', raising=False)
    try:
        yield tmp_path
    finally:
        for handler in list(root.handlers):
            handler.close()
        root.handlers[:] = saved_handlers
        root.level = saved_level
        logmod._configured = saved_configured


# ── console level resolution ────────────────────────────────────────────────

@pytest.mark.parametrize('flag', ['1', 'true', 'True', '  1  '])
def test_biosuite_debug_env_selects_debug_console(fresh_setup, monkeypatch, flag):
    monkeypatch.setenv('BIOSUITE_DEBUG', flag)
    logmod._setup_root()
    assert [h.level for h in _console_handlers()] == [logging.DEBUG]


@pytest.mark.parametrize('flag', ['0', 'false', '', '  ', 'yes'])
def test_other_debug_flags_leave_the_console_at_info(fresh_setup, monkeypatch, flag):
    monkeypatch.setenv('BIOSUITE_DEBUG', flag)
    logmod._setup_root()
    assert [h.level for h in _console_handlers()] == [logging.INFO]


@pytest.mark.parametrize('name,expected', [
    ('WARNING', logging.WARNING), ('warning', logging.WARNING),
    ('  ERROR  ', logging.ERROR), ('DEBUG', logging.DEBUG),
    ('CRITICAL', logging.CRITICAL),
    # VERBOSE is registered with addLevelName() at import, so the custom level
    # is reachable through BIOSUITE_LOG_LEVEL as well as through BIOSUITE_DEBUG.
    ('verbose', logmod.VERBOSE),
])
def test_named_log_level_env_is_parsed(fresh_setup, monkeypatch, name, expected):
    monkeypatch.setenv('BIOSUITE_LOG_LEVEL', name)
    logmod._setup_root()
    assert [h.level for h in _console_handlers()] == [expected]


@pytest.mark.parametrize('name', ['NOT_A_LEVEL', '30', '5'])
def test_unparseable_log_level_falls_back_to_info(fresh_setup, monkeypatch, name):
    monkeypatch.setenv('BIOSUITE_LOG_LEVEL', name)
    logmod._setup_root()
    assert [h.level for h in _console_handlers()] == [logging.INFO]


def test_no_env_leaves_console_at_info_and_root_verbose(fresh_setup):
    logmod._setup_root()
    assert [h.level for h in _console_handlers()] == [logging.INFO]
    assert logging.getLogger('biosuite').level == logmod.VERBOSE
    assert logmod._configured is True


def test_setup_root_is_idempotent(fresh_setup):
    logmod._setup_root()
    first = list(logging.getLogger('biosuite').handlers)
    logmod._setup_root()
    assert logging.getLogger('biosuite').handlers == first


def test_the_console_handler_uses_the_color_formatter(fresh_setup):
    logmod._setup_root()
    assert all(type(h.formatter) is logmod.ColorFormatter for h in _console_handlers())


# ── optional file handler ───────────────────────────────────────────────────

def test_the_file_handler_lands_inside_the_redirected_home(fresh_setup):
    logmod._setup_root()
    files = [h.baseFilename for h in _file_handlers()]
    assert len(files) == 1
    assert files[0].startswith(str(fresh_setup)), files[0]
    assert os.sep.join(['.biosuite', 'logs']) in files[0]
    assert files[0].endswith('.log')

    logmod.get_logger('biosuite.fileprobe').warning("written to disk %s", "42")
    for handler in _file_handlers():
        handler.flush()
    with open(files[0], encoding='utf-8') as handle:
        text = handle.read()
    assert "written to disk 42" in text
    assert "biosuite.fileprobe" in text


def test_a_file_logger_failure_does_not_stop_console_logging(fresh_setup):
    """``except Exception: pass`` is deliberate: a read-only home must not break
    the library, but it must also not take the console handler with it."""
    (fresh_setup / '.biosuite').write_text('this is a file, not a directory')
    logmod._setup_root()
    assert _file_handlers() == []
    assert [h.level for h in _console_handlers()] == [logging.INFO]


# ── helper functions ────────────────────────────────────────────────────────

def test_log_performance_without_details_omits_the_parentheses(caplog):
    with caplog.at_level(logging.INFO, logger='biosuite.performance'):
        logmod.log_performance('align_reads', 2.5)
    assert [(r.name, r.getMessage()) for r in caplog.records] == [
        ('biosuite.performance', 'align_reads completed in 2.5ms')]


def test_log_performance_with_details_appends_them(caplog):
    with caplog.at_level(logging.INFO, logger='biosuite.performance'):
        logmod.log_performance('align_reads', 125.04, '300 reads')
    assert [r.getMessage() for r in caplog.records] == [
        'align_reads completed in 125.0ms (300 reads)']


@pytest.mark.parametrize('status,level', [
    ('started', logging.DEBUG), ('completed', logging.INFO),
    ('failed', logging.ERROR), ('retrying', logging.INFO), ('', logging.INFO),
])
def test_log_step_status_to_level_with_details(caplog, status, level):
    with caplog.at_level(logging.DEBUG, logger='biosuite'):
        logmod.log_step('align', 'align_reads', status, details='5 sequences')
    assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
        ('biosuite.align', level, 'align_reads: %s (5 sequences)' % status)]


def test_log_step_without_details_has_no_trailing_parentheses(caplog):
    with caplog.at_level(logging.DEBUG, logger='biosuite'):
        logmod.log_step('align', 'align_reads', 'completed')
    assert [r.getMessage() for r in caplog.records] == ['align_reads: completed']


def test_log_error_without_exception_logs_the_plain_message(caplog):
    with caplog.at_level(logging.ERROR, logger='biosuite'):
        logmod.log_error('nothing went wrong here')
    record = caplog.records[-1]
    assert (record.name, record.getMessage(), record.exc_info) == (
        'biosuite', 'nothing went wrong here', None)


def test_log_error_with_an_exception_formats_the_message_and_keeps_the_traceback(caplog):
    """The `%s: %s` form is why the exception is passed at all: without
    `record.getMessage()` the audit line would read literally `%s: %s`."""
    try:
        raise ValueError("boom")
    except ValueError as exc:
        with caplog.at_level(logging.ERROR, logger='biosuite'):
            logmod.log_error('caught error', exc=exc, module='test')
    record = caplog.records[-1]
    assert (record.name, record.levelno, record.getMessage()) == (
        'biosuite.test', logging.ERROR, 'caught error: boom')
    assert record.exc_info[0] is ValueError
    assert 'ValueError: boom' in caplog.text


def test_log_warning_defaults_to_the_root_namespace(caplog):
    with caplog.at_level(logging.WARNING, logger='biosuite'):
        logmod.log_warning('disk nearly full')
    assert [(r.name, r.getMessage()) for r in caplog.records] == [
        ('biosuite', 'disk nearly full')]


# ── get_logger namespaces ───────────────────────────────────────────────────

def test_get_logger_prefixes_only_when_needed():
    assert logmod.get_logger('core.msa').name == 'biosuite.core.msa'
    assert logmod.get_logger('biosuite.already.prefixed').name == 'biosuite.already.prefixed'
    assert logmod.get_logger().name == 'biosuite'
    assert logmod.get_logger(None) is logmod.get_logger('biosuite')


def test_the_verbose_level_name_is_registered():
    assert logging.getLevelName(logmod.VERBOSE) == 'VERBOSE'
    assert logmod.VERBOSE == 5
