"""The GUI error paths against the central log.

The dialog the user sees is transient, so every genuine failure must also reach
``logger.exception()`` — one record, with the traceback — while the dialog still
appears exactly as before.  Expected validation refusals are not failures and
must not be logged, and no log line may carry private file contents.
"""
import logging
import threading

import pytest

from biosuite.core.workflow.pipeline import Pipeline

PIPELINE_TEXT = "reverse:reverse_complement\ntranslate:translate"
WORKFLOW_LOGGER = "biosuite.gui.tabs.workflow"
WINDOW_LOGGER = "biosuite.gui.main_window"


def _explode():
    raise RuntimeError("background blew up")


class InlineThread:
    """A ``threading.Thread`` stand-in that runs its target the moment it starts.

    ``_run_bg`` hands its work to a daemon thread and marshals the callbacks back
    with ``after()``, which the session fixture runs synchronously; with this stub
    the whole round trip happens inside the call, so the log record and the dialog
    can both be asserted the instant ``_run_bg`` returns.
    """

    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None, *, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


@pytest.fixture(autouse=True)
def inline_worker_threads(monkeypatch):
    monkeypatch.setattr(threading, "Thread", InlineThread)


def _records_from(caplog, logger_name):
    return [record for record in caplog.records if record.name == logger_name]


# ─── Unexpected failures ─────────────────────────────────────────────────────

def test_a_pipeline_failure_is_logged_once_with_its_traceback_and_still_dialogs(
        ui, monkeypatch, caplog):
    def boom(self, *args, **kwargs):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(Pipeline, "run", boom)
    ui.set("pipe_text", PIPELINE_TEXT)

    ui.app._run_pipeline()

    assert ui.messages_of("error") == ["pipeline exploded"], \
        "logging must not replace the dialog the user was already shown"
    records = _records_from(caplog, WORKFLOW_LOGGER)
    assert len(records) == 1, "one failure, one record"
    record = records[0]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None, \
        "the traceback is what makes the entry usable for diagnosis"
    assert "GUI pipeline run failed" in caplog.text
    assert "RuntimeError" in caplog.text and "in boom" in caplog.text
    assert "[message redacted]" in caplog.text
    assert "pipeline exploded" not in caplog.text
    assert not _records_from(caplog, WINDOW_LOGGER), \
        "the tab already logged the failure; the window must not repeat it"


def test_a_background_failure_is_logged_once_and_still_dialogs(ui, caplog):
    ui.app._run_bg(_explode)

    assert ui.messages_of("error") == ["background blew up"]
    records = _records_from(caplog, WINDOW_LOGGER)
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert "Background task failed" in caplog.text
    assert "RuntimeError" in caplog.text and "in _explode" in caplog.text
    assert "background blew up" not in caplog.text
    assert not _records_from(caplog, WORKFLOW_LOGGER)


# ─── Sensitive values ────────────────────────────────────────────────────────

def test_a_corrupt_csv_is_named_by_path_and_its_contents_stay_out_of_the_log(
        ui, tmp_path, caplog):
    """A real unreadable file: the failure is diagnosable, the rows are not logged."""
    heap = tmp_path / "heap.csv"
    heap.write_bytes(b"\xff\xffSECRETGENEDATA\xff\nmore private rows\n")

    ui.open_path = str(heap)
    ui.app._gwas_load()

    errors = ui.messages_of("error")
    assert len(errors) == 1 and errors[0].startswith("Could not load CSV:")
    records = _records_from(caplog, WORKFLOW_LOGGER)
    assert len(records) == 1 and records[0].exc_info is not None
    assert "UnicodeDecodeError" in caplog.text and "in _gwas_load" in caplog.text
    assert str(heap) not in caplog.text
    assert "SECRETGENEDATA" not in caplog.text, \
        "the log names the file; it must never carry the file's contents"


# ─── Expected validation ─────────────────────────────────────────────────────

def test_a_validation_refusal_dialogs_without_being_logged_as_a_failure(
        ui, caplog):
    ui.clear("pipe_text")

    ui.app._run_pipeline()

    assert ui.messages_of("warning") == ["Enter pipeline steps first."]
    assert not _records_from(caplog, WORKFLOW_LOGGER), \
        "a refused run is expected behaviour, not a failure"
    assert not _records_from(caplog, WINDOW_LOGGER)


def test_exception_messages_and_chains_never_reach_gui_logs(ui, monkeypatch, caplog):
    marker = "SYNTHETIC_PRIVATE_TOKEN_53"

    def fail(self):
        try:
            raise ValueError(marker)
        except ValueError as cause:
            raise RuntimeError("private file contents: " + marker) from cause

    monkeypatch.setattr(Pipeline, "run", fail)
    ui.set("pipe_text", PIPELINE_TEXT)
    ui.app._run_pipeline()

    assert ui.messages_of("error") == ["private file contents: " + marker]
    records = _records_from(caplog, WORKFLOW_LOGGER)
    assert len(records) == 1 and records[0].exc_info is not None
    assert "RuntimeError" in caplog.text and "in fail" in caplog.text
    assert marker not in caplog.text
    assert marker not in logging.Formatter().format(records[0])
    assert records[0].exc_info[1].__cause__ is None
    assert records[0].exc_info[1].__context__ is None


def test_a_private_callable_is_not_represented_in_the_background_log(ui, caplog):
    marker = "SYNTHETIC_PRIVATE_CALLABLE_53"

    class PrivateWork:
        def __repr__(self):
            raise AssertionError("logging must not inspect private callable state")

        def __call__(self):
            raise RuntimeError(marker)

    ui.app._run_bg(PrivateWork())

    assert ui.messages_of("error") == [marker]
    records = _records_from(caplog, WINDOW_LOGGER)
    assert len(records) == 1 and records[0].exc_info is not None
    assert "RuntimeError" in caplog.text and "in __call__" in caplog.text
    assert marker not in caplog.text


@pytest.mark.parametrize("handler,target", [
    ("_go_search", "biosuite.core.go_browser.GOBrowser.search"),
    ("_run_epitope", "biosuite.core.epitope.predict_t_cell_epitopes"),
])
def test_user_input_is_not_copied_into_operational_logs(
        ui, monkeypatch, caplog, handler, target):
    marker = "SYNTHETIC_PRIVATE_INPUT_53"

    def fail(*args, **kwargs):
        raise ValueError(marker)

    monkeypatch.setattr(target, fail)
    ui.set("go_search", marker)
    ui.set("epi_hla", marker)
    ui.set("epi_seq", "ACDEFGHIKLMNPQRSTVWY")
    getattr(ui.app, handler)()

    assert ui.messages_of("error") == [marker]
    records = _records_from(caplog, WORKFLOW_LOGGER)
    assert len(records) == 1 and records[0].exc_info is not None
    assert "ValueError" in caplog.text and "in fail" in caplog.text
    assert marker not in caplog.text
