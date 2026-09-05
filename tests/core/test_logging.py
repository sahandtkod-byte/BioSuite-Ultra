"""Logging regression tests (NEW-14).

``ColorFormatter.format`` used ``record.msg`` instead of ``record.getMessage()``,
so every %-style call in the package logged its raw template.  The
security-relevant instance was ``"Failed admin login attempt from %s"``, which
recorded a literal ``%s`` instead of the address - making the audit log useless
for identifying an attacker.  Building the line by hand also discarded any
traceback passed via ``exc_info``.
"""
import logging

import pytest

from biosuite.core.log import ColorFormatter, get_logger


def _record(msg, args=(), level=logging.WARNING, **kwargs):
    return logging.LogRecord(name="biosuite.test", level=level, pathname=__file__,
                             lineno=1, msg=msg, args=args, exc_info=None, **kwargs)


# ── %-style interpolation ───────────────────────────────────────────────────

def test_percent_style_arguments_are_interpolated():
    out = ColorFormatter().format(_record("value is %s", ("42",)))
    assert "value is 42" in out
    assert "%s" not in out


def test_the_failed_login_audit_line_records_the_address():
    """The security case: a literal %s here defeats the audit log."""
    out = ColorFormatter().format(
        _record("Failed admin login attempt from %s", ("203.0.113.7",)))
    assert "203.0.113.7" in out
    assert "%s" not in out


def test_multiple_arguments_are_interpolated():
    out = ColorFormatter().format(_record("%s took %d ms", ("align", 12)))
    assert "align took 12 ms" in out


def test_percent_d_is_interpolated():
    out = ColorFormatter().format(_record("found %d peaks", (7,)))
    assert "found 7 peaks" in out


def test_a_message_without_arguments_is_unchanged():
    out = ColorFormatter().format(_record("plain message"))
    assert "plain message" in out


def test_a_literal_percent_sign_survives_when_there_are_no_args():
    out = ColorFormatter().format(_record("100% complete"))
    assert "100% complete" in out


def test_non_string_messages_are_rendered():
    out = ColorFormatter().format(_record({"a": 1}))
    assert "{'a': 1}" in out


# ── exc_info / stack_info are not dropped ───────────────────────────────────

def test_exception_tracebacks_are_included():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = logging.LogRecord(
            name="biosuite.test", level=logging.ERROR, pathname=__file__,
            lineno=1, msg="failed", args=(), exc_info=sys.exc_info())
    out = ColorFormatter().format(record)
    assert "ValueError" in out
    assert "boom" in out
    assert "Traceback" in out


def test_stack_info_is_included():
    record = _record("with stack")
    record.stack_info = "Stack (most recent call last):\n  fake frame"
    out = ColorFormatter().format(record)
    assert "fake frame" in out


# ── level and formatting basics ─────────────────────────────────────────────

@pytest.mark.parametrize("level,name", [
    (logging.DEBUG, "DEBUG"), (logging.INFO, "INFO"),
    (logging.WARNING, "WARNING"), (logging.ERROR, "ERROR"),
    (logging.CRITICAL, "CRITICAL"),
])
def test_the_level_name_appears(level, name):
    out = ColorFormatter().format(_record("m", level=level))
    assert name in out


def test_the_logger_name_appears():
    out = ColorFormatter().format(_record("m"))
    assert "biosuite.test" in out


# ── end-to-end through a real logger ────────────────────────────────────────

def test_interpolation_works_through_a_real_logger(caplog):
    logger = get_logger("biosuite.e2e")
    with caplog.at_level(logging.WARNING, logger="biosuite.e2e"):
        logger.warning("Failed admin login attempt from %s", "198.51.100.4")
    assert "198.51.100.4" in caplog.text
    assert "%s" not in caplog.text


def test_get_logger_namespaces_under_biosuite():
    assert get_logger("core.sequence").name.startswith("biosuite")
    assert get_logger().name == "biosuite"


def test_get_logger_is_idempotent():
    assert get_logger("biosuite.same") is get_logger("biosuite.same")


def test_the_console_handler_does_not_default_to_debug():
    """A library must not spray DEBUG at every user's console."""
    import biosuite.core.log as log_module
    root = logging.getLogger("biosuite")
    log_module._setup_root()
    console = [h for h in root.handlers
               if isinstance(h, logging.StreamHandler)
               and not isinstance(h, logging.FileHandler)]
    assert console, "expected a console handler"
    assert all(h.level >= logging.INFO for h in console)
