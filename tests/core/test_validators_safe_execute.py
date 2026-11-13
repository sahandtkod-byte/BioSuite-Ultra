"""``safe_execute`` — the (result, error) contract and where the failure is reported.

The wrapper is used by callers who want to carry on after an analysis fails, so two
properties matter: it must never hide *the caller's own mistake* (`func` not being
callable was turned into a default value plus a traceback on stderr), and its report
has to reach the place the application actually looks.  The old version printed the
traceback with `traceback.print_exc()`, which lands on stderr — invisible in the GUI
and out of the CLI's log — while the module's logger, which both of them collect,
was never called.
"""
import logging

import pytest

from biosuite.core import validators as v


def records_of_this_module(caplog):
    """The records `validators.py` emitted — `caplog` also carries every other
    logger's output, and `at_level(logger=...)` filters nothing."""
    return [r for r in caplog.records if r.name == 'biosuite.core.validators']


# ── the happy path ──────────────────────────────────────────────────────────

def test_a_successful_call_returns_its_value_and_no_error():
    assert v.safe_execute(lambda a, b=0: a + b, 20, b=22) == (42, None)


def test_arguments_keep_their_positional_and_keyword_places():
    def labelled(label, *rest, marker=None):
        return (label, rest, marker)

    result, error = v.safe_execute(labelled, 'run', 1, 2, marker='m')
    assert result == ('run', (1, 2), 'm')
    assert error is None


def test_a_keyword_named_func_reaches_the_wrapped_function_not_the_wrapper():
    """`func` is positional-only, so a function that takes a `func=` keyword can
    still be executed safely."""
    result, error = v.safe_execute(lambda func=1: func, func=7)
    assert (result, error) == (7, None)


def test_a_call_that_returns_None_is_not_reported_as_a_failure():
    result, error = v.safe_execute(lambda: None)
    assert (result, error) == (None, None)


# ── the failure path ────────────────────────────────────────────────────────

def test_a_failure_returns_the_default_and_the_exception_itself(caplog):
    def boom():
        raise ZeroDivisionError("division by zero")

    with caplog.at_level(logging.ERROR):
        result, error = v.safe_execute(boom, _default="fallback")
    assert result == "fallback"
    assert isinstance(error, ZeroDivisionError)
    assert str(error) == "division by zero"

    logged = records_of_this_module(caplog)
    assert [r.levelno for r in logged] == [logging.ERROR]
    assert logged[0].getMessage() == (
        "boom failed: ZeroDivisionError: division by zero"
    )
    assert logged[0].exc_info[0] is ZeroDivisionError, "the traceback is the point"


def test_the_name_of_a_lambda_is_reported_as_it_was_written(caplog):
    with caplog.at_level(logging.ERROR):
        v.safe_execute(lambda: 1 / 0, _default=None)
    assert records_of_this_module(caplog)[0].getMessage() == (
        "<lambda> failed: ZeroDivisionError: division by zero")


def test_logging_can_be_turned_off_and_the_error_is_still_returned(caplog):
    def quiet():
        raise ValueError("quiet failure")

    with caplog.at_level(logging.DEBUG):
        result, error = v.safe_execute(quiet, _log_errors=False)
    assert (result, error) == (None, error)
    assert str(error) == "quiet failure"
    assert records_of_this_module(caplog) == []


def test_nothing_is_written_to_stderr_because_the_report_goes_to_the_logger(caplog,
                                                                           capsys):
    def boom():
        raise RuntimeError("reported once")

    with caplog.at_level(logging.ERROR):
        v.safe_execute(boom)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "RuntimeError: reported once" in caplog.text


@pytest.mark.parametrize("not_callable, message", [
    pytest.param(None, "safe_execute needs a callable, got NoneType.", id="none"),
    pytest.param("analyze", "safe_execute needs a callable, got str.", id="string"),
    pytest.param(42, "safe_execute needs a callable, got int.", id="number"),
    pytest.param([1, 2], "safe_execute needs a callable, got list.", id="list"),
])
def test_a_missing_function_is_an_error_not_a_failed_analysis(not_callable, message):
    """Measured on the old code: `safe_execute(None)` returned
    `(None, TypeError("'NoneType' object is not callable"))` and printed a
    traceback, so a caller who passed the wrong object got a plausible-looking
    "the analysis failed" answer."""
    with pytest.raises(TypeError) as excinfo:
        v.safe_execute(not_callable)
    assert str(excinfo.value) == message


def test_a_keyboard_interrupt_is_not_collected_as_a_result():
    """The wrapper catches `Exception`; a cancellation is the caller's signal and
    has to leave through the top."""
    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        v.safe_execute(interrupted)


def test_the_default_keyword_is_not_passed_on_to_the_wrapped_function():
    def accepts(**kwargs):
        return kwargs

    result, error = v.safe_execute(accepts, _default='unused')
    assert (result, error) == ({}, None)
