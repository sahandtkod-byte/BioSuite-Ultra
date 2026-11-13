"""``retry_on_error`` — the retry budget, the backoff, and the log line it leaves.

Three things were measured on the old version and are pinned against it here: a
mis-shaped parameter was discovered at the worst possible moment — `delay=-1`
replaced the caller's own exception with `ValueError: sleep length must be
non-negative`, and `exceptions=[ValueError]` (a list, not a tuple) replaced it with
`TypeError: catching classes that do not inherit from BaseException is not a
tuple`; `exceptions=()` silently disabled retrying; and a function that only
succeeded on its third attempt said nothing anywhere, although the module already
had a logger.  The budget check itself used to run *after* the loop, reporting a
`ValueError` for a call that had never been attempted.

The sleeps are recorded through a monkeypatch, so no test waits.
"""
import logging

import pytest

from biosuite.core import validators as v


@pytest.fixture
def recorded_sleeps(monkeypatch):
    """Replace the module's sleep so backoff is observable without waiting."""
    calls = []
    monkeypatch.setattr(v.time, "sleep", lambda seconds: calls.append(seconds))
    return calls


def _messages(caplog):
    """Only the records this module's logger emitted — `caplog` also carries every
    other logger's output, and `at_level(logger=...)` filters nothing."""
    records = [r for r in caplog.records if r.name == 'biosuite.core.validators']
    return [r.getMessage() for r in records]


# ── the retry loop ──────────────────────────────────────────────────────────

def test_a_transient_failure_recovers_and_reports_the_backoff_it_used(recorded_sleeps,
                                                                     caplog):
    attempts = []

    @v.retry_on_error(max_retries=3, delay=0.5, exceptions=(KeyError,))
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise KeyError("transient")
        return "recovered"

    with caplog.at_level(logging.WARNING):
        assert flaky() == "recovered"
    assert len(attempts) == 3
    assert recorded_sleeps == [0.5, 1.0]
    assert _messages(caplog) == [
        "flaky failed with KeyError: 'transient'; retry 1 of 3 in 0.5 second(s)",
        "flaky failed with KeyError: 'transient'; retry 2 of 3 in 1.0 second(s)",
    ]
    assert {r.levelno for r in caplog.records if r.name == 'biosuite.core.validators'} == \
        {logging.WARNING}


def test_an_undeclared_exception_escapes_on_the_first_attempt(recorded_sleeps, caplog):
    attempts = []

    @v.retry_on_error(max_retries=3, delay=0.5, exceptions=(KeyError,))
    def wrong_type():
        attempts.append(1)
        raise ValueError("not retryable")

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError) as excinfo:
            wrong_type()
    assert str(excinfo.value) == "not retryable"
    assert len(attempts) == 1
    assert recorded_sleeps == []
    assert _messages(caplog) == []


def test_the_last_exception_survives_after_the_budget_is_spent(recorded_sleeps, caplog):
    attempts = []

    @v.retry_on_error(max_retries=2, delay=1.0)
    def always_fails():
        attempts.append(1)
        raise RuntimeError(f"attempt {len(attempts)}")

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError) as excinfo:
            always_fails()
    assert str(excinfo.value) == "attempt 3"          # 1 call + 2 retries
    assert recorded_sleeps == [1.0, 2.0]
    assert _messages(caplog) == [
        "always_fails failed with RuntimeError: attempt 1; retry 1 of 2 in 1.0 second(s)",
        "always_fails failed with RuntimeError: attempt 2; retry 2 of 2 in 2.0 second(s)",
    ]


def test_zero_retries_means_exactly_one_call_and_no_log(recorded_sleeps, caplog):
    attempts = []

    @v.retry_on_error(max_retries=0, delay=1.0)
    def once():
        attempts.append(1)
        raise RuntimeError("single shot")

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            once()
    assert len(attempts) == 1
    assert recorded_sleeps == []
    assert _messages(caplog) == []


def test_a_single_exception_class_is_read_as_one_class_not_a_character_list(recorded_sleeps):
    attempts = []

    @v.retry_on_error(max_retries=2, delay=0.25, exceptions=ValueError)
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError("again")
        return 'ok'

    assert flaky() == 'ok'
    assert len(attempts) == 3
    assert recorded_sleeps == [0.25, 0.5]


def test_arguments_and_the_result_pass_through_the_retry_decorator(recorded_sleeps):
    seen = []

    @v.retry_on_error(max_retries=1, delay=0.1)
    def add(a, b=0, *, label=None):
        seen.append((a, b, label))
        return a + b

    assert add(20, 22, label='sum') == 42
    assert seen == [(20, 22, 'sum')]
    assert recorded_sleeps == []
    assert add.__name__ == 'add'


# ── the parameters are checked where they are written ───────────────────────

@pytest.mark.parametrize("kwargs, message", [
    pytest.param(dict(max_retries=-1),
                 "retry_on_error max_retries must be a non-negative integer, got -1.",
                 id="negative-budget"),
    pytest.param(dict(max_retries='2'),
                 "retry_on_error max_retries must be a non-negative integer, got '2'.",
                 id="string-budget"),
    pytest.param(dict(max_retries=1.5),
                 "retry_on_error max_retries must be a non-negative integer, got 1.5.",
                 id="float-budget"),
    pytest.param(dict(max_retries=True),
                 "retry_on_error max_retries must be a non-negative integer, got True.",
                 id="bool-budget"),
    pytest.param(dict(delay=-1),
                 "retry_on_error delay must be a finite, non-negative number, got -1.",
                 id="negative-delay"),
    pytest.param(dict(delay=float('nan')),
                 "retry_on_error delay must be a finite, non-negative number, got nan.",
                 id="nan-delay"),
    pytest.param(dict(delay='1'),
                 "retry_on_error delay must be a finite, non-negative number, got '1'.",
                 id="string-delay"),
    pytest.param(dict(exceptions=[ValueError]),
                 "retry_on_error exceptions must be an exception class or a non-empty "
                 "tuple of them, got [<class 'ValueError'>].", id="list-of-classes"),
    pytest.param(dict(exceptions='ValueError'),
                 "retry_on_error exceptions must be an exception class or a non-empty "
                 "tuple of them, got 'ValueError'.", id="string-of-a-name"),
    pytest.param(dict(exceptions=()),
                 "retry_on_error exceptions must be an exception class or a non-empty "
                 "tuple of them, got ().", id="empty-tuple"),
    pytest.param(dict(exceptions=(ValueError, "KeyError")),
                 "retry_on_error exceptions must be an exception class or a non-empty "
                 "tuple of them, got (<class 'ValueError'>, 'KeyError').",
                 id="mixed-tuple"),
])
def test_a_misshapen_retry_parameter_is_refused_when_the_decorator_is_built(kwargs,
                                                                           message):
    """Measured on the old code, `delay=-1` reached `time.sleep` on the first
    failure and the caller's own exception was replaced by
    `ValueError: sleep length must be non-negative`; `exceptions=[ValueError]` was
    replaced by a `TypeError` about the except clause; and `exceptions=()` simply
    stopped retrying, in silence."""
    def build():
        @v.retry_on_error(**kwargs)
        def never_runs():
            raise AssertionError("the function must not be called")
        return never_runs

    with pytest.raises(ValueError) as excinfo:
        build()
    assert str(excinfo.value) == message


def test_a_zero_delay_and_the_default_budget_are_both_accepted(recorded_sleeps):
    """The positive control for the refusals above: the legal spellings still build
    and still retry."""
    calls = []

    @v.retry_on_error(max_retries=1, delay=0)
    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise OSError("first attempt")
        return 'ok'

    assert flaky() == 'ok'
    assert len(calls) == 2
    assert recorded_sleeps == [0]


def test_the_default_budget_and_delay_are_three_retries_at_one_second():
    """Pinned so a change to the defaults is a decision, not a drift."""
    import inspect

    defaults = {name: parameter.default
                for name, parameter in inspect.signature(v.retry_on_error).parameters.items()}
    assert defaults == {'max_retries': 3, 'delay': 1.0, 'exceptions': None}
