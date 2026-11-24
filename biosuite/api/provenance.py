"""WS5 provenance bridge: request/job correlation into the core audit trail.

The middleware and the job executor bind a
:class:`~biosuite.core.provenance.Correlation` (the request id resolved by the
``X-Request-ID`` middleware plus the capability being run) for the duration of
their work.  Every step the core ``ProvenanceTracker`` records inside that
window then carries the submitter's ``request_id``/``capability_id`` without
any caller passing ids through and without changing the core's public
signatures.

WS8 addition: the job executor also records one bounded provenance step per
finished job.  ``jobs.service`` hands the executor a recorder callback backed
by the process-wide tracker (``_record_job_provenance`` here in the API
layer); the executor never touches the tracker directly, so
``biosuite.api`` depends on ``biosuite.core`` and not the other way round.

Nothing here stores or logs API keys or other credentials: the only values
propagated are correlation ids and capability names.
"""
from biosuite.core.provenance import (
    Correlation,
    bind_correlation,
    current_correlation,
    reset_correlation,
)

__all__ = ['Correlation', 'bind_correlation', 'current_correlation',
           'reset_correlation', 'run_with_correlation']


def run_with_correlation(request_id, capability_id, function, *args, **kwargs):
    """Run ``function`` with a bound correlation, restoring the old one after.

    The job worker uses this so a job executes under its submitter's
    correlation (with the job's own ``capability_id``), isolated from whatever
    else the worker process is doing.
    """
    token = bind_correlation(request_id, capability_id)
    try:
        return function(*args, **kwargs)
    finally:
        reset_correlation(token)
