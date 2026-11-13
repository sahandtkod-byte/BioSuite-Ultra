"""One failure guard for every interactive plot demo.

Each demo used to carry its own ``except Exception as e: print(f"Error: {e}")``.
That printed the message but threw the traceback away, named no stage, and left
any figure the demo had already created open for the life of the process — a
process-lifetime leak in a long CLI session, measured in cluster 28 for
``biological_plots`` and still present in ``specialized_plots`` until cluster 30.

Two modules needing the same guard is what this module is: ``biological_plots``
owns eighteen demos and ``specialized_plots`` four, and the second copy was the
one that kept the leak.
"""
import contextlib

import matplotlib.pyplot as plt

from ..core.log import get_logger

logger = get_logger(__name__)


@contextlib.contextmanager
def demo_guard(name):
    """Run one interactive demo: report a failure and release its figures.

    The line a user sees stays ``Error: <message>`` — the closed-stdin and
    cancellation suites pin that text — but the traceback is logged, the failing
    demo is named in the log, and a figure the demo created before it died is
    closed on the way out.
    """
    figures_before = set(plt.get_fignums())
    try:
        yield
    except Exception as exc:                # a demo reports, it does not raise
        logger.exception("%s failed", name)
        print(f"Error: {exc}")
    finally:
        for fignum in set(plt.get_fignums()) - figures_before:
            plt.close(plt.figure(fignum))
