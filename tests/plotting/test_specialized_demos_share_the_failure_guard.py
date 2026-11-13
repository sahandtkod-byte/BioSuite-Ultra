"""The four specialized demos run inside the guard the other eighteen already use.

``specialized_plots`` kept its own copies of ``except Exception as e: print(...)``:
no logger, no demo name, and a figure opened before a failure stayed open for the
life of the process.  Cluster 28 removed exactly that from ``biological_plots``;
this is the same object, imported, so the two modules cannot drift apart again.
"""
import logging

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import biological_plots as bp
from biosuite.plotting import specialized_plots as sp
from biosuite.plotting._demo_guard import demo_guard


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_both_modules_import_the_same_guard():
    assert sp._demo is bp._demo is demo_guard


def test_the_guard_names_the_demo_in_the_log(caplog):
    with caplog.at_level(logging.ERROR, logger="biosuite.plotting._demo_guard"):
        with demo_guard("unit_demo"):
            raise RuntimeError("boom")

    records = [r for r in caplog.records if r.name == "biosuite.plotting._demo_guard"]
    assert [r.getMessage() for r in records] == ["unit_demo failed"], \
        [r.getMessage() for r in records]
    assert records[0].exc_info is not None, "the traceback was thrown away"


def test_the_user_still_reads_the_same_line(capsys):
    with demo_guard("unit_demo"):
        raise RuntimeError("boom")

    assert capsys.readouterr().out == "Error: boom\n"


def test_a_figure_opened_before_the_failure_is_closed(capsys):
    with demo_guard("unit_demo"):
        plt.subplots()
        raise RuntimeError("boom")

    assert plt.get_fignums() == [], "the half-drawn figure stayed registered"
    capsys.readouterr()


def test_a_figure_that_already_existed_is_left_alone(capsys):
    """The guard closes what the demo opened, not what it found."""
    kept = plt.figure()

    with demo_guard("unit_demo"):
        plt.subplots()
        raise RuntimeError("boom")

    assert plt.get_fignums() == [kept.number], plt.get_fignums()
    capsys.readouterr()


def test_a_demo_failure_reaches_the_guard_with_its_own_name(run_demo, caplog):
    """The sankey demo's malformed input, end to end: print, log, no figure."""
    with caplog.at_level(logging.ERROR, logger="biosuite.plotting._demo_guard"):
        report = run_demo(sp.sankey_diagram, answers={
            "use default data": "n",
            "node labels": "A,B",
            "source indices": "not-a-number",
        })

    assert "Error:" in report["stdout"]
    assert report["images"] == []
    assert report["figures_open"] == 0
    assert "sankey_diagram failed" in [r.getMessage() for r in caplog.records], \
        [r.getMessage() for r in caplog.records]
