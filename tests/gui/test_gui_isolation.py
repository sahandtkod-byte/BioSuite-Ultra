"""The session-scoped app returns to its built state between tests.

One BioSuiteApp serves the whole GUI session, so whatever a test leaves behind
— typed text, an open pyplot figure, a stray window, a replaced widget, a
runtime attribute — arrives in the next test.  The autouse ``fresh_app_state``
teardown in conftest runs ``restore_built_state`` after every test, failed ones
included; these tests pin that guarantee down deterministically, without sleeps
and without a deliberately failing test in the suite.  Every test here also
passes when run alone, because the restored state equals the built state.
"""
import os
import pathlib
import subprocess
import sys
import tkinter as tk

import matplotlib.pyplot as plt
import pytest

BUILT_PIPELINE_TEXT = "reverse:reverse_complement\ntranslate:translate"


# ─── Sequential session ──────────────────────────────────────────────────────

def test_a_pipeline_run_leaves_output_behind_in_the_shared_session(ui):
    ui.set("pipe_text", "reverse:reverse_complement")
    ui.app._run_pipeline()

    assert ui.text("pipe_results").strip() != "", \
        "the run really did leave state behind for the next test"


def test_the_next_test_starts_from_the_built_state(ui):
    assert ui.text("pipe_results").strip() == "", \
        "one test's results must not still be on screen in the next"
    assert ui.text("pipe_text").rstrip("\n") == BUILT_PIPELINE_TEXT, \
        "typed steps must not survive into the next test's box"


def test_figures_a_test_leaves_open_do_not_reach_the_next_one(ui):
    own = plt.figure("isolation-probe")

    assert plt.get_fignums() == [own.number], \
        "the teardown closed every figure an earlier test left open"


def test_a_test_may_leave_figures_open_without_costing_the_next_one(ui):
    plt.figure("left open on purpose")
    plt.figure("and a second one")

    assert len(plt.get_fignums()) == 2


# ─── Cleanup helpers ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("parent_name", [None, "seq_text"])
def test_a_stray_window_is_destroyed_and_the_app_survives_the_sweep(ui, parent_name):
    parent = getattr(ui.app, parent_name) if parent_name else ui.app
    stray = tk.Toplevel(parent)
    stray.withdraw()
    assert stray.winfo_exists()

    ui.reset()

    assert not stray.winfo_exists(), "no window a test opened may outlive it"
    assert ui.app.winfo_exists(), "the sweep must spare the application itself"


def test_a_widget_a_test_replaced_with_a_stand_in_is_given_back(ui):
    original = ui.app.seq_text

    class _Fake:
        def get(self, *args, **kwargs):
            return "not a widget"

    ui.app.seq_text = _Fake()
    ui.reset()

    assert ui.app.seq_text is original, \
        "the next test must not meet a stand-in where a real box belongs"


def test_runtime_state_a_test_created_is_taken_back_out(ui):
    ui.app._gwas_data = object()
    ui.app._last_digest_result = object()
    ui.app._svcnv_data = object()
    ui.app._resize_scheduled = True
    ui.app._plot_history.append({"fig": None, "name": "leftover"})

    ui.reset()

    assert not hasattr(ui.app, "_gwas_data"), \
        "the app did not have this attribute when it was built"
    assert not hasattr(ui.app, "_resize_scheduled")
    assert not hasattr(ui.app, "_last_digest_result")
    assert not hasattr(ui.app, "_svcnv_data")
    assert ui.app._plot_history == []
    assert ui.app._plot_history_index == -1


def test_the_session_ends_on_the_tab_and_status_it_built_with(ui):
    built_status = ui.status()
    ui.app._show_frame("help")
    ui.app._set_status("GWAS: analysed 2000 SNPs")
    assert ui.app._current_frame == "help"

    ui.reset()

    assert ui.app._current_frame == "plots"
    assert ui.status() == built_status, \
        "the status line returns to what the app started the test with"


def test_state_survives_a_test_body_that_raises_midway(ui, app_reset):
    """The same restore runs in teardown after a failed test; exercised directly
    here, because a deliberately failing test cannot live in the suite."""
    ui.set("pipe_text", "typed, then the test blew up")
    plt.figure()
    with pytest.raises(RuntimeError, match="died halfway"):
        raise RuntimeError("died halfway")

    app_reset()

    assert ui.text("pipe_text").rstrip("\n") == BUILT_PIPELINE_TEXT
    assert plt.get_fignums() == []


def test_gallery_selection_and_results_return_to_the_full_catalogue(ui):
    from biosuite.gui.themes import PLOT_CATEGORIES

    ui.app._select_category("Mathematical")
    ui.set("plot_search", "sine")
    ui.app._apply_plot_search()
    expected_filtered = [(name, plot_id, "Mathematical")
                         for name, plot_id in PLOT_CATEGORIES["Mathematical"]
                         if "sine" in name.lower()]
    assert ui.app._plot_items == expected_filtered
    assert 0 < len(expected_filtered) < len(PLOT_CATEGORIES["Mathematical"])

    ui.reset()

    expected = sum(len(plots) for plots in PLOT_CATEGORIES.values())
    assert ui.app._selected_cat == "All"
    assert len(ui.app._plot_items) == expected
    assert ui.text("plot_search") == ""
    assert ui.text("plot_count_label") == f"{expected} plots available"


def test_actual_fixture_cleanup_runs_after_a_failed_test(tmp_path):
    repo = pathlib.Path(__file__).resolve().parents[2]
    child_test = tmp_path / "test_failed_session.py"
    child_test.write_text(
        "import tkinter as tk\n"
        "import matplotlib.pyplot as plt\n"
        "def test_fail_midway(ui):\n"
        "    ui.set('pipe_text', 'leftover steps')\n"
        "    ui.set('pipe_results', 'leftover output')\n"
        "    ui.app._gwas_data = object()\n"
        "    ui.app._show_frame('help')\n"
        "    window = tk.Toplevel(ui.app.seq_text)\n"
        "    window.withdraw()\n"
        "    figure = plt.figure()\n"
        "    assert window.winfo_exists() and plt.fignum_exists(figure.number)\n"
        "    raise RuntimeError('intentional isolated failure')\n"
        "def test_next_starts_clean(ui):\n"
        "    assert ui.text('pipe_text').strip() == "
        "'reverse:reverse_complement\\ntranslate:translate'\n"
        "    assert ui.text('pipe_results').strip() == ''\n"
        "    assert plt.get_fignums() == []\n"
        "    assert not hasattr(ui.app, '_gwas_data')\n"
        "    assert ui.app._current_frame == 'plots'\n"
        "    children = list(ui.app.winfo_children())\n"
        "    while children:\n"
        "        child = children.pop()\n"
        "        assert not isinstance(child, tk.Toplevel)\n"
        "        children.extend(child.winfo_children())\n",
        encoding="utf-8")
    runner = (
        "import importlib.util, sys, pytest\n"
        "spec = importlib.util.spec_from_file_location('gui_state_probe', sys.argv[1])\n"
        "plugin = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = plugin\n"
        "spec.loader.exec_module(plugin)\n"
        "raise SystemExit(pytest.main([sys.argv[2], '-q', '-p', 'no:cacheprovider', "
        "'--rootdir', sys.argv[3]], plugins=[plugin]))\n")
    env = dict(os.environ,
               PYTHONPATH=str(repo) + os.pathsep + str(repo / "audit_output/.probe_env"),
               PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               BIOSUITE_CONFIG_DIR=str(tmp_path / "config"),
               HOME=str(tmp_path), USERPROFILE=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c", runner, str(repo / "tests/gui/conftest.py"),
         str(child_test), str(repo)], cwd=repo, env=env, capture_output=True,
        text=True, timeout=120)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed, 1 passed" in result.stdout, result.stdout + result.stderr
    assert "intentional isolated failure" in result.stdout
