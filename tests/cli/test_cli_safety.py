"""Regression tests for CLI safety (BSU-003, BSU-014) and exit codes.

These run the installed entry point in a subprocess, so they exercise exactly
what a user gets from a shell - argument parsing, exit status and all.
"""
import os
import subprocess
import sys

import pytest

ENV = {**os.environ, "MPLBACKEND": "Agg", "BIOSUITE_QUIET": "1"}


def _run(args, stdin_text=""):
    return subprocess.run(
        [sys.executable, "-m", "biosuite.cli.menu", *args],
        input=stdin_text, capture_output=True, text=True, timeout=180, env=ENV)


def _run_cli(args, stdin_text=""):
    """Invoke main_cli the way the console script does."""
    code = (
        "import sys;"
        "from biosuite.cli.menu import main_cli;"
        f"sys.argv=['biosuite', *{args!r}];"
        "sys.exit(main_cli())"
    )
    return subprocess.run([sys.executable, "-c", code], input=stdin_text,
                          capture_output=True, text=True, timeout=180, env=ENV)


# ── BSU-003: no code execution from menu input ──────────────────────────────

RCE_MARKER = "/tmp/biosuite-cli-rce-marker"
RCE_PAYLOAD = f'__import__("os").system("touch {RCE_MARKER}")'


@pytest.fixture(autouse=True)
def _clean_marker():
    for path in (RCE_MARKER, RCE_MARKER + "2"):
        if os.path.exists(path):
            os.unlink(path)
    yield
    for path in (RCE_MARKER, RCE_MARKER + "2"):
        if os.path.exists(path):
            os.unlink(path)


def test_pipeline_builder_does_not_execute_python(tmp_path):
    """Menu option 92 used to eval() whatever the user typed."""
    result = _run_cli([], stdin_text=f"92\n1\nstep1={RCE_PAYLOAD}\nACGT\n0\n")
    assert not os.path.exists(RCE_MARKER), "menu option 92 executed user code"
    assert "Unknown function" in result.stdout or "Traceback" not in result.stdout


def test_batch_builder_does_not_execute_python():
    """Menu option 93 used to eval() the function name."""
    result = _run_cli([], stdin_text=f"93\n{RCE_PAYLOAD}\nACGT\n0\n")
    assert not os.path.exists(RCE_MARKER), "menu option 93 executed user code"
    assert "Unknown function" in result.stdout


def test_batch_builder_still_runs_a_legitimate_function():
    result = _run_cli([], stdin_text="93\ngc_content\nATCG,GGCC\n0\n")
    assert "2 done" in result.stdout or "done" in result.stdout.lower()
    assert result.returncode == 0


def test_safe_resolver_rejects_arbitrary_imports():
    from biosuite.cli.menu import resolve_safe_callable
    for payload in ("os:system", "subprocess:Popen", "builtins:eval",
                    "shutil:rmtree", "__import__('os').system",
                    "biosuite.core.sequence:__builtins__"):
        with pytest.raises(ValueError):
            resolve_safe_callable(payload)


def test_safe_resolver_accepts_the_documented_forms():
    from biosuite.cli.menu import SAFE_FUNCTIONS, resolve_safe_callable
    for alias in SAFE_FUNCTIONS:
        assert callable(resolve_safe_callable(alias))
    assert resolve_safe_callable("biosuite.core.sequence:gc_content")("ATGC") == 50.0


def test_no_eval_or_exec_in_the_cli_module():
    import ast
    import pathlib

    from biosuite.cli import menu
    tree = ast.parse(pathlib.Path(menu.__file__).read_text(encoding="utf8"))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert called.isdisjoint({"eval", "exec", "compile"})


# ── BSU-014: the session survives errors ────────────────────────────────────

def test_eof_at_the_menu_prompt_exits_cleanly():
    result = _run_cli([], stdin_text="")
    assert result.returncode == 0
    assert "Traceback" not in result.stderr
    assert "Goodbye" in result.stdout


def test_a_failing_action_returns_to_the_menu():
    result = _run_cli([], stdin_text="75\n((((notnewick\n0\n")
    assert result.returncode == 0
    assert "Traceback" not in result.stderr
    assert "session is still running" in result.stdout
    assert "Goodbye" in result.stdout


def test_invalid_menu_option_does_not_crash():
    result = _run_cli([], stdin_text="not-an-option\n0\n")
    assert result.returncode == 0
    assert "Traceback" not in result.stderr


def test_invalid_port_for_the_api_launcher_is_reported():
    result = _run_cli([], stdin_text="90\nnotaport\n0\n")
    assert result.returncode == 0
    assert "Invalid port" in result.stdout
    assert "Traceback" not in result.stderr


def test_missing_input_file_is_reported_not_raised():
    result = _run_cli([], stdin_text="28\n/nonexistent/path/x.fasta\n0\n")
    assert result.returncode == 0
    assert "Traceback" not in result.stderr


# ── exit codes ──────────────────────────────────────────────────────────────

def test_successful_command_exits_zero():
    result = _run_cli(["gc", "ATCGATCG"])
    assert result.returncode == 0
    assert "50" in result.stdout


def test_missing_argument_is_exit_code_two():
    result = _run_cli(["gc"])
    assert result.returncode == 2
    assert "Usage:" in result.stdout
    assert "Traceback" not in result.stderr


def test_unknown_command_is_exit_code_two():
    result = _run_cli(["definitely-not-a-command"])
    assert result.returncode == 2
    assert "Unknown command" in result.stdout


def test_version_flag_exits_zero():
    result = _run_cli(["--version"])
    assert result.returncode == 0


# ── BSU-NEW: the hwe command was structurally impossible ────────────────────

def test_hardy_weinberg_command_works():
    """`biosuite hwe 10 20 30` raised TypeError for every possible input.

    The registry passed three positional arguments to a function that takes a
    single dict, so the command could never succeed.
    """
    result = _run_cli(["hwe", "10", "20", "30"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    # chi2 for AA=10 Aa=20 aa=30 is 3.75 (p = 0.0528).
    assert "3.75" in result.stdout or "3.7" in result.stdout


def test_hardy_weinberg_rejects_non_numeric_input():
    result = _run_cli(["hwe", "ten", "20", "30"])
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
