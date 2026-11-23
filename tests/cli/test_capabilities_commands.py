"""WS6: the non-interactive ``manifest`` and ``doctor`` subcommands.

Real subprocess runs check stdout/stderr/exit codes and manifest determinism;
in-process runs with monkeypatched seams cover the doctor branches (required
engine missing, catalog failure) without touching the filesystem or network.
No test launches an external executable and no test performs network I/O.
"""
import importlib
import json
import subprocess
import sys

import pytest

from biosuite.cli.capabilities import REQUIRED_ENGINES, run_doctor, run_manifest
from biosuite.cli.menu import COMMAND_REGISTRY, _import_func

_RUNNER = "import sys; from biosuite.cli.menu import main_cli; sys.exit(main_cli(sys.argv[1:]))"


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-c", _RUNNER, *args],
        capture_output=True, text=True, timeout=300,
    )


@pytest.fixture
def _no_side_channels(monkeypatch):
    """Fail the test if doctor tries to launch an executable or use the network."""
    import socket
    import subprocess as sp

    def _forbidden(*args, **kwargs):
        raise AssertionError("external process launch attempted")

    monkeypatch.setattr(sp, "Popen", _forbidden)
    monkeypatch.setattr(sp, "run", _forbidden)
    monkeypatch.setattr(sp, "check_output", _forbidden)
    monkeypatch.setattr(socket.socket, "connect", _forbidden)


# ------------------------------------------------------------------ manifest

def test_manifest_prints_deterministic_json_to_stdout():
    from biosuite.capabilities.catalog import list_capabilities

    first = _run_cli("manifest")
    assert first.returncode == 0, first.stderr
    assert first.stderr == ""
    manifest = json.loads(first.stdout)  # stdout must be pure JSON
    assert manifest["total"] == len(list_capabilities())
    assert manifest["total"] == len(manifest["capabilities"])
    assert first.stdout.endswith("\n")


def test_manifest_bytes_are_identical_across_two_processes():
    first = _run_cli("manifest")
    second = _run_cli("manifest")
    assert first.returncode == 0 and second.returncode == 0
    assert first.stdout == second.stdout


def test_manifest_carries_no_runtime_state():
    manifest = json.loads(_run_cli("manifest").stdout)
    assert set(manifest) == {
        "schema_version", "total", "mcp_exposed_total", "capabilities", "mcp_tools"
    }
    assert manifest["schema_version"] == "1.0.0"
    assert "timestamp" not in manifest and "versions" not in manifest
    for entry in manifest["capabilities"]:
        assert set(entry["requires"]) == {"tools", "packages"}


def test_manifest_registry_entry_is_well_formed():
    entry = COMMAND_REGISTRY["manifest"]
    assert entry["args"] == []
    func = _import_func(entry["func"])
    assert callable(func)
    assert func.__name__ == "cli_manifest"


def test_manifest_catalog_error_goes_to_stderr_with_nonzero_exit(monkeypatch):
    import biosuite.capabilities.catalog as catalog

    def _boom():
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(catalog, "list_capabilities", _boom)
    code, out, err = run_manifest()
    assert code == 3
    assert out == ""
    assert "RuntimeError" in err and "catalog unavailable" in err


def test_manifest_serialization_error_is_reported(monkeypatch):
    import biosuite.capabilities.manifest as manifest_module

    monkeypatch.setattr(manifest_module, "manifest_json",
                        lambda manifest: (_ for _ in ()).throw(ValueError("no")))
    code, out, err = run_manifest()
    assert code == 3
    assert out == "" and err != ""


# ------------------------------------------------------------------- doctor

def test_doctor_reports_availability_without_launching_anything(_no_side_channels):
    # Import first: the block patches subprocess/socket and must be active
    # only for the doctor call itself, not for module imports.
    importlib.import_module("biosuite.capabilities.catalog")
    import biosuite.capabilities.discovery as discovery

    assert not hasattr(discovery, "has_tool")
    code, out, err = run_doctor()
    assert code == 0, err
    assert "runnable" not in out  # runnability is never claimed
    assert "located" in out


def test_doctor_labels_missing_optional_tool_but_exits_zero(monkeypatch, _no_side_channels):
    import biosuite.capabilities.discovery as discovery

    monkeypatch.setattr(discovery, "_probe_tool",
                        lambda name: discovery.ToolStatus(name=name, available=False))
    code, out, err = run_doctor()
    assert code == 0  # missing optional tools are warnings only
    assert "tool blastn: not located" in out
    assert "warning: optional dependency blastn was not found" in out


def test_doctor_missing_required_engine_exits_one(monkeypatch, _no_side_channels):
    monkeypatch.setattr(sys.modules["biosuite.cli.capabilities"],
                        "REQUIRED_ENGINES", ("blastn",))
    import biosuite.capabilities.discovery as discovery

    monkeypatch.setattr(discovery, "_probe_tool",
                        lambda name: discovery.ToolStatus(name=name, available=False))
    code, out, err = run_doctor()
    assert code == 1
    assert "error: required engine blastn is missing" in out
    assert "missing required engine(s): blastn" in err


def test_doctor_required_engine_present_exits_zero(monkeypatch, _no_side_channels):
    monkeypatch.setattr(sys.modules["biosuite.cli.capabilities"],
                        "REQUIRED_ENGINES", ("blastn",))
    import biosuite.capabilities.discovery as discovery

    monkeypatch.setattr(discovery, "_probe_tool",
                        lambda name: discovery.ToolStatus(name=name, available=True))
    code, out, err = run_doctor()
    assert code == 0, err
    assert "error: required engine blastn is missing" not in out
    assert "tool blastn: located" in out


def test_doctor_catalog_error_is_reported(monkeypatch, _no_side_channels):
    import biosuite.capabilities.catalog as catalog

    def _boom():
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(catalog, "list_capabilities", _boom)
    code, out, err = run_doctor()
    assert code == 3
    assert out == "" and "catalog unavailable" in err


def test_doctor_import_failure_is_reported(monkeypatch):
    # `from ..capabilities.catalog import ...` fails when the module entry in
    # sys.modules is None: the CLI reports the failure instead of tracing it.
    monkeypatch.setitem(sys.modules, "biosuite.capabilities.catalog", None)
    code, out, err = run_doctor()
    assert code == 3
    assert out == ""
    assert "doctor failed: ModuleNotFoundError" in err


def test_doctor_required_engine_absent_from_discovery_exits_one(monkeypatch, _no_side_channels):
    # A required engine that discovery never even reported must still be
    # flagged missing (and only once), not silently skipped.
    monkeypatch.setattr(sys.modules["biosuite.cli.capabilities"],
                        "REQUIRED_ENGINES", ("no-such-engine",))
    code, out, err = run_doctor()
    assert code == 1
    assert "tool no-such-engine: not located" in out
    assert out.count("error: required engine no-such-engine is missing") == 1
    assert "missing required engine(s): no-such-engine" in err


def test_manifest_cli_failure_writes_stderr_and_exits_three(monkeypatch, capsys):
    import biosuite.cli.capabilities as cap_cli

    monkeypatch.setattr(cap_cli, "run_manifest",
                        lambda: (3, "", "manifest failed: RuntimeError: no\n"))
    with pytest.raises(SystemExit) as excinfo:
        cap_cli.cli_manifest()

    assert excinfo.value.code == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "manifest failed: RuntimeError: no" in captured.err


def test_doctor_registry_entry_is_well_formed():
    entry = COMMAND_REGISTRY["doctor"]
    assert entry["args"] == []
    func = _import_func(entry["func"])
    assert func.__name__ == "cli_doctor"


def test_default_required_engines_are_empty():
    assert REQUIRED_ENGINES == ()


def test_doctor_subprocess_exit_zero_and_no_stderr():
    proc = _run_cli("doctor")
    assert proc.returncode == 0, proc.stderr
    assert "catalog:" in proc.stdout
