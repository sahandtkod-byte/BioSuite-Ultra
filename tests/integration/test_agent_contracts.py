"""WS8: agent-contract end-to-end checks.

These run the installed ``biosuite`` console script as a real subprocess — the
way an agent actually drives the tool — and hold ``docs/for_agents.md`` to
every claim it makes: the manifest bytes, the doctor labels, the exit codes,
the job allowlist and the documented import paths.  A documented promise that
the code does not keep is a defect; these tests keep the two honest.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from importlib import resources

REPO = pathlib.Path(__file__).resolve().parents[2]

JOB_ROUTES_DOCUMENTED = (
    "POST /api/v1/jobs",
    "GET /api/v1/jobs/{id}",
    "GET /api/v1/jobs/{id}/result",
    "POST /api/v1/jobs/{id}/cancel",
)


def _run_biosuite(*args, cwd=None):
    """Run the console script exactly as an agent would; return the process."""
    exe = pathlib.Path(sys.executable).with_name("biosuite")
    argv = [str(exe)] if exe.exists() else [sys.executable, "-m", "biosuite.cli.menu"]
    return subprocess.run(argv + list(args), capture_output=True, text=True,
                          cwd=cwd, timeout=300)


def _packaged_manifest_text() -> str:
    return (resources.files("biosuite.capabilities")
            .joinpath("capabilities.json").read_text(encoding="utf-8"))


# ── `biosuite manifest` — the machine-readable discovery contract ────────────

def test_manifest_subprocess_prints_deterministic_pure_json():
    first = _run_biosuite("manifest")
    second = _run_biosuite("manifest")

    assert first.returncode == 0, first.stderr
    assert first.stderr == "", "stdout carries output only on success"
    manifest = json.loads(first.stdout)  # stdout is nothing but the JSON document
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["total"] == len(manifest["capabilities"]) > 0
    assert first.stdout == second.stdout, "manifest bytes must not vary run to run"


def test_manifest_subprocess_matches_the_packaged_static_manifest():
    proc = _run_biosuite("manifest")

    assert proc.returncode == 0
    assert proc.stdout == _packaged_manifest_text(), (
        "the CLI manifest and the packaged capabilities.json must be byte-identical")


def test_manifest_is_identical_from_a_foreign_working_directory(tmp_path):
    proc = _run_biosuite("manifest", cwd=tmp_path)

    assert proc.returncode == 0
    assert proc.stdout == _packaged_manifest_text()


# ── `biosuite doctor` — the availability-report contract ─────────────────────

def test_doctor_subprocess_reports_the_catalog_with_the_documented_labels():
    from biosuite.capabilities.catalog import list_capabilities

    proc = _run_biosuite("doctor")

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == "", "stdout carries output only on success"
    lines = proc.stdout.splitlines()
    assert lines[0] == f"catalog: {len(list_capabilities())} capabilities"
    for line in lines[1:]:
        if line.startswith("tool "):
            assert re.fullmatch(r"tool \S+: (not )?located", line), line
        elif line.startswith("package "):
            assert re.fullmatch(r"package \S+: (not )?importable", line), line
        else:
            assert line.startswith("warning: optional dependency "), line
    assert "runnable" not in proc.stdout, "doctor never claims a located tool runs"


def test_manifest_and_doctor_take_no_arguments():
    for command in ("manifest", "doctor"):
        proc = _run_biosuite(command, "extra")

        assert proc.returncode == 2, (command, proc.stdout)
        assert "Too many arguments" in proc.stdout, command


# ── docs/for_agents.md — every documented claim is checked against the code ──

def test_the_documented_import_paths_resolve():
    from biosuite.capabilities import manifest  # noqa: F401
    from biosuite.capabilities.catalog import list_capabilities  # noqa: F401
    from biosuite.capabilities.discovery import discover_availability  # noqa: F401

    assert callable(list_capabilities) and callable(discover_availability)
    assert callable(manifest.manifest_json)


def test_the_documented_exit_codes_match_the_command_contract():
    from biosuite.cli import capabilities as cap_cli

    doc = (REPO / "docs" / "for_agents.md").read_text(encoding="utf-8")
    for code, phrase in (("0", "no required engine is missing"),
                         ("1", "a required engine is missing"),
                         ("3", "failed to load")):
        assert f"| {code} |" in doc, f"exit code {code} is not documented"
        assert phrase in doc, f"the meaning of exit {code} is not documented"
    assert "REQUIRED_ENGINES" in doc, "the extension point is not documented"
    assert isinstance(cap_cli.REQUIRED_ENGINES, tuple)


def test_the_documented_job_surface_matches_the_api_and_allowlist():
    doc = (REPO / "docs" / "for_agents.md").read_text(encoding="utf-8")
    for route in JOB_ROUTES_DOCUMENTED:
        assert route in doc, f"{route} is not documented"

    from biosuite.api.job_executor import ALLOWLIST
    assert ALLOWLIST == frozenset({"sequence.gc_content"}), (
        "docs/for_agents.md says the allowlist is currently sequence.gc_content")
    assert "sequence.gc_content" in doc

    from biosuite.api import app
    paths = set()
    for route in app.routes:
        if getattr(route, "path", None):
            paths.add(route.path)
        # Job routes live behind a sub-router, which app.routes does not
        # flatten; the same expansion the legacy-contract tests rely on.
        original = getattr(route, "original_router", None)
        if original is not None:
            paths.update(sub.path for sub in original.routes)
    for route in JOB_ROUTES_DOCUMENTED:
        path = route.split(" ", 1)[1]
        assert path in paths, f"{path} is documented but not served"


def test_the_documented_surface_counts_match_the_cli():
    from biosuite.cli.menu import COMMAND_REGISTRY

    doc = (REPO / "docs" / "for_agents.md").read_text(encoding="utf-8")
    measured = len(COMMAND_REGISTRY)
    assert measured == 22, f"subcommand count changed to {measured}; update the docs"
    assert f"{measured} documented subcommands" in doc
    assert {"manifest", "doctor"} <= set(COMMAND_REGISTRY)
    assert "117 options" in doc
