"""WS8: the packaged manifest survives distribution end to end.

``capabilities.json`` exists so that an *installed* BioSuite Ultra — a wheel on
a machine with no source checkout — can serve the same deterministic manifest
byte for byte.  These checks exercise the distribution path the way a consumer
would: through ``importlib.resources`` and the console script, from working
directories that are not the repository, and against the packaging metadata
that decides whether the artifact ships at all.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from importlib import resources

REPO = pathlib.Path(__file__).resolve().parents[2]


def _utf8_env():
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


def _generated_manifest_text() -> str:
    from biosuite.capabilities.manifest import build_manifest, manifest_json
    return manifest_json(build_manifest())


def _packaged_manifest_text() -> str:
    return (resources.files("biosuite.capabilities")
            .joinpath("capabilities.json").read_text(encoding="utf-8"))


def _python_snippet(code: str, cwd) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, encoding="utf-8", env=_utf8_env(),
                          cwd=cwd, timeout=300)


def test_the_packaged_manifest_is_readable_through_importlib_resources():
    data = json.loads(_packaged_manifest_text())

    assert data["schema_version"] == "1.0.0"
    assert data["total"] == len(data["capabilities"]) > 0


def test_a_foreign_working_directory_sees_the_same_bytes():
    """A consumer's cwd must not change what importlib.resources hands back."""
    code = (
        "from importlib import resources;"
        "print(resources.files('biosuite.capabilities')"
        ".joinpath('capabilities.json').read_text(encoding='utf-8'), end='')"
    )
    proc = _python_snippet(code, cwd=REPO)
    elsewhere = _python_snippet(code, cwd=REPO.parent)

    assert proc.returncode == 0 and elsewhere.returncode == 0
    assert proc.stdout == elsewhere.stdout == _packaged_manifest_text()


def test_the_console_script_serves_the_packaged_manifest_outside_the_repo(tmp_path):
    """The end-to-end distribution contract: installed tool, foreign cwd, one truth."""
    exe = pathlib.Path(sys.executable).with_name("biosuite")
    argv = [str(exe)] if exe.exists() else [sys.executable, "-m", "biosuite.cli.menu"]
    proc = subprocess.run(argv + ["manifest"], capture_output=True, text=True,
                          encoding="utf-8", env=_utf8_env(),
                          cwd=tmp_path, timeout=300)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == _packaged_manifest_text() == _generated_manifest_text()


def test_the_packaging_metadata_ships_the_static_manifest():
    """The artifact reaches a wheel only if the packaging config lets it through."""
    manifest_in = (REPO / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include biosuite *.json" in manifest_in, (
        "the sdist drops biosuite/capabilities/capabilities.json")

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools.packages.find]" in pyproject
    assert 'include = ["biosuite*"]' in pyproject, (
        "the biosuite.capabilities package must be collected for the wheel")
    assert "include-package-data = false" not in pyproject, (
        "package data (capabilities.json) must not be excluded from the wheel")
