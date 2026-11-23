"""Import-contract tests: the package must stay stdlib-only at import time."""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "biosuite" / "capabilities"

THIRD_PARTY_MODULES = frozenset(
    {"fastapi", "pydantic", "numpy", "pandas", "Bio", "scanpy", "anndata"}
)

STDLIB_OK = frozenset(sys.stdlib_module_names)


def _imported_root_names(path: pathlib.Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize(
    "py_file", sorted(PACKAGE_ROOT.glob("*.py")), ids=lambda p: p.name
)
def test_capability_modules_import_only_stdlib_or_self(py_file):
    roots = _imported_root_names(py_file)
    offending = {
        name
        for name in roots
        if name not in STDLIB_OK and name not in {"biosuite", "__future__"}
    }
    assert not offending, f"{py_file.name} imports non-stdlib: {sorted(offending)}"


@pytest.mark.parametrize(
    "module_name",
    sorted(
        {p.stem for p in PACKAGE_ROOT.glob("*.py")} - {"__init__"}
    ),
)
def test_importing_capability_module_does_not_pull_third_party_packages(
    module_name, monkeypatch
):
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    code = (
        "import sys; import biosuite.capabilities."
        + module_name
        + "; offenders = {'fastapi','pydantic','numpy','pandas','Bio','scanpy','anndata'} & set(sys.modules)"
        "; assert not offenders, offenders"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PACKAGE_ROOT.parents[1]),
    )
    assert result.returncode == 0, result.stderr


def test_package_import_does_not_import_core_modules():
    code = (
        "import sys; import biosuite.capabilities"
        "; leaked = [m for m in sys.modules if m.startswith('biosuite.core')]"
        "; assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PACKAGE_ROOT.parents[1]),
    )
    assert result.returncode == 0, result.stderr


def test_capability_modules_never_discover_at_runtime():
    for py_file in PACKAGE_ROOT.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "walk" and isinstance(
                    node.func.value, ast.Name
                ) and node.func.value.id == "ast":
                    pytest.fail(f"runtime AST discovery in {py_file.name}")
