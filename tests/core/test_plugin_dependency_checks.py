"""How a plugin's declared dependencies are checked before it is loaded.

The check imports each dependency by the name it is written with, so the way a
requirement string is parsed is what decides whether an installed plugin is
refused for a package that is present.  `numpy~=1.26`, `numpy[extra]>=1.0`,
`numpy;python_version<'3.12'` and a padded `  numpy >= 1.24 ` all mean numpy, and
all used to be reported as missing.
"""
import importlib.metadata

import pytest

from biosuite.core import plugin as pl


class EntryPoints:
    def select(self, group):
        return []


@pytest.fixture(autouse=True)
def _no_entry_points(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: EntryPoints())


def _manager_with(deps, name="dep"):
    manager = pl.PluginManager()
    manager.plugins[name] = pl.PluginInfo(name=name, version="1.0", description="d",
                                          author="a", module_path="m",
                                          dependencies=list(deps))
    return manager


# ── the requirement parser ──────────────────────────────────────────────────

@pytest.mark.parametrize("requirement, module", [
    ("numpy", "numpy"),
    ("numpy>=1.24", "numpy"),
    ("numpy~=1.26", "numpy"),
    ("numpy!=1.0", "numpy"),
    ("numpy>1", "numpy"),
    ("numpy<2", "numpy"),
    ("numpy==1.26.4", "numpy"),
    ("numpy[extra]>=1.0", "numpy"),
    ("  numpy >= 1.24 ", "numpy"),
    ("numpy;python_version<'3.12'", "numpy"),
    ("python-dateutil", "python-dateutil"),
])
def test_the_module_a_requirement_names(requirement, module):
    assert pl._requirement_module(requirement) == module


@pytest.mark.parametrize("empty", ["", "   ", "\t"])
def test_a_blank_requirement_names_no_module(empty):
    assert pl._requirement_module(empty) == ""


def test_numpy_is_recognised_as_installed(capsys):
    """The parser would be unfalsifiable if every requirement parsed to something
    that is missing anyway: these three all mean numpy, which is here."""
    manager = _manager_with(["numpy>=1.24", "numpy~=1.26", "numpy[extra]>=1.0"])
    assert manager.load_plugin("dep") is False
    out = capsys.readouterr().out
    assert "missing dependencies" not in out, \
        "a versioned requirement on an installed package must not block the load"
    assert "was discovered but no loadable implementation" in out, \
        "the refusal is the missing entry point, which is the next step after deps"


# ── what the manager answers ────────────────────────────────────────────────

def test_an_installed_dependency_does_not_block_the_load(capsys):
    manager = _manager_with(["numpy"])
    assert manager.load_plugin("dep") is False
    out = capsys.readouterr().out
    assert "missing dependencies" not in out
    assert "was discovered but no loadable implementation" in out


def test_a_version_specifier_on_an_installed_package_still_loads(capsys):
    """The refusal these tests replaced: 'numpy~=1.26' was parsed as a module
    named 'numpy~=1.26' and the plugin was blocked on a package that is there."""
    manager = _manager_with(["numpy~=1.26", "numpy>=1.24", "numpy;python_version<'3.12'"])
    manager.load_plugin("dep")
    out = capsys.readouterr().out
    assert "missing dependencies" not in out


def test_a_genuinely_missing_dependency_is_named(capsys):
    manager = _manager_with(["totally_absent_pkg"])
    assert manager.load_plugin("dep") is False
    out = capsys.readouterr().out
    assert "missing dependencies: totally_absent_pkg" in out
    assert "pip install totally_absent_pkg" in out


def test_an_installed_and_a_missing_dependency_are_told_apart(capsys):
    manager = _manager_with(["numpy>=1.24", "totally_absent_pkg"])
    assert manager.load_plugin("dep") is False
    out = capsys.readouterr().out
    assert "totally_absent_pkg" in out
    assert "numpy>=1.24" not in out, "an installed dependency must not be listed"


def test_a_versioned_requirement_is_quoted_in_the_hint(capsys):
    """`pip install numpy>=1.24` in a shell redirects stdout into a file named
    '=1.24'; the quotes are part of the instruction."""
    manager = _manager_with(["totally_absent_pkg>=2.0"])
    manager.load_plugin("dep")
    out = capsys.readouterr().out
    assert 'pip install "totally_absent_pkg>=2.0"' in out
    assert "pip install totally_absent_pkg>=2.0\n" not in out


def test_several_missing_dependencies_are_all_listed(capsys):
    manager = _manager_with(["absent_one", "absent_two>=1.0"])
    manager.load_plugin("dep")
    out = capsys.readouterr().out
    assert "absent_one, absent_two>=1.0" in out
    assert '"absent_two>=1.0"' in out


def test_the_module_name_caveat_is_stated(capsys):
    """A distribution whose module is named differently (python-dateutil over
    dateutil) cannot be checked by import name, and the user has to be told."""
    manager = _manager_with(["python-dateutil"])
    manager.load_plugin("dep")
    assert "has to list the module name" in capsys.readouterr().out


def test_a_blank_dependency_is_ignored_and_said(capsys):
    manager = _manager_with(["", "numpy"])
    assert manager.load_plugin("dep") is False
    assert "lists an empty dependency" in capsys.readouterr().out


def test_a_dependency_that_raises_on_import_is_reported_with_its_reason(capsys,
                                                                        monkeypatch):
    """An ImportError is one answer; a package that is installed and blows up with
    something else is a dependency the manager cannot rely on either — and the
    reason is the only clue the user gets."""
    real_import = importlib.import_module

    def exploding_import(name, *args, **kwargs):
        if name == "half_broken":
            raise RuntimeError("undefined symbol at import time")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(pl.importlib, "import_module", exploding_import)
    manager = _manager_with(["half_broken"])
    assert manager.load_plugin("dep") is False
    out = capsys.readouterr().out
    assert "could not be imported (RuntimeError: undefined symbol at import time)" in out
    assert "half_broken" in out
    assert "treating it as missing" in out


@pytest.mark.parametrize("bad", ["", "   "])
def test_no_dependency_exception_escapes_a_load(bad):
    """`import_module('')` used to raise ValueError out of a function documented
    to return True or False."""
    manager = _manager_with([bad])
    assert manager.load_plugin("dep") is False
