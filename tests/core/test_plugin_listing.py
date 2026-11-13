"""What `list_plugins()` prints, including the state `load_all()` obeys.

The table used to show a plugin that the manager would refuse to load exactly as it
showed a loadable one, because the enabled flag was never printed — and the CLI's
only view of the plugin system is that table.
"""
import importlib.metadata

import pytest

from biosuite.core import plugin as pl


class EntryPoints:
    def __init__(self, entries=()):
        self._entries = list(entries)

    def select(self, group):
        return list(self._entries) if group == "biosuite.plugins" else []


class EP:
    def __init__(self, name, loaded, project="demo-dist"):
        self.name = name
        self._loaded = loaded
        self.dist = type("D", (), {"project_name": project})()

    def load(self):
        return self._loaded


@pytest.fixture(autouse=True)
def _no_plugins_installed(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: EntryPoints())


class GoodPlugin(pl.BioSuitePlugin):
    def name(self):
        return "good"

    def version(self):
        return "2.1"

    def description(self):
        return "a well behaved plugin"

    def author(self):
        return "Tester"

    def register(self, app):
        pass


def _info(name, version="1.0", description="d", author="a", enabled=True):
    return pl.PluginInfo(name=name, version=version, description=description,
                         author=author, module_path="m", enabled=enabled)


def test_the_empty_manager_says_how_to_get_plugins(capsys):
    pl.PluginManager().list_plugins()
    out = capsys.readouterr().out
    assert "No plugins found." in out
    assert "pip install biosuite-plugin-<name>" in out
    assert "biosuite/core/plugin.py for documentation" in out


def test_a_plugin_is_listed_with_its_metadata(capsys):
    manager = pl.PluginManager()
    manager.plugins["good"] = _info("good", version="2.1",
                                    description="a well behaved plugin",
                                    author="Tester")
    manager.list_plugins()
    out = capsys.readouterr().out
    assert "Found 1 plugin(s):" in out
    assert "a well behaved plugin" in out
    assert "[loaded]" not in out
    assert "[disabled]" not in out


def test_a_loaded_plugin_is_marked(capsys, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points",
                        lambda: EntryPoints([EP("good", GoodPlugin)]))
    manager = pl.PluginManager()
    manager.plugins["good"] = _info("good")
    manager.list_plugins()
    assert "[loaded]" not in capsys.readouterr().out
    manager.load_plugin("good")
    manager.list_plugins()
    assert "[loaded]" in capsys.readouterr().out


def test_a_disabled_plugin_is_marked_as_disabled(capsys):
    """load_all() skips it; the table has to say why."""
    manager = pl.PluginManager()
    manager.plugins["off"] = _info("off", enabled=False)
    manager.list_plugins()
    out = capsys.readouterr().out
    assert "[disabled]" in out
    assert "off" in out
    assert manager.load_all() == 0


def test_a_loaded_and_disabled_plugin_shows_both_states(capsys):
    manager = pl.PluginManager()
    info = _info("odd", enabled=False)
    manager.plugins["odd"] = info
    manager.loaded["odd"] = GoodPlugin()
    manager.list_plugins()
    out = capsys.readouterr().out
    assert "[loaded]" in out and "[disabled]" in out


def test_several_plugins_are_all_listed(capsys):
    manager = pl.PluginManager()
    manager.plugins["a"] = _info("a", description="first")
    manager.plugins["b"] = _info("b", description="second", enabled=False)
    manager.list_plugins()
    out = capsys.readouterr().out
    assert "Found 2 plugin(s):" in out
    assert "first" in out and "second" in out


def test_the_header_row_names_the_four_columns(capsys):
    manager = pl.PluginManager()
    manager.plugins["a"] = _info("a")
    manager.list_plugins()
    header = capsys.readouterr().out.splitlines()[2]
    assert header.startswith("Name")
    for column in ("Version", "Author", "Description"):
        assert column in header


def test_a_long_description_does_not_break_the_row_shape(capsys):
    manager = pl.PluginManager()
    manager.plugins["a"] = _info("a", description="x" * 60)
    manager.list_plugins()
    row = [line for line in capsys.readouterr().out.splitlines()
           if line.startswith("a ") or line.startswith("a" + " " * 24)][0]
    assert "x" * 60 in row
