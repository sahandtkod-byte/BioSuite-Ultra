"""Discovering the local drop-in plugins under ``~/.biosuite/plugins``.

A folder dropped there is the workflow the module documents for people who are not
ready to publish a package, and it is invisible to importlib.metadata — which is why
discovery has to hand the loaded class to `load_plugin()` itself.
"""
import logging
import os

import pytest

from biosuite.core import plugin as pl

LOCAL_SOURCE = """
from biosuite.core.plugin import BioSuitePlugin


class Plugin(BioSuitePlugin):
    def name(self):
        return "local_one"

    def version(self):
        return "0.3"

    def description(self):
        return "installed by dropping a folder in ~/.biosuite/plugins"

    def author(self):
        return "Local"

    def dependencies(self):
        return []

    def register(self, app):
        pass
"""


@pytest.fixture
def install_entry_points_nothing(monkeypatch):
    """Local folders are the only subject here, so the metadata lookup is
    switched off to make that visible in the results."""
    import importlib.metadata

    class EntryPoints:
        def select(self, group):
            return []

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: EntryPoints())


def _write_local_plugin(home, folder, body):
    path = os.path.join(str(home), ".biosuite", "plugins", folder)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "__init__.py"), "w", encoding="utf-8") as handle:
        handle.write(body)
    return path


def test_a_local_folder_is_discovered_and_loadable(fake_home, install_entry_points_nothing):
    _write_local_plugin(fake_home, "local_one", LOCAL_SOURCE)
    manager = pl.PluginManager()
    assert manager.discover() == ["local_one"]
    info = manager.plugins["local_one"]
    assert info.module_path.endswith(os.path.join("plugins", "local_one", "__init__.py"))
    assert manager.load_plugin("local_one") is True
    assert isinstance(manager.get_plugin("local_one"), pl.BioSuitePlugin)
    assert manager.get_plugin("local_one").name() == "local_one"


def test_a_local_plugin_is_loaded_from_the_cached_class_without_entry_points(
        fake_home, install_entry_points_nothing, monkeypatch):
    """Prove the entry-point machinery is never consulted for a drop-in plugin."""
    _write_local_plugin(fake_home, "local_one", LOCAL_SOURCE)
    manager = pl.PluginManager()
    manager.discover()
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "entry_points",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("entry points must not be consulted")))
    assert manager.load_plugin("local_one") is True


def test_without_a_plugins_folder_discovery_is_quiet(fake_home,
                                                    install_entry_points_nothing):
    manager = pl.PluginManager()
    assert manager.discover() == []
    assert manager.plugins == {}


def test_folders_are_visited_in_name_order(fake_home, install_entry_points_nothing):
    """Two folders that both define the same plugin name would otherwise have a
    winner decided by the filesystem."""
    _write_local_plugin(fake_home, "a_first", LOCAL_SOURCE.replace("local_one", "same"))
    _write_local_plugin(fake_home, "z_second",
                        LOCAL_SOURCE.replace("local_one", "same")
                        .replace('return "0.3"', 'return "later"'))
    manager = pl.PluginManager()
    assert manager.discover() == ["same"]
    assert manager.plugins["same"].version == "later"


# ── folders that are not plugins ────────────────────────────────────────────

@pytest.mark.parametrize("folder, body, expected_log", [
    ("raises", "raise ImportError('nope')\n", "Failed to load local plugin raises"),
    ("no_plugin", "X = 1\n", "has no Plugin class"),
    ("not_a_plugin", "class Plugin:\n    pass\n", "not a BioSuitePlugin"),
    ("abstract", "from biosuite.core.plugin import BioSuitePlugin\n"
                 "class Plugin(BioSuitePlugin):\n    pass\n",
     "Failed to instantiate plugin from local folder abstract"),
])
def test_a_folder_that_cannot_be_a_plugin_is_skipped_with_a_reason(
        fake_home, install_entry_points_nothing, caplog, folder, body, expected_log):
    _write_local_plugin(fake_home, folder, body)
    with caplog.at_level(logging.WARNING):
        manager = pl.PluginManager()
        assert manager.discover() == []
    assert manager.plugins == {}
    assert expected_log in caplog.text


def test_a_positive_control_next_to_the_broken_folders(fake_home,
                                                       install_entry_points_nothing,
                                                       caplog):
    """One unusable folder must not cost the usable one beside it."""
    _write_local_plugin(fake_home, "broken", "raise ImportError('nope')\n")
    _write_local_plugin(fake_home, "local_one", LOCAL_SOURCE)
    with caplog.at_level(logging.WARNING):
        manager = pl.PluginManager()
        assert manager.discover() == ["local_one"]
    assert "Failed to load local plugin broken" in caplog.text


def test_a_stray_file_in_the_plugins_folder_is_ignored(fake_home,
                                                      install_entry_points_nothing):
    stray = os.path.join(str(fake_home), ".biosuite", "plugins")
    os.makedirs(stray, exist_ok=True)
    with open(os.path.join(stray, "notes.txt"), "w", encoding="utf-8") as handle:
        handle.write("not a plugin\n")
    manager = pl.PluginManager()
    assert manager.discover() == []


def test_an_empty_folder_with_no_init_is_ignored(fake_home,
                                                 install_entry_points_nothing):
    os.makedirs(os.path.join(str(fake_home), ".biosuite", "plugins", "empty_folder"),
                exist_ok=True)
    manager = pl.PluginManager()
    assert manager.discover() == []


# ── the local folder against an installed distribution ──────────────────────

def test_a_local_folder_that_shadows_an_installed_plugin_says_so(
        fake_home, monkeypatch, caplog):
    """The drop-in copy wins — the user put it there — but the installed copy
    being silently replaced made the version in list_plugins() a mystery."""
    import importlib.metadata

    class Installed(pl.BioSuitePlugin):
        def name(self):
            return "drop_a"

        def version(self):
            return "1.0 (installed distribution)"

        def description(self):
            return "d"

        def author(self):
            return "a"

        def register(self, app):
            pass

    class EntryPoints:
        def select(self, group):
            return [type("EP", (), {"name": "drop_a", "dist": type(
                "D", (), {"project_name": "demo-dist"})(),
                "load": lambda self: Installed})()]

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: EntryPoints())
    _write_local_plugin(
        fake_home, "shadow",
        LOCAL_SOURCE.replace("local_one", "drop_a")
        .replace('return "0.3"', 'return "9.9 (shadow)"'))

    with caplog.at_level(logging.WARNING):
        manager = pl.PluginManager()
        discovered = manager.discover()
    assert discovered == ["drop_a"], "one name, reported once"
    assert manager.plugins["drop_a"].version == "9.9 (shadow)"
    assert "shadows the earlier copy from demo-dist" in caplog.text


def test_the_local_plugin_can_be_reloaded_after_unload(fake_home,
                                                       install_entry_points_nothing):
    _write_local_plugin(fake_home, "local_one", LOCAL_SOURCE)
    manager = pl.PluginManager()
    manager.discover()
    assert manager.load_plugin("local_one") is True
    manager.unload_plugin("local_one")
    assert manager.load_plugin("local_one") is True
