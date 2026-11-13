"""The persisted enable/disable state in ``~/.biosuite/plugins.json``.

A hand-edited JSON file is ordinary input.  Two shapes used to be fatal — a
top-level list and a per-plugin value that was not a mapping — and a third was
worse than fatal: `"enabled": "false"` is a non-empty string, so the plugin the
user had switched off stayed switched on, in silence.
"""
import json
import logging
import os

import pytest

from biosuite.core import plugin as pl


def _info(name="cfg_demo", enabled=True):
    return pl.PluginInfo(name=name, version="1.0", description="d", author="a",
                         module_path="m", enabled=enabled)


@pytest.fixture
def manager(fake_home):
    m = pl.PluginManager()
    m.plugins["cfg_demo"] = _info()
    return m


def _write_config(manager, text):
    os.makedirs(os.path.dirname(manager._config_path), exist_ok=True)
    with open(manager._config_path, "w", encoding="utf-8") as handle:
        handle.write(text)


# ── the round trip ──────────────────────────────────────────────────────────

def test_a_saved_state_comes_back_from_disk(manager, fake_home):
    manager.plugins["cfg_demo"].enabled = False
    assert manager.save_config() is True
    assert manager._config_path.startswith(str(fake_home))
    assert os.path.exists(manager._config_path)
    manager.plugins["cfg_demo"].enabled = True
    manager.load_config()
    assert manager.plugins["cfg_demo"].enabled is False


def test_the_file_holds_the_state_and_the_version_it_was_saved_with(manager):
    manager.plugins["cfg_demo"].enabled = False
    manager.save_config()
    with open(manager._config_path, encoding="utf-8") as handle:
        assert json.load(handle) == {"cfg_demo": {"enabled": False, "version": "1.0"}}


def test_saving_several_plugins_keeps_them_all(manager):
    manager.plugins["second"] = _info(name="second", enabled=False)
    manager.save_config()
    with open(manager._config_path, encoding="utf-8") as handle:
        saved = json.load(handle)
    assert sorted(saved) == ["cfg_demo", "second"]
    assert saved["second"]["enabled"] is False


def test_no_file_yet_is_not_an_error(manager, fake_home, caplog):
    assert not os.path.exists(manager._config_path)
    with caplog.at_level(logging.WARNING):
        manager.load_config()
    assert manager.plugins["cfg_demo"].enabled is True
    assert caplog.text == ""


# ── a file the user edited ──────────────────────────────────────────────────

def test_corrupt_json_is_logged_and_leaves_the_state_alone(manager, caplog):
    _write_config(manager, "{ not json")
    with caplog.at_level(logging.WARNING):
        manager.load_config()
    assert manager.plugins["cfg_demo"].enabled is True
    assert "Could not read plugin config" in caplog.text


def test_an_unreadable_file_is_reported_not_raised(manager, caplog, monkeypatch):
    _write_config(manager, '{"cfg_demo": {"enabled": true}}')

    def refuse(*args, **kwargs):
        raise OSError("device is gone")

    monkeypatch.setattr("builtins.open", refuse)
    with caplog.at_level(logging.WARNING):
        manager.load_config()
    assert "Could not read plugin config" in caplog.text


def test_a_top_level_list_does_not_crash_the_loader(manager, caplog):
    """The old code called .items() on whatever json.load returned."""
    _write_config(manager, "[1, 2]")
    with caplog.at_level(logging.WARNING):
        manager.load_config()
    assert "not a mapping of plugin name to settings" in caplog.text
    assert manager.plugins["cfg_demo"].enabled is True


def test_a_plugin_entry_that_is_not_a_mapping_is_ignored(manager, caplog):
    _write_config(manager, '{"cfg_demo": "not a dict"}')
    with caplog.at_level(logging.WARNING):
        manager.load_config()
    assert "holds 'not a dict', not a mapping" in caplog.text
    assert manager.plugins["cfg_demo"].enabled is True


@pytest.mark.parametrize("raw", ['"false"', "0", "1", "null", "[true]", '{"a": 1}'])
def test_an_enabled_that_is_not_true_or_false_is_not_guessed(manager, caplog, raw):
    """`"false"` is a non-empty string, and bool('false') is True: reading it as
    the flag kept a switched-off plugin switched on."""
    _write_config(manager, '{"cfg_demo": {"enabled": %s}}' % raw)
    manager.plugins["cfg_demo"].enabled = True
    with caplog.at_level(logging.WARNING):
        manager.load_config()
    assert manager.plugins["cfg_demo"].enabled is True, "nothing was applied"
    assert "which is not true or false" in caplog.text
    assert "keeping it as enabled=True" in caplog.text


def test_a_real_false_is_applied(manager):
    _write_config(manager, '{"cfg_demo": {"enabled": false}}')
    manager.plugins["cfg_demo"].enabled = True
    manager.load_config()
    assert manager.plugins["cfg_demo"].enabled is False


def test_a_missing_enabled_key_changes_nothing(manager):
    _write_config(manager, '{"cfg_demo": {"version": "9"}}')
    manager.load_config()
    assert manager.plugins["cfg_demo"].enabled is True


def test_an_unknown_plugin_in_the_file_is_ignored(manager, caplog):
    _write_config(manager, '{"ghost": {"enabled": false}}')
    with caplog.at_level(logging.WARNING):
        manager.load_config()
    assert manager.plugins["cfg_demo"].enabled is True
    assert "ghost" not in caplog.text


# ── the state reaching the lifecycle ────────────────────────────────────────

def test_discover_applies_the_saved_state_so_load_all_honours_it(fake_home, monkeypatch):
    """Nothing in the product called load_config(), so a plugin the user had
    disabled was loaded again on every start."""
    import importlib.metadata

    class Off(pl.BioSuitePlugin):
        def name(self):
            return "cfg"

        def version(self):
            return "1"

        def description(self):
            return "d"

        def author(self):
            return "a"

        def register(self, app):
            pass

    class EntryPoints:
        def select(self, group):
            return [type("EP", (), {"name": "cfg", "dist": type(
                "D", (), {"project_name": "demo-dist"})(),
                "load": lambda self: Off})()]

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: EntryPoints())
    config_dir = os.path.join(str(fake_home), ".biosuite")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "plugins.json"), "w", encoding="utf-8") as fh:
        fh.write('{"cfg": {"enabled": false, "version": "1"}}')

    manager = pl.PluginManager()
    assert manager.discover() == ["cfg"]
    assert manager.plugins["cfg"].enabled is False
    assert manager.load_all() == 0


def test_save_config_reports_a_failure_it_could_not_write(manager, monkeypatch, caplog):
    def refuse(*args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr("builtins.open", refuse)
    with caplog.at_level(logging.WARNING):
        assert manager.save_config() is False
    assert "Could not write plugin config" in caplog.text
    assert "Permission denied" in caplog.text
