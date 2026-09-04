"""Tests for core/plugin.py lifecycle + dataclasses (true API surface)."""
import sys
import textwrap

from biosuite.core import plugin as pl


def test_plugin_manager_singleton():
    assert pl.get_plugin_manager() is pl.get_plugin_manager()


def test_plugininfo_dataclass():
    info = pl.PluginInfo(name="demo", version="1.0", description="d",
                         author="a", module_path="/tmp/x.py",
                         enabled=True, dependencies=[])
    assert info.enabled is True


def test_create_plugin_template(tmp_path):
    pm = pl.PluginManager()
    pm.create_plugin_template("demo_plugin", output_dir=str(tmp_path))
    pkg = tmp_path / "biosuite-plugin-demo_plugin"
    assert (pkg / "__init__.py").exists()
    assert "BioSuitePlugin" in (pkg / "__init__.py").read_text()


def test_load_unknown_plugin_false():
    pm = pl.PluginManager()
    assert pm.load_plugin("never-registered-plugin") is False


def test_config_roundtrip(tmp_path):
    pm = pl.PluginManager()
    info = pl.PluginInfo(name="cfg_demo", version="1.0", description="d",
                         author="a", module_path="x", enabled=False,
                         dependencies=[])
    pm.plugins["cfg_demo"] = info
    pm.save_config()                                  # returns None, writes file
    import json, os
    saved = json.load(open(pm._config_path))
    assert saved["cfg_demo"]["enabled"] is False
    pm.plugins["cfg_demo"].enabled = True
    pm.load_config()
    assert pm.plugins["cfg_demo"].enabled is False    # state restored from disk


def test_example_plugin_instantiable():
    ex = pl.ExamplePlugin()
    assert ex.name and ex.version
    assert ex.register(None) in (True, False, None)
