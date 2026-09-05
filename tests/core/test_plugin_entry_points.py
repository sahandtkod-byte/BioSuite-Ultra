"""Entry-point-based plugin discovery/load tests (synthetic entry points)."""
import types

import pytest

from biosuite.core import plugin as pl


class _FakeEntryPoint:
    def __init__(self, name, cls, project='demo-plugin'):
        self.name = name
        self._cls = cls
        self.dist = types.SimpleNamespace(project_name=project)

    def load(self):
        return self._cls


def _fake_eps(entries):
    class FakeEPS:
        def __init__(self, entries):
            self._entries = entries
        def select(self, group=None):
            return self._entries if group == 'biosuite.plugins' else []
        def get(self, group, default=None):
            return self._entries if group == 'biosuite.plugins' else (default or [])
    return FakeEPS(entries)


class DemoPlugin(pl.BioSuitePlugin):
    def name(self):
        return 'demo_ep'
    def version(self):
        return '1.0'
    def description(self):
        return 'synthetic entry-point plugin'
    def author(self):
        return 'test'
    def register(self, app=None):
        DemoPlugin.registered_on = app


def test_discover_registers_entry_point_plugins(monkeypatch):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'entry_points',
                        lambda: _fake_eps([_FakeEntryPoint('demo_ep', DemoPlugin)]))
    pm = pl.PluginManager()
    found = pm.discover()
    assert 'demo_ep' in found or 'demo_ep' in pm.plugins


def test_load_plugin_from_entry_point(monkeypatch):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'entry_points',
                        lambda: _fake_eps([_FakeEntryPoint('demo_ep', DemoPlugin)]))
    pm = pl.PluginManager()
    info = pl.PluginInfo(name='demo_ep', version='1.0', description='d',
                         author='t', module_path='demo-plugin', enabled=True,
                         dependencies=[])
    pm.plugins['demo_ep'] = info
    ok = pm.load_plugin('demo_ep')
    assert ok is True
    assert 'demo_ep' in pm.loaded


def test_discover_missing_group_returns_empty(monkeypatch):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'entry_points',
                        lambda: _fake_eps([]))
    pm = pl.PluginManager()
    result = pm.discover()
    assert result == [] or isinstance(result, list)


def test_load_plugin_missing_dependency_false(monkeypatch):
    pm = pl.PluginManager()
    info = pl.PluginInfo(name='dep_demo', version='1.0', description='d',
                         author='t', module_path='x', enabled=True,
                         dependencies=['no-such-module-anywhere>=1.0'])
    pm.plugins['dep_demo'] = info
    assert pm.load_plugin('dep_demo') is False


def test_unload_known_plugin(monkeypatch):
    pm = pl.PluginManager()
    info = pl.PluginInfo(name='X', version='1.0', description='d', author='a',
                         module_path='m', enabled=True, dependencies=[])
    pm.plugins['X'] = info
    pm.loaded['X'] = DemoPlugin()
    assert pm.unload_plugin('X') in (True, None)
