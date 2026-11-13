"""Discovering plugins through `biosuite.plugins` entry points.

Discovery is the one place that runs other people's packages at import time, so
every branch here is a way a third-party install can be wrong — and none of them
is allowed to take BioSuite's startup down with it, or to be skipped in silence.
"""
import importlib.metadata
import logging

import pytest

from biosuite.core import plugin as pl


class EntryPoint:
    """A stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name, loaded, project="demo-dist"):
        self.name = name
        self._loaded = loaded
        self.dist = type("Distribution", (), {"project_name": project})()

    def load(self):
        if isinstance(self._loaded, Exception):
            raise self._loaded
        return self._loaded


def install_entry_points(monkeypatch, entries):
    """Replace the metadata lookup for one test, group selection included."""

    class EntryPoints:
        def select(self, group):
            return list(entries) if group == "biosuite.plugins" else []

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

    def dependencies(self):
        return ["numpy"]

    def register(self, app):
        self.app = app


def _discovery(monkeypatch, entries, fake_home=None):
    install_entry_points(monkeypatch, entries)
    manager = pl.PluginManager()
    return manager, manager.discover()


# ── the happy path ──────────────────────────────────────────────────────────

def test_a_registered_plugin_is_discovered_with_its_metadata(monkeypatch):
    install_entry_points(monkeypatch, [EntryPoint("good", GoodPlugin)])
    manager = pl.PluginManager()
    assert manager.discover() == ["good"]
    info = manager.plugins["good"]
    assert (info.version, info.description, info.author) == \
        ("2.1", "a well behaved plugin", "Tester")
    assert info.dependencies == ["numpy"]
    assert info.enabled is True


def test_module_path_holds_the_distribution_name_not_its_repr(monkeypatch):
    """load_plugin() compares this field against ep.dist.project_name; storing
    str(ep.dist) — a repr like '<PathDistribution ...>' — made that match
    permanently false."""
    install_entry_points(monkeypatch, [EntryPoint("good", GoodPlugin,
                                                 project="biosuite-plugin-good")])
    manager = pl.PluginManager()
    manager.discover()
    assert manager.plugins["good"].module_path == "biosuite-plugin-good"


def test_an_entry_point_without_a_distribution_is_still_usable(monkeypatch):
    ep = EntryPoint("good", GoodPlugin)
    del ep.dist
    install_entry_points(monkeypatch, [ep])
    manager = pl.PluginManager()
    assert manager.discover() == ["good"]
    assert manager.plugins["good"].module_path == "unknown"


def test_the_entry_point_key_may_differ_from_the_plugin_name(monkeypatch):
    """Registration under a different key than Plugin.name() is legal and must
    still resolve — the fallback is the distribution name."""
    install_entry_points(monkeypatch,
                         [EntryPoint("registered_as_something_else", GoodPlugin)])
    manager = pl.PluginManager()
    assert manager.discover() == ["good"]
    assert manager.plugins["good"].module_path == "demo-dist"


def test_several_plugins_are_discovered_in_one_pass(monkeypatch):
    class Other(GoodPlugin):
        def name(self):
            return "other"

    install_entry_points(monkeypatch, [EntryPoint("good", GoodPlugin),
                                       EntryPoint("other", Other)])
    manager = pl.PluginManager()
    assert sorted(manager.discover()) == ["good", "other"]


def test_nothing_registered_discovers_nothing(monkeypatch):
    manager, discovered = _discovery(monkeypatch, [])
    assert discovered == []
    assert manager.plugins == {}


def test_the_lookup_asks_for_the_biosuite_plugins_group_and_nothing_else(monkeypatch):
    """Every supported Python answers entry_points() with an object that has
    select(); the group name is the only thing the scan may guess at."""
    import importlib.metadata
    asked = []

    class EntryPoints:
        def select(self, group):
            asked.append(group)
            return []

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: EntryPoints())
    pl.PluginManager().discover()
    assert asked, "the scan never consulted the group at all"
    assert set(asked) == {"biosuite.plugins"}


def test_an_entry_point_object_without_select_is_a_type_error_not_a_silent_miss(
        monkeypatch):
    """The dict-shaped 3.9 answer used to be accommodated; the project supports
    >=3.10, where it cannot happen, so nothing may swallow it either."""
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: {})
    with pytest.raises(AttributeError):
        pl.PluginManager().discover()


# ── what one bad package costs ──────────────────────────────────────────────

def test_an_entry_point_that_raises_costs_one_log_line(monkeypatch, caplog):
    install_entry_points(monkeypatch, [EntryPoint("boom", RuntimeError("bad module")),
                                        EntryPoint("good", GoodPlugin)])
    with caplog.at_level(logging.ERROR):
        manager = pl.PluginManager()
        assert manager.discover() == ["good"]
    assert "Failed to load plugin entry point boom: bad module" in caplog.text


def test_an_entry_point_that_loaded_a_function_is_said_to_be_one(monkeypatch, caplog):
    """issubclass() on a function raised TypeError and the log blamed loading."""
    install_entry_points(monkeypatch, [EntryPoint("func", lambda: GoodPlugin)])
    with caplog.at_level(logging.ERROR):
        manager = pl.PluginManager()
        assert manager.discover() == []
    assert "loaded a function, not a class" in caplog.text


def test_an_entry_point_that_is_not_a_plugin_says_which_class_it_was(
        monkeypatch, caplog):
    class NotAPlugin:
        pass

    install_entry_points(monkeypatch, [EntryPoint("other", NotAPlugin),
                                       EntryPoint("good", GoodPlugin)])
    with caplog.at_level(logging.ERROR):
        manager = pl.PluginManager()
        assert manager.discover() == ["good"]
    assert "NotAPlugin, which does not inherit from BioSuitePlugin" in caplog.text


def test_an_abstract_plugin_class_is_reported_as_instantiation_failing(
        monkeypatch, caplog):
    install_entry_points(monkeypatch, [EntryPoint("abs", pl.BioSuitePlugin)])
    with caplog.at_level(logging.ERROR):
        manager = pl.PluginManager()
        assert manager.discover() == []
    assert "Failed to instantiate plugin from entry point abs" in caplog.text
    assert "abstract" in caplog.text


@pytest.mark.parametrize("bad_name", ["", "   ", None, 7, ["list"]])
def test_a_plugin_that_cannot_name_itself_is_skipped_with_a_reason(
        monkeypatch, caplog, bad_name):
    class Nameless(GoodPlugin):
        def name(self):
            return bad_name

    install_entry_points(monkeypatch, [EntryPoint("nameless", Nameless)])
    with caplog.at_level(logging.ERROR):
        manager = pl.PluginManager()
        assert manager.discover() == []
    assert "needs a non-empty string name" in caplog.text
    assert manager.plugins == {}, \
        f"a plugin keyed on {bad_name!r} would be unloadable from the CLI"


def test_a_positive_control_that_a_usable_name_is_kept(monkeypatch):
    """The refusal above is only meaningful if an ordinary name still lands."""
    install_entry_points(monkeypatch, [EntryPoint("good", GoodPlugin)])
    manager = pl.PluginManager()
    assert list(manager.discover()) == ["good"]


def test_string_dependencies_are_read_as_one_requirement(monkeypatch):
    """A plugin returning 'numpy' used to be read one character at a time, and
    load_plugin then reported n, u, m, p and y as missing packages."""
    class StringDeps(GoodPlugin):
        def name(self):
            return "strdeps"

        def dependencies(self):
            return "numpy"

    install_entry_points(monkeypatch, [EntryPoint("strdeps", StringDeps)])
    manager = pl.PluginManager()
    manager.discover()
    assert manager.plugins["strdeps"].dependencies == ["numpy"]


@pytest.mark.parametrize("deps, expected", [
    (["numpy", "pandas>=2.0"], ["numpy", "pandas>=2.0"]),
    (("numpy",), ["numpy"]),
    (["numpy", "", None, 5], ["numpy"]),
    (None, []),
    (5, []),
])
def test_dependency_lists_are_normalised(monkeypatch, deps, expected):
    class Deps(GoodPlugin):
        def name(self):
            return "deps"

        def dependencies(self):
            return deps

    install_entry_points(monkeypatch, [EntryPoint("deps", Deps)])
    manager = pl.PluginManager()
    manager.discover()
    assert manager.plugins["deps"].dependencies == expected


@pytest.mark.parametrize("deps", [["numpy", "", None, 5], None, 5])
def test_a_useless_dependency_entry_is_logged(monkeypatch, caplog, deps):
    class Deps(GoodPlugin):
        def name(self):
            return "deps"

        def dependencies(self):
            return deps

    install_entry_points(monkeypatch, [EntryPoint("deps", Deps)])
    with caplog.at_level(logging.WARNING):
        pl.PluginManager().discover()
    assert "dependenc" in caplog.text


# ── collisions ──────────────────────────────────────────────────────────────

def test_two_entry_points_that_share_a_name_are_reported_and_returned_once(
        monkeypatch, caplog):
    class Clone(GoodPlugin):
        def version(self):
            return "9.9 clone"

    install_entry_points(monkeypatch, [EntryPoint("a", GoodPlugin),
                                       EntryPoint("b", Clone)])
    with caplog.at_level(logging.WARNING):
        manager = pl.PluginManager()
        discovered = manager.discover()
    assert discovered == ["good"], "the same name must not be listed twice"
    assert "shadows the earlier copy" in caplog.text
