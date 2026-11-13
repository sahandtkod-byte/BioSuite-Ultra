"""The example plugin BioSuite ships, the PluginInfo record and the global manager.

`ExamplePlugin` is what the module docstring tells a newcomer to copy, and
`get_plugin_manager()` is the shared instance an application is meant to register
plugins against — both are public surface, so their answers are pinned here rather
than left to the discovery machinery.
"""
import pytest

from biosuite.core import plugin as pl


# ── PluginInfo ──────────────────────────────────────────────────────────────

def test_the_record_carries_what_the_table_prints():
    info = pl.PluginInfo(name="demo", version="1.0", description="d", author="a",
                         module_path="/tmp/x.py")
    assert (info.name, info.version, info.description, info.author) == \
        ("demo", "1.0", "d", "a")
    assert info.module_path == "/tmp/x.py"


def test_the_defaults_are_enabled_and_no_dependencies():
    info = pl.PluginInfo(name="demo", version="1.0", description="d", author="a",
                         module_path="m")
    assert info.enabled is True
    assert info.dependencies == []


def test_two_records_do_not_share_a_dependency_list():
    """A mutable default on a dataclass is only safe because of default_factory;
    without it the second plugin would inherit the first one's requirements."""
    first = pl.PluginInfo(name="a", version="1", description="d", author="a",
                          module_path="m")
    second = pl.PluginInfo(name="b", version="1", description="d", author="a",
                           module_path="m")
    first.dependencies.append("numpy")
    assert second.dependencies == []


def test_module_path_holds_the_distribution_for_an_entry_point_plugin(monkeypatch):
    """The field is documented as 'where it came from', which is a project name
    for an installed plugin and a file path for a local one."""
    import importlib.metadata

    class EP:
        name = "good"
        dist = type("D", (), {"project_name": "biosuite-plugin-good"})()

        @staticmethod
        def load():
            class Good(pl.BioSuitePlugin):
                def name(self):
                    return "good"

                def version(self):
                    return "1"

                def description(self):
                    return "d"

                def author(self):
                    return "a"

                def register(self, app):
                    pass
            return Good

    class EntryPoints:
        def select(self, group):
            return [EP()]

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: EntryPoints())
    manager = pl.PluginManager()
    manager.discover()
    assert manager.plugins["good"].module_path == "biosuite-plugin-good"


# ── ExamplePlugin ───────────────────────────────────────────────────────────

def test_the_example_plugin_answers_the_metadata_it_promises():
    example = pl.ExamplePlugin()
    assert example.name() == "example"
    assert example.version() == "1.0.0"
    assert example.description() == "Example plugin for demonstration"
    assert example.author() == "BioSuite Team"
    assert example.dependencies() == []


def test_the_example_plugin_satisfies_the_abstract_contract():
    """It is the template every plugin copies, so it must be instantiable without
    overriding anything."""
    assert isinstance(pl.ExamplePlugin(), pl.BioSuitePlugin)
    assert pl.ExamplePlugin().register(None) is None


def test_the_example_demo_function_really_uses_the_core():
    assert pl.ExamplePlugin().reverse_complement_demo("ATCG") == "CGAT"
    assert pl.ExamplePlugin().reverse_complement_demo("") == ""


def test_the_example_plugin_can_be_discovered_like_a_real_one(monkeypatch):
    import importlib.metadata

    class EP:
        name = "example"

        @staticmethod
        def load():
            return pl.ExamplePlugin

    class EntryPoints:
        def select(self, group):
            return [EP()]

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: EntryPoints())
    manager = pl.PluginManager()
    assert manager.discover() == ["example"]
    assert manager.load_plugin("example") is True


# ── the global manager ──────────────────────────────────────────────────────

def test_the_global_manager_is_created_once():
    first = pl.get_plugin_manager()
    assert isinstance(first, pl.PluginManager)
    assert first is pl.get_plugin_manager()


def test_the_global_manager_keeps_what_was_put_on_it(monkeypatch):
    """Callers register the app on the shared instance; a second call must not
    hand back a fresh manager that forgot it."""
    monkeypatch.setattr(pl, "_default_manager", None)
    manager = pl.get_plugin_manager()
    sentinel = object()
    manager.app = sentinel
    assert pl.get_plugin_manager().app is sentinel


def test_the_global_manager_starts_empty(monkeypatch):
    monkeypatch.setattr(pl, "_default_manager", None)
    manager = pl.get_plugin_manager()
    assert manager.plugins == {}
    assert manager.loaded == {}
