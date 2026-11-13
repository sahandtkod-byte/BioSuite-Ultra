"""`PluginManager.load_plugin()`, `_activate()` and `unload_plugin()`.

Loading is where a plugin's own code runs, so a failure must leave the manager
knowing the truth: an instance that raised in `register()` used to be filed as
loaded anyway, and one that raised in `on_unload()` could never be unloaded.
"""
import importlib.metadata
import logging

import pytest

from biosuite.core import plugin as pl


class EntryPoints:
    def __init__(self, entries):
        self._entries = entries

    def select(self, group):
        return list(self._entries) if group == "biosuite.plugins" else []


class EP:
    def __init__(self, name, loaded, project="demo-dist"):
        self.name = name
        self._loaded = loaded
        self.dist = type("D", (), {"project_name": project})()

    def load(self):
        if isinstance(self._loaded, Exception):
            raise self._loaded
        return self._loaded


def install(monkeypatch, entries):
    monkeypatch.setattr(importlib.metadata, "entry_points",
                        lambda: EntryPoints(entries))


class GoodPlugin(pl.BioSuitePlugin):
    loaded_calls = 0
    unloaded_calls = 0
    register_calls = 0

    def name(self):
        return "good"

    def version(self):
        return "2.1"

    def description(self):
        return "a well behaved plugin"

    def author(self):
        return "Tester"

    def register(self, app):
        GoodPlugin.register_calls += 1
        self.app = app

    def on_load(self):
        GoodPlugin.loaded_calls += 1

    def on_unload(self):
        GoodPlugin.unloaded_calls += 1


def _info(name="good", enabled=True, deps=(), module_path="demo-dist"):
    return pl.PluginInfo(name=name, version="1.0", description="d", author="a",
                         module_path=module_path, enabled=enabled,
                         dependencies=list(deps))


@pytest.fixture(autouse=True)
def _reset_counters():
    GoodPlugin.loaded_calls = 0
    GoodPlugin.unloaded_calls = 0
    GoodPlugin.register_calls = 0


# ── the plain answers ───────────────────────────────────────────────────────

def test_an_unknown_name_is_refused_and_tells_the_user_to_discover(capsys):
    manager = pl.PluginManager()
    assert manager.load_plugin("ghost") is False
    assert "Plugin 'ghost' not found. Run discover() first." in capsys.readouterr().out


def test_loading_twice_answers_with_the_existing_state(monkeypatch, capsys):
    install(monkeypatch, [EP("good", GoodPlugin)])
    manager = pl.PluginManager()
    manager.plugins["good"] = _info()
    assert manager.load_plugin("good") is True
    before = GoodPlugin.loaded_calls
    assert manager.load_plugin("good") is True
    assert GoodPlugin.loaded_calls == before, "the plugin was not started twice"
    assert "already loaded" in capsys.readouterr().out


def test_a_discovered_plugin_that_cannot_be_reached_says_so(monkeypatch, capsys):
    """A bare False after discover() proudly listed the plugin read like a bug in
    the plugin rather than a missing implementation."""
    install(monkeypatch, [EP("other", GoodPlugin, project="elsewhere")])
    manager = pl.PluginManager()
    manager.plugins["gone"] = _info(name="gone", module_path="no-such-dist")
    assert manager.load_plugin("gone") is False
    assert "was discovered but no loadable implementation" in capsys.readouterr().out


def test_an_entry_point_that_explodes_becomes_a_reported_error(monkeypatch, capsys):
    install(monkeypatch, [EP("good", RuntimeError("metadata db gone"))])
    manager = pl.PluginManager()
    manager.plugins["good"] = _info()
    assert manager.load_plugin("good") is False
    assert "Error loading plugin 'good': metadata db gone" in capsys.readouterr().out


def test_a_plugin_that_cannot_be_constructed_is_an_error_not_a_crash(monkeypatch,
                                                                     capsys):
    class Boom(GoodPlugin):
        def __init__(self):
            raise RuntimeError("cannot initialise")

    install(monkeypatch, [EP("boom", Boom)])
    manager = pl.PluginManager()
    manager.plugins["good"] = _info()
    assert manager.load_plugin("good") is False
    assert "Error loading plugin 'good'" in capsys.readouterr().out
    assert manager.loaded == {}


# ── what a successful load does ─────────────────────────────────────────────

def test_on_load_runs_and_the_instance_is_reachable(monkeypatch):
    install(monkeypatch, [EP("good", GoodPlugin)])
    manager = pl.PluginManager()
    manager.plugins["good"] = _info()
    assert manager.load_plugin("good") is True
    assert GoodPlugin.loaded_calls == 1
    assert isinstance(manager.get_plugin("good"), GoodPlugin)


def test_register_runs_only_once_an_application_exists(monkeypatch):
    install(monkeypatch, [EP("good", GoodPlugin)])
    manager = pl.PluginManager()
    manager.plugins["good"] = _info()
    manager.load_plugin("good")
    assert GoodPlugin.register_calls == 0
    assert not hasattr(manager.get_plugin("good"), "app")

    sentinel = object()
    manager.unload_plugin("good")
    manager.app = sentinel
    manager.load_plugin("good")
    assert GoodPlugin.register_calls == 1
    assert manager.get_plugin("good").app is sentinel


def test_an_app_that_is_present_but_falsy_is_still_an_app(monkeypatch):
    """`if self.app:` skipped registration for any object that happened to
    answer False to len()/bool()."""
    class EmptyApp:
        def __bool__(self):
            return False

    install(monkeypatch, [EP("good", GoodPlugin)])
    manager = pl.PluginManager()
    manager.plugins["good"] = _info()
    manager.app = EmptyApp()
    assert manager.load_plugin("good") is True
    assert GoodPlugin.register_calls == 1


def test_the_printed_version_is_the_one_discovery_recorded(monkeypatch, capsys):
    install(monkeypatch, [EP("good", GoodPlugin)])
    manager = pl.PluginManager()
    manager.plugins["good"] = _info()
    manager.load_plugin("good")
    assert "Loaded plugin: good v1.0" in capsys.readouterr().out


# ── a plugin that fails part way through ────────────────────────────────────

def test_a_register_failure_does_not_leave_the_plugin_loaded(monkeypatch, capsys):
    class RegisterBoom(GoodPlugin):
        def name(self):
            return "boom_register"

        def register(self, app):
            raise RuntimeError("the plugin's GUI hook exploded")

    install(monkeypatch, [EP("boom", RegisterBoom)])
    manager = pl.PluginManager()
    manager.plugins["boom_register"] = _info(name="boom_register")
    manager.app = object()
    assert manager.load_plugin("boom_register") is False
    assert manager.loaded == {}, \
        "the instance used to stay loaded, and a later load called it 'already loaded'"
    assert manager.get_plugin("boom_register") is None
    assert "Error loading plugin 'boom_register': the plugin's GUI hook exploded" \
        in capsys.readouterr().out


def test_a_register_failure_lets_the_plugin_clean_itself_up(monkeypatch):
    class RegisterBoom(GoodPlugin):
        cleaned = False

        def name(self):
            return "boom_register"

        def register(self, app):
            raise RuntimeError("the plugin's GUI hook exploded")

        def on_unload(self):
            RegisterBoom.cleaned = True

    install(monkeypatch, [EP("boom", RegisterBoom)])
    manager = pl.PluginManager()
    manager.plugins["boom_register"] = _info(name="boom_register")
    manager.app = object()
    manager.load_plugin("boom_register")
    assert RegisterBoom.cleaned, "on_load() ran, so on_unload() has to get its turn"


def test_a_cleanup_that_fails_after_a_failed_register_is_logged(caplog):
    class Both(GoodPlugin):
        def name(self):
            return "both"

        def register(self, app):
            raise RuntimeError("register exploded")

        def on_unload(self):
            raise RuntimeError("cleanup exploded")

    manager = pl.PluginManager()
    manager.app = object()          # register() only runs when an app exists
    info = _info(name="both")
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="register exploded"):
            manager._activate("both", Both, info)
    assert "its own cleanup failed too" in caplog.text
    assert "register exploded" not in caplog.text, "the cause is printed, not logged twice"


# ── unloading ───────────────────────────────────────────────────────────────

def test_unload_calls_the_hook_and_forgets_the_instance(monkeypatch, capsys):
    install(monkeypatch, [EP("good", GoodPlugin)])
    manager = pl.PluginManager()
    manager.plugins["good"] = _info()
    manager.load_plugin("good")
    manager.unload_plugin("good")
    assert GoodPlugin.unloaded_calls == 1
    assert manager.loaded == {}
    assert "Unloaded plugin: good" in capsys.readouterr().out


def test_unloading_something_not_loaded_says_so(capsys):
    manager = pl.PluginManager()
    manager.unload_plugin("ghost")
    assert "Plugin 'ghost' is not loaded." in capsys.readouterr().out


def test_a_failing_on_unload_still_frees_the_plugin(monkeypatch, caplog):
    class UnloadBoom(GoodPlugin):
        def name(self):
            return "unload_boom"

        def on_unload(self):
            raise RuntimeError("cleanup exploded")

    install(monkeypatch, [EP("u", UnloadBoom)])
    manager = pl.PluginManager()
    manager.plugins["unload_boom"] = _info(name="unload_boom")
    manager.load_plugin("unload_boom")
    with caplog.at_level(logging.WARNING):
        manager.unload_plugin("unload_boom")
    assert manager.loaded == {}, \
        "the exception used to escape and leave the plugin impossible to unload"
    assert "raised during on_unload() and was forgotten anyway" in caplog.text
    assert "cleanup exploded" in caplog.text


def test_a_plugin_can_be_loaded_again_after_its_cleanup_failed(monkeypatch):
    class UnloadBoom(GoodPlugin):
        def name(self):
            return "unload_boom"

        def on_unload(self):
            raise RuntimeError("cleanup exploded")

    install(monkeypatch, [EP("u", UnloadBoom)])
    manager = pl.PluginManager()
    manager.plugins["unload_boom"] = _info(name="unload_boom")
    manager.load_plugin("unload_boom")
    manager.unload_plugin("unload_boom")
    assert manager.load_plugin("unload_boom") is True


def test_a_distribution_with_two_entry_points_loads_the_one_that_was_asked_for(
        monkeypatch, capsys):
    """Both plugins come from one distribution, so the distribution-name fallback
    used to take whichever entry point came first and file it under the other
    name."""
    class Other(GoodPlugin):
        def name(self):
            return "other"

    install(monkeypatch, [EP("registered_under_a_key", GoodPlugin),
                          EP("other_key", Other)])
    manager = pl.PluginManager()
    manager.plugins["good"] = _info(name="good")
    manager.plugins["other"] = _info(name="other")
    assert manager.load_plugin("other") is True
    assert manager.get_plugin("other").name() == "other", \
        "the GoodPlugin entry point was the one being activated"
    assert manager.load_plugin("good") is True
    assert manager.get_plugin("good").name() == "good"


def test_a_distribution_fallback_never_activates_a_plugin_that_denies_being_it(
        monkeypatch, capsys):
    """The recorded distribution matched, but no class from it answers to the
    requested name after a reinstall — that is the 'discovered but unreachable'
    answer, not a wrong plugin."""
    class Unrelated(GoodPlugin):
        def name(self):
            return "unrelated"

    install(monkeypatch, [EP("whatever", Unrelated)])
    manager = pl.PluginManager()
    manager.plugins["good"] = _info(name="good")
    assert manager.load_plugin("good") is False
    assert "was discovered but no loadable implementation" in capsys.readouterr().out
    assert manager.loaded == {}


# ── load_all and set_app ────────────────────────────────────────────────────

def test_load_all_counts_only_the_plugins_that_actually_loaded(monkeypatch):
    install(monkeypatch, [EP("good", GoodPlugin)])
    manager = pl.PluginManager()
    manager.plugins["good"] = _info()
    manager.plugins["needy"] = _info(name="needy", deps=["totally_absent_pkg"])
    assert manager.load_all() == 1
    assert set(manager.loaded) == {"good"}


def test_load_all_skips_what_is_disabled(monkeypatch, capsys):
    install(monkeypatch, [EP("good", GoodPlugin)])
    manager = pl.PluginManager()
    manager.plugins["off"] = _info(name="off", enabled=False)
    manager.plugins["good"] = _info()
    assert manager.load_all() == 1
    assert "Plugin 'off' is disabled; skipping." in capsys.readouterr().out
    assert set(manager.loaded) == {"good"}


def test_load_all_on_an_empty_manager(monkeypatch):
    install(monkeypatch, [])
    assert pl.PluginManager().load_all() == 0


def test_set_app_re_registers_what_is_already_loaded(monkeypatch):
    install(monkeypatch, [EP("good", GoodPlugin)])
    manager = pl.PluginManager()
    manager.plugins["good"] = _info()
    manager.load_plugin("good")
    first = object()
    manager.set_app(first)
    assert manager.get_plugin("good").app is first
    second = object()
    manager.set_app(second)
    assert manager.get_plugin("good").app is second


def test_one_broken_re_registration_does_not_stop_the_others(monkeypatch, caplog):
    class BadRe(GoodPlugin):
        def name(self):
            return "bad"

        def register(self, app):
            if app is not None and getattr(app, "explode", False):
                raise RuntimeError("re-register exploded")

    install(monkeypatch, [EP("bad", BadRe), EP("good", GoodPlugin)])
    manager = pl.PluginManager()
    manager.plugins["bad"] = _info(name="bad")
    manager.plugins["good"] = _info()
    manager.load_plugin("bad")
    manager.load_plugin("good")

    class App:
        explode = True

    with caplog.at_level(logging.WARNING):
        manager.set_app(App())
    assert "failed to re-register with the new application" in caplog.text
    assert manager.get_plugin("good").app is not None
