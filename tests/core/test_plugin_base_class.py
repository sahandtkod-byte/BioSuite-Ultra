"""The contract every BioSuite plugin inherits.

`BioSuitePlugin` is what a third-party package has to satisfy, so its abstract
methods, its optional hooks and the lifecycle the manager actually runs are the
product's public interface to other people's code.
"""
import pytest

from biosuite.core import plugin as pl


def test_the_five_required_methods_must_be_implemented():
    class Half(pl.BioSuitePlugin):
        def name(self):
            return "half"

    with pytest.raises(TypeError, match="abstract"):
        Half()


@pytest.mark.parametrize("missing", ["name", "version", "description", "author",
                                     "register"])
def test_each_required_method_is_required(missing):
    """Dropping any one of the five must be what stops the class from
    instantiating — a contract with a silently optional member is no contract."""
    body = {method: (lambda self, _m=method: _m) for method in
            ("name", "version", "description", "author", "register")
            if method != missing}
    cls = type("Incomplete", (pl.BioSuitePlugin,), body)
    with pytest.raises(TypeError, match="abstract"):
        cls()


def test_a_complete_plugin_instantiates_and_answers_the_metadata():
    class Full(pl.BioSuitePlugin):
        def name(self):
            return "full"

        def version(self):
            return "1.2"

        def description(self):
            return "a complete plugin"

        def author(self):
            return "Tester"

        def register(self, app):
            self.app = app

    plugin = Full()
    assert (plugin.name(), plugin.version(), plugin.description(), plugin.author()) \
        == ("full", "1.2", "a complete plugin", "Tester")


def test_the_optional_hooks_do_nothing_by_default():
    class Bare(pl.BioSuitePlugin):
        def name(self):
            return "bare"

        def version(self):
            return "1"

        def description(self):
            return "d"

        def author(self):
            return "a"

        def register(self, app):
            pass

    plugin = Bare()
    assert plugin.dependencies() == []
    assert plugin.on_load() is None
    assert plugin.on_unload() is None


def test_a_subclass_may_override_only_the_hooks_it_cares_about():
    class Overrides(pl.BioSuitePlugin):
        def name(self):
            return "o"

        def version(self):
            return "1"

        def description(self):
            return "d"

        def author(self):
            return "a"

        def register(self, app):
            pass

        def dependencies(self):
            return ["numpy>=1.24"]

        def on_load(self):
            return "loaded"

    assert Overrides().dependencies() == ["numpy>=1.24"]
    assert Overrides().on_load() == "loaded"
    assert Overrides().on_unload() is None


def test_register_is_documented_for_the_app_it_will_actually_receive():
    """The docstring used to promise None for a headless run; the manager only
    calls register() once an application has been set, and the docstring now says
    which of the two is true."""
    doc = pl.BioSuitePlugin.register.__doc__
    assert "set_app" in doc
    assert "only once an application exists" in doc
    assert "None for headless" not in doc
