"""
Plugin architecture for BioSuite third-party extensions.

Allows community members to add new analysis modules without modifying
core BioSuite code. Plugins are pip-installable packages that register
themselves via entry points.

Usage:
    # Creating a plugin:
    # 1. Create a Python package named biosuite-plugin-<name>
    # 2. Define a class inheriting from BioSuitePlugin
    # 3. Register via entry_points in setup.py/pyproject.toml

    # In your plugin's pyproject.toml:
    # [project.entry-points."biosuite.plugins"]
    # my_plugin = "my_plugin:MyPlugin"

    # Using plugins in BioSuite:
    from biosuite.core.plugin import PluginManager
    pm = PluginManager()
    pm.discover()
    pm.list_plugins()
"""
import importlib
import importlib.util
import json
import keyword
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import __version__
from .log import get_logger

logger = get_logger(__name__)

# A plugin name becomes a directory, an importable module and a pip requirement,
# so it is restricted to what all three of those can hold: no path separators,
# no shell metacharacters, no leading digit.
PLUGIN_NAME_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_-]*\Z')

# Requirements the generated plugin template writes into its pyproject.  The
# floor tracks this package's own version rather than a remembered number.
TEMPLATE_REQUIRES_PYTHON = ">=3.10"
TEMPLATE_BIOSUITE_FLOOR = f">={__version__.split('.')[0]}.0"

# Requirement specifiers that may follow a dependency name in a plugin's
# dependencies() list.
_REQUIREMENT_SEPARATORS = re.compile(r'[<>=!~;\[\s]')


def _dist_name(ep) -> str:
    """The distribution an entry point came from, as load_plugin() wants to see it.

    ``str(ep.dist)`` is a repr such as ``"<PathDistribution ...>"``, and
    load_plugin() compares this field against ``ep.dist.project_name``; storing
    the repr left that fallback match permanently false.
    """
    dist = getattr(ep, "dist", None)
    return getattr(dist, "project_name", None) or "unknown"


def _plugin_entries():
    """The registered 'biosuite.plugins' entry points.

    ``importlib.metadata.entry_points()`` answers with an EntryPoints object on
    every Python this project supports (``requires-python`` is >=3.10 and CI runs
    3.10/3.11/3.12), so there is no dict-shaped legacy API to accommodate and no
    missing metadata module to survive.
    """
    from importlib.metadata import entry_points
    return entry_points().select(group='biosuite.plugins')


def _requirement_module(requirement: str) -> str:
    """The module a requirement string names, or '' when it names nothing usable.

    ``numpy>=1.24``, ``numpy~=1.26``, ``numpy[extra]>=1.0`` and
    ``numpy;python_version<'3.12'`` all mean "numpy": splitting on '>=' and '=='
    alone handed back the specifier as a module name, and every plugin that asked
    for a versioned dependency was refused as missing it.
    """
    name = requirement.strip()
    if not name:
        return ''
    match = _REQUIREMENT_SEPARATORS.search(name)
    if match:
        name = name[:match.start()]
    return name.strip()


def _pip_target(requirement: str) -> str:
    """Quote a requirement so the printed pip command does the printed thing.

    ``pip install numpy>=1.24`` in a shell redirects stdout to a file called
    ``=1.24``; the quotes are part of the instruction, not decoration.
    """
    return f'"{requirement}"' if requirement.strip() != _requirement_module(requirement) \
        else requirement.strip()


@dataclass
class PluginInfo:
    """Information about a loaded plugin.

    ``module_path`` is where the plugin came from, in the one form load_plugin()
    can compare: the distribution name for an entry-point plugin, the file path of
    ``~/.biosuite/plugins/<folder>/__init__.py`` for a local one.
    """
    name: str
    version: str
    description: str
    author: str
    module_path: str
    enabled: bool = True
    dependencies: list = field(default_factory=list)


class BioSuitePlugin(ABC):
    """Base class for all BioSuite plugins.

    To create a plugin:
    1. Create a class that inherits from BioSuitePlugin
    2. Implement all required methods
    3. Register via entry_points in your package's pyproject.toml

    Example:
        from biosuite.core.plugin import BioSuitePlugin

        class MyPlugin(BioSuitePlugin):
            def name(self): return "my-analysis"
            def version(self): return "1.0.0"
            def description(self): return "My custom analysis"
            def author(self): return "Your Name"
            def register(self, app):
                # Add your functions to the app
                app.add_analysis("my_analysis", self.my_function)
            def my_function(self, data):
                return {"result": "custom analysis"}
    """

    @abstractmethod
    def name(self) -> str:
        """Return the plugin name."""

    @abstractmethod
    def version(self) -> str:
        """Return the plugin version."""

    @abstractmethod
    def description(self) -> str:
        """Return a brief description of the plugin."""

    @abstractmethod
    def author(self) -> str:
        """Return the plugin author name."""

    @abstractmethod
    def register(self, app):
        """Register plugin with the BioSuite application.

        Args:
            app: the BioSuiteApp the manager was given through set_app().  The
                manager calls this only once an application exists — a headless
                load() leaves the plugin instantiated with on_load() run but
                never registered, which is why PluginManager.load_plugin()
                returns True without a GUI ever hearing from the plugin.
        """

    def dependencies(self) -> List[str]:
        """Return list of required pip packages. Override if needed."""
        return []

    def on_load(self):  # noqa: B027 - optional hook, deliberately not abstract
        """Called after plugin is loaded. Override for initialization."""
        return None

    def on_unload(self):  # noqa: B027 - optional hook, deliberately not abstract
        """Called before plugin is unloaded. Override for cleanup."""
        return None


class PluginManager:
    """Discover, load, and manage BioSuite plugins.

    Plugins are discovered via Python entry points under 'biosuite.plugins'.
    They can also be loaded manually from file paths.

    Usage:
        pm = PluginManager()
        pm.discover()  # Find all installed plugins
        pm.list_plugins()  # Show available plugins
        pm.load_plugin("my-plugin")  # Load specific plugin
    """

    def __init__(self):
        self.plugins: Dict[str, PluginInfo] = {}
        self.loaded: Dict[str, BioSuitePlugin] = {}
        # Local (~/.biosuite/plugins) plugin classes cannot be re-found through
        # entry points, so discovery has to hand the class to load_plugin().
        self._local_classes: Dict[str, type] = {}
        self.app = None
        self._config_path = os.path.join(
            os.path.expanduser('~'), '.biosuite', 'plugins.json'
        )

    def discover(self):
        """Discover installed BioSuite plugins and the local drop-in folders.

        Entry points under 'biosuite.plugins' are the pip-installed route;
        ``~/.biosuite/plugins/<folder>/__init__.py`` exposing ``Plugin`` is the
        un-installed route, and it shadows an installed plugin of the same name
        (which is logged, because the user has two copies and one winner).

        The enabled/disabled state saved in ``plugins.json`` is applied here, so
        that ``load_all()`` honours it without every caller having to remember a
        second step.
        """
        discovered = []

        for ep in _plugin_entries():
            try:
                plugin_class = ep.load()
            except Exception as e:
                logger.error("Failed to load plugin entry point %s: %s", ep.name, e)
                continue
            if not isinstance(plugin_class, type):
                logger.error("Plugin entry point %s loaded a %s, not a class: %r",
                             ep.name, type(plugin_class).__name__, plugin_class)
                continue
            if not issubclass(plugin_class, BioSuitePlugin):
                logger.error("Plugin entry point %s is %s, which does not inherit "
                             "from BioSuitePlugin, so it was skipped",
                             ep.name, plugin_class.__name__)
                continue
            self._register(plugin_class, _dist_name(ep), discovered,
                           source=f"entry point {ep.name}")

        # Also check for local plugins in ~/.biosuite/plugins/
        local_dir = os.path.join(os.path.expanduser('~'), '.biosuite', 'plugins')
        if os.path.exists(local_dir):
            for item in sorted(os.listdir(local_dir)):
                plugin_path = os.path.join(local_dir, item, '__init__.py')
                if os.path.exists(plugin_path):
                    try:
                        spec = importlib.util.spec_from_file_location(
                            f"biosuite_plugin_{item}", plugin_path
                        )
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        if not hasattr(mod, 'Plugin'):
                            logger.warning("Local plugin folder %s has no Plugin "
                                           "class, so it was skipped", item)
                            continue
                        if not isinstance(mod.Plugin, type) or \
                                not issubclass(mod.Plugin, BioSuitePlugin):
                            logger.warning("Local plugin folder %s defines Plugin "
                                           "as something that is not a "
                                           "BioSuitePlugin, so it was skipped", item)
                            continue
                        self._register(mod.Plugin, plugin_path, discovered,
                                       source=f"local folder {item}",
                                       remember_class=True)
                    except Exception as e:
                        logger.warning("Failed to load local plugin %s: %s", item, e)

        self.load_config()
        return discovered

    def _register(self, plugin_class, module_path, discovered, source,
                  remember_class=False):
        """Add one discovered plugin, or say why it is not usable."""
        try:
            instance = plugin_class()
            name = instance.name()
            info = PluginInfo(
                name=name,
                version=instance.version(),
                description=instance.description(),
                author=instance.author(),
                module_path=module_path,
                dependencies=self._dependency_list(instance, source)
            )
        except Exception as e:
            logger.error("Failed to instantiate plugin from %s: %s", source, e)
            return
        if not isinstance(name, str) or not name.strip():
            logger.error("Plugin from %s names itself %r; a plugin needs a non-empty "
                         "string name to be loadable, so it was skipped",
                         source, name)
            return
        existing = self.plugins.get(name)
        if existing is not None:
            logger.warning("Plugin %r from %s shadows the earlier copy from %s; "
                           "the later one is what will load",
                           name, source, existing.module_path)
        self.plugins[name] = info
        if remember_class:
            self._local_classes[name] = plugin_class
        if name not in discovered:
            discovered.append(name)

    @staticmethod
    def _dependency_list(instance, source) -> list:
        """Dependencies as a list of requirement strings.

        A plugin returning the bare string 'numpy' used to be read one character
        at a time by load_plugin(), which then reported n, u, m, p and y as
        missing packages.
        """
        deps = instance.dependencies()
        if isinstance(deps, str):
            deps = [deps]
        if not isinstance(deps, (list, tuple)):
            logger.warning("Plugin from %s returned dependencies() as %s, which is "
                           "not a list of requirement strings; ignoring it",
                           source, type(deps).__name__)
            return []
        clean = []
        for dep in deps:
            if not isinstance(dep, str) or not dep.strip():
                logger.warning("Plugin from %s listed dependency %r, which is not a "
                               "requirement string; ignoring it", source, dep)
                continue
            clean.append(dep.strip())
        return clean

    def list_plugins(self):
        """Print all discovered plugins.

        The enabled flag is printed because load_all() acts on it; a table that
        hides it lists a plugin the manager will refuse to load as just another
        plugin.
        """
        if not self.plugins:
            print("No plugins found.")
            print("\nTo install plugins:")
            print("  pip install biosuite-plugin-<name>")
            print("\nTo create a plugin:")
            print("  See biosuite/core/plugin.py for documentation")
            return

        print(f"Found {len(self.plugins)} plugin(s):\n")
        print(f"{'Name':<25} {'Version':<10} {'Author':<20} Description")
        print("-" * 80)
        for name, info in self.plugins.items():
            state = ""
            if name in self.loaded:
                state = " [loaded]"
            if not info.enabled:
                state += " [disabled]"
            print(f"{info.name:<25} {info.version:<10} {info.author:<20} "
                  f"{info.description}{state}")

    def load_plugin(self, plugin_name: str):
        """Load a specific plugin by name.

        Args:
            plugin_name: name of the plugin to load.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if plugin_name in self.loaded:
            print(f"Plugin '{plugin_name}' is already loaded.")
            return True

        if plugin_name not in self.plugins:
            print(f"Plugin '{plugin_name}' not found. Run discover() first.")
            return False

        info = self.plugins[plugin_name]

        # Check dependencies
        missing = []
        for dep in info.dependencies:
            module = _requirement_module(dep)
            if not module:
                print(f"Plugin '{plugin_name}' lists an empty dependency; ignoring it.")
                continue
            try:
                importlib.import_module(module)
            except ImportError:
                missing.append(dep)
            except Exception as e:
                # A package that is installed but blows up on import is not a
                # plugin's dependency the manager can rely on either, and the
                # reason is the only clue the user will get.
                print(f"Plugin '{plugin_name}' dependency '{dep}' could not be "
                      f"imported ({type(e).__name__}: {e}); treating it as missing.")
                missing.append(dep)

        if missing:
            print(f"Plugin '{plugin_name}' missing dependencies: {', '.join(missing)}")
            print(f"Install with: pip install "
                  f"{' '.join(_pip_target(dep) for dep in missing)}")
            print("The check imports each dependency by the name it is written "
                  "with, so a distribution whose module is named differently "
                  "('python-dateutil' vs 'dateutil') has to list the module name.")
            return False

        # Find and load the plugin class
        try:
            local_class = self._local_classes.get(plugin_name)
            if local_class is not None:
                return self._activate(plugin_name, local_class, info)

            same_distribution = []
            for ep in _plugin_entries():
                if ep.name == plugin_name:
                    return self._activate(plugin_name, ep.load(), info)
                if getattr(getattr(ep, "dist", None), "project_name",
                           None) == info.module_path:
                    same_distribution.append(ep)
            # No entry point is registered under this name, so a plugin that came
            # from the recorded distribution is the next best guess — but only if
            # the class itself answers to the requested name.  Two entry points
            # from one distribution used to be interchangeable here, which loaded
            # the wrong plugin under the right name.
            for ep in same_distribution:
                candidate = ep.load()
                # A class that cannot be constructed is a load failure with a
                # reason, not a non-match: the reason is what the caller needs.
                if candidate().name() == plugin_name:
                    return self._activate(plugin_name, candidate, info)
        except Exception as e:
            print(f"Error loading plugin '{plugin_name}': {e}")
            return False

        # No entry point (and no local class) matched: without this the caller
        # only saw a bare False after discover() had proudly listed the plugin.
        print(f"Plugin '{plugin_name}' was discovered but no loadable "
              f"implementation was found for it.")
        return False

    def _activate(self, plugin_name: str, plugin_class, info: PluginInfo) -> bool:
        """Instantiate a discovered plugin class and register it with the app.

        The instance is remembered only once every step that can fail has
        succeeded: inserting it before ``register()`` left a half-loaded plugin
        in ``self.loaded``, which the next ``load_plugin()`` called "already
        loaded" and ``get_plugin()`` handed out as if it were working.
        """
        instance = plugin_class()
        instance.on_load()
        try:
            if self.app is not None:
                instance.register(self.app)
        except Exception:
            try:
                instance.on_unload()
            except Exception as cleanup_error:
                logger.warning("Plugin '%s' failed to register and its own cleanup "
                               "failed too: %s", plugin_name, cleanup_error)
            raise
        self.loaded[plugin_name] = instance
        print(f"Loaded plugin: {plugin_name} v{info.version}")
        return True

    def load_all(self):
        """Load every discovered plugin that is not disabled in the config."""
        loaded = 0
        for name, info in self.plugins.items():
            # ``enabled`` is the state save_config()/load_config() persist; a
            # plugin disabled there used to be loaded anyway.
            if not info.enabled:
                print(f"Plugin '{name}' is disabled; skipping.")
                continue
            if self.load_plugin(name):
                loaded += 1
        return loaded

    def unload_plugin(self, plugin_name: str):
        """Unload a loaded plugin.

        Args:
            plugin_name: name of the plugin to unload.

        The plugin's own cleanup is allowed to fail: the instance is going away
        either way, and an exception escaping here used to leave it in
        ``self.loaded`` where no later call could ever unload it again.
        """
        if plugin_name in self.loaded:
            instance = self.loaded.pop(plugin_name)
            try:
                instance.on_unload()
            except Exception as e:
                logger.warning("Plugin '%s' raised during on_unload() and was "
                               "forgotten anyway: %s", plugin_name, e)
            print(f"Unloaded plugin: {plugin_name}")
        else:
            print(f"Plugin '{plugin_name}' is not loaded.")

    def get_plugin(self, plugin_name: str) -> Optional[BioSuitePlugin]:
        """Get a loaded plugin instance.

        Args:
            plugin_name: name of the plugin.

        Returns:
            Plugin instance or None.
        """
        return self.loaded.get(plugin_name)

    def set_app(self, app):
        """Set the BioSuite application instance for plugin registration.

        Args:
            app: BioSuiteApp instance.

        One plugin's broken ``register()`` must not stop the others from being
        told about the new application.
        """
        self.app = app
        # Re-register loaded plugins with new app
        for name, plugin in self.loaded.items():
            try:
                plugin.register(app)
            except Exception as e:
                logger.warning("Plugin '%s' failed to re-register with the new "
                               "application: %s", name, e)

    def save_config(self) -> bool:
        """Save plugin configuration (enabled/disabled state).

        Returns:
            True when the file was written; False, with the reason logged, when
            it was not — a state the user believes is saved is a state that must
            be known to be saved.
        """
        config = {}
        for name, info in self.plugins.items():
            config[name] = {
                'enabled': info.enabled,
                'version': info.version,
            }

        try:
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            with open(self._config_path, 'w', encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except OSError as e:
            logger.warning("Could not write plugin config %s: %s",
                           self._config_path, e)
            return False
        return True

    def load_config(self):
        """Apply the saved enable/disable state to the discovered plugins.

        A hand-edited plugins.json is ordinary input: a value that is not a
        mapping for one plugin, or an 'enabled' that is not a bool, is reported
        and skipped instead of raising out of the loader or turning the string
        "false" into a truthy enabled state.
        """
        if not os.path.exists(self._config_path):
            return

        try:
            with open(self._config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # Swallowing this made a corrupt plugins.json indistinguishable
            # from "no config yet": the user's enable/disable state silently
            # stopped being applied with no clue why.
            logger.warning("Could not read plugin config %s: %s",
                           self._config_path, e)
            return

        if not isinstance(config, dict):
            logger.warning("Plugin config %s holds a %s, not a mapping of plugin "
                           "name to settings; nothing was applied",
                           self._config_path, type(config).__name__)
            return

        for name, settings in config.items():
            if name not in self.plugins:
                continue
            if not isinstance(settings, dict):
                logger.warning("Plugin config entry %r holds %r, not a mapping of "
                               "settings; it was ignored", name, settings)
                continue
            if 'enabled' not in settings:
                continue
            enabled = settings['enabled']
            if not isinstance(enabled, bool):
                logger.warning("Plugin %r is saved as enabled=%r, which is not "
                               "true or false; keeping it as enabled=%s",
                               name, enabled, self.plugins[name].enabled)
                continue
            self.plugins[name].enabled = enabled

    def create_plugin_template(self, plugin_name: str, output_dir: str = '.'):
        """Create a template for a new BioSuite plugin.

        Args:
            plugin_name: name for the new plugin.  It has to work as a directory
                name, as an importable module name and as part of a shell
                command, so anything outside ``PLUGIN_NAME_PATTERN`` is refused
                rather than written out.
            output_dir: directory to create the plugin in.

        The package is written as ``biosuite-plugin-<name>/<module>/__init__.py``
        with the entry point aimed at ``<module>:Plugin``.  The earlier layout put
        ``__init__.py`` straight in the dashed project folder and pointed the
        entry point at a module name that could never be imported, so the
        generated plugin failed at ``ep.load()`` with ModuleNotFoundError.
        """
        if not isinstance(plugin_name, str) or not PLUGIN_NAME_PATTERN.match(plugin_name):
            raise ValueError(
                f"plugin_name must start with a letter or underscore and continue "
                f"with letters, digits, '-' or '_'; got {plugin_name!r}")

        module = plugin_name.replace('-', '_')
        if keyword.iskeyword(module):
            # 'class' passes the pattern and is still not an importable module
            # name, and the entry point the template writes would never load.
            raise ValueError(
                f"plugin_name {plugin_name!r} would become the Python keyword "
                f"{module!r} as a module name; choose another name")

        plugin_dir = os.path.join(output_dir, f"biosuite-plugin-{plugin_name}")
        package_dir = os.path.join(plugin_dir, module)
        os.makedirs(package_dir, exist_ok=True)

        # Create __init__.py
        init_content = f'''"""
BioSuite Plugin: {plugin_name}
"""
from biosuite.core.plugin import BioSuitePlugin


class Plugin(BioSuitePlugin):
    """BioSuite plugin for {plugin_name}."""

    def name(self) -> str:
        return "{plugin_name}"

    def version(self) -> str:
        return "0.1.0"

    def description(self) -> str:
        return "A BioSuite plugin for {plugin_name} analysis"

    def author(self) -> str:
        return "Your Name"

    def dependencies(self):
        return []  # Add pip packages here, e.g., ["numpy>=1.24"]

    def register(self, app):
        """Register plugin with BioSuite."""
        # Add your analysis functions here
        pass

    def my_analysis(self, data):
        """Example analysis function."""
        return {{"result": "analysis complete", "input": data}}
'''

        with open(os.path.join(package_dir, '__init__.py'), 'w', encoding="utf-8") as f:
            f.write(init_content)

        # Create pyproject.toml
        toml_content = f'''[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "biosuite-plugin-{plugin_name}"
version = "0.1.0"
description = "BioSuite plugin: {plugin_name}"
requires-python = "{TEMPLATE_REQUIRES_PYTHON}"
dependencies = [
    "biosuite{TEMPLATE_BIOSUITE_FLOOR}",
]

[project.entry-points."biosuite.plugins"]
{plugin_name} = "{module}:Plugin"

[tool.setuptools]
packages = ["{module}"]
'''

        with open(os.path.join(plugin_dir, 'pyproject.toml'), 'w', encoding="utf-8") as f:
            f.write(toml_content)

        # Create README.md
        readme_content = f'''# BioSuite Plugin: {plugin_name}

A plugin for BioSuite Ultra bioinformatics platform.

## Layout

```
biosuite-plugin-{plugin_name}/
    pyproject.toml
    README.md
    {module}/
        __init__.py      # defines class Plugin
```

## Installation

```bash
pip install .
```

## Usage

```python
from biosuite.core.plugin import PluginManager

pm = PluginManager()
pm.discover()
pm.load_plugin("{plugin_name}")
```

## Without installing it

Copy the `{module}` folder into `~/.biosuite/plugins/{module}` and BioSuite's
plugin manager will discover it as a local plugin.
'''

        with open(os.path.join(plugin_dir, 'README.md'), 'w', encoding="utf-8") as f:
            f.write(readme_content)

        print(f"Created plugin template at: {plugin_dir}")
        print("\nNext steps:")
        print(f"  1. Edit {plugin_dir}/{module}/__init__.py with your analysis code")
        print(f'  2. Test: pip install -e "{plugin_dir}"')
        print(f'  3. Or drop "{module}" into ~/.biosuite/plugins/ to use it '
              f"without installing")
        print("  4. Publish: python -m twine upload dist/*")


# ── Example Plugins (built-in) ──────────────────────────────────────────────

class ExamplePlugin(BioSuitePlugin):
    """Example plugin demonstrating the plugin API."""

    def name(self) -> str:
        return "example"

    def version(self) -> str:
        return "1.0.0"

    def description(self) -> str:
        return "Example plugin for demonstration"

    def author(self) -> str:
        return "BioSuite Team"

    def register(self, app):
        """Register example analysis with BioSuite."""
        pass

    def reverse_complement_demo(self, seq):
        """Demo function using BioSuite core."""
        from biosuite.core.sequence import reverse_complement
        return reverse_complement(seq)


# ── Global plugin manager ────────────────────────────────────────────────────

_default_manager = None

def get_plugin_manager():
    """Get or create the global plugin manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = PluginManager()
    return _default_manager
