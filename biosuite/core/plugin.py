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
import os
import sys
import json
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
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
        pass

    @abstractmethod
    def version(self) -> str:
        """Return the plugin version."""
        pass

    @abstractmethod
    def description(self) -> str:
        """Return a brief description of the plugin."""
        pass

    @abstractmethod
    def author(self) -> str:
        """Return the plugin author name."""
        pass

    @abstractmethod
    def register(self, app):
        """Register plugin with the BioSuite application.

        Args:
            app: BioSuiteApp instance (GUI) or None for headless.
        """
        pass

    def dependencies(self) -> List[str]:
        """Return list of required pip packages. Override if needed."""
        return []

    def on_load(self):
        """Called after plugin is loaded. Override for initialization."""
        pass

    def on_unload(self):
        """Called before plugin is unloaded. Override for cleanup."""
        pass


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
        self.app = None
        self._config_path = os.path.join(
            os.path.expanduser('~'), '.biosuite', 'plugins.json'
        )

    def discover(self):
        """Discover all installed BioSuite plugins via entry points.

        Scans for packages that register 'biosuite.plugins' entry points.
        """
        discovered = []

        try:
            from importlib.metadata import entry_points
            eps = entry_points()
            # Python 3.12+ returns a SelectableGroups, older returns dict
            plugin_eps = eps.select(group='biosuite.plugins') if hasattr(eps, 'select') else eps.get('biosuite.plugins', [])
            for ep in plugin_eps:
                try:
                    plugin_class = ep.load()
                    if issubclass(plugin_class, BioSuitePlugin):
                        instance = plugin_class()
                        info = PluginInfo(
                            name=instance.name(),
                            version=instance.version(),
                            description=instance.description(),
                            author=instance.author(),
                            module_path=str(getattr(ep, 'dist', 'unknown')),
                            dependencies=instance.dependencies()
                        )
                        self.plugins[info.name] = info
                        discovered.append(info.name)
                except Exception as e:
                    print(f"Warning: Failed to load plugin {ep.name}: {e}")
        except ImportError:
            pass

        # Also check for local plugins in ~/.biosuite/plugins/
        local_dir = os.path.join(os.path.expanduser('~'), '.biosuite', 'plugins')
        if os.path.exists(local_dir):
            for item in os.listdir(local_dir):
                plugin_path = os.path.join(local_dir, item, '__init__.py')
                if os.path.exists(plugin_path):
                    try:
                        spec = importlib.util.spec_from_file_location(
                            f"biosuite_plugin_{item}", plugin_path
                        )
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        if hasattr(mod, 'Plugin'):
                            instance = mod.Plugin()
                            if isinstance(instance, BioSuitePlugin):
                                info = PluginInfo(
                                    name=instance.name(),
                                    version=instance.version(),
                                    description=instance.description(),
                                    author=instance.author(),
                                    module_path=plugin_path,
                                    dependencies=instance.dependencies()
                                )
                                self.plugins[info.name] = info
                                discovered.append(info.name)
                    except Exception as e:
                        print(f"Warning: Failed to load local plugin {item}: {e}")

        return discovered

    def list_plugins(self):
        """Print all discovered plugins."""
        if not self.plugins:
            print("No plugins found.")
            print("\nTo install plugins:")
            print("  pip install biosuite-plugin-<name>")
            print("\nTo create a plugin:")
            print("  See biosuite/core/plugin.py for documentation")
            return

        print(f"Found {len(self.plugins)} plugin(s):\n")
        print(f"{'Name':<25} {'Version':<10} {'Author':<20} {'Description'}")
        print("-" * 80)
        for name, info in self.plugins.items():
            loaded = " [loaded]" if name in self.loaded else ""
            print(f"{info.name:<25} {info.version:<10} {info.author:<20} {info.description}{loaded}")

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
            try:
                importlib.import_module(dep.split('>=')[0].split('==')[0])
            except ImportError:
                missing.append(dep)

        if missing:
            print(f"Plugin '{plugin_name}' missing dependencies: {', '.join(missing)}")
            print(f"Install with: pip install {' '.join(missing)}")
            return False

        # Find and load the plugin class
        try:
            from importlib.metadata import entry_points
            eps = entry_points()
            plugin_eps = eps.select(group='biosuite.plugins') if hasattr(eps, 'select') else eps.get('biosuite.plugins', [])
            for ep in plugin_eps:
                if ep.name == plugin_name or (hasattr(ep, 'dist') and
                    ep.dist and getattr(ep.dist, 'project_name', None) == info.module_path):
                    plugin_class = ep.load()
                    instance = plugin_class()
                    instance.on_load()
                    self.loaded[plugin_name] = instance
                    if self.app:
                        instance.register(self.app)
                    print(f"Loaded plugin: {plugin_name} v{info.version}")
                    return True
        except Exception as e:
            print(f"Error loading plugin '{plugin_name}': {e}")
            return False

        return False

    def load_all(self):
        """Load all discovered plugins."""
        loaded = 0
        for name in self.plugins:
            if self.load_plugin(name):
                loaded += 1
        return loaded

    def unload_plugin(self, plugin_name: str):
        """Unload a loaded plugin.

        Args:
            plugin_name: name of the plugin to unload.
        """
        if plugin_name in self.loaded:
            self.loaded[plugin_name].on_unload()
            del self.loaded[plugin_name]
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
        """
        self.app = app
        # Re-register loaded plugins with new app
        for name, plugin in self.loaded.items():
            plugin.register(app)

    def save_config(self):
        """Save plugin configuration (enabled/disabled state)."""
        config = {}
        for name, info in self.plugins.items():
            config[name] = {
                'enabled': info.enabled,
                'version': info.version,
            }

        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
        with open(self._config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_config(self):
        """Load plugin configuration."""
        if not os.path.exists(self._config_path):
            return

        try:
            with open(self._config_path) as f:
                config = json.load(f)
            for name, settings in config.items():
                if name in self.plugins:
                    self.plugins[name].enabled = settings.get('enabled', True)
        except (json.JSONDecodeError, OSError):
            pass

    def create_plugin_template(self, plugin_name: str, output_dir: str = '.'):
        """Create a template for a new BioSuite plugin.

        Args:
            plugin_name: name for the new plugin.
            output_dir: directory to create the plugin in.
        """
        plugin_dir = os.path.join(output_dir, f"biosuite-plugin-{plugin_name}")
        os.makedirs(plugin_dir, exist_ok=True)

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

        with open(os.path.join(plugin_dir, '__init__.py'), 'w') as f:
            f.write(init_content)

        # Create pyproject.toml
        toml_content = f'''[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "biosuite-plugin-{plugin_name}"
version = "0.1.0"
description = "BioSuite plugin: {plugin_name}"
requires-python = ">=3.9"
dependencies = [
    "biosuite>=4.0",
]

[project.entry-points."biosuite.plugins"]
{plugin_name} = "{plugin_name}:Plugin"
'''

        with open(os.path.join(plugin_dir, 'pyproject.toml'), 'w') as f:
            f.write(toml_content)

        # Create README.md
        readme_content = f'''# BioSuite Plugin: {plugin_name}

A plugin for BioSuite Ultra bioinformatics platform.

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
'''

        with open(os.path.join(plugin_dir, 'README.md'), 'w') as f:
            f.write(readme_content)

        print(f"Created plugin template at: {plugin_dir}")
        print(f"\nNext steps:")
        print(f"  1. Edit {plugin_dir}/__init__.py with your analysis code")
        print(f"  2. Test: pip install -e {plugin_dir}")
        print(f"  3. Publish: python -m twine dist/*")


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
