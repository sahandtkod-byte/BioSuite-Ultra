"""``biosuite.plotting`` publishes its own functions and nothing else.

``biosuite/plotting/__init__.py`` is eleven ``from .x import *`` lines and none of
those modules declared ``__all__``, so a star import pulled in each module's own
imports too: ``numpy`` as ``np``, ``pandas`` as ``pd``, ``os``, ``matplotlib``'s
``plt``, ``plotly.graph_objects`` as ``go``, plus ``math``, ``random`` and
``HAS_PLOTLY``.  Measured: 157 attributes on the package, 8 of them foreign
modules or flags.  ``biosuite/core/__init__.py`` had exactly this bug and was
fixed by declaring its names; the same leak in a namespace a user is expected to
star-import is worse, because it silently rebinds ``os`` in the caller's file.
"""
import importlib
import os
import pathlib
import re
import types

import pytest

import biosuite.plotting as plotting

STARED = ['biological_plots', 'conservation_plots', 'genome_browser',
          'interactive_plots', 'math_plots', 'network_plots', 'plasmid_map',
          'sequence_viewer', 'specialized_plots', 'synteny', 'upset_plots']

#: Names that reached the package only because a module imported them.
FOREIGN = ['np', 'pd', 'os', 'plt', 'go', 'math', 'random', 'HAS_PLOTLY',
           'sys', 'json', 're', 'itertools', 'warnings', 'annotations',
           'matplotlib', 'collections', 'subprocess']


def _modules():
    return [importlib.import_module(f'biosuite.plotting.{name}') for name in STARED]


@pytest.mark.parametrize('name', FOREIGN)
def test_a_module_the_submodules_import_is_not_re_exported(name):
    assert not hasattr(plotting, name), f"{name} leaked out of a star import"


def test_no_attribute_of_the_package_is_a_foreign_module():
    """The submodules themselves are legitimate attributes of the package; the
    foreign ones are the modules a submodule imported."""
    leaked = [name for name, value in vars(plotting).items()
              if isinstance(value, types.ModuleType)
              and not name.startswith('__')
              and not getattr(value, '__name__', '').startswith('biosuite.plotting')]

    assert leaked == [], leaked


def test_every_star_imported_module_declares_its_public_names():
    missing = [module.__name__ for module in _modules()
               if getattr(module, '__all__', None) is None]

    assert missing == []


def test_every_declared_name_actually_exists_and_is_public():
    """A hand-written ``__all__`` that drifts is worse than none: the star import
    raises AttributeError at import time, in whichever module was edited last."""
    broken, private = [], []
    for module in _modules():
        for name in module.__all__:
            if not hasattr(module, name):
                broken.append(f'{module.__name__}.{name}')
            if name.startswith('_'):
                private.append(f'{module.__name__}.{name}')

    assert broken == []
    assert private == []


def test_a_star_import_brings_the_functions_and_nothing_else(tmp_path):
    # __file__ -> .../biosuite/plotting/__init__.py, so the repo is three levels up;
    # getting this wrong imports some *other* biosuite off the interpreter.
    repo = pathlib.Path(plotting.__file__).resolve().parents[2]
    script = tmp_path / 'star.py'
    script.write_text('from biosuite.plotting import *\n'
                      'print(sorted(n for n in dir() if not n.startswith("_")))\n',
                      encoding='utf-8')
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(repo), str(repo / 'audit_output' / '.probe_env')]))
    result = subprocess.run([sys.executable, str(script)], capture_output=True,
                            text=True, cwd=str(repo), env=env)
    assert result.returncode == 0, result.stderr
    names = eval(result.stdout.strip())  # the script prints a literal list of names

    assert 'volcano_plot' in names and 'sine_plot' in names
    assert 'os' not in names and 'np' not in names and 'HAS_PLOTLY' not in names
    assert 'matplotlib' not in names and 'pandas' not in names


def test_the_package_still_exports_every_module_s_own_definitions():
    """The guard against over-strict ``__all__`` lists: nothing that was
    reachable from the package before is now missing."""
    expected = set()
    for module in _modules():
        for node in module.__dict__.values():
            if isinstance(node, types.FunctionType) and node.__module__ == module.__name__ \
                    and not node.__name__.startswith('_'):
                expected.add(node.__name__)
    missing = sorted(name for name in expected if not hasattr(plotting, name))

    assert missing == []
    # 105 public plotting functions in the package, 13 of them in plot_api.py,
    # which __init__.py does not star-import: 92 have to be reachable here.
    assert len(expected) >= 92, f"only {len(expected)} public plotting functions found"


def test_the_re_exports_are_the_modules_own_objects():
    for module in _modules():
        for name in module.__all__:
            value = getattr(module, name)
            if isinstance(value, types.FunctionType):
                assert getattr(plotting, name) is value, f"{name} was shadowed"
