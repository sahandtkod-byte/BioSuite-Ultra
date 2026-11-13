"""Every biosuite module logs through the central get_logger factory.

Seventeen modules used to create their loggers with a bare
``logging.getLogger(__name__)``.  The call is correct Python but skips
:func:`biosuite.core.log.get_logger`, whose side effect is the one-time
configuration of the ``biosuite`` root logger (console handler, level,
optional file sink).  A module that skips it logs into an unconfigured
root: importing ``biosuite.api.config`` alone, for example, dropped its
credential warnings on the floor.  The factory call also makes every
module logger a child of ``biosuite`` so the file sink sees them.

This file drives one subprocess that imports the seventeen modules with
nothing else imported first — in-process assertions would prove nothing,
because conftest already configures logging by the time a test runs.
"""
import json
import logging
import os
import pathlib
import subprocess
import sys

import biosuite.core.log

MODULES = [
    'biosuite.api',
    'biosuite.api.config',
    'biosuite.api.auth',
    'biosuite.api.security',
    'biosuite.api.server',
    'biosuite.cli.menu',
    'biosuite.core.databases',
    'biosuite.core.file_formats',
    'biosuite.core.go_browser',
    'biosuite.core.md_simulation',
    'biosuite.core.metagenomics',
    'biosuite.core.metabolism',
    'biosuite.core.ml_phylogeny',
    'biosuite.core.pathway_viz',
    'biosuite.plotting._demo_guard',
    'biosuite.plotting.biological_plots',
    'biosuite.plotting.plot_api',
]

PROBE = json.dumps(MODULES)

_child = '''
import importlib, json, logging, sys
report = {}
for name in json.loads(sys.argv[1]):
    module = importlib.import_module(name)
    report[name] = {
        "module_logger_is_named_logger": getattr(module, "logger", None)
            is logging.getLogger(name),
        "does_not_bind_logging": "logging" not in vars(module),
    }
root = logging.getLogger("biosuite")
print("REPORT=" + json.dumps({
    "modules": report,
    "root_handler_count": len(root.handlers),
}))
'''


def _repo():
    return pathlib.Path(biosuite.core.log.__file__).resolve().parents[2]


def _run_child():
    repo = _repo()
    env = dict(os.environ,
               PYTHONPATH=str(repo) + os.pathsep + str(repo / 'audit_output' / '.probe_env'),
               PYTHONDONTWRITEBYTECODE='1',
               MPLBACKEND='Agg')
    return subprocess.run([sys.executable, '-c', _child, PROBE],
                          capture_output=True, text=True, env=env, cwd=str(repo))


def _report():
    result = _run_child()
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith('REPORT=')]
    assert len(lines) == 1, f"expected one REPORT line, got {lines}"
    return json.loads(lines[0].removeprefix('REPORT='))


def test_a_bare_interpreter_that_imports_a_module_gets_a_configured_root_logger():
    report = _report()
    assert report['root_handler_count'] >= 1, \
        "get_logger never configured the biosuite root logger"


def test_every_module_uses_get_logger_and_is_a_child_of_the_biosuite_root():
    report = _report()
    bad = {name: facts for name, facts in report['modules'].items()
           if not facts['module_logger_is_named_logger']}
    assert bad == {}, \
        "module.logger is not logging.getLogger(module.__name__) for: " + str(bad)


def test_no_module_carries_its_own_logging_import():
    report = _report()
    bound = {name: facts for name, facts in report['modules'].items()
             if not facts['does_not_bind_logging']}
    assert bound == {}, \
        "these modules still bind a private 'logging' name: " + str(bound)
