"""Lightweight CLI operations never load the plotting stack (CQ-07).

``run.py`` used to import ``biological_plots`` and apply the theme before
dispatching anything, so ``--help``, config inspection and single commands
like ``biosuite gc`` all paid a full Matplotlib import.  The plotting
imports now live in the branches that draw, and these subprocess tests pin
the split: a bare import, ``--help``, config inspection and a plain command
run with no matplotlib in ``sys.modules``, while a plotting command still
reaches and executes the real plotting functionality.

Every case runs in a fresh interpreter — in-process assertions prove
nothing, because importing any test module has already loaded matplotlib.
"""
import os
import pathlib
import subprocess
import sys

import biosuite.cli.menu


def _repo():
    return pathlib.Path(biosuite.cli.menu.__file__).resolve().parents[2]


def _env(tmp_path=None):
    repo = _repo()
    env = dict(os.environ,
               PYTHONPATH=str(repo) + os.pathsep + str(repo / 'audit_output' / '.probe_env'),
               PYTHONDONTWRITEBYTECODE='1',
               MPLBACKEND='Agg')
    return env


def _run(code, cwd=None):
    return subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, env=_env(), cwd=str(cwd or _repo()))


def test_help_still_works_and_exits_zero():
    result = subprocess.run([sys.executable, 'run.py', '--help'],
                            capture_output=True, text=True, env=_env(), cwd=str(_repo()))
    assert result.returncode == 0, result.stderr
    assert 'usage' in result.stdout.lower()


def test_importing_run_and_the_menu_loads_no_matplotlib():
    result = _run("import sys\n"
                  "import run\n"
                  "import biosuite.cli.menu\n"
                  "assert 'matplotlib' not in sys.modules, \\\n"
                  "    [m for m in sys.modules if 'matplot' in m]\n")
    assert result.returncode == 0, result.stderr


def test_config_inspection_loads_no_matplotlib():
    result = _run("import sys\n"
                  "from biosuite.core.utils import config\n"
                  "assert config['theme']\n"
                  "assert 'matplotlib' not in sys.modules, \\\n"
                  "    [m for m in sys.modules if 'matplot' in m]\n"
                  "print(config['theme'])\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_a_basic_command_runs_without_matplotlib():
    result = _run("import sys\n"
                  "sys.argv = ['biosuite', 'gc', 'ATCGATCG']\n"
                  "from biosuite.cli.menu import main_cli\n"
                  "code = main_cli()\n"
                  "assert code == 0, code\n"
                  "assert 'matplotlib' not in sys.modules, \\\n"
                  "    [m for m in sys.modules if 'matplot' in m]\n")
    assert result.returncode == 0, result.stderr
    assert '50' in result.stdout, 'ATCGATCG is 50% GC; the command must have run'


def test_a_plotting_command_still_draws(tmp_path):
    data = tmp_path / 'snps.csv'
    data.write_text('chrom,pos,p_value\n'
                    '1,100,1e-4\n1,200,3e-6\n2,300,1e-2\n2,400,5e-9\n',
                    encoding='utf-8')
    code = ("import sys\n"
            "sys.argv = ['biosuite', 'manhattan', r'" + str(data) + "']\n"
            "from biosuite.cli.menu import main_cli\n"
            "code = main_cli()\n"
            "assert code == 0, code\n"
            "assert 'matplotlib' in sys.modules, 'the plot path must load matplotlib'\n")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, env=_env(), cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'manhattan.png').exists(), result.stdout
