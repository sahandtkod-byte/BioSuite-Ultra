"""``biosuite/gui/__init__.py``: the lazy entry point a GUI launch starts from.

``from biosuite.gui import BioSuiteApp`` runs the package's ``__getattr__``, which
imports the real window module only at the point of use, turns an unknown name
into an AttributeError instead of a silent ``None``, and keeps Tk out of the
process until the app is actually asked for.  The last promise can only be
observed in a clean interpreter, so one test runs one.
"""
import subprocess
import sys
from pathlib import Path

import pytest

import biosuite.gui

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_asking_for_the_app_hands_back_the_window_class_it_wraps(gui_app):
    from biosuite.gui import BioSuiteApp
    import biosuite.gui.main_window as main_window

    assert BioSuiteApp is main_window.BioSuiteApp
    assert isinstance(gui_app, BioSuiteApp)
    assert biosuite.gui.BioSuiteApp is BioSuiteApp, "the lazy import must answer twice"


def test_an_unknown_name_is_reported_as_missing_rather_than_none():
    with pytest.raises(AttributeError) as excinfo:
        biosuite.gui.BioSuiteSplashScreenIsNotExported

    message = str(excinfo.value)
    assert "BioSuiteSplashScreenIsNotExported" in message
    assert "biosuite.gui" in message


def test_the_app_is_advertised_before_the_package_has_imported_it():
    assert "BioSuiteApp" not in vars(biosuite.gui)
    assert dir(biosuite.gui) == ["BioSuiteApp"]


def test_a_broken_gui_import_names_the_bindings_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "biosuite.gui.main_window", None)
    with pytest.raises(ImportError) as excinfo:
        biosuite.gui.BioSuiteApp

    message = str(excinfo.value)
    assert "tkinter" in message and "customtkinter" in message
    assert "python3-tk" in message
    assert "Original error" in message
    assert excinfo.value.__cause__ is not None


def test_importing_the_package_headlessly_leaves_tk_out_of_the_process():
    script = "\n".join([
        "import sys, biosuite, biosuite.gui, biosuite.gui.text_parsing",
        "print(any(m == 'tkinter' or m.startswith('tkinter.') for m in sys.modules))",
        "print(any(m == 'customtkinter' or m.startswith('customtkinter.') for m in sys.modules))",
        f"print(biosuite.__file__.startswith({str(REPO_ROOT)!r}))",
    ])

    result = subprocess.run([sys.executable, "-c", script], cwd=str(REPO_ROOT),
                            capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-3:] == ["False", "False", "True"], result.stdout
