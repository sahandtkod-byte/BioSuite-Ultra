"""``biosuite.notebook`` without IPython, and without ipywidgets.

Two ``try: import`` guards decide which halves of the module exist, so a user who
installed the package without the ``notebook`` extra still gets a module that
imports and still offers the ``quick_*`` helpers.  The guards are decided while the
module body runs, so testing the other side means executing the module again with
the dependency hidden from the import system.

It is executed into a *fresh* module object rather than through ``importlib.reload``:
a reload keeps whatever the previous import defined, so a class the guard skipped
would still be visible and the assertions below would be measuring the wrong
module.  It also leaves the shared ``biosuite.notebook`` untouched for the rest of
the session.
"""
import contextlib
import importlib.util
import sys

import pandas as pd
import pytest

import biosuite.notebook as nbmod

#: Names defined inside one of the two optional-dependency guards.
GUARDED_NAMES = ("BioSuiteMagics", "load_ipython_extension", "SequenceAnalyzer",
                 "AlignmentViewer", "PlotExplorer")

#: What each guarded name needs, mirroring the module's own _OPTIONAL_EXPORTS map.
GUARD_REQUIREMENTS = {
    "BioSuiteMagics": "IPython",
    "load_ipython_extension": "IPython",
    "SequenceAnalyzer": "ipywidgets",
    "AlignmentViewer": "ipywidgets",
    "PlotExplorer": "ipywidgets",
}

#: The surface available on an install without either optional group.
BARE_EXPORTS = {"HAS_IPYTHON", "HAS_WIDGETS", "quick_gc", "quick_translate",
                "quick_align", "quick_blast"}


def absent_because(module, name):
    """Assert a guarded name is absent *and* names the dependency that provides it.

    ``hasattr`` is not usable here: the module's PEP 562 ``__getattr__`` answers a
    missing optional export with an informative ImportError (so a star import on a
    reduced install fails with the fix, not a bare "cannot import name"), and
    ``hasattr`` only catches AttributeError.
    """
    with pytest.raises(ImportError) as excinfo:
        getattr(module, name)
    message = str(excinfo.value)
    assert name in message
    assert GUARD_REQUIREMENTS[name] in message


def star_exported(module):
    """What ``from <module> import *`` actually puts in a namespace."""
    namespace = {}
    exec(f"from {module.__name__} import *", namespace)
    return {name for name in namespace if not name.startswith("__")}


@contextlib.contextmanager
def _executed_without(*packages):
    """Run the notebook module with ``packages`` hidden; yield the resulting module.

    A finder alone is not enough: both packages are already in ``sys.modules`` by
    the time this runs, and an import never consults a finder for a module that is
    already there.  The cache entries are parked for the duration and put back
    afterwards.
    """
    for package in packages:
        pytest.importorskip(package)      # hiding something absent proves nothing

    class HiddenFinder:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in packages:
                raise ImportError(f"{name} is hidden by this test")
            return None

    finder = HiddenFinder()
    parked = {name: module for name, module in sys.modules.items()
              if name.split(".")[0] in packages}
    for name in parked:
        del sys.modules[name]
    sys.meta_path.insert(0, finder)
    alias = "notebook_without_" + "_and_".join(packages)
    spec = importlib.util.spec_from_file_location(alias, nbmod.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(parked)
        sys.modules.pop(alias, None)


def test_the_module_still_imports_without_ipython():
    with _executed_without("IPython") as module:
        assert module.HAS_IPYTHON is False
        absent_because(module, "BioSuiteMagics")
        absent_because(module, "load_ipython_extension")
        assert module.quick_gc("ATCGATCG") == pytest.approx(50.0)


def test_the_module_keeps_the_magics_without_ipywidgets():
    with _executed_without("ipywidgets") as module:
        assert module.HAS_WIDGETS is False
        absent_because(module, "SequenceAnalyzer")
        absent_because(module, "AlignmentViewer")
        absent_because(module, "PlotExplorer")
        assert hasattr(module, "BioSuiteMagics")
        assert module.quick_align("ATCG", "ATCG")[:2] == ("ATCG", "ATCG")


def test_biostats_can_display_a_frame_without_ipywidgets():
    """The regression: ``display`` is an IPython name that sat in the widgets guard.

    ``%biostats`` on a DataFrame printed the shape and then died with
    ``NameError: name 'display' is not defined`` on any install that had IPython
    but not ipywidgets, which is the common case outside Jupyter.
    """
    pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell  # noqa: E402

    with _executed_without("ipywidgets") as module:
        shell = InteractiveShell.instance()
        shell.user_ns["frame"] = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        magics = module.BioSuiteMagics(shell=shell)
        try:
            magics.biostats("frame")
        finally:
            shell.user_ns.pop("frame", None)

        assert module.HAS_WIDGETS is False
        assert callable(module.display)


def test_no_magic_and_no_widget_survive_without_either_package():
    with _executed_without("IPython", "ipywidgets") as module:
        assert module.HAS_IPYTHON is False
        assert module.HAS_WIDGETS is False
        for name in GUARDED_NAMES:
            absent_because(module, name)
        assert module.quick_translate("ATGAAATTTTAA") == "MKF*"


def test_the_session_module_is_untouched_by_all_of_that():
    """The reason a fresh module object is used: the real one keeps what it had.

    Asserted against the module's own flags rather than fixed values, so the test
    says the same thing on an install that never had ipywidgets.
    """
    assert nbmod.quick_gc("ATCG") == 50.0
    if nbmod.HAS_IPYTHON:
        assert isinstance(nbmod.BioSuiteMagics, type)
        assert callable(nbmod.load_ipython_extension)
    if nbmod.HAS_WIDGETS:
        assert hasattr(nbmod, "SequenceAnalyzer")
        assert hasattr(nbmod, "PlotExplorer")


# ── the export contract (CQ-01) ──────────────────────────────────────────────

def test_a_star_import_exposes_exactly_the_available_names():
    """``__all__`` used to advertise guarded names unconditionally, so a bare
    install died on ``from biosuite.notebook import *`` with AttributeError."""
    namespace = {}
    exec("from biosuite.notebook import *", namespace)
    assert star_exported(nbmod) == set(nbmod.__all__)
    for name in nbmod.__all__:
        assert name in namespace, f"{name} is in __all__ but star import missed it"


def test_a_bare_install_star_imports_the_quick_helpers():
    with _executed_without("IPython", "ipywidgets") as module:
        exported = star_exported(module)
        assert exported == BARE_EXPORTS
        assert set(module.__all__) == BARE_EXPORTS


def test_a_bare_install_advertises_only_widgets_but_not_magics():
    with _executed_without("IPython") as module:
        exported = star_exported(module)
        assert BARE_EXPORTS | {"SequenceAnalyzer", "AlignmentViewer",
                               "PlotExplorer"} == exported
        assert "BioSuiteMagics" not in exported
        assert "load_ipython_extension" not in exported


def test_a_widgets_less_install_star_imports_the_magics():
    with _executed_without("ipywidgets") as module:
        exported = star_exported(module)
        assert BARE_EXPORTS | {"BioSuiteMagics", "load_ipython_extension"} == exported
        assert "SequenceAnalyzer" not in exported


def test_a_full_install_star_imports_every_advertised_name():
    pytest.importorskip("IPython")
    pytest.importorskip("ipywidgets")
    exported = star_exported(nbmod)
    assert exported == set(GUARDED_NAMES) | BARE_EXPORTS


def test_an_explicit_import_of_an_absent_optional_name_says_the_fix():
    """The failure must be informative: which dependency, and how to install it."""
    with _executed_without("IPython", "ipywidgets") as module:
        absent_because(module, "SequenceAnalyzer")
        absent_because(module, "load_ipython_extension")


def test_an_unknown_attribute_still_raises_attribute_error():
    with _executed_without("IPython", "ipywidgets") as module:
        with pytest.raises(AttributeError):
            getattr(module, "no_such_attribute_anywhere")
