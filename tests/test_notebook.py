"""Tests for biosuite.notebook subpackage imports."""
import pytest


class TestNotebookImports:
    """Verify notebook sub-modules can be imported."""

    def test_import_magics_module(self):
        from biosuite.notebook import magics
        assert hasattr(magics, "load_ipython_extension")

    def test_import_widgets_module(self):
        from biosuite.notebook import widgets
        assert hasattr(widgets, "SequenceAnalyzer")
        assert hasattr(widgets, "AlignmentViewer")
        assert hasattr(widgets, "PlotExplorer")

    def test_import_from_notebook_package(self):
        from biosuite.notebook import magics, widgets
        assert magics is not None
        assert widgets is not None

    def test_quick_gc(self):
        from biosuite.notebook.widgets import quick_gc
        result = quick_gc("ATCGATCG")
        assert result == 50.0

    def test_quick_translate(self):
        from biosuite.notebook.widgets import quick_translate
        result = quick_translate("ATGAAATTTTAA")
        assert result == "MKF*"
