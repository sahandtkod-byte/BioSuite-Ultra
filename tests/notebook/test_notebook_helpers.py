"""Tests for notebook helper functions + extension contract."""
import pytest


def test_quick_helpers_match_core():
    from biosuite.core.alignment import needleman_wunsch
    from biosuite.core.sequence import gc_content, translate
    from biosuite.notebook import quick_align, quick_gc, quick_translate
    seq = "ATCGATCGATCG"
    assert quick_gc(seq) == gc_content(seq)
    assert quick_translate("ATGAAATTTTAA") == translate("ATGAAATTTTAA")
    assert quick_align("ACTG", "ACTG", "nw") == needleman_wunsch("ACTG", "ACTG")
    assert quick_align("ACTG", "ACTG", "sw")[2] >= 0


def test_load_ipython_extension_defined():
    import biosuite.notebook as nb
    if nb.HAS_IPYTHON:
        assert callable(nb.load_ipython_extension)


def test_magics_exposed_when_ipython_present():
    ip = pytest.importorskip("IPython")
    import biosuite.notebook as nb
    names = [n for n in dir(nb.BioSuiteMagics) if not n.startswith('_')]
    assert {'biosuite', 'biostats', 'bioimport'} <= set(names)


def test_widgets_classes_present_when_available():
    pytest.importorskip("ipywidgets")
    import biosuite.notebook as nb
    for cls in ('SequenceAnalyzer', 'AlignmentViewer', 'PlotExplorer'):
        assert hasattr(nb, cls)
