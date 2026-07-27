"""
Jupyter notebook integration for BioSuite.

Provides IPython magic commands and ipywidgets for interactive analysis
in Jupyter notebooks and Google Colab.

Usage in Jupyter:
    %pip install biosuite-ultra
    %load_ext biosuite.notebook.magics

    # Quick analysis
    %biosuite gc ATCGATCGATCG

    # Interactive widgets
    from biosuite.notebook.widgets import SequenceAnalyzer
    analyzer = SequenceAnalyzer()
    analyzer.show()
"""

from biosuite.notebook.magics import (
    load_ipython_extension,
    HAS_IPYTHON,
)

from biosuite.notebook.widgets import (
    SequenceAnalyzer,
    AlignmentViewer,
    PlotExplorer,
    quick_gc,
    quick_translate,
    quick_align,
    quick_blast,
    HAS_WIDGETS,
)

# BioSuiteMagics is only available when IPython is installed
if HAS_IPYTHON:
    from biosuite.notebook.magics import BioSuiteMagics

__all__ = [
    # magics
    "load_ipython_extension", "HAS_IPYTHON",
    # widgets
    "SequenceAnalyzer", "AlignmentViewer", "PlotExplorer",
    "quick_gc", "quick_translate", "quick_align", "quick_blast",
    "HAS_WIDGETS",
]
