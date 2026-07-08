BioSuite Ultra Documentation
============================

**The most comprehensive open-source bioinformatics platform.**

BioSuite Ultra v4.1.0 is a full-stack bioinformatics platform with 53 analysis modules,
36+ visualization types, a cyberpunk GUI, and a 99+ option CLI — all in pure Python.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   installation
   tutorials/index
   api/index
   gui_guide
   cli_guide
   plugins
   contributing
   changelog

Features
--------

- **53 Analysis Modules**: Sequence analysis, alignment, phylogenetics, transcriptomics,
  single-cell, proteomics, metabolomics, and more
- **36+ Visualization Types**: Volcano, PCA, Manhattan, Heatmap, Interactive Plotly, and more
- **Parallel Processing**: Multi-threaded/multi-process execution for all modules
- **100+ Restriction Enzymes**: Comprehensive enzyme database for molecular cloning
- **Cyberpunk GUI**: 29 analysis tabs with scrollable sidebar
- **CLI with 99+ Options**: Professional CLI menu with organized sections
- **Pure Python**: No external binaries required — works with just ``pip install``
- **Dual-Mode Architecture**: Optional external tools (BLAST+, Clustal Omega) for speed

Quick Start
-----------

Install BioSuite Ultra:

.. code-block:: bash

   pip install biosuite-ultra

Launch the CLI:

.. code-block:: bash

   python run.py

Launch the GUI:

.. code-block:: bash

   python run.py --gui

Launch the REST API:

.. code-block:: bash

   python -m biosuite.api.server

Use in Python:

.. code-block:: python

   from biosuite.core.sequence import gc_content, reverse_complement, translate

   gc = gc_content("ATCGATCG")  # 50.0
   rc = reverse_complement("ATCG")  # "CGAT"
   protein = translate("ATGAAATTTTAA")  # "MKF"

Parallel Processing:

.. code-block:: python

   from biosuite.core.parallel import parallel_gc_content

   sequences = ["ATCG...", "GCTA...", ...]
   gc_values = parallel_gc_content(sequences, workers=8)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
