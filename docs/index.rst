BioSuite Ultra Documentation
============================

An integrated, pure-Python bioinformatics platform.

BioSuite Ultra provides 47 analysis modules, 105 public plotting functions, molecular cloning
with a 169-enzyme restriction table, and reproducible workflows — usable as a Python library,
a command-line interface, a desktop GUI or a REST API.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   tutorials/index
   api/index

Features
--------

- **47 analysis modules**: sequence analysis, alignment, phylogenetics, transcriptomics,
  genomics/NGS, single-cell, population genetics, proteomics, structure and metabolism
- **105 public plotting functions** across 12 modules; the GUI catalogue lists 40 plot types
- **Parallel processing**: thread- and process-based execution helpers
- **169 restriction enzymes** with recognition sites and cut positions for molecular cloning
- **Desktop GUI**: CustomTkinter application with 11 tabs and three themes
- **CLI**: interactive menu with 99 options plus 19 direct subcommands
- **REST API**: 38 FastAPI endpoints under ``/api/*``
- **Pure Python core**: no external bioinformatics binaries required
- **Optional external tools**: BLAST+, Clustal Omega, MUSCLE and MAFFT are used when present

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
