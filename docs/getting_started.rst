Getting Started
===============

This guide will help you get started with BioSuite Ultra v4.1.0.

Installation
------------

Via PyPI (recommended):

.. code-block:: bash

   pip install biosuite-ultra

With all optional features:

.. code-block:: bash

   pip install "biosuite-ultra[full]"

From source:

.. code-block:: bash

   git clone https://github.com/sahandtouri/BioSuite-Ultra.git
   cd BioSuite-Ultra
   pip install -r requirements.txt

Quick Start
-----------

CLI Mode
~~~~~~~~

Launch the interactive CLI menu:

.. code-block:: bash

   python run.py

GUI Mode
~~~~~~~~

Launch the modern graphical interface:

.. code-block:: bash

   python run.py --gui

REST API
~~~~~~~~

Launch the REST API server:

.. code-block:: bash

   python -m biosuite.api.server

Open http://localhost:8000/docs for Swagger UI.

Python API
~~~~~~~~~~

Import and use functions directly:

.. code-block:: python

   from biosuite.core.sequence import gc_content, reverse_complement, translate

   # GC content
   gc = gc_content("ATCGATCG")
   print(f"GC content: {gc}%")

   # Reverse complement
   rc = reverse_complement("ATCG")
   print(f"Reverse complement: {rc}")

   # Translation
   protein = translate("ATGAAATTTTAA")
   print(f"Protein: {protein}")

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Process large datasets faster:

.. code-block:: python

   from biosuite.core.parallel import parallel_gc_content

   # Process 10,000 sequences in parallel
   sequences = ["ATCG...", "GCTA...", ...]  # 10,000 sequences
   gc_values = parallel_gc_content(sequences, workers=8)

Molecular Cloning
~~~~~~~~~~~~~~~~~

Restriction digest with 100+ enzymes:

.. code-block:: python

   from biosuite.core.cloning import simulate_digestion

   result = simulate_digestion(plasmid_seq, enzyme="EcoRI")
   print(f"Generated {len(result['fragments'])} fragments")

CRISPR Guide Design
~~~~~~~~~~~~~~~~~~~

Design guide RNAs:

.. code-block:: python

   from biosuite.core.crispr import design_guides

   result = design_guides(target_sequence, pam_type='SpCas9')
   for guide in result.guides[:5]:
       print(f"{guide.sequence} (score={guide.score:.3f})")

Interactive Plots
~~~~~~~~~~~~~~~~~

Generate interactive Plotly plots:

.. code-block:: python

   from biosuite.plotting.plot_api import volcano, pca
   import numpy as np

   # Create volcano plot
   fc = np.random.normal(0, 1.5, 500)
   pvals = np.random.uniform(0, 1, 500)
   fig = volcano(fc, pvals, interactive=True)
   fig.show()  # In Jupyter
   fig.write_html("volcano.html")  # Export

Jupyter Integration
~~~~~~~~~~~~~~~~~~~

Use magic commands in Jupyter notebooks:

.. code-block:: python

   %load_ext biosuite.notebook.magic

   # Quick analysis
   %biosuite gc ATCGATCG

   # Import all functions
   %bioimport all

Next Steps
----------

- :doc:`tutorials/index` — Step-by-step tutorials
- :doc:`api/index` — Full API reference
- :doc:`gui_guide` — GUI user guide
- :doc:`cli_guide` — CLI reference

What's New in v4.1.0
---------------------

- **Parallel Processing**: Multi-threaded/multi-process execution
- **100+ Restriction Enzymes**: Expanded from 18 to 100+ enzymes
- **Better Bayesian Phylogeny**: Real MCMC with Jukes-Cantor model
- **Improved MD Simulation**: Velocity Verlet integrator
- **30+ Bug Fixes**: Across all modules
- **New Documentation**: CHANGELOG.md, CONTRIBUTING.md
