Tutorials
=========

.. toctree::
   :maxdepth: 1

   01_sequence_analysis
   02_alignment
   03_differential_expression
   04_single_cell
   05_crispr_guide_rna
   06_metagenomics
   07_machine_learning
   08_advanced_visualization

Tutorial 1: Sequence Analysis
-----------------------------

Learn the basics of sequence analysis with BioSuite:

- Reading FASTA/FASTQ files
- Computing GC content
- Reverse complement
- Translation
- Sequence statistics

See: ``examples/tutorial_01_sequence_analysis.py``

Tutorial 2: Pairwise Alignment
------------------------------

Master alignment algorithms:

- Needleman-Wunsch (global alignment)
- Smith-Waterman (local alignment)
- Score comparison
- Gap penalty effects

See: ``examples/tutorial_02_alignment.py``

Tutorial 3: Differential Expression
------------------------------------

RNA-seq analysis workflow:

- Count matrix normalization (CPM, TPM, DESeq2)
- Differential expression testing
- Volcano plot visualization
- VST transformation

See: ``examples/tutorial_03_differential_expression.py``

Tutorial 4: Single-Cell Analysis
--------------------------------

scRNA-seq pipeline with Scanpy:

- Quality control
- Normalization
- Dimensionality reduction (PCA, UMAP)
- Clustering
- Marker gene detection

See: ``examples/tutorial_04_single_cell.py``

Tutorial 5: CRISPR Guide RNA Design
------------------------------------

Design guide RNAs for CRISPR experiments:

- PAM site finding
- Guide scoring
- Off-target analysis

See: ``examples/tutorial_05_crispr.py``

Tutorial 6: Metagenomics
-------------------------

Taxonomic classification and diversity:

- k-mer based classification
- 16S rRNA analysis
- Alpha/beta diversity

See: ``examples/tutorial_06_metagenomics.py``

Tutorial 7: Machine Learning
-----------------------------

Apply ML to biological data:

- Random Forest classification
- SVM classification
- Feature importance
- Cross-validation

See: ``examples/tutorial_07_machine_learning.py``

Tutorial 8: Advanced Visualization
-----------------------------------

Publication-quality figures:

- Interactive Plotly plots
- Custom themes
- Export options

See: ``examples/tutorial_08_visualization.py``

New in v4.1.0
--------------

Tutorial 9: Parallel Processing
--------------------------------

Process large datasets faster:

- Multi-threaded execution
- Batch processing
- Progress tracking

See: ``from biosuite.core.parallel import parallel_map``

Tutorial 10: Molecular Cloning
-------------------------------

Complete molecular cloning toolkit:

- Restriction digest with 100+ enzymes
- PCR simulation
- Ligation efficiency
- Virtual gel electrophoresis

See: ``from biosuite.core.cloning import simulate_digestion``
