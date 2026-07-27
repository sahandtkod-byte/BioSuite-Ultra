# BioSuite Ultra

**The most comprehensive open-source bioinformatics platform.**

47 analysis modules · 36+ visualization types · CLI + REST API + PyQt6 GUI + Jupyter notebooks

---

## Features

- **Sequence Analysis** — GC content, reverse complement, translation, ORF finding
- **Alignment** — Needleman-Wunsch, Smith-Waterman, MSA (MAFFT/Clustal/MUSCLE)
- **Phylogenetics** — UPGMA, Neighbor-Joining, ML, Bayesian
- **Molecular Cloning** — Restriction enzymes, Gibson Assembly, Golden Gate, PCR simulation
- **NGS Pipeline** — QC, trimming, alignment, variant calling, quantification
- **Expression Analysis** — DESeq2-style normalization, differential expression, enrichment
- **Metagenomics** — 16S classification, diversity metrics
- **CRISPR** — Guide RNA design, off-target scoring
- **Visualization** — 36+ plot types (volcano, PCA, Manhattan, heatmap, etc.)
- **REST API** — 50+ endpoints with OpenAPI docs
- **GUI** — PyQt6 desktop application with 13 analysis tabs
- **Jupyter** — Magic commands and interactive widgets

## Quick Install

```bash
pip install biosuite-ultra

# With optional features
pip install biosuite-ultra[api]      # REST API
pip install biosuite-ultra[gui]      # PyQt6 GUI
pip install biosuite-ultra[notebook] # Jupyter integration
pip install biosuite-ultra[full]     # Everything
```

## Quick Start

```python
from biosuite.core.sequence import gc_content, translate

# GC content
gc = gc_content("ATCGATCGATCG")
print(f"GC: {gc}%")  # GC: 50.0%

# Translate DNA to protein
protein = translate("ATGAAATTTTAA")
print(f"Protein: {protein}")  # Protein: MKF*
```

## Links

- [GitHub](https://github.com/sahandtkod-byte/BioSuite-Ultra)
- [PyPI](https://pypi.org/project/biosuite-ultra/)
- [API Docs](http://localhost:8000/docs) (when running)
- [Changelog](development/changelog.md)
