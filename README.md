# BioSuite Ultra

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-5.5.0-blueviolet)
![Modules](https://img.shields.io/badge/Analysis%20Modules-47-orange)
![Plots](https://img.shields.io/badge/Plotting%20Functions-105-yellow)
![Enzymes](https://img.shields.io/badge/Restriction%20Enzymes-169-brightgreen)
![Tests](https://img.shields.io/badge/Tests-2493%20passed-brightgreen)
![PyPI](https://img.shields.io/pypi/v/biosuite-ultra.svg)
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21256296.svg)
![JOSS](https://img.shields.io/badge/JOSS-Submitted-blue.svg)

**BioSuite Ultra is an open-source, pure-Python bioinformatics platform for integrated sequence, genomics, transcriptomics, population genetics, molecular cloning, visualization, and multi-omics analysis.**

BioSuite Ultra provides **47 analysis modules**, **105 public plotting functions**, **169 restriction enzymes**, a desktop GUI, a professional CLI, a REST API, Jupyter integration, reproducible workflows, and provenance tracking.

The project is designed as a unified bioinformatics environment that can be used as a **Python library, command-line application, desktop application, REST service, or web application backend**.

**Free and open source under the MIT License.**

---

## Current Release: v5.5.0

**BioSuite Ultra v5.5.0 is the current production release.**

This release focuses on **stability, API reliability, security hardening, reproducibility, and production readiness** across the Python library, CLI, GUI, and REST API.

### Highlights

* **47 bioinformatics analysis modules**
* **105 public plotting functions**
* **169 restriction enzymes**
* **99 CLI menu options**
* **19 direct CLI subcommands**
* **11 GUI tabs**
* **38 documented REST API endpoints**
* Reproducible workflows and provenance tracking
* REST API authentication and rate limiting
* Secure file-path handling within the configured data directory
* Python 3.10, 3.11, and 3.12 support
* Comprehensive automated test suite
* Docker and Docker Compose support
* Jupyter integration
* Pure-Python implementations with optional external-tool acceleration where available

### Quality Status

The v5.5.0 development and validation cycle includes:

* **2,493 tests passed**
* **14 tests skipped**
* **0 failed tests**
* Ruff static analysis passing
* Security regression tests passing
* Package build and distribution checks passing
* CodeQL analysis with zero alerts

---

## Why BioSuite Ultra?

BioSuite Ultra brings many common bioinformatics workflows into one consistent Python ecosystem.

Instead of requiring a different interface for every analysis, users can access the same underlying software through:

```text
Python Library
      │
      ├── CLI
      ├── Desktop GUI
      ├── REST API
      └── Jupyter
```

The architecture keeps the scientific analysis layer independent from presentation and transport layers.

This allows the same core functionality to be reused locally, through automation, or as a backend service.

---

# Features

## 47 Analysis Modules

BioSuite Ultra covers a broad range of computational biology and bioinformatics workflows.

| Domain              | Examples                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| Sequence Analysis   | FASTA/FASTQ processing, GC content, translation, reverse complement, ORF analysis, primer design |
| Alignment           | Needleman-Wunsch, Smith-Waterman, sequence similarity search, multiple sequence alignment        |
| Phylogenetics       | Distance matrices, UPGMA, neighbor joining, phylogenetic analysis                                |
| Transcriptomics     | CPM/TPM normalization, differential expression, enrichment analysis                              |
| Genomics / NGS      | BAM/VCF processing, variant analysis and sequencing workflows                                    |
| Single-Cell         | QC, normalization, PCA, dimensionality reduction and clustering workflows                        |
| Proteomics          | Protein and structural analysis                                                                  |
| Epigenomics         | Methylation and chromatin-related analysis                                                       |
| Metagenomics        | K-mer classification, 16S analysis, alpha/beta diversity                                         |
| Metabolomics        | Peak and feature analysis, statistical analysis, PCA                                             |
| Population Genetics | HWE, FST, Tajima's D, LD and nucleotide diversity                                                |
| CRISPR              | Guide design, PAM detection and off-target scoring                                               |
| Metabolism          | Flux balance analysis and knockout simulation                                                    |
| Machine Learning    | Random Forest, SVM, feature selection and model evaluation                                       |
| GWAS                | Association analysis, Manhattan/QQ visualization and lead variant analysis                       |
| Epitope Prediction  | T-cell and B-cell epitope analysis                                                               |
| Molecular Cloning   | Restriction digestion, PCR, ligation, Gibson assembly and plasmid analysis                       |
| Workflow            | Pipeline execution, batch processing and report generation                                       |
| Databases           | NCBI, UniProt, PDB and KEGG integrations                                                         |

---

# Molecular Cloning

BioSuite Ultra includes an integrated molecular cloning toolkit for computational cloning workflows.

### Available tools

| Tool                   | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| **Restriction Digest** | Simulate single and double restriction digests         |
| **PCR Simulation**     | Primer annealing, amplification and product analysis   |
| **Ligation**           | Computational insert/vector ligation                   |
| **Gibson Assembly**    | Overlap-based assembly design                          |
| **Plasmid Maps**       | Circular plasmid visualization                         |
| **Virtual Gel**        | Simulated agarose gel results                          |
| **Sequence Viewer**    | Linear sequence visualization and feature highlighting |

The current release includes **169 restriction enzymes** with recognition-site and cut-position information.

---

# Visualization

BioSuite Ultra provides **105 public plotting functions** across scientific and bioinformatics workflows.

Visualization capabilities include:

* Volcano plots
* PCA
* Manhattan plots
* MA plots
* Heatmaps
* Clustered heatmaps
* Boxplots
* Bar plots
* Scatter plots
* Time-series plots
* QQ plots
* Venn diagrams
* Violin plots
* Raincloud plots
* Ridge plots
* Dot plots
* GSEA visualization
* Motif logos
* Sankey diagrams
* UMAP visualization
* Network visualization
* UpSet plots
* Genome browser visualization
* Sequence logos
* Conservation plots
* Synteny visualization
* Plasmid maps
* Virtual gels
* Alignment visualization
* Interactive Plotly-based visualization

---

# Parallel Processing

BioSuite Ultra provides built-in parallel processing utilities for supported workloads.

```python
from biosuite.core.parallel import parallel_gc_content

sequences = ["ATCG...", "GCTA...", "..."]

gc_values = parallel_gc_content(
    sequences,
    workers=8,
)
```

Batch processing can also be used for large collections of sequences or analysis jobs.

---

# 169 Restriction Enzymes

The molecular cloning subsystem includes a database of **169 restriction enzymes**.

```python
from biosuite.core.utils import (
    RESTRICTION_ENZYMES,
    RESTRICTION_ENZYMES_SITES,
)

print(len(RESTRICTION_ENZYMES))

site = RESTRICTION_ENZYMES_SITES["EcoRI"]
print(site)
```

Restriction enzymes can be used directly with computational digestion workflows:

```python
from biosuite.core.cloning import simulate_digestion

result = simulate_digestion(
    plasmid_seq,
    enzyme="EcoRI",
)
```

---

# Dual-Mode Architecture

BioSuite Ultra is designed to support a consistent analysis interface while allowing optional acceleration through external bioinformatics software.

Conceptually:

```python
def analyze(input, ...):
    if external_tool_available():
        return run_external(input, ...)

    return run_builtin(input, ...)
```

Where an external implementation is available and configured, it can be used for acceleration or specialized workflows.

The built-in Python implementations provide a self-contained computational path for supported functionality.

This architecture keeps the core analysis layer independent from the GUI, CLI, API, and deployment environment.

---

# Interfaces

## Python Library

BioSuite Ultra can be used directly from Python.

```python
from biosuite.core.sequence import (
    gc_content,
    reverse_complement,
    translate,
)

gc = gc_content("ATCGATCG")
rc = reverse_complement("ATCG")
protein = translate("ATGAAATTTTAA")

print(gc)
print(rc)
print(protein)
```

---

## Command-Line Interface

BioSuite Ultra provides:

* **99 CLI menu options**
* **19 direct CLI subcommands**

The CLI covers sequence analysis, alignment, genomics, statistics, visualization, molecular cloning, workflows, and other supported modules.

Run:

```bash
biosuite
```

or:

```bash
python run.py
```

---

## Desktop GUI

The desktop interface currently contains **11 GUI tabs** covering major BioSuite workflows.

Features include:

* Dark and light themes
* Keyboard shortcuts
* Progress indicators
* Plot history
* API configuration
* Molecular cloning workflows
* Interactive analysis controls
* Built-in help

Launch the GUI with:

```bash
python -m biosuite
```

or:

```bash
python run.py --gui
```

---

## REST API

BioSuite Ultra includes a FastAPI-based REST interface with **38 documented API endpoints** covering major analysis and workflow operations.

API groups include:

* Sequence
* Alignment
* BLAST
* Expression
* Population Genetics
* GWAS
* CRISPR
* Epitope Prediction
* Metagenomics
* Databases
* File Operations
* Plotting
* Provenance
* Workflows
* Modules
* Administration

Start the API locally:

```bash
python -m biosuite.api.server
```

Then open:

```text
http://localhost:8000/docs
```

for the interactive Swagger/OpenAPI documentation.

### API Security

API endpoints require an `X-API-Key`.

Administrative endpoints additionally require JWT authentication.

File operations are restricted to the configured BioSuite data directory.

API requests are rate-limited.

See [`API_GUIDE.md`](API_GUIDE.md) for configuration and authentication details.

---

# Reproducibility and Provenance

BioSuite Ultra includes workflow and provenance capabilities designed to make computational analyses easier to reproduce.

Provenance information can be recorded and queried through the platform's provenance subsystem and REST API.

This enables workflows to retain information about analysis operations and computational history rather than treating each execution as an isolated command.

---

# Installation

## PyPI

The recommended installation method is PyPI:

```bash
pip install biosuite-ultra
```

Install the current release explicitly:

```bash
pip install biosuite-ultra==5.5.0
```

Optional features:

```bash
pip install "biosuite-ultra[full]"
```

---

## From Source

```bash
git clone https://github.com/sahandtkod-byte/BioSuite-Ultra.git

cd BioSuite-Ultra

pip install -e .
```

---

## Docker

Pull the latest container:

```bash
docker pull sahandtkod/biosuite-ultra:latest
```

Run the REST API:

```bash
docker run -p 8000:8000 sahandtkod/biosuite-ultra
```

Docker Compose configurations are also included in the repository.

---

# Jupyter

BioSuite Ultra includes Jupyter integration for notebook-based workflows and computational exploration.

The repository contains example notebooks demonstrating selected analysis workflows.

---

# Quick Examples

## GC Content

```python
from biosuite.core.sequence import gc_content

print(gc_content("ATCGATCG"))
```

## Reverse Complement

```python
from biosuite.core.sequence import reverse_complement

print(reverse_complement("ATCG"))
```

## Translation

```python
from biosuite.core.sequence import translate

print(translate("ATGAAATTTTAA"))
```

## CRISPR Guide Design

```python
from biosuite.core.crispr import design_guides

result = design_guides(
    target_sequence,
    pam_type="SpCas9",
    guide_length=20,
)

for guide in result.guides[:5]:
    print(
        guide.sequence,
        guide.score,
    )
```

## Differential Expression

```python
from biosuite.core.expression import differential_expression

result = differential_expression(
    counts_df,
    conditions=["ctrl", "ctrl", "treat", "treat"],
)

print(result["num_upregulated"])
print(result["num_downregulated"])
```

## Plasmid Visualization

```python
from biosuite.plotting.plasmid_map import (
    create_sample_plasmid,
    draw_plasmid,
)

fig = create_sample_plasmid()
fig.savefig("pUC19_map.png", dpi=150)
```

---

# Architecture

```text
BioSuite-Ultra/
│
├── biosuite/
│   ├── core/          # Scientific analysis layer
│   ├── plotting/      # Visualization functions
│   ├── gui/           # Desktop GUI
│   ├── cli/           # Command-line interface
│   ├── api/           # REST API
│   └── notebook/      # Jupyter integration
│
├── tests/             # Automated test suite
├── examples/          # Tutorials and notebooks
├── docs/              # Documentation
│
├── run.py             # Application entry point
├── pyproject.toml     # Package configuration
├── Dockerfile         # Container definition
├── docker-compose.yml # Docker Compose configuration
└── CHANGELOG.md       # Release history
```

### Dependency Direction

The architecture is intentionally layered:

```text
Core
 ↓
Plotting
 ↓
Interfaces
 ├── CLI
 ├── GUI
 ├── REST API
 └── Notebook
```

Scientific analysis modules do not depend on GUI, CLI, or API layers.

This separation allows the same computational functionality to be reused across different interfaces.

---

# Dependencies

## Core Dependencies

BioSuite Ultra uses scientific Python libraries including:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
biopython
networkx
plotly
tqdm
```

Additional dependencies are used by optional functionality.

## Optional Bioinformatics Tools

Some specialized workflows can integrate with external tools when installed, including:

```text
BLAST+
Clustal Omega
MUSCLE
MAFFT
BWA
Bowtie2
FreeBayes
MACS2
RAxML
IQ-TREE
MrBayes
SPAdes
MEGAHIT
Kraken2
AutoDock Vina
OpenMM
```

External tools are optional and are not required for the platform's built-in Python functionality where a built-in implementation is available.

---

# Testing

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

Coverage:

```bash
python -m pytest tests/ \
    --cov=biosuite \
    --cov-report=html
```

Parallel test execution:

```bash
python -m pytest tests/ -n auto
```

Current v5.5.0 validation status:

```text
2,493 passed
14 skipped
0 failed
```

Additional validation includes static analysis, security regression testing, package build checks, and CodeQL analysis.

---

# Documentation

Project documentation and API documentation are available in the repository.

Key resources include:

* [`API_GUIDE.md`](API_GUIDE.md)
* [`DEVELOPMENT.md`](DEVELOPMENT.md)
* [`CONTRIBUTING.md`](CONTRIBUTING.md)
* [`CHANGELOG.md`](CHANGELOG.md)
* [`docs/`](docs/)

---

# Contributing

Contributions, bug reports, feature requests, documentation improvements, and scientific feedback are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

---

# License

BioSuite Ultra is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

---

# Citation

If you use BioSuite Ultra in research, please cite the software release:

```bibtex
@software{biosuite2026,
  author = {Touri, Sahand},
  title = {BioSuite Ultra: Comprehensive Open-Source Bioinformatics Platform},
  year = {2026},
  version = {5.5.0},
  doi = {10.5281/zenodo.21256296},
  url = {https://github.com/sahandtkod-byte/BioSuite-Ultra}
}
```

DOI:

https://doi.org/10.5281/zenodo.21256296

---

# Project Links

* **GitHub:** https://github.com/sahandtkod-byte/BioSuite-Ultra
* **PyPI:** https://pypi.org/project/biosuite-ultra/
* **PyPI v5.5.0:** https://pypi.org/project/biosuite-ultra/5.5.0/
* **DOI:** https://doi.org/10.5281/zenodo.21256296
* **Issues:** https://github.com/sahandtkod-byte/BioSuite-Ultra/issues
* **JOSS Paper:** https://joss.theoj.org/papers/6efd11d9995ddc82d5d76403c32a4a2d

---

# Contributors

We thank the contributors and community members who have helped improve BioSuite Ultra.

* **Faiz Mulla** — REST API authentication and rate limiting

---

## Project Status

**BioSuite Ultra v5.5.0 is the current production release.**

The project is under active development, with ongoing work focused on:

* Scientific validation
* Reproducible research workflows
* API and web integration
* Documentation
* Performance
* Security
* Community contributions
* Research use cases
