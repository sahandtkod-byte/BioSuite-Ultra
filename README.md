# BioSuite Ultra

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-1444%20passing-brightgreen)
![Modules](https://img.shields.io/badge/Modules-47-orange)
![Lines](https://img.shields.io/badge/Lines-33%2C600%2B-yellow)
![Version](https://img.shields.io/badge/Version-4.2.2-blueviolet)
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21256296.svg)
![Cloning](https://img.shields.io/badge/Molecular%20Cloning-Free-brightgreen)
![JOSS](https://img.shields.io/badge/JOSS-Submitted-blue.svg)

**The most comprehensive open-source bioinformatics platform.**

BioSuite Ultra is a full-stack bioinformatics platform with 47 analysis modules, 26 visualization types (123 functions), a cyberpunk GUI, a 99+ option CLI, and SnapGene-killer molecular cloning tools — all in pure Python. No external bioinformatics tools required. **100% free.**

---

## What's New in v4.2.2

- **Parallel Processing**: Multi-threaded/multi-process execution for all modules
- **100+ Restriction Enzymes**: Expanded from 18 to 100+ enzymes
- **Better Bayesian Phylogeny**: Real MCMC sampling with Jukes-Cantor model
- **Improved MD Simulation**: Velocity Verlet integrator, Berendsen thermostat
- **Bug Fixes**: 30+ bug fixes across all modules
- **Better Documentation**: Comprehensive changelog and improved docs

---

## 🚀 Quick Start

```bash
# Install
pip install biosuite-ultra

# Run GUI
python -m biosuite

# Or use Docker
docker pull sahandtkod/biosuite-ultra:latest
docker run -p 8000:8000 sahandtkod/biosuite-ultra
```

**Binder:** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sahandtkod-byte/BioSuite-Ultra/main)

---

## Features

### 47 Analysis Modules

| Domain | Modules | Coverage |
|--------|---------|----------|
| Sequence Analysis | FASTA/FASTQ I/O, GC%, translation, reverse complement, ORF finder, primer design, restriction enzymes, codon usage | 85% |
| Alignment | Needleman-Wunsch, Smith-Waterman, BLAST (k-mer), MSA (progressive + Clustal/MUSCLE/MAFFT) | 75% |
| Phylogenetics | p-distance, UPGMA, NJ, ML (RAxML/IQ-TREE), Bayesian (MrBayes + MCMC) | 90% |
| Transcriptomics | CPM/TPM/DESeq2 normalization, differential expression (NB GLM), GO/KEGG enrichment | 70% |
| NGS/Genomics | BAM/VCF parsing, read alignment (BWA/Bowtie2), variant calling, SV/CNV detection | 70% |
| Single-Cell | Scanpy-based scRNA-seq pipeline (QC, normalization, PCA, UMAP, clustering) | 85% |
| Proteins | PDB analysis, ESMFold structure prediction, molecular docking | 55% |
| Epigenomics | Bisulfite methylation, DMR detection, ATAC-seq peak analysis | 45% |
| Metagenomics | K-mer classifier, 16S rRNA pipeline, alpha/beta diversity | 70% |
| Metabolomics | Peak detection, ANOVA, feature alignment, PCA | 55% |
| Population Genetics | HWE, FST, Tajima's D, LD, PCA, nucleotide diversity | 75% |
| CRISPR | Guide RNA design, PAM finding (SpCas9, SaCas9, Cas12a), off-target scoring | 75% |
| Metabolism | Flux balance analysis (FBA), knockout simulation | 60% |
| Machine Learning | Random Forest, SVM, SHAP, cross-validation, feature selection | 55% |
| Workflow | Pipeline builder, batch processor, HTML report generator | 85% |
| GO/Pathways | GO browser, pathway visualization (KEGG-style maps) | 65% |
| GWAS | Chi-squared test, Manhattan/QQ plots, lead SNP detection | 75% |
| Epitope Prediction | T-cell (MHC binding), B-cell (surface propensity), linear epitopes | 75% |
| **Molecular Cloning** | **Plasmid maps, restriction digest, virtual gel, PCR simulation, ligation, Gibson assembly** | **90%** |
| **Parallel Processing** | **Multi-threaded execution, batch processing, progress tracking** | **NEW** |

### Molecular Cloning Tools 🧬

BioSuite includes a complete molecular cloning suite — features that SnapGene charges $350/year for:

| Tool | Description | SnapGene Equivalent |
|------|-------------|-------------------|
| **Restriction Digest** | Simulate single/double digests with 100+ enzymes | ✅ Same |
| **PCR Simulation** | Primer annealing, extension, cycling with Tm calculation | ✅ Same |
| **Ligation** | Insert:vector ratios, T4 ligase efficiency | ✅ Same |
| **Gibson Assembly** | Overlap-based cloning design | ✅ Same |
| **Plasmid Maps** | Circular rendering with annotated features | ✅ Same |
| **Virtual Gel** | Agarose gel simulation from digest results | ✅ Same |
| **Sequence Viewer** | Linear display with feature highlighting | ✅ Same |

**All FREE. No subscriptions. No trials. No limits.**

### Parallel Processing ⚡

Process large datasets faster with built-in parallel execution:

```python
from biosuite.core.parallel import parallel_map, parallel_gc_content
from biosuite.core.sequence import gc_content

# Process 10,000 sequences in parallel
sequences = ["ATCG...", "GCTA...", ...]  # 10,000 sequences
gc_values = parallel_gc_content(sequences, workers=8)

# Or use the batch processor for large datasets
from biosuite.core.parallel import ParallelBatchProcessor
processor = ParallelBatchProcessor(workers=4)
results = processor.process(gc_content, sequences, batch_size=1000)
print(f"Processed {processor.stats['completed']} sequences in {processor.stats['time']:.1f}s")
```

### 100+ Restriction Enzymes 🧪

Full database of Type II restriction enzymes used in molecular biology:

```python
from biosuite.core.utils import RESTRICTION_ENZYMES, RESTRICTION_ENZYMES_SITES

# List all available enzymes
print(f"Available enzymes: {len(RESTRICTION_ENZYMES)}")

# Get enzyme recognition site
site = RESTRICTION_ENZYMES_SITES['EcoRI']  # 'GAATTC'

# Use in restriction digest
from biosuite.core.cloning import simulate_digestion
result = simulate_digestion(plasmid_seq, enzyme='EcoRI')
```

### 36+ Visualization Types

Volcano, PCA, Manhattan, MA, Venn, Barplot, Boxplot, Heatmap, Scatter, Time Series, QQ-plot, Clustered Heatmap, Circos, Alignment Viewer, Violin, Raincloud, Ridge, Dot Plot, GSEA, Motif Logo, Sankey, UMAP, Network (PPI/Regulatory/Metabolic), UpSet, Genome Browser, Interactive (Plotly), Sequence Logo, Conservation, Synteny Dotplot, Plasmid Map, Virtual Gel, and more.

### Dual-Mode Architecture

Every module follows a consistent pattern:
```python
def analyze(input, ...):
    # Try external tool first (fast)
    if _has_external_tool():
        return _run_external(input, ...), {"engine": "external"}
    # Fall back to pure Python (always works)
    return _run_builtin(input, ...), {"engine": "builtin"}
```

### Cyberpunk GUI

- 29 analysis tabs with scrollable sidebar
- 3 themes: Dark-Green-Cyber, Dark-Purple-Cyber, Light-Blue-Cyber
- Keyboard shortcuts (Ctrl+S, Ctrl+Q, F1, F5, Escape)
- Progress bars for long operations
- Plot history (last 10 plots)
- API key configuration panel
- 15 built-in help guides
- Molecular cloning tab with plasmid viewer

### CLI with 99+ Options

Professional CLI menu with organized sections for every analysis type.

---

## Installation

### Via PyPI (recommended)
```bash
pip install biosuite-ultra
```

### Install with all optional features
```bash
pip install "biosuite-ultra[full]"
```

### Windows Users — If `pip install` fails on pysam

pysam needs C build tools. Two options:

**Option A: Visual Studio Build Tools**
1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run installer → select **"Desktop development with C++"** → Install
3. Open **"x64 Native Tools Command Prompt for VS"** (search in Start Menu)
4. Run: `pip install pysam`

**Option B: Use Conda (easier)**
1. Install Anaconda: https://anaconda.com/download
2. Run: `conda install -c bioconda pysam`

### From source
```bash
git clone https://github.com/sahandtkod-byte/BioSuite-Ultra.git
cd BioSuite-Ultra
pip install -r requirements.txt
```

---

## Quick Start

### CLI Mode
```bash
python run.py
```

### GUI Mode
```bash
python run.py --gui
```

### REST API
```bash
python -m biosuite.api.server
# Open http://localhost:8000/docs for Swagger UI
```

**Authentication:** All endpoints require an `X-API-Key` header. Admin endpoints (`/api/v1/admin/*`) additionally require a JWT `Authorization: Bearer <token>` obtained from `/api/v1/admin/login`. Requests are rate-limited to 100/minute. See [API_GUIDE.md](API_GUIDE.md#rest-api-server-authentication) for setup and env vars.

### Programmatic API

#### Basic Sequence Analysis
```python
from biosuite.core.sequence import gc_content, reverse_complement, translate

gc = gc_content("ATCGATCG")  # 50.0
rc = reverse_complement("ATCG")  # "CGAT"
protein = translate("ATGAAATTTTAA")  # "MKF"
```

#### Parallel Processing
```python
from biosuite.core.parallel import parallel_align_pairs

# Align 1000 sequence pairs in parallel
pairs = [("ATCG", "ATCG"), ("GCTA", "GCTA"), ...]  # 1000 pairs
results = parallel_align_pairs(pairs, algorithm='needleman_wunsch', workers=8)
```

#### Molecular Cloning
```python
from biosuite.core.cloning import simulate_digestion, simulate_pcr

# Restriction digest with 100+ enzymes
result = simulate_digestion(plasmid_seq, enzyme="EcoRI")
print(f"Generated {len(result['fragments'])} fragments")

# PCR simulation
pcr_result = simulate_pcr(template, forward_primer, reverse_primer, cycles=30)
print(f"PCR product: {pcr_result['product_size']} bp")
```

#### CRISPR Guide Design
```python
from biosuite.core.crispr import design_guides

result = design_guides(target_sequence, pam_type='SpCas9', guide_length=20)
for guide in result.guides[:5]:
    print(f"{guide.sequence} (score={guide.score:.3f})")
```

#### Differential Expression
```python
from biosuite.core.expression import differential_expression

result = differential_expression(counts_df, conditions=['ctrl', 'ctrl', 'treat', 'treat'])
print(f"Up-regulated: {result['num_upregulated']}")
print(f"Down-regulated: {result['num_downregulated']}")
```

#### Plasmid Maps
```python
from biosuite.plotting.plasmid_map import create_sample_plasmid, draw_plasmid

fig = create_sample_plasmid()
fig.savefig("pUC19_map.png", dpi=150)
```

---

## Architecture

```
BioSuite-Ultra/
├── biosuite/                  # Main package (84 files, 26,000+ lines)
│   ├── core/                    # 45 analysis modules
│   │   ├── parallel.py          # Parallel processing utilities
│   │   ├── sequence.py          # FASTA/FASTQ I/O, GC%, translation
│   │   ├── alignment.py         # NW/SW alignment, MSA
│   │   ├── blast.py             # Sequence similarity search
│   │   ├── assembly.py          # Genome assembly
│   │   ├── ngs.py               # NGS analysis (BAM/VCF)
│   │   ├── crispr.py            # CRISPR guide design
│   │   ├── cloning.py           # Molecular cloning
│   │   ├── expression.py        # Differential expression
│   │   ├── databases.py         # Database searches
│   │   ├── ...                  # 35+ more modules
│   │   └── utils.py             # Shared utilities (100+ enzymes)
│   ├── plotting/                # 13 visualization modules
│   ├── gui/                     # Cyberpunk GUI (29 tabs)
│   ├── cli/                     # CLI menu (99+ options)
│   ├── api/                     # REST API (42+ endpoints)
│   └── notebook/                # Jupyter integration
├── tests/                       # 1,089+ tests
├── examples/                    # 8 tutorials + 5 notebooks
├── docs/                        # Sphinx documentation
├── run.py                       # Entry point
├── pyproject.toml               # Package configuration
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml           # Multi-service Docker Compose
└── CHANGELOG.md                 # Version history
```

---

## Dependencies

### Core (required)
```
numpy>=1.24, pandas>=2.0, matplotlib>=3.7, seaborn>=0.12
scipy>=1.10, scikit-learn>=1.3, customtkinter>=5.2
tqdm>=4.65, biopython>=1.81, networkx>=3.0, plotly>=5.0
```

### Optional (for specific modules)
```
goatools>=1.3, gseapy>=1.0, cutadapt>=4.0
scanpy>=1.9, anndata>=0.9, scikit-bio>=0.5
shap>=0.42, statsmodels>=0.14, umap-learn>=0.5
fastapi>=0.100, uvicorn>=0.23
```

### External Tools (optional, for speed)
```
BLAST+, Clustal Omega, MUSCLE, MAFFT
BWA, Bowtie2, FreeBayes, MACS2
RAxML, IQ-TREE, MrBayes
SPAdes, MEGAHIT, Kraken2
AutoDock Vina, OpenMM
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=biosuite --cov-report=html

# Run parallel tests
python -m pytest tests/ -n auto
```

---

## Docker

```bash
# Build and run CLI
docker-compose up biosuite

# Build and run REST API
docker-compose up biosuite-api

# Build and run Jupyter
docker-compose up jupyter
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use BioSuite Ultra in your research, please cite:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21256296.svg)](https://doi.org/10.5281/zenodo.21256296)

```bibtex
@software{biosuite2026,
  author = {Sahand Touri},
  title = {BioSuite Ultra: Comprehensive Open-Source Bioinformatics Platform},
  year = {2026},
  version = {4.2.2},
  doi = {10.5281/zenodo.21256296},
  url = {https://github.com/sahandtkod-byte/BioSuite-Ultra}
}
```

---

## Links

- **GitHub**: https://github.com/sahandtkod-byte/BioSuite-Ultra
- **DOI**: https://doi.org/10.5281/zenodo.21256296
- **PyPI**: https://pypi.org/project/biosuite-ultra/
- **Issues**: https://github.com/sahandtkod-byte/BioSuite-Ultra/issues
- **JOSSPaper**: https://joss.theoj.org/papers/6efd11d9995ddc82d5d76403c32a4a2d

---

## Contributors

We thank the following people for their contributions to BioSuite Ultra:

- **Faiz Mulla** ([@faizmullaa](https://github.com/faizmullaa)) — India — REST API authentication and rate limiting

