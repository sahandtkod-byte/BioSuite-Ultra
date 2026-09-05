# BioSuite Ultra

[![CI](https://github.com/sahandtkod-byte/BioSuite-Ultra/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sahandtkod-byte/BioSuite-Ultra/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/biosuite-ultra.svg)](https://pypi.org/project/biosuite-ultra/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/sahandtkod-byte/BioSuite-Ultra)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An integrated, pure-Python bioinformatics platform: 47 analysis modules for sequence,
genomics, transcriptomics and population genetics, a 105-function plotting library, molecular
cloning with a 169-enzyme restriction table, and reproducible workflows — usable as a Python
library, a CLI, a desktop GUI or a REST API.

---

## Overview

Most bioinformatics work involves stitching together a dozen tools with incompatible input
formats, installation requirements and output conventions. BioSuite Ultra takes the opposite
approach: a single installable Python package that covers the common analysis path end to end,
with no external binaries required for its core functionality.

Everything runs on the scientific Python stack (NumPy, SciPy, pandas, scikit-learn, Biopython,
matplotlib). Where an external tool such as BLAST+, MUSCLE or MAFFT is available, BioSuite will
use it; where it is not, a built-in pure-Python implementation takes over so that a pipeline
still runs. Those built-in engines are labelled as such in the code and in this document —
see [Scientific scope and limitations](#scientific-scope-and-limitations).

## Why BioSuite Ultra?

- **One dependency, not twelve.** `pip install biosuite-ultra` provides sequence analysis,
  alignment, phylogenetics, expression analysis, variant handling, cloning and plotting.
- **Four interfaces over one core.** The same functions back the library, CLI, GUI and REST
  API, so an interactive exploration can be turned into a script or a service without a rewrite.
- **Reproducibility is built in.** A provenance tracker records parameters and inputs, and
  pipelines are seedable and re-runnable.
- **Graceful degradation, honestly labelled.** Optional accelerators are used when present;
  the pure-Python fallbacks state their approximations rather than hiding them.
- **Tested and scanned on every change.** 2,477 tests run on Python 3.10, 3.11 and 3.12, with
  Ruff, a dedicated security-regression suite and CodeQL in the same pipeline.

---

## Feature matrix

| Area | Capability |
|---|---|
| Sequence analysis | FASTA/FASTQ I/O, GC content, translation, reverse complement, ORF finding, codon usage, primer design, sequence validation |
| Alignment | Needleman–Wunsch and Smith–Waterman (exact DP), k-mer BLAST-style search, progressive MSA with optional Clustal Omega / MUSCLE / MAFFT back ends |
| Phylogenetics | p-distance, UPGMA, neighbour-joining, maximum-likelihood and Bayesian (MCMC) tree inference |
| Transcriptomics | CPM/TPM normalisation, differential expression, transcript quantification, GO/KEGG enrichment |
| Genomics / NGS | Read trimming, read alignment, assembly, variant calling, structural-variant and CNV detection, epigenomics and ChIP-seq peak calling |
| Single cell | scRNA-seq workflow (QC, normalisation, dimensionality reduction, clustering) via the optional `scanpy` stack |
| Population genetics | Allele and genotype frequencies, Hardy–Weinberg testing, F-statistics, GWAS |
| Proteomics & structure | Epitope prediction, structure handling and prediction, molecular docking, molecular-dynamics simulation |
| Metabolism | Metabolic modelling, metabolomics, pathway visualisation |
| Machine learning | Feature construction and model training helpers for biological data (`bio_ml`) |
| Molecular cloning | Restriction digestion with 169 enzymes, PCR simulation, plasmid maps, virtual gel electrophoresis |
| Visualisation | 12 plotting modules, 105 public plotting functions, 40 plot types in the GUI catalogue; matplotlib and Plotly back ends |
| Workflows | Declarative pipelines, batch processing across samples, report generation |
| Provenance | Parameter/input capture and audit records for reproducible runs |
| Databases | NCBI/Ensembl-style lookups and a local GO browser |
| Interfaces | Python library, interactive CLI (99 menu options, 19 direct subcommands), CustomTkinter GUI (11 tabs), FastAPI REST API (38 endpoints) |

---

## Installation

Requires **Python 3.10, 3.11 or 3.12**.

```bash
pip install biosuite-ultra
```

The base install covers the core scientific functionality. Optional extras add the other
interfaces and heavier scientific dependencies:

| Extra | Install | Adds |
|---|---|---|
| `api` | `pip install "biosuite-ultra[api]"` | FastAPI, Uvicorn, JWT auth, rate limiting |
| `gui` | `pip install "biosuite-ultra[gui]"` | CustomTkinter desktop application |
| `notebook` | `pip install "biosuite-ultra[notebook]"` | IPython magics and ipywidgets |
| `bio` | `pip install "biosuite-ultra[bio]"` | goatools, gseapy, scanpy, scikit-bio, cobra and other domain packages |
| `dev` | `pip install "biosuite-ultra[dev]"` | pytest, pytest-cov, Ruff and the test toolchain |
| `full` | `pip install "biosuite-ultra[full]"` | All of the above |

Extras combine: `pip install "biosuite-ultra[api,gui,notebook]"`.

> **The GUI also needs Tk**, which cannot be installed from PyPI. On Debian/Ubuntu:
> `sudo apt-get install python3-tk`. The `gui` extra provides CustomTkinter; Tk itself comes
> from your operating system's Python packaging.

### From source

```bash
git clone https://github.com/sahandtkod-byte/BioSuite-Ultra.git
cd BioSuite-Ultra
pip install -e ".[api,gui,notebook,dev]"
pytest tests/ -q
```

---

## Quick start

```python
from biosuite.core.sequence import gc_content, reverse_complement, translate

gc_content("ATCGATCG")            # 50.0
reverse_complement("ATCG")        # 'CGAT'
translate("ATGAAATTTTAA")         # 'MKF*'  (stop codon retained)
```

A slightly more scientific example — align two sequences and score the result with the exact
dynamic-programming implementation:

```python
from biosuite.core.alignment import needleman_wunsch

aligned1, aligned2, score = needleman_wunsch("ACGTACGTAG", "ACGTTACGAG")
print(aligned1)   # ACG-TACGTAG
print(aligned2)   # ACGTTACG-AG
print(score)      # 5
```

Differential expression over a counts matrix returns a tidy `pandas.DataFrame`:

```python
import pandas as pd
from biosuite.core.expression import differential_expression

counts = pd.read_csv("counts.csv", index_col=0)          # genes x samples
res = differential_expression(counts, conditions=["ctrl", "ctrl", "treat", "treat"])
print(res.columns.tolist())                               # ['gene', 'log2FC', 'pvalue', 'padj']
print(res.nsmallest(10, "padj"))
```

---

## Interfaces

BioSuite Ultra is one platform with four ways to use it. They share the same core functions, so
results are identical whichever you choose.

| Interface | Best for |
|---|---|
| **Python library** | Scripting, notebooks, integration into an existing pipeline |
| **CLI** | Quick one-off calculations and shell pipelines |
| **GUI** | Interactive exploration, teaching, and users who prefer not to write code |
| **REST API** | Serving analyses to other applications or over a network |

### Command line

The `biosuite` console script runs an interactive menu with 99 options, or executes a
subcommand directly:

```bash
biosuite                              # interactive menu
biosuite gc ATCGATCG                  # GC content
biosuite translate ATCGATCG           # translate DNA
biosuite nw AGTACGCA TATGC            # Needleman-Wunsch alignment
biosuite --version
```

Available subcommands: `api`, `blast`, `crispr`, `digest`, `epitope`, `gc`, `gui`, `gwas`,
`hwe`, `manhattan`, `nw`, `pca`, `primers`, `revcomp`, `stats`, `sw`, `theme`, `translate`,
`volcano`.

### Desktop GUI

```bash
biosuite-gui           # console script
biosuite --gui         # or via the main CLI
python run.py --gui    # from a source checkout
```

A CustomTkinter application organised into 11 tabs (sequence analysis, transcriptomics,
genomics, visualisation, cloning, metabolomics, survival, databases, workflow, advanced, help),
with a plot catalogue of 40 chart types and three colour themes.

### REST API

```bash
pip install "biosuite-ultra[api]"
python -m biosuite.api.server        # or: biosuite api --port 8000
```

The API exposes 38 endpoints under `/api/*`, plus `/health`. It **will not start without
credentials configured** — see [Security](#security).

```bash
curl -X POST "http://localhost:8000/api/v1/sequence/gc-content" \
     -H "X-API-Key: $BIOSUITE_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"sequence": "ATCGATCG"}'
```

Endpoint groups: `sequence`, `alignment`, `blast`, `phylogeny`, `expression`, `popgen`, `gwas`,
`crispr`, `epitope`, `metagenomics`, `database`, `file`, `plotting`, `provenance`, `workflow`,
`modules`, `admin`. Full details in [API_GUIDE.md](API_GUIDE.md).

### Notebooks

```bash
pip install "biosuite-ultra[notebook]"
```

`biosuite.notebook` provides IPython magics and ipywidgets helpers. Five worked notebooks are
in [`examples/`](examples/), alongside eight standalone tutorial scripts.

---

## Molecular cloning

A restriction table of **169 enzymes** with recognition sites and cut positions drives
digestion, fragment prediction and virtual gel electrophoresis:

```python
from biosuite.core.cloning import simulate_digestion

result = simulate_digestion(plasmid_sequence, enzyme="EcoRI")
print(result["fragments"], result["sizes"], result["cuts"])
```

Plasmid maps are drawn from a `PlasmidMap` object:

```python
from biosuite.plotting.plasmid_map import create_sample_plasmid, draw_plasmid

plasmid = create_sample_plasmid()
fig = draw_plasmid(plasmid)
fig.savefig("plasmid_map.png", dpi=150)
```

---

## Architecture

```
biosuite/
├── core/        47 analysis modules (44 domain modules + core/workflow/)
│   ├── sequence, alignment, msa, phylogeny, blast, crispr, cloning, ...
│   ├── provenance.py        run-level audit records
│   └── workflow/            pipeline.py, batch.py, report.py
├── plotting/    12 modules, 105 public plotting functions (matplotlib + Plotly)
├── cli/         menu.py — interactive menu and subcommand dispatch
├── gui/         CustomTkinter application: main_window.py, themes.py, tabs/ (11)
├── api/         FastAPI app, auth.py, security.py, server.py
└── notebook/    IPython magics and widgets
```

The dependency direction is one-way: **core → plotting → interfaces**. Analysis modules never
import from `gui`, `cli` or `api`, so the scientific code is usable headless and the GUI is an
optional layer. GUI and notebook imports are lazy, so importing `biosuite` does not require Tk
or IPython.

---

## Reproducibility and provenance

`biosuite.core.provenance` records the parameters, inputs and environment of an analysis so a
result can be traced back to the call that produced it. Pipelines in `biosuite.core.workflow`
accept explicit random seeds and keep per-run state isolated, so repeated execution of the same
pipeline with the same seed produces the same output. The REST API exposes provenance records
through its `/api/v1/provenance/*` endpoints.

---

## Security

The API is designed to fail closed.

- **No default credentials.** The server entry point (`python -m biosuite.api.server`)
  refuses to bind a socket and exits non-zero unless `BIOSUITE_API_KEY`,
  `BIOSUITE_JWT_SECRET` and `BIOSUITE_ADMIN_PASSWORD` are set. There is no built-in admin
  password and no fallback signing secret. Importing the `app` object in-process without them
  (as tests do) generates ephemeral random values and warns; those never serve traffic and do
  not survive a restart.
- **Layered authentication.** All endpoints require an `X-API-Key` header; `/api/v1/admin/*`
  additionally requires a JWT bearer token from `/api/v1/admin/login`. Login attempts are
  rate-limited.
- **Confined file access.** File endpoints resolve paths only inside `BIOSUITE_DATA_DIR`.
  Absolute paths, `..` traversal, percent-encoded traversal, dotfiles and symlinks that escape
  the directory are rejected.
- **Restricted CORS.** Origins come from `BIOSUITE_CORS_ORIGINS`; arbitrary origins are not
  reflected, and credentials are disabled when a wildcard is configured.
- **Documentation endpoints are closed by default.** `/docs` and `/openapi.json` require
  authentication unless `BIOSUITE_DEV_MODE=1`.

Configuration is via environment variables — see [`.env.example`](.env.example) and
[`biosuite_config.example.json`](biosuite_config.example.json). Never commit real secrets.

These are engineering controls, not a guarantee. The project has not undergone an external
security audit, and no claim of suitability for handling regulated or clinical data is made.
To report a vulnerability, see [SECURITY.md](.github/SECURITY.md).

> **Operators upgrading from an earlier revision:** credentials committed to this repository's
> history in the past must be treated as compromised and rotated. See the security notice in
> [CHANGELOG.md](CHANGELOG.md).

---

## Testing and quality

Every push and pull request runs the full suite on three interpreters, plus linting, a
security-regression suite, a package build and CodeQL analysis.

| Check | Status at the latest verified run |
|---|---|
| Tests — Python 3.10 | 2477 passed, 14 skipped, 0 failed |
| Tests — Python 3.11 | 2477 passed, 14 skipped, 0 failed |
| Tests — Python 3.12 | 2477 passed, 14 skipped, 0 failed |
| Ruff (`E9,F,B` as a hard gate) | pass |
| Security regression suite | pass |
| Package build (`build` + `twine check`) | pass |
| CodeQL (python + actions) | pass, zero alerts |

The 14 skips are optional scientific dependencies (`scanpy`, `gseapy`, `torch`/ESM,
`umap-learn`) that are not installed in CI. Coverage is measured on every CI run; the project
does not claim full coverage.

Run the suite locally with:

```bash
pytest tests/ -q                     # full suite
pytest tests/api tests/cli -q        # security-relevant subset
ruff check biosuite/ --select E9,F,B
```

### CI matrix

| | |
|---|---|
| Python | 3.10, 3.11, 3.12 |
| OS | ubuntu-latest |
| Linting | Ruff (`E9,F,B` blocking; broader rule set advisory) |
| Static analysis | CodeQL — `python` and `actions` |
| Packaging | `python -m build` and `twine check` |

---

## Scientific scope and limitations

BioSuite Ultra is research software. Where an algorithm is simplified, heuristic or dependent
on external data, it is labelled as such in the code and its docstring. In particular:

- Built-in engines for BLAST-style search, read alignment and docking are **pure-Python
  approximations** intended to keep pipelines runnable without external binaries. Install the
  corresponding external tool for production-scale or publication-grade work.
- Docking scores are reported in **arbitrary units**, not kcal/mol.
- Enrichment analysis requires ontology data via the optional `bio` extra; without it the
  relevant functions report that the dependency is missing rather than returning a result.
- Structure prediction and molecular dynamics are simplified implementations for teaching and
  exploration, not replacements for dedicated MD or structure-prediction software.

No part of this project is validated for clinical or diagnostic use.

---

## Project status

The current development version is **5.5.0**, which is **not yet published to PyPI** — the
latest released version is **5.0.3**. Version 5.5.0 is a correctness, security and
reproducibility release; see [CHANGELOG.md](CHANGELOG.md) for the full list of changes.

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](.github/CODE_OF_CONDUCT.md) first.

```bash
git clone https://github.com/sahandtkod-byte/BioSuite-Ultra.git
cd BioSuite-Ultra
pip install -e ".[api,gui,notebook,dev]"
pytest tests/ -q
ruff check biosuite/ --select E9,F,B
```

New scientific behaviour should come with a test that fails before the change and passes after
it. The structural counts quoted in this README — analysis modules, REST endpoints, plotting
functions, plot types, restriction enzymes, GUI tabs, CLI menu options and supported Python
versions — are asserted by `tests/test_documentation_accuracy.py`, so if you add one, update
the documentation in the same commit or CI will fail. The pass/skip figures in the table above
are refreshed from the latest verified CI run and are not test-locked.

## Security reporting

Please do **not** open a public issue for security problems. Reporting instructions, scope and
response expectations are in [SECURITY.md](.github/SECURITY.md).

## License

Released under the [MIT License](LICENSE).

## Citation

If you use BioSuite Ultra in your research, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff):

```bibtex
@software{biosuite_ultra,
  author  = {Sahand Touri},
  title   = {BioSuite Ultra: Comprehensive Open-Source Bioinformatics Platform},
  year    = {2026},
  version = {5.5.0},
  doi     = {10.5281/zenodo.21256296},
  url     = {https://github.com/sahandtkod-byte/BioSuite-Ultra}
}
```

## Links

- **Repository**: https://github.com/sahandtkod-byte/BioSuite-Ultra
- **PyPI**: https://pypi.org/project/biosuite-ultra/
- **Issues**: https://github.com/sahandtkod-byte/BioSuite-Ultra/issues
- **API guide**: [API_GUIDE.md](API_GUIDE.md)
- **Deployment**: [DEPLOY.md](DEPLOY.md)

## Contributors

- **Faiz Mulla** ([@faizmullaa](https://github.com/faizmullaa)) — REST API authentication and
  rate limiting
