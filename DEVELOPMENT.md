# BioSuite Ultra — Developer Guide

## Project Overview
BioSuite Ultra is a pure-Python bioinformatics platform. All numbers below were
measured from the tree — keep them accurate if you change the code.

- **47 analysis modules** in `biosuite/core/` (+ `biosuite/core/workflow/`)
- **12 plotting modules** exposing **167 functions** (105 public); the GUI plot catalogue lists **40 plot types** in 9 categories
- **11 GUI tabs**, **117 CLI menu options**, **38 REST API endpoints** (under `/api/*`)
- **169 restriction enzymes** (`RESTRICTION_ENZYMES` in `biosuite/core/utils.py`)
- **42,665 lines** of Python across 90 modules in `biosuite/`
- **460 test files**; the run recorded under Current Status is the authoritative figure

## Architecture
```
biosuite/
├── core/           # 44 domain modules + core/workflow/ (3 modules)
├── gui/            # customtkinter app: main_window.py, themes.py, dialogs.py, tabs/ (11)
├── api/            # FastAPI app: __init__.py (38 endpoints under /api/*), auth.py, security.py, server.py
├── cli/            # menu.py — interactive CLI, 117 options
├── notebook/       # Jupyter magics + ipywidgets (optional IPython deps, guarded)
└── plotting/       # 12 modules, matplotlib + plotly backends
tests/              # 460 files
benchmarks/         # pytest-benchmark suite
examples/           # 8 tutorial scripts + 5 notebooks
docs/               # Sphinx
```

## Key Files
- `run.py` — GUI launcher entry point
- `biosuite/__init__.py` — **single source of truth for `__version__`**
- `biosuite_config.json` — user configuration (theme, API keys)
- `biotools.json` — external tool definitions
- `pyproject.toml` — package metadata; extras: `bio`, `gui`, `cloning`, `api`, `notebook`, `parallel`, `full`, `dev`

## Version Policy
`biosuite/__init__.py::__version__` is the only hard-coded version string in Python code.
`pyproject.toml`, `CITATION.cff`, `docs/conf.py`, `README.md` badge and the CHANGELOG must be
updated together on a release. The API, CLI banner, `--version` flag and GUI labels all import
`__version__` — never hard-code a version in them again.

## Code Style
- Python 3.10+ (CI matrix: 3.10, 3.11, 3.12)
- Type hints on public functions, Google-style docstrings
- Max line length 100
- Plotting functions must **return the figure** and never call `plt.show()` or `plt.close()` —
  the caller (GUI/CLI) owns display and cleanup
- Heavy/optional deps (pymc, torch, openmm, scanpy, cobra) must be imported lazily inside
  functions, never at module level

## Build & Test
```bash
pip install -e ".[dev]"
pytest tests/ -q                 # full suite: see Current Status for the measured run
pytest tests/test_sequence.py -v # single file
pytest tests/ --cov=biosuite     # coverage
```
The API tests need `fastapi` + `httpx`; they read the key from `BIOSUITE_API_KEY`.

## Current Status
- Version: **6.0.0**
- Last measured run: **9,068 passed, 17 skipped, 0 failed** in 16:09 on Python 3.13.3,
  collected 9,085 (`coverage run --branch --source=biosuite -m pytest tests -q`, this
  checkout — the instrumented run is the one quoted, so the coverage figures below come
  from the same pass; `audit_output/PROGRESS.md` records how each cluster's run was
  measured and re-pinned).
- Branch coverage of `biosuite/`: **100 %** — 21,711 statements with 0 missed and
  7,022 branches with 0 partial, measured over the same tree with
  `coverage run --branch --source=biosuite -m pytest tests -q`.
- The 17 skips are 13 optional scientific extras (7 scanpy, 4 goatools, 2 ESM/torch),
  2 tests that need POSIX mode bits, and 2 modules whose engine (`scanpy`, `goatools`)
  is absent; they are reported by `pytest -rs` and are not deferred failures.
- CI runs the same suite on 3.10/3.11/3.12 with Tk bindings installed, plus lint,
  security, coverage and package-build jobs (`.github/workflows/ci.yml`). The figures
  above are from the local run; a CI log is not readable from this checkout, so no CI
  count is quoted here. The two `security` job file lists were re-run here as well:
  257 tests, 1 skipped, no failures.
- Published on PyPI as `biosuite-ultra`
- Zenodo DOI: 10.5281/zenodo.21256296

## Rules
- Never break existing tests
- Always add tests for new functions
- Keep backward compatibility
- New domain logic goes under `biosuite/core/`; never mix GUI code into core
- Do not commit build artifacts, `.bak` files, PNGs or release archives — they are gitignored
