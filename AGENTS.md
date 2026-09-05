# BioSuite Ultra — Agent Instructions

## Project Overview
BioSuite Ultra is a pure-Python bioinformatics platform. All numbers below were
measured from the tree — keep them accurate if you change the code.

- **47 analysis modules** in `biosuite/core/` (+ `biosuite/core/workflow/`)
- **12 plotting modules** exposing **123 functions** (105 public); the GUI plot catalogue lists **40 plot types** in 9 categories
- **11 GUI tabs**, **99 CLI menu options**, **38 REST API endpoints** (under `/api/*`)
- **169 restriction enzymes** (`RESTRICTION_ENZYMES` in `biosuite/core/utils.py`)
- **~32.5K lines** of Python in `biosuite/`
- **130 test files**; CI reports **2,477 passed, 14 skipped** on 3.10/3.11/3.12

## Architecture
```
biosuite/
├── core/           # 44 domain modules + core/workflow/ (3 modules)
├── gui/            # customtkinter app: main_window.py, themes.py, dialogs.py, tabs/ (11)
├── api/            # FastAPI app: __init__.py (38 endpoints under /api/*), auth.py, security.py, server.py
├── cli/            # menu.py — interactive CLI, 99 options
├── notebook/       # Jupyter magics + ipywidgets (optional IPython deps, guarded)
└── plotting/       # 12 modules, matplotlib + plotly backends
tests/              # 130 files
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
pytest tests/ -q                 # full suite: 2,477 passed + 14 skipped in CI
pytest tests/test_sequence.py -v # single file
pytest tests/ --cov=biosuite     # coverage
```
The API tests need `fastapi` + `httpx`; they read the key from `BIOSUITE_API_KEY`.

## Current Status
- Version: **5.5.0**
- Tests in CI: **2,477 passed, 14 skipped, 0 failed** on 3.10/3.11/3.12
- Tests locally without tkinter: **2,343 passed, 141 failed, 14 skipped** — the 141 GUI tests
  cannot import tkinter in a headless checkout; they run and pass in the matrix
- Published on PyPI as `biosuite-ultra`
- Zenodo DOI: 10.5281/zenodo.21256296

## Rules
- Never break existing tests
- Always add tests for new functions
- Keep backward compatibility
- New domain logic goes under `biosuite/core/`; never mix GUI code into core
- Do not commit build artifacts, `.bak` files, PNGs or release archives — they are gitignored
