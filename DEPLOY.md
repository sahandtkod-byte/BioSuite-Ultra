# BioSuite PyPI Deployment Guide

## Project Info
- **Package name:** `biosuite-ultra`
- **Internal module:** `biosuite`
- **PyPI page:** https://pypi.org/project/biosuite-ultra/
- **Source:** `C:\Users\SAHAND\Desktop\python\BioSuite-Ultra`
- **Python:** 3.10+ (tested on 3.10, 3.11, 3.12)
- **Build system:** setuptools + wheel

## How to Update and Publish

### Step 1: Bump version in pyproject.toml
Open `pyproject.toml` and change the version number. PyPI rejects duplicate versions.

```toml
version = "X.Y.Z"
```

Version rules:
- Patch (bugfix): 5.5.0 → 5.5.1
- Minor (new features): 5.5.0 → 5.6.0
- Major (breaking changes): 5.5.0 → 6.0.0

### Step 2: Clean old build files
```powershell
Remove-Item "dist\biosuite-ultra*" -Force -ErrorAction SilentlyContinue
Remove-Item "build" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "biosuite-ultra.egg-info" -Recurse -Force -ErrorAction SilentlyContinue
```

### Step 3: Build
```bash
cd C:\Users\SAHAND\Desktop\python\BioSuite-Ultra
python -m build
```

Creates two files in `dist/`:
- `biosuite-ultra-X.Y.Z-py3-none-any.whl`
- `biosuite-ultra-X.Y.Z.tar.gz`

### Step 4: Upload to PyPI
```bash
twine upload dist/biosuite-ultra-X.Y.Z*
```

Credentials are stored in `C:\Users\SAHAND\.pypirc` (API token). No prompt needed.

### Step 5: Verify
- Check: https://pypi.org/project/biosuite-ultra/X.Y.Z/
- Test install: `pip install biosuite-ultra==X.Y.Z`

## pyproject.toml Structure

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "biosuite-ultra"
version = "X.Y.Z"          # <-- CHANGE THIS EACH TIME
description = "..."
dependencies = [...]        # All deps go here (not optional)

[project.scripts]
biosuite = "biosuite.cli.menu:main_cli"
biosuite-gui = "biosuite.cli.menu:main_cli_gui"

[tool.setuptools.packages.find]
include = ["biosuite*"]
```

## User Install Commands
```bash
pip install biosuite-ultra              # basic install
pip install biosuite-ultra==X.Y.Z      # specific version
pip install --upgrade biosuite-ultra    # get latest
pip install "biosuite-ultra[full]"      # all optional features
```

## After Install, Users Get
- `biosuite` command → CLI menu
- `biosuite-gui` command → launches GUI
- `from biosuite.core.sequence import ...` → Python API
- `from biosuite.core.parallel import ...` → Parallel processing

## Common Errors

| Error | Fix |
|-------|-----|
| `File already exists` | Version already on PyPI. Bump version number. |
| `Invalid distribution` | Clean old files from `dist/` before building. |
| `403 Forbidden` | API token expired. Create new at https://pypi.org/manage/account/token/ |
| `twine: command not found` | Run `pip install twine` |

## Current Version
**v5.5.0** — see [CHANGELOG.md](CHANGELOG.md) for the full entry.

Note that 5.5.0 is the version in this source tree; the latest version published to PyPI is
5.0.3.

### What's New
- Scientific correctness fixes to pairwise alignment, ChIP-seq peak calling and coverage,
  docking, multiple sequence alignment and differential expression
- Security hardening: no default credentials, confined file access, restricted CORS,
  authenticated documentation endpoints
- 169 restriction enzymes for molecular cloning
- Reproducible workflows with provenance tracking
- CI on Python 3.10, 3.11 and 3.12 with Ruff, a security regression suite and CodeQL
