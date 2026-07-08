# BioSuite PyPI Deployment Guide

## Project Info
- **Package name:** `biosuite-ultra`
- **Internal module:** `biosuite`
- **PyPI page:** https://pypi.org/project/biosuite-ultra/
- **Source:** `C:\Users\SAHAND\Desktop\python\BioSuite-Ultra`
- **Python:** 3.9+
- **Build system:** setuptools + wheel

## How to Update and Publish

### Step 1: Bump version in pyproject.toml
Open `pyproject.toml` and change the version number. PyPI rejects duplicate versions.

```toml
version = "X.Y.Z"
```

Version rules:
- Patch (bugfix): 4.1.0 → 4.1.1
- Minor (new features): 4.1.0 → 4.2.0
- Major (breaking changes): 4.1.0 → 5.0.0

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
**v4.1.0** — July 7, 2026

### What's New
- Parallel processing module
- 100+ restriction enzymes
- Better Bayesian phylogeny
- Improved MD simulation
- 30+ bug fixes
- New documentation (CHANGELOG.md, CONTRIBUTING.md)
