# Changelog

All notable changes to BioSuite Ultra will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.0] - 2026-07-27

### 🔒 Security (CRITICAL)
- **Removed all hardcoded credentials** — `admin/biosuite123` default removed from `api/security.py`
- **JWT secret** auto-generated on first run, stored in `~/.biosuite/.jwt_secret`
- **API key** now required via `BIOSUITE_API_KEY` env var (no default)
- **Admin credentials** required via `BIOSUITE_ADMIN_USER` + `BIOSUITE_ADMIN_PASSWORD` env vars
- **Password hashing** — PBKDF2-SHA256 (stdlib, no extra deps) with constant-time verification
- **Constant-time API key comparison** via `hmac.compare_digest()` (prevents timing attacks)
- **Replaced 15 `tempfile.mktemp()` calls** with secure `tempfile.mkstemp()` across 8 files

### ⚡ Performance
- **Lazy imports** — core module import time reduced from **10.65s → 0.33s** (32x faster)
- Heavy dependencies (numpy, pandas, scipy, matplotlib) loaded on-demand, not at import time
- CLI startup now under 0.5s

### 🚀 New Features
- **Streaming FASTA/FASTQ readers** — `iter_fasta()` and `iter_fastq()` generators for memory-efficient parsing of large files
- **Jupyter notebook modules** — `biosuite.notebook.magics` and `biosuite.notebook.widgets` as separate importable modules
- **Structured logging** — log rotation (10MB/5 backups), JSON formatter option, configurable via `BIOSUITE_LOG_LEVEL`
- **Request size limits** — API rejects requests >50MB (configurable via `BIOSUITE_MAX_REQUEST_SIZE_MB`)
- **Sequence length validation** — API enforces max 10M characters on sequence inputs
- **`_DictWrapper`** — config/session dicts now auto-persist changes

### 🏗️ Architecture
- **Exception-based validation** — all validators raise `ValueError` (consistent with Python conventions)
- **Pydantic models** for API request/response validation with field constraints
- **Optional dependency groups** — `pip install biosuite-ultra[api,gui,notebook,bio,dev]`
- **Mypy config** added to `pyproject.toml`

### 🧪 Testing
- **63 tests** across 5 test files (sequence, alignment, expression, notebook, security)
- Parametrized tests for sequence operations, alignment, and normalization
- Security tests for password hashing and verification
- Streaming reader tests

### 📦 DevOps
- **GitHub Actions CI** — lint (ruff), type-check (mypy), security scan (bandit), tests (pytest on Python 3.10-3.12)
- **Docker** — multi-stage Dockerfile (API + GUI), docker-compose.yml
- **Benchmarks** — pytest-benchmark suite for core operations
- **ADRs** — 5 architecture decision records documenting key choices

### 🔧 Internal
- Version bumped to **5.0.0** (breaking: removed hardcoded credentials, requires env vars)
- Updated `pyproject.toml` with optional dependencies, dev tools, and classifiers
- `README.md` updated with new features and setup instructions
