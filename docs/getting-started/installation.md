# Installation

## Requirements

- Python 3.10 or later
- pip or uv package manager

## Install from PyPI

```bash
pip install biosuite-ultra
```

## Install from Source

```bash
git clone https://github.com/sahandtkod-byte/BioSuite-Ultra.git
cd BioSuite-Ultra
pip install -e .
```

## Optional Dependencies

BioSuite uses optional dependency groups to keep the base install lightweight:

| Group | Command | What it includes |
|-------|---------|------------------|
| **api** | `pip install biosuite-ultra[api]` | FastAPI, uvicorn, JWT auth, rate limiting |
| **gui** | `pip install biosuite-ultra[gui]` | PyQt6 desktop application |
| **notebook** | `pip install biosuite-ultra[notebook]` | IPython magics, ipywidgets |
| **bio** | `pip install biosuite-ultra[bio]` | GO tools, GSEA, Scanpy, ete3 |
| **dev** | `pip install biosuite-ultra[dev]` | pytest, ruff, mypy, bandit |
| **full** | `pip install biosuite-ultra[full]` | Everything above |

## Docker

```bash
docker compose up api     # Start REST API on port 8000
docker compose --profile gui up gui   # Start GUI (headless)
```

## Verify Installation

```bash
python -c "import biosuite; print(biosuite.__version__)"
# 5.0.0
```
