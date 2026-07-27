# Contributing

## Development Setup

```bash
git clone https://github.com/sahandtkod-byte/BioSuite-Ultra.git
cd BioSuite-Ultra
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v --tb=short
```

## Code Quality

```bash
ruff check biosuite/ --fix    # Lint
ruff format biosuite/         # Format
mypy biosuite/ --ignore-missing-imports  # Type check
bandit -r biosuite/ -ll       # Security scan
```

## Commit Convention

```
type(scope): description

feat: add new feature
fix: bug fix
refactor: code restructuring
docs: documentation
test: adding tests
chore: maintenance
```
