# Contributing to BioSuite Ultra

Thank you for your interest in contributing to BioSuite Ultra! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)

---

## Code of Conduct

Be respectful, inclusive, and constructive. We're here to build great software together.

---

## How to Contribute

### Types of Contributions

1. **Bug Fixes** - Fix issues in existing modules
2. **New Features** - Add new analysis modules or improve existing ones
3. **Documentation** - Improve docs, tutorials, or examples
4. **Tests** - Add or improve test coverage
5. **Performance** - Optimize existing code

### First-Time Contributors

Look for issues labeled:
- `good first issue` - Simple fixes perfect for beginners
- `help wanted` - Issues where we need community help
- `documentation` - Documentation improvements

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/BioSuite-Ultra.git
cd BioSuite-Ultra
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 4. Create Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Code Style

### General Rules

1. **Pure Python First** - All code must work with just `pip install`
2. **No External Binaries Required** - External tools are optional speed boosts
3. **Dual-Mode Pattern** - Every module should have:
   - `_run_builtin()` - Pure Python fallback
   - `_run_external()` - Optional external tool
   - `check_*_tools()` - Detect available tools

### Python Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to all public functions
- Keep functions under 50 lines when possible
- Use meaningful variable names

### Example Module Structure

```python
"""
Module description.

Provides analysis of [topic] using [methods].
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Constants
CONSTANT_VALUE = 42

# Data classes
@dataclass
class Result:
    """Result of analysis."""
    value: float
    message: str = ""

# Private functions
def _run_builtin(input_data, **kwargs):
    """Pure Python implementation."""
    # Always works, no external dependencies
    pass

def _run_external(input_data, **kwargs):
    """External tool implementation (optional)."""
    # Faster but requires external tool
    pass

def check_tools():
    """Check which tools are available."""
    return {'external_tool': _has_tool('tool_name')}

# Public API
def analyze(input_data, tool='auto'):
    """Analyze input data.
    
    Args:
        input_data: Input data
        tool: 'auto', 'external', or 'builtin'
    
    Returns:
        Result object
    """
    if tool in ('external', 'auto') and check_tools()['external_tool']:
        result = _run_external(input_data)
        if result:
            return Result(value=result, message="Using external tool")
    
    return Result(value=_run_builtin(input_data), message="Using builtin")
```

---

## Testing

### Writing Tests

1. **One test file per module** - `test_module_name.py`
2. **Test both builtin and external** - When external tools available
3. **Test edge cases** - Empty inputs, invalid data, etc.
4. **Use fixtures** - Create reusable test data
5. **Aim for 80%+ coverage**

### Example Test

```python
"""Tests for module_name."""
import pytest
from biosuite.core.module_name import analyze, _run_builtin


class TestAnalyze:
    def test_basic_analysis(self):
        result = analyze("test_input")
        assert result is not None
        assert result.value > 0
    
    def test_empty_input(self):
        result = analyze("")
        assert result is not None
    
    def test_invalid_input(self):
        with pytest.raises(ValueError):
            analyze(None)


class TestBuiltin:
    def test_builtin_returns_result(self):
        result = _run_builtin("test_input")
        assert result is not None
```

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_module_name.py -v

# With coverage
python -m pytest tests/ --cov=biosuite.core.module_name

# Parallel (faster)
python -m pytest tests/ -n auto
```

---

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated if needed
- [ ] No sensitive data (API keys, passwords) committed

### 2. PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Test addition

## Testing
- [ ] All existing tests pass
- [ ] New tests added
- [ ] Tested locally

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

### 3. Review Process

1. Automated tests run on your PR
2. Maintainer reviews code
3. Address any feedback
4. Merge when approved

---

## Reporting Bugs

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Import '...'
2. Run function '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment**
- OS: [e.g., Windows 10, Ubuntu 22.04]
- Python version: [e.g., 3.11.0]
- BioSuite version: [e.g., 4.1.0]

**Additional context**
Any other context about the problem.
```

---

## Requesting Features

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem. Ex. "I'm always frustrated when..."

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Add any other context about the feature request.
```

---

## Project Structure

```
BioSuite-Ultra/
├── biosuite/              # Main package
│   ├── core/                # Analysis modules
│   ├── plotting/            # Visualization modules
│   ├── gui/                 # GUI components
│   ├── cli/                 # CLI menu
│   ├── api/                 # REST API
│   └── notebook/            # Jupyter integration
├── tests/                   # Test suite
├── examples/                # Tutorials and examples
├── docs/                    # Documentation
├── run.py                   # Entry point
├── pyproject.toml           # Package config
└── requirements.txt         # Dependencies
```

---

## Key Principles

1. **100% Free** - No paid dependencies, ever
2. **Pure Python First** - Works with just `pip install`
3. **Dual-Mode** - External tools optional, builtin always works
4. **Well-Tested** - Comprehensive test coverage
5. **Well-Documented** - Clear docstrings and examples

---

## Questions?

Open a GitHub issue or start a discussion. We're happy to help!

---

Thank you for contributing to BioSuite Ultra! 🧬
