"""Shared pytest configuration.

Two jobs:

1. Keep the test-suite from touching (or writing into) the developer's real
   environment: matplotlib runs headless and BioSuite's user configuration is
   redirected into a per-session temporary directory.  Historically the config
   lived next to the source tree, so running the tests mutated — and got the
   API-key placeholders committed into — the git-tracked
   ``biosuite_config.json`` (see BSU-004).
2. Give the API tests a deterministic, *explicitly configured* set of
   credentials.  They are set here rather than baked into the application so
   that the production code paths keep failing closed when nothing is
   configured.
"""
from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

# A per-run configuration directory, created before any biosuite import so that
# module-level constants pick it up.
_CONFIG_DIR = tempfile.mkdtemp(prefix="biosuite-tests-config-")
os.environ["BIOSUITE_CONFIG_DIR"] = _CONFIG_DIR

# Deterministic test credentials.  ``setdefault`` keeps an operator's own
# values (e.g. in CI) working.
os.environ.setdefault("BIOSUITE_API_KEY", "test-api-key-not-a-real-secret")
os.environ.setdefault("BIOSUITE_JWT_SECRET", "test-jwt-secret-not-a-real-secret")
os.environ.setdefault("BIOSUITE_ADMIN_PASSWORD", "test-admin-password")


@pytest.fixture(autouse=True)
def _isolated_user_config(tmp_path, monkeypatch):
    """Point per-user state at a fresh directory for every test."""
    monkeypatch.setenv("BIOSUITE_CONFIG_DIR", str(tmp_path / "biosuite-config"))
