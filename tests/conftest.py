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

# The same trick for everything that resolves `os.path.expanduser('~')`: log.py,
# plugin.py and enrichment.py build ~/.biosuite/... paths from the home
# directory rather than from BIOSUITE_CONFIG_DIR, so without this the suite
# writes its daily log file, its plugin config and its GO cache into the
# developer's real home.  Measured on 2026-10-07: a plugin test overwrote
# ~/.biosuite/plugins.json.  Home is redirected before the first biosuite import,
# because log.py resolves the path once, when its file handler is installed.
_FAKE_HOME = tempfile.mkdtemp(prefix="biosuite-tests-home-")
os.environ["HOME"] = _FAKE_HOME
os.environ["USERPROFILE"] = _FAKE_HOME

# Deterministic test credentials.  ``setdefault`` keeps an operator's own
# values (e.g. in CI) working.
os.environ.setdefault("BIOSUITE_API_KEY", "test-api-key-not-a-real-secret")
os.environ.setdefault("BIOSUITE_JWT_SECRET", "test-jwt-secret-not-a-real-secret")
os.environ.setdefault("BIOSUITE_ADMIN_PASSWORD", "test-admin-password")


@pytest.fixture(autouse=True)
def _isolated_user_config(tmp_path, monkeypatch):
    """Point per-user state at a fresh directory for every test."""
    monkeypatch.setenv("BIOSUITE_CONFIG_DIR", str(tmp_path / "biosuite-config"))


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """A home directory the test may fill in, for code that resolves '~'.

    Plugin discovery reads ``~/.biosuite/plugins`` and stores its state in
    ``~/.biosuite/plugins.json``; a test that wants to plant a plugin, or to
    assert what a corrupt config does, needs its own home rather than the
    session-wide one.
    """
    home = tmp_path / "home"
    (home / ".biosuite").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@pytest.fixture(autouse=True)
def _release_figures():
    """Close every figure a test leaves behind.

    Figures live in a process-global pyplot registry, and 58 tests across
    ``tests/core``, ``tests/integration`` and ``tests/plotting`` open one without
    closing it.  Accumulated, that reached pyplot's 20-figure warning; measured
    per test, it means a test that asserts "no figure left open" is really
    measuring whatever earlier tests leaked.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")

