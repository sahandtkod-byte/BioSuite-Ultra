"""Regression tests for caching and user-configuration handling.

* BSU-008 - ``CachedResult`` keyed entries on ``str(args)``.  numpy abbreviates
  large arrays, so two different arrays produced the same key and one call's
  result was returned for the other's input.  Eviction also ignored recency,
  so it was not an LRU despite the ``access_order`` bookkeeping.
* BSU-004 - user configuration (including API keys) was written next to the
  source tree into a git-tracked file.
"""
import json
import os

import numpy as np
import pytest

from biosuite.core import utils


# ── BSU-008: cache keys ─────────────────────────────────────────────────────

def test_large_arrays_that_stringify_identically_are_distinct_keys():
    calls = []

    def total(array):
        calls.append(array.copy())
        return float(array.sum())

    cached = utils.CachedResult(total)
    a = np.arange(10_000, dtype=float)
    b = a.copy()
    b[5_000] = 999_999.0          # differs only inside numpy's "..." elision

    assert str(a) == str(b), "precondition: the old key function collides here"
    assert cached(a) == a.sum()
    assert cached(b) == b.sum()
    assert len(calls) == 2, "the second array was served from the first's entry"


def test_cache_returns_the_stored_value_on_a_genuine_hit():
    calls = []
    cached = utils.CachedResult(lambda x: calls.append(x) or x * 2)
    a = np.arange(1000, dtype=float)
    assert cached(a) is not None
    cached(a.copy())              # equal contents -> same key
    assert len(calls) == 1


def test_distinct_dtypes_and_shapes_do_not_collide():
    calls = []
    cached = utils.CachedResult(lambda x: calls.append(x) or x.dtype.str)
    assert cached(np.zeros(4, dtype=np.int64)) == '<i8'
    assert cached(np.zeros(4, dtype=np.float64)) == '<f8'
    assert cached(np.zeros((2, 2), dtype=np.float64)) == '<f8'
    assert len(calls) == 3


def test_kwargs_are_part_of_the_key():
    calls = []

    def f(x, scale=1):
        calls.append((x, scale))
        return x * scale

    cached = utils.CachedResult(f)
    assert cached(3, scale=1) == 3
    assert cached(3, scale=2) == 6
    assert len(calls) == 2


def test_eviction_is_least_recently_used_not_least_recently_inserted():
    calls = []
    cached = utils.CachedResult(lambda x: calls.append(x) or x, maxsize=2)
    cached(1)
    cached(2)
    cached(1)          # refreshes 1; 2 is now the least recently used
    cached(3)          # must evict 2, not 1
    assert len(cached) == 2
    cached(1)
    assert calls == [1, 2, 3], "key 1 was evicted despite being used last"


def test_cache_respects_maxsize():
    cached = utils.CachedResult(lambda x: x, maxsize=3)
    for i in range(20):
        cached(i)
    assert len(cached) <= 3


def test_ttl_expiry_still_works():
    calls = []
    cached = utils.CachedResult(lambda x: calls.append(x) or x, ttl=0.0)
    cached(1)
    cached(1)
    assert len(calls) == 2


# ── BSU-004: configuration location ─────────────────────────────────────────

def test_config_is_written_to_the_user_config_dir_not_the_repository(tmp_path,
                                                                     monkeypatch):
    monkeypatch.setenv("BIOSUITE_CONFIG_DIR", str(tmp_path))
    cfg = utils.load_config()
    cfg["theme"] = "unit-test-theme"
    utils.save_config(cfg)

    written = tmp_path / "biosuite_config.json"
    assert written.exists()
    assert json.loads(written.read_text())["theme"] == "unit-test-theme"

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(utils.__file__))))
    assert not os.path.exists(os.path.join(repo_root, "biosuite_config.json")), \
        "configuration must never be written into the source tree"


def test_saved_config_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOSUITE_CONFIG_DIR", str(tmp_path))
    utils.save_config({**utils.DEFAULT_CONFIG, "marker": "xyz"})
    assert utils.load_config()["marker"] == "xyz"


def test_api_keys_are_not_world_readable(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOSUITE_CONFIG_DIR", str(tmp_path))
    utils.save_config({**utils.DEFAULT_CONFIG,
                       "api_keys": {"ncbi_api_key": "secret"}})
    mode = os.stat(tmp_path / "biosuite_config.json").st_mode & 0o777
    assert mode == 0o600, f"config holding API keys is mode {oct(mode)}"


def test_save_config_reports_failure_instead_of_swallowing_it(tmp_path,
                                                              monkeypatch):
    """`except OSError: pass` told the user their API key had been saved."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setenv("BIOSUITE_CONFIG_DIR", str(blocked / "sub"))
    with pytest.raises(OSError):
        utils.save_config(dict(utils.DEFAULT_CONFIG))


def test_config_directory_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("BIOSUITE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    if os.name != "nt":
        assert utils.user_config_dir() == str(tmp_path / "biosuite")
