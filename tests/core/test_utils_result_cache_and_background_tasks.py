"""The result cache's key function, its LRU/TTL paths, and a background task that dies.

`CachedResult` has to answer three questions without ever calling the wrapped
function twice for the same input: what is a key (arrays are hashed by content, not
identity), is this entry still fresh, and who gets evicted when the cache is full.
Each is a separate branch, and the expired-entry tests build their own ages by
hand rather than sleeping, so they fail only when the logic is wrong.
"""
import logging
import time

import numpy as np
import pytest

from biosuite.core import utils as ut


class Counting:
    """A callable that counts how often the wrapped function actually ran."""

    def __init__(self):
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return value * 2


# ─── what makes two calls the same call ──────────────────────────────────────

@pytest.mark.parametrize("value, kind", [
    (b"abc", "bytes"),
    (bytearray(b"abc"), "bytes"),
    ("text", "str"),
    (7, "int"),
    (7.5, "float"),
    (True, "bool"),
    (None, "NoneType"),
    ([1, 2], "list"),
    ((1, 2), "tuple"),
    ({"a": 1}, "dict"),
    ({1, 2}, "set"),
])
def test_a_key_part_names_the_type_it_hashed(value, kind):
    assert ut.CachedResult._key_part(value)[0] == kind


def test_two_equal_arrays_get_the_same_key_and_a_different_array_does_not():
    key_part = ut.CachedResult._key_part
    a = np.arange(6.0)
    b = np.arange(6.0)
    c = np.arange(6.0).reshape(2, 3)

    assert key_part(a) == key_part(b), "content, not identity, is the key"
    assert key_part(a)[1:] != key_part(c)[1:], "the shape is part of the key"
    assert key_part(a)[0] == "ndarray"


def test_an_object_that_can_be_dumped_is_keyed_on_its_dump():
    class Frame:
        def to_dict(self):
            return {"x": [1, 2]}

    assert ut.CachedResult._key_part(Frame())[0] == "to_dict"

    class Broken:
        def to_dict(self):
            raise RuntimeError("no dump for you")

    assert ut.CachedResult._key_part(Broken())[0] == "repr", \
        "an object that cannot be dumped falls back to its type, not to a crash"


def test_an_object_without_a_dump_is_keyed_on_its_repr():
    class Plain:
        pass

    value = Plain()
    key = ut.CachedResult._key_part(value)

    assert key[0] == "repr" and key[1] == "Plain"
    assert key[2] == repr(value)


def test_a_cached_call_runs_the_function_once_per_key():
    counter = Counting()
    cached = ut.CachedResult(counter)

    assert cached(3) == 6
    assert cached(3) == 6
    assert cached(4) == 8

    assert counter.calls == [3, 4], "the second identical call was recomputed"
    assert len(cached.cache) == 2


# ─── freshness and eviction ──────────────────────────────────────────────────

def _age(cache, key, seconds):
    """Backdate one entry so the ttl logic can be tested without waiting."""
    value, created = cache.cache[key]
    cache.cache[key] = (value, created - seconds)


def test_a_hit_refreshes_recency_and_an_expired_entry_is_recomputed():
    counter = Counting()
    cached = ut.CachedResult(counter, maxsize=2, ttl=100.0)
    key = cached.make_key((3,), {})

    cached(3)
    cached(3)
    assert cached.access_order == [key], "a hit does not duplicate the recency entry"

    _age(cached, key, 500)
    assert cached(3) == 6, "the entry was recomputed, not returned stale"
    assert counter.calls == [3, 3], "an expired entry must not count as fresh"


def test_is_expired_knows_a_missing_key_and_an_entry_with_no_deadline():
    counter = Counting()
    timed = ut.CachedResult(counter, ttl=100.0)
    forever = ut.CachedResult(counter)

    assert timed.is_expired("never-stored") is True
    timed.cache["fresh"] = (1, time.monotonic())
    assert timed.is_expired("fresh") is False

    forever.cache["kept"] = (1, time.monotonic() - 10_000)
    assert forever.is_expired("kept") is False, "no ttl means no entry is ever old"
    assert forever.is_expired("absent") is True, "a key that is not there cannot be fresh"


def test_expired_entries_are_cleaned_out_on_access():
    counter = Counting()
    cached = ut.CachedResult(counter, maxsize=8, ttl=1.0)
    cached(1)
    cached(2)
    _age(cached, cached.make_key((1,), {}), 30)

    now = time.monotonic()
    cached._evict_expired(now)

    assert cached.access_order == [cached.make_key((2,), {})]
    assert list(cached.cache) == [cached.make_key((2,), {})], \
        "an evicted entry must leave both the cache and the recency list"


def test_a_full_cache_drops_the_least_recently_used_entry():
    counter = Counting()
    cached = ut.CachedResult(counter, maxsize=2)

    cached(1)
    cached(2)
    cached(1)          # 1 is now the most recent
    cached(3)

    assert len(cached.cache) == 2
    assert cached.make_key((2,), {}) not in cached.cache, "2 was the oldest"
    assert set(cached.access_order) == {cached.make_key((1,), {}), cached.make_key((3,), {})}


def test_a_cache_whose_recency_list_ran_empty_still_frees_a_slot():
    counter = Counting()
    cached = ut.CachedResult(counter, maxsize=1)
    cached(1)
    cached.access_order.clear()      # the state the defence exists for

    cached(2)

    assert list(cached.cache) == [cached.make_key((2,), {})]


# ─── a background task that raises ───────────────────────────────────────────

def test_a_failing_background_task_tells_the_callback_it_failed(caplog):
    def boom():
        raise ValueError("no")

    seen = []
    with caplog.at_level(logging.ERROR, logger="biosuite.core.utils"):
        ut.run_in_background(boom, callback=seen.append).join(timeout=5)

    assert seen == [None], "the callback is told the task produced nothing"
    assert "Background task boom failed: ValueError: no" in caplog.text, caplog.text
    records = [r for r in caplog.records if r.name == "biosuite.core.utils"]
    assert [r.exc_info[0] for r in records] == [ValueError], \
        "the failure is reported with its traceback"


def test_a_callback_that_raises_is_logged_and_does_not_lose_the_result(caplog):
    """The result was already decided; the report on the way out must not vanish
    into an "Exception in thread" printed by the interpreter."""
    def explodes(_value):
        raise RuntimeError("callback broke")

    with caplog.at_level(logging.ERROR):
        ut.run_in_background(lambda: 7, callback=explodes).join(timeout=5)
    records = [r for r in caplog.records if r.name == "biosuite.core.utils"]
    assert [r.getMessage() for r in records] == [
        "Background callback explodes failed: RuntimeError: callback broke"]
    assert records[0].exc_info[0] is RuntimeError


def test_a_background_task_without_a_callback_still_finishes():
    seen = []

    ut.run_in_background(seen.append, 5).join(timeout=5)

    assert seen == [5]
