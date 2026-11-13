"""The shared reporting helpers in ``utils`` and the two bookkeeping states the
cache has to survive.

Each ``report_*`` helper is the text a user reads under a plot, so the numbers it
prints are compared with an independent computation rather than with itself.  The
cache tests build states the normal call path cannot produce — an entry present in
the map but missing from the recency list — because that is exactly when the
bookkeeping has to keep working instead of raising.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from biosuite.core import utils as ut


# ─── the scatter report ──────────────────────────────────────────────────────

def test_a_single_point_cannot_be_correlated_and_says_so(caplog):
    with caplog.at_level(logging.INFO, logger="biosuite.core.utils"):
        assert ut.report_scatter_stats([1.0], [2.0]) is None

    assert "Insufficient data for correlation" in caplog.text, caplog.text


def test_the_scatter_report_carries_the_pearson_coefficient_and_its_p_value():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 5.5, 7.0, 10.5]

    reported = ut.report_scatter_stats(x, y)
    expected = sp_stats.pearsonr(x, y)

    assert reported['r'] == pytest.approx(expected.statistic)
    assert reported['p'] == pytest.approx(expected.pvalue)


# ─── the two plot annotations ────────────────────────────────────────────────

def test_three_groups_get_an_anova_bracket_instead_of_a_ttest():
    frame = pd.DataFrame({"group": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
                          "value": [1.0, 1.2, 0.9, 1.1,
                                    5.0, 5.2, 4.9, 5.1,
                                    9.0, 9.2, 8.9, 9.1]})
    fig, ax = plt.subplots()

    ut.add_ttest_to_boxplot(frame, "group", "value", ax)

    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 1, texts
    assert texts[0].startswith("ANOVA p = "), texts

    _, expected = sp_stats.f_oneway(*[frame[frame.group == g]["value"] for g in "abc"])
    assert texts[0] == f"ANOVA p = {expected:.4f}", (texts, expected)
    plt.close(fig)


def test_two_groups_get_the_pairwise_test():
    frame = pd.DataFrame({"group": ["a"] * 4 + ["b"] * 4,
                          "value": [1.0, 1.2, 0.9, 1.1, 5.0, 5.2, 4.9, 5.1]})
    fig, ax = plt.subplots()

    ut.add_ttest_to_boxplot(frame, "group", "value", ax)

    assert [t.get_text() for t in ax.texts] == [
        f"t-test p = {sp_stats.ttest_ind(frame[frame.group == 'a']['value'],
                                         frame[frame.group == 'b']['value']).pvalue:.4f}"]
    plt.close(fig)


def test_the_regression_equation_is_written_out_with_its_fit():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 3.0, 5.0, 7.0]
    fig, ax = plt.subplots()

    ut.add_regression_eq(x, y, ax)

    text = ax.texts[0].get_text()
    assert text.splitlines()[0] == "y = 2.000x + 1.000", text
    assert text.splitlines()[1].startswith("R² = 1.000"), text
    plt.close(fig)


# ─── the glass panel on an axes that has no spines ───────────────────────────

def test_a_polar_axes_has_no_top_spine_and_still_gets_styled():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="polar")

    ut.apply_glass_ax(ax)          # must not raise on the missing 'top' spine

    assert len(ax.patches) == 0, "no panel is added where there is no frame to match"
    plt.close(fig)


# ─── the cache's awkward bookkeeping states ─────────────────────────────────

class Counter:
    def __init__(self):
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return value


def test_an_entry_missing_from_the_recency_list_is_still_found_and_re_listed():
    cached = ut.CachedResult(Counter(), maxsize=4, ttl=100.0)
    key = cached.make_key((1,), {})
    cached.cache[key] = ("kept", __import__("time").monotonic())

    assert cached(1) == "kept"
    assert cached.access_order == [key], "the hit put the key back where it belongs"


def test_an_expired_entry_absent_from_the_recency_list_is_recomputed():
    import time

    counter = Counter()
    cached = ut.CachedResult(counter, maxsize=4, ttl=10.0)
    key = cached.make_key((2,), {})
    cached.cache[key] = ("stale", time.monotonic() - 99)

    assert cached(2) == 2, "the value was computed, not the stale one returned"
    assert counter.calls == [2]


def test_expired_entries_are_dropped_from_both_records_when_the_clock_is_left_out():
    cached = ut.CachedResult(Counter(), maxsize=4, ttl=10.0)
    key = cached.make_key((3,), {})
    cached.cache[key] = (1, 0.0)              # created at the epoch: long expired
    cached.access_order.append(key)

    cached._evict_expired()                   # no explicit now

    assert cached.cache == {} and cached.access_order == []


def test_an_expired_key_that_is_not_in_the_recency_list_does_not_raise():
    cached = ut.CachedResult(Counter(), maxsize=4, ttl=10.0)
    cached.cache[cached.make_key((4,), {})] = (1, 0.0)

    cached._evict_expired()

    assert cached.cache == {}


def test_bytes_like_values_are_hashed_by_content_not_identity():
    cached = ut.CachedResult(Counter(), maxsize=4)

    first = cached(b"ABCD")

    assert cached(b"ABCD") is first
    assert len(cached.cache) == 1, "equal bytes became one key, not two"


def test_a_failing_background_task_without_a_callback_still_logs(caplog):
    def boom():
        raise ValueError("quiet failure")

    with caplog.at_level(logging.ERROR, logger="biosuite.core.utils"):
        ut.run_in_background(boom).join(timeout=5)

    assert "Background task boom failed: ValueError: quiet failure" in caplog.text, \
        caplog.text
    assert "ValueError: quiet failure" in caplog.text.split("failed:", 1)[1], \
        "the traceback travels with the message, not instead of it"
    records = [r for r in caplog.records if r.name == "biosuite.core.utils"]
    assert [(r.levelname, r.exc_info[0].__name__) for r in records] == \
        [("ERROR", "ValueError")]


def test_a_fasta_read_by_the_simple_reader_tolerates_blank_lines_inside_a_record(tmp_path):
    path = tmp_path / "blank.fa"
    path.write_text(">first\nAC\n\nGT\n>second\nTT\n", encoding="utf-8")

    assert ut.read_fasta_simple(str(path)) == [("first", "ACGT"), ("second", "TT")]
