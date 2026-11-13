"""Targeted unit tests for uncovered core/utils.py helpers."""
import logging
import re
import threading
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from biosuite.core import utils


# ── threading helpers ────────────────────────────────────────────────────────

def test_run_in_background_with_callback():
    done = threading.Event()
    got = {}

    def slow(x):
        time.sleep(0.05)
        return x * 2

    def cb(result):
        got['r'] = result
        done.set()

    th = utils.run_in_background(slow, 21, callback=cb)
    assert th.is_alive()
    assert done.wait(timeout=5)
    th.join(timeout=5)
    assert got['r'] == 42


def test_run_with_progress_callback_fires():
    # run_with_progress injects the callback as a kwarg into the callee
    calls = []

    def work(x, progress_callback=None):
        progress_callback(1, 1)
        return x + 1

    out = utils.run_with_progress(work, lambda cur, total: calls.append((cur, total)), 7)
    assert out == 8
    assert calls == [(1, 1)]


# ── CachedResult ─────────────────────────────────────────────────────────────

def test_cached_result_hits_and_ttl():
    calls = []

    def f(x):
        calls.append(x)
        return x * 10

    cached = utils.CachedResult(f, maxsize=8, ttl=None)
    assert cached(3) == 30 and cached(3) == 30
    assert calls == [3]                       # second hit came from cache
    cached.clear()
    assert len(cached) == 0
    assert cached(3) == 30
    assert calls == [3, 3]

    expiring = utils.CachedResult(f, ttl=60)
    assert expiring(5) == 50
    key = next(iter(expiring.cache))
    value, created = expiring.cache[key]
    expiring.cache[key] = (value, created - 61)      # age it by hand, no sleeping
    assert expiring.is_expired(key) is True
    assert expiring(5) == 50
    assert calls == [3, 3, 5, 5], "the expired entry was recomputed, not returned"
    assert expiring.is_expired(expiring.make_key((999,), {})) is True, \
        "a key that was never stored counts as expired"


@pytest.mark.parametrize("kwargs, message", [
    pytest.param({'ttl': 0.0},
                 "CachedResult ttl must be None or a positive finite number of seconds, "
                 "got 0.0."),
    pytest.param({'ttl': -5},
                 "CachedResult ttl must be None or a positive finite number of seconds, "
                 "got -5."),
    pytest.param({'ttl': float('nan')},
                 "CachedResult ttl must be None or a positive finite number of seconds, "
                 "got nan."),
    pytest.param({'maxsize': 0},
                 "CachedResult maxsize must be an integer of at least 1, got 0."),
    pytest.param({'maxsize': None},
                 "CachedResult maxsize must be an integer of at least 1, got None."),
    pytest.param({'maxsize': True},
                 "CachedResult maxsize must be an integer of at least 1, got True."),
])
def test_a_cache_that_could_never_hold_anything_is_refused(kwargs, message):
    """`ttl=0` expired every entry the moment it was written and `maxsize=0` crashed
    the eviction loop with `StopIteration` — both are configuration mistakes that
    have to be said where they are made."""
    with pytest.raises(ValueError) as excinfo:
        utils.CachedResult(lambda x: x, **kwargs)
    assert str(excinfo.value) == message


def test_cached_result_lru_eviction():
    def g(x):
        return x

    small = utils.CachedResult(g, maxsize=2)
    small(1), small(2), small(3)              # eviction of oldest key
    assert len(small) <= 2


# ── config / sessions (temp home) ────────────────────────────────────────────

@pytest.fixture()
def tmp_app_home(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, 'get_app_dir', lambda: {'config': str(tmp_path / 'cfg'),
                                                       'session': str(tmp_path / 'sess')})
    return tmp_path


def test_config_roundtrip(tmp_app_home, monkeypatch):
    monkeypatch.setenv('BIOSUITE_CONFIG_DIR', str(tmp_app_home))
    cfg = dict(utils.config)
    cfg['marker'] = 'xyz'
    utils.config.update(cfg)
    try:
        utils.save_config(utils.config)
        loaded = utils.load_config()
        assert loaded.get('marker') == 'xyz'
    finally:
        pass


def test_api_key_roundtrip(tmp_app_home):
    utils.set_api_key('unittest_svc_circle', 'KEY123')
    assert utils.get_api_key('unittest_svc_circle') == 'KEY123'
    utils.set_api_key('unittest_svc_circle', '')


# ── safe inputs (quiet mode -> defaults, no stdin) ──────────────────────────

def test_safe_inputs_quiet_returns_default(monkeypatch):
    monkeypatch.setitem(utils.config, 'quiet', True)
    assert utils.safe_float_input('p?', 1.5) == 1.5
    assert utils.safe_int_input('n?', 9) == 9


def test_safe_float_parses_user(monkeypatch):
    monkeypatch.setitem(utils.config, 'quiet', False)
    monkeypatch.setattr('builtins.input', lambda _: '3.25')
    assert utils.safe_float_input('x?', 0.0) == 3.25
    monkeypatch.setattr('builtins.input', lambda _: 'oops')
    assert utils.safe_float_input('x?', 0.5) == 0.5     # invalid -> default


# ── data loading ─────────────────────────────────────────────────────────────

def test_load_dataframe_safe(tmp_path):
    csv = tmp_path / 'd.csv'
    csv.write_text('a,b\n1,2\n3,4\n')
    df = utils.load_dataframe_safe(str(csv))
    assert list(df.columns) == ['a', 'b'] and len(df) == 2
    assert utils.load_dataframe_safe(str(tmp_path / 'missing.csv')) is None
    empty = tmp_path / 'e.csv'
    empty.write_text('')
    assert utils.load_dataframe_safe(str(empty)) is None


# ── downsampling ─────────────────────────────────────────────────────────────

def test_downsample_keeps_size_and_bounds():
    x = np.arange(10000)
    y = np.arange(10000) * 2
    xs, ys = utils.maybe_downsample(x, y, max_points=100)
    assert len(xs) == 100 and len(ys) == 100
    smallx, smally = utils.maybe_downsample(x[:10], y[:10])
    assert len(smallx) == 10


# ── pandas-backed report stats ───────────────────────────────────────────────

def test_report_boxplot_and_volcano_stats():
    df = pd.DataFrame({'grp': ['A'] * 5 + ['B'] * 5,
                       'val': [1, 2, 3, 4, 100, 5, 6, 7, 8, 9]})
    st = utils.report_boxplot_stats(df, 'grp', 'val')
    assert set(st) == {'A', 'B'}
    assert st['A']['median'] == pytest.approx(3.0)

    counts = utils.report_volcano_stats(np.array([2.0, -2.0, 0.1]),
                                        np.array([0.001, 0.001, 0.9]),
                                        fc_thresh=1.0, p_thresh=0.05)
    assert counts == {'up': 1, 'down': 1, 'total': 2}


def test_volcano_counts_need_both_series_and_agree_on_length():
    """Measured on the old code: two plain Python lists raised
    `TypeError: '<' not supported between instances of 'list' and 'float'`, and a
    2-gene fold-change series beside a 1-value p series broadcast into the answer
    `{'up': 0, 'down': 0}` instead of an error."""
    assert utils.report_volcano_stats([2.0, -2.0, 0.1], [0.001, 0.001, 0.9],
                                      1.0, 0.05) == {'up': 1, 'down': 1, 'total': 2}
    with pytest.raises(ValueError, match=re.escape(
            "report_volcano_stats got 2 log2 fold changes but 1 p-values; "
            "every gene needs both.")):
        utils.report_volcano_stats(np.array([1.0, 2.0]), np.array([0.1]), 1.0, 0.05)
    with pytest.raises(ValueError, match=re.escape(
            "needs 1-D fold-change and p-value series, got shapes (2, 2) and (2,).")):
        utils.report_volcano_stats(np.array([[1.0, 2.0], [3.0, 4.0]]),
                                   np.array([0.1, 0.2]), 1.0, 0.05)


def test_the_manhattan_report_counts_hits_and_names_the_top_snk(caplog):
    """'p' holds -log10(p) in this package's frames, and a hit is anything above
    -log10(threshold)."""
    df = pd.DataFrame({'p': [6.0, 8.5, 20.0, 1.0],
                       'chrom': ['1', '1', '2', '2'], 'pos': [1, 2, 3, 4]})
    with caplog.at_level(logging.INFO):
        assert utils.report_manhattan_stats(df) == {'significant': 2, 'total': 4}
    assert "Significant SNPs: 2" in caplog.text
    assert "Top SNP: 2:3 with -log10(p)=20.00" in caplog.text

    df_low = df.assign(p=[0.1, 0.2, 0.3, 0.4])
    with caplog.at_level(logging.INFO):
        assert utils.report_manhattan_stats(df_low) == {'significant': 0, 'total': 4}
    assert "No SNPs reached genome-wide significance." in caplog.text


@pytest.mark.parametrize("frame, needed", [
    pytest.param(pd.DataFrame({'x': [1]}), "['chrom', 'p', 'pos']", id="nothing-usable"),
    pytest.param(pd.DataFrame({'p': [9.0], 'chrom': ['1']}), "['pos']", id="no-position"),
])
def test_a_manhattan_frame_that_cannot_answer_is_said_not_raised(caplog, frame, needed):
    """Measured on the old code: a frame without 'p' returned None from a function
    annotated `-> Dict[str, int]`, and one with 'p' but no 'chrom' raised
    `KeyError: 'chrom'` while printing the report line."""
    with caplog.at_level(logging.INFO):
        assert utils.report_manhattan_stats(frame) == {}
    assert f"Manhattan statistics needs the columns {needed}" in caplog.text


def test_report_boxplot_stats_omits_groups_without_values():
    df = pd.DataFrame({'grp': ['A', 'A', 'B', 'B'],
                       'val': [1.0, 2.0, np.nan, np.nan]})
    st = utils.report_boxplot_stats(df, 'grp', 'val')
    assert set(st) == {'A'}, "a group whose values are all NaN has no quartiles"
    assert st['A'] == {'median': 1.5, 'iqr': 0.5, 'outliers': 0}


def test_report_pca_stats_reports_the_first_two_components(caplog):
    pca = PCA(n_components=2).fit(np.array(
        [[0.0, 0.0, 5.0], [0.1, 0.0, 5.2], [5.0, 4.9, 0.0], [5.2, 5.0, 0.2]]))

    with caplog.at_level("INFO", logger="biosuite.core.utils"):
        stats = utils.report_pca_stats(pca)

    ratios = pca.explained_variance_ratio_
    assert stats == {'var1': ratios[0], 'var2': ratios[1]}
    assert "PCA Statistics:" in caplog.text
    assert f"PC1 explains {ratios[0] * 100:.2f}% of variance" in caplog.text
    assert f"PC2 explains {ratios[1] * 100:.2f}% of variance" in caplog.text


def test_report_scatter_stats(tmp_path):
    x = [1, 2, 3, 4, 5]
    y = [2.1, 4.0, 5.9, 8.1, 9.9]
    r = utils.report_scatter_stats(x, y)
    assert 0.99 <= r['r'] <= 1.0


# ── sequence helpers ─────────────────────────────────────────────────────────

def test_read_fasta_simple(tmp_path):
    fa = tmp_path / 'x.fa'
    fa.write_text('>id1\nACGT\n>id2 desc\nTGCA\n')
    recs = utils.read_fasta_simple(str(fa))
    assert ('id1', 'ACGT') in recs
    assert len(recs) == 2 and recs[1][1] == 'TGCA'


def test_reverse_complement_dna():
    assert utils.reverse_complement_dna('ACGTN') == 'NACGT'


def test_has_tool_missing():
    assert utils.has_tool('surely-not-a-real-tool-xyz') is False
