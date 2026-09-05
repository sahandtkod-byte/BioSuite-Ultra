"""Regression tests for sequence_viewer review fixes."""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

from biosuite.plotting.sequence_viewer import (
    _find_orfs_forward, _find_orfs_reverse, _revcomp,
    draw_gc_content_plot, draw_orf_map, draw_sequence_view,
    draw_translation_view, create_sequence_overview,
)


# ── GC sliding window ───────────────────────────────────────────────────────
def test_gc_first_base_counts():
    # 'G' + 99 'A' -> exactly 1% GC in a 100-bp window; old off-by-one
    # excluded the FIRST base of the window and reported 0%.
    seq = 'G' + 'A' * 99
    fig = draw_gc_content_plot(seq, window=100)
    ax = fig.axes[0]
    line = ax.lines[0]
    assert line.get_ydata()[0] == pytest.approx(1.0)


def test_gc_all_g_is_100():
    fig = draw_gc_content_plot('G' * 100, window=50)
    assert np.allclose(fig.axes[0].lines[0].get_ydata(), 100.0)


def test_gc_none_window_is_full():
    fig = draw_gc_content_plot('GCGCGC' + 'A' * 94, window=100)
    assert fig.axes[0].lines[0].get_ydata()[0] == pytest.approx(6.0)


# ── Reverse-ORF coordinate mapping ──────────────────────────────────────────
def test_reverse_orf_exact_coordinates():
    rc = 'ATGCCGCCGCCCTAA' + 'T' * 3   # M P P P * on rc, 16..19 bp tail
    seq = _revcomp(rc)
    L = len(seq)
    orfs = _find_orfs_reverse(seq, min_aa=1)
    assert len(orfs) == 1
    o = orfs[0]
    # rc span [0, 12) excl. stop (forward convention: end = stop index)
    # -> orig (exclusive-end) [L-12, L)
    assert o['end'] == L - 0
    assert o['start'] == L - 12
    assert o['frame'] == -1
    assert o['length'] == 4


def test_forward_orf_basics():
    orfs = _find_orfs_forward('ATGCCGCCCTAA' + 'ATG', min_aa=1)
    assert orfs and orfs[0]['length'] == 3
    assert orfs[0]['end'] == 9


# ── dashboard / embedding safety ────────────────────────────────────────────
def test_translation_view_stays_in_subplot():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)
    keep = fig.add_subplot(gs[0, 0]); keep.set_title('keeper', fontsize=6)
    fig.add_subplot(gs[0, 1])
    fig.add_subplot(gs[1, 0])
    slot = fig.add_subplot(gs[1, 1])
    draw_translation_view('ATGCCCGCTAAATG', frame=1, ax=slot)
    # 2x2 grid: keeper panels preserved + 2 replacement axes in the slot
    assert len(fig.axes) == 5
    titles = [a.get_title() for a in fig.axes]
    assert any('keeper' in t for t in titles)
    plt.close('all')


def test_sequence_view_does_not_nuke_shared_figure():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 2)
    sentinel = fig.add_subplot(gs[0]); sentinel.plot([0, 1], [0, 1])
    slot = fig.add_subplot(gs[1])
    draw_sequence_view('ATGC' * 40, ax=slot)
    assert len(fig.axes) > 1
    assert sentinel in fig.axes  # other panels must survive
    plt.close('all')


def test_overview_four_panels_render():
    fig = create_sequence_overview('ATGCCCGACTAAATGCGC' * 6 + 'ATGAAATAA')
    assert fig is not None and len(fig.axes) >= 4


def test_orf_map_with_precomputed():
    orfs = _find_orfs_forward('ATGGGGGGGGG' + 'TAA' + 'A' * 30 + 'ATGAAATAA',
                              min_aa=2)
    fig = draw_orf_map('ATGGGGGGGGG' + 'TAA' + 'A' * 30 + 'ATGAAATAA', orfs=orfs)
    assert fig is not None
