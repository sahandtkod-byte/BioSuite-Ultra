"""Regression tests: upset exclusivity partition + synteny stats + hist guard."""
import matplotlib
matplotlib.use('Agg')
import pytest
from biosuite.plotting.upset_plots import (
    compute_upset_matrix, compute_set_statistics, plot_upset,
)
from biosuite.plotting.synteny import (
    compute_synteny_score, plot_dotplot, plot_synteny_dotplot,
)


def test_upset_rows_partition_union():
    """Exclusive intersections must exactly partition the union."""
    sets = {'A': {1, 2, 3}, 'B': {2, 3, 4}, 'C': {3, 4, 5}}
    _, matrix, counts = compute_upset_matrix(sets)
    assert sum(counts) == len(set().union(*sets.values()))
    assert matrix  # rows only for non-empty


def test_upset_exclusive_semantics():
    sets = {'A': {1, 2}, 'B': {2, 3}}
    labels, matrix, counts = compute_upset_matrix(sets)
    # rows: AB={2}, A={1}, B={3} -> counts 1,1,1; AB row must exist
    assert counts == [1, 1, 1]
    assert any(m == [1, 1] for m in matrix)


def test_upset_renders_plot():
    fig = plot_upset({'A': {1, 2, 3}, 'B': {2, 3, 4}})
    assert fig is not None


def test_upset_no_data_external_ax():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # all-empty sets -> zero intersection rows -> text path, no crash
    assert plot_upset({'A': set(), 'B': set()}, ax=ax) is None
    plt.close('all')


def test_set_statistics_jaccard():
    st = compute_set_statistics({'A': {1, 2}, 'B': {2, 3}})
    assert st['sizes'] == {'A': 2, 'B': 2}
    assert st['pairwise_jaccard'][('A', 'B')] == pytest.approx(1 / 3)
    assert st['unique_per_set'] == {'A': 1, 'B': 1}


def test_synteny_score_identity():
    order = ['a', 'b', 'c', 'd']
    score, pairs = compute_synteny_score(order, order)
    assert score == pytest.approx(1.0)
    assert len(pairs) == 6  # C(4,2) collinear


def test_synteny_score_reversed():
    order = ['a', 'b', 'c', 'd']
    score, pairs = compute_synteny_score(order, list(reversed(order)))
    assert score == pytest.approx(0.0)


def test_dotplot_histogram_guard_with_external_ax():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # seq with a self match + external ax + show_histograms: previously
    # crashed on ax_histx=None ; now silently skips the marginals
    out = plot_dotplot('ACGTACGT', 'ACGTACGT', word_size=3,
                       ax=ax, show_histograms=True)
    assert out is fig
    plt.close('all')


def test_synteny_dotplot_no_common():
    fig = plot_synteny_dotplot(['a', 'b'], ['x', 'y'])
    assert fig is not None
