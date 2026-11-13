"""A grouped plot needs groups that have something in them.

``boxplot({})`` reached ``ax.boxplot([], tick_labels=[])`` and answered with
matplotlib's own "Dimensions of labels and X must be compatible", while the
Plotly branch returned a perfectly valid figure with **no traces** — the same
empty chart either way, described differently or not at all.  A one-value group
was drawn as a violin, which is a density estimate: one point has no shape, and
the figure claimed a distribution the sample cannot support.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import plot_api as pa

GROUPS = {'ctrl': [1.0, 2.0, 3.0, 2.5], 'trt': [4.0, 5.0, 4.5, 6.0]}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_two_groups_still_draw_two_named_boxes():
    fig = pa.boxplot(GROUPS)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ['ctrl', 'trt']
    assert ax.get_lines(), "the whiskers and medians are lines, and they are drawn"


@pytest.mark.parametrize('plot', ['boxplot', 'violin'])
def test_no_groups_at_all_is_refused_rather_than_drawn(plot):
    with pytest.raises(ValueError, match=f'{plot}\\(\\) was given no groups'):
        getattr(pa, plot)({})


@pytest.mark.parametrize('plot', ['boxplot', 'violin'])
def test_the_refusal_reaches_the_interactive_branch_too(plot):
    """Without the guard Plotly answers with an empty figure and no complaint."""
    pytest.importorskip('plotly.graph_objects')

    with pytest.raises(ValueError, match='was given no groups'):
        getattr(pa, plot)({}, interactive=True)


def test_a_group_without_values_is_named_by_group():
    with pytest.raises(ValueError, match=r'at least 1 value\(s\) per group') as info:
        pa.boxplot({'ctrl': [1.0, 2.0], 'blank': []})

    assert 'blank=0' in str(info.value)
    assert 'ctrl' not in str(info.value).split('have fewer')[1], "only the thin group is listed"


def test_a_violin_needs_two_values_before_it_can_estimate_a_shape():
    with pytest.raises(ValueError, match=r'at least 2 value\(s\) per group') as info:
        pa.violin({'ctrl': [1.0, 2.0, 3.0], 'lonely': [4.0]})

    assert 'lonely=1' in str(info.value)


def test_two_values_is_enough_for_a_violin_and_is_not_refused():
    fig = pa.violin({'a': [1.0, 2.0], 'b': [3.0, 5.0]})

    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ['a', 'b']


def test_a_thing_that_is_not_a_mapping_is_refused_as_a_type():
    with pytest.raises(TypeError, match=r'needs a dict of group name -> values, got list'):
        pa.boxplot([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(TypeError, match='needs a dict of group name'):
        pa.violin('ctrl')
