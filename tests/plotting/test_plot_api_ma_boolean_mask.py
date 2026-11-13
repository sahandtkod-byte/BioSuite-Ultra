"""The significance mask of an MA plot, in whatever shape it arrives.

``ma(sig=...)`` documented "boolean array" and used ``~sig``, so a Python list —
the shape a caller reads out of a results table — died with
``TypeError: bad operand type for unary ~: 'list'``.  A mask of the wrong length
then produced numpy's own complaint about axis sizes, and the zero line that is
supposed to separate up- from down-regulation was drawn in black on a background
the theme paints near-black (#1a1a1a), so it was not drawn at all in practice.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import plot_api as pa

MEAN = [5.0, 6.0, 7.0]
LOG_FC = [1.0, -1.0, 0.2]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _sizes(ax):
    return sorted(len(np.asarray(c.get_offsets())) for c in ax.collections)


def test_a_python_list_of_booleans_selects_the_points_it_names():
    fig = pa.ma(MEAN, LOG_FC, sig=[True, False, True])

    ax = fig.axes[0]
    assert _sizes(ax) == [1, 2], "one not significant, two significant"
    picked = np.vstack([np.asarray(c.get_offsets()) for c in ax.collections])
    assert sorted(picked[:, 1].tolist()) == [-1.0, 0.2, 1.0]


def test_zeros_and_ones_are_accepted_as_the_same_mask():
    from_list = pa.ma(MEAN, LOG_FC, sig=[1, 0, 1])
    from_bools = pa.ma(MEAN, LOG_FC, sig=[True, False, True])

    assert _sizes(from_list.axes[0]) == _sizes(from_bools.axes[0]) == [1, 2]


def test_the_interactive_branch_accepts_a_list_mask_too():
    pytest.importorskip('plotly.graph_objects')

    fig = pa.ma(MEAN, LOG_FC, sig=[True, False, True], interactive=True)

    assert [trace.name for trace in fig.data] == ['Not significant', 'Significant']
    assert [len(trace.x) for trace in fig.data] == [1, 2]


def test_a_mask_that_is_not_boolean_is_named_with_an_example():
    with pytest.raises(ValueError, match=r'booleans or 0/1') as info:
        pa.ma(MEAN, LOG_FC, sig=['yes', 'no', 'yes'])

    assert "'yes'" in str(info.value)

    with pytest.raises(ValueError, match=r'1 other value'):
        pa.ma(MEAN, LOG_FC, sig=[0.5, 1, 0])


def test_a_mask_of_the_wrong_length_is_refused_before_numpy_is():
    with pytest.raises(ValueError, match=r'3 genes and a sig mask of 2 entries'):
        pa.ma(MEAN, LOG_FC, sig=[True, False])

    with pytest.raises(ValueError, match=r'1-D boolean mask, got a 2-D'):
        pa.ma(MEAN, LOG_FC, sig=[[True, False, True]])


def test_hover_names_have_to_line_up_with_the_genes_they_belong_to():
    with pytest.raises(ValueError, match=r'3 genes and 2 names'):
        pa.ma(MEAN, LOG_FC, gene_names=['TP53', 'BRCA1'])


def test_the_expression_arrays_have_to_match_each_other():
    with pytest.raises(ValueError, match=r'mean_expression=3, log_fc=2'):
        pa.ma(MEAN, [1.0, 2.0])


def test_a_negative_fold_change_cutoff_is_refused_when_no_mask_is_given():
    with pytest.raises(ValueError, match=r'ma\(\) needs fc_thresh >= 0'):
        pa.ma(MEAN, LOG_FC, fc_thresh=-1.0)


def test_no_mask_means_the_fold_change_cutoff_decides():
    fig = pa.ma(MEAN, LOG_FC, fc_thresh=0.5)

    assert _sizes(fig.axes[0]) == [1, 2], "|0.2| is the only gene under 0.5"


def test_the_zero_line_is_visible_on_the_dark_axes():
    """Positive control first: the theme really is dark, or 'not black' would
    prove nothing."""
    fig = pa.ma(MEAN, LOG_FC, sig=[True, False, True])
    ax = fig.axes[0]
    background = np.asarray(ax.get_facecolor())[:3]
    line = np.asarray(matplotlib.colors.to_rgb(ax.lines[0].get_color()))

    assert background.max() < 0.2, f"axes background {background} was expected to be dark"
    assert line.min() > background.max() + 0.5, \
        f"a {line} zero line on a {background} background cannot be seen"
