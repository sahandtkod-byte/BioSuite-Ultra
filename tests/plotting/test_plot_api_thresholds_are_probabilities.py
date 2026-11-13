"""A p-value cutoff is a probability, and here it is checked like one.

``volcano(p_thresh=0)`` set its significance line to ``-log10(0)``, which is
``+inf``: matplotlib draws nothing and the figure read as "no threshold was
applied" while the numpy divide-by-zero warning went to whichever stream the
caller was not watching.  ``p_thresh=-1`` gave ``nan``, and in
``manhattan(threshold=-1)`` the same arithmetic hid the genome-wide line.  A
p-value itself outside [0, 1] was clipped into the most extreme point on the
plot instead of refused.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import plot_api as pa

LOG2FC = np.array([2.5, -3.0, 0.1, 1.2, -0.4])
PVALUES = np.array([0.001, 0.002, 0.60, 0.01, 0.40])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _hline(fig):
    return float(fig.axes[0].lines[0].get_ydata()[0])


def _vlines(fig):
    """The x positions of the two vertical guides matplotlib kept."""
    return [float(line.get_xdata()[0]) for line in fig.axes[0].lines[1:3]]


# ─── volcano: the cutoffs ────────────────────────────────────────────────────

def test_the_default_cutoffs_still_draw_the_line_and_both_guides():
    """Positive control: the legal call is unchanged, so the refusals below are
    not a guard that fires for everything."""
    fig = pa.volcano(LOG2FC, PVALUES, fc_thresh=1.0, p_thresh=0.05)

    assert _hline(fig) == pytest.approx(1.3010299956639813)
    assert _vlines(fig) == [-1.0, 1.0]


@pytest.mark.parametrize('bad', [0.0, -1.0, 1.5, float('nan'), float('inf'), '0.05'])
def test_a_volcano_cutoff_that_is_not_a_probability_is_refused_by_name(bad):
    with pytest.raises(ValueError) as info:
        pa.volcano(LOG2FC, PVALUES, p_thresh=bad)

    assert 'p_thresh' in str(info.value)


def test_a_zero_p_cutoff_says_what_it_would_have_drawn():
    with pytest.raises(ValueError, match=r'p_thresh.*above 0.*infinity'):
        pa.volcano(LOG2FC, PVALUES, p_thresh=0.0)


def test_a_negative_p_cutoff_names_the_arithmetic_it_breaks():
    with pytest.raises(ValueError, match='not a real number'):
        pa.volcano(LOG2FC, PVALUES, p_thresh=-1.0)


@pytest.mark.parametrize('bad', [-1.0, -0.001, float('nan'), 'fc'])
def test_a_negative_fold_change_cutoff_is_refused(bad):
    """``abs(log2fc) >= -1`` is true for every gene, so nothing was filtered and
    the two guide lines moved to the wrong side of the axis."""
    with pytest.raises(ValueError, match='fc_thresh') as info:
        pa.volcano(LOG2FC, PVALUES, fc_thresh=bad)

    assert 'every gene' in str(info.value) or 'finite number' in str(info.value)


def test_a_fold_change_cutoff_of_zero_means_no_fold_change_filter():
    fig = pa.volcano(LOG2FC, PVALUES, fc_thresh=0.0, p_thresh=0.05)

    assert _vlines(fig) == [0.0, 0.0]
    sizes = sorted(len(np.asarray(c.get_offsets())) for c in fig.axes[0].collections)
    assert sizes == [2, 3], "every gene clears a cutoff of 0, so only p < 0.05 splits them"


# ─── volcano: the data ───────────────────────────────────────────────────────

def test_pvalues_outside_the_unit_interval_are_counted_and_named():
    with pytest.raises(ValueError, match=r'1 of 5 pvalues outside \[0, 1\]') as info:
        pa.volcano(LOG2FC, np.array([0.001, 3.0, 0.60, 0.01, 0.40]))

    assert '3.0' in str(info.value)

    with pytest.raises(ValueError, match=r'2 of 5 pvalues outside'):
        pa.volcano(LOG2FC, np.array([-0.5, 0.002, 0.60, 0.01, 1.2]))


def test_a_p_value_of_exactly_zero_or_one_is_still_legal():
    fig = pa.volcano([2.0, 1.0], [0.0, 1.0])

    heights = np.concatenate([np.asarray(c.get_offsets())[:, 1]
                              for c in fig.axes[0].collections])
    assert sorted(np.round(heights, 3).tolist()) == [-0.0, 300.0]


def test_a_p_value_that_is_not_a_number_cannot_be_plotted_as_one():
    """NaN reaches us from a failed upstream statistic; -log10(NaN) is NaN and
    matplotlib drops the point, so the volcano silently lost a gene."""
    with pytest.raises(ValueError, match=r'1 of 5 pvalues outside \[0, 1\]') as info:
        pa.volcano(LOG2FC, np.array([0.001, np.nan, 0.60, 0.01, 0.40]))

    assert 'nan' in str(info.value)

    with pytest.raises(ValueError, match=r'1 of 3 pvalues outside'):
        pa.qqplot([0.1, np.inf, 0.3])


def test_arrays_that_describe_different_numbers_of_genes_are_refused_together():
    with pytest.raises(ValueError, match=r'log2fc=3, pvalues=2') as info:
        pa.volcano([1.0, 2.0, 3.0], [0.01, 0.5])

    assert 'same number of entries' in str(info.value)


def test_hover_names_have_to_line_up_with_the_genes_they_belong_to():
    with pytest.raises(ValueError, match=r'5 genes and 2 names'):
        pa.volcano(LOG2FC, PVALUES, gene_names=['TP53', 'BRCA1'])


def test_an_empty_result_set_is_refused_rather_than_drawn():
    with pytest.raises(ValueError, match=r'volcano\(\) needs at least one value in log2fc'):
        pa.volcano([], [])


# ─── manhattan: the same arithmetic, a different function ────────────────────

CHROMS = ['chr1', 'chr1', 'chr2']
POSITIONS = [100, 900, 200]
GWAS_P = [0.001, 0.4, 0.9]


def test_the_genome_wide_line_is_drawn_where_a_real_threshold_puts_it():
    fig = pa.manhattan(CHROMS, POSITIONS, GWAS_P, threshold=5e-8)

    assert _hline(fig) == pytest.approx(7.301029995663981)


@pytest.mark.parametrize('bad', [0.0, -1.0, 2.0, float('nan')])
def test_a_manhattan_threshold_that_is_not_a_probability_is_refused(bad):
    with pytest.raises(ValueError, match='threshold') as info:
        pa.manhattan(CHROMS, POSITIONS, GWAS_P, threshold=bad)

    assert 'probability' in str(info.value) or 'finite' in str(info.value)


def test_fractional_positions_are_refused_instead_of_truncated():
    with pytest.raises(ValueError, match=r'are not whole numbers \(for example 1\.9\)') as info:
        pa.manhattan(['1', '2'], [1.9, 2.7], [0.5, 0.5])

    assert 'moves the point to the left' in str(info.value)


def test_a_negative_position_is_refused():
    with pytest.raises(ValueError, match=r'negative position\(s\)'):
        pa.manhattan(['1', '2'], [100, -5], [0.5, 0.5])


def test_the_three_manhattan_arrays_have_to_agree_about_the_snp_count():
    with pytest.raises(ValueError, match=r'chromosomes=3, positions=3, pvalues=2'):
        pa.manhattan(CHROMS, POSITIONS, [0.5, 0.5])
