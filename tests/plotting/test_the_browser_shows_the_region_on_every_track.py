"""Every row shows the window that was asked for, on the right chromosome.

The region's start and end reached only the coverage, signal and variant drawers;
a BED-only view fell back to matplotlib's auto limits (measured ``xlim 0..1`` with
a patch at x=0..100), and a BED or VCF holding two chromosomes drew both rows of
data on one coordinate axis as if chr2's position 5000 were chr1's.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import genome_browser as gb


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_a_bed_only_view_uses_the_region_for_its_x_axis():
    track = {'type': 'bed', 'name': 'B',
             'data': {'regions': [('chr1', 20, 40, 'a', 0), ('chr1', 50, 70, 'b', 0)]}}

    fig = gb.plot_genome_tracks([track], region=('chr1', 10, 90))

    assert fig.axes[0].get_xlim() == (10, 90)


def test_a_second_chromosome_is_left_out_and_counted():
    track = {'type': 'bed', 'name': 'B',
             'data': {'regions': [('chr1', 20, 40, 'a', 0), ('chr2', 20, 40, 'b', 0)]}}

    fig = gb.plot_genome_tracks([track], region=('chr1', 0, 100))

    assert len(fig.axes[0].patches) == 1
    assert fig.axes[0].get_ylabel() == "B — 1 region(s) on other chromosomes not shown"


def test_the_variant_row_is_filtered_the_same_way():
    track = {'type': 'variant', 'name': 'V',
             'data': {'variants': [{'chrom': 'chr1', 'pos': 10, 'ref': 'A', 'alt': 'G'},
                                   {'chrom': 'chr2', 'pos': 10, 'ref': 'A', 'alt': 'G'}]}}

    fig = gb.plot_genome_tracks([track], region=('chr1', 0, 50))

    assert len(fig.axes[0].lines) == 1
    assert fig.axes[0].get_ylabel() == \
        "V — 1 variant(s) on other chromosomes not shown"


def test_a_window_that_holds_every_variant_says_nothing():
    """The false branch of the filter's note: nothing was left out."""
    track = {'type': 'variant', 'name': 'V',
             'data': {'variants': [{'chrom': 'chr1', 'pos': 10, 'ref': 'A', 'alt': 'G'}]}}

    fig = gb.plot_genome_tracks([track], region=('chr1', 0, 50))

    assert fig.axes[0].get_ylabel() == "V"


def test_an_open_region_leaves_the_variant_axis_to_the_data():
    track = {'type': 'variant', 'name': 'V',
             'data': {'variants': [{'chrom': 'chr1', 'pos': 10, 'ref': 'A', 'alt': 'G'},
                                   {'chrom': 'chr1', 'pos': 30, 'ref': 'A', 'alt': 'G'}]}}

    fig = gb.plot_genome_tracks([track], region=('chr1', 0, None))

    low, high = fig.axes[0].get_xlim()
    assert low <= 10 and high >= 30, "matplotlib frames the markers themselves"


def test_a_hand_built_variant_without_a_chromosome_is_kept():
    """It cannot be filtered by a chromosome, so it must not be dropped by one."""
    track = {'type': 'variant', 'name': 'V',
             'data': {'variants': [{'pos': 10, 'ref': 'A', 'alt': 'G'}]}}

    fig = gb.plot_genome_tracks([track], region=('chr1', 0, 50))

    assert len(fig.axes[0].lines) == 1
    assert fig.axes[0].get_ylabel() == "V"


def test_a_clean_window_carries_no_note():
    """The positive control: nothing is said when nothing was left out."""
    track = {'type': 'bed', 'name': 'B',
             'data': {'regions': [('chr1', 20, 40, 'a', 0)]}}

    fig = gb.plot_genome_tracks([track], region=('chr1', 0, 100))

    assert fig.axes[0].get_ylabel() == "B"


def test_the_rows_share_the_window_between_them():
    coverage = {'type': 'coverage', 'name': 'C',
                'data': {'positions': np.array([0, 500, 1000]),
                         'coverage': np.array([1, 2, 3])}}
    bed = {'type': 'bed', 'name': 'B',
           'data': {'regions': [('chr1', 100, 200, 'a', 0)]}}

    fig = gb.plot_genome_tracks([coverage, bed], region=('chr1', 100, 900))

    assert [ax.get_xlim() for ax in fig.axes] == [(100, 900), (100, 900)], \
        "sharex plus one explicit window on every row"


def test_an_open_region_does_not_set_an_axis():
    """A region with no end cannot be an x range; the drawers fall back to the data."""
    track = {'type': 'coverage', 'name': 'C',
             'data': {'positions': np.array([0, 10]), 'coverage': np.array([1, 2])}}

    fig = gb.plot_genome_tracks([track], region=('chr1', 100, None))

    assert fig.axes[0].get_xlim() == (0, 10), "the data's own span"
