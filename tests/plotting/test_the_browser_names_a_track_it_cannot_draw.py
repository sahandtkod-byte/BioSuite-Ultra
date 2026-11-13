"""A row the browser cannot draw says so, and a track's data has to be the right shape.

An unknown track type produced a blank row with no word — a typo like ``'gene'``
read as "this region has no features".  A list handed where a dict of arrays
belongs raised ``AttributeError: 'list' object has no attribute 'get'`` from three
frames down, and two arrays of different lengths died inside matplotlib's
``fill_between`` accounting.
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


def test_a_track_type_the_browser_does_not_have_says_so_in_its_row():
    fig = gb.plot_genome_tracks([{'type': 'gene', 'name': 'genes'}])

    assert [t.get_text() for t in fig.axes[0].texts] == ["unknown track type 'gene'"]
    assert fig.axes[0].get_ylabel() == "genes"


def test_data_that_is_not_a_dict_of_arrays_is_refused_by_track_name():
    with pytest.raises(ValueError, match=r"track 'b' carries a list as 'data'"):
        gb.plot_genome_tracks([{'type': 'bed', 'name': 'b', 'data': ['chr1', 1, 5]}])


@pytest.mark.parametrize("track_type, first, second", [
    ("coverage", "positions", "coverage"),
    ("signal", "positions", "values"),
])
def test_two_arrays_of_different_lengths_name_both_counts(track_type, first, second):
    track = {'type': track_type, 'name': 't',
             'data': {first: [0, 10, 20], second: [1, 2]}}

    with pytest.raises(ValueError) as excinfo:
        gb.plot_genome_tracks([track])

    message = str(excinfo.value)
    assert f"the {track_type} track got 3 {first} and 2 {second}" in message, message


def test_equal_arrays_still_draw():
    """The positive control: the check is about the mismatch, not about size."""
    track = {'type': 'signal', 'name': 's',
             'data': {'positions': np.array([0, 10]), 'values': np.array([2, 4])}}

    fig = gb.plot_genome_tracks([track])

    assert len(fig.axes[0].collections) == 1
