"""A matplotlib colormap name has to work in the Plotly branch too.

``heatmap(cmap=)`` is documented as a matplotlib colormap name and the static
branch accepts every one matplotlib knows.  The interactive branch passed the
string straight to Plotly's ``colorscale``, whose own list only overlaps by
accident: 'viridis', 'magma' and 'Inferno' came through, while **every**
matplotlib-only name died — ``heatmap(cmap='coolwarm', interactive=True)``
raised "Invalid value of type 'builtins.str' received for the 'colorscale'
property of heatmap", and 'seismic' the same.  Two backends, one argument, only
one of which could be used.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import plot_api as pa

GRID = np.array([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _as_pairs(scale):
    """Plotly hands a colourscale back as tuples; compare on the values."""
    return [[float(stop), colour] for stop, colour in scale]


def _expected_scale(name, stops=10):
    """Sampled the way matplotlib resolves it: as spelled, then lowercase."""
    try:
        colormap = matplotlib.colormaps.get_cmap(name)
    except ValueError:
        colormap = matplotlib.colormaps.get_cmap(name.lower())
    positions = np.linspace(0.0, 1.0, stops)
    out = []
    for position in positions:
        red, green, blue = (int(round(255 * c)) for c in colormap(position)[:3])
        out.append([float(position), f'#{red:02x}{green:02x}{blue:02x}'])
    return out


@pytest.mark.parametrize('name', ['viridis', 'magma', 'RdBu', 'coolwarm', 'seismic',
                                  'Greys', 'jet', 'twilight', 'cividis', 'plasma'])
def test_every_matplotlib_colormap_name_becomes_a_ten_stop_scale(name):
    """The names that worked before (viridis, magma, RdBu, Greys, jet, cividis,
    plasma) come out unchanged in meaning; the ones that raised (coolwarm,
    seismic, twilight) now work as well."""
    scale = pa._plotly_colorscale(name)

    assert len(scale) == 10
    assert [stop for stop, _colour in scale] == pytest.approx(np.linspace(0, 1, 10))
    assert _as_pairs(scale) == _expected_scale(name)


@pytest.mark.parametrize('name', ['coolwarm', 'seismic', 'twilight_shifted', 'Paired',
                                  'hsv', 'Binary', 'terrain'])
def test_the_interactive_heatmap_accepts_the_names_that_used_to_raise(name):
    pytest.importorskip('plotly.graph_objects')

    fig = pa.heatmap(GRID, cmap=name, interactive=True)

    assert _as_pairs(fig.data[0].colorscale) == _expected_scale(name)


def test_the_first_stop_is_the_colormap_own_starting_colour():
    """Cross-check against the colour the static branch would have painted, so a
    broken sampling cannot pass by being self-consistent."""
    scale = pa._plotly_colorscale('inferno')
    red, green, blue = (int(round(255 * c)) for c in matplotlib.colormaps['inferno'](0.0)[:3])

    assert (red, green, blue) == (0, 0, 4), scale[0]
    assert scale[0][0] == 0.0
    assert scale[0][1] == '#000004', "inferno starts at (nearly) black"


@pytest.mark.parametrize('name', ['Earth', 'aggrnyl', 'portland', 'plotly3'])
def test_a_name_only_plotly_knows_is_passed_through_untouched(name):
    """A matplotlib-only lookup must not swallow the names Plotly keeps for
    itself; the string is handed on and Plotly resolves it."""
    assert pa._plotly_colorscale(name) == name


@pytest.mark.parametrize('spelling', ['Binary', 'binary', 'CoolWarm', 'RdBu_r'])
def test_a_matplotlib_name_works_whichever_case_the_caller_typed(spelling):
    """matplotlib's registry is lowercase and Plotly keeps its own list; 'Binary'
    was in neither for the interactive branch, so it raised from Plotly."""
    scale = pa._plotly_colorscale(spelling)

    assert _as_pairs(scale) == _expected_scale(spelling)

    pytest.importorskip('plotly.graph_objects')
    fig = pa.heatmap(GRID, cmap=spelling, interactive=True)
    assert _as_pairs(fig.data[0].colorscale) == _expected_scale(spelling)


def test_a_scale_the_caller_built_by_hand_is_left_alone():
    given = [[0.0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']]

    assert pa._plotly_colorscale(given) is given

    pytest.importorskip('plotly.graph_objects')
    fig = pa.heatmap(GRID, cmap=given, interactive=True)
    assert _as_pairs(fig.data[0].colorscale) == [[0.0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']]


def test_the_default_colormap_still_matches_the_static_figure():
    pytest.importorskip('plotly.graph_objects')

    fig = pa.heatmap(GRID, interactive=True)
    static = pa.heatmap(GRID)

    assert _as_pairs(fig.data[0].colorscale) == _expected_scale('viridis')
    assert static.axes[0].images[0].cmap.name == 'viridis'


def test_a_colormap_object_is_allowed_because_it_has_a_name():
    pytest.importorskip('plotly.graph_objects')

    fig = pa.heatmap(GRID, cmap=matplotlib.colormaps['coolwarm'], interactive=True)

    assert _as_pairs(fig.data[0].colorscale) == _expected_scale('coolwarm')
