"""The shapes `plot_virtual_gel` accepts for lanes, labels and colours.

Three separate silent failures lived here.  A flat size mixed into a multi-lane
call died as ``TypeError: 'int' object is not iterable``.  A single label or
colour handed in as a bare string was indexed character by character, so lane 1
was labelled "S".  A colour outside the seven-name table was replaced by the
lane's default, drawing a confident gel in a colour nobody asked for.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from biosuite.core import cloning as cl


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_a_flat_size_inside_a_multi_lane_call_is_named_by_lane_number():
    with pytest.raises(ValueError) as exc:
        cl.plot_virtual_gel([[1000, 200], 3000])
    assert str(exc.value) == (
        "lane 2 of fragment_sizes is 3000, not a list of fragment sizes. "
        "A multi-lane gel takes a list of lists; a single sample takes a flat list.")


def test_a_tuple_of_lanes_is_drawn_as_one_lane_per_entry():
    fig = cl.plot_virtual_gel(((1000, 200), (3000,)), ladder=[1000])
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == [
        "1000", "1000 bp", "200 bp", "Sample 1", "3000 bp", "Sample 2", "Ladder"]


def test_a_bare_label_string_labels_one_lane_instead_of_its_first_letter():
    fig = cl.plot_virtual_gel([[1000]], ladder=[1000], lane_labels="One")
    assert [t.get_text() for t in fig.axes[0].texts] == [
        "1000", "1000 bp", "One", "Ladder"]


def test_a_bare_colour_string_colours_one_lane():
    fig = cl.plot_virtual_gel([[1000]], ladder=[1000], sample_colors="cyan")
    glows = [c.get_facecolor()[0] for c in fig.axes[0].collections
             if c.get_zorder() == 2]
    assert tuple(round(v, 2) for v in glows[1]) == (0.05, 0.8, 0.8, 0.18)


@pytest.mark.parametrize("value", [42, {"green": 1}, 3.5])
def test_a_lane_label_that_is_neither_a_list_nor_a_string_is_refused_by_name(value):
    with pytest.raises(ValueError) as exc:
        cl.plot_virtual_gel([[1000]], lane_labels=value)
    assert str(exc.value) == (
        f"lane_labels must be a list or a single string, got {type(value).__name__}")


def test_the_house_names_keep_their_branded_band_and_glow_pair():
    assert cl._lane_palette("cyan") == ((0.15, 0.85, 0.85), (0.05, 0.8, 0.8, 0.18))
    assert cl._lane_palette("GREEN") == ((0.15, 0.85, 0.25), (0.05, 0.8, 0.15, 0.18))


def test_any_matplotlib_colour_can_colour_a_lane():
    band, glow = cl._lane_palette("#00cccc")
    assert band == (0.0, 0.8, 0.8)
    assert glow == (0.0, 0.8, 0.8, 0.18)
    assert cl._lane_palette((0.2, 0.6, 0.9))[0] == (0.2, 0.6, 0.9)
    assert cl._lane_palette("orange")[0] == pytest.approx((1.0, 0.6470588, 0.0))


def test_an_unusable_colour_raises_instead_of_drawing_the_default_green():
    with pytest.raises(ValueError) as exc:
        cl._lane_palette("nosuchcolor")
    assert str(exc.value) == "Unusable lane colour: 'nosuchcolor'"
    assert isinstance(exc.value.__cause__, ValueError)


def test_a_lane_colour_that_cannot_be_parsed_refuses_the_whole_gel_by_name():
    with pytest.raises(ValueError) as exc:
        cl.plot_virtual_gel([[1000], [500]], sample_colors=["cyan", "nosuchcolor"])
    assert str(exc.value) == "Unusable lane colour: 'nosuchcolor'"
