"""A feature that cannot sit on the map is refused, and a wrapped one is reported as such.

``add_feature`` took anything: a feature ending past the plasmid's size was drawn
beyond the top of the circle while the report showed a start and an end that do not
exist, a negative coordinate reached the polar transform, and a feature written
across the origin (start after end) was printed with a **negative** size in the
report.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import plasmid_map as pm


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_a_feature_past_the_end_of_the_map_is_refused_by_name():
    plasmid = pm.PlasmidMap(name="p", size=100)

    with pytest.raises(ValueError) as excinfo:
        plasmid.add_feature(pm.PlasmidFeature("late", 90, 120))

    message = str(excinfo.value)
    assert "feature 'late' ends at 120 on a 100 bp plasmid" in message, message
    assert "crosses the origin is written with its start after its end" in message


@pytest.mark.parametrize("start, end", [(-5, 10), (10, -5)])
def test_a_negative_coordinate_is_refused_at_construction(start, end):
    with pytest.raises(ValueError, match="has a negative coordinate"):
        pm.PlasmidFeature("bad", start, end)


def test_a_feature_ending_exactly_at_the_size_is_allowed():
    plasmid = pm.PlasmidMap(name="p", size=100)

    plasmid.add_feature(pm.PlasmidFeature("whole", 0, 100))

    assert plasmid.features[0].span(100) == 100


def test_the_keyword_form_is_checked_too():
    plasmid = pm.PlasmidMap(name="p", size=10)

    with pytest.raises(ValueError, match="ends at 11 on a 10 bp plasmid"):
        plasmid.add_feature("late", start=5, end=11)


def test_a_feature_across_the_origin_is_still_allowed_and_drawn():
    plasmid = pm.PlasmidMap(name="p", size=1000, sequence="A" * 1000)
    plasmid.add_feature(pm.PlasmidFeature("wrap", 900, 100))

    fig = pm.draw_plasmid(plasmid)

    angles = fig.axes[0].collections[0].get_paths()[0].vertices[:, 0]
    assert angles.max() > 2 * 3.141592653589793, "the arc runs past the top"


def test_a_wrapped_feature_is_reported_with_its_wrapped_length():
    plasmid = pm.PlasmidMap(name="p", size=1000, sequence="A" * 1000)
    plasmid.add_feature(pm.PlasmidFeature("wrap", 600, 300))

    rows = [ln for ln in pm.format_plasmid_report(plasmid).splitlines() if "wrap" in ln]

    assert len(rows) == 1
    assert "700" in rows[0], "600 -> 300 across the origin is 700 bp, not -300"
    assert "(wraps)" in rows[0], rows[0]


def test_a_map_without_a_size_cannot_check_bounds_but_still_stores_features():
    """The check needs a scale; a map with size 0 keeps its features for later."""
    plasmid = pm.PlasmidMap(name="p")
    plasmid.add_feature(pm.PlasmidFeature("any", 0, 500))

    assert plasmid.features[0].end == 500
