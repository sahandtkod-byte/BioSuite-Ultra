"""A Circos plot needs a whole circle to spend.

The demo added sectors into a dict keyed by name, so typing the same name twice
for "Number of sectors: 2" silently produced one sector — the circle was drawn
with a slice the user never asked for and nothing was said about the lost one.
Sizes are free-form floats, and every size zero made the arc arithmetic divide by
zero, which the demo's own handler turned into one ``Error: float division by
zero`` line with the already-created polar figure left open for the life of the
process. A link whose sector name was mistyped was dropped by ``continue``.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import biological_plots as bp


@pytest.mark.parametrize("sizes", [["0", "0"], ["5", "-5"]])
def test_sector_sizes_that_sum_to_nothing_are_refused_before_any_figure(
        run_demo, sizes):
    report = run_demo(bp.circos_plot, quiet=False, answers={
        "use default data": "n", "number of sectors": "2",
        "name:": ["alpha", "beta"], "size of": sizes,
        "number of links": "1", "from sector": "alpha", "to sector": "beta"})

    out = report["stdout"]
    assert report["figures_open"] == 0, out
    assert "Circos needs a positive total angle budget" in out, out
    assert "sum to 0.0" in out, out
    assert report["images"] == [], "a refused plot wrote a file anyway"


def test_the_default_sectors_still_draw_a_circle(run_demo):
    report = run_demo(bp.circos_plot, quiet=False, answers={"use default data": "y"})

    assert report["images"] == ["circos.png"], report["stdout"]
    assert "Error:" not in report["stdout"], report["stdout"]
    assert report["points_drawn"] == 0, "the links are lines, not scatter points"


def test_a_sector_name_given_twice_is_reported_and_the_earlier_size_is_replaced(
        run_demo):
    report = run_demo(bp.circos_plot, quiet=False, answers={
        "use default data": "n", "number of sectors": "2",
        "name:": ["same", "same"], "size of": ["7", "3"],
        "number of links": "1", "from sector": "same", "to sector": "same"})

    out = report["stdout"]
    assert "Sector name 'same' was given twice" in out, out
    assert "1 of the 2 sectors asked for survive" in out, out
    assert "Error:" not in out, out


def test_a_link_to_an_unknown_sector_is_counted_by_name(run_demo):
    report = run_demo(bp.circos_plot, quiet=False, answers={
        "use default data": "n", "number of sectors": "2",
        "name:": ["alpha", "beta"], "size of": ["5", "5"],
        "number of links": "2", "from sector": ["alpha", "beta"],
        "to sector": ["beta", "gamma"]})

    out = report["stdout"]
    assert "1 link(s) named a sector that does not exist" in out, out
    assert "beta->gamma" in out, out
    assert report["images"] == ["circos.png"], out


def test_links_the_sectors_can_hold_are_drawn_without_an_apology(run_demo):
    report = run_demo(bp.circos_plot, quiet=False, answers={
        "use default data": "n", "number of sectors": "2",
        "name:": ["alpha", "beta"], "size of": ["5", "5"],
        "number of links": "1", "from sector": "alpha", "to sector": "beta"})

    out = report["stdout"]
    assert "named a sector that does not exist" not in out, out
    assert report["images"] == ["circos.png"], out
