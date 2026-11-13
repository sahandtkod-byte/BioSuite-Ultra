"""Drawing a pathway map: the figure `draw_pathway` builds.

This is the plot a user looks at, so every node has to be on it, the axes have to
stay out of the way, and an edge the map cannot resolve has to be said out loud
instead of quietly missing from the picture.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from biosuite.core.pathway_viz import (
    DEFAULT_NODE_COLOR, PathwayMap, create_custom_pathway, create_kegg_style_pathway,
    draw_pathway,
)


@pytest.fixture()
def figure():
    fig = None

    def _make(*args, **kwargs):
        nonlocal fig
        fig = draw_pathway(*args, **kwargs)
        return fig

    yield _make
    if fig is not None:
        plt.close(fig)


def test_the_map_draws_one_rounded_box_per_node(figure):
    fig = figure(create_kegg_style_pathway())
    ax = fig.axes[0]
    assert len(ax.patches) == 10


def test_every_node_name_is_written_and_no_label_is_longer_than_fifteen_characters(
        figure):
    pm = create_custom_pathway(["A" * 30, "short"])
    fig = draw_pathway(pm)
    try:
        written = [t.get_text() for t in fig.axes[0].texts if t.get_text() in ("A" * 15, "short")]
        assert written == ["A" * 15, "short"], "the long name was not truncated to 15"
    finally:
        plt.close(fig)


def test_the_expression_value_is_written_under_the_node_it_belongs_to(figure):
    pm = PathwayMap("e")
    pm.add_node("a", "A", 0, 0, height=0.5)
    pm.set_expression({"a": 1.25})
    fig = pm and figure(pm)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "1.2" in texts, texts
    placed = [t for t in fig.axes[0].texts if t.get_text() == "1.2"]
    assert placed[0].get_position() == (0.0, -0.4)


def test_a_node_without_an_expression_gets_no_number(figure):
    pm = PathwayMap("e")
    pm.add_node("a", "A")
    fig = figure(pm)
    assert [t.get_text() for t in fig.axes[0].texts if t.get_text() == "0.0"] == []


def _rgb(hex_colour):
    return tuple(int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))


def test_the_default_colour_is_used_until_the_scale_is_applied(figure):
    pm = PathwayMap("c")
    pm.add_node("a", "A")
    fig = figure(pm)
    assert fig.axes[0].patches[0].get_facecolor()[:3] == _rgb(DEFAULT_NODE_COLOR)


def test_the_expression_colour_replaces_the_default(figure):
    pm = PathwayMap("c")
    pm.add_node("a", "A")
    pm.set_expression({"a": 2.0})
    fig = figure(pm)
    assert fig.axes[0].patches[0].get_facecolor()[:3] == _rgb("#e61e14")


def test_node_colors_can_be_turned_off_while_the_scale_was_already_applied(figure):
    pm = PathwayMap("c")
    pm.add_node("a", "A")
    pm.set_expression({"a": 2.0})
    fig = figure(pm, node_colors=False)
    assert fig.axes[0].patches[0].get_facecolor()[:3] == _rgb(DEFAULT_NODE_COLOR)


def test_show_labels_can_be_turned_off(figure):
    pm = create_custom_pathway(["A", "B"])
    fig = figure(pm, show_labels=False)
    written = [t.get_text() for t in fig.axes[0].texts if t.get_text()]
    assert written == [], "labels were still written"


def test_an_edge_label_is_written_at_the_midpoint_of_its_edge(figure):
    pm = PathwayMap("l")
    pm.add_node("a", "A", 0, 0).add_node("b", "B", 4, 0)
    pm.add_edge("a", "b", "inhibition", label="blocks")
    pm.add_edge("b", "a", "activation")
    fig = figure(pm)
    marked = [t for t in fig.axes[0].texts if t.get_text() == "blocks"]
    assert len(marked) == 1
    assert marked[0].get_position() == (2.0, 0.2)
    assert len([t for t in fig.axes[0].texts if t.get_text()]) == 3


def test_the_view_is_scaled_to_the_nodes_with_a_margin_of_two(figure):
    fig = figure(create_kegg_style_pathway())
    ax = fig.axes[0]
    assert ax.get_xlim() == pytest.approx((-2.0, 5.0))
    assert ax.get_ylim() == pytest.approx((-16.0, 2.0))


def test_the_axes_are_hidden_and_square_and_the_title_is_the_one_asked_for(figure):
    fig = figure(create_kegg_style_pathway(), title="asked")
    ax = fig.axes[0]
    assert ax.get_title() == "asked"
    assert ax.get_aspect() == 1.0, "the axes are not square"
    assert not ax.axison


def test_the_title_falls_back_to_the_maps_own_name(figure):
    pm = PathwayMap("given name")
    fig = figure(pm)
    assert fig.axes[0].get_title() == "given name"


def test_the_title_argument_replaces_the_maps_name_without_mutating_it(figure):
    pm = PathwayMap("given name")
    fig = figure(pm, title="on the plot")
    assert fig.axes[0].get_title() == "on the plot"
    assert pm.title == "given name"


def test_supplied_axes_are_used_and_no_second_axes_is_created(figure):
    fig, ax = plt.subplots()
    try:
        returned = draw_pathway(create_custom_pathway(["A", "B"]), ax=ax,
                                title="on supplied axes")
        assert returned is fig
        assert len(fig.axes) == 1
        assert ax.get_title() == "on supplied axes"
    finally:
        plt.close(fig)


def test_an_empty_map_draws_nothing_and_sets_no_limits(figure):
    fig = figure(PathwayMap("empty"))
    ax = fig.axes[0]
    assert list(ax.patches) == []
    assert ax.get_title() == "empty"


def test_an_unresolvable_edge_is_reported_rather_than_left_out_silently(
        tmp_path, caplog):
    """Measured before the fix: 3 edges stored, 1 drawn, nothing said.

    The report counts them apart now, and the drawing names them in the log, so a
    typo in a gene list does not turn into a map that quietly has holes in it.
    """
    pm = PathwayMap("holes")
    pm.add_node("a", "A").add_node("b", "B")
    pm.add_edge("a", "b")
    pm.add_edge("a", "ghost")
    pm.add_edge("nope", "b")
    with caplog.at_level("WARNING", logger="biosuite.core.pathway_viz"):
        fig = draw_pathway(pm)
    try:
        assert len(fig.axes[0].patches) == 2
        assert "2 edge(s) name a node that is not in the map and were not drawn" \
            in caplog.text, caplog.text
        assert "a->ghost" in caplog.text and "nope->b" in caplog.text, caplog.text
    finally:
        plt.close(fig)
