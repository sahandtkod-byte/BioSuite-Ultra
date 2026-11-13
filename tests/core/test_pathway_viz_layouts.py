"""Where pathway nodes are placed: the grid and the linear layouts.

The layout is the only thing that decides whether a drawn map is readable, and both
of them take a number the caller chooses.
"""
import pytest

from biosuite.core.pathway_viz import PathwayMap, create_custom_pathway


@pytest.fixture()
def seven():
    return create_custom_pathway([f"G{i}" for i in range(7)])


def _positions(pm):
    return [(node.x, node.y) for node in pm.nodes.values()]


def test_the_grid_fills_a_row_before_wrapping(seven):
    seven.layout_grid(n_cols=3)
    assert _positions(seven) == [
        (0, 0), (3, 0), (6, 0),
        (0, -2), (3, -2), (6, -2),
        (0, -4)]


def test_the_grid_keeps_one_column_per_slot_whatever_the_column_count(seven):
    seven.layout_grid(n_cols=1)
    assert [x for x, _y in _positions(seven)] == [0] * 7
    assert [y for _x, y in _positions(seven)] == [0, -2, -4, -6, -8, -10, -12]


def test_a_column_count_larger_than_the_map_holds_one_row(seven):
    seven.layout_grid(n_cols=10)
    assert [y for _x, y in _positions(seven)] == [0] * 7


def test_the_spacings_are_applied_to_the_axis_they_name(seven):
    seven.layout_grid(n_cols=2, spacing_x=5, spacing_y=4)
    assert _positions(seven)[:4] == [(0, 0), (5, 0), (0, -4), (5, -4)]


@pytest.mark.parametrize("n_cols", [0, -1, -3, 0.5, 2.5, None, True, "3"])
def test_a_column_count_that_is_not_a_whole_positive_number_is_refused(seven, n_cols):
    """Measured before the fix: 0 raised ZeroDivisionError.

    0.5 was worse than the crash — `i % 0.5` is 0 for every node, so all seven
    stacked into one overlapping column and the layout reported success.
    """
    with pytest.raises(ValueError) as exc:
        seven.layout_grid(n_cols=n_cols)
    assert f"n_cols must be a whole number of at least 1, got {n_cols!r}" in str(exc.value)


def test_a_rejected_column_count_leaves_the_previous_layout_in_place(seven):
    seven.layout_linear(spacing=3)
    with pytest.raises(ValueError):
        seven.layout_grid(n_cols=0)
    assert _positions(seven) == [(0, 0), (3, 0), (6, 0), (9, 0), (12, 0), (15, 0), (18, 0)]


def test_the_linear_layout_spaces_nodes_along_one_row(seven):
    seven.layout_linear(spacing=2)
    assert _positions(seven) == [(x, 0) for x in (0, 2, 4, 6, 8, 10, 12)]


def test_the_linear_layout_accepts_a_custom_spacing(seven):
    seven.layout_linear(spacing=10)
    assert _positions(seven)[-1] == (60, 0)


def test_an_empty_map_has_an_empty_layout_of_either_kind():
    pm = PathwayMap("none")
    pm.layout_grid(n_cols=4)
    pm.layout_linear()
    assert _positions(pm) == []


def test_a_single_node_is_laid_out_without_a_neighbour():
    pm = create_custom_pathway(["only"])
    pm.layout_grid(n_cols=3)
    assert _positions(pm) == [(0, 0)]
    assert pm.edges == []


def test_the_builder_places_nodes_in_a_line_before_any_layout_call():
    pm = create_custom_pathway(["A", "B", "C"])
    assert _positions(pm) == [(0, 0), (3, 0), (6, 0)]
    assert [(e.source, e.target) for e in pm.edges] == [("g0", "g1"), ("g1", "g2")]


def test_add_node_carries_the_geometry_options_through():
    pm = PathwayMap("g")
    pm.add_node("a", "A", 1, 2, node_type="compound", width=2.5, height=1.5)
    node = pm.nodes["a"]
    assert (node.x, node.y, node.node_type, node.width, node.height) == (
        1, 2, "compound", 2.5, 1.5)


def test_the_builders_chain_so_a_map_can_be_written_in_one_expression():
    pm = PathwayMap("chain")
    assert pm.add_node("a", "A").add_node("b", "B").add_edge("a", "b") is pm
    assert (len(pm.nodes), len(pm.edges)) == (2, 1)


def test_a_node_id_written_twice_keeps_the_last_definition_only_once():
    pm = PathwayMap("dup")
    pm.add_node("a", "first")
    pm.add_node("a", "second")
    assert list(pm.nodes) == ["a"] and pm.nodes["a"].name == "second"


def test_an_edge_is_stored_exactly_once_even_when_it_is_added_twice():
    """The map keeps what it was given; de-duplication is not implied by add_edge.

    Pinned because the report counts these edges, and a caller that re-added one
    should see the count move rather than the list stay mysteriously still.
    """
    pm = PathwayMap("dup")
    pm.add_node("a", "A").add_node("b", "B")
    pm.add_edge("a", "b").add_edge("a", "b")
    assert len(pm.edges) == 2
