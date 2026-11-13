"""The text report of a pathway map, and the two ways a map is built for it.

`format_pathway_report` is what the CLI option and the GUI panel print, so its
counts are the only description of the map a user who never sees the plot gets.
"""
import pytest

from biosuite.core.pathway_viz import (
    PathwayMap, create_custom_pathway, create_kegg_style_pathway, format_pathway_report,
)


def test_the_sample_map_is_reported_with_its_nodes_and_its_reactions():
    text = format_pathway_report(create_kegg_style_pathway())
    lines = text.splitlines()
    assert lines[:4] == ["Pathway: MAPK Signaling Pathway", "Nodes: 10", "Edges: 9", ""]
    assert lines[4:14] == ["  EGF", "  EGFR", "  GRB2", "  SOS", "  RAS", "  RAF", "  MEK",
                           "  ERK", "  c-Myc", "  c-Fos"]
    assert lines[15] == "Connections:"
    assert lines[16] == "  EGF --activation--> EGFR"
    assert lines[-2:] == ["  ERK --activation--> c-Myc", "  ERK --activation--> c-Fos"]
    assert len(lines) == 25


def test_expression_values_appear_beside_the_node_they_belong_to():
    pm = PathwayMap("expr")
    pm.add_node("a", "A").add_node("b", "B")
    pm.set_expression({"a": 1.5, "b": -2.0})
    lines = format_pathway_report(pm).splitlines()
    assert "  A (expr=1.50)" in lines and "  B (expr=-2.00)" in lines


def test_a_map_with_no_edges_says_so_instead_of_printing_an_empty_section():
    pm = PathwayMap("lonely")
    pm.add_node("a", "A")
    assert format_pathway_report(pm).splitlines()[-2:] == [
        "Connections:", "  (none: no edge has both of its nodes in this map)"]


def test_an_empty_map_is_reported_as_zero_of_everything():
    assert format_pathway_report(PathwayMap("empty")).splitlines()[:3] == [
        "Pathway: empty", "Nodes: 0", "Edges: 0"]


def test_edges_that_name_no_node_are_counted_apart_from_the_drawn_ones():
    """Measured before the fix: 'Edges: 3' above one listed connection.

    The number the report gave was the number of PathwayEdge objects, while the
    list under it — and the drawing — could only ever show the resolvable ones.
    """
    pm = PathwayMap("holes")
    pm.add_node("a", "A").add_node("b", "B")
    pm.add_edge("a", "b")
    pm.add_edge("a", "ghost")
    lines = format_pathway_report(pm).splitlines()
    assert lines[2] == "Edges: 2 (1 not drawn: their endpoints are not in the map)"
    assert lines[-1] == "  A --activation--> B"


def test_a_map_whose_edges_are_all_unresolvable_lists_none_but_says_why():
    pm = PathwayMap("broken")
    pm.add_node("a", "A")
    pm.add_edge("a", "ghost")
    lines = format_pathway_report(pm).splitlines()
    assert lines[2] == "Edges: 1 (1 not drawn: their endpoints are not in the map)"
    assert lines[-1] == "  (none: no edge has both of its nodes in this map)"


def test_an_unlabelled_edge_type_still_names_itself_in_the_connection_line():
    pm = PathwayMap("kinds")
    pm.add_node("a", "A").add_node("b", "B")
    pm.add_edge("a", "b", "inhibition")
    assert format_pathway_report(pm).splitlines()[-1] == "  A --inhibition--> B"


def test_a_custom_pathway_chains_the_genes_in_the_order_they_were_given():
    pm = create_custom_pathway(["EGF", "EGFR", "ERK"])
    assert [(e.source, e.target) for e in pm.edges] == [("g0", "g1"), ("g1", "g2")]
    assert list(pm.nodes) == ["g0", "g1", "g2"]


@pytest.mark.parametrize("connections", [
    [(0, 7)], [(-1, 0)], [(0, 3), (2, 0)], [(0, 1.5)], [(0, None)],
])
def test_a_connection_naming_no_gene_is_refused_at_the_moment_it_is_given(connections):
    """Measured before the fix: (0, 7) became an edge to 'g7' and vanished.

    The drawing skipped it, the report counted it, and the map held a node id no
    gene was ever assigned.
    """
    with pytest.raises(ValueError) as exc:
        create_custom_pathway(["A", "B", "C"], connections=connections)
    assert "which is not a gene position in a list of 3 names" in str(exc.value)


def test_connections_are_allowed_to_cross_and_skip_so_a_real_pathway_can_be_drawn():
    pm = create_custom_pathway(["A", "B", "C", "D"],
                               connections=[(0, 3), (1, 2), (3, 1)])
    assert [(e.source, e.target) for e in pm.edges] == [
        ("g0", "g3"), ("g1", "g2"), ("g3", "g1")]
    assert not pm.unresolvable_edges()


def test_an_empty_gene_list_makes_an_empty_map_rather_than_failing():
    pm = create_custom_pathway([])
    assert format_pathway_report(pm).splitlines()[:3] == [
        "Pathway: Custom Pathway", "Nodes: 0", "Edges: 0"]


def test_a_gene_list_of_one_has_no_edges_to_connect():
    pm = create_custom_pathway(["solo"])
    assert pm.edges == [] and list(pm.nodes) == ["g0"]


def test_the_sample_map_is_self_consistent_for_drawing_and_reporting():
    pm = create_kegg_style_pathway()
    text = format_pathway_report(pm)
    assert text.count("--activation-->") == len(pm.edges) == 9
    for edge in pm.edges:
        assert edge.source in pm.nodes and edge.target in pm.nodes
