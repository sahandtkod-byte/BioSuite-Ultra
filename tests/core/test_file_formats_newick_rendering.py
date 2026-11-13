"""Newick output: tree_to_newick and tree_to_ascii.

These two are how a tree leaves the program — one feeds files to other tools, the
other is what the user reads on screen — so both have to survive a round trip
through parse_newick without losing a label or hiding a branch length.
"""
import pytest

from biosuite.core import file_formats as ff


def test_a_simple_tree_is_written_back_exactly_as_it_came_in():
    assert ff.tree_to_newick(ff.parse_newick("(A:0.1,B:0.2)C:0.3;")) == "(A:0.1,B:0.2)C:0.3;"


def test_an_unnamed_internal_node_is_still_a_valid_newick_string():
    assert ff.tree_to_newick(ff.parse_newick("((A:1,B:1):0.5,C:2);")) == "((A:1,B:1):0.5,C:2):0;"


def test_a_short_branch_keeps_its_order_of_magnitude():
    """'%.6g' is what stops 1e-5 from being written as 0.

    Fixed decimals rounded a short but real branch out of the file, and the tree
    that came back from parse_newick was a different tree.
    """
    assert ff.tree_to_newick(ff.parse_newick("(A:1e-05,B:0.2);")) == "(A:1e-05,B:0.2):0;"


@pytest.mark.parametrize("name", ["a b", "a,b", "a:b", "a[3]", "a;b", "a(1)", "'q'"])
def test_a_label_holding_a_reserved_character_survives_the_round_trip(name):
    node = ff.TreeNode(name=name, branch_length=0.5)
    text = ff.tree_to_newick(node)
    assert ff.parse_newick(text).name == name


def test_an_escaped_quote_is_written_the_way_the_format_reads_it_back():
    assert ff._newick_label("q'uote") == "'q''uote'"
    assert ff._newick_label("plain") == "plain"
    assert ff._newick_label("") == ""


def test_a_round_trip_through_both_writers_keeps_every_label_and_length():
    tree = ff.parse_newick("(A:0.1,(B:0.2,C:0.3)D:0.4,E:1e-05)root:0.6;")
    again = ff.parse_newick(ff.tree_to_newick(tree))
    assert again.name == "root" and again.branch_length == pytest.approx(0.6)
    assert [c.name for c in again.children] == ["A", "D", "E"]
    assert again.children[1].children[1].branch_length == pytest.approx(0.3)
    assert again.children[2].branch_length == pytest.approx(1e-05)


def test_a_named_root_and_its_two_children_are_three_indented_lines():
    assert ff.tree_to_ascii(ff.parse_newick("(A:0.1,B:0.2)C:0.3;")) == [
        "└── C (0.300)",
        "    ├── A (0.100)",
        "    └── B (0.200)",
    ]


def test_an_unnamed_node_is_described_by_the_children_it_holds():
    assert ff.tree_to_ascii(ff.parse_newick("((A:0,B:1)root);")) == [
        "└── (1 children)",
        "    └── root",
        "        ├── A",
        "        └── B (1.000)",
    ]


def test_a_branch_of_zero_length_is_not_shown_as_a_number():
    lines = ff.tree_to_ascii(ff.parse_newick("(A:0,B:1)C;"))
    assert lines[1] == "    ├── A"


def test_a_negative_branch_is_printed_rather_than_left_out():
    """Neighbor-joining produces these, and a silent omission reads as 'no length'.

    The condition this replaces asked for a length greater than zero, so the taxon
    below appeared with no number at all and the reader could only assume the input
    tree had not carried one.
    """
    lines = ff.tree_to_ascii(ff.parse_newick("((A:-0.5,B:0.5)root);"))
    assert lines[2] == "        ├── A (-0.500)"
    assert lines[3] == "        └── B (0.500)"


def test_a_non_finite_branch_is_printed_as_the_number_it_is():
    lines = ff.tree_to_ascii(ff.parse_newick("((A:NaN,B:1)root);"))
    assert lines[2] == "        ├── A (nan)"


def test_a_two_level_tree_draws_the_vertical_continuation_of_its_first_branch():
    assert ff.tree_to_ascii(ff.parse_newick("((human:0.1,chimp:0.1):0.05,mouse:0.2);")) == [
        "└── (2 children)",
        "    ├── (2 children) (0.050)",
        "    │   ├── human (0.100)",
        "    │   └── chimp (0.100)",
        "    └── mouse (0.200)",
    ]


def test_a_single_label_tree_is_one_line_with_no_branch():
    assert ff.tree_to_ascii(ff.parse_newick("AB;")) == ["└── AB"]
