"""parse_newick: what counts as one tree, and what the parser refuses to read.

The trees handed to this parser usually come from a file the user points the GUI or
CLI at, so a string that is not one tree has to be said out loud rather than
answered with whichever part happened to parse.
"""
import math

import pytest

from biosuite.core import file_formats as ff


def test_a_three_taxon_tree_reads_as_one_internal_node_and_three_leaves():
    tree = ff.parse_newick("(A:0.1,B:0.2,C:0.3)root;")
    assert tree.name == "root" and tree.is_leaf is False
    assert [(c.name, c.branch_length, c.is_leaf) for c in tree.children] == [
        ("A", 0.1, True), ("B", 0.2, True), ("C", 0.3, True)]


def test_a_nested_tree_keeps_its_shape():
    tree = ff.parse_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);")
    assert len(tree.children) == 2
    assert [len(c.children) for c in tree.children] == [2, 2]
    assert tree.children[0].children[0].branch_length == pytest.approx(0.1)


def test_a_label_without_a_branch_length_is_zero_and_stays_a_leaf():
    leaf = ff.parse_newick("(A,B)C;").children[0]
    assert (leaf.name, leaf.branch_length, leaf.is_leaf) == ("A", 0.0, True)


def test_a_single_label_string_is_a_one_leaf_tree():
    tree = ff.parse_newick("AB;")
    assert (tree.name, tree.children, tree.is_leaf) == ("AB", [], True)


def test_quoted_labels_keep_spaces_commas_and_doubled_quotes():
    tree = ff.parse_newick("('has a space':0.5,'it''s':0.25,B:1)C:0;")
    assert tree.name == "C"
    assert [(c.name, c.branch_length) for c in tree.children] == [
        ("has a space", 0.5), ("it's", 0.25), ("B", 1.0)]


def test_an_unterminated_quote_consumes_the_rest_of_the_string():
    tree = ff.parse_newick("'A")
    assert (tree.name, tree.branch_length, tree.is_leaf) == ("A", 0.0, True)


def test_surrounding_whitespace_and_the_terminator_are_not_part_of_the_label():
    assert ff.parse_newick("  (A:0.1,B:0.2)C:0.3;  ").name == "C"
    assert ff.parse_newick("(A:0.1,B:0.2)C:0.3").name == "C"


@pytest.mark.parametrize("text,message", [
    ("", "empty Newick string"),
    ("   ", "empty Newick string"),
    (";", "empty Newick string"),
    ("   ;  ", "empty Newick string"),
])
def test_an_string_without_a_tree_says_it_is_empty(text, message):
    with pytest.raises(ValueError) as exc:
        ff.parse_newick(text)
    assert str(exc.value) == message


@pytest.mark.parametrize("text,message", [
    ("(A:0.1,B:0.2", "unbalanced parentheses in Newick string"),
    ("((A,B)", "unbalanced parentheses in Newick string"),
    ("(A,'unterminated,B)C;", "unbalanced parentheses in Newick string"),
    ("(A:x,B:0.2)C;", "invalid branch length 'x' after label 'A'"),
    ("(A:1:2,B:0.2)C;", "invalid branch length '1:2' after label 'A'"),
])
def test_an_unbalanced_tree_or_a_non_numeric_branch_names_what_was_wrong(text, message):
    with pytest.raises(ValueError) as exc:
        ff.parse_newick(text)
    assert str(exc.value) == message


@pytest.mark.parametrize("text,fragment", [
    ("(A:0.1,B:0.2)C:0.3 ; trailing junk", "'trailing junk'"),
    ("(A,B)C;(D,E)F;", "'(D,E)F'"),
    ("(A,B)C;root;", "'root'"),
])
def test_a_second_tree_or_junk_after_the_first_is_refused_not_ignored(text, fragment):
    """Reading only the first element used to look like a successful parse.

    A file that holds a forest, or a label written after the terminator, came back
    as the first tree with no complaint, so the user drew one tree and believed the
    file had been read whole.
    """
    with pytest.raises(ValueError) as exc:
        ff.parse_newick(text)
    assert "unexpected text after the first Newick tree" in str(exc.value)
    assert fragment in str(exc.value), str(exc.value)


def test_the_position_named_in_that_error_is_where_the_reading_stopped():
    with pytest.raises(ValueError) as exc:
        ff.parse_newick("(A,B)C;(D,E)F;")
    assert "at position 6" in str(exc.value)


def test_an_empty_sibling_is_kept_as_an_unnamed_node():
    """Newick allows an unnamed node, so an empty field is not a parse error.

    Pinning it here keeps the decision visible: (A,,B) is three children, the
    middle one without a label, not a dropped row.
    """
    assert [c.name for c in ff.parse_newick("(A,,B);").children] == ["A", "", "B"]


def test_a_negative_branch_length_is_kept_because_the_methods_do_produce_one():
    """Neighbor-joining trees really do carry negative branches.

    Refusing them would reject files the field produces, so the parser keeps the
    number and the renderer is the place that has to show it honestly.
    """
    assert ff.parse_newick("((A:-0.5,B:0.5)root);").children[0].children[0].branch_length == -0.5


@pytest.mark.parametrize("text", ["((A:NaN,B:1)root);", "((A:inf,B:1)root);"])
def test_a_non_finite_branch_length_is_not_invented_but_is_passed_through(text):
    lengths = [c.branch_length for c in ff.parse_newick(text).children[0].children]
    assert len(lengths) == 2
    assert any(math.isnan(v) or math.isinf(v) for v in lengths), lengths


def test_a_tree_string_that_is_not_a_string_is_a_type_error():
    with pytest.raises(TypeError) as exc:
        ff.parse_newick(None)
    assert "needs a Newick string, got NoneType" in str(exc.value)
