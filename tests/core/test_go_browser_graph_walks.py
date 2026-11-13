"""Walking the ontology graph: ancestors, and the depth-limited DAG view.

Both walks hand the CLI and the GUI what a user reads as "the terms above this one"
and "the subgraph below it", so a node that does not exist must not appear in either.
"""
import pytest

from biosuite.core import go_browser as gb
from biosuite.core.go_browser import GOBrowser


class Item:
    """A goatools-shaped OBO term: id, name, namespace and parent term objects."""

    def __init__(self, go_id, parents=(), namespace="biological_process"):
        self.id = go_id
        self.name = go_id
        self.namespace = namespace
        self.parents = list(parents)
        setattr(self, "def", f"definition of {go_id}")


def _dag_browser(terms, tmp_path, monkeypatch):
    """A browser fed by a stand-in GODag, built over a real (empty) OBO file."""
    path = tmp_path / "go.obo"
    path.write_text("stub\n", encoding="utf-8")
    monkeypatch.setattr(gb, "HAS_GOATOOLS", True)
    monkeypatch.setattr(gb, "GODag", lambda p: terms)
    return GOBrowser(str(path))


@pytest.fixture()
def browser():
    return GOBrowser()


def test_the_ancestor_list_starts_with_the_queried_term_itself(browser):
    """Documented shape: element 0 is the term asked for, the rest are above it."""
    assert [t.go_id for t in browser.get_ancestors("GO:0005829")] == [
        "GO:0005829", "GO:0005737", "GO:0005575"]


def test_ancestors_of_a_root_are_the_root_alone(browser):
    assert [t.go_id for t in browser.get_ancestors("GO:0005575")] == ["GO:0005575"]


def test_ancestors_of_an_unknown_id_are_no_terms(browser):
    assert browser.get_ancestors("GO:9999999") == []


def test_a_term_reached_by_two_routes_is_visited_once(tmp_path, monkeypatch):
    """A diamond: the shared ancestor appears one time, not once per path."""
    top = Item("GO:top")
    left = Item("GO:left", [top])
    right = Item("GO:right", [top])
    bottom = Item("GO:bottom", [left, right])
    dag = _dag_browser({t.id: t for t in (top, left, right, bottom)}, tmp_path, monkeypatch)
    assert [t.go_id for t in dag.get_ancestors("GO:bottom")] == [
        "GO:bottom", "GO:left", "GO:right", "GO:top"]


def test_a_parent_that_is_not_in_the_loaded_file_is_not_a_term(tmp_path, monkeypatch):
    """A dangling parent id is skipped by the term lookups rather than invented."""
    orphan = Item("GO:orphan", ["GO:not-loaded"])
    dag = _dag_browser({orphan.id: orphan}, tmp_path, monkeypatch)
    assert dag.get_parents("GO:orphan") == []
    assert [t.go_id for t in dag.get_ancestors("GO:orphan")] == ["GO:orphan"]
    assert dag.get_term("GO:not-loaded") is None
    # The asserted edge itself is kept: the children index maps the id the file
    # named, so the orphan is still reachable from it even though that parent was
    # never loaded as a term.
    assert [t.go_id for t in dag.get_children("GO:not-loaded")] == ["GO:orphan"]


def test_the_dag_view_is_breadth_first_and_stops_at_the_requested_depth(browser):
    nodes = browser.get_dag("GO:0005575", depth=1)
    assert [level for _gid, level in nodes] == [0] + [1] * 6
    assert nodes[0] == ("GO:0005575", 0)
    assert ("GO:0005829", 2) not in nodes


def test_depth_zero_is_the_queried_term_alone(browser):
    assert browser.get_dag("GO:0005575", depth=0) == [("GO:0005575", 0)]


def test_a_child_reached_twice_still_appears_once_in_the_dag(tmp_path, monkeypatch):
    top = Item("GO:top")
    left = Item("GO:left", [top])
    right = Item("GO:right", [top, left])
    dag = _dag_browser({t.id: t for t in (top, left, right)}, tmp_path, monkeypatch)
    ids = [go_id for go_id, _level in dag.get_dag("GO:top", depth=3)]
    assert ids == ["GO:top", "GO:left", "GO:right"]


def test_an_unknown_root_gives_an_empty_subgraph_not_a_node_that_does_not_exist(browser):
    """Measured before the fix: get_dag('GO:9999999') returned [('GO:9999999', 0)]."""
    assert browser.get_dag("GO:9999999") == []
    assert browser.get_dag("GO:9999999", depth=3) == []


def test_a_term_with_no_children_is_a_one_node_subgraph(browser):
    assert browser.get_dag("GO:0006915") == [("GO:0006915", 0)]


@pytest.mark.parametrize("depth", [-1, -10])
def test_a_negative_depth_is_refused_rather_than_answering_with_nothing(browser, depth):
    with pytest.raises(ValueError) as exc:
        browser.get_dag("GO:0005575", depth=depth)
    assert f"depth must be 0 or more, got {depth}" in str(exc.value)


@pytest.mark.parametrize("depth", [1.5, "2", None, True])
def test_a_depth_that_is_not_a_whole_number_is_refused(browser, depth):
    with pytest.raises(ValueError) as exc:
        browser.get_dag("GO:0005575", depth=depth)
    assert f"depth must be a whole number of levels, got {depth!r}" in str(exc.value)


def test_the_default_depth_reaches_the_grandchildren_of_a_root(browser):
    nodes = browser.get_dag("GO:0005575")
    assert max(level for _gid, level in nodes) == 2
    assert ("GO:0005829", 2) in nodes
