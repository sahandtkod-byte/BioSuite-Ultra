"""Looking a GO term up: search, get_term, parents, children and namespaces.

These are the four calls the CLI's GO option and the GUI's GO browser make, so an
answer they give for a query that asked nothing is an answer the user reads as fact.
"""
import pytest

from biosuite.core.go_browser import GOBrowser, GOTerm


@pytest.fixture()
def browser():
    return GOBrowser()


def test_a_name_is_found_whatever_case_it_is_asked_in(browser):
    assert [t.go_id for t in browser.search("Apoptotic")] == ["GO:0006915"]
    assert [t.go_id for t in browser.search("apoptotic")] == ["GO:0006915"]


def test_surrounding_space_is_not_part_of_the_word_the_user_meant(browser):
    assert [t.go_id for t in browser.search("  nucleus  ")] == ["GO:0005634"]


def test_a_term_can_be_found_by_the_id_it_was_given(browser):
    assert [t.go_id for t in browser.search("go:0006915")] == ["GO:0006915"]


def test_a_query_that_asks_nothing_is_refused_rather_than_answered_with_everything(
        browser, caplog):
    """Measured before the fix: '' matched all 31 terms.

    Both front ends strip the field themselves today, but the function is public,
    and "the whole ontology" is not the answer to an empty question — the user's
    next click decides whether that list looked complete.
    """
    for empty in ("", "   ", "\t", "\n"):
        with pytest.raises(ValueError) as exc:
            browser.search(empty)
        assert "needs a non-empty query" in str(exc.value)
        assert "get_namespace_terms" in str(exc.value)


@pytest.mark.parametrize("query", [None, 7, ["kinase"], b"kinase"])
def test_a_query_that_is_not_text_is_a_type_error(browser, query):
    with pytest.raises(TypeError) as exc:
        browser.search(query)
    assert "needs a string query" in str(exc.value)


def test_a_word_no_term_carries_answers_with_nothing(browser):
    assert browser.search("nothing-matches-this") == []


def test_get_term_answers_with_the_record_or_nothing_at_all(browser):
    term = browser.get_term("GO:0005737")
    assert (term.go_id, term.name, term.namespace) == ("GO:0005737", "cytoplasm", "CC")
    assert browser.get_term("GO:9999999") is None


def test_get_term_needs_the_exact_id_it_was_stored_under(browser):
    assert browser.get_term("go:0005737") is None


def test_parents_are_the_terms_above_and_none_for_a_root(browser):
    assert [t.go_id for t in browser.get_parents("GO:0005829")] == ["GO:0005737"]
    assert browser.get_parents("GO:0005575") == []
    assert browser.get_parents("GO:9999999") == []


def test_children_are_listed_once_each(browser):
    kids = [t.go_id for t in browser.get_children("GO:0016020")]
    assert kids == ["GO:0005886"]


def test_children_of_an_unknown_id_are_no_children(browser):
    assert browser.get_children("GO:9999999") == []


@pytest.mark.parametrize("namespace,expected", [("BP", 12), ("MF", 9), ("CC", 10)])
def test_each_namespace_holds_the_terms_the_subset_defines_for_it(browser, namespace,
                                                                  expected):
    assert len(browser.get_namespace_terms(namespace)) == expected


def test_a_namespace_written_in_lower_case_matches_nothing(browser):
    """The codes are case-sensitive, which is what the consumers pass ('BP')."""
    assert browser.get_namespace_terms("bp") == []


def test_a_namespace_that_does_not_exist_is_reported_as_having_no_terms(browser):
    assert browser.get_namespace_terms("NOPE") == []


def test_a_term_carries_its_id_name_namespace_parents_and_definition():
    term = GOTerm("GO:1", "kinase activity", "MF", ["GO:0"], "Catalysis.")
    assert (term.go_id, term.name, term.namespace, term.parents, term.definition) == (
        "GO:1", "kinase activity", "MF", ["GO:0"], "Catalysis.")
    assert repr(term) == "GOTerm(GO:1, kinase activity, MF)"
    assert GOTerm("GO:2", "x", "BP").parents == []
