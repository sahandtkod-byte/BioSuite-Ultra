"""The GUI module search: the matcher's rules and the sidebar wiring.

``search_score`` in ``biosuite.gui.text_parsing`` is pure string logic — the
headless tests here pin its rules (every token must match, in any order; exact
substring beats a one-letter typo; Persian synonyms and digits are folded).
The ``gui_app`` tests drive the real sidebar search box and the result
buttons it builds.
"""
import pytest

from biosuite.gui.text_parsing import _one_letter_off, normalise_search_text, search_score


def test_the_typo_helper_refuses_different_lengths():
    assert _one_letter_off("abc", "abcd") is False
    assert _one_letter_off("abcd", "abce") is True


def _matches(query, haystack):
    return search_score(query, haystack) > 0


def test_an_exact_word_matches_its_item():
    assert _matches("alignment", "Alignments")


def test_a_partial_word_matches():
    assert _matches("align", "Alignments")


def test_case_does_not_matter():
    assert _matches("ALIGNMENTS", "Alignments")


def test_accents_are_folded():
    assert search_score("Métabolomics", "Metabolomics") == 1.0


def test_token_order_does_not_matter():
    assert _matches("rna single", "Single-Cell RNA-seq")
    assert _matches("single rna", "Single-Cell RNA-seq")


def test_every_token_must_match():
    assert search_score("align zzz", "Alignments") == 0.0


def test_a_persian_term_finds_its_module():
    assert _matches("تراز", "Alignments")


def test_a_persian_number_is_folded_to_ascii():
    assert _matches("۱۶", "16S rRNA Pipeline")


def test_a_one_letter_typo_still_matches_but_scores_lower():
    assert search_score("alignmant", "Alignments") == 0.6
    assert search_score("alignment", "Alignments") == 1.0


def test_a_short_token_gets_no_typo_license():
    assert search_score("ali", "Alignments") > 0      # a real prefix
    assert search_score("alz", "Alignments") == 0.0   # distance 1, but too short


def test_an_empty_query_matches_nothing():
    assert search_score("   ", "Alignments") == 0.0


def test_a_query_with_no_relation_scores_zero():
    assert search_score("crispr", "Alignments") == 0.0


# ── Sidebar wiring ──────────────────────────────────────────────────────────

def test_the_sidebar_has_a_search_entry(gui_app):
    import customtkinter as ctk
    assert isinstance(gui_app.sidebar_search, ctk.CTkEntry)
    assert len(gui_app._module_index) == len(gui_app.sidebar_buttons)


def test_typing_filters_the_sidebar_and_the_best_match_opens(gui_app):
    app = gui_app
    try:
        app.sidebar_search.insert(0, "align")
        app._on_module_search()
        assert app._search_result_buttons, "no results for 'align'"
        first_label = app._search_result_buttons[0].cget("text")
        assert "Alignments" in first_label

        app._open_best_match()
        assert app._current_frame == "alignment"
    finally:
        app.sidebar_search.delete(0, "end")
        app._on_module_search()
        assert app._search_result_buttons == []


def test_a_persian_query_finds_the_alignment_module(gui_app):
    app = gui_app
    try:
        app.sidebar_search.insert(0, "تراز")
        app._on_module_search()
        labels = [b.cget("text") for b in app._search_result_buttons]
        assert any("Alignments" in l for l in labels)
    finally:
        app.sidebar_search.delete(0, "end")
        app._on_module_search()


def test_a_query_nothing_matches_leaves_no_results(gui_app):
    app = gui_app
    before = app._current_frame
    try:
        app.sidebar_search.insert(0, "xyzzy-not-a-module")
        app._on_module_search()
        assert app._search_result_buttons == []
        app._open_best_match()  # no results: the empty guard must not crash
        assert app._current_frame == before
    finally:
        app.sidebar_search.delete(0, "end")
        app._on_module_search()
