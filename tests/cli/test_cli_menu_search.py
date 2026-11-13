"""The CLI menu search (option ``s`` and the ``search`` subcommand).

``search_menu`` parses the text ``print_menu`` generates, so the search always
describes the menu that is actually displayed — there is no second list to
drift.  The tests here pin that contract: keyword, number and section matches,
case-insensitivity, and the empty / no-match answers.
"""
import argparse
import contextlib
import io
import re

import pytest

from biosuite.cli import menu


def _search(query):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        menu.search_menu(query)
    return out.getvalue()


def _search_plain(query):
    return re.sub(r"\x1b\[[0-9;]*m", "", _search(query))


def test_a_keyword_finds_the_item_with_its_number_and_section():
    out = _search_plain("blast")
    assert "Menu search for 'blast'" in out
    assert "── SEQUENCE ANALYSIS" in out
    assert "20  BLAST Sequence Search" in out


def test_the_match_is_case_insensitive():
    out = _search_plain("BLAST")
    assert "20  BLAST Sequence Search" in out


def test_a_number_finds_its_item():
    out = _search_plain("77")
    assert "── ADVANCED VISUALIZATION" in out
    assert "77  UpSet Plot (Multi-set)" in out


def test_a_section_word_finds_its_items():
    out = _search_plain("cloning")
    assert "── MOLECULAR CLONING" in out
    assert "l  Plasmid Map Viewer" in out


def test_an_empty_query_prints_usage():
    out = _search_plain("   ")
    assert "Usage: search <keyword>" in out


def test_a_query_nothing_matches_says_so():
    out = _search_plain("xyzzy-not-a-tool")
    assert "No menu items match 'xyzzy-not-a-tool'." in out


def test_the_menu_advertises_the_search_option():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        menu.print_menu()
    import re
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue())
    assert "s Search Menu" in plain


def test_the_search_option_is_dispatched(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "blast")
    menu._handle_menu_choice("s")
    out = capsys.readouterr().out
    assert "BLAST Sequence Search" in out


def test_the_search_subcommand_runs_non_interactively():
    entry = menu.COMMAND_REGISTRY["search"]
    assert entry["func"] == "biosuite.cli.menu:search_menu"
    assert entry["args"] == ["query"]

    ns = argparse.Namespace(query="volcano")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = menu._execute_command("search", ns)
    assert code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue())
    assert "1  Volcano Plot" in plain
