"""Which ontology a GOBrowser actually holds.

The browser can be built from an OBO file or from the built-in subset, and the two
differ by three orders of magnitude in size.  A user who cannot tell them apart
reads "not found" as a fact about Gene Ontology rather than about this session.
"""
import sys
import types

import pytest

from biosuite.core import go_browser as gb
from biosuite.core.go_browser import BASIC_GO_TERMS, GOBrowser


def test_the_default_browser_reports_the_subset_it_loaded():
    browser = GOBrowser()
    assert browser.source == f"built-in GO subset ({len(BASIC_GO_TERMS)} terms, offline)"
    assert len(browser.terms) == len(BASIC_GO_TERMS) == 31


def test_no_obo_file_means_no_warning_because_nothing_was_asked_for(caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.go_browser"):
        GOBrowser()
    assert caplog.text == ""


@pytest.mark.parametrize("obo_file", [None, "", "/data/go-basic.obo"])
def test_an_obo_argument_with_goatools_missing_loads_the_subset_and_says_which(
        caplog, obo_file):
    assert gb.HAS_GOATOOLS is False, "goatools is absent in this environment"
    with caplog.at_level("WARNING", logger="biosuite.core.go_browser"):
        browser = GOBrowser(obo_file)
    assert len(browser.terms) == len(BASIC_GO_TERMS)
    assert browser.source.startswith("built-in GO subset")
    if obo_file:
        assert f"{obo_file} was asked for, but goatools is not installed" in caplog.text
        assert "pip install goatools" in caplog.text, caplog.text
    else:
        assert caplog.text == ""


def test_a_missing_obo_file_is_named_as_missing_rather_than_silently_replaced(
        tmp_path, monkeypatch, caplog):
    """The cause of the fallback has to be distinguishable from the other causes.

    One shared silent fallback made "goatools is not installed", "the file is not
    there" and "the file is there and was read" look identical on screen.
    """
    monkeypatch.setattr(gb, "HAS_GOATOOLS", True)
    missing = str(tmp_path / "go-basic.obo")
    with caplog.at_level("WARNING", logger="biosuite.core.go_browser"):
        browser = GOBrowser(missing)
    assert f"{missing} was asked for but is not there" in caplog.text, caplog.text
    assert len(browser.terms) == len(BASIC_GO_TERMS)


def test_an_obo_file_that_exists_is_loaded_and_named_as_the_source(tmp_path, monkeypatch):
    class Item:
        """A goatools OBO term: name, namespace, parent term objects and 'def'."""

        def __init__(self, go_id, name, namespace, parents, definition):
            self.id = go_id
            self.name = name
            self.namespace = namespace
            self.parents = parents
            setattr(self, "def", definition)

    root = Item("GO:0008150", "biological_process", "biological_process", [],
                "A biological process.")
    child = Item("GO:0006281", "DNA repair", "biological_process", [root],
                 "The restoration of DNA.")

    monkeypatch.setattr(gb, "HAS_GOATOOLS", True)
    monkeypatch.setattr(gb, "GODag",
                        lambda path: {"GO:0008150": root, "GO:0006281": child})
    path = tmp_path / "go-basic.obo"
    path.write_text("fake obo\n", encoding="utf-8")

    browser = GOBrowser(str(path))
    assert browser.source == f"GO OBO file {path}"
    assert len(browser.terms) == 2
    assert browser.get_term("GO:0006281").namespace == "BP"
    assert browser.get_term("GO:0006281").parents == ["GO:0008150"]
    assert browser.get_term("GO:0008150").parents == []
    assert browser.get_term("GO:0006281").definition == "The restoration of DNA."
    assert [t.go_id for t in browser.get_children("GO:0008150")] == ["GO:0006281"]


def test_parents_are_mapped_by_id_one_by_one_not_by_the_first_of_them(
        tmp_path, monkeypatch, caplog):
    """A parent with no GO id is dropped and counted, not kept as an object.

    The loader used to decide for the whole list from the first parent: if that one
    had an `id`, the remaining entries were copied unchanged, and any object among
    them became a children-map key that no GO-id lookup could ever reach.
    """
    class Item:
        def __init__(self, go_id, name, parents):
            self.id, self.name, self.namespace = go_id, name, "biological_process"
            self.parents = parents

    class NoId:
        namespace = "biological_process"

    root = Item("GO:0000001", "root", [])
    odd = NoId()
    mixed = Item("GO:0000002", "mixed", [root, odd, "GO:0000001"])
    monkeypatch.setattr(gb, "HAS_GOATOOLS", True)
    monkeypatch.setattr(gb, "GODag", lambda path: {root.id: root, mixed.id: mixed})
    path = tmp_path / "mixed.obo"
    path.write_text("fake\n", encoding="utf-8")

    with caplog.at_level("WARNING", logger="biosuite.core.go_browser"):
        browser = GOBrowser(str(path))
    assert browser.get_term("GO:0000002").parents == ["GO:0000001"]
    assert "1 parent reference(s) carried no GO id" in caplog.text, caplog.text
    assert all(isinstance(key, str) for key in browser.children)
    assert [t.go_id for t in browser.get_children("GO:0000001")] == ["GO:0000002"]


def test_a_model_that_cannot_be_parsed_is_not_answered_with_a_subset(
        tmp_path, monkeypatch):
    """An unreadable file the caller named is an error, not a reason to guess.

    Silently answering with 31 built-in terms would let a corrupt OBO masquerade as
    a real ontology, and every lookup would then report "not in GO".
    """
    def explode(path):
        raise OSError("truncated obo file")

    monkeypatch.setattr(gb, "HAS_GOATOOLS", True)
    monkeypatch.setattr(gb, "GODag", explode)
    path = tmp_path / "broken.obo"
    path.write_text("garbage\n", encoding="utf-8")
    with pytest.raises(OSError) as exc:
        GOBrowser(str(path))
    assert "truncated obo file" in str(exc.value)


def test_the_namespace_codes_are_the_two_letter_forms_every_consumer_filters_on():
    assert gb.GO_NAMESPACE_CODES == {"biological_process": "BP",
                                     "molecular_function": "MF",
                                     "cellular_component": "CC"}
    assert {t.namespace for t in GOBrowser().terms.values()} == {"BP", "MF", "CC"}


def test_the_browser_uses_goatools_when_the_package_is_installed():
    """The import-time branch that decides whether an OBO file can be read at all.

    With goatools absent this line is never executed in a normal session, and a
    module that silently stopped picking the engine up would look identical.
    """
    import inspect
    import types

    fake_parser = types.ModuleType("goatools.obo_parser")
    marker = object()
    fake_parser.GODag = marker
    fake_pkg = types.ModuleType("goatools")
    fake_pkg.obo_parser = fake_parser
    source = inspect.getsource(gb)

    fresh = types.ModuleType("go_browser_with_goatools")
    fresh.__package__ = "biosuite.core"
    module_sys_modules = dict(sys.modules)
    try:
        sys.modules["goatools"] = fake_pkg
        sys.modules["goatools.obo_parser"] = fake_parser
        exec(compile(source, inspect.getfile(gb), "exec"), fresh.__dict__)
    finally:
        sys.modules.clear()
        sys.modules.update(module_sys_modules)
    assert fresh.HAS_GOATOOLS is True
    assert fresh.GODag is marker
    assert fresh.GOBrowser().source == (
        f"built-in GO subset ({len(BASIC_GO_TERMS)} terms, offline)")


def test_the_browser_holds_no_graph_library_of_its_own(monkeypatch):
    """The DAG walks are hand-rolled; the module never needed networkx.

    It used to import networkx under a `HAS_NX` flag that no code read and a comment
    claiming downstream compatibility — an inert knob that looked like a dependency.
    """
    for name in list(sys.modules):
        if name == "networkx":
            monkeypatch.delitem(sys.modules, name)
    fresh = types.ModuleType("go_browser_without_networkx")
    fresh.__package__ = "biosuite.core"
    with open(gb.__file__, encoding="utf-8") as handle:
        source = handle.read()
    exec(compile(source, gb.__file__, "exec"), fresh.__dict__)
    assert not hasattr(fresh, "HAS_NX") and not hasattr(fresh, "nx")
    browser = fresh.GOBrowser()
    assert browser.get_children("GO:0005575")
