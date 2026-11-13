"""The Databases tab: strictly offline search, format parsing and the API-key writer.

``_run_db_search`` normally reaches NCBI, UniProt, PDB, KEGG and Ensembl over HTTP,
so every core search function is replaced by a recorder that hands back a canned
``DBResult``: the tests then prove the typed query is the only argument that left the
tab, that the box shows what the real ``format_search_results`` makes of that object,
and that a raising search function is reported by message.  ``_save_api_keys`` and
``_clear_api_keys`` are steered away from the user's real configuration file by
redirecting ``get_config_file`` into ``tmp_path``, and they check the JSON that was
actually written there.
"""
import json
import os

import pytest

from biosuite.core import databases as core_databases
from biosuite.core import utils
from biosuite.core.databases import DBResult, format_search_results
from biosuite.core.file_formats import (
    format_bed_summary,
    format_gff_summary,
    parse_bed,
    parse_gff,
    parse_newick,
    parse_stockholm,
    tree_to_ascii,
)

SEARCH_FUNCTIONS = ("search_ncbi", "search_uniprot", "search_pdb",
                    "search_kegg", "search_ensembl")

API_ENTRIES = ("api_ncbi_email", "api_ncbi_key", "api_uniprot_email",
               "api_kegg_email", "api_af_email")

KEY_NAMES = ("ncbi_email", "ncbi_api_key", "uniprot_email", "kegg_email",
             "alphafold_email")


def _canned(db, query):
    return DBResult(source=db, query=query, records=[
        {"id": f"{db}-ID-1", "title": f"record for {query}", "score": "42"}])


@pytest.fixture
def searches(monkeypatch):
    """Replace every core search by a recorder; nothing in the tab may touch the network."""
    calls = []

    for name in SEARCH_FUNCTIONS:
        db = name.split("_", 1)[1]

        def make(db=db, name=name):
            def fake(query, *args, **kwargs):
                assert not args and not kwargs, "the tab invented search arguments"
                calls.append((name, query))
                return _canned(db, query)
            return fake

        monkeypatch.setattr(core_databases, name, make())

    def fake_all(query, *args, **kwargs):
        calls.append(("search_all", query))
        return {db: _canned(db, query) for db in ("ncbi", "uniprot")}

    monkeypatch.setattr(core_databases, "search_all", fake_all)
    return calls


@pytest.fixture
def sandboxed_config(tmp_path, monkeypatch):
    """Point the config file at tmp_path and leave the user's own one exactly as found."""
    real_path = utils.get_config_file()
    before = open(real_path, "rb").read() if os.path.exists(real_path) else None
    written = tmp_path / "config" / "biosuite_config.json"
    monkeypatch.setattr(utils, "get_config_file", lambda: str(written))

    snapshot = json.loads(json.dumps(utils.config))
    yield written

    utils.config.clear()
    utils.config.update(snapshot)
    after = open(real_path, "rb").read() if os.path.exists(real_path) else None
    assert after == before, f"the tab wrote to the real user config at {real_path}"


@pytest.fixture(autouse=True)
def _ncbi_selected(ui):
    """The app is session-scoped; every test starts from the default database."""
    ui.app.db_combo.set("NCBI")


def _fill_keys(ui, values):
    for name, value in zip(API_ENTRIES, values):
        ui.set(name, value)


def _expected_keys(values):
    return dict(zip(KEY_NAMES, (v.strip() for v in values)))


# ─── Database search (offline) ───────────────────────────────────────────────

def test_db_search_refuses_an_empty_query(ui, searches):
    ui.clear("db_query")
    ui.set("db_result", "earlier hits")

    ui.app._run_db_search()

    assert ui.messages_of("warning") == ["Enter a search term."]
    assert searches == [], "a refused search must not go out at all"
    assert ui.text("db_result") == "earlier hits\n"


@pytest.mark.parametrize("label,function,source", [
    ("NCBI", "search_ncbi", "ncbi"),
    ("UniProt", "search_uniprot", "uniprot"),
    ("PDB", "search_pdb", "pdb"),
    ("KEGG", "search_kegg", "kegg"),
    ("Ensembl", "search_ensembl", "ensembl"),
])
def test_each_dropdown_choice_calls_its_own_database(ui, searches, label, function, source):
    ui.app.db_combo.set(label)
    ui.set("db_query", "  BRCA1  ")
    ui.clear("db_result")

    ui.app._run_db_search()

    assert searches == [(function, "BRCA1")], "only the chosen database may be queried"
    expected = format_search_results(_canned(source, "BRCA1"))
    assert ui.text("db_result").strip() == expected.strip()
    assert f"{source}-ID-1" in ui.text("db_result")
    assert ui.status() == "Database search complete"
    assert ui.messages == []


def test_the_all_choice_queries_every_database_at_once(ui, searches):
    ui.app.db_combo.set("All")
    ui.set("db_query", "insulin")
    ui.clear("db_result")

    ui.app._run_db_search()

    assert searches == [("search_all", "insulin")]
    box = ui.text("db_result")
    assert box.strip() == format_search_results(
        {db: _canned(db, "insulin") for db in ("ncbi", "uniprot")}).strip()
    assert "=== NCBI Results ===" in box and "=== UNIPROT Results ===" in box
    assert "record for insulin" in box


def test_the_database_choice_cannot_be_typed_over(ui):
    """A typed value used to fall through to NCBI, so the user's chosen database was
    silently replaced; the picker is now a closed list."""
    assert ui.app.db_combo.cget("state") == "readonly"


def test_an_unknown_database_is_refused_not_redirected(ui, searches):
    """``set()`` can still hand the widget a value outside its list, and NCBI is not
    a safe answer to "ChEMBL": refuse it and leave the previous results alone."""
    ui.app.db_combo.set("ChEMBL")
    ui.set("db_query", "aspirin")
    ui.set("db_result", "earlier hits")

    ui.app._run_db_search()

    assert searches == [], "an unknown database must not query anything"
    assert ui.messages_of("error") == [
        "Unknown database: 'ChEMBL'. Choose NCBI, UniProt, PDB, KEGG, Ensembl or All."]
    assert ui.text("db_result") == "earlier hits\n"


def test_db_search_shows_a_database_that_refused_to_answer(ui, monkeypatch):
    def boom(query, *args, **kwargs):
        raise OSError("UniProt is unreachable")

    monkeypatch.setattr(core_databases, "search_uniprot", boom)
    ui.app.db_combo.set("UniProt")
    ui.set("db_query", "P53")
    ui.set("db_result", "earlier hits")
    ui.app._set_status("waiting for a search")

    ui.app._run_db_search()

    assert ui.messages_of("error") == ["UniProt is unreachable"]
    assert ui.text("db_result") == "earlier hits\n"
    assert ui.status() == "waiting for a search", "a failed search must not claim success"


def test_db_search_shows_the_message_of_an_empty_answer_set(ui, monkeypatch):
    """No records is not an exception, so the box must say it in the core's own words."""
    monkeypatch.setattr(core_databases, "search_pdb",
                        lambda query, *a, **k: DBResult(source="pdb", query=query))
    ui.app.db_combo.set("PDB")
    ui.set("db_query", "nothinghere")
    ui.clear("db_result")

    ui.app._run_db_search()

    rendered = ui.text("db_result").strip()
    assert rendered == format_search_results(DBResult(source="pdb",
                                                     query="nothinghere")).strip()
    assert rendered == "No records found.", rendered
    assert ui.status() == "Database search complete"


# ─── File format parsing ─────────────────────────────────────────────────────

def test_parse_refuses_without_a_file(ui):
    ui.clear("ff_path")
    ui.set("ff_result", "earlier parse")

    ui.app._run_parse()

    assert ui.messages_of("warning") == ["Select a file to parse."]
    assert ui.text("ff_result") == "earlier parse\n"


def test_parse_bed_summary_counts_the_intervals(ui, tmp_path):
    path = tmp_path / "regions.bed"
    path.write_text("chr1\t1000\t2000\tgeneA\t0\t+\n"
                    "chr1\t3000\t4500\tgeneB\t50\t-\n"
                    "chr2\t100\t600\tgeneC\t.\t+\n", encoding="utf-8")
    records = parse_bed(str(path))
    assert len(records) == 3, "the fixture BED really holds three intervals"

    ui.set("ff_path", str(path))
    ui.app._run_parse()

    box = ui.text("ff_result")
    assert box.strip() == format_bed_summary(records).strip()
    assert "Records: 3" in box and "Chromosomes: 2" in box
    assert "Total bp: 3,000" in box, box


@pytest.mark.parametrize("suffix", [".gff", ".gtf"])
def test_parse_gff_summarises_the_feature_types(ui, tmp_path, suffix):
    path = tmp_path / f"annotation{suffix}"
    path.write_text("chr1\tbiosuite\tgene\t100\t900\t.\t+\t.\tID=g1;Name=BRCA1\n"
                    "chr1\tbiosuite\texon\t100\t180\t.\t+\t.\tID=e1;Parent=g1\n"
                    "chr1\tbiosuite\tgene\t2000\t2900\t.\t-\t.\tID=g2;Name=EGFR\n",
                    encoding="utf-8")
    records = parse_gff(str(path))
    assert len(records) == 3

    ui.set("ff_path", str(path))
    ui.app._run_parse()

    box = ui.text("ff_result")
    assert box.strip() == format_gff_summary(records).strip()
    assert "gene: 2" in box and "exon: 1" in box, box


def test_parse_newick_draws_the_tree_in_the_box(ui, tmp_path):
    path = tmp_path / "tree.nwk"
    newick = "(A:0.1,(B:0.2,C:0.3)AB:0.4)Root;"
    path.write_text(newick + "\n", encoding="utf-8")
    lines = tree_to_ascii(parse_newick(newick))
    assert len(lines) == 5, "root, A, AB, B and C"

    ui.set("ff_path", str(path))
    ui.app._run_parse()

    box = ui.text("ff_result")
    assert box.strip() == "\n".join(lines).strip()
    for line in lines:
        assert line in box, "every branch of the tree must be shown"


def test_parse_stockholm_lists_the_sequences_and_truncates_long_ones(ui, tmp_path):
    path = tmp_path / "aln.sto"
    head, tail = "ACGT" * 15, "GG" * 20
    long_seq = head + tail          # 100 columns, only the first 60 may be shown
    short_seq = "ACGT-acgt"
    path.write_text("# STOCKHOLM 1.0\n"
                    f"seqA {long_seq}\n"
                    f"seqB {short_seq}\n"
                    "#=GF ID   demoaln\n"
                    "//\n", encoding="utf-8")
    data = parse_stockholm(str(path))
    assert data["alignment"] == {"seqA": long_seq, "seqB": short_seq}

    ui.set("ff_path", str(path))
    ui.app._run_parse()

    box = ui.text("ff_result")
    assert box.strip() == (f"Sequences: {len(data['alignment'])}\n"
                           f"  seqA: {head}...\n"
                           f"  seqB: {short_seq}\n").strip()
    assert head in box and tail not in box, "the box must show 60 columns and an ellipsis"
    assert f"  seqB: {short_seq}\n" in box, "a short sequence gets no ellipsis"


def test_parse_names_a_format_it_does_not_know(ui):
    ui.set("ff_path", os.path.join("nowhere", "matrix.xyz"))
    ui.set("ff_result", "earlier parse")

    ui.app._run_parse()

    assert ui.messages_of("info") == [
        "Unknown format: .xyz\nSupported: .bed, .gff, .gtf, .nwk, .newick, "
        ".tree, .nhx, .sto, .stockholm, .aln"]
    assert ui.text("ff_result") == "earlier parse\n"


def test_parse_reports_a_tree_it_cannot_read(ui, tmp_path):
    path = tmp_path / "broken.nwk"
    path.write_text("(A:0.1,B:0.2;", encoding="utf-8")
    try:
        parse_newick(path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        expected = str(exc)
    else:
        raise AssertionError("parse_newick accepted unbalanced parentheses")

    ui.set("ff_path", str(path))
    ui.clear("ff_result")
    ui.app._run_parse()

    assert ui.messages_of("error") == [expected]
    assert ui.text("ff_result").strip() == ""


# ─── API keys ────────────────────────────────────────────────────────────────

def test_saving_keys_writes_the_json_the_entries_hold(ui, sandboxed_config):
    values = ("me@example.com ", " SECRET ", "u@example.com",
              "k@example.com", "a@example.com")
    _fill_keys(ui, values)

    ui.app._save_api_keys()

    saved = json.loads(sandboxed_config.read_text(encoding="utf-8"))
    assert saved["api_keys"] == _expected_keys(values)
    assert utils.config["api_keys"] == saved["api_keys"]
    assert saved["theme"] == utils.config["theme"], "the rest of the config survives"
    assert ui.text("apikey_status") == "API keys saved successfully!"
    assert ui.status() == "API keys saved"
    assert ui.messages == []


def test_the_saved_config_file_stays_inside_the_sandbox(ui, sandboxed_config):
    _fill_keys(ui, ("me@example.com", "SECRET", "", "", ""))

    ui.app._save_api_keys()

    written = os.path.realpath(str(sandboxed_config))
    assert written.startswith(os.path.realpath(str(sandboxed_config.parent)))
    assert os.listdir(sandboxed_config.parent) == ["biosuite_config.json"], \
        "the atomic writer must not leave a temporary file behind"
    assert os.path.realpath(str(sandboxed_config)) != os.path.realpath(
        os.path.join(os.environ.get("APPDATA", ""), "BioSuite", "biosuite_config.json"))


def test_the_api_key_page_survives_a_config_without_a_key_map(ui, monkeypatch):
    """A hand-edited ``biosuite_config.json`` can hold ``"api_keys": null``.  The
    page is built at startup with no guard around it there, so it must fall back
    to an empty map instead of raising AttributeError on ``None.get``."""
    import customtkinter as ctk

    from biosuite.gui.tabs import databases as tab

    class Builder:
        pass

    stub = Builder()
    stub.T = ui.app.T
    stub.content = ctk.CTkFrame(ui.app.content)
    stub.frames = {}
    for name in ("_section_header", "_card", "_label", "_input_entry",
                 "_action_button", "_save_api_keys", "_clear_api_keys"):
        setattr(stub, name, getattr(ui.app, name))
    monkeypatch.setattr(tab, "config", {"theme": "dark-green", "api_keys": None})
    try:
        tab.DatabasesTabMixin._build_apikey_frame(stub)
        values = [getattr(stub, name).get() for name in API_ENTRIES]
    finally:
        stub.content.destroy()

    assert values == [""] * 5


def test_clearing_keys_empties_the_entries_and_the_file(ui, sandboxed_config):
    _fill_keys(ui, ("me@example.com", "SECRET", "u@example.com", "k@example.com",
                    "a@example.com"))
    ui.app._save_api_keys()
    assert json.loads(sandboxed_config.read_text(encoding="utf-8"))["api_keys"], \
        "positive control: the keys were really on disk before clearing"

    ui.app._clear_api_keys()

    for name in API_ENTRIES:
        assert ui.text(name) == "", f"{name} still holds a key"
    saved = json.loads(sandboxed_config.read_text(encoding="utf-8"))
    assert saved["api_keys"] == {}
    assert utils.config["api_keys"] == {}
    assert ui.text("apikey_status") == "All keys cleared."
    assert ui.status() == "API keys cleared"


def test_clearing_leaves_the_other_settings_alone(ui, sandboxed_config):
    _fill_keys(ui, ("me@example.com", "SECRET", "u@example.com", "", ""))
    ui.app._save_api_keys()
    before = json.loads(sandboxed_config.read_text(encoding="utf-8"))

    ui.app._clear_api_keys()

    after = json.loads(sandboxed_config.read_text(encoding="utf-8"))
    assert after["api_keys"] == {}
    assert {k: v for k, v in after.items() if k != "api_keys"} == \
           {k: v for k, v in before.items() if k != "api_keys"}, \
        "clearing keys must not drop the theme or the plot settings"


def test_the_shared_config_dict_comes_back_unchanged(ui, monkeypatch, tmp_path):
    """The save mutates the module-global ``utils.config``; the sandbox must undo it."""
    original = json.loads(json.dumps(utils.config))
    written = tmp_path / "nested" / "biosuite_config.json"
    monkeypatch.setattr(utils, "get_config_file", lambda: str(written))
    snapshot = json.loads(json.dumps(utils.config))
    _fill_keys(ui, ("sentinel@example.com", "SENTINEL", "", "", ""))

    try:
        ui.app._save_api_keys()
        assert utils.config["api_keys"]["ncbi_api_key"] == "SENTINEL"
    finally:
        utils.config.clear()
        utils.config.update(snapshot)

    assert utils.config == original, "a key written in a test leaked into the session config"
    assert json.loads(written.read_text(encoding="utf-8"))["api_keys"]["ncbi_api_key"] \
        == "SENTINEL"
