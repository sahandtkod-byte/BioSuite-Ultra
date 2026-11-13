"""The GO ontology and gene2go cache behind ORA.

These helpers reach the network on the user's behalf and leave files behind in
``~/.biosuite``, so three things have to hold: a cache that is reused instead of
re-fetched, a transfer that only replaces the cache once it is complete, and an
empty answer that says which of the three causes it had.
"""
import os
import time
import types
import urllib.request

import pytest

from biosuite.core import enrichment as en


@pytest.fixture
def home_free(monkeypatch, tmp_path):
    """Point every ``~/.biosuite`` helper at tmp_path instead of the real home."""
    def expanduser(path):
        return str(tmp_path) if path == "~" else path
    monkeypatch.setattr(os.path, "expanduser", expanduser)
    return tmp_path


def test_the_obo_is_reused_from_the_cache_without_any_download(home_free, monkeypatch):
    cached = home_free / ".biosuite" / "go-basic.obo"
    cached.parent.mkdir(parents=True)
    cached.write_text("format-version: 1.2\n", encoding="utf-8")

    def no_network(*args, **kwargs):
        raise AssertionError("the cache was ignored and a download started")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert en._download_go_obo() == str(cached)


def test_the_obo_is_written_from_the_http_response(home_free, monkeypatch):
    seen = {}

    class _Response:
        def read(self):
            return b"format-version: 1.2\n[Term]\nid: GO:0000001\n"

    def fake_urlopen(url, timeout):
        seen["url"] = url
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    path = en._download_go_obo()
    assert seen == {"url": "http://purl.obolibrary.org/obo/go-basic.obo", "timeout": 60}
    assert path == str(home_free / ".biosuite" / "go-basic.obo")
    with open(path, "rb") as handle:
        assert handle.read().startswith(b"format-version: 1.2")


def test_a_failed_obo_download_is_reported_as_no_file(home_free, monkeypatch):
    def refuse(url, timeout):
        raise OSError("no route to purl.obolibrary.org")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert en._download_go_obo() is None
    assert not (home_free / ".biosuite" / "go-basic.obo").exists()


def test_an_unknown_organism_short_circuits_before_any_download(home_free, monkeypatch):
    calls = []
    monkeypatch.setattr(en, "_download_if_needed", lambda *args: calls.append(args))
    assert en._get_associations("martian") == {}
    assert calls == []


@pytest.mark.parametrize("organism, taxid", [
    ("human", 9606),
    ("Homo sapiens", 9606),
    ("mouse", 10090),
    ("4932", 4932),
], ids=["human", "latin name", "mouse", "taxid string"])
def test_each_supported_organism_maps_to_a_taxid_and_fetches_only_gene2go(
        home_free, monkeypatch, organism, taxid):
    """gene2go is the only file this module reads.

    NCBI's per-organism ``gene_info.gz`` used to be fetched on the same 30-day
    cycle and never opened: every association comes from gene2go's own columns, so
    that second transfer was cost without use — on a metered connection that is a
    defect, not an optimisation.
    """
    gene2go = home_free / ".biosuite" / "gene2go.gz"
    gene2go.parent.mkdir(parents=True, exist_ok=True)
    gene2go.write_bytes(b"placeholder")
    requested = []

    def record(filepath, url, max_age_days=30):
        requested.append((os.path.basename(filepath), url))
        return True

    monkeypatch.setattr(en, "_download_if_needed", record)
    associations = en._get_associations(organism)

    assert requested == [("gene2go.gz", "ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz")], \
        requested
    assert associations == {}, "the placeholder is not a gene2go table"
    assert taxid == en.ORGANISM_TAXID[str(organism).lower()]


def test_gene2go_is_filtered_to_the_requested_taxid(home_free, monkeypatch):
    rows = [
        "#TaxID\tEntrezGeneID\tGO_ID\tQuestion\tEvidence\tWith\tCategory",
        "\t",
        "9606\t7157\tGO:0006915\tIB\tIEA\t-\tP\t",
        "9606\t7157\tGO:0006915\tduplicate row\tIEA\t-\tP\t",
        "9606\t7157\tGO:0007049\tIB\tIEA\t-\tP\t",
        "10090\t26517\tGO:0000001\tmouse only\tIEA\t-\tP\t",
        "not_a_taxid\t1\tGO:0000002\tbad\tIEA\t-\tP\t",
        "9606\tonly-one-column",
    ]
    import gzip
    path = home_free / ".biosuite"
    path.mkdir(parents=True, exist_ok=True)
    with gzip.open(path / "gene2go.gz", "wt", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")
    (path / "gene_info.gz").write_bytes(b"")
    monkeypatch.setattr(en, "_download_if_needed",
                        lambda *args, **kwargs: True)
    associations = en._get_associations("human")
    assert associations == {"7157": {"GO:0006915", "GO:0007049"}}


def test_a_gene2go_that_cannot_be_fetched_is_not_parsed_as_an_empty_table(
        home_free, monkeypatch):
    """The download answers whether a usable file is there; without that the
    caller read a missing file as "no associations for this taxid"."""
    calls = []
    monkeypatch.setattr(en, "_download_if_needed",
                        lambda *args, **kwargs: calls.append(args) or False)
    assert en._get_associations("human") == {}
    assert len(calls) == 1, "the parse is skipped once the cache is known to be absent"


def test_a_missing_gene2go_file_yields_no_associations_rather_than_an_exception(
        home_free, monkeypatch):
    monkeypatch.setattr(en, "_download_if_needed", lambda *args, **kwargs: None)
    assert en._get_associations("human") == {}


def test_a_fresh_cache_entry_is_not_downloaded_again(home_free, monkeypatch):
    path = home_free / "gene2go.gz"
    path.write_bytes(b"cached")
    calls = []
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        lambda url, dest: calls.append((url, dest)))
    en._download_if_needed(str(path), "ftp://example.invalid/gene2go.gz", max_age_days=30)
    assert calls == []
    assert path.read_bytes() == b"cached"


def test_a_stale_cache_entry_is_retrieved_into_a_temporary_file_then_renamed(home_free,
                                                                             monkeypatch):
    """The transfer never writes into the cache itself.

    `urlretrieve` streams straight into its destination, so an interrupted FTP call
    left half a gene2go.gz behind that then counted as a fresh cache entry for 30
    days — the caller's next error was "Error parsing gene2go".
    """
    path = home_free / "gene2go.gz"
    path.write_bytes(b"stale")
    old = time.time() - 45 * 86400
    os.utime(path, (old, old))
    calls = []

    def retrieve(url, dest):
        calls.append((url, dest))
        with open(dest, "wb") as handle:
            handle.write(b"fresh from the server")

    monkeypatch.setattr(urllib.request, "urlretrieve", retrieve)
    assert en._download_if_needed(str(path), "ftp://example.invalid/gene2go.gz",
                                  max_age_days=30) is True

    assert len(calls) == 1, calls
    url, dest = calls[0]
    assert url == "ftp://example.invalid/gene2go.gz"
    assert dest != str(path) and dest.endswith(".part"), dest
    assert path.read_bytes() == b"fresh from the server"
    assert not os.path.exists(dest), "the temporary file was left behind"


def test_an_interrupted_download_does_not_replacement_the_cached_copy(home_free,
                                                                      monkeypatch):
    """A transfer that produces nothing keeps the stale file rather than an empty one."""
    path = home_free / "gene2go.gz"
    path.write_bytes(b"stale but readable")
    old = time.time() - 45 * 86400
    os.utime(path, (old, old))

    def writes_nothing(url, dest):
        open(dest, "wb").close()

    monkeypatch.setattr(urllib.request, "urlretrieve", writes_nothing)

    assert en._download_if_needed(str(path), "ftp://example.invalid/gene2go.gz",
                                  max_age_days=30) is True
    assert path.read_bytes() == b"stale but readable"


def test_a_failed_download_reports_the_previous_cache_as_absent(home_free, monkeypatch):
    """False only when nothing usable is left for the caller to read."""
    (home_free / ".biosuite").mkdir(exist_ok=True)
    path = home_free / "gene2go.gz"

    def refuse(url, dest):
        raise OSError("ftp is blocked on this network")

    monkeypatch.setattr(urllib.request, "urlretrieve", refuse)

    assert en._download_if_needed(str(path), "ftp://example.invalid/gene2go.gz") is False
    assert not path.exists()


def test_a_failed_download_leaves_the_caller_with_no_file(home_free, monkeypatch):
    def refuse(url, dest):
        raise OSError("ftp is blocked on this network")

    monkeypatch.setattr(urllib.request, "urlretrieve", refuse)
    path = home_free / "gene_info.gz"
    en._download_if_needed(str(path), "ftp://example.invalid/gene_info.gz")
    assert not path.exists()


def test_associations_are_grouped_by_gene_and_blank_rows_are_ignored(tmp_path):
    path = tmp_path / "assoc.tsv"
    path.write_text("G1\tGO:0000001\nG1\tGO:0000001\nG2\tGO:0000002\nG3\n\n", encoding="utf-8")
    loaded = en._load_associations_file(str(path))
    assert loaded == {"G1": {"GO:0000001"}, "G2": {"GO:0000002"}}
    assert "G3" not in loaded


def test_an_unreadable_association_file_is_reported_as_empty(tmp_path):
    assert en._load_associations_file(str(tmp_path / "absent.tsv")) == {}


def test_a_gene2go_with_rows_only_for_other_taxids_says_so(home_free, monkeypatch, caplog):
    """The file was read and was fine; it just holds nothing for this organism.

    That is a third cause, distinct from "no file" and "unparsable file", and the
    caller used to fold all three into "Could not load gene associations for human".
    """
    import gzip
    import logging
    rows = ["10090\t26517\tGO:0000001\tIB\tIEA\t-\tP\t",
            "10090\t26517\tGO:0000002\tIB\tIEA\t-\tP\t"]
    (home_free / ".biosuite").mkdir(exist_ok=True)
    with gzip.open(home_free / ".biosuite" / "gene2go.gz", "wt", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    monkeypatch.setattr(en, "_download_if_needed", lambda *a, **k: True)

    with caplog.at_level(logging.WARNING, logger="biosuite.core.enrichment"):
        assert en._get_associations("human") == {}

    assert "No GO associations found for taxid 9606" in caplog.text, caplog.text
    assert "skipped" not in caplog.text, caplog.text


def test_rows_that_are_not_records_are_counted_without_losing_the_good_ones(
        home_free, monkeypatch, caplog):
    import gzip
    import logging
    rows = ["#comment line", "\t", "9606\t7157\tGO:0006915\tIB\tIEA\t-\tP\t",
            "not_a_taxid\t1\tGO:0000002\tbad\tIEA\t-\tP\t"]
    (home_free / ".biosuite").mkdir(exist_ok=True)
    with gzip.open(home_free / ".biosuite" / "gene2go.gz", "wt", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    monkeypatch.setattr(en, "_download_if_needed", lambda *a, **k: True)

    with caplog.at_level(logging.WARNING, logger="biosuite.core.enrichment"):
        associations = en._get_associations("human")

    assert associations == {"7157": {"GO:0006915"}}
    assert "skipped 2 gene2go row(s)" in caplog.text, caplog.text
    assert "#comment" not in caplog.text, "a header line is not a malformed row"


def test_an_unreadable_gene2go_cache_is_named_as_the_failure(home_free, monkeypatch,
                                                            caplog):
    """A directory where the file should be is an OSError, not an empty table."""
    import logging
    (home_free / ".biosuite").mkdir(exist_ok=True)
    (home_free / ".biosuite" / "gene2go.gz").mkdir()
    monkeypatch.setattr(en, "_download_if_needed", lambda *a, **k: True)

    with caplog.at_level(logging.ERROR, logger="biosuite.core.enrichment"):
        assert en._get_associations("human") == {}

    assert "Could not read the gene2go cache" in caplog.text, caplog.text


def test_an_ontology_that_downloads_but_cannot_be_written_is_no_ontology(home_free,
                                                                        monkeypatch,
                                                                        caplog):
    """The transfer succeeded and the cache did not; the caller must get no path."""
    import builtins
    import logging
    cache = home_free / ".biosuite"
    cache.mkdir(exist_ok=True)
    real_open = builtins.open

    def refuse(path, mode="r", *args, **kwargs):
        if str(path).endswith("go-basic.obo") and "w" in mode:
            raise OSError(28, "No space left on device")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: types.SimpleNamespace(read=lambda: b"obo bytes"))
    monkeypatch.setattr(builtins, "open", refuse)

    with caplog.at_level(logging.WARNING, logger="biosuite.core.enrichment"):
        assert en._download_go_obo() is None

    assert "could not write" in caplog.text, caplog.text
    assert not any(child.is_file() for child in cache.iterdir()), list(cache.iterdir())


def test_a_partial_download_that_refuses_to_be_deleted_is_still_a_failure(home_free,
                                                                         monkeypatch,
                                                                         caplog):
    import logging
    (home_free / ".biosuite").mkdir(exist_ok=True)
    path = home_free / "gene2go.gz"

    def refuse(url, dest):
        with open(dest, "wb") as handle:
            handle.write(b"half a file")
        raise OSError("connection reset")

    def refuse_unlink(target):
        raise OSError("device busy")

    monkeypatch.setattr(urllib.request, "urlretrieve", refuse)
    monkeypatch.setattr(os, "unlink", refuse_unlink)

    with caplog.at_level(logging.DEBUG, logger="biosuite.core.enrichment"):
        assert en._download_if_needed(str(path), "ftp://example.invalid/gene2go.gz") is False

    assert not path.exists(), "the partial file was promoted into the cache"
    assert "could not remove the partial download" in caplog.text, caplog.text
