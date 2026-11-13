"""`classify_16s_rna()` on sequences handed to it in memory.

The scorer underneath is a local window match, so a four-base read used to come
back as 'Escherichia coli' at 100 % confidence and 'classified'.  The number is
still reported; what changed is that the result now says the read was too short to
support it, and that a database which is not a mapping is refused in words.
"""
import logging

import pandas as pd
import pytest

from biosuite.core import metagenomics as mg

ECOLI = mg.SILVA_16S_DB["Escherichia coli"]
STAPH = mg.SILVA_16S_DB["Staphylococcus aureus"]


def test_a_reference_sequence_is_matched_at_full_identity():
    res = mg.classify_16s_rna([("q1", ECOLI)])
    call = res.classifications[0]
    assert call["taxonomy"] == "Escherichia coli"
    assert call["identity"] == 1.0
    assert call["confidence"] == 100.0
    assert call["status"] == "classified"
    assert res.engine == "16S"
    assert res.message == "Classified 1 16S sequences"


def test_a_query_of_nobody_resembles_comes_back_unclassified():
    """'N' occurs in no reference here, so this is a guaranteed zero rather than
    a nearest-neighbour borrow."""
    res = mg.classify_16s_rna([("q", "N" * len(ECOLI))])
    call = res.classifications[0]
    assert call["taxonomy"] == "Unclassified"
    assert call["identity"] == 0.0
    assert call["status"] == "low_confidence"


@pytest.mark.parametrize("length", [1, 2, 4, 8, 19])
def test_a_read_shorter_than_the_window_is_called_too_short(length):
    """Its identity is a prefix comparison of a few bases: the number stays in
    the result, but it no longer dresses up as a call."""
    res = mg.classify_16s_rna([("q", ECOLI[:length])])
    call = res.classifications[0]
    assert call["status"] == "too_short"
    assert call["identity"] == 1.0


def test_a_read_exactly_the_window_long_is_not_called_short():
    res = mg.classify_16s_rna([("q", ECOLI[:mg.IDENTITY_WINDOW])])
    assert res.classifications[0]["status"] == "classified"


def test_the_boundary_between_the_two_statuses_is_the_window():
    short = mg.classify_16s_rna([("q", ECOLI[:mg.IDENTITY_WINDOW - 1])])
    exact = mg.classify_16s_rna([("q", ECOLI[:mg.IDENTITY_WINDOW])])
    assert [short.classifications[0]["status"],
            exact.classifications[0]["status"]] == ["too_short", "classified"]


def test_lowercase_input_is_matched_like_uppercase():
    res = mg.classify_16s_rna([("q", ECOLI.lower())])
    assert res.classifications[0]["taxonomy"] == "Escherichia coli"
    assert res.classifications[0]["identity"] == 1.0


def test_a_custom_database_replaces_the_builtin_one():
    res = mg.classify_16s_rna([("q1", ECOLI), ("q2", "N" * len(ECOLI))],
                              db={"Only taxon": ECOLI})
    assert [c["identity"] for c in res.classifications] == [1.0, 0.0]
    assert [c["taxonomy"] for c in res.classifications] == ["Only taxon",
                                                            "Unclassified"]
    assert res.abundance_table["relative_abundance"].tolist() == [50.0, 50.0]


def test_the_abundance_table_counts_each_called_taxon_once_per_read():
    res = mg.classify_16s_rna([("q1", ECOLI), ("q2", ECOLI), ("q3", STAPH)])
    table = res.abundance_table
    assert table["taxon"].tolist() == ["Escherichia coli", "Staphylococcus aureus"]
    assert table["count"].tolist() == [2, 1]
    assert table["relative_abundance"].tolist() == [pytest.approx(66.6666667, abs=1e-4),
                                                    pytest.approx(33.3333333, abs=1e-4)]


def test_nothing_in_nothing_out():
    res = mg.classify_16s_rna([])
    assert res.classifications == []
    assert res.abundance_table.empty
    assert list(res.abundance_table.columns) == list(mg.ABUNDANCE_COLUMNS)
    assert res.message == "Classified 0 16S sequences"


def test_an_empty_database_matches_nothing_and_says_so_in_the_log(caplog):
    res = mg.classify_16s_rna([("q", ECOLI)], db={})
    assert res.classifications[0]["taxonomy"] == "Unclassified"
    with caplog.at_level(logging.WARNING):
        mg.classify_16s_rna([("q", ECOLI)], db={})
    assert "empty reference database" in caplog.text


@pytest.mark.parametrize("bad_db", [[], ["taxon"], "taxon", 5, None if False else 0])
def test_a_database_that_is_not_a_mapping_is_refused(bad_db):
    with pytest.raises(TypeError, match="db must be a mapping"):
        mg.classify_16s_rna([("q", ECOLI)], db=bad_db)


def test_records_must_be_name_and_sequence_pairs():
    with pytest.raises((TypeError, ValueError)):
        mg.classify_16s_rna([("only_one_element",)])
    with pytest.raises(TypeError):
        mg.classify_16s_rna(None)


def test_a_longer_query_still_finds_the_reference_inside_it():
    """The window slides over the query, so flanking sequence must not hide the
    match — this is what the scorer is for."""
    padded = "NNNNNNNNNN" + ECOLI + "NNNNNNNNNN"
    res = mg.classify_16s_rna([("q", padded)])
    assert res.classifications[0]["taxonomy"] == "Escherichia coli"
    assert res.classifications[0]["identity"] == 1.0
    assert res.classifications[0]["status"] == "classified"


def test_the_dataframe_is_typed_even_with_one_taxon():
    res = mg.classify_16s_rna([("q", ECOLI)])
    assert isinstance(res.abundance_table, pd.DataFrame)
    assert list(res.abundance_table.columns) == list(mg.ABUNDANCE_COLUMNS)
