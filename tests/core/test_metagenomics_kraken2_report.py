"""How a Kraken2 --output report is read by metagenomics.

The per-read line is ``<status>\\t<read id>\\t<taxid>\\t<length>\\t<k-mer hits>``.
Two mistakes lived here: the taxon was taken from column 5 (the k-mer hit list),
and every status letter was treated as a read — including the ``D`` clade-summary
rows, which name a taxon rather than a read and were inventing reads whose id was
a taxid.
"""
import logging

import pytest

from biosuite.core import metagenomics as mg

REPORT = (
    "C\tread1\t561\t50\t561:0 561:1\n"
    "U\tread2\t0\t50\t0:50\n"
    "C\tread3\t561\t50\t561:0 561:1\n"
)


@pytest.fixture
def report(tmp_path):
    def _write(text, name="k.report"):
        path = tmp_path / name
        path.write_text(text, newline="")
        return str(path)
    return _write


def test_the_assignment_lives_in_the_third_column(report):
    res = mg._parse_kraken2_output(report(REPORT))
    assert res.engine == "kraken2"
    assert res.classifications == [
        {"read_id": "read1", "taxon": "561"},
        {"read_id": "read2", "taxon": "0"},
        {"read_id": "read3", "taxon": "561"},
    ]


def test_only_classified_reads_reach_the_abundance_table(report):
    res = mg._parse_kraken2_output(report(REPORT))
    table = res.abundance_table
    assert table["taxon"].tolist() == ["561"]
    assert table["count"].tolist() == [2]
    # The denominator is every read row, classified or not: 2 of 3 reads.
    assert table["relative_abundance"].tolist() == [pytest.approx(66.6666667, abs=1e-4)]


def test_the_message_separates_classified_from_not(report):
    res = mg._parse_kraken2_output(report(REPORT))
    assert res.message == "Kraken2: 3 reads (2 classified, 1 not)"


def test_a_clade_summary_row_is_not_a_read(report, caplog):
    """A 'D' row is kraken2 telling the reader which taxa below the LCA carry
    reads; it has a taxid where the read id belongs."""
    res = mg._parse_kraken2_output(report(REPORT + "D\t561\t2\t2\t561:0 561:1\n"))
    assert [c["read_id"] for c in res.classifications] == ["read1", "read2", "read3"]
    assert res.abundance_table["count"].tolist() == [2]
    assert res.message == "Kraken2: 3 reads (2 classified, 1 not); 1 clade-summary line(s) ignored"
    with caplog.at_level(logging.WARNING):
        mg._parse_kraken2_output(report("D\t561\t2\t2\t561:0\n"))
    assert "1 clade-summary row(s)" in caplog.text


def test_a_row_shorter_than_three_fields_is_dropped_and_counted(report, caplog):
    res = mg._parse_kraken2_output(report("C\tonly_two_fields\n" + REPORT))
    assert len(res.classifications) == 3
    assert "only_two_fields" not in [c["read_id"] for c in res.classifications]
    with caplog.at_level(logging.WARNING):
        mg._parse_kraken2_output(report("C\tbroken\n"))
    assert "1 row(s) that are not kraken2 read rows" in caplog.text


def test_an_unknown_status_letter_is_not_silently_a_read(report, caplog):
    """kraken2 only writes C, U and D. A row with any other first field is either
    a different tool or a corrupt file, and neither belongs in a read count."""
    res = mg._parse_kraken2_output(report("X\tread9\t561\t50\t561:0\n" + REPORT))
    assert [c["read_id"] for c in res.classifications] == ["read1", "read2", "read3"]
    with caplog.at_level(logging.WARNING):
        mg._parse_kraken2_output(report("X\tread9\t561\t50\t561:0\n"))
    assert "1 row(s) that are not kraken2 read rows" in caplog.text


def test_a_four_field_line_is_still_a_read(report):
    """Requiring five fields used to drop every unclassified read, which shrank
    the denominator and inflated the remaining percentages."""
    res = mg._parse_kraken2_output(report("U\tread2\t0\t50\n"))
    assert res.classifications == [{"read_id": "read2", "taxon": "0"}]
    assert res.abundance_table.empty
    assert res.message == "Kraken2: 1 reads (0 classified, 1 not)"


def test_an_empty_report_reads_no_reads(report):
    res = mg._parse_kraken2_output(report(""))
    assert res.classifications == []
    assert res.abundance_table.empty
    assert list(res.abundance_table.columns) == list(mg.ABUNDANCE_COLUMNS)
    assert res.message == "Kraken2: 0 reads (0 classified, 0 not)"


def test_a_report_of_only_unclassified_reads_has_an_empty_but_typed_table(report):
    res = mg._parse_kraken2_output(report("U\tr1\t0\t50\t0:50\n"))
    assert len(res.classifications) == 1
    assert res.abundance_table.empty
    assert list(res.abundance_table.columns) == list(mg.ABUNDANCE_COLUMNS)


def test_taxids_are_kept_as_the_numbers_kraken2_writes(report):
    """No name lookup happens here: the table column is kraken2's taxid, and the
    report labels it as such rather than pretending it is a species name."""
    res = mg._parse_kraken2_output(report("C\tr1\t561\t50\t561:0\n"))
    assert res.abundance_table["taxon"].tolist() == ["561"]
    assert all(isinstance(t, str) for t in res.abundance_table["taxon"])


def test_a_taxon_appearing_once_of_three_reads_is_a_third(report):
    res = mg._parse_kraken2_output(report(
        "C\tr1\t561\t50\t561:0\nC\tr2\t1224\t50\t1224:0\nU\tr3\t0\t50\t0:5\n"))
    table = res.abundance_table
    assert table["relative_abundance"].tolist() == [pytest.approx(33.3333333, abs=1e-4),
                                                    pytest.approx(33.3333333, abs=1e-4)]
