"""`format_metagenomics_report()` and what its lines claim.

The report is the only thing a CLI or GUI user sees.  It used to print "Reads
classified: 3" for a run in which all three reads came back Unknown, list at most
ten taxa without saying that it had stopped, and describe kraken2's taxids as if
they were species names.
"""
import pandas as pd
import pytest

from biosuite.core import metagenomics as mg

ECOLI_KMER = 'ATCGATCGATCGATCG'
BACILLUS = 'TTTTAAAACCCCGGGG'
STAPH = 'GCTAGCTAGCTAGCTA'
LACTO = 'AAAACCCCGGGGTTTT'


@pytest.fixture
def fastq(tmp_path):
    def _write(reads, name="reads.fastq"):
        path = tmp_path / name
        path.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n"
                                for i, s in enumerate(reads)), newline="")
        return str(path)
    return _write


@pytest.fixture
def kraken_report(tmp_path):
    def _write(body, name="k.report"):
        path = tmp_path / name
        path.write_text(body, newline="")
        return str(path)
    return _write


# ── the headline numbers ────────────────────────────────────────────────────

def test_the_header_names_the_engine_and_the_read_count(fastq):
    path = fastq([ECOLI_KMER, BACILLUS])
    text = mg.format_metagenomics_report(mg._builtin_classify(path))
    assert text.splitlines()[0] == "=== Metagenomics Report ==="
    assert "Engine: builtin" in text
    assert "Reads: 2" in text


def test_nothing_is_claimed_as_classified_when_nothing_was(fastq):
    """'Reads classified: 3' was printed for three reads that were all Unknown."""
    path = fastq(['N' * 20] * 3)
    text = mg.format_metagenomics_report(mg._builtin_classify(path))
    assert "Reads: 3" in text
    assert "Assigned to a taxon: 0 read(s)" in text
    assert "Reads classified" not in text


def test_the_assigned_count_comes_from_the_abundance_table(fastq):
    path = fastq([ECOLI_KMER, ECOLI_KMER, 'N' * 20])
    text = mg.format_metagenomics_report(mg._builtin_classify(path))
    assert "Assigned to a taxon: 2 of 3 read(s)" in text


def test_an_empty_result_does_not_invent_a_claim_about_reads():
    text = mg.format_metagenomics_report(mg.TaxonomyResult(engine="builtin"))
    assert "Reads: 0" in text
    assert "Assigned to a taxon" not in text, "no reads, so nothing to report on"


# ── the taxa block ──────────────────────────────────────────────────────────

def test_top_taxa_are_listed_with_their_share(fastq):
    path = fastq([ECOLI_KMER, BACILLUS])
    text = mg.format_metagenomics_report(mg._builtin_classify(path))
    assert "\nTop taxa:" in text
    assert "Escherichia coli: 50.0%" in text
    assert "Bacillus subtilis: 50.0%" in text


def test_every_taxa_line_carries_a_percentage(fastq):
    path = fastq([ECOLI_KMER, BACILLUS, STAPH, LACTO])
    text = mg.format_metagenomics_report(mg._builtin_classify(path))
    block = text.split("Top taxa:")[1]
    assert len([line for line in block.splitlines() if line.startswith("  ")]) == 4
    assert block.count("%") == 4


def test_a_taxa_list_longer_than_the_report_rows_announces_the_rest():
    """Four signature taxa can never overflow the page, so the truncation is
    measured on a table that does: eleven taxa, ten printed, one named."""
    table = pd.DataFrame({
        "taxon": [f"taxon{i}" for i in range(11)],
        "count": [1] * 11,
        "relative_abundance": [100 / 11] * 11,
    })
    res = mg.TaxonomyResult(engine="builtin",
                            classifications=[{"read_id": "r", "taxon": "x"}] * 11,
                            abundance_table=table)
    text = mg.format_metagenomics_report(res)
    assert "taxon0" in text and "taxon9" in text
    assert "taxon10" not in text
    assert f"... and {11 - mg.TAXA_REPORT_ROWS} more taxon(s)." in text


def test_a_kraken2_run_labels_its_numbers_as_taxids(kraken_report):
    """Kraken2 --output carries taxids, not names; the old label invited the
    reader to read '561' as a species."""
    res = mg._parse_kraken2_output(kraken_report(
        "C\tread1\t561\t50\t561:0\nC\tread2\t561\t50\t561:0\n"))
    text = mg.format_metagenomics_report(res)
    assert "Top taxa (kraken2 taxids):" in text
    assert "561: 100.0%" in text
    assert "Reads: 2" in text


def test_no_taxa_block_appears_when_the_table_is_empty(fastq):
    path = fastq(['N' * 20])
    text = mg.format_metagenomics_report(mg._builtin_classify(path))
    assert "Top taxa" not in text


# ── the note line ───────────────────────────────────────────────────────────

def test_the_note_carries_the_classifiers_own_summary(fastq):
    path = fastq([ECOLI_KMER])
    text = mg.format_metagenomics_report(mg._builtin_classify(path))
    assert "Note: Built-in classifier: 1 reads, 1 taxa" in text


def test_a_result_without_a_message_has_no_note_line():
    text = mg.format_metagenomics_report(mg.TaxonomyResult(engine="none"))
    assert "Note" not in text


def test_a_file_that_does_not_exist_explains_itself():
    res = mg.classify_reads("/absent/reads.fastq")
    text = mg.format_metagenomics_report(res)
    assert "Engine: none" in text
    assert "File not found: /absent/reads.fastq" in text
    assert "Reads: 0" in text
