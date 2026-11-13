"""The built-in k-mer classifier of metagenomics and its signature table.

`biosuite.core.metagenomics._builtin_classify` votes per read; the earlier version
compared each read against the running totals of the whole file, so every read after
the first collapsed onto whichever taxon happened to win first.  Its `k` argument is
also checked here, because the signature table is a fixed length and a different `k`
silently classified nothing.
"""
import logging

import pytest

from biosuite.core import metagenomics as mg

ECOLI = 'ATCGATCGATCGATCG'
STAPH = 'GCTAGCTAGCTAGCTA'
BACILLUS = 'TTTTAAAACCCCGGGG'
LACTO = 'AAAACCCCGGGGTTTT'


@pytest.fixture
def fastq(tmp_path):
    def _write(reads, name="reads.fastq"):
        path = tmp_path / name
        path.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n"
                                for i, s in enumerate(reads)), newline="")
        return str(path)
    return _write


# ── the signature table ─────────────────────────────────────────────────────

def test_signature_table_is_uniform_length():
    """SIGNATURE_KMER_LENGTH is derived from the first key, so the classifier is
    only correct while every signature has that same length."""
    lengths = {len(kmer) for kmer in mg.KNOWN_TAXA}
    assert lengths == {mg.SIGNATURE_KMER_LENGTH}
    assert mg.SIGNATURE_KMER_LENGTH == 16


def test_every_signature_names_a_known_taxon():
    assert set(mg.KNOWN_TAXA.values()) == {
        'Escherichia coli', 'Staphylococcus aureus', 'Bacillus subtilis',
        'Lactobacillus acidophilus'}
    for kmer in mg.KNOWN_TAXA:
        assert set(kmer) <= set('ACGT'), f"{kmer} is not a DNA k-mer"


# ── voting ──────────────────────────────────────────────────────────────────

def test_a_staph_read_after_three_ecoli_reads_stays_staph(fastq):
    """The global-collapse bug: the fourth read matched Staph clearly and was
    still assigned to the taxon that happened to be ahead in the running count."""
    path = fastq([ECOLI + "A" * 20, ECOLI + "A" * 20, ECOLI + "A" * 20,
                  "C" * 20 + STAPH])
    res = mg._builtin_classify(path)
    assert [c['taxon'] for c in res.classifications] == [
        'Escherichia coli', 'Escherichia coli', 'Escherichia coli',
        'Staphylococcus aureus']


def test_a_read_with_two_votes_for_the_same_taxon_wins_that_taxon(fastq):
    path = fastq([ECOLI + "NNNN" + ECOLI])
    res = mg._builtin_classify(path)
    assert res.classifications[0]['taxon'] == 'Escherichia coli'
    assert res.abundance_table['count'].tolist() == [1]


def test_a_lowercase_read_is_matched_like_an_uppercase_one(fastq):
    """Reads arrive from sequencers in either case; comparing only the raw string
    made every lowercase read Unknown."""
    path = fastq([ECOLI.lower()])
    assert mg._builtin_classify(path).classifications[0]['taxon'] == 'Escherichia coli'


def test_an_ambiguous_read_is_reported_as_unknown_not_guessed(fastq):
    path = fastq(['N' * 20])
    res = mg._builtin_classify(path)
    assert res.classifications == [{'read_id': 'r0', 'taxon': 'Unknown'}]
    assert res.abundance_table.empty
    assert res.message == "Built-in classifier: 1 reads, 0 taxa"


def test_a_read_shorter_than_the_signature_never_matches(fastq):
    path = fastq(['ATCG'])
    assert mg._builtin_classify(path).classifications[0]['taxon'] == 'Unknown'


def test_abundance_is_the_share_of_reads_and_sums_to_at_most_100(fastq):
    path = fastq([ECOLI + "A" * 4, ECOLI + "A" * 4, STAPH + "A" * 4])
    res = mg._builtin_classify(path)
    table = res.abundance_table
    assert table['relative_abundance'].sum() == pytest.approx(100.0)
    assert table.iloc[0]['taxon'] == 'Escherichia coli'
    assert table.iloc[0]['relative_abundance'] == pytest.approx(66.6666667, abs=1e-4)


def test_unknown_reads_keep_the_denominator_honest(fastq):
    """One classified read out of four is 25 % of the file, not 100 %: the old
    percentage was over classified reads only and could exceed 100 %."""
    path = fastq([ECOLI, 'N' * 20, 'N' * 20, 'N' * 20])
    table = mg._builtin_classify(path).abundance_table
    assert table['relative_abundance'].tolist() == [pytest.approx(25.0)]


def test_the_empty_abundance_table_still_has_its_columns(fastq):
    path = fastq(['N' * 20])
    assert list(mg._builtin_classify(path).abundance_table.columns) == \
        list(mg.ABUNDANCE_COLUMNS)


def test_no_read_matching_anything_is_written_to_the_log(fastq, caplog):
    path = fastq(['N' * 20, 'N' * 20])
    with caplog.at_level(logging.WARNING):
        mg._builtin_classify(path)
    assert "None of the 2 read(s)" in caplog.text
    assert "16-mer signature" in caplog.text
    assert "4 reference taxon(s)" in caplog.text


def test_a_read_that_matches_something_leaves_no_log_line(fastq, caplog):
    path = fastq([ECOLI])
    with caplog.at_level(logging.WARNING):
        mg._builtin_classify(path)
    assert caplog.text == ""


# ── the k knob ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_k", [8, 12, 20, 0, -1, 16.0, "16", True, None])
def test_k_is_refused_unless_it_is_the_signature_length(bad_k):
    """`k` used to be accepted and ignored: the table is 16-mers, so any other
    window silently matched nothing and every read came back Unknown."""
    if bad_k is None:
        assert mg._check_kmer_length(None) == mg.SIGNATURE_KMER_LENGTH
        return
    with pytest.raises(ValueError, match="would classify nothing|whole k-mer length"):
        mg._check_kmer_length(bad_k)


def test_the_default_k_classifies_the_signatures():
    assert mg._check_kmer_length(mg.SIGNATURE_KMER_LENGTH) == mg.SIGNATURE_KMER_LENGTH
