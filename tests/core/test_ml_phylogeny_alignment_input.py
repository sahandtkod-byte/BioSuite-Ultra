"""What ml_phylogeny does with alignments it cannot turn into a tree.

The built-in engine reads FASTA with Biopython and neighbour-joining with a
distance matrix; every route below is a different way that input can be wrong —
absent, unreadable, empty, one sequence, ragged, or a directory — and each has to
come back saying which it was instead of returning an empty tree or a traceback.
"""
import os

import pytest

from biosuite.core import ml_phylogeny as mp

SEQS = [("s1", "ACGTACGTAA"), ("s2", "ACGTACGTAC"), ("s3", "ACGAACGTAA"),
        ("s4", "ACGTGCGTAA")]


@pytest.fixture
def alignment_file(tmp_path):
    path = tmp_path / "aln.fasta"
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in SEQS), encoding="utf-8")
    return str(path)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# ── the file itself ─────────────────────────────────────────────────────────

def test_an_absent_file_is_named_and_no_engine_claims_to_have_run(tmp_path):
    missing = str(tmp_path / "gone.fasta")
    result = mp.build_tree(missing)
    assert result.engine == "none"
    assert result.message == f"File not found: {missing}"
    assert result.newick_tree == ""
    assert result.support_values == {}


def test_a_directory_is_not_mistaken_for_an_alignment(tmp_path, alignment_file):
    """os.path.exists() is true for a directory, so the reader has to be the one
    that says what went wrong."""
    result = mp.build_tree(str(tmp_path))
    assert result.engine == "builtin"
    assert result.message.startswith("Error reading alignment:")
    assert "Permission denied" in result.message or "Is a directory" in result.message


def test_an_empty_file_says_biopython_found_no_records(tmp_path):
    result = mp._builtin_phylogeny(_write(tmp_path, "empty.fasta", ""))
    assert "No records found in handle" in result.message


def test_junk_is_reported_with_biopythons_own_hint(tmp_path):
    result = mp._builtin_phylogeny(_write(tmp_path, "junk.fasta", "# not fasta\n%%%%\n"))
    assert result.message.startswith("Error reading alignment:")
    assert "Pearson" in result.message, "Biopython's own hint must reach the caller"
    assert result.newick_tree == ""


def test_an_unreadable_alignment_is_written_to_the_log(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        mp._builtin_phylogeny(_write(tmp_path, "junk2.fasta", "# x\n%%%%\n"))
    assert "Could not read" in caplog.text
    assert "junk2.fasta" in caplog.text


# ── the shape of the alignment ──────────────────────────────────────────────

def test_ragged_rows_are_refused_by_the_reader(tmp_path):
    """An *alignment* means equal lengths; Bio gives the real reason and the
    message carries it."""
    result = mp._builtin_phylogeny(
        _write(tmp_path, "ragged.fasta", ">a\nACGT\n>b\nACGTACGTACGT\n"))
    assert "Sequences must all be the same length" in result.message


def test_one_sequence_is_a_tree_of_one_taxon(tmp_path):
    """Biopython's NJ accepts it; the report must not pretend a clade was
    inferred."""
    result = mp._builtin_phylogeny(_write(tmp_path, "one.fasta", ">s1\nACGTACGTAA\n"),
                                   bootstrap=0)
    assert result.engine == "builtin"
    assert result.newick_tree == "s1:0.00000;"
    assert result.support_values == {}


def test_two_identical_sequences_produce_no_split(tmp_path):
    text = ">a\nACGTACGTACGTACGT\n>b\nACGTACGTACGTACGT\n"
    result = mp._builtin_phylogeny(_write(tmp_path, "same.fasta", text), bootstrap=0)
    assert result.engine == "builtin"
    assert sorted(t.name for t in mp.parse_newick(result.newick_tree).get_terminals()) \
        == ["a", "b"]


def test_four_sequences_produce_a_resolved_tree(alignment_file):
    result = mp._builtin_phylogeny(alignment_file, bootstrap=0)
    assert result.newick_tree == \
        "((s1:0.00000,s2:0.10000)Inner1:0.00000,s3:0.10000,s4:0.10000)Inner2:0.00000;"


def test_a_neighbour_joining_error_message_replaces_the_success_message(alignment_file,
                                                                       monkeypatch):
    monkeypatch.setattr(mp, "_nj_tree_from_alignment",
                        lambda a: (None, "distance matrix is singular"))
    result = mp._builtin_phylogeny(alignment_file, bootstrap=0)
    assert result.engine == "builtin"
    assert result.message == "distance matrix is singular"
    assert result.newick_tree == ""
    assert result.log_likelihood is None


def test_the_alignment_is_read_only_once_per_call(alignment_file, monkeypatch):
    """Bootstrapping resamples columns; it must not re-open the file per
    replicate."""
    reads = []
    real_read = mp.AlignIO.read

    def counting_read(*args, **kwargs):
        reads.append(args[0])
        return real_read(*args, **kwargs)

    monkeypatch.setattr(mp.AlignIO, "read", counting_read)
    mp._builtin_phylogeny(alignment_file, bootstrap=5, seed=1)
    assert reads == [alignment_file]
