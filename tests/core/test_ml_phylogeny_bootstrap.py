"""Regression tests for ml_phylogeny's bootstrap resampling.

Support is counted per internal clade and divided by every replicate that was
attempted, so the number of replicates that produced no tree is a fact the caller
has to see: an empty support dict from a failing resampling run and an empty one
from an unresolved tree used to print the same "with 20 bootstrap replicates".
"""
import logging

import pytest

from biosuite.core import ml_phylogeny as M

pytestmark = pytest.mark.skipif(not M.HAS_BIO, reason="biopython missing")

FASTA = """>human
ACGTACGTACGTACGTACGTACGTACGTACGT
>chimp
ACGTACGTACGTACGTACGTACGTACGTACGA
>mouse
AGGTACGGACGTACGTTCGTACGTACGTACGT
>rat
AGGTACGGACGTACGTTTGTACGTACGTACGT
"""

ALL_TAXA = {'human', 'chimp', 'mouse', 'rat'}


def _alignment():
    from io import StringIO
    from Bio import AlignIO
    return AlignIO.read(StringIO(FASTA), 'fasta')


def _write_fasta(tmp_path):
    path = tmp_path / 'a.fasta'
    path.write_text(FASTA, encoding="utf-8")
    return str(path)


def _nj_failing_after_the_first(real, keep):
    """A stand-in whose first `keep` calls work and whose later ones fail.

    `_builtin_phylogeny` makes the first NJ call for the tree itself, so keep=1
    means "the tree is fine and every replicate fails".
    """
    calls = []

    def nj(alignment):
        calls.append(1)
        if len(calls) <= keep:
            return real(alignment)
        return None, "distance matrix is singular"
    return nj


# ── the happy path ──────────────────────────────────────────────────────────

def test_bootstrap_support_is_clade_frequency():
    support, failed = M._bootstrap_support(_alignment(), n_replicates=30)
    assert support
    assert failed == 0
    for key, value in support.items():
        assert 0 < value <= 1.0
        assert set(key) <= ALL_TAXA
        assert len(key) > 1, "support is per non-trivial clade"


def test_bootstrap_close_pair_supported():
    support, _failed = M._bootstrap_support(_alignment(), n_replicates=50)
    pairs = [set(key) for key in support]
    assert any(pair == {'human', 'chimp'} or pair == {'mouse', 'rat'}
               for pair in pairs)


def test_builtin_phylogeny_end_to_end(tmp_path):
    res = M._builtin_phylogeny(_write_fasta(tmp_path), bootstrap=10)
    assert res.engine == 'builtin'
    assert 'chimp' in res.newick_tree
    assert res.support_values
    assert res.message == "Neighbor-Joining tree with 10 bootstrap replicates"


def test_no_replicates_means_no_support_and_no_division():
    support, failed = M._bootstrap_support(_alignment(), n_replicates=0)
    assert (support, failed) == ({}, 0)


def test_zero_replicates_is_said_instead_of_counted(tmp_path):
    res = M._builtin_phylogeny(_write_fasta(tmp_path), bootstrap=0)
    assert res.message == "Neighbor-Joining tree without bootstrap support"
    assert res.support_values == {}


def test_a_successful_run_leaves_the_log_quiet(caplog):
    with caplog.at_level(logging.WARNING):
        support, failed = M._bootstrap_support(_alignment(), n_replicates=5, seed=1)
    assert failed == 0
    assert support
    assert caplog.text == ""


# ── failing replicates ──────────────────────────────────────────────────────

def test_replicates_that_produce_no_tree_are_counted(monkeypatch):
    monkeypatch.setattr(M, "_nj_tree_from_alignment",
                        lambda a: (None, "distance matrix is singular"))
    support, failed = M._bootstrap_support(_alignment(), n_replicates=4, seed=7)
    assert (support, failed) == ({}, 4)


def test_replicate_failures_are_written_to_the_log(monkeypatch, caplog):
    monkeypatch.setattr(M, "_nj_tree_from_alignment",
                        lambda a: (None, "distance matrix is singular"))
    with caplog.at_level(logging.WARNING):
        M._bootstrap_support(_alignment(), n_replicates=3, seed=1)
    assert "3 of 3 bootstrap replicates produced no tree" in caplog.text
    assert "distance matrix is singular" in caplog.text
    assert "support values are fractions of all 3 replicates" in caplog.text


def test_every_replicate_failing_shows_up_in_the_message(monkeypatch, tmp_path):
    real = M._nj_tree_from_alignment
    monkeypatch.setattr(M, "_nj_tree_from_alignment", _nj_failing_after_the_first(real, 1))
    res = M._builtin_phylogeny(_write_fasta(tmp_path), bootstrap=3, seed=1)
    assert res.support_values == {}
    assert res.message == ("Neighbor-Joining tree; none of the 3 bootstrap replicates "
                           "resolved a clade, so no support values")


def test_a_partial_replicate_failure_is_said_in_the_message(monkeypatch, tmp_path):
    real = M._nj_tree_from_alignment
    monkeypatch.setattr(M, "_nj_tree_from_alignment", _nj_failing_after_the_first(real, 2))
    res = M._builtin_phylogeny(_write_fasta(tmp_path), bootstrap=4, seed=1)
    assert res.message == ("Neighbor-Joining tree with 4 bootstrap replicates "
                           "(3 produced no tree)")
    assert res.support_values, "the one replicate that worked still contributes"


def test_support_is_a_fraction_of_every_replicate_attempted(monkeypatch):
    """Dividing by the replicates that worked would quietly inflate support."""
    real = M._nj_tree_from_alignment
    calls = []

    def nj_fails_every_other(alignment):
        calls.append(1)
        return real(alignment) if len(calls) % 2 else (None, "singular")

    monkeypatch.setattr(M, "_nj_tree_from_alignment", nj_fails_every_other)
    support, failed = M._bootstrap_support(_alignment(), n_replicates=10, seed=3)
    assert failed == 5
    assert support
    assert all(value <= 0.5 + 1e-9 for value in support.values()), \
        "at most 5 of the 10 replicates could contribute"


# ── the replicate-count argument ────────────────────────────────────────────

@pytest.mark.parametrize("bad", [-1, -100, 1.5, "20", True, None])
def test_a_replicate_count_that_is_not_a_count_is_refused(bad):
    with pytest.raises((ValueError, TypeError)):
        M._bootstrap_support(_alignment(), n_replicates=bad)


def test_a_missing_alignment_resamples_nothing():
    assert M._bootstrap_support(None) == ({}, 0)


def test_a_duplicate_sequence_alignment_resolves_no_clade(tmp_path):
    """NJ runs on two identical sequences; there is simply no non-trivial split
    to report, which the message must not describe as a failure."""
    path = tmp_path / 'two.fasta'
    path.write_text(">a\nACGTACGTACGTACGT\n>b\nACGTACGTACGTACGT\n", encoding="utf-8")
    res = M._builtin_phylogeny(str(path), bootstrap=5, seed=1)
    assert res.support_values == {}
    assert "none of the 5 bootstrap replicates resolved a clade" in res.message
