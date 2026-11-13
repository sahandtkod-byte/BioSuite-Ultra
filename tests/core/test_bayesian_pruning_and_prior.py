"""Felsenstein pruning and the branch-length prior.

The pruning step is where the alignment meets the tree, so this file pins it against
plain arithmetic rather than against the module's own output: an all-identical
alignment under a near-zero tree must score `n_sites · ln(0.25)` because the root has
one state available and four equiprobable choices, and an all-ambiguous alignment must
score exactly 0.0 because every site contributes ln(1).  Both are the invariants a
wrong matrix or a wrong stationary frequency would break.

Two lifetimes belong here too.  `_pruning` reads `self._p_cache`, which used to be
created inside `log_likelihood` and deleted again on the way out, so calling `_pruning`
on an internal node first raised `AttributeError: 'TreeSampler' object has no attribute
'_p_cache'`; and a child with no usable branch length is scored at
`PRUNING_BRANCH_FLOOR`, a number the log-likelihood genuinely depends on (measured on a
4-taxon tree: -44.416 at 0.001 against -108.858 at 1e-12).
"""
import io
import math

import numpy as np
import pytest
from Bio import AlignIO, Phylo
from Bio.Phylo import Newick

from biosuite.core.bayesian_phylogeny import (
    PRUNING_BRANCH_FLOOR, TreeSampler,
)

FASTA_4 = ">A\nACGTACGT\n>B\nACGAACGT\n>C\nACGTGCGT\n>D\nTCGTACGA\n"
FASTA_AMBIGUOUS = ">A\nACGT\n>B\nANNT\n>C\nACGA\n"
FASTA_ALL_N = ">A\nNNNN\n>B\nNNNN\n>C\nNNNN\n"
FASTA_IDENTICAL = ">A\nACGTACGT\n>B\nACGTACGT\n>C\nACGTACGT\n>D\nACGTACGT\n"
FIXED_TREE = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"


def sampler_for(fasta, seed=1):
    return TreeSampler(AlignIO.read(io.StringIO(fasta), 'fasta'), seed=seed)


def tree_of(newick):
    return Phylo.read(io.StringIO(newick), 'newick')


@pytest.fixture
def sampler():
    return sampler_for(FASTA_4)


def test_pruning_is_uninformative_for_an_unknown_taxon_and_for_ambiguous_bases(sampler):
    ghost = Newick.Clade(name="ghost")
    assert np.array_equal(sampler._pruning(ghost, 0), np.ones(4))
    assert sampler.seq_matrix[1].tolist() == [0, 1, 2, 0, 0, 1, 2, 3]

    ambiguous = sampler_for(FASTA_AMBIGUOUS, seed=2)
    assert ambiguous.seq_matrix[1].tolist() == [0, -1, -1, 3], \
        "N codes as ambiguous, not as a base"
    assert ambiguous.log_likelihood(ambiguous._initial_tree()) == pytest.approx(
        -9.959198392826796)


def test_pruning_an_internal_node_does_not_need_a_likelihood_call_first(sampler):
    """`_p_cache` is built by `log_likelihood`, and `_pruning` reads it — so a sampler
    that had not been asked for a likelihood yet could not be pruned at all."""
    instance = sampler_for(FASTA_4)
    internal = instance._initial_tree().root
    vector = instance._pruning(internal, 0)
    assert len(vector) == 4
    assert np.all(np.isfinite(vector))


def test_the_transition_matrices_are_cached_per_branch_length_and_reused(sampler):
    tree = sampler._initial_tree()
    assert sampler._p_cache == {}, "the cache is emptied when a new tree is scored"
    first = sampler.log_likelihood(tree)
    distinct_lengths = {round(max(c.branch_length or PRUNING_BRANCH_FLOOR, 1e-10), 10)
                        for c in tree.find_clades() if c is not tree.root}
    assert len(sampler._p_cache) == len(distinct_lengths)
    assert sampler.log_likelihood(tree) == pytest.approx(first)


def test_a_branch_with_no_length_is_scored_at_the_floor_and_not_as_identity(sampler):
    """`max(child.branch_length or 0.001, 1e-10)` treats a missing length and an exact
    zero alike.  That is a modelling choice with a measured cost, so it is pinned here
    instead of left inside an `or`."""
    def terminals_at(branch_length):
        leaves = [(name, branch_length) for name in ("A", "B", "C", "D")]
        newick = ("((" + f"A:{leaves[0][1]},B:{leaves[1][1]})"
                  f":0.1,(C:{leaves[2][1]},D:{leaves[3][1]}):0.1);")
        return tree_of(newick)

    at_zero = sampler.log_likelihood(terminals_at(0.0))
    at_floor = sampler.log_likelihood(terminals_at(PRUNING_BRANCH_FLOOR))
    at_epsilon = sampler.log_likelihood(terminals_at(1e-12))
    assert at_zero == pytest.approx(at_floor), "an exact 0 is scored at the floor"
    assert at_zero == pytest.approx(-44.41636525621503)
    assert at_epsilon == pytest.approx(-108.85826336825164), \
        "the identity limit is a different number, so the floor is not decoration"


def test_an_all_identical_alignment_scores_sites_times_ln_quarter(sampler):
    """The plain-arithmetic oracle: one state is available at every site, the root has
    four equiprobable choices, and a near-zero branch makes no difference."""
    instance = sampler_for(FASTA_IDENTICAL)
    near_zero = "((A:1e-9,B:1e-9):1e-9,(C:1e-9,D:1e-9):1e-9);"
    assert instance.log_likelihood(tree_of(near_zero)) == pytest.approx(
        8 * math.log(0.25), abs=1e-6)


def test_an_all_ambiguous_alignment_scores_exactly_zero_whatever_the_tree():
    """Every leaf is uninformative, so the root vector stays [1,1,1,1] and each site
    contributes ln(0.25·4) = ln(1) = 0.  A likelihood that drifts from 0.0 here means
    the stationary frequencies or the gap coding have moved."""
    instance = sampler_for(FASTA_ALL_N)
    for newick in (FIXED_TREE, "((A:0.4,B:0.01):0.2,(C:0.9,D:0.05):0.3);"):
        assert instance.log_likelihood(tree_of(newick)) == pytest.approx(0.0)


def test_identical_data_fits_the_same_tree_far_better_than_random_data():
    """One tree, two alignments — the comparison the earlier test could not make,
    because it scored each alignment under its own neighbour-joining tree."""
    identical = sampler_for(FASTA_IDENTICAL.replace("ACGTACGT", "ACGTACGTACGTACGT"))
    random_dna = (">A\nACGTACGTACGTACGT\n>B\nTTGCAAGCTTAGCCGG\n"
                  ">C\nGATCGATCCTAGGCTA\n>D\nCTAGGGATCCATGCAT\n")
    scored = sampler_for(random_dna)
    tree = tree_of(FIXED_TREE)
    assert identical.log_likelihood(tree) == pytest.approx(-31.552236016089694)
    assert scored.log_likelihood(tree) == pytest.approx(-114.21818796778808)
    assert identical.log_likelihood(tree) > scored.log_likelihood(tree)


def test_log_prior_and_posterior_reject_a_non_positive_branch(sampler):
    zero = Phylo.read(io.StringIO("((A:0.0,B:0.1):0.1,C:0.1,D:0.1);"), 'newick')
    assert sampler.log_prior(zero) == -float('inf')
    assert sampler.log_posterior(zero) == -float('inf')

    healthy = sampler._initial_tree()
    prior = sampler.log_prior(healthy)
    branches = [c.branch_length for c in healthy.find_clades()
                if c is not healthy.root and c.branch_length is not None]
    assert prior == pytest.approx(sum(math.log(10.0) - 10.0 * b for b in branches))
    assert sampler.log_posterior(healthy) == pytest.approx(
        sampler.log_likelihood(healthy) + prior)


def test_the_root_branch_is_outside_the_prior(sampler):
    """`log_prior` skips `tree.root`, and MrBayes-style Newick puts a length there for
    the whole tree — changing it must not move the prior."""
    base = "((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);"
    with_root = base[:-1] + ":0.7;"
    assert tree_of(with_root).root.branch_length == pytest.approx(0.7), \
        "the length really lands on the root, which is the clade the prior skips"
    assert sampler.log_prior(tree_of(base)) == pytest.approx(
        sampler.log_prior(tree_of(with_root)))
