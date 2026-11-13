"""`run_mcmc`: the chain itself — what it records, and that it can be reproduced.

The chain is the part of this module a user cannot check by eye, so the pins are on
counts (generations recorded, samples kept, one scoring per generation) and on the
invariant the sampler claims for itself: `best_tree` is the highest-posterior tree the
chain met, so its likelihood cannot be below the tree the chain started from.  The
acceptance rate and the best tree are pinned with a seed, because an MCMC run without
one is not a reproducible analysis.
"""
import io
import math

import pytest
from Bio import AlignIO, Phylo

from biosuite.core.bayesian_phylogeny import TreeSampler

FASTA_LONGER = (">A\nACGTACGTACGTACGT\n>B\nACGAACGTACATACGT\n"
                ">C\nACGTGCGTACGTAGGT\n>D\nTCGTACGAACGTACGA\n")


def new_sampler(seed=3, fasta=FASTA_LONGER):
    return TreeSampler(AlignIO.read(io.StringIO(fasta), 'fasta'), seed=seed)


def newick_of(tree):
    buf = io.StringIO()
    Phylo.write(tree, buf, 'newick')
    return buf.getvalue().strip()


def test_the_chain_records_every_generation_and_keeps_the_thinned_samples():
    sampler = new_sampler()
    samples, ll_chain, best_tree, accept = sampler.run_mcmc(600, burn_in_frac=0.2, thin=10)
    assert len(ll_chain) == 600, "one log-likelihood per generation, accepted or not"
    assert len(samples) == 48, "generations 120..590 every 10th after a 20% burn-in"
    assert accept == pytest.approx(0.9233333333333333), "554 of 600 proposals accepted"
    assert ll_chain[0] == pytest.approx(-51.17228258158445)
    assert samples[0][1] == pytest.approx(-50.11590710801565), "the first kept sample"


def test_the_best_tree_is_the_highest_posterior_the_chain_met():
    sampler = new_sampler()
    initial = sampler.log_likelihood(sampler._initial_tree())
    assert initial == pytest.approx(-51.143104991259754), \
        "the unperturbed NJ tree, which the chain starts from"
    _samples, _chain, best_tree, _accept = sampler.run_mcmc(600, burn_in_frac=0.2, thin=10)
    assert sampler.log_likelihood(best_tree) == pytest.approx(-50.22877704006137)
    assert sampler.log_likelihood(best_tree) >= initial, \
        "a 'best' tree worse than the starting tree means the tracker is broken"
    assert sorted(c.name for c in best_tree.get_terminals()) == ["A", "B", "C", "D"]


def test_the_same_seed_reproduces_the_chain_and_a_second_run_on_the_sampler_does_not():
    """Two chains drawn from one sampler must differ — the PSRF diagnostic compares
    them, and identically-seeded chains would agree by construction."""
    first = new_sampler().run_mcmc(600, burn_in_frac=0.2, thin=10)
    again = new_sampler().run_mcmc(600, burn_in_frac=0.2, thin=10)
    assert [ll for _newick, ll in first[0]] == [ll for _newick, ll in again[0]]

    sampler = new_sampler()
    one = sampler.run_mcmc(600, burn_in_frac=0.2, thin=10)
    two = sampler.run_mcmc(600, burn_in_frac=0.2, thin=10)
    assert [ll for _newick, ll in one[0]] != [ll for _newick, ll in two[0]]


@pytest.mark.parametrize("n_gen, thin, expected_samples", [
    pytest.param(20, 3, 6, id="twenty-generations"),
    pytest.param(20, 1, 16, id="every-post-burnin-generation"),
    pytest.param(1, 5, 1, id="one-generation-keeps-one-sample"),
])
def test_thinning_counts_the_way_the_message_reports_it(n_gen, thin, expected_samples):
    samples, ll_chain, _best, _accept = new_sampler(seed=7).run_mcmc(
        n_gen, burn_in_frac=0.2, thin=thin)
    assert len(ll_chain) == n_gen
    burn_in = int(n_gen * 0.2)
    kept = sum(1 for g in range(n_gen) if g >= burn_in and (g - burn_in) % thin == 0)
    assert len(samples) == kept == expected_samples


def test_every_sampled_newick_parses_and_carries_the_same_four_taxa():
    samples, _chain, _best, _accept = new_sampler(seed=5).run_mcmc(
        40, burn_in_frac=0.2, thin=4)
    assert len(samples) == 8
    for newick, likelihood in samples:
        tree = Phylo.read(io.StringIO(newick), 'newick')
        assert sorted(c.name for c in tree.get_terminals()) == ["A", "B", "C", "D"]
        assert newick.endswith(";")
        assert math.isfinite(likelihood)
