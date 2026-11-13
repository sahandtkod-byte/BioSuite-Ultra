"""The two moves the MCMC proposes: a branch-length scale and an NNI swap.

Both must return a copy — the sampler keeps scoring the current tree after proposing,
so a proposal that mutated its input would move the chain without the Metropolis test
ever being asked — and the branch move must return the Hastings correction the
acceptance ratio needs.  A log-normal multiplier is not symmetric:
`q(old|new)/q(new|old) = new/old`, so the returned term is `log(new/old)` exactly,
while NNI is symmetric on tree space and contributes nothing.
"""
import io
import math

import pytest
from Bio import AlignIO, Phylo
from Bio.Phylo import Newick

from biosuite.core.bayesian_phylogeny import TreeSampler

FASTA_2 = ">A\nACGTACGT\n>B\nACGAACGT\n"
FASTA_3 = ">A\nACGT\n>B\nACGA\n>C\nACGC\n"
FASTA_4 = ">A\nACGTACGT\n>B\nACGAACGT\n>C\nACGTGCGT\n>D\nTCGTACGA\n"


def sampler_for(fasta, seed):
    return TreeSampler(AlignIO.read(io.StringIO(fasta), 'fasta'), seed=seed)


def newick_of(tree):
    buf = io.StringIO()
    Phylo.write(tree, buf, 'newick')
    return buf.getvalue().strip()


def branch_lengths(tree):
    return sorted(round(c.branch_length, 12) for c in tree.find_clades()
                  if c.branch_length is not None)


@pytest.mark.parametrize("fasta, expected_root_degree", [
    pytest.param(FASTA_4, 2, id="four-taxon-trifurcation-collapsed"),
    pytest.param(FASTA_2, 2, id="two-taxon-root-left-alone"),
])
def test_initial_tree_is_made_bifurcating_only_when_the_root_has_three_children(fasta,
                                                                               expected_root_degree):
    instance = sampler_for(fasta, seed=3)
    tree = instance._initial_tree(perturb=0.15)
    assert len(tree.root.clades) == expected_root_degree
    assert all(c.branch_length and c.branch_length > 0 for c in tree.find_clades()
               if c is not tree.root), "NNI needs a bifurcating root and positive branches"


def test_the_perturbation_moves_branch_lengths_and_leaves_none_of_them_zero():
    instance = sampler_for(FASTA_4, seed=3)
    plain = instance._initial_tree()
    perturbed = instance._initial_tree(perturb=0.4)
    assert branch_lengths(perturbed) != branch_lengths(plain)
    assert all(length > 0 for length in branch_lengths(perturbed)), \
        "the floor is 1e-4, and `log_prior` rejects a non-positive branch outright"


def test_branch_proposal_on_a_tree_without_branches_is_a_no_op():
    sampler = sampler_for(FASTA_4, seed=1)
    lonely = Newick.Tree(root=Newick.Clade(name="only"))
    proposed, log_hastings = sampler._propose_bl(lonely)
    assert log_hastings == 0.0
    assert proposed.root.name == "only"
    assert proposed is not lonely, "the sampler must not mutate the current tree"


def test_branch_proposal_moves_exactly_one_branch_and_returns_its_hastings_correction():
    sampler = sampler_for(FASTA_4, seed=11)
    current = sampler._initial_tree()
    before = [c.branch_length for c in current.find_clades()
              if c is not current.root and c.branch_length is not None]

    proposed, log_h = sampler._propose_bl(current)

    after = [c.branch_length for c in proposed.find_clades()
             if c is not proposed.root and c.branch_length is not None]
    changed = [(old, new) for old, new in zip(before, after) if old != new]
    assert len(changed) == 1, "one branch per move, or log(new/old) is not the ratio"
    old, new = changed[0]
    assert log_h == pytest.approx(math.log(new / old))
    assert math.copysign(1, log_h) == math.copysign(1, new - old), \
        "a lengthening move is corrected upward and a shortening one downward"
    assert [c.branch_length for c in current.find_clades()
            if c is not current.root and c.branch_length is not None] == before, \
        "the tree it was proposed from is untouched"


def test_nni_without_an_internal_edge_returns_the_same_topology():
    sampler = sampler_for(FASTA_3, seed=4)
    tree = sampler._initial_tree()
    proposed, log_hastings = sampler._propose_nni(tree)
    assert log_hastings == 0.0, "a move that cannot be made contributes no Hastings term"
    assert sorted(c.name for c in proposed.get_terminals()) == ["A", "B", "C"]
    assert newick_of(proposed) == newick_of(tree), "nothing was swapped, so nothing changed"


def test_nni_rearranges_a_four_taxon_tree_without_touching_its_numbers():
    """The root of a collapsed NJ tree leaves one internal edge below the root, so a
    four-taxon chain does have NNI moves available — 20 of 20 attempts changed the
    topology on this alignment and seed."""
    sampler = sampler_for(FASTA_4, seed=9)
    tree = sampler._initial_tree(perturb=0.2)
    expected_branches = branch_lengths(tree)

    changed = 0
    for _ in range(20):
        proposed, log_h = sampler._propose_nni(tree)
        assert log_h == 0.0, "NNI is symmetric on tree space: no correction applies"
        assert sorted(c.name for c in proposed.get_terminals()) == ["A", "B", "C", "D"]
        assert branch_lengths(proposed) == expected_branches, \
            "the swap moves branches between edges, it does not change their values"
        assert proposed is not tree
        if newick_of(proposed) != newick_of(tree):
            changed += 1
    assert changed == 20
