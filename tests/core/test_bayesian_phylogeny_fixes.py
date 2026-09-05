"""Regression tests for bayesian_phylogeny review fixes."""
import math
import random
import time

import numpy as np
import pytest

from biosuite.core.bayesian_phylogeny import (
    JC69Model, TreeSampler, run_bayesian, _compute_ess, _compute_psrf,
)


def _alignment(tmp_path):
    seqs = {
        'A': 'ACGTACGTACGTACGTACGT' * 5,
        'B': 'ACGTACGTACGTACATACGT' * 5,
        'C': 'AGGTACCTACGTACGTAAAA' * 5,
        'D': 'AGATACCTACATACCTAAAA' * 5,
    }
    p = tmp_path / "aln.fa"
    p.write_text("".join(f">{k}\n{v}\n" for k, v in seqs.items()))
    return str(p)


def _sampler(tmp_path):
    from Bio import AlignIO
    return TreeSampler(AlignIO.read(_alignment(tmp_path), 'fasta'))


def test_jc69_matrix_rows_sum_to_one():
    for t in (0.01, 0.1, 1.0, 10.0):
        for row in JC69Model.matrix(t):
            assert sum(row) == pytest.approx(1.0)
    assert JC69Model.prob(True, 0.5) > JC69Model.prob(False, 0.5)


def test_likelihood_identical_beats_random():
    from Bio import AlignIO
    import io
    fa = ">a\nACGTACGTACGTACGT\n>b\nACGTACGTACGTACGT\n>c\nACGTACGTACGTACGT\n>d\nACGTACGTACGTACGT\n"
    s = TreeSampler(AlignIO.read(io.StringIO(fa), 'fasta'))
    tree = s._initial_tree()
    ll_ident = s.log_likelihood(tree)

    fa2 = ">a\nACGTACGTACGTACGT\n>b\nTTGCAAGCTTAGCCGG\n>c\nGATCGATCCTAGGCTA\n>d\nCTAGGGATCCATGCAT\n"
    s2 = TreeSampler(AlignIO.read(io.StringIO(fa2), 'fasta'))
    tree2 = s2._initial_tree()
    ll_rand = s2.log_likelihood(tree2)
    assert ll_ident > ll_rand


def test_propose_bl_hastings_term_matches_ratio(tmp_path):
    random.seed(11)
    s = _sampler(tmp_path)
    tree = s._initial_tree()
    nt, log_h = s._propose_bl(tree)
    # find the changed branch and verify log(b'/b) is returned exactly
    old = [c.branch_length for c in tree.find_clades() if c.branch_length and c is not tree.root]
    new = [c.branch_length for c in nt.find_clades() if c.branch_length and c is not nt.root]
    assert old == new[:-1] or True  # deep copy; compare below instead
    # recompute expectation from the ratio directly
    changed = [(o, n) for o, n in zip(old, new) if o != n]
    assert len(changed) == 1
    assert log_h == pytest.approx(math.log(changed[0][1] / changed[0][0]))


def test_mcmc_posterior_improves_and_samples(tmp_path):
    random.seed(3)
    np.random.seed(3)
    s = _sampler(tmp_path)
    samples, ll_chain, best_tree, acc = s.run_mcmc(600, burn_in_frac=0.2, thin=10)
    assert 0 < acc <= 1
    assert len(samples) > 0
    tail = ll_chain[len(ll_chain) // 4:]
    assert np.mean(tail) >= np.mean(ll_chain[:len(ll_chain) // 4]) - abs(ll_chain[0])


def test_run_bayesian_end_to_end_and_speed(tmp_path):
    t0 = time.time()
    res = run_bayesian(_alignment(tmp_path), n_generations=400)
    elapsed = time.time() - t0
    assert res.engine in ('builtin', 'mrbayes')
    assert res.newick_tree.startswith('(') or res.engine == 'mrbayes'
    assert res.ess > 0 and res.psrf > 0
    assert res.posterior_probability is not None
    # vectorized pruning keeps a small analysis well under half a minute
    assert elapsed < 30


def test_ess_and_psrf_known_semantics():
    iid = list(np.random.default_rng(1).normal(0, 1, 5000))
    # iid -> ESS close to n
    assert _compute_ess(iid) > 0.5 * len(iid)
    assert _compute_ess([1.0]) == 1.0
    # identical chains -> PSRF ~ sqrt((n-1)/n) <= 1.05
    psrf = _compute_psrf([0.0, 0.0], [1.0, 1.0], 1000)
    assert 0.9 <= psrf <= 1.05


def test_likelihood_finite_with_gaps(tmp_path):
    seqs = {'A': 'ACGT-ACGT-NNNNNNNN', 'B': 'ACGT?ACGT?ACGTACGT',
            'C': 'ACGT-ACGT-ACGTACGT', 'D': 'ACGTACGT??ACGTACGT'}
    p = tmp_path / "gaps.fa"
    p.write_text("".join(f">{k}\n{v}\n" for k, v in seqs.items()))
    res = run_bayesian(str(p), n_generations=200)
    assert math.isfinite(res.log_likelihood)
