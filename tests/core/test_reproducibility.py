"""Reproducibility regression tests (BSU-028).

Stochastic analyses used the *global* ``random`` / ``np.random`` state, so
identical inputs produced different numbers on every run and a published
result could not be reproduced.  These tests prove that (a) an explicit seed
makes a run bit-reproducible, (b) different seeds still explore differently
(the seed is not silently ignored), and (c) deterministic helpers really are
deterministic.
"""
import numpy as np
import pytest

pytest.importorskip("Bio", reason="Biopython required for phylogeny")

from biosuite.core.bayesian_phylogeny import run_bayesian  # noqa: E402
from biosuite.core.ml_phylogeny import build_tree  # noqa: E402
from biosuite.core.utils import maybe_downsample  # noqa: E402

ALIGNMENT = """>taxonA
ACGTACGTACGTAAGGTTCCAACGTACGTACGTAAGGTTCC
>taxonB
ACGTACGTACGTAAGGTTCCAACGTACGTACGTAAGCTTCC
>taxonC
ACGTTCGTACGTAAGGTACCAACGTTCGTACGTAAGGTACC
>taxonD
ACGAACGTTCGTAAGGTACCAACGAACGTTCGTAAGGTACC
>taxonE
TCGTACGTACGAAAGGTTGCATCGTACGTACGAAAGGTTGC
"""


@pytest.fixture
def alignment_file(tmp_path):
    path = tmp_path / "align.fasta"
    path.write_text(ALIGNMENT)
    return str(path)


# ── deterministic downsampling ──────────────────────────────────────────────

def test_downsample_is_deterministic_and_ordered():
    """It used to call an unseeded np.random.choice without sorting."""
    x = np.arange(1000.0)
    y = x * 3.0
    first_x, first_y = maybe_downsample(x, y, max_points=50)
    for _ in range(5):
        again_x, again_y = maybe_downsample(x, y, max_points=50)
        assert np.array_equal(first_x, again_x)
        assert np.array_equal(first_y, again_y)
    assert np.all(np.diff(first_x) > 0), "downsampled x must stay ordered"
    assert first_x[0] == x[0] and first_x[-1] == x[-1], "endpoints must be kept"
    assert np.array_equal(first_y, first_x * 3.0), "x/y pairing must be preserved"


def test_downsample_accepts_plain_lists():
    """The signature says List[Any]; it used to raise TypeError on a list."""
    x = list(range(500))
    y = [v * 2 for v in x]
    dx, dy = maybe_downsample(x, y, max_points=10)
    assert isinstance(dx, list) and len(dx) == 10
    assert dy == [v * 2 for v in dx]


def test_downsample_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        maybe_downsample([1, 2, 3], [1, 2], max_points=2)


# ── seeded bootstrap ────────────────────────────────────────────────────────

def test_bootstrap_support_is_reproducible_with_a_seed(alignment_file):
    a = build_tree(alignment_file, method="builtin", bootstrap=30, seed=1234)
    b = build_tree(alignment_file, method="builtin", bootstrap=30, seed=1234)
    assert a.support_values == b.support_values
    assert a.support_values, "bootstrap produced no clade support at all"


def test_bootstrap_seed_is_actually_used(alignment_file):
    """A seed that changes nothing would mean it is being ignored."""
    seeds = [build_tree(alignment_file, method="builtin", bootstrap=30, seed=s).support_values
             for s in (1, 2, 3, 4, 5, 6)]
    assert any(s != seeds[0] for s in seeds[1:]), (
        "every seed gave identical support - the seed is not reaching the RNG, "
        "or the bootstrap is not resampling")


def test_bootstrap_support_values_are_valid_frequencies(alignment_file):
    result = build_tree(alignment_file, method="builtin", bootstrap=20, seed=7)
    for clade, value in result.support_values.items():
        assert 0.0 <= value <= 1.0, f"support for {clade} out of range: {value}"


# ── seeded MCMC ─────────────────────────────────────────────────────────────

def test_bayesian_mcmc_is_reproducible_with_a_seed(alignment_file):
    a = run_bayesian(alignment_file, n_generations=200, tool="builtin", seed=99)
    b = run_bayesian(alignment_file, n_generations=200, tool="builtin", seed=99)
    assert a.newick_tree == b.newick_tree
    assert a.log_likelihood == pytest.approx(b.log_likelihood, rel=0, abs=0)


def test_bayesian_mcmc_seed_is_actually_used(alignment_file):
    results = [run_bayesian(alignment_file, n_generations=200, tool="builtin",
                            seed=s).log_likelihood for s in (11, 22, 33, 44)]
    assert len(set(results)) > 1, (
        "every seed gave the identical log-likelihood - the chain is not "
        "stochastic or the seed is ignored")
