"""The built-in chain: what one call to `_builtin_bayesian` promises and reports.

These are the outcomes a user of the CLI or the API sees, so each one is pinned on the
returned fields and on the message text — including the two cases the module used to get
wrong in opposite directions: a request for 50 generations running 800 of them, and a
chain too short to diagnose reporting a *perfect* convergence proxy.
"""
import pytest

from biosuite.core import bayesian_phylogeny as bp
from biosuite.core.bayesian_phylogeny import TreeSampler, _builtin_bayesian

FASTA_4 = ">A\nACGTACGT\n>B\nACGAACGT\n>C\nACGTGCGT\n>D\nTCGTACGA\n"
FASTA_2 = ">A\nACGTACGT\n>B\nACGAACGT\n"
FASTA_LONGER = (">A\nACGTACGTACGTACGT\n>B\nACGAACGTACATACGT\n"
                ">C\nACGTGCGTACGTAGGT\n>D\nTCGTACGAACGTACGA\n")
FASTA_TINY = ">A\nA\n>B\nC\n>C\nG\n"


def alignment(tmp_path, text, name="aln.fasta"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def counted(monkeypatch):
    """Replace `TreeSampler.log_likelihood` with a wrapper that counts evaluations.

    The chain scores the current tree once and then one proposed tree per generation,
    so `2 chains x (n_gen + 1)` is exactly the work a call did — a bound a machine can't
    argue with, where the wall-clock assertion this replaces could."""
    calls = []
    original = TreeSampler.log_likelihood

    def counting(self, tree):
        calls.append(1)
        return original(self, tree)

    monkeypatch.setattr(TreeSampler, "log_likelihood", counting)
    return calls


@pytest.mark.parametrize("requested, expected_evaluations", [
    pytest.param(50, 102, id="fifty"),
    pytest.param(500, 1002, id="five-hundred"),
])
def test_the_requested_number_of_generations_is_the_number_that_runs(counted, tmp_path,
                                                                     requested,
                                                                     expected_evaluations):
    """Measured before the fix: `n_generations=50` and `n_generations=500` on this
    4-taxon x 200-site alignment both ran 800 generations and 1,602 evaluations,
    because a data-sized term `max(n_generations, min(n_taxa * n_sites, 50000))`
    overrode the argument.  A request for a quick look could cost 50,000 generations
    of pure-Python MCMC with nothing said about it."""
    wide = ">A\n" + "ACGT" * 50 + "\n>B\n" + "ACGAACGT" * 25 + "\n" \
           + ">C\n" + "ACGTGCGT" * 25 + "\n>D\n" + "TCGTACGA" * 25 + "\n"
    result = _builtin_bayesian(alignment(tmp_path, wide), n_generations=requested, seed=1)
    assert len(counted) == expected_evaluations
    assert result.engine == "builtin"


def test_the_chain_reports_the_thinning_it_applied(counted, tmp_path):
    _builtin_bayesian(alignment(tmp_path, FASTA_4), n_generations=60, seed=5)
    assert len(counted) == 122, "2 chains x (60 + 1) scorings"


def test_posterior_probability_is_only_reported_when_it_can_be_wrong(tmp_path):
    """Four samples or fewer cannot make this statistic anything but 1.0: the largest
    standardised deviation a set of n points reaches is sqrt(n-1) sigma, which is under
    2 for n <= 4.  The module reported that guaranteed 1.0 as a convergence-quality
    score; it now reports 0.0 — nothing computed — and says so in the message."""
    tiny = alignment(tmp_path, FASTA_TINY, "tiny.fasta")
    result = _builtin_bayesian(tiny, n_generations=1, seed=3)
    assert (result.engine, result.num_samples) == ("builtin", 1)
    assert result.posterior_probability == 0.0
    assert result.ess == pytest.approx(1.0)
    assert result.psrf == pytest.approx(1.0)
    assert "posterior proxy needs >=5 samples, 1 kept" in result.message
    assert "PSRF needs >=2 post-burnin iterations per chain" in result.message


def test_a_chain_too_short_for_the_proxy_produces_no_numpy_warning(tmp_path):
    """`np.var(half, ddof=1)` on a one-iteration half-chain is nan with a
    RuntimeWarning, and nan reached the report as "PSRF: nan"."""
    import warnings as _warnings
    tiny = alignment(tmp_path, FASTA_TINY, "tiny2.fasta")
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = _builtin_bayesian(tiny, n_generations=2, seed=3)
    assert [(w.category.__name__, str(w.message)) for w in caught] == []
    assert result.num_samples == 2
    assert result.psrf == pytest.approx(1.0)
    assert result.posterior_probability == 0.0


def test_eight_samples_report_the_proxy_they_can_support(tmp_path):
    """`n_generations=10` on the same 3-taxon file keeps 8 samples, which is enough for
    the proxy and for a real PSRF — measured 1.0 and 1.317."""
    result = _builtin_bayesian(alignment(tmp_path, FASTA_TINY, "tiny3.fasta"),
                               n_generations=10, seed=3)
    assert (result.num_samples, result.posterior_probability) == (8, pytest.approx(1.0))
    assert result.psrf == pytest.approx(1.317)
    assert result.ess == pytest.approx(8.0)
    assert "thin=1" in result.message
    assert "needs >=5 samples" not in result.message


def test_posterior_probability_counts_two_sigma_samples_and_is_not_clade_support(tmp_path):
    """PSRF 1.752 says the chains disagree; the 0.95 "posterior" still looks excellent."""
    result = _builtin_bayesian(alignment(tmp_path, FASTA_LONGER), n_generations=400,
                               sample_freq=20, seed=1)
    assert result.num_samples == 20
    assert result.posterior_probability == pytest.approx(0.95), \
        "19 of 20 samples within 2 sigma"
    assert result.ess == pytest.approx(5.1)
    assert result.psrf == pytest.approx(1.752), "well above the 1.1 convergence threshold"
    assert result.log_likelihood == pytest.approx(-51.27)
    assert result.newick_tree.rstrip().endswith(";")
    assert "2 chains, burn-in 20%" in result.message and "thin=16" in result.message


def test_an_unreadable_alignment_is_not_reported_as_a_run(tmp_path):
    """The `engine` field names the engine that ran.  Its own docstring said a bad
    input gives 'none', and three pins had frozen 'builtin' for it — which let the
    report print a converged-looking table around a parse error."""
    junk = alignment(tmp_path, "# comment before sequence\n%%%%\n", "junk.fasta")
    result = _builtin_bayesian(junk)
    assert result.engine == "none"
    assert result.newick_tree == ""
    assert result.num_samples == 0
    assert result.message.startswith("Error:")


def test_two_taxa_names_the_file_and_the_count(tmp_path):
    result = _builtin_bayesian(alignment(tmp_path, FASTA_2, "two.fasta"), n_generations=10)
    assert result.engine == "none"
    assert result.message == (
        "Need ≥3 taxa for Bayesian phylogenetic analysis, "
        f"{alignment(tmp_path, FASTA_2, 'two.fasta')} holds 2")


def test_the_seed_is_what_makes_a_run_reproducible(tmp_path):
    """MCMC is stochastic, so the only honesty about a repeated analysis is the seed:
    the same seed gives the same tree, and a different seed does not."""
    path = alignment(tmp_path, FASTA_LONGER)
    first = _builtin_bayesian(path, n_generations=40, seed=11)
    second = _builtin_bayesian(path, n_generations=40, seed=11)
    third = _builtin_bayesian(path, n_generations=40, seed=12)
    assert first.newick_tree == second.newick_tree
    assert first.newick_tree != third.newick_tree
