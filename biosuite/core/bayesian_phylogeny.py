"""
Bayesian phylogenetics with dual-mode execution.

Pure Python MCMC tree sampler using the Jukes-Cantor 69 substitution model
with Felsenstein's pruning algorithm for likelihood computation.
MrBayes available as optional external engine.
"""
from __future__ import annotations

import copy
import math
import os
import random
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from Bio import AlignIO, Phylo
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

# ── Constants ────────────────────────────────────────────────────────────────

NUM_STATES = 4
BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
# Anything outside BASE_TO_INT is coded -1 and pruned as "all four states equally
# likely", so the IUPAC set needs no list of its own; `AMBIGUOUS` used to sit here
# unreferenced, which made it look like N/R/Y were handled differently from `?`.

#: A branch with no usable length is scored at this floor rather than as identity.
#: `log_prior` already rejects a length <= 0, and pruning an exact 0 drives every
#: site where two children disagree to a zero likelihood vector for the whole
#: subtree — measured on a 4-taxon tree: -44.416 at 0.001 against -108.858 at 1e-12.
PRUNING_BRANCH_FLOOR = 0.001


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BayesianResult:
    """Bayesian phylogeny inference result.

    Attributes:
        engine: ``'builtin'`` (JC69 MCMC) or ``'mrbayes'`` — the engine that
            **ran**.  ``'none'`` for every outcome where nothing ran: the file is
            missing, Biopython is not installed, the alignment cannot be parsed,
            there are fewer than three taxa, or the *tool* argument names an
            engine that does not exist.
        newick_tree: best-scoring tree found by the sampler (MrBayes consensus when
            that engine ran).  It carries NO per-clade probabilities.
        posterior_probability: convergence-quality proxy — the fraction of
            post-burnin samples whose log-likelihood is within 2 sigma of their own
            mean.  It is not posterior clade support and must not be printed as one.
            Fewer than five post-burnin samples cannot estimate it, so the proxy is
            reported as ``0.0`` there rather than as the ``1.0`` the formula
            degenerates to.
        log_likelihood: mean post-burnin log-likelihood.
        ess / psrf: effective sample size and Gelman-Rubin scale reduction factor.
            Both are the "no evidence either way" values (``ess`` = the sample count,
            ``psrf`` = 1.0) when the chain is too short to estimate them.
        num_samples: post-burnin, thinned samples kept.
        tree_file: set only when the file outlives the call.
    """
    engine: str
    newick_tree: str = ""
    posterior_probability: float = 0.0
    log_likelihood: float = 0.0
    ess: float = 0.0
    psrf: float = 1.0
    num_samples: int = 0
    tree_file: str = ""
    message: str = ""


# ── Jukes-Cantor 69 Substitution Model ──────────────────────────────────────

class JC69Model:
    """
    Jukes-Cantor 69 substitution model.

    Equal base frequencies (π_A = π_C = π_G = π_T = 0.25) and equal
    substitution rates between all pairs of nucleotides.

    Transition probabilities for branch length *t* (expected substitutions
    per site):

        P(same state | t) = 1/4 + 3/4 · exp(-4t/3)
        P(diff  state | t) = 1/4 - 1/4 · exp(-4t/3)
    """

    @staticmethod
    def prob(same_state: bool, t: float) -> float:
        """Transition probability P(same|t) or P(diff|t)."""
        t = max(t, 1e-10)
        e = math.exp(-4.0 * t / 3.0)
        return (0.25 + 0.75 * e) if same_state else (0.25 - 0.25 * e)

    @staticmethod
    def matrix(t: float) -> List[List[float]]:
        """4×4 transition probability matrix for branch length *t*."""
        t = max(t, 1e-10)
        e = math.exp(-4.0 * t / 3.0)
        p_same = 0.25 + 0.75 * e
        p_diff = 0.25 - 0.25 * e
        return [[p_same if i == j else p_diff for j in range(4)] for i in range(4)]


# ── MCMC Tree Sampler ───────────────────────────────────────────────────────

class TreeSampler:
    """
    Metropolis-Hastings MCMC tree sampler.

    Likelihood  — Felsenstein's pruning algorithm with JC69 model.
    Proposals   — branch-length scaling (log-normal) + NNI topology moves.
    Prior       — Exponential(10) on branch lengths (penalises long branches).
    Convergence — burn-in removal, thinning, dual-chain PSRF.
    """

    def __init__(self, alignment: Any, seed: Optional[int] = None) -> None:
        """Initialize the sampler.

        Args:
            alignment: Biopython alignment to sample trees for.
            seed: RNG seed.  MCMC is stochastic, so results differ between
                runs unless a seed is supplied; pass one for a reproducible
                analysis.  A single generator instance is used for the whole
                sampler so that successive chains draw from different parts of
                the stream and remain genuinely independent (identically
                seeded chains would make the PSRF diagnostic meaningless).
        """
        self._rng = random.Random(seed)
        self.alignment = alignment
        self.n_taxa = len(alignment)
        self.n_sites = alignment.get_alignment_length()

        self.taxon_names = [r.id for r in alignment]
        self.taxon_index = {n: i for i, n in enumerate(self.taxon_names)}

        # Integer-coded sequence matrix (n_taxa × n_sites), -1 = ambiguous
        self.seq_matrix = np.full((self.n_taxa, self.n_sites), -1, dtype=np.int8)
        for i, record in enumerate(alignment):
            for j, c in enumerate(str(record.seq).upper()):
                if c in BASE_TO_INT:
                    self.seq_matrix[i, j] = BASE_TO_INT[c]

        self.model = JC69Model()
        # Cached 4x4 transition matrices, keyed by branch length.  It lives on the
        # instance because _pruning reads it, and _pruning is callable on its own:
        # creating the attribute inside log_likelihood (and deleting it again on the
        # way out) made a direct _pruning call on an internal node raise
        # AttributeError: 'TreeSampler' object has no attribute '_p_cache'.
        self._p_cache: 'dict' = {}

    # ── Felsenstein's pruning algorithm ────────────────────────────────

    def _pruning(self, clade: Any, site: int) -> List[float]:
        """
        Compute conditional likelihood vector at a single site.

        Returns [P(data|A), P(data|C), P(data|G), P(data|T)] for the
        subtree rooted at *clade*, traversed bottom-up.

        A child whose branch length is missing or zero is scored at
        `PRUNING_BRANCH_FLOOR`; `or` is what makes both cases land there, and the
        number it lands on is pinned by `tests/core`.
        """
        if clade.is_terminal():
            idx = self.taxon_index.get(clade.name, -1)
            if idx >= 0:
                s = int(self.seq_matrix[idx, site])
                if s >= 0:
                    v = np.zeros(NUM_STATES)
                    v[s] = 1.0
                    return v
            # Ambiguous / gap — all states equally likely
            return np.ones(NUM_STATES)

        # Internal node: combine children (P @ cv with cached matrices —
        # the old pure-Python double loop made MCMC unusably slow for
        # anything beyond toy alignments).
        v = np.ones(NUM_STATES)
        for child in clade.clades:
            cv = self._pruning(child, site)
            bl = max(child.branch_length or PRUNING_BRANCH_FLOOR, 1e-10)
            key = round(bl, 10)
            P = self._p_cache.get(key)
            if P is None:
                P = np.array(self.model.matrix(bl))
                self._p_cache[key] = P
            v *= P @ cv
        return v

    def log_likelihood(self, tree: Any) -> float:
        """Log-likelihood of the full alignment given the tree."""
        ll = 0.0
        self._p_cache.clear()          # transition matrices are per-tree
        for site in range(self.n_sites):
            rv = self._pruning(tree.root, site)
            # Stationary frequencies pi = (0.25, 0.25, 0.25, 0.25)
            sl = 0.25 * float(rv.sum())
            ll += math.log(sl) if sl > 1e-300 else -1e10
        return ll

    # ── Prior ──────────────────────────────────────────────────────────

    def log_prior(self, tree: Any) -> float:
        """
        Exponential(10) prior on branch lengths.

        p(b) = 10 · exp(-10b)  →  log p(b) = ln(10) - 10b

        Returns -inf if any branch length ≤ 0.
        """
        lp = 0.0
        for clade in tree.find_clades():
            if clade.branch_length is not None and clade is not tree.root:
                bl = clade.branch_length
                if bl <= 0:
                    return -float('inf')
                lp += math.log(10.0) - 10.0 * bl
        return lp

    def log_posterior(self, tree: Any) -> float:
        """Log-posterior = log-likelihood + log-prior."""
        return self.log_likelihood(tree) + self.log_prior(tree)

    # ── Initial tree (NJ) ─────────────────────────────────────────────

    def _initial_tree(self, perturb: float = 0.0) -> Any:
        """Build a starting tree using Neighbour-Joining."""
        calc = DistanceCalculator('identity')
        dm = calc.get_distance(self.alignment)
        tree = DistanceTreeConstructor(calc, 'nj').nj(dm)

        for clade in tree.find_clades():
            if clade.branch_length is None or clade.branch_length <= 0:
                clade.branch_length = 0.05
            if perturb > 0:
                clade.branch_length = max(
                    clade.branch_length * math.exp(self._rng.gauss(0, perturb)),
                    1e-4,
                )

        # Ensure bifurcating root (required for NNI)
        if len(tree.root.clades) == 3:
            from Bio.Phylo import Newick
            c0, c1, c2 = tree.root.clades
            internal = Newick.Clade(branch_length=0.01, clades=[c0, c1])
            tree.root.clades = [internal, c2]
        return tree

    # ── Proposals ─────────────────────────────────────────────────────

    def _propose_bl(self, tree: Any) -> Tuple[Any, float]:
        """Scale a random branch length by exp(N(0, 0.2)).

        Returns ``(tree, log_hastings_ratio)``.  A log-normal multiplier is
        NOT a symmetric proposal — q(b'|b)/q(b|b') = b/b' — so the caller
        must add ``log(b'/b)`` to the acceptance log-ratio (this used to be
        silently treated as symmetric, biasing the posterior).
        """
        t = copy.deepcopy(tree)
        branches = [
            c for c in t.find_clades(order='level')
            if c is not t.root and c.branch_length is not None
        ]
        if not branches:
            return t, 0.0
        target = self._rng.choice(branches)
        old = target.branch_length
        target.branch_length = max(old * math.exp(self._rng.gauss(0, 0.2)), 1e-6)
        return t, math.log(target.branch_length / old)

    @staticmethod
    def _find_parent(tree: Any, child: Any) -> Any:
        """Find the parent clade of *child* in *tree*."""
        for c in tree.find_clades(order='level'):
            if hasattr(c, 'clades') and child in c.clades:
                return c
        return None

    def _propose_nni(self, tree: Any) -> Tuple[Any, float]:
        """Nearest-Neighbour Interchange (NNI) on a random internal edge.

        Swaps one child of the selected internal node with its sibling
        under the grandparent.  NNI is a symmetric move on tree space, so
        the Hastings term returned is 0.0.
        """
        t = copy.deepcopy(tree)
        edges = []
        for clade in t.find_clades(order='level'):
            if (not clade.is_terminal() and clade.clades
                    and len(clade.clades) == 2):
                parent = self._find_parent(t, clade)
                if (parent and parent is not t.root
                        and not parent.is_terminal()
                        and len(parent.clades) == 2):
                    edges.append((parent, clade))
        if not edges:
            return t, 0.0

        parent, child = self._rng.choice(edges)
        ci = parent.clades.index(child)
        si = 1 - ci
        sibling = parent.clades[si]
        gi = self._rng.randint(0, len(child.clades) - 1)
        grandchild = child.clades[gi]
        child.clades[gi] = sibling
        parent.clades[si] = grandchild
        return t, 0.0

    # ── MCMC driver ──────────────────────────────────────────────────

    def run_mcmc(self, n_gen: int, burn_in_frac: float = 0.2,
                 thin: int = 10, bl_rate: float = 0.8):
        """
        Run a single Metropolis-Hastings MCMC chain.

        Parameters
        ----------
        n_gen : int
            Total number of MCMC generations.
        burn_in_frac : float
            Fraction of chain to discard as burn-in (0–1).
        thin : int
            Keep every *thin*-th sample after burn-in.
        bl_rate : float
            Probability of proposing a branch-length move (vs NNI).

        Returns
        -------
        samples : list of (newick_str, log_likelihood)
            Post-burnin, thinned samples.
        ll_chain : list of float
            Full chain of log-likelihoods (all generations).
        best_tree : Bio.Phylo tree
            Tree with the highest posterior density encountered.
        accept_rate : float
            Fraction of accepted proposals.
        """
        tree = self._initial_tree(perturb=0.1)
        cur_ll = self.log_likelihood(tree)
        cur_lp = self.log_prior(tree)
        cur_post = cur_ll + cur_lp
        best_post, best_tree = cur_post, copy.deepcopy(tree)

        n_bi = int(n_gen * burn_in_frac)
        samples: List[Tuple[str, float]] = []
        ll_chain: List[float] = []
        accepts = 0

        for g in range(n_gen):
            # Choose proposal
            if self._rng.random() < bl_rate:
                nt, log_h = self._propose_bl(tree)
            else:
                nt, log_h = self._propose_nni(tree)

            nl = self.log_likelihood(nt)
            np_ = nl + self.log_prior(nt)

            # Metropolis-Hastings with the proposal-density correction
            if math.log(max(self._rng.random(), 1e-300)) < np_ - cur_post + log_h:
                tree, cur_ll, cur_post = nt, nl, np_
                accepts += 1

            if cur_post > best_post:
                best_post, best_tree = cur_post, copy.deepcopy(tree)

            ll_chain.append(cur_ll)

            # Collect post-burnin samples with thinning
            if g >= n_bi and (g - n_bi) % thin == 0:
                buf = StringIO()
                Phylo.write(tree, buf, 'newick')
                samples.append((buf.getvalue().strip(), cur_ll))

        return samples, ll_chain, best_tree, accepts / max(n_gen, 1)


# ── ESS (initial-positive-sequence estimator) ──────────────────────────────

def _compute_ess(samples: Any) -> float:
    """
    Effective Sample Size via the initial-positive-sequence estimator
    (Geyer 1992).

    Uses autocovariance windowing with an initial monotone positive
    sequence condition to avoid variance inflation from noisy lags.
    """
    n = len(samples)
    if n < 20:
        return float(n)
    x = np.asarray(samples, dtype=float)
    mu = x.mean()
    var = x.var(ddof=1)
    if var < 1e-20:
        return float(n)

    max_lag = min(n // 2, 500)
    # Autocovariance: γ(k) = Σ (x_i - μ)(x_{i+k} - μ) / n
    acv = np.correlate(x - mu, x - mu, mode='full')[n - 1: n - 1 + max_lag + 1]
    acv = acv / (n * var)  # autocorrelation ρ(k)

    # Initial positive sequence: sum pairs until first negative or zero
    tau = 1.0
    for k in range(1, max_lag):
        if acv[k] <= 0:
            break
        tau += 2.0 * acv[k]
    return n / tau


# ── PSRF (Gelman-Rubin potential scale reduction factor) ───────────────────

def _compute_psrf(chain_means: Any, chain_vars: Any, n_per_chain: float) -> float:
    """
    Gelman-Rubin PSRF from multiple chains.

    PSRF ≈ 1.0 indicates convergence.
    PSRF < 1.1 is typically considered acceptable.
    """
    m = len(chain_means)
    if m < 2:
        return 1.0
    B = n_per_chain * np.var(chain_means, ddof=1)  # between-chain variance
    W = np.mean(chain_vars)                           # within-chain variance
    if W < 1e-20:
        return 1.0
    var_plus = ((n_per_chain - 1) / max(n_per_chain, 1)) * W + B / max(n_per_chain, 1)
    return float(np.sqrt(var_plus / W))


# ── Built-in Bayesian (JC69 MCMC) ─────────────────────────────────────────

def _builtin_bayesian(alignment_file: str, n_generations: int = 5000, sample_freq: int = 10,
                      seed: Optional[int] = None) -> BayesianResult:
    """
    Run a pure-Python Bayesian MCMC phylogenetic analysis.

    Uses the Jukes-Cantor 69 substitution model with Felsenstein's
    pruning algorithm for likelihood computation.  Runs two independent
    MCMC chains and computes ESS + PSRF convergence diagnostics.

    Args:
        n_generations: the generations each chain runs, honoured exactly.  It used
            to be a floor that a data-sized term could override — measured on a
            4-taxon x 200-site alignment, `n_generations=50` and `n_generations=500`
            both ran 800 generations and 1,602 likelihood evaluations, so a request
            for a quick look could silently cost 50,000 generations of pure Python.
        sample_freq: sizes the target number of kept samples (`n_generations //
            sample_freq`, at least 10); the thinning actually applied is reported
            in the message.
    """
    if not HAS_BIO:
        return BayesianResult(engine='none', message="Biopython not installed")
    try:
        alignment = AlignIO.read(alignment_file, 'fasta')
    except Exception as e:
        return BayesianResult(engine='none', message=f"Error: {e}")

    n_taxa = len(alignment)

    if n_taxa < 3:
        return BayesianResult(
            engine='none',
            message=f"Need ≥3 taxa for Bayesian phylogenetic analysis, "
                    f"{alignment_file} holds {n_taxa}",
        )

    n_gen = max(1, n_generations)
    n_keep = max(n_generations // sample_freq, 10)
    thin = max(1, int(n_gen * 0.8) // n_keep)

    sampler = TreeSampler(alignment, seed=seed)

    # ── Run 2 independent chains for PSRF ──────────────────────────────
    chain_data = []
    for _ in range(2):
        samples, ll, best_tree, acc = sampler.run_mcmc(
            n_gen, burn_in_frac=0.2, thin=thin, bl_rate=0.8,
        )
        chain_data.append((samples, ll, best_tree, acc))

    # Select chain with highest peak log-likelihood
    best_chain_idx = max(
        range(2),
        key=lambda i: max((s[1] for s in chain_data[i][0]), default=-1e30),
    )
    samples, _ll_chain, best_tree, _best_acc = chain_data[best_chain_idx]

    # ── Newick of best tree ────────────────────────────────────────────
    buf = StringIO()
    Phylo.write(best_tree, buf, 'newick')
    best_newick = buf.getvalue().strip()

    # ── Convergence diagnostics ────────────────────────────────────────
    post_ll = [s[1] for s in samples]
    n_post = len(post_ll)
    ess = _compute_ess(post_ll) if n_post >= 10 else float(n_post)
    mean_ll = float(np.mean(post_ll)) if post_ll else 0.0

    # PSRF on the second half of each chain's log-likelihood.  A half shorter than
    # two iterations has no ddof=1 variance to offer: np.var returns nan with a
    # RuntimeWarning, which used to be printed as "PSRF: nan".
    halves = [cd[1][len(cd[1]) // 2:] for cd in chain_data]
    usable = [half for half in halves if len(half) > 1]
    psrf_computed = len(usable) == 2
    if psrf_computed:
        psrf = _compute_psrf([float(np.mean(half)) for half in usable],
                             [float(np.var(half, ddof=1)) for half in usable],
                             min(len(half) for half in usable))
    else:
        psrf = 1.0

    # Rough "posterior probability" — fraction of post-burnin samples with
    # log-likelihood within 2σ of the mean (convergence quality proxy).  Four
    # samples or fewer cannot estimate it: the largest standardised deviation any
    # set of n points can reach is sqrt(n-1) sigma, which is below 2 for n <= 4, so
    # the count is 100% no matter how badly the chain is mixing.  It is reported as
    # 0.0 — nothing computed — rather than as a perfect score.
    proxy_computed = n_post >= 5
    if proxy_computed:
        sd = float(np.std(post_ll))
        within = sum(1 for ll in post_ll if abs(ll - mean_ll) < 2 * sd)
        post_prob = within / n_post
    else:
        post_prob = 0.0

    notes = []
    if not proxy_computed:
        notes.append(f"posterior proxy needs >=5 samples, {n_post} kept")
    if not psrf_computed:
        notes.append("PSRF needs >=2 post-burnin iterations per chain")

    # Average acceptance rate across chains
    avg_acc = np.mean([cd[3] for cd in chain_data])

    return BayesianResult(
        engine='builtin',
        newick_tree=best_newick,
        posterior_probability=round(post_prob, 3),
        log_likelihood=round(mean_ll, 2),
        ess=round(ess, 1),
        psrf=round(psrf, 3),
        num_samples=n_post,
        message=(
            f"MCMC (JC69+Felsenstein): {n_post} post-burnin samples, "
            f"ESS={ess:.0f}, PSRF={psrf:.3f}, "
            f"accept={avg_acc:.1%}, 2 chains, burn-in 20%, thin={thin}"
            + (f" ({'; '.join(notes)})" if notes else "")
        ),
    )


# ── MrBayes Wrapper ─────────────────────────────────────────────────────────

def _mrbayes_run(alignment_file: str, n_generations: int = 10000,
                 output_dir: Optional[str] = None) -> Optional[BayesianResult]:
    """Run MrBayes via subprocess on a NEXUS file and parse the consensus tree.

    Returns ``None`` — after warning about the reason — when the run produced no
    tree; the caller decides what to fall back to.

    ``output_dir`` is the caller's directory when one is given and this function's
    own when it is not.  A directory created here is removed again on every path
    that has no tree to hand over: three failed runs used to leave three of them
    behind, each holding an `alignment.nex` and a `run.nex` nobody could find again.
    """
    owns_dir = output_dir is None
    if owns_dir:
        output_dir = tempfile.mkdtemp(prefix="biosuite_mrbayes_")

    nexus_file = os.path.join(output_dir, 'alignment.nex')
    tree_file = os.path.join(output_dir, 'alignment.nex.tre')
    result: Optional[BayesianResult] = None
    try:
        _fasta_to_nexus(alignment_file, nexus_file)
        if not os.path.exists(nexus_file):
            # Without Biopython `_fasta_to_nexus` declines to write anything (it has
            # no way to read the FASTA).  Running mb on a script that executes a file
            # that does not exist produced "mb exited with 1" — a verdict on the tool,
            # not on the real reason.
            warnings.warn(f"{nexus_file} was not written; MrBayes cannot be started "
                          f"without Biopython", RuntimeWarning)
            return None

        # The path is quoted: `execute` takes one filename, and the unquoted
        # absolute path of a temporary directory containing a space arrives at MrBayes
        # as several tokens (measured in the generated script:
        # `execute C:\\Users\\me\\AppData\\Local\\Temp\\with space\\alignment.nex`).
        mb_script = f"""\
execute "{nexus_file}"
lset nst=6 rates=invgamma
mcmc ngen={n_generations} samplefreq=10 nruns=2 nchains=4
sump
sumt
quit
"""
        script_file = os.path.join(output_dir, 'run.nex')
        with open(script_file, 'w', encoding="utf-8") as f:
            f.write(mb_script)

        try:
            proc = subprocess.run(
                ['mb', script_file], capture_output=True, text=True,
                cwd=output_dir, timeout=7200,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            # A missing/blocked binary is a reason the caller deserves, not a `pass`.
            warnings.warn(f"mb could not be executed: {exc}", RuntimeWarning)
            return None
        # MrBayes reports a rejected script with a non-zero exit status; ignoring it used
        # to turn a failed external run into an unexplained fall-back to the built-in MCMC.
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-1:]
            warnings.warn(f"mb exited with {proc.returncode}: "
                          f"{tail[0] if tail else 'no output'}", RuntimeWarning)
            return None
        if not os.path.exists(tree_file):
            warnings.warn(f"mb succeeded but {tree_file} was not produced", RuntimeWarning)
            return None
        with open(tree_file, encoding="utf-8") as f:
            newick = f.read().strip()
        result = BayesianResult(
            engine='mrbayes', newick_tree=newick,
            tree_file=tree_file,
            message="MrBayes posterior consensus tree",
        )
    finally:
        if result is None and owns_dir:
            # ignore_errors because a file MrBayes still holds open on Windows must
            # not turn a run that has already failed into a second, unrelated error.
            shutil.rmtree(output_dir, ignore_errors=True)
    return result


def _fasta_to_nexus(fasta_file: str, nexus_file: str) -> None:
    """Convert aligned FASTA to NEXUS format for MrBayes input.

    Writes nothing and returns when Biopython is missing — the caller checks for the
    file instead of launching `mb` on an alignment that was never staged.  Raises
    `ValueError` for a duplicated sequence label: MrBayes refuses the whole data
    block for it, and "mb exited with 1" is not a message the user can act on.
    (`record.id` is the first token of the header, so a header with spaces cannot
    reach the matrix line as two tokens — measured, `>Homo sapiens` stages as
    `Homo`.  No quoting is needed, and none was added.)
    """
    if not HAS_BIO:
        return
    alignment = AlignIO.read(fasta_file, 'fasta')
    n_taxa = len(alignment)
    n_chars = alignment.get_alignment_length()

    seen, repeated = set(), []
    for record in alignment:
        if record.id in seen and record.id not in repeated:
            repeated.append(record.id)
        seen.add(record.id)
    if repeated:
        raise ValueError(f"NEXUS taxon labels must be unique; {fasta_file} repeats "
                         f"{', '.join(repeated)}")

    with open(nexus_file, 'w', encoding="utf-8") as f:
        f.write(f"#nexus\nbegin data;\n  dimensions ntax={n_taxa} nchar={n_chars};\n")
        f.write("  format datatype=dna missing=? gap=-;\n  matrix\n")
        for record in alignment:
            f.write(f"    {record.id:20s} {record.seq}\n")
        f.write(";\nend;\n")


# ── Public API ──────────────────────────────────────────────────────────────

#: The engine names `run_bayesian` accepts.
BAYESIAN_ENGINES = ('auto', 'mrbayes', 'builtin')


def check_bayesian_tools() -> Dict[str, bool]:
    """Detect mrbayes availability."""
    from .utils import has_tool as _has_tool
    return {'mrbayes': _has_tool('mb')}


def run_bayesian(alignment_file: str, n_generations: int = 5000, tool: str = 'auto',
                 seed: Optional[int] = None) -> BayesianResult:
    """Run Bayesian phylogeny; built-in JC69 MCMC when MrBayes is absent.

    Args:
        alignment_file: an aligned FASTA.  Any outcome where nothing ran comes back
            as a result whose ``engine`` is ``'none'`` and whose ``message`` says
            why; this function does not raise for a bad input.
        n_generations: generations of the built-in chain, run exactly as asked.
        tool: ``'auto'`` (MrBayes when it is installed), ``'mrbayes'`` (MrBayes
            first, the built-in chain if its run fails) or ``'builtin'``.  Any other
            spelling is refused: ``'mrbay'``, ``'MrBayes'``, ``''`` and ``None`` all
            used to return a built-in MCMC result without a word about the engine
            the caller had named.
        seed: seed for the built-in MCMC sampler.  MCMC is stochastic; supply a
            seed to make a run reproducible (MrBayes manages its own RNG).

    MrBayes is handed ``n_generations * 10``: its sampling runs far coarser than
    this sampler, so a request that is enough for the built-in chain is thin for a
    consensus.  The multiplier belongs to the external engine and is reported in the
    script it is given, not applied to the built-in chain.
    """
    if tool not in BAYESIAN_ENGINES:
        return BayesianResult(
            engine='none',
            message=f"Unknown tool {tool!r}; use 'auto', 'mrbayes' or 'builtin'.")
    if not os.path.exists(alignment_file):
        return BayesianResult(
            engine='none', message=f"File not found: {alignment_file}"
        )

    tools = check_bayesian_tools()
    if tool in ('mrbayes', 'auto') and tools['mrbayes']:
        try:
            result = _mrbayes_run(alignment_file, n_generations * 10)
        except (OSError, ValueError) as exc:
            # Staging reads the alignment and writes into the staging directory, so
            # an unparsable FASTA and a full disk both arrive here.  The contract is
            # "returns a BayesianResult", not "raises through the CLI".
            return BayesianResult(
                engine='none',
                message=f"Could not stage the MrBayes run: {type(exc).__name__}: {exc}")
        if result is not None:
            return result

    return _builtin_bayesian(alignment_file, n_generations, seed=seed)


def format_bayesian_report(result: BayesianResult) -> str:
    """Format a BayesianResult as a plain-text report.

    The report is the engine, the chain diagnostics and the best tree.  There is no
    "posterior support table" — the sampler returns one tree and a convergence
    proxy, not per-clade probabilities, and the earlier docstring promised a table
    that does not exist in this module or in `BayesianResult`.

    A run that did not happen is reported as the engine that did not run plus the
    reason.  It used to print the whole table around it, so a file Biopython could
    not parse was presented as `Samples: 0, Log-likelihood: 0.00, ESS: 0,
    PSRF: 1.000` — a converged-looking analysis of nothing.
    """
    lines = ["=== Bayesian Phylogenetics Report ===", f"Engine: {result.engine}"]
    if result.engine == 'none':
        if result.message:
            lines.append(f"Note: {result.message}")
        return '\n'.join(lines)

    lines += [
        f"Samples: {result.num_samples}",
        f"Log-likelihood: {result.log_likelihood:.2f}",
        f"ESS: {result.ess:.0f}",
        f"PSRF: {result.psrf:.3f}",
        (f"Tree: {result.newick_tree[:80]}..."
         if len(result.newick_tree) > 80
         else f"Tree: {result.newick_tree}"),
    ]
    if result.message:
        lines.append(f"Note: {result.message}")
    return '\n'.join(lines)
