"""
Bayesian phylogenetics with dual-mode execution.

Pure Python MCMC tree sampler using the Jukes-Cantor 69 substitution model
with Felsenstein's pruning algorithm for likelihood computation.
MrBayes available as optional external engine.
"""
import copy
import math
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from io import StringIO
from typing import List, Tuple

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
AMBIGUOUS = {'N', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V', '-', '?'}


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BayesianResult:
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

    def __init__(self, alignment):
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

    # ── Felsenstein's pruning algorithm ────────────────────────────────

    def _pruning(self, clade, site: int) -> List[float]:
        """
        Compute conditional likelihood vector at a single site.

        Returns [P(data|A), P(data|C), P(data|G), P(data|T)] for the
        subtree rooted at *clade*, traversed bottom-up.
        """
        if clade.is_terminal():
            idx = self.taxon_index.get(clade.name, -1)
            if idx >= 0:
                s = int(self.seq_matrix[idx, site])
                if s >= 0:
                    v = [0.0] * NUM_STATES
                    v[s] = 1.0
                    return v
            # Ambiguous / gap — all states equally likely
            return [1.0] * NUM_STATES

        # Internal node: combine children
        v = [1.0] * NUM_STATES
        for child in clade.clades:
            cv = self._pruning(child, site)
            bl = max(child.branch_length or 0.001, 1e-10)

            # Matrix–vector product: transition probs × child likelihood
            nv = [0.0] * NUM_STATES
            for parent_state in range(NUM_STATES):
                for child_state in range(NUM_STATES):
                    nv[parent_state] += (
                        self.model.prob(parent_state == child_state, bl)
                        * cv[child_state]
                    )
            # Multiply across children (conditionally independent)
            for s in range(NUM_STATES):
                v[s] *= nv[s]
        return v

    def log_likelihood(self, tree) -> float:
        """Log-likelihood of the full alignment given the tree."""
        ll = 0.0
        for site in range(self.n_sites):
            rv = self._pruning(tree.root, site)
            # Stationary frequencies π = (0.25, 0.25, 0.25, 0.25)
            sl = 0.25 * sum(rv)
            ll += math.log(sl) if sl > 1e-300 else -1e10
        return ll

    # ── Prior ──────────────────────────────────────────────────────────

    def log_prior(self, tree) -> float:
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

    def log_posterior(self, tree) -> float:
        """Log-posterior = log-likelihood + log-prior."""
        return self.log_likelihood(tree) + self.log_prior(tree)

    # ── Initial tree (NJ) ─────────────────────────────────────────────

    def _initial_tree(self, perturb: float = 0.0):
        """Build a starting tree using Neighbour-Joining."""
        calc = DistanceCalculator('identity')
        dm = calc.get_distance(self.alignment)
        tree = DistanceTreeConstructor(calc, 'nj').nj(dm)

        for clade in tree.find_clades():
            if clade.branch_length is None or clade.branch_length <= 0:
                clade.branch_length = 0.05
            if perturb > 0:
                clade.branch_length = max(
                    clade.branch_length * math.exp(random.gauss(0, perturb)),
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

    def _propose_bl(self, tree):
        """Scale a random branch length by exp(N(0, 0.2))."""
        t = copy.deepcopy(tree)
        branches = [
            c for c in t.find_clades(order='level')
            if c is not t.root and c.branch_length is not None
        ]
        if not branches:
            return t
        target = random.choice(branches)
        target.branch_length = max(
            target.branch_length * math.exp(random.gauss(0, 0.2)), 1e-6
        )
        return t

    @staticmethod
    def _find_parent(tree, child):
        """Find the parent clade of *child* in *tree*."""
        for c in tree.find_clades(order='level'):
            if hasattr(c, 'clades') and child in c.clades:
                return c
        return None

    def _propose_nni(self, tree):
        """
        Nearest-Neighbour Interchange (NNI) on a random internal edge.

        Swaps one child of the selected internal node with its sibling
        under the grandparent.
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
            return t

        parent, child = random.choice(edges)
        ci = parent.clades.index(child)
        si = 1 - ci
        sibling = parent.clades[si]
        gi = random.randint(0, len(child.clades) - 1)
        grandchild = child.clades[gi]
        child.clades[gi] = sibling
        parent.clades[si] = grandchild
        return t

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
            if random.random() < bl_rate:
                nt = self._propose_bl(tree)
            else:
                nt = self._propose_nni(tree)

            nl = self.log_likelihood(nt)
            np_ = nl + self.log_prior(nt)

            # Metropolis-Hastings (symmetric proposals → log ratio only)
            if math.log(max(random.random(), 1e-300)) < np_ - cur_post:
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

        return samples, ll_chain, best_tree, accepts / n_gen


# ── ESS (initial-positive-sequence estimator) ──────────────────────────────

def _compute_ess(samples) -> float:
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

def _compute_psrf(chain_means, chain_vars, n_per_chain) -> float:
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
    var_plus = ((n_per_chain - 1) / n_per_chain) * W + B / n_per_chain
    return float(np.sqrt(var_plus / W))


# ── Built-in Bayesian (JC69 MCMC) ─────────────────────────────────────────

def _builtin_bayesian(alignment_file, n_generations=5000, sample_freq=10):
    """
    Run a pure-Python Bayesian MCMC phylogenetic analysis.

    Uses the Jukes-Cantor 69 substitution model with Felsenstein's
    pruning algorithm for likelihood computation.  Runs two independent
    MCMC chains and computes ESS + PSRF convergence diagnostics.
    """
    if not HAS_BIO:
        return BayesianResult(engine='builtin', message="Biopython not installed")
    try:
        alignment = AlignIO.read(alignment_file, 'fasta')
    except Exception as e:
        return BayesianResult(engine='builtin', message=f"Error: {e}")

    n_taxa = len(alignment)
    n_sites = alignment.get_alignment_length()

    if n_taxa < 3:
        return BayesianResult(
            engine='builtin',
            message="Need ≥3 taxa for Bayesian phylogenetic analysis",
        )

    # Scale chain length to data size, with a reasonable cap
    n_gen = max(n_generations, min(n_taxa * n_sites, 50000))
    n_keep = max(n_generations // sample_freq, 10)
    thin = max(1, int(n_gen * 0.8) // n_keep)

    sampler = TreeSampler(alignment)

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
    samples, ll_chain, best_tree, best_acc = chain_data[best_chain_idx]

    # ── Newick of best tree ────────────────────────────────────────────
    buf = StringIO()
    Phylo.write(best_tree, buf, 'newick')
    best_newick = buf.getvalue().strip()

    # ── Convergence diagnostics ────────────────────────────────────────
    post_ll = [s[1] for s in samples] if samples else []
    n_post = len(post_ll)

    ess = _compute_ess(post_ll) if n_post >= 10 else float(n_post)
    mean_ll = float(np.mean(post_ll)) if post_ll else 0.0

    # PSRF on second half of each chain's log-likelihood
    half_lens = [len(cd[1]) // 2 for cd in chain_data]
    ll_halves = [
        cd[1][len(cd[1]) // 2:] for cd in chain_data
    ]
    chain_means = [float(np.mean(ll)) for ll in ll_halves if ll]
    chain_vars = [float(np.var(ll, ddof=1)) for ll in ll_halves if ll]
    n_per = min((len(ll) for ll in ll_halves if ll), default=0)
    psrf = _compute_psrf(chain_means, chain_vars, n_per) if n_per > 1 else 1.0

    # Rough "posterior probability" — fraction of post-burnin samples with
    # log-likelihood within 2σ of the mean (convergence quality proxy)
    if n_post >= 5 and len(post_ll) > 1:
        sd = float(np.std(post_ll))
        within = sum(1 for ll in post_ll if abs(ll - mean_ll) < 2 * sd)
        post_prob = within / n_post
    else:
        post_prob = 1.0

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
        ),
    )


# ── MrBayes Wrapper ─────────────────────────────────────────────────────────

def _mrbayes_run(alignment_file, n_generations=10000, output_dir=None):
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    nexus_file = os.path.join(output_dir, 'alignment.nex')
    _fasta_to_nexus(alignment_file, nexus_file)

    mb_script = f"""\
execute {nexus_file}
lset nst=6 rates=invgamma
mcmc ngen={n_generations} samplefreq=10 nruns=2 nchains=4
sump
sumt
quit
"""
    script_file = os.path.join(output_dir, 'run.nex')
    with open(script_file, 'w') as f:
        f.write(mb_script)

    try:
        r = subprocess.run(
            ['mb', script_file], capture_output=True, text=True,
            cwd=output_dir, timeout=7200,
        )
        tree_file = os.path.join(output_dir, 'alignment.nex.tre')
        if os.path.exists(tree_file):
            with open(tree_file) as f:
                newick = f.read().strip()
            return BayesianResult(
                engine='mrbayes', newick_tree=newick,
                tree_file=tree_file,
                message="MrBayes posterior consensus tree",
            )
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _fasta_to_nexus(fasta_file, nexus_file):
    if not HAS_BIO:
        return
    alignment = AlignIO.read(fasta_file, 'fasta')
    n_taxa = len(alignment)
    n_chars = alignment.get_alignment_length()
    with open(nexus_file, 'w') as f:
        f.write(f"#nexus\nbegin data;\n  dimensions ntax={n_taxa} nchar={n_chars};\n")
        f.write("  format datatype=dna missing=? gap=-;\n  matrix\n")
        for record in alignment:
            f.write(f"    {record.id:20s} {record.seq}\n")
        f.write(";\nend;\n")


# ── Public API ──────────────────────────────────────────────────────────────

def check_bayesian_tools():
    from .utils import has_tool as _has_tool
    return {'mrbayes': _has_tool('mb')}


def run_bayesian(alignment_file, n_generations=5000, tool='auto'):
    if not os.path.exists(alignment_file):
        return BayesianResult(
            engine='none', message=f"File not found: {alignment_file}"
        )

    tools = check_bayesian_tools()
    if tool in ('mrbayes', 'auto') and tools['mrbayes']:
        result = _mrbayes_run(alignment_file, n_generations * 10)
        if result:
            return result

    return _builtin_bayesian(alignment_file, n_generations)


def format_bayesian_report(result):
    lines = [
        "=== Bayesian Phylogenetics Report ===",
        f"Engine: {result.engine}",
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
