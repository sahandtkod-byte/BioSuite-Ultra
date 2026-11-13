"""
Phylogenetics with dual-mode execution.

The built-in engine is neighbour-joining plus bootstrap — it is **not** maximum
likelihood.  A maximum-likelihood tree exists only when RAxML-NG or IQ-TREE is
installed and actually completes; ``PhyloResult.engine`` reports which of the two
happened, and a silent fall-back to NJ emits a PerformanceWarning.
"""
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field

from .log import get_logger
from .utils import PerformanceWarning, has_tool as _has_tool

logger = get_logger(__name__)

try:
    from Bio import AlignIO
    from Bio import Phylo as _BioPhylo
    from Bio.Phylo import draw  # noqa: F401
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    HAS_BIO = True
except ImportError:
    HAS_BIO = False


# The methods build_tree() will accept.  'builtin' and its long-standing alias
# 'nj' both ask for neighbour-joining directly; 'auto' asks for the best engine
# that is installed.
METHODS = ('auto', 'builtin', 'nj', 'raxml', 'iqtree')


@dataclass
class PhyloResult:
    engine: str
    newick_tree: str = ""
    support_values: dict = field(default_factory=dict)
    # Always empty here: build_tree() deletes the engine's working directory
    # before it returns, so no path out of this module is usable.  A tree file
    # that outlives the call is bayesian_phylogeny's shape, not this one.
    tree_file: str = ""
    # None means "no likelihood was computed", which is also what a
    # neighbour-joining tree means; 0.0 would be a measured value.
    log_likelihood: float | None = None
    message: str = ""


def check_phylo_tools():
    """Which external phylogenetics engines are usable.

    ``raxml`` means RAxML-NG specifically: ``_raxml_run`` emits RAxML-NG flags
    (``--all``, ``--msa``, ``--bs-trees``) that the legacy RAxML-8 binary does not
    accept, so a legacy install must not be reported as a usable ML engine.
    ``raxml_legacy`` is informational only.
    """
    return {'raxml': _has_tool('raxml-ng'),
            'raxml_legacy': _has_tool('RAxML'),
            'iqtree': _has_tool('iqtree')}


# ── Pure Python NJ + Bootstrap ─────────────────────────────────────────────

def _nj_tree_from_alignment(alignment):
    if not HAS_BIO:
        return None, "Biopython not installed"
    calculator = DistanceCalculator('identity')
    dm = calculator.get_distance(alignment)
    constructor = DistanceTreeConstructor(calculator, method='nj')
    tree = constructor.nj(dm)
    return tree, None


def _newick_from_tree(tree):
    """Serialise a Biopython tree to Newick text.

    ``Bio.Phylo.write`` is the supported entry point.  The previous code called the
    module-level ``NewickIO.write``, which raises ``TypeError: 'Tree' object is not
    iterable`` on Biopython 1.87/1.88, and swallowed it into ``str(tree)`` — that
    produced a Python repr which a lenient Newick reader happily parsed into a tree
    of fake taxa such as ``rooted=False``.  Errors now propagate.
    """
    import io
    if not HAS_BIO:
        raise RuntimeError("Bio.Phylo is not installed, so no tree can be serialised")
    buf = io.StringIO()
    _BioPhylo.write(tree, buf, "newick")
    return buf.getvalue().strip()


def _bootstrap_support(alignment, n_replicates=100, seed=None):
    """Bootstrap CLADE support frequencies (0..1 keyed by taxa tuples).

    Columns are resampled WITH replacement; support is counted per internal
    clade split, not per whole-tree string.

    Args:
        alignment: Biopython alignment to resample.
        n_replicates: number of bootstrap replicates, zero or more.
        seed: RNG seed.  Bootstrapping is stochastic, so support values differ
            between runs unless a seed is given; pass one to make an analysis
            reproducible.  ``None`` keeps the historical non-deterministic
            behaviour.

    Returns:
        ``(support, failed)`` — ``failed`` is how many replicates produced no
        tree at all.  Counting it matters because a resampling run that fails
        every time is indistinguishable from a tree with no resolved clades
        otherwise, and the caller has to be able to say which happened.
    """
    if not HAS_BIO or alignment is None:
        return {}, 0
    if not isinstance(n_replicates, int) or isinstance(n_replicates, bool) \
            or n_replicates < 0:
        raise ValueError(
            f"n_replicates must be a whole number of bootstrap replicates, "
            f"got {n_replicates!r}")

    import random
    from collections import Counter

    rng = random.Random(seed)

    from Bio.Align import MultipleSeqAlignment
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    align_len = alignment.get_alignment_length()

    def _clade_splits(tree):
        """Non-trivial splits: frozenset of terminals under each internal clade."""
        splits = set()
        all_terms = set(t.name for t in tree.get_terminals())
        for clade in tree.get_nonterminals():
            names = frozenset(t.name for t in clade.get_terminals())
            if 1 < len(names) < len(all_terms):
                splits.add(names)
        return splits

    clade_counts = Counter()
    failed = 0
    failure_reason = ""
    for _rep in range(n_replicates):
        cols = rng.choices(range(align_len), k=align_len)
        boot_records = [
            SeqRecord(Seq(''.join(record.seq[c] for c in cols)), id=record.id, description="")
            for record in alignment
        ]
        boot_align = MultipleSeqAlignment(boot_records)
        tree, err = _nj_tree_from_alignment(boot_align)
        if tree:
            for split in _clade_splits(tree):
                clade_counts[split] += 1
        else:
            failed += 1
            failure_reason = failure_reason or str(err)

    if failed:
        logger.warning("%d of %d bootstrap replicates produced no tree (%s); "
                       "support values are fractions of all %d replicates",
                       failed, n_replicates, failure_reason, n_replicates)

    support = {tuple(sorted(split)): count / n_replicates
               for split, count in clade_counts.items()}
    return support, failed


def _builtin_phylogeny(alignment_file, bootstrap=100, seed=None):
    if not HAS_BIO:
        return PhyloResult(engine='builtin', message="Biopython not installed")

    try:
        alignment = AlignIO.read(alignment_file, 'fasta')
    except Exception as e:
        logger.warning("Could not read %s as a FASTA alignment: %s", alignment_file, e)
        return PhyloResult(engine='builtin', message=f"Error reading alignment: {e}")

    tree, err = _nj_tree_from_alignment(alignment)
    if err:
        return PhyloResult(engine='builtin', message=err)

    newick = _newick_from_tree(tree)

    support, failed = _bootstrap_support(alignment, n_replicates=bootstrap, seed=seed)
    if bootstrap > 0:
        if support:
            message = f"Neighbor-Joining tree with {bootstrap} bootstrap replicates"
            if failed:
                message += f" ({failed} produced no tree)"
        else:
            message = (f"Neighbor-Joining tree; none of the {bootstrap} bootstrap "
                       f"replicates resolved a clade, so no support values")
    else:
        message = "Neighbor-Joining tree without bootstrap support"

    return PhyloResult(
        engine='builtin',
        newick_tree=newick,
        support_values=support,
        message=message
    )


# ── External Tool Wrappers ──────────────────────────────────────────────────

def _remove_work_dir(path):
    """Delete an engine's working directory, saying so when that failed.

    ``shutil.rmtree(ignore_errors=True)`` left a directory behind in silence on
    every run whose artefacts were still held open, and the leak was invisible.
    """
    try:
        shutil.rmtree(path)
    except OSError as exc:
        logger.warning("Could not remove the temporary analysis directory %s: %s",
                       path, exc)


def _raxml_run(alignment_file, output_dir, model='GTRGAMMA', bootstrap=100):
    prefix = os.path.join(output_dir, 'tree')
    cmd = ['raxml-ng', '--all', '--msa', alignment_file, '--model', model,
           '--bs-trees', str(bootstrap), '--prefix', prefix, '--force']
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.warn(f"raxml-ng could not be executed: {exc}", RuntimeWarning)
        return None
    # RAxML-NG reports usage errors with a non-zero exit status; ignoring it used
    # to turn a failed ML run into a silent NJ result.
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-1:]
        warnings.warn(f"raxml-ng exited with {proc.returncode}: "
                      f"{tail[0] if tail else 'no output'}", RuntimeWarning)
        return None
    tree_file = prefix + '.raxml.bestTree'
    if os.path.exists(tree_file):
        with open(tree_file, encoding='utf-8') as f:
            newick = f.read().strip()
        # The score itself lives in run.log, which this wrapper does not read;
        # leaving log_likelihood as None says "not reported" instead of 0.0's
        # "measured as zero".
        return PhyloResult(engine='raxml', newick_tree=newick,
                          message="RAxML-NG maximum likelihood tree")
    warnings.warn(f"raxml-ng succeeded but {tree_file} was not produced", RuntimeWarning)
    return None


def _iqtree_run(alignment_file, output_dir, model='GTR+G', bootstrap=100):
    # IQ-TREE derives its output prefix from -s and writes <prefix>.treefile next
    # to the alignment; '-o' is the *outgroup* option, so passing it previously
    # both mis-rooted the analysis and left the wrapper looking for a file IQ-TREE
    # never writes.  The alignment is staged in its own subdirectory so the
    # artefacts, and their cleanup, stay ours — and so that staging can never
    # collide with the caller's own file.
    stage_dir = os.path.join(output_dir, "iqtree-stage")
    work_aln = os.path.join(stage_dir, os.path.basename(alignment_file))
    try:
        os.makedirs(stage_dir, exist_ok=True)
        if os.path.abspath(work_aln) != os.path.abspath(alignment_file):
            shutil.copyfile(alignment_file, work_aln)
    except OSError as exc:
        warnings.warn(f"IQ-TREE alignment could not be staged: {exc}", RuntimeWarning)
        return None
    cmd = ['iqtree', '-s', work_aln, '-m', model, '-bb', str(bootstrap), '-nt', '1']
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.warn(f"iqtree could not be executed: {exc}", RuntimeWarning)
        return None
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-1:]
        warnings.warn(f"iqtree exited with {proc.returncode}: "
                      f"{tail[0] if tail else 'no output'}", RuntimeWarning)
        return None
    tree_file = work_aln + '.treefile'
    if os.path.exists(tree_file):
        with open(tree_file, encoding='utf-8') as f:
            newick = f.read().strip()
        return PhyloResult(engine='iqtree', newick_tree=newick,
                          message="IQ-TREE maximum likelihood tree")
    warnings.warn(f"iqtree succeeded but {tree_file} was not produced", RuntimeWarning)
    return None


# ── Public API ──────────────────────────────────────────────────────────────

def build_tree(alignment_file, method='auto', model='auto', bootstrap=100,
               seed=None):
    """Build a phylogenetic tree.

    Args:
        alignment_file: a FASTA alignment; the call fails loudly when it is
            missing.
        method: one of ``METHODS``.  Anything else is a programming error and
            raises, because a misspelled engine used to be answered with a
            neighbour-joining tree and no warning.
        model: passed to the external engine, or ``'auto'`` for its default.
        bootstrap: number of bootstrap replicates for the built-in engine
            (zero or more); external tools manage their own support.
        seed: seed for the bootstrap resampling of the built-in engine; pass a
            value for a reproducible analysis (external tools manage their own
            RNG).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}, got {method!r}")
    if not isinstance(bootstrap, int) or isinstance(bootstrap, bool) or bootstrap < 0:
        raise ValueError(f"bootstrap must be a whole number of replicates, "
                         f"got {bootstrap!r}")

    if not os.path.exists(alignment_file):
        return PhyloResult(engine='none', message=f"File not found: {alignment_file}")

    # A call that asks for neighbour-joining by name never needs to know what is
    # on the PATH, and a PATH scan is not free.
    if method in ('builtin', 'nj'):
        return _builtin_phylogeny(alignment_file, bootstrap, seed=seed)

    tools = check_phylo_tools()

    # A requested ML engine that is absent, or present but failing, must not
    # degrade into neighbour-joining in silence: the caller believes it got ML.
    fell_back = False

    if method in ('raxml', 'auto') and tools['raxml']:
        out_dir = tempfile.mkdtemp()
        try:
            result = _raxml_run(alignment_file, out_dir,
                               model=model if model != 'auto' else 'GTRGAMMA',
                               bootstrap=bootstrap)
        finally:
            # The work directory dies with the call; the tree is returned in
            # newick_tree, so no path out of here is usable (N17).
            _remove_work_dir(out_dir)
        if result:
            return result
        fell_back = True
    elif method == 'raxml':
        fell_back = True

    if method in ('iqtree', 'auto') and tools['iqtree']:
        out_dir = tempfile.mkdtemp()
        try:
            result = _iqtree_run(alignment_file, out_dir,
                                model=model if model != 'auto' else 'GTR+G',
                                bootstrap=bootstrap)
        finally:
            _remove_work_dir(out_dir)
        if result:
            return result
        fell_back = True
    elif method == 'iqtree':
        fell_back = True

    if fell_back:
        warnings.warn(
            f"method={method!r} requested a maximum-likelihood engine but none "
            f"completed; returning the built-in neighbour-joining tree, which is "
            f"NOT maximum likelihood (engine='builtin').",
            PerformanceWarning, stacklevel=2)

    return _builtin_phylogeny(alignment_file, bootstrap, seed=seed)


def parse_newick(newick_str):
    """Read a Newick string into a ``Bio.Phylo`` tree, or return None.

    None covers both "Biopython is not installed" and "this text is not a tree
    the reader accepts".  The reason is logged rather than raised because the
    documented contract of this helper is a tree-or-None, but silently: an
    empty string, a non-string and a multi-tree string used to be three
    indistinguishable Nones.
    """
    if not HAS_BIO:
        logger.warning("parse_newick() cannot read a tree without Biopython")
        return None
    from io import StringIO

    from Bio import Phylo
    try:
        return Phylo.read(StringIO(newick_str), 'newick')
    except Exception as exc:
        shown = newick_str[:60] if isinstance(newick_str, str) else repr(newick_str)
        logger.warning("Could not read a Newick tree from %r: %s", shown, exc)
        return None


def format_phylo_report(result):
    lines = [
        "=== Phylogenetic Analysis Report ===",
        f"Engine: {result.engine}",
    ]
    if result.log_likelihood is not None:
        lines.append(f"Log-likelihood: {result.log_likelihood:.2f}")
    if result.newick_tree:
        if len(result.newick_tree) > 100:
            lines.append(f"Tree: {result.newick_tree[:100]}... "
                         f"({len(result.newick_tree)} characters in total)")
        else:
            lines.append(f"Tree: {result.newick_tree}")
    else:
        lines.append("Tree: (none)")
    if result.support_values:
        lines.append(f"Bootstrap clusters: {len(result.support_values)}")
    if result.message:
        lines.append(f"Note: {result.message}")
    return '\n'.join(lines)
