"""
Maximum Likelihood phylogenetics with dual-mode execution.

Pure Python neighbor-joining + bootstrap as default, RAxML/IQ-TREE as optional.
"""
import os
import subprocess
import tempfile
import numpy as np
from dataclasses import dataclass, field

try:
    from Bio import AlignIO
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    from Bio.Phylo import draw
    HAS_BIO = True
except ImportError:
    HAS_BIO = False


@dataclass
class PhyloResult:
    engine: str
    newick_tree: str = ""
    support_values: dict = field(default_factory=dict)
    tree_file: str = ""
    log_likelihood: float = 0.0
    message: str = ""


from .utils import has_tool as _has_tool


def check_phylo_tools():
    return {'raxml': _has_tool('raxml-ng') or _has_tool('RAxML'),
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
    import io
    buf = io.StringIO()
    try:
        from Bio.Phylo import NewickIO
        NewickIO.write(tree, buf)
    except Exception:
        buf.write(str(tree))
    return buf.getvalue().strip()


def _bootstrap_support(alignment, n_replicates=100):
    if not HAS_BIO or alignment is None:
        return {}
    support = {}
    from Bio import AlignIO
    import random
    align_len = alignment.get_alignment_length()

    trees_seen = {}
    for rep in range(n_replicates):
        cols = random.sample(range(align_len), align_len)
        boot_records = []
        for record in alignment:
            new_seq = ''.join(record.seq[c] for c in sorted(cols))
            from Bio.SeqRecord import SeqRecord
            from Bio.Seq import Seq
            boot_records.append(SeqRecord(Seq(new_seq), id=record.id))

        from Bio.Align import MultipleSeqAlignment
        boot_align = MultipleSeqAlignment(boot_records)
        tree, _ = _nj_tree_from_alignment(boot_align)
        if tree:
            clades = str(tree)
            trees_seen[clades] = trees_seen.get(clades, 0) + 1

    return {k: v / n_replicates for k, v in trees_seen.items()}


def _builtin_phylogeny(alignment_file, bootstrap=100):
    if not HAS_BIO:
        return PhyloResult(engine='builtin', message="Biopython not installed")

    try:
        alignment = AlignIO.read(alignment_file, 'fasta')
    except Exception as e:
        return PhyloResult(engine='builtin', message=f"Error reading alignment: {e}")

    tree, err = _nj_tree_from_alignment(alignment)
    if err:
        return PhyloResult(engine='builtin', message=err)

    newick = _newick_from_tree(tree)

    support = {}
    if bootstrap > 0:
        support = _bootstrap_support(alignment, n_replicates=bootstrap)

    return PhyloResult(
        engine='builtin',
        newick_tree=newick,
        support_values=support,
        message=f"Neighbor-Joining tree with {bootstrap} bootstrap replicates"
    )


# ── External Tool Wrappers ──────────────────────────────────────────────────

def _raxml_run(alignment_file, output_dir, model='GTRGAMMA', bootstrap=100):
    cmd = ['raxml-ng', '--all', '--msa', alignment_file, '--model', model,
           '--bs-trees', str(bootstrap), '--out-prefix', output_dir + '/tree',
           '--force']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        tree_file = output_dir + '/tree.raxml.bestTree'
        if os.path.exists(tree_file):
            with open(tree_file) as f:
                newick = f.read().strip()
            return PhyloResult(engine='raxml', newick_tree=newick,
                              tree_file=tree_file,
                              message="RAxML maximum likelihood tree")
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _iqtree_run(alignment_file, output_dir, model='GTR+G', bootstrap=100):
    cmd = ['iqtree', '-s', alignment_file, '-m', model, '-bb', str(bootstrap),
           '-nt', '1', '-o', output_dir + '/tree']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        tree_file = output_dir + '/tree.treefile'
        if os.path.exists(tree_file):
            with open(tree_file) as f:
                newick = f.read().strip()
            return PhyloResult(engine='iqtree', newick_tree=newick,
                              tree_file=tree_file,
                              message="IQ-TREE maximum likelihood tree")
    except (OSError, subprocess.SubprocessError):
        pass
    return None


# ── Public API ──────────────────────────────────────────────────────────────

def build_tree(alignment_file, method='auto', model='auto', bootstrap=100):
    if not os.path.exists(alignment_file):
        return PhyloResult(engine='none', message=f"File not found: {alignment_file}")

    tools = check_phylo_tools()

    if method in ('raxml', 'auto') and tools['raxml']:
        out_dir = tempfile.mkdtemp()
        result = _raxml_run(alignment_file, out_dir,
                           model=model if model != 'auto' else 'GTRGAMMA',
                           bootstrap=bootstrap)
        if result:
            return result

    if method in ('iqtree', 'auto') and tools['iqtree']:
        out_dir = tempfile.mkdtemp()
        result = _iqtree_run(alignment_file, out_dir,
                            model=model if model != 'auto' else 'GTR+G',
                            bootstrap=bootstrap)
        if result:
            return result

    return _builtin_phylogeny(alignment_file, bootstrap)


def parse_newick(newick_str):
    if not HAS_BIO:
        return None
    from io import StringIO
    from Bio.Phylo import NewickIO
    try:
        return NewickIO.read(StringIO(newick_str))
    except Exception:
        return None


def format_phylo_report(result):
    lines = [
        "=== Phylogenetic Analysis Report ===",
        f"Engine: {result.engine}",
        f"Log-likelihood: {result.log_likelihood:.2f}" if result.log_likelihood else "",
        f"Tree: {result.newick_tree[:100]}..." if len(result.newick_tree) > 100 else f"Tree: {result.newick_tree}",
    ]
    if result.support_values:
        lines.append(f"Bootstrap clusters: {len(result.support_values)}")
    if result.message:
        lines.append(f"Note: {result.message}")
    return '\n'.join(lines)
