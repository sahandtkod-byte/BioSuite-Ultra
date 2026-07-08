"""
Genome assembly with dual-mode execution.

Pure Python overlap-graph assembler and assembly statistics as default,
SPAdes/MEGAHIT as optional faster tools.

The built-in assembler constructs an overlap graph from reads, merges
overlapping reads into contigs via a greedy traversal, and then
detects unitigs (maximal non-branching paths).  This replaces the
original simple greedy assembler with a proper overlap-based approach.
"""
import os
import subprocess
import tempfile
import numpy as np
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class AssemblyResult:
    engine: str
    num_contigs: int = 0
    total_length: int = 0
    n50: int = 0
    l50: int = 0
    gc_content: float = 0.0
    max_contig: int = 0
    min_contig: int = 0
    contig_lengths: list = field(default_factory=list)
    output_fasta: str = ""
    message: str = ""


from .utils import has_tool as _has_tool


def check_assembly_tools():
    return {'spades': _has_tool('spades.py') or _has_tool('spades'),
            'megahit': _has_tool('megahit-core')}


def _compute_assembly_stats(contigs):
    lengths = sorted([len(c) for c in contigs], reverse=True)
    if not lengths:
        return AssemblyResult(engine='builtin', message="No contigs")

    total = sum(lengths)
    n50 = _compute_n50(lengths)
    l50 = _compute_l50(lengths)
    gc = sum(c.count('G') + c.count('C') for c in contigs) / max(total, 1) * 100

    return AssemblyResult(
        engine='builtin',
        num_contigs=len(contigs),
        total_length=total,
        n50=n50,
        l50=l50,
        gc_content=gc,
        max_contig=lengths[0],
        min_contig=lengths[-1],
        contig_lengths=lengths
    )


def _compute_n50(lengths):
    total = sum(lengths)
    cumsum = 0
    for l in lengths:
        cumsum += l
        if cumsum >= total / 2:
            return l
    return 0


def _compute_l50(lengths):
    total = sum(lengths)
    cumsum = 0
    for i, l in enumerate(lengths):
        cumsum += l
        if cumsum >= total / 2:
            return i + 1
    return 0


def _builtin_assembly(reads_file, output_fasta=None):
    """Run the built-in overlap-graph assembler.

    Reads are loaded, an overlap graph is constructed, a greedy
    traversal merges reads into contigs, and unitig detection is
    applied to produce the final contig set.
    """
    if output_fasta is None:
        output_fasta = tempfile.mktemp(suffix='.fasta')

    reads = _load_reads_fasta(reads_file)
    if not reads:
        return _compute_assembly_stats([])

    # Build overlap graph
    graph = _build_overlap_graph(reads, min_overlap=15)

    # Merge overlapping reads into contigs
    contigs = _greedy_traversal_assembly(reads, graph)

    # Detect unitigs (maximal non-branching paths)
    contigs = _detect_unitigs(reads, contigs, graph)

    stats = _compute_assembly_stats(contigs)
    stats.output_fasta = output_fasta
    stats.message = (
        f"Built-in overlap-graph assembly: {stats.num_contigs} contigs, "
        f"N50={stats.n50}"
    )

    # Write FASTA
    with open(output_fasta, 'w') as f:
        for i, c in enumerate(contigs):
            f.write(f">contig_{i+1} len={len(c)}\n")
            for j in range(0, len(c), 80):
                f.write(c[j:j+80] + '\n')

    return stats


def _load_reads_fasta(reads_file):
    """Load reads from a FASTA or FASTQ file.

    Auto-detects format by checking the first character (@ = FASTQ,
    > = FASTA).

    Returns:
        List of (header, sequence) tuples.
    """
    reads = []
    with open(reads_file) as f:
        first_line = f.readline()
        if not first_line:
            return reads
        f.seek(0)

        if first_line.startswith('@'):
            # FASTQ: 4 lines per record (header, seq, +, quality)
            while True:
                header = f.readline()
                if not header:
                    break
                seq = f.readline().strip()
                f.readline()  # + line
                f.readline()  # quality line
                if seq:
                    reads.append((header.strip()[1:], seq))
        else:
            # FASTA
            header = None
            buf = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if header is not None and buf:
                        reads.append((header, ''.join(buf)))
                    header = line[1:]
                    buf = []
                elif line:
                    buf.append(line)
            if header is not None and buf:
                reads.append((header, ''.join(buf)))
    return reads


def _compute_suffix_prefix_overlap(s1, s2, min_overlap=15):
    """Compute the longest suffix of s1 that is a prefix of s2.

    Args:
        s1: First sequence.
        s2: Second sequence.
        min_overlap: Minimum overlap length to consider.

    Returns:
        Length of the suffix-prefix overlap, or 0 if below min_overlap.
    """
    max_k = min(len(s1), len(s2))
    # Start from the largest possible overlap and work down
    for k in range(max_k, min_overlap - 1, -1):
        if s1[-k:] == s2[:k]:
            return k
    return 0


def _build_overlap_graph(reads, min_overlap=15):
    """Build an overlap graph from a list of reads.

    Each node is a read index.  A directed edge i -> j means read i's
    suffix overlaps read j's prefix by at least ``min_overlap`` bases.

    Args:
        reads: List of (header, sequence) tuples.
        min_overlap: Minimum overlap length required for an edge.

    Returns:
        Dict mapping read_index -> list of (target_index, overlap_len)
        for the best (longest) overlap to each target.
    """
    n = len(reads)
    graph = defaultdict(list)
    seqs = [seq for _, seq in reads]

    # For efficiency, group reads by their last k-mer (k=min_overlap)
    # to reduce the number of pairwise comparisons
    suffix_index = defaultdict(list)
    for i, seq in enumerate(seqs):
        if len(seq) >= min_overlap:
            kmer = seq[-min_overlap:]
            suffix_index[kmer].append(i)

    prefix_index = defaultdict(list)
    for i, seq in enumerate(seqs):
        if len(seq) >= min_overlap:
            kmer = seq[:min_overlap]
            prefix_index[kmer].append(i)

    # Find overlaps using k-mer index for candidate filtering
    best_overlaps = {}  # (i,j) -> overlap_len
    for i, seq_i in enumerate(seqs):
        suffix_kmer = seq_i[-min_overlap:] if len(seq_i) >= min_overlap else seq_i
        candidates = set()
        # Check suffix k-mer matches
        if suffix_kmer in prefix_index:
            candidates.update(prefix_index[suffix_kmer])
        # Also do a broader check: look at last min_overlap bases
        for plen in range(min_overlap, min(len(seq_i), 2 * min_overlap)):
            suffix = seq_i[-plen:]
            for j, seq_j in enumerate(seqs):
                if i == j:
                    continue
                if seq_j.startswith(suffix):
                    candidates.add(j)

        for j in candidates:
            if i == j:
                continue
            olap = _compute_suffix_prefix_overlap(
                seqs[i], seqs[j], min_overlap
            )
            if olap > 0:
                key = (i, j)
                if key not in best_overlaps or olap > best_overlaps[key]:
                    best_overlaps[key] = olap

    # Build adjacency list (best overlap per target)
    best_for_target = {}
    for (i, j), olap in best_overlaps.items():
        if j not in best_for_target or olap > best_for_target[j][1]:
            best_for_target[j] = (i, olap)

    for (i, j), olap in best_overlaps.items():
        # Keep only the best overlap for each source -> target
        graph[i].append((j, olap))

    # Trim to best overlap per target
    trimmed = defaultdict(list)
    for i, targets in graph.items():
        best = {}
        for j, olap in targets:
            if j not in best or olap > best[j]:
                best[j] = olap
        for j, olap in best.items():
            trimmed[i].append((j, olap))

    return dict(trimmed)


def _greedy_traversal_assembly(reads, graph, min_overlap=15):
    """Merge reads into contigs by greedy graph traversal.

    Starting from the longest unused read, follow the overlap graph
    greedily to extend the contig.  Reads are consumed as they are
    placed.

    Args:
        reads: List of (header, sequence) tuples.
        graph: Overlap graph from _build_overlap_graph.
        min_overlap: Minimum overlap for merging.

    Returns:
        List of assembled contig sequences (strings).
    """
    n = len(reads)
    used = [False] * n
    seqs = [seq for _, seq in reads]

    # Sort read indices by sequence length (longest first)
    order = sorted(range(n), key=lambda i: len(seqs[i]), reverse=True)

    contigs = []
    for start in order:
        if used[start]:
            continue
        used[start] = True
        current = seqs[start]

        # Extend forward
        changed = True
        while changed:
            changed = False
            best_j, best_olap = -1, 0
            for j, olap in graph.get(start, []):
                if not used[j] and olap > best_olap:
                    best_j, best_olap = j, olap
            if best_j >= 0:
                # Merge: append non-overlapping portion
                current += seqs[best_j][best_olap:]
                used[best_j] = True
                start = best_j
                changed = True

        contigs.append(current)

    return contigs


def _detect_unitigs(reads, contigs, graph, min_overlap=15):
    """Detect unitigs from the overlap graph.

    A unitig is a maximal non-branching path in the overlap graph.
    Here we refine contigs by checking for unambiguous extensions at
    each end — if a contig's terminal read has exactly one outgoing
    edge with sufficient overlap, we extend; otherwise we stop.

    Args:
        reads: List of (header, sequence) tuples.
        contigs: Current contig list from _greedy_traversal_assembly.
        graph: Overlap graph.
        min_overlap: Minimum overlap.

    Returns:
        Refined list of contig sequences.
    """
    if not graph:
        return contigs

    # Build reverse graph for backward extension
    rev_graph = defaultdict(list)
    for i, targets in graph.items():
        for j, olap in targets:
            rev_graph[j].append((i, olap))

    refined = []
    seqs = [seq for _, seq in reads]

    # Map each contig back to its constituent reads (approximate)
    # by finding which reads were merged — we track via read-set overlap
    for contig in contigs:
        # Try to find the last read in this contig by checking
        # which read's sequence appears at the end
        tail_read = -1
        for i, seq in enumerate(seqs):
            if contig.endswith(seq) and len(seq) > min_overlap:
                tail_read = i
                break

        if tail_read >= 0:
            # Check for unambiguous forward extension (unitig property)
            fwd_targets = graph.get(tail_read, [])
            if len(fwd_targets) == 1:
                j, olap = fwd_targets[0]
                if not used_in_any(contigs, seqs[j], seqs, min_overlap):
                    contig = contig + seqs[j][olap:]

        refined.append(contig)

    return refined


def used_in_any(contigs, seq, seqs, min_overlap):
    """Check if a sequence is already represented in any contig."""
    for c in contigs:
        if len(seq) >= min_overlap and seq[:min_overlap] in c:
            return True
        if len(seq) >= min_overlap and seq[-min_overlap:] in c:
            return True
    return False


# ── External Tool Wrappers ──────────────────────────────────────────────────

def _spades_assemble(reads_file, output_dir, threads=4):
    cmd = ['spades.py', '-s', reads_file, '-o', output_dir,
           '--only-assembler', '-t', str(threads)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        contig_file = os.path.join(output_dir, 'contigs.fasta')
        if os.path.exists(contig_file):
            return contig_file
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _megahit_assemble(reads_file, output_dir, threads=4):
    cmd = ['megahit-core', '-1', reads_file, '-o', output_dir,
           '--min-contig-len', '200', '-t', str(threads)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        contig_file = os.path.join(output_dir, 'final.contigs.fa')
        if os.path.exists(contig_file):
            return contig_file
    except (OSError, subprocess.SubprocessError):
        pass
    return None


# ── Public API ──────────────────────────────────────────────────────────────

def assemble(reads_file, output_fasta=None, tool='auto', threads=4):
    if not os.path.exists(reads_file):
        return AssemblyResult(engine='none', message=f"Reads file not found: {reads_file}")

    tools = check_assembly_tools()

    if tool in ('spades', 'auto') and tools['spades']:
        out_dir = tempfile.mkdtemp()
        contig_file = _spades_assemble(reads_file, out_dir, threads)
        if contig_file:
            return _parse_assembly_fasta(contig_file, 'spades')

    if tool in ('megahit', 'auto') and tools['megahit']:
        out_dir = tempfile.mkdtemp()
        contig_file = _megahit_assemble(reads_file, out_dir, threads)
        if contig_file:
            return _parse_assembly_fasta(contig_file, 'megahit')

    if output_fasta is None:
        output_fasta = tempfile.mktemp(suffix='.fasta')
    return _builtin_assembly(reads_file, output_fasta)


def _parse_assembly_fasta(fasta_file, engine):
    contigs = []
    name, buf = None, []
    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name:
                    contigs.append(''.join(buf))
                name = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line)
    if name:
        contigs.append(''.join(buf))

    stats = _compute_assembly_stats(contigs)
    stats.engine = engine
    stats.output_fasta = fasta_file
    stats.message = f"{engine}: {stats.num_contigs} contigs, N50={stats.n50}"
    return stats


def format_assembly_report(result):
    lines = [
        "=== Genome Assembly Report ===",
        f"Engine: {result.engine}",
        f"Contigs: {result.num_contigs}",
        f"Total length: {result.total_length:,} bp",
        f"N50: {result.n50:,} bp",
        f"L50: {result.l50}",
        f"GC content: {result.gc_content:.1f}%",
        f"Longest contig: {result.max_contig:,} bp",
    ]
    if result.message:
        lines.append(f"Note: {result.message}")
    return '\n'.join(lines)
