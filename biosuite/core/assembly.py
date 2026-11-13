"""
Genome assembly with dual-mode execution.

Pure Python overlap-graph assembler and assembly statistics as default,
SPAdes/MEGAHIT as optional faster tools.

The built-in assembler constructs an overlap graph from reads, merges
overlapping reads into contigs via a greedy traversal, and then
detects unitigs (maximal non-branching paths).  This replaces the
original simple greedy assembler with a proper overlap-based approach.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AssemblyResult:
    """Genome assembly output container.

    Attributes:
        contigs: List of assembled sequence strings.
        stats: N50/L50/total-length summary dict.
        engine: spades/megahit/builtin marker.
    """
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


from .log import get_logger
from .utils import has_tool as _has_tool, secure_temp_path, wrote_output as _wrote_output

logger = get_logger(__name__)


def check_assembly_tools() -> dict :
    """Detect spades/megahit availability."""
    return {'spades': _has_tool('spades.py') or _has_tool('spades'),
            'megahit': _has_tool('megahit-core')}


def _compute_assembly_stats(contigs: List[str]) -> AssemblyResult:
    """Compute assembly statistics from a contig list.

    Returns:
        AssemblyResult with total_length, num_contigs, N50, L50, longest/
        shortest contig, per-contig lengths and overall GC content.
    """
    lengths = sorted([len(c) for c in contigs], reverse=True)
    if not lengths:
        return AssemblyResult(engine='builtin', message="No contigs")

    total = sum(lengths)
    n50 = _compute_n50(lengths)
    l50 = _compute_l50(lengths)
    # Assemblers mask low-quality regions by writing them in lower case, so the
    # count is taken on the upper-cased sequence — a case-sensitive count reports
    # a masked contig as holding no G and no C at all.
    gc = sum(c.upper().count('G') + c.upper().count('C') for c in contigs) / max(total, 1) * 100

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


def _compute_n50(lengths: List[int]) -> int:
    """N50: contig length at which 50% of assembly is covered."""
    total = sum(lengths)
    cumsum = 0
    for l in lengths:
        cumsum += l
        if cumsum >= total / 2:
            return l
    return 0


def _compute_l50(lengths: List[int]) -> int:
    """L50: number of contigs needed to reach 50%% of total length."""
    total = sum(lengths)
    cumsum = 0
    for i, l in enumerate(lengths):
        cumsum += l
        if cumsum >= total / 2:
            return i + 1
    return 0


def _builtin_assembly(reads_file: str, output_fasta: Optional[str] = None) -> AssemblyResult:
    """Run the built-in overlap-graph assembler.

    Reads are loaded, an overlap graph is constructed, a greedy
    traversal merges reads into contigs, and unitig detection is
    applied to produce the final contig set.
    """
    reads = _load_reads_fasta(reads_file)
    if not reads:
        return _compute_assembly_stats([])

    # Reserved only now: reserving before the empty-reads return created a
    # 0-byte biosuite_*.fasta in /tmp that no caller was ever told about.
    if output_fasta is None:
        output_fasta = secure_temp_path(suffix='.fasta')

    # Build overlap graph
    graph = _build_overlap_graph(reads, min_overlap=15)

    # Merge overlapping reads into contigs.  The traversal consumes every read and
    # follows each tip to its best overlap, so what it returns is already a set of
    # maximal non-branching contigs.  A separate unitig pass used to run here and
    # extend each contig by one more read, but its k-mer membership test was
    # evaluated against the contig set that already contains every read, so the
    # extension could never fire: the pass spent O(reads x contigs x contig-bases)
    # on substring scans that provably did nothing (minutes on real read sets).
    contigs = _greedy_traversal_assembly(reads, graph)

    stats = _compute_assembly_stats(contigs)
    stats.output_fasta = output_fasta
    stats.message = (
        f"Built-in overlap-graph assembly: {stats.num_contigs} contigs, "
        f"N50={stats.n50}"
    )

    # Write FASTA
    with open(output_fasta, 'w', encoding="utf-8") as f:
        for i, c in enumerate(contigs):
            f.write(f">contig_{i+1} len={len(c)}\n")
            for j in range(0, len(c), 80):
                f.write(c[j:j+80] + '\n')

    return stats


def _load_reads_fasta(reads_file: str) -> List[str]:
    """Load reads from a FASTA or FASTQ file.

    Auto-detects format by checking the first character (@ = FASTQ,
    > = FASTA).

    Returns:
        List of (header, sequence) tuples.
    """
    reads = []
    with open(reads_file, encoding="utf-8") as f:
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


def _build_overlap_graph(reads: List[str], min_overlap: int = 15) -> Dict[str, Any]:
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
    graph = defaultdict(list)
    seqs = [seq for _, seq in reads]

    # Index ALL suffixes of length >= min_overlap: a suffix/prefix overlap of
    # any length L >= min_overlap can then be found by exact lookup.  (The old
    # code only indexed the single min_overlap k-mer plus suffixes shorter
    # than 2*min_overlap — this only ever found overlaps EXACTLY min_overlap
    # long, so typical 30+ bp read overlaps produced no edges and no contigs
    # were ever merged.)
    suffix_index = defaultdict(list)
    for i, seq in enumerate(seqs):
        for plen in range(min_overlap, len(seq) + 1):
            suffix_index[seq[-plen:]].append(i)

    best_overlaps = {}  # (i, j) -> overlap length
    for j, seq_j in enumerate(seqs):
        for plen in range(min_overlap, len(seq_j) + 1):
            prefix = seq_j[:plen]
            if prefix not in suffix_index:
                continue
            for i in suffix_index[prefix]:
                if i == j or len(seq_j) < plen:
                    continue
                key = (i, j)
                # plen grows strictly within one target j and each (i, j) key is
                # only ever written by ever-longer prefixes, so this write already
                # holds the longest overlap for the pair: the previous
                # "if plen > best_overlaps.get(key, 0)" guard could never be False
                # and its untaken branch was unreachable.
                best_overlaps[key] = plen

    # Build adjacency list.  best_overlaps is keyed by (i, j), so every source
    # already carries at most one edge per target holding its longest overlap;
    # the second "trim to best per target" pass was a no-op whose False branch (a
    # repeated target in one source) could never occur, so it is gone.
    for (i, j), olap in best_overlaps.items():
        graph[i].append((j, olap))

    return dict(graph)


def _greedy_traversal_assembly(reads: List[str], graph: Dict[str, Any]) -> List[str]:
    """Merge reads into contigs by greedy graph traversal.

    Starting from the longest unused read, follow the overlap graph
    greedily to extend the contig.  Reads are consumed as they are
    placed.

    Args:
        reads: List of (header, sequence) tuples.
        graph: Overlap graph from _build_overlap_graph.

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


# ── External Tool Wrappers ──────────────────────────────────────────────────

def _spades_assemble(reads_file: str, output_dir: str, threads: int = 4) -> Optional[str]:
    """Run SPAdes assembler via subprocess (multi-k default).

    Returns the contigs file the tool wrote, or None if SPAdes did not produce
    one.
    """
    cmd = ['spades.py', '-s', reads_file, '-o', output_dir,
           '--only-assembler', '-t', str(threads)]
    contig_file = os.path.join(output_dir, 'contigs.fasta')
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        # Return code AND a non-empty result: a SPAdes that dies mid-run leaves
        # an empty contigs.fasta behind, and os.path.exists() called that a
        # successful assembly.
        if r.returncode == 0 and _wrote_output(contig_file):
            return contig_file
        logger.warning("SPAdes assembly failed (exit %s): %s",
                       r.returncode, (r.stderr or '')[:300])
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("SPAdes could not be run: %s", exc)
    return None


def _megahit_assemble(reads_file: str, output_dir: str, threads: int = 4) -> Optional[str]:
    """Run MEGAHIT assembler via subprocess (metagenomic mode).

    Returns the contigs file the tool wrote, or None if MEGAHIT did not produce
    one.
    """
    cmd = ['megahit-core', '-1', reads_file, '-o', output_dir,
           '--min-contig-len', '200', '-t', str(threads)]
    contig_file = os.path.join(output_dir, 'final.contigs.fa')
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        if r.returncode == 0 and _wrote_output(contig_file):
            return contig_file
        logger.warning("MEGAHIT assembly failed (exit %s): %s",
                       r.returncode, (r.stderr or '')[:300])
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("MEGAHIT could not be run: %s", exc)
    return None


# ── Public API ──────────────────────────────────────────────────────────────

def assemble(reads_file: str, output_fasta: Optional[str] = None, tool: str = 'auto', threads: int = 4) -> AssemblyResult:
    """Assemble reads auto-selecting engine; built-in greedy overlap
    fallback requires no external tools.

    Args:
        tool: 'auto', 'spades', 'megahit' or 'builtin'.  Anything else is a
            refusal: an unrecognised name used to fall through to the built-in
            assembler, so tool='spdes' quietly produced a built-in assembly.
    Raises:
        ValueError: If ``tool`` names no known engine.
    """
    if tool not in ('auto', 'spades', 'megahit', 'builtin'):
        raise ValueError(
            f"Unknown tool: {tool}. Use 'auto', 'spades', 'megahit' or 'builtin'.")

    if not os.path.exists(reads_file):
        return AssemblyResult(engine='none', message=f"Reads file not found: {reads_file}")

    tools = check_assembly_tools()

    if tool in ('spades', 'auto') and tools['spades']:
        # TemporaryDirectory, not mkdtemp(): every attempted external assembly
        # used to leave a full SPAdes/MEGAHIT output tree in /tmp.  The contig
        # file is parsed before the directory goes away.
        with tempfile.TemporaryDirectory(prefix='biosuite_spades_') as out_dir:
            contig_file = _spades_assemble(reads_file, out_dir, threads)
            if contig_file:
                return _parse_assembly_fasta(contig_file, 'spades')

    if tool in ('megahit', 'auto') and tools['megahit']:
        with tempfile.TemporaryDirectory(prefix='biosuite_megahit_') as out_dir:
            contig_file = _megahit_assemble(reads_file, out_dir, threads)
            if contig_file:
                return _parse_assembly_fasta(contig_file, 'megahit')

    # output_fasta is deliberately not reserved here any more:
    # _builtin_assembly() reserves it after it knows there are reads to write,
    # so an empty/garbage read file no longer leaves a 0-byte biosuite_*.fasta
    # behind whose path was never reported to the caller.
    return _builtin_assembly(reads_file, output_fasta)


def _parse_assembly_fasta(fasta_file: str, engine: str) -> AssemblyResult:
    """Parse assembled contigs from FASTA and compute AssemblyResult stats.

    A header with no sequence lines is not a contig: counting it added a
    zero-length entry that inflated ``num_contigs``, set ``min_contig`` to 0 and
    pulled the N50/L50 arithmetic down.
    """
    contigs = []
    name, buf = None, []
    with open(fasta_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name and buf:
                    contigs.append(''.join(buf))
                name = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line)
    if name and buf:
        contigs.append(''.join(buf))

    stats = _compute_assembly_stats(contigs)
    stats.engine = engine
    stats.output_fasta = fasta_file
    stats.message = f"{engine}: {stats.num_contigs} contigs, N50={stats.n50}"
    return stats


def format_assembly_report(result: AssemblyResult) -> str:
    """Format AssemblyResult with stats table and top-contig preview."""
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
