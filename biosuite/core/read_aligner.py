"""
Short read alignment with dual-mode execution.

Provides a pure Python seed-and-extend aligner as default, with BWA/Bowtie2
as optional faster alternatives when installed.
"""
import os
import subprocess
import tempfile
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field


@dataclass
class Alignment:
    read_id: str
    reference_id: str
    position: int
    strand: str
    cigar: str
    mapping_quality: int
    score: int
    edit_distance: int
    is_primary: bool = True


@dataclass
class AlignmentReport:
    tool: str
    engine: str
    total_reads: int = 0
    mapped_reads: int = 0
    unmapped_reads: int = 0
    mapping_rate: float = 0.0
    avg_mapping_quality: float = 0.0
    alignments: list = field(default_factory=list)
    output_file: str = ""
    message: str = ""


from .utils import has_tool as _has_tool


def check_aligner_tools():
    return {'bwa': _has_tool('bwa'), 'bowtie2': _has_tool('bowtie2')}


# ── Pure Python Seed-and-Extend Aligner ─────────────────────────────────────

def _build_suffix_index(reference, k=15):
    index = defaultdict(list)
    for i in range(len(reference) - k + 1):
        kmer = reference[i:i + k]
        if 'N' not in kmer:
            index[kmer].append(i)
    return index


def _build_fasta_index(fasta_file):
    refs = {}
    name, seq = None, []
    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name:
                    refs[name] = ''.join(seq)
                name = line[1:].split()[0]
                seq = []
            elif line:
                seq.append(line)
    if name:
        refs[name] = ''.join(seq)
    return refs


def _seed_and_extend(read_seq, ref_seq, ref_name, index, k=15, seed_threshold=3):
    seeds = []
    for i in range(len(read_seq) - k + 1):
        kmer = read_seq[i:i + k]
        if kmer in index:
            for pos in index[kmer]:
                seeds.append((i, pos))

    if len(seeds) < seed_threshold:
        return None

    clusters = defaultdict(list)
    for read_pos, ref_pos in seeds:
        offset = ref_pos - read_pos
        clusters[offset].append((read_pos, ref_pos))

    best_offset = max(clusters, key=lambda x: len(clusters[x]))
    best_seeds = clusters[best_offset]

    if len(best_seeds) < seed_threshold:
        return None

    ref_start = max(0, best_seeds[0][1] - 10)
    ref_end = min(len(ref_seq), best_seeds[-1][1] + len(read_seq) + 10)
    target = ref_seq[ref_start:ref_end]

    matches = sum(1 for rp, rep in best_seeds if rep < len(ref_seq) and ref_seq[rep] == read_seq[rp])
    total_seeds = len(best_seeds)
    identity = matches / total_seeds * 100 if total_seeds > 0 else 0

    cigar = f"{len(read_seq)}M"
    score = matches * 2 - (total_seeds - matches) * 3

    return Alignment(
        read_id="",
        reference_id=ref_name,
        position=best_seeds[0][1],
        strand="+",
        cigar=cigar,
        mapping_quality=min(60, int(identity * 0.6)),
        score=score,
        edit_distance=total_seeds - matches
    )


def _builtin_align_reads(reads_file, reference_file, output_file=None):
    refs = _build_fasta_index(reference_file)
    alignments = []
    total = 0
    mapped = 0
    qualities = []

    ref_indices = {}
    for name, seq in refs.items():
        ref_indices[name] = _build_suffix_index(seq, k=15)

    with open(reads_file) as f:
        while True:
            header = f.readline()
            if not header:
                break
            seq = f.readline().strip()
            f.readline()
            f.readline()
            if not seq:
                break
            total += 1
            read_id = header.strip().lstrip('@').split()[0]

            best_hit = None
            best_score = -999
            for ref_name, ref_seq in refs.items():
                hit = _seed_and_extend(seq, ref_seq, ref_name, ref_indices[ref_name])
                if hit and hit.score > best_score:
                    best_hit = hit
                    best_score = hit.score

            if best_hit:
                best_hit.read_id = read_id
                alignments.append(best_hit)
                mapped += 1
                qualities.append(best_hit.mapping_quality)
            else:
                alignments.append(Alignment(
                    read_id=read_id, reference_id="*", position=0,
                    strand="*", cigar="*", mapping_quality=0,
                    score=0, edit_distance=0, is_primary=False
                ))

    report = AlignmentReport(
        tool='builtin_seed_extend',
        engine='builtin',
        total_reads=total,
        mapped_reads=mapped,
        unmapped_reads=total - mapped,
        mapping_rate=mapped / total * 100 if total > 0 else 0,
        avg_mapping_quality=float(np.mean(qualities)) if qualities else 0,
        alignments=alignments,
        message=f"Built-in aligner: {mapped}/{total} reads mapped"
    )

    if output_file:
        _write_sam(report, output_file, refs)
        report.output_file = output_file

    return report


def _write_sam(report, output_file, refs):
    with open(output_file, 'w') as f:
        f.write("@HD\tVN:1.6\tSO:coordinate\n")
        for name in refs:
            f.write(f"@SQ\tSN:{name}\tLN:{len(refs[name])}\n")
        for aln in report.alignments:
            flag = 4 if aln.reference_id == "*" else 0
            f.write(f"{aln.read_id}\t{flag}\t{aln.reference_id}\t{aln.position + 1}\t"
                    f"{aln.mapping_quality}\t{aln.cigar}\t*\t0\t0\t*\t*\n")


# ── External Tool Wrappers ──────────────────────────────────────────────────

def _bwa_align(reference, reads_r1, output_bam, threads=1):
    idx_cmd = ['bwa', 'index', reference]
    try:
        subprocess.run(idx_cmd, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None

    with tempfile.NamedTemporaryFile(suffix='.sam', delete=False) as tmp:
        sam_file = tmp.name

    cmd = ['bwa', 'mem', '-t', str(threads), reference, reads_r1]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode == 0:
            with open(sam_file, 'w') as f:
                f.write(r.stdout)
            return sam_file
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _bowtie2_align(reference, reads_r1, output_bam, threads=1):
    with tempfile.NamedTemporaryFile(suffix='.sam', delete=False) as tmp:
        sam_file = tmp.name
    prefix = tempfile.mktemp()
    try:
        subprocess.run(['bowtie2-build', reference, prefix], capture_output=True, timeout=600)
        cmd = ['bowtie2', '-x', prefix, '-U', reads_r1, '--threads', str(threads), '-S', sam_file]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode == 0:
            return sam_file
    except (OSError, subprocess.SubprocessError):
        pass
    return None


# ── Public API ──────────────────────────────────────────────────────────────

def align_reads(reference_file, reads_file, output_file=None, tool='auto', threads=1):
    if not os.path.exists(reference_file):
        return AlignmentReport(tool='none', engine='none', message=f"Reference not found: {reference_file}")
    if not os.path.exists(reads_file):
        return AlignmentReport(tool='none', engine='none', message=f"Reads not found: {reads_file}")

    tools = check_aligner_tools()

    if tool in ('bwa', 'auto') and tools['bwa']:
        sam = _bwa_align(reference_file, reads_file, output_file, threads)
        if sam:
            return AlignmentReport(tool='bwa', engine='bwa',
                                   message="Using BWA-MEM (external)",
                                   output_file=sam)

    if tool in ('bowtie2', 'auto') and tools['bowtie2']:
        sam = _bowtie2_align(reference_file, reads_file, output_file, threads)
        if sam:
            return AlignmentReport(tool='bowtie2', engine='bowtie2',
                                   message="Using Bowtie2 (external)",
                                   output_file=sam)

    if output_file is None:
        output_file = tempfile.mktemp(suffix='.sam')
    return _builtin_align_reads(reads_file, reference_file, output_file)


def format_alignment_report(report):
    lines = [
        "=== Read Alignment Report ===",
        f"Engine: {report.engine}",
        f"Total reads: {report.total_reads:,}",
        f"Mapped: {report.mapped_reads:,} ({report.mapping_rate:.1f}%)",
        f"Unmapped: {report.unmapped_reads:,}",
        f"Avg MAPQ: {report.avg_mapping_quality:.1f}",
    ]
    if report.message:
        lines.append(f"Note: {report.message}")
    return '\n'.join(lines)
