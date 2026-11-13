"""
Short read alignment with dual-mode execution.

Provides a pure Python seed-and-extend aligner as default, with BWA/Bowtie2
as optional faster alternatives when installed.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Alignment:
    """One aligned read with mapping coordinates and flags.

    Attributes:
        qname: Read identifier.
        flag: SAM flag integer (strand/paired state).
        chrom: Reference chromosome.
        pos: 1-based leftmost mapped position.
        mapq: Mapping quality score.
        cigar: CIGAR alignment string.
    """
    read_id: str
    reference_id: str
    position: int
    strand: str
    cigar: str
    mapping_quality: int
    score: int
    edit_distance: int
    is_primary: bool = True
    sequence: str = ""


@dataclass
class AlignmentReport:
    """Batch aligner output summary.

    Attributes:
        alignments: List of Alignment records.
        total_reads/unmapped: Counters for report headers.
        engine: builtin/bwa/bowtie2 marker.
    """
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


from .utils import has_tool as _has_tool, wrote_output as _wrote_output


def check_aligner_tools() -> dict :
    """Detect bwa/bowtie2 availability on PATH."""
    return {'bwa': _has_tool('bwa'), 'bowtie2': _has_tool('bowtie2')}


# ── Pure Python Seed-and-Extend Aligner ─────────────────────────────────────

def _build_suffix_index(reference: str, k: int = 15) -> Dict[str, List[int]]:
    """Index reference k-mers for seed lookup during alignment."""
    index = defaultdict(list)
    for i in range(len(reference) - k + 1):
        kmer = reference[i:i + k]
        if 'N' not in kmer:
            index[kmer].append(i)
    return index


def _build_fasta_index(fasta_file: str) -> Dict[str, str]:
    """Load reference FASTA into memory index for the built-in engine."""
    refs = {}
    name, seq = None, []
    with open(fasta_file, encoding="utf-8") as f:
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


def _seed_and_extend(read_seq: str, ref_seq: str, ref_name: str, index: Any, k: int = 15, seed_threshold: int = 3) -> Optional[Dict[str, Any]]:
    """Seed-and-extend alignment of one read against the reference index.

    Ungapped scoring: after seed clustering fixes the mapping offset, the
    identity and edit distance are computed base-by-base over the aligned
    window, and read bases that fall off either end of the reference are
    soft-clipped (substitutions only — no gap placement, no
    reverse-complement search; use BWA/Bowtie2 for those).
    """
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

    # Mapping offset = consensus (ref_pos - read_pos) of the seed cluster.
    # The old code used seeds[0][1] as the position — that ignores the
    # read-side offset and shifts the locus right whenever the read has a
    # 5' region without seeds.
    offset = best_offset

    # Only the read bases that still land on the reference are aligned: what
    # hangs off either end is soft-clipped.  Comparing from index 0 of the read
    # after clamping the position to zero scored the hanging bases as
    # mismatches at the wrong locus, and a CIGAR of len(read_seq)M consumes
    # more reference than the contig holds, which is not a valid SAM record.
    first = max(0, -offset)
    last = min(len(read_seq), len(ref_seq) - offset)
    # aligned >= 1 for any seed cluster: a seed needs len(read_seq) >= k and
    # len(ref_seq) >= k, and |offset| <= len(read_seq) - k by construction.
    aligned = last - first

    matches = 0
    for i in range(first, last):
        if ref_seq[offset + i] == read_seq[i]:
            matches += 1
    identity = matches / aligned * 100
    mismatches = aligned - matches

    position = offset + first
    cigar = "".join([
        f"{first}S" if first else "",
        f"{aligned}M",
        f"{len(read_seq) - last}S" if last < len(read_seq) else "",
    ])
    score = matches * 2 - mismatches * 3

    return Alignment(
        read_id="",
        reference_id=ref_name,
        position=position,
        strand="+",
        cigar=cigar,
        mapping_quality=min(60, int(identity * 0.6)),
        score=score,
        edit_distance=mismatches
    )


def _builtin_align_reads(reads_file: str, reference_file: str, output_file: Optional[str] = None) -> AlignmentReport:
    """Pure-Python multi-read alignment using suffix seeding."""
    refs = _build_fasta_index(reference_file)
    alignments = []
    total = 0
    mapped = 0
    qualities = []

    ref_indices = {}
    for name, seq in refs.items():
        ref_indices[name] = _build_suffix_index(seq, k=15)

    with open(reads_file, encoding="utf-8") as f:
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
                best_hit.sequence = seq
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


def _write_sam(report: AlignmentReport, output_file: str, refs: Dict[str, str]) -> None:
    """Emit alignments in SAM format with computed MAPQ/flags."""
    with open(output_file, 'w', encoding="utf-8") as f:
        f.write("@HD\tVN:1.6\tSO:coordinate\n")
        for name in refs:
            f.write(f"@SQ\tSN:{name}\tLN:{len(refs[name])}\n")
        for aln in report.alignments:
            flag = 4 if aln.reference_id == "*" else 0
            # Emit the read sequence (and flat high qualities) so downstream
            # pileup tools — e.g. our variant caller — can use this SAM.
            seq = aln.sequence if aln.sequence and flag != 4 else "*"
            qual = "I" * len(seq) if seq != "*" else "*"
            f.write(f"{aln.read_id}\t{flag}\t{aln.reference_id}\t{aln.position + 1}\t"
                    f"{aln.mapping_quality}\t{aln.cigar}\t*\t0\t0\t{seq}\t{qual}\n")


# ── External Tool Wrappers ──────────────────────────────────────────────────

def _drop_scratch(path: Optional[str]) -> None:
    """Remove a scratch SAM this module created but did not hand back."""
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _summarize_external_sam(sam_file: str) -> Dict[str, Any]:
    """Count alignment records in a SAM written by an external aligner.

    Only the counters are read back: ``_bwa_align``/``_bowtie2_align`` promised
    to "parse SAM output" but never did, so every external-engine report said
    ``Total reads: 0`` while the SAM on disk was full.
    """
    total = mapped = 0
    qualities: List[int] = []
    with open(sam_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith('@'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 11:
                continue
            total += 1
            if int(fields[1]) & 4:
                continue
            mapped += 1
            qualities.append(int(fields[4]))
    return {
        'total_reads': total,
        'mapped_reads': mapped,
        'unmapped_reads': total - mapped,
        'mapping_rate': mapped / total * 100 if total > 0 else 0,
        'avg_mapping_quality': float(np.mean(qualities)) if qualities else 0,
    }


def _bwa_align(reference: str, reads_r1: str, output_bam: str, threads: int = 1) -> Optional[str]:
    """Run BWA-MEM via subprocess and write its SAM to *output_bam*.

    Returns the path of the SAM produced, or None if the tool did not run.
    """
    idx_cmd = ['bwa', 'index', reference]
    try:
        subprocess.run(idx_cmd, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None

    # The caller's path is the output: writing only into a private temp file
    # made align_reads(output_file=...) report success while the file the
    # caller asked for was never created.
    scratch: Optional[str] = None
    if output_bam:
        sam_file = output_bam
    else:
        with tempfile.NamedTemporaryFile(suffix='.sam', delete=False) as tmp:
            sam_file = scratch = tmp.name

    cmd = ['bwa', 'mem', '-t', str(threads), reference, reads_r1]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        # bwa mem writes the SAM to stdout, so exit 0 with empty stdout is a
        # failure; without this guard an empty file reached the report.
        if r.returncode == 0 and r.stdout.strip():
            with open(sam_file, 'w', encoding="utf-8") as f:
                f.write(r.stdout)
            return sam_file
    except (OSError, subprocess.SubprocessError):
        pass
    _drop_scratch(scratch)
    return None


def _bowtie2_align(reference: str, reads_r1: str, output_bam: str, threads: int = 1) -> Optional[str]:
    """Run bowtie2 via subprocess and write its SAM to *output_bam*.

    Returns the path of the SAM produced, or None if the tool did not run.
    """
    scratch: Optional[str] = None
    if output_bam:
        sam_file = output_bam
    else:
        with tempfile.NamedTemporaryFile(suffix='.sam', delete=False) as tmp:
            sam_file = scratch = tmp.name
    # The index prefix gets its own directory: mkstemp() left the placeholder
    # file (and every .bt2 sidecar bowtie2-build writes next to it) behind on
    # every run, successful or not.
    with tempfile.TemporaryDirectory(prefix='biosuite_bowtie2_') as idx_dir:
        prefix = os.path.join(idx_dir, 'bowtie2idx')
        try:
            subprocess.run(['bowtie2-build', reference, prefix], capture_output=True, timeout=600)
            cmd = ['bowtie2', '-x', prefix, '-U', reads_r1, '--threads', str(threads), '-S', sam_file]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if r.returncode == 0 and _wrote_output(sam_file):
                return sam_file
        except (OSError, subprocess.SubprocessError):
            pass
    _drop_scratch(scratch)
    return None


# ── Public API ──────────────────────────────────────────────────────────────

def align_reads(reference_file: str, reads_file: str, output_file: Optional[str] = None, tool: str = 'auto', threads: int = 1) -> AlignmentReport:
    """Align reads auto-selecting best available engine.

    Prefers bwa > bowtie2 > built-in; never raises on missing tools.
    """
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
                                   output_file=sam,
                                   **_summarize_external_sam(sam))

    if tool in ('bowtie2', 'auto') and tools['bowtie2']:
        sam = _bowtie2_align(reference_file, reads_file, output_file, threads)
        if sam:
            return AlignmentReport(tool='bowtie2', engine='bowtie2',
                                   message="Using Bowtie2 (external)",
                                   output_file=sam,
                                   **_summarize_external_sam(sam))

    if output_file is None:
        _fd, output_file = tempfile.mkstemp(suffix=".sam")
        os.close(_fd)
    return _builtin_align_reads(reads_file, reference_file, output_file)


def format_alignment_report(report: AlignmentReport) -> str:
    """Format AlignmentReport with mapping-rate breakdown table."""
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
