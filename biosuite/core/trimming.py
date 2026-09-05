"""
FASTQ quality trimming with dual-mode execution.

Uses Cutadapt if installed, otherwise falls back to a pure Python trimmer.
Works out of the box — no external tools required.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

ADAPTERS = {
    'illumina_nextera': 'CTGTCTCTTATACACATCT',
    'illumina_truseq_rna': 'AGATCGGAAGAGCACACGTCT',
    'illumina_truseq_dna': 'AGATCGGAAGAGCGTCGTGTAG',
    'polya': 'AAAAAAAAAA',
    'polyg': 'GGGGGGGGGG',
}


@dataclass
class TrimReport:
    """FASTQ trimming run summary.

    Attributes:
        reads_in/reads_out: Read counts before/after filtering.
        bases_trimmed: Total bases removed by quality/adapter cuts.
        engine: cutadapt/builtin marker.
    """
    input_file: str
    output_file: str
    total_reads: int = 0
    reads_trimmed: int = 0
    reads_removed: int = 0
    avg_quality_before: float = 0.0
    avg_quality_after: float = 0.0
    adapter_trimmed: int = 0
    engine: str = "builtin"
    message: str = ""


def check_trimming_tools() -> Dict[str, bool]:
    """Detect cutadapt availability."""
    tools = {'cutadapt': False}
    try:
        r = subprocess.run(['cutadapt', '--version'], capture_output=True, text=True, timeout=10)
        tools['cutadapt'] = r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return tools


# ── Pure Python Trimmer ─────────────────────────────────────────────────────

def _resolve_adapters(adapter: str, adapter_name: Optional[str]) -> list:
    """Build the adapter candidate list.

    'auto' tries every known adapter sequence (exact substring match — a
    heuristic; cutadapt does proper error-tolerant alignment when present).
    """
    if adapter_name and adapter_name in ADAPTERS:
        return [ADAPTERS[adapter_name]]
    if adapter != 'auto' and adapter not in ('', None):
        return [adapter] if adapter in ADAPTERS.values() else [ADAPTERS.get(adapter, adapter)]
    return list(ADAPTERS.values())  # 'auto': try them all


def _find_adapter(seq: str, adapters: list) -> int:
    """Return the position of the earliest adapter hit in *seq*, or -1."""
    best = -1
    for ad in adapters:
        pos = seq.find(ad)
        if 0 <= pos and (best < 0 or pos < best):
            best = pos
    return best


def _trim_one(seq: str, qual: str, adapters: list, quality_threshold: int):
    """Adapter + 3' quality trim of one record; returns (seq, qual)."""
    if adapters:
        pos = _find_adapter(seq, adapters)
        if pos >= 0:
            seq, qual = seq[:pos], qual[:pos]
    trim_pos = len(qual)
    while trim_pos > 0 and ord(qual[trim_pos - 1]) - 33 < quality_threshold:
        trim_pos -= 1
    return seq[:trim_pos], qual[:trim_pos]


def _pure_python_trim(input_file: str, output_file: str, quality_threshold: int = 20,
                      min_length: int = 36, adapters: Optional[list] = None,
                      adapter_seq: Optional[str] = None) -> TrimReport:
    """Trim FASTQ reads using pure Python — no external tools needed.

    Accepts ``adapters`` (list) or the legacy ``adapter_seq`` (single str).
    """
    if adapters is None and adapter_seq:
        adapters = [adapter_seq]
    total = 0
    trimmed = 0
    removed = 0
    adapter_hits = 0
    qual_before_sum = 0
    qual_after_sum = 0
    base_count = 0
    after_base_count = 0

    with open(input_file) as fin, open(output_file, 'w') as fout:
        while True:
            header = fin.readline()
            if not header:
                break
            seq = fin.readline().rstrip('\n')
            plus = fin.readline()
            qual = fin.readline().rstrip('\n')

            if not seq or not qual:
                break

            total += 1

            # Adapter trimming
            if adapters:
                pos = _find_adapter(seq, adapters)
                if pos >= 0:
                    adapter_hits += 1

            old_len = len(seq)
            qual_scores = [ord(c) - 33 for c in qual]
            qual_before_sum += sum(qual_scores)
            base_count += len(qual_scores)

            seq, qual = _trim_one(seq, qual, adapters, quality_threshold)

            if len(seq) < old_len:
                trimmed += 1

            qual_after_scores = [ord(c) - 33 for c in qual]
            qual_after_sum += sum(qual_after_scores)
            after_base_count += len(qual_after_scores)

            # Length filter
            if len(seq) < min_length:
                removed += 1
                continue

            fout.write(f"{header}{seq}\n{plus}{qual}\n")

    report = TrimReport(
        input_file=input_file,
        output_file=output_file,
        total_reads=total,
        reads_trimmed=trimmed,
        reads_removed=removed,
        avg_quality_before=qual_before_sum / base_count if base_count > 0 else 0,
        avg_quality_after=qual_after_sum / max(after_base_count, 1) if after_base_count > 0 else 0,
        adapter_trimmed=adapter_hits,
        engine="builtin",
        message="Using built-in quality trimmer"
    )
    return report


# ── Cutadapt Wrapper ─────────────────────────────────────────────────────────

def _cutadapt_trim(input_file: str, output_file: str, quality_threshold: int = 20,
                   min_length: int = 36, adapter_seq: Optional[str] = None) -> TrimReport:
    """Run cutadapt via subprocess on single-end input."""
    cmd = ['cutadapt', '-q', str(quality_threshold),
           '--minimum-length', str(min_length),
           '-o', output_file, input_file]
    if adapter_seq:
        cmd.extend(['-a', adapter_seq])

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            return None
        return _parse_cutadapt_stderr(r.stderr, input_file, output_file)
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_cutadapt_stderr(stderr: str, input_file: str, output_file: str) -> TrimReport:
    """Extract read/base counts from cutadapt stderr summary block."""
    report = TrimReport(input_file=input_file, output_file=output_file, engine='cutadapt')
    for line in stderr.split('\n'):
        line = line.strip()
        if 'Total reads processed:' in line:
            report.total_reads = int(line.split(':')[1].strip().replace(',', '').split('(')[0].strip())
        elif 'Reads with adapters:' in line:
            val = line.split(':')[1].strip().replace(',', '').split('(')[0].strip()
            report.adapter_trimmed = int(val)
        elif 'Reads that were too short:' in line:
            val = line.split(':')[1].strip().replace(',', '').split('(')[0].strip()
            report.reads_removed = int(val)
        elif 'Reads written (passing filters):' in line:
            val = line.split(':')[1].strip().replace(',', '').split('(')[0].strip()
            report.reads_trimmed = report.total_reads - int(val)
    report.message = "Using Cutadapt (external)"
    return report


# ── Public API ──────────────────────────────────────────────────────────────

def trim_fastq(input_file: str, output_file: Optional[str] = None, quality_threshold: int = 20,
               min_length: int = 36, adapter: str = 'auto', adapter_name: Optional[str] = None) -> TrimReport:
    """Single-end trim auto-selecting cutadapt or built-in sliding window."""
    if not os.path.exists(input_file):
        return TrimReport(input_file=input_file, output_file='',
                         message=f"File not found: {input_file}")

    if output_file is None:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_trimmed.fastq"

    adapters = _resolve_adapters(adapter, adapter_name)
    # 'auto' for cutadapt means "no -a flag" (cutadapt cannot guess);
    # the built-in engine instead tries every known adapter sequence.
    cutadapt_adapter = adapters[0] if (adapter_name or adapter != 'auto') and adapters else None

    # Try Cutadapt first
    if check_trimming_tools()['cutadapt']:
        result = _cutadapt_trim(input_file, output_file, quality_threshold,
                               min_length, cutadapt_adapter)
        if result is not None:
            return result

    # Pure Python fallback
    return _pure_python_trim(input_file, output_file, quality_threshold,
                            min_length, adapters)


def trim_pair_end(input_r1: str, input_r2: str, output_r1: Optional[str] = None, output_r2: Optional[str] = None,
                  quality_threshold: int = 20, min_length: int = 36, adapter: str = 'auto') -> TrimReport:
    """Paired-end trim keeping mate synchronization intact."""
    if output_r1 is None:
        output_r1 = os.path.splitext(input_r1)[0] + '_trimmed.fastq'
    if output_r2 is None:
        output_r2 = os.path.splitext(input_r2)[0] + '_trimmed.fastq'

    # Try paired-end Cutadapt
    if check_trimming_tools()['cutadapt']:
        adapter_seq = ADAPTERS.get(adapter, adapter) if adapter != 'auto' else None
        cmd = ['cutadapt', '-q', str(quality_threshold), '--minimum-length', str(min_length),
               '-o', output_r1, '-p', output_r2, input_r1, input_r2]
        if adapter_seq:
            cmd.extend(['-a', adapter_seq, '-A', adapter_seq])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if r.returncode == 0:
                report = _parse_cutadapt_stderr(r.stderr,
                    f"{input_r1} + {input_r2}", f"{output_r1} + {output_r2}")
                return report
        except (OSError, subprocess.SubprocessError):
            pass

    # Fallback: lockstep pure-Python trim that keeps mates synchronized —
    # dropping one mate of a pair would corrupt EVERY downstream tool
    # (the previous implementation trimmed the files independently and
    # silently desynchronized them).
    return _pure_python_trim_pair(input_r1, input_r2, output_r1, output_r2,
                                  quality_threshold, min_length, adapter)


def _pure_python_trim_pair(input_r1: str, input_r2: str, output_r1: str, output_r2: str,
                           quality_threshold: int = 20, min_length: int = 36,
                           adapter: str = 'auto') -> TrimReport:
    """Lockstep paired-end trim: a pair is written only if BOTH mates pass."""
    adapters = _resolve_adapters(adapter, None)
    total = kept = trimmed = adapter_hits = 0

    with open(input_r1) as f1, open(input_r2) as f2, \
            open(output_r1, 'w') as o1, open(output_r2, 'w') as o2:
        while True:
            rec1 = [f1.readline() for _ in range(4)]
            rec2 = [f2.readline() for _ in range(4)]
            if not rec1[0] or not rec2[0]:
                break
            seqs = [rec1[1].rstrip('\n'), rec2[1].strip('\n')]
            quals = [rec1[3].rstrip('\n'), rec2[3].rstrip('\n')]
            if not all(seqs) or not all(quals):
                break
            total += 1
            old_lens = [len(seqs[0]), len(seqs[1])]

            for i in range(2):
                if adapters and _find_adapter(seqs[i], adapters) >= 0:
                    adapter_hits += 1
                seqs[i], quals[i] = _trim_one(seqs[i], quals[i], adapters, quality_threshold)
                if len(seqs[i]) < old_lens[i]:
                    trimmed += 1

            if min(len(seqs[0]), len(seqs[1])) < min_length:
                continue  # discard the WHOLE pair — mates stay in lockstep
            kept += 1
            o1.write(f"{rec1[0]}{seqs[0]}\n{rec1[2]}{quals[0]}\n")
            o2.write(f"{rec2[0]}{seqs[1]}\n{rec2[2]}{quals[1]}\n")

    report = TrimReport(
        input_file=f"{input_r1} + {input_r2}",
        output_file=f"{output_r1} + {output_r2}",
        total_reads=total * 2,
        reads_trimmed=trimmed,
        reads_removed=(total - kept) * 2,
        adapter_trimmed=adapter_hits,
        engine="builtin",
        message=(f"Built-in lockstep paired trimmer: {kept}/{total} pairs kept "
                 f"(whole pairs dropped to preserve mate sync)"),
    )
    return report


def analyze_fastq_quality(filepath: str, max_reads: int = 100000) -> Dict[str, Any]:
    """Per-cycle quality profile for FASTQ QC reporting."""
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}

    qualities = []
    lengths = []
    count = 0

    with open(filepath) as f:
        while count < max_reads:
            header = f.readline()
            if not header:
                break
            seq = f.readline().strip()
            f.readline()
            qual = f.readline().strip()
            if qual:
                scores = [ord(c) - 33 for c in qual]
                qualities.extend(scores)
                lengths.append(len(seq))
            count += 1

    if not qualities:
        return {"error": "No quality data extracted."}

    q = np.array(qualities)
    rl = np.array(lengths)
    return {
        'total_reads': count,
        'read_length_mean': float(rl.mean()),
        'quality_mean': float(q.mean()),
        'quality_median': float(np.median(q)),
        'percent_above_q20': float((q >= 20).sum() / len(q) * 100),
        'percent_above_q30': float((q >= 30).sum() / len(q) * 100),
    }


def format_trim_report(report: TrimReport) -> str:
    """Format TrimReport with before/after retention statistics."""
    lines = [
        "=== FASTQ Trimming Report ===",
        f"Engine: {report.engine}",
        f"Input: {report.input_file}",
        f"Output: {report.output_file}",
    ]
    if report.message:
        lines.append(f"Status: {report.message}")
    if report.total_reads > 0:
        lines.extend([
            f"Total reads: {report.total_reads:,}",
            f"Reads trimmed: {report.reads_trimmed:,}",
            f"Reads removed (too short): {report.reads_removed:,}",
            f"Adapters trimmed: {report.adapter_trimmed:,}",
            f"Mean quality before: {report.avg_quality_before:.1f}",
            f"Mean quality after: {report.avg_quality_after:.1f}",
        ])
    return '\n'.join(lines)
