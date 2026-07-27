"""
FASTQ quality trimming with dual-mode execution.

Uses Cutadapt if installed, otherwise falls back to a pure Python trimmer.
Works out of the box — no external tools required.
"""
import os
import subprocess
import tempfile
import numpy as np
from dataclasses import dataclass

ADAPTERS = {
    'illumina_nextera': 'CTGTCTCTTATACACATCT',
    'illumina_truseq_rna': 'AGATCGGAAGAGCACACGTCT',
    'illumina_truseq_dna': 'AGATCGGAAGAGCGTCGTGTAG',
    'polya': 'AAAAAAAAAA',
    'polyg': 'GGGGGGGGGG',
}


@dataclass
class TrimReport:
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


def check_trimming_tools():
    tools = {'cutadapt': False}
    try:
        r = subprocess.run(['cutadapt', '--version'], capture_output=True, text=True, timeout=10)
        tools['cutadapt'] = r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return tools


# ── Pure Python Trimmer ─────────────────────────────────────────────────────

def _pure_python_trim(input_file, output_file, quality_threshold=20,
                      min_length=36, adapter_seq=None):
    """Trim FASTQ reads using pure Python — no external tools needed."""
    total = 0
    trimmed = 0
    removed = 0
    adapter_hits = 0
    qual_before_sum = 0
    qual_after_sum = 0
    base_count = 0

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
            if adapter_seq:
                pos = seq.find(adapter_seq)
                if pos >= 0:
                    seq = seq[:pos]
                    qual = qual[:pos]
                    adapter_hits += 1

            qual_scores = [ord(c) - 33 for c in qual]
            qual_before_sum += sum(qual_scores)
            base_count += len(qual_scores)

            # Quality trimming from 3' end
            trim_pos = len(qual_scores)
            while trim_pos > 0 and qual_scores[trim_pos - 1] < quality_threshold:
                trim_pos -= 1

            if trim_pos < len(seq):
                trimmed += 1

            seq = seq[:trim_pos]
            qual = qual[:trim_pos]
            qual_after_scores = [ord(c) - 33 for c in qual]
            qual_after_sum += sum(qual_after_scores)

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
        avg_quality_after=qual_after_sum / max(base_count - trimmed, 1) if base_count > 0 else 0,
        adapter_trimmed=adapter_hits,
        engine="builtin",
        message="Using built-in quality trimmer"
    )
    return report


# ── Cutadapt Wrapper ─────────────────────────────────────────────────────────

def _cutadapt_trim(input_file, output_file, quality_threshold=20,
                   min_length=36, adapter_seq=None):
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


def _parse_cutadapt_stderr(stderr, input_file, output_file):
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

def trim_fastq(input_file, output_file=None, quality_threshold=20,
               min_length=36, adapter='auto', adapter_name=None):
    if not os.path.exists(input_file):
        return TrimReport(input_file=input_file, output_file='',
                         message=f"File not found: {input_file}")

    if output_file is None:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_trimmed.fastq"

    # Resolve adapter
    adapter_seq = None
    if adapter_name and adapter_name in ADAPTERS:
        adapter_seq = ADAPTERS[adapter_name]
    elif adapter != 'auto':
        adapter_seq = adapter

    # Try Cutadapt first
    if check_trimming_tools()['cutadapt']:
        result = _cutadapt_trim(input_file, output_file, quality_threshold,
                               min_length, adapter_seq)
        if result is not None:
            return result

    # Pure Python fallback
    return _pure_python_trim(input_file, output_file, quality_threshold,
                            min_length, adapter_seq)


def trim_pair_end(input_r1, input_r2, output_r1=None, output_r2=None,
                  quality_threshold=20, min_length=36, adapter='auto'):
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

    # Fallback: trim each file independently
    r1 = trim_fastq(input_r1, output_r1, quality_threshold, min_length, adapter)
    r2 = trim_fastq(input_r2, output_r2, quality_threshold, min_length, adapter)
    r1.input_file = f"{input_r1} + {input_r2}"
    r1.output_file = f"{output_r1} + {output_r2}"
    r1.total_reads += r2.total_reads
    r1.reads_trimmed += r2.reads_trimmed
    r1.reads_removed += r2.reads_removed
    r1.message = "Using built-in trimmer (paired-end processed independently)"
    return r1


def analyze_fastq_quality(filepath, max_reads=100000):
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


def format_trim_report(report):
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
