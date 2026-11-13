"""
FASTQ quality trimming with dual-mode execution.

Uses Cutadapt if installed, otherwise falls back to a pure Python trimmer.
Works out of the box — no external tools required.

The built-in engine cuts adapters (exact substring match) and trims the 3'
quality tail below the threshold; it is not a sliding window and it does not
touch the 5' end.  A read whose length falls below ``min_length`` after those
cuts is dropped whole.
"""
from __future__ import annotations

import os
import subprocess
import warnings
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

# Phred+33 spans '!' (Q0) to '~' (Q93); anything outside it is not a quality.
_QUALITY_MIN, _QUALITY_MAX = ord('!'), ord('~')


@dataclass
class TrimReport:
    """FASTQ trimming run summary.

    Attributes:
        input_file/output_file: paths the run read and wrote (the paired engine
            names both mates in one string, and a run that never started leaves
            ``output_file`` empty).
        total_reads: records read from the input.
        reads_trimmed: records whose sequence was cut (built-in engine) or, for
            cutadapt, records reported as carrying an adapter.
        reads_removed: records dropped because they came out too short.
        adapter_trimmed: records with an adapter hit.
        avg_quality_before: mean Phred quality per base over the input.
        avg_quality_after: mean Phred quality per base over the records actually
            written to the output.
        engine: ``cutadapt`` or ``builtin``.
        message: what the run did, or why it did nothing.
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
    # A probe is allowed to fail in every way a probe can fail — the binary is
    # absent, not executable, killed by the timeout.  None of them means
    # "cutadapt is installed".
    except (OSError, subprocess.SubprocessError):
        pass
    return tools


# ── Pure Python Trimmer ─────────────────────────────────────────────────────

def _resolve_adapters(adapter: str, adapter_name: Optional[str]) -> list:
    """Build the adapter candidate list.

    ``adapter_name`` wins over ``adapter`` when both are given, and an unknown
    name is refused — silently falling back to 'auto' would trim with five
    adapters the user never asked for.  ``adapter`` may be a key of ``ADAPTERS``
    or a literal sequence; ``'auto'`` tries every known adapter sequence (exact
    substring match — a heuristic; cutadapt does proper error-tolerant alignment
    when present).
    """
    if adapter_name is not None:
        if adapter_name not in ADAPTERS:
            raise ValueError(
                f"unknown adapter_name {adapter_name!r}; known names are "
                f"{', '.join(sorted(ADAPTERS))}, or pass the adapter sequence in adapter="
            )
        return [ADAPTERS[adapter_name]]
    if adapter != 'auto' and adapter not in ('', None):
        return [ADAPTERS.get(adapter, adapter)]
    return list(ADAPTERS.values())  # 'auto': try them all


def _find_adapter(seq: str, adapters: list) -> int:
    """Return the position of the earliest adapter hit in *seq*, or -1."""
    best = -1
    for ad in adapters:
        pos = seq.find(ad)
        if 0 <= pos and (best < 0 or pos < best):
            best = pos
    return best


def _read_record(handle, filepath: str):
    """Read one FASTQ record, or return None at end of file.

    The trimmer writes what it reads, so a record it does not understand must
    stop the run with a message instead of being guessed at: measured on the old
    loop, a quality line running into the next header produced a report of
    "1 read processed" beside an output file containing one mangled record, and a
    record whose quality line was longer than its sequence was written back
    unchanged — an invalid FASTQ handed to whatever runs next.
    """
    while True:
        header = handle.readline()
        if not header:
            return None
        if not header.strip():
            continue  # blank lines between records are tolerated, as in read_fastq
        seq = handle.readline().rstrip('\n')
        plus = handle.readline()
        qual = handle.readline().rstrip('\n')
        name = header.split()[0]
        if not name.startswith('@'):
            raise ValueError(f"malformed FASTQ in {filepath}: expected '@' at the start "
                             f"of a record, found {name[:20]!r}.")
        if not plus.startswith('+'):
            raise ValueError(f"malformed FASTQ in {filepath}: expected a '+' separator "
                             f"after {name!r}, found {plus.rstrip(chr(10))[:20]!r}.")
        if not seq or not qual:
            raise ValueError(f"malformed FASTQ in {filepath}: record {name!r} has an "
                             f"empty sequence or quality line.")
        if len(seq) != len(qual):
            raise ValueError(f"malformed FASTQ in {filepath}: record {name!r} has "
                             f"{len(seq)} bases but {len(qual)} quality scores.")
        bad = [c for c in qual if not _QUALITY_MIN <= ord(c) <= _QUALITY_MAX]
        if bad:
            raise ValueError(f"malformed FASTQ in {filepath}: record {name!r} carries "
                             f"{len(bad)} character(s) outside the Phred+33 range, "
                             f"starting with {bad[0]!r} (Q{ord(bad[0]) - 33}).")
        return header, seq, plus, qual


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


def _check_cut_parameters(quality_threshold, min_length, caller) -> None:
    if isinstance(quality_threshold, bool) or not isinstance(quality_threshold, int):
        raise ValueError(f"{caller} quality_threshold must be an integer Phred score, "
                         f"got {quality_threshold!r}.")
    if isinstance(min_length, bool) or not isinstance(min_length, int) or min_length < 0:
        raise ValueError(f"{caller} min_length must be a non-negative integer, "
                         f"got {min_length!r}.")


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

    with open(input_file, encoding="utf-8") as fin, open(output_file, 'w', encoding="utf-8") as fout:
        while True:
            record = _read_record(fin, input_file)
            if record is None:
                break
            header, seq, plus, qual = record

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

            # Length filter
            if len(seq) < min_length:
                removed += 1
                continue

            # "After" describes the file that was written, not the reads that were
            # dropped on the way to it.
            qual_after_scores = [ord(c) - 33 for c in qual]
            qual_after_sum += sum(qual_after_scores)
            after_base_count += len(qual_after_scores)

            fout.write(f"{header}{seq}\n{plus}{qual}\n")

    report = TrimReport(
        input_file=input_file,
        output_file=output_file,
        total_reads=total,
        reads_trimmed=trimmed,
        reads_removed=removed,
        avg_quality_before=qual_before_sum / base_count if base_count > 0 else 0,
        avg_quality_after=qual_after_sum / after_base_count if after_base_count > 0 else 0,
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
    except (OSError, subprocess.SubprocessError) as exc:
        _warn_cutadapt_fallback(exc)
        return None
    if r.returncode != 0:
        _warn_cutadapt_fallback(f"exit code {r.returncode}: {(r.stderr or '').strip()[:200]}")
        return None
    return _parse_cutadapt_stderr(r.stderr, input_file, output_file)


def _warn_cutadapt_fallback(reason) -> None:
    """Say out loud that the external engine was present but did not finish.

    Without this the caller gets a built-in report with no sign that cutadapt was
    asked first and failed, which is how a bad adapter argument reads as a clean run.
    """
    warnings.warn(f"cutadapt did not finish, falling back to the built-in trimmer: "
                  f"{reason}", RuntimeWarning, stacklevel=3)


def _cutadapt_count(line: str, label: str) -> int:
    """The integer a cutadapt summary line reports, or a message naming the line."""
    value = line.split(':')[1].strip().replace(',', '').split('(')[0].strip()
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"unexpected cutadapt summary line: {line.strip()!r} "
                         f"(no count after {label!r})") from None


def _parse_cutadapt_stderr(stderr: str, input_file: str, output_file: str) -> TrimReport:
    """Extract read/base counts from cutadapt stderr summary block.

    ``reads_trimmed`` is the adapter count, not "processed minus written": the
    difference is the number of reads that were *not written*, which is already
    what ``reads_removed`` reports, so the old assignment counted the same reads
    twice under two different labels.  Cutadapt's summary gives no separate
    per-read count of quality trimming, so the two fields carry the same number
    here — and the built-in engine, which does count cuts per read, keeps its own.
    """
    report = TrimReport(input_file=input_file, output_file=output_file, engine='cutadapt')
    for line in stderr.split('\n'):
        line = line.strip()
        if 'Total reads processed:' in line:
            report.total_reads = _cutadapt_count(line, 'Total reads processed')
        elif 'Reads with adapters:' in line:
            report.adapter_trimmed = _cutadapt_count(line, 'Reads with adapters')
            report.reads_trimmed = report.adapter_trimmed
        elif 'Reads that were too short:' in line:
            report.reads_removed = _cutadapt_count(line, 'Reads that were too short')
    report.message = "Using Cutadapt (external)"
    return report


# ── Public API ──────────────────────────────────────────────────────────────

def trim_fastq(input_file: str, output_file: Optional[str] = None, quality_threshold: int = 20,
               min_length: int = 36, adapter: str = 'auto', adapter_name: Optional[str] = None) -> TrimReport:
    """Single-end trim, cutadapt first and the built-in 3' trimmer as fallback."""
    _check_cut_parameters(quality_threshold, min_length, "trim_fastq")
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
    _check_cut_parameters(quality_threshold, min_length, "trim_pair_end")
    for missing in (input_r1, input_r2):
        if not os.path.exists(missing):
            return TrimReport(input_file=missing, output_file='',
                              message=f"File not found: {missing}")

    adapters = _resolve_adapters(adapter, None)
    if output_r1 is None:
        output_r1 = os.path.splitext(input_r1)[0] + '_trimmed.fastq'
    if output_r2 is None:
        output_r2 = os.path.splitext(input_r2)[0] + '_trimmed.fastq'

    # Try paired-end Cutadapt
    if check_trimming_tools()['cutadapt']:
        adapter_seq = adapters[0] if adapter != 'auto' else None
        cmd = ['cutadapt', '-q', str(quality_threshold), '--minimum-length', str(min_length),
               '-o', output_r1, '-p', output_r2, input_r1, input_r2]
        if adapter_seq:
            cmd.extend(['-a', adapter_seq, '-A', adapter_seq])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if r.returncode == 0:
                return _parse_cutadapt_stderr(
                    r.stderr, f"{input_r1} + {input_r2}", f"{output_r1} + {output_r2}")
            _warn_cutadapt_fallback(f"exit code {r.returncode}: "
                                    f"{(r.stderr or '').strip()[:200]}")
        except (OSError, subprocess.SubprocessError) as exc:
            _warn_cutadapt_fallback(exc)

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
    # The paired engine used to leave both quality means at their 0.0 default, and
    # the report printed "Mean quality before: 0.0" over a file of Q40 reads.
    qual_before_sum = qual_after_sum = base_count = after_base_count = 0

    with open(input_r1, encoding="utf-8") as f1, open(input_r2, encoding="utf-8") as f2, \
            open(output_r1, 'w', encoding="utf-8") as o1, open(output_r2, 'w', encoding="utf-8") as o2:
        while True:
            rec1 = _read_record(f1, input_r1)
            rec2 = _read_record(f2, input_r2)
            if rec1 is None or rec2 is None:
                if (rec1 is None) != (rec2 is None):
                    raise ValueError(
                        f"paired FASTQ files disagree: {input_r1} and {input_r2} do "
                        f"not hold the same number of records, so mate order cannot "
                        f"be preserved.")
                break
            seqs = [rec1[1], rec2[1]]
            quals = [rec1[3], rec2[3]]
            total += 1
            old_lens = [len(seqs[0]), len(seqs[1])]
            for q in quals:
                scores = [ord(c) - 33 for c in q]
                qual_before_sum += sum(scores)
                base_count += len(scores)

            for i in range(2):
                if adapters and _find_adapter(seqs[i], adapters) >= 0:
                    adapter_hits += 1
                seqs[i], quals[i] = _trim_one(seqs[i], quals[i], adapters, quality_threshold)
                if len(seqs[i]) < old_lens[i]:
                    trimmed += 1

            if min(len(seqs[0]), len(seqs[1])) < min_length:
                continue  # discard the WHOLE pair — mates stay in lockstep
            kept += 1
            for q in quals:
                scores = [ord(c) - 33 for c in q]
                qual_after_sum += sum(scores)
                after_base_count += len(scores)
            o1.write(f"{rec1[0]}{seqs[0]}\n{rec1[2]}{quals[0]}\n")
            o2.write(f"{rec2[0]}{seqs[1]}\n{rec2[2]}{quals[1]}\n")

    report = TrimReport(
        input_file=f"{input_r1} + {input_r2}",
        output_file=f"{output_r1} + {output_r2}",
        total_reads=total * 2,
        reads_trimmed=trimmed,
        reads_removed=(total - kept) * 2,
        avg_quality_before=qual_before_sum / base_count if base_count > 0 else 0,
        avg_quality_after=qual_after_sum / after_base_count if after_base_count > 0 else 0,
        adapter_trimmed=adapter_hits,
        engine="builtin",
        message=(f"Built-in lockstep paired trimmer: {kept}/{total} pairs kept "
                 f"(whole pairs dropped to preserve mate sync)"),
    )
    return report


def analyze_fastq_quality(filepath: str, max_reads: int = 100000) -> Dict[str, Any]:
    """Pooled quality statistics over at most ``max_reads`` records.

    Every number is per base cycle across those records — there is no per-cycle
    vector in the result, which is what a cycle profile would need.
    """
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    if isinstance(max_reads, bool) or not isinstance(max_reads, int) or max_reads < 1:
        return {"error": f"max_reads must be a positive integer, got {max_reads!r}."}

    qualities = []
    lengths = []
    count = 0

    try:
        with open(filepath, encoding="utf-8") as f:
            while count < max_reads:
                record = _read_record(f, filepath)
                if record is None:
                    break
                count += 1
                seq, qual = record[1], record[3]
                qualities.extend(ord(c) - 33 for c in qual)
                lengths.append(len(seq))
    except ValueError as exc:
        return {"error": str(exc)}

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
