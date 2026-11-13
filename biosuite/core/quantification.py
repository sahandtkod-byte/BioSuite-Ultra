"""
RNA-seq transcript quantification with dual-mode execution.

Uses Salmon/Kallisto if installed, otherwise falls back to a pure Python
k-mer based pseudo-alignment quantifier.
"""
import json
import os
import subprocess
import tempfile
import warnings

from .utils import has_tool, secure_temp_path, smart_open, wrote_output
from .log import get_logger
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = get_logger(__name__)


@dataclass
class QuantResult:
    tool: str
    sample_name: str
    num_transcripts: int = 0
    num_mapped_reads: int = 0
    # None means the engine gave no total read count, so no rate could be computed.
    # It used to be 0.0 there, which reported a successful Salmon run as "0.0%".
    mapping_rate: float | None = 0.0
    abundance_file: str = ""
    tpm_values: list | None = None
    num_reads_values: list | None = None
    transcript_ids: list | None = None
    engine: str = "builtin"
    message: str = ""

    def to_dataframe(self):
        if self.transcript_ids is None:
            return pd.DataFrame()
        return pd.DataFrame({
            'transcript_id': self.transcript_ids,
            'tpm': self.tpm_values or [0] * len(self.transcript_ids),
            'num_reads': self.num_reads_values or [0] * len(self.transcript_ids),
        })


def check_quantification_tools():
    # `salmon version` is not a valid subcommand — salmon only understands
    # -v/--version, and the wrong flag made salmon report unavailable even
    # when installed.  Probe PATH first, then a --version smoke test.
    tools = {}
    for t in ('salmon', 'kallisto'):
        if not has_tool(t):
            tools[t] = False
            continue
        try:
            r = subprocess.run([t, '--version'], capture_output=True, text=True, timeout=10)
            tools[t] = r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            tools[t] = False
    return tools


# ── Pure Python K-mer Quantifier ────────────────────────────────────────────

def _build_transcript_index(transcripts, k=31):
    """Build k-mer -> set(transcript) mapping from transcript sequences.

    A k-mer that occurs multiple times inside ONE transcript must not
    multiply that transcript's votes (the old list-based index counted
    every occurrence, inflating repetitive transcripts).
    """
    index = defaultdict(set)
    lengths = {}
    for name, seq in transcripts:
        seq_upper = seq.upper()
        lengths[name] = len(seq_upper)
        for i in range(len(seq_upper) - k + 1):
            kmer = seq_upper[i:i + k]
            if 'N' not in kmer:
                index[kmer].add(name)
    return {kmer: sorted(ids) for kmer, ids in index.items()}, lengths


def _pseudo_align_read(read_seq, index, k=31, min_hits=2):
    """Find best matching transcript for a read using k-mer voting."""
    read_upper = read_seq.upper()
    votes = Counter()
    for i in range(len(read_upper) - k + 1):
        kmer = read_upper[i:i + k]
        if kmer in index:
            for transcript_id in index[kmer]:
                votes[transcript_id] += 1

    if not votes:
        return None, 0

    best, count = votes.most_common(1)[0]
    if count < min_hits:
        return None, 0
    return best, count


def _builtin_quantify(reads_file, transcripts, k=31, sample_name='sample'):
    """Pure Python k-mer based quantification."""
    too_short = [name for name, seq in transcripts if len(seq) < k]
    if too_short:
        warnings.warn(
            f"{len(too_short)} transcript(s) shorter than k={k} cannot be "
            f"quantified by the built-in k-mer engine (TPM forced to 0): "
            f"{', '.join(too_short[:3])}{'...' if len(too_short) > 3 else ''}",
            stacklevel=2,
        )
    index, lengths = _build_transcript_index(transcripts, k=k)
    alt_indices = {}  # k-> index cache for reads shorter than the global k
    counts = Counter()
    total_reads = 0
    mapped_reads = 0

    # smart_open: a plain open(encoding='utf-8') died on the first byte of a
    # .fastq.gz — 0x8b, the gzip magic — with UnicodeDecodeError, although .gz is
    # the normal delivery format for sequencer output.
    with smart_open(reads_file) as f:
        rejected = []
        record_start = 0
        while True:
            header = f.readline()
            if not header:
                break
            header = header.rstrip("\n")
            if not header.strip():
                continue
            record_start += 1
            seq = (f.readline() or "").strip()
            plus = (f.readline() or "").strip()
            f.readline()

            if not header.startswith('@'):
                rejected.append(f"record {record_start}: header does not start with '@'")
                continue
            if not plus.startswith('+'):
                rejected.append(
                    f"record {record_start}: separator line is {plus[:12]!r}, expected '+'")
                continue
            if not seq:
                # The record is already fully consumed (four lines read above),
                # so an empty sequence line skips that one record.  It used to
                # break the loop, which silently discarded every read after it
                # and reported a mapping rate computed on a truncated file.
                continue
            total_reads += 1
            # Reads shorter than k previously matched NOTHING.  Clip k for
            # that read and use a lazily-built index at the shorter k.
            k_read = min(k, len(seq))
            if k_read < 8:
                continue
            if k_read == k:
                use_index = index
            else:
                if k_read not in alt_indices:
                    alt_indices[k_read], _ = _build_transcript_index(transcripts, k=k_read)
                use_index = alt_indices[k_read]
            n_kmers = len(seq) - k_read + 1
            transcript, hits = _pseudo_align_read(
                seq, use_index, k=k_read, min_hits=min(2, n_kmers))
            if transcript:
                counts[transcript] += 1
                mapped_reads += 1

    if rejected:
        warnings.warn(
            f"quantify_reads: skipped {len(rejected)} malformed FASTQ record(s) in "
            f"{reads_file}; first: {rejected[0]}", stacklevel=2)

    # Compute TPM
    transcript_ids = [name for name, _ in transcripts]
    rpk = np.zeros(len(transcript_ids))
    for i, tid in enumerate(transcript_ids):
        # No guessed length: every transcript got an entry in `lengths` above, and
        # inventing 1000 nt for a missing one would silently change its abundance.
        length_kb = lengths[tid] / 1000.0
        rpk[i] = counts.get(tid, 0) / length_kb if length_kb > 0 else 0

    scaling = rpk.sum() / 1e6 if rpk.sum() > 0 else 1
    tpm = rpk / scaling

    mapping_rate = mapped_reads / total_reads * 100 if total_reads > 0 else 0
    note = f"Using built-in k-mer quantifier ({mapped_reads}/{total_reads} reads mapped)"
    if rejected:
        note += f"; {len(rejected)} record(s) skipped as malformed FASTQ"
    if too_short:
        note += f"; {len(too_short)} transcript(s) shorter than k={k} cannot be quantified"

    return QuantResult(
        tool='builtin_kmer',
        sample_name=sample_name,
        num_transcripts=len(transcript_ids),
        num_mapped_reads=mapped_reads,
        mapping_rate=mapping_rate,
        tpm_values=tpm.tolist(),
        num_reads_values=[counts.get(tid, 0) for tid in transcript_ids],
        transcript_ids=transcript_ids,
        engine='builtin',
        message=note
    )


def _read_fasta(transcripts_file):
    """Read FASTA and return list of (id, sequence) tuples."""
    from .utils import read_fasta_simple
    return read_fasta_simple(transcripts_file)


# ── External Tool Wrappers ──────────────────────────────────────────────────

def _external_mapping_stats(run_dir, df, count_column):
    """Return (mapped read count, mapping rate or None) for an external run.

    Both abundance tables give per-transcript numbers and no total read count, and
    neither field was ever filled in here, so a *successful* Salmon or Kallisto run
    printed `Mapped reads: 0 / Mapping rate: 0.0%` in the CLI and GUI reports.
    Salmon also writes ``meta_info.json`` next to ``quant.sf`` with the real numbers;
    when that file is absent the assigned-read sum is still known while the rate is
    not, and an unknown rate is reported as ``None`` rather than as zero.
    """
    meta_path = os.path.join(run_dir, 'meta_info.json') if run_dir else None
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, encoding='utf-8') as fh:
                meta = json.load(fh)
            return (int(round(float(meta['num_mapped_reads']))),
                    float(meta['percent_mapped']))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Could not read %s (%s); falling back to the abundance "
                           "table for the read count.", meta_path, exc)
    try:
        assigned = int(round(float(df[count_column].sum())))
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("External abundance table has no usable %r column (%s).",
                       count_column, exc)
        assigned = 0
    return assigned, None


def _salmon_quant(reads_r1, reads_r2, index_dir, output_dir, sample_name, threads):
    cmd = ['salmon', 'quant', '-i', index_dir, '-l', 'A', '-r', reads_r1,
           '-o', os.path.join(output_dir, sample_name), '--threads', str(threads),
           '--validateMappings']
    if reads_r2:
        cmd.extend(['--paired', '-r', reads_r2])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            logger.error("Salmon quantification failed: exit %d, %s",
                         r.returncode, (r.stderr or '').strip()[:200])
            return None
        qf = os.path.join(output_dir, sample_name, 'quant.sf')
        if not os.path.exists(qf):
            return None
        df = pd.read_csv(qf, sep='\t')
        mapped, rate = _external_mapping_stats(os.path.join(output_dir, sample_name),
                                               df, 'NumReads')
        return QuantResult(tool='salmon', sample_name=sample_name,
                          num_transcripts=len(df), abundance_file=qf,
                          num_mapped_reads=mapped, mapping_rate=rate,
                          tpm_values=df['TPM'].tolist(),
                          num_reads_values=df['NumReads'].tolist(),
                          transcript_ids=df['Name'].tolist(),
                          engine='salmon', message="Using Salmon (external)")
    except Exception as e:
        logger.error(f"Salmon quantification failed: {e}")
        return None


def _kallisto_quant(reads_r1, reads_r2, index_file, output_dir, sample_name, threads):
    cmd = ['kallisto', 'quant', '-i', index_file, '-o', output_dir,
           '-b', '100', '--threads', str(threads)]
    if reads_r2:
        cmd.extend([reads_r1, reads_r2])
    else:
        cmd.extend(['--single', '-l', '200', '-s', '20', reads_r1])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            logger.error("Kallisto quantification failed: exit %d, %s",
                         r.returncode, (r.stderr or '').strip()[:200])
            return None
        af = os.path.join(output_dir, 'abundance.tsv')
        if not os.path.exists(af):
            return None
        df = pd.read_csv(af, sep='\t')
        # kallisto's abundance.tsv has no meta_info.json partner and no total read
        # count, so the rate stays None (= not assessed) here.
        mapped, rate = _external_mapping_stats(None, df, 'est_counts')
        return QuantResult(tool='kallisto', sample_name=sample_name,
                          num_transcripts=len(df), abundance_file=af,
                          num_mapped_reads=mapped, mapping_rate=rate,
                          tpm_values=df['tpm'].tolist(),
                          # abundance.tsv headers are
                          # target_id/length/eff_length/est_counts/tpm; the
                          # singular 'est_count' read here raised KeyError on
                          # every run, and the bare except below turned that
                          # into "kallisto unavailable" even when it had worked.
                          num_reads_values=df['est_counts'].tolist(),
                          transcript_ids=df['target_id'].tolist(),
                          engine='kallisto', message="Using Kallisto (external)")
    except Exception as e:
        logger.error(f"Kallisto quantification failed: {e}")
        return None


# ── Index Building ──────────────────────────────────────────────────────────

def build_salmon_index(transcriptome_fasta, index_dir=None, threads=1):
    # The tool is probed before anything is created: salmon's index step wants an
    # empty directory, and creating it first left an empty `salmon_idx_*` directory
    # behind for every failed build (measured: three calls, three leaked directories).
    if not has_tool('salmon'):
        logger.info("Salmon is not installed; no index will be built.")
        return None
    if index_dir is None:
        index_dir = tempfile.mkdtemp(prefix='salmon_idx_')
    os.makedirs(index_dir, exist_ok=True)
    cmd = ['salmon', 'index', '-t', transcriptome_fasta, '-i', index_dir,
           '-k', '31', '--threads', str(threads)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            logger.error("Salmon index failed: exit %d, %s", r.returncode,
                         (r.stderr or '').strip()[:200])
        return index_dir if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("Salmon index could not run: %s", exc)
        return None


def kallisto_index(transcriptome_fasta, index_file=None, threads=1):
    if not has_tool('kallisto'):
        logger.info("Kallisto is not installed; no index will be built.")
        return None
    if index_file is None:
        # Reserved only now: secure_temp_path() creates the path, so asking for it
        # before the availability check left a 0-byte .idx behind on every failure.
        index_file = secure_temp_path(suffix='.idx')
    cmd = ['kallisto', 'index', '-i', index_file, '-k', '31',
           '--threads', str(threads), transcriptome_fasta]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            logger.error("Kallisto index failed: exit %d, %s", r.returncode,
                         (r.stderr or '').strip()[:200])
        return index_file if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("Kallisto index could not run: %s", exc)
        return None


# ── Public API ──────────────────────────────────────────────────────────────

def salmon_quant(reads_r1, reads_r2=None, index_dir=None,
                 transcriptome_fasta=None, output_dir=None,
                 sample_name='sample', threads=1):
    """Quantify with Salmon, building an index from the transcriptome if asked.

    Raises:
        ValueError: when neither ``index_dir`` nor ``transcriptome_fasta`` is given.
            That used to answer the same "Salmon not available" message a missing
            installation produces, which pointed the caller at the wrong fix.
    """
    if index_dir is None and not transcriptome_fasta:
        raise ValueError(
            "salmon_quant needs a Salmon index: pass index_dir=..., or "
            "transcriptome_fasta=... so one can be built.")
    if index_dir is None and transcriptome_fasta:
        index_dir = build_salmon_index(transcriptome_fasta, threads=threads)
    if index_dir and os.path.exists(index_dir):
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='salmon_quant_')
        result = _salmon_quant(reads_r1, reads_r2, index_dir, output_dir,
                              sample_name, threads)
        if result:
            return result
    return QuantResult(tool='salmon', sample_name=sample_name,
                      message="Salmon not available. Use quantify_reads() for built-in.")


def kallisto_quant(reads_r1, reads_r2=None, index_file=None,
                   transcriptome_fasta=None, output_dir=None,
                   sample_name='sample', threads=1):
    """Quantify with Kallisto, building an index from the transcriptome if asked.

    Raises:
        ValueError: when neither ``index_file`` nor ``transcriptome_fasta`` is given.
    """
    if index_file is None and not transcriptome_fasta:
        raise ValueError(
            "kallisto_quant needs a Kallisto index: pass index_file=..., or "
            "transcriptome_fasta=... so one can be built.")
    if index_file is None:
        index_file = kallisto_index(transcriptome_fasta, threads=threads)
    # wrote_output: secure_temp_path creates the path, so exists() alone would
    # treat a failed index build as success and quantify against a 0-byte index.
    if index_file and wrote_output(index_file):
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='kallisto_quant_')
        result = _kallisto_quant(reads_r1, reads_r2, index_file, output_dir,
                                sample_name, threads)
        if result:
            return result
    return QuantResult(tool='kallisto', sample_name=sample_name,
                      message="Kallisto not available. Use quantify_reads() for built-in.")


def quantify_reads(reads_file, transcriptome_fasta, sample_name='sample', k=31):
    """Quantify RNA-seq reads against a transcriptome using built-in engine.

    Args:
        reads_file: Path to single-end FASTQ file.
        transcriptome_fasta: Path to transcriptome FASTA file.
        sample_name: Sample identifier.
        k: K-mer size (default: 31).

    Returns:
        QuantResult object with TPM values.
    """
    if not os.path.exists(reads_file):
        return QuantResult(tool='builtin', sample_name=sample_name,
                          message=f"Reads file not found: {reads_file}")
    if not os.path.exists(transcriptome_fasta):
        return QuantResult(tool='builtin', sample_name=sample_name,
                          message=f"Transcriptome not found: {transcriptome_fasta}")

    transcripts = _read_fasta(transcriptome_fasta)
    if not transcripts:
        return QuantResult(tool='builtin', sample_name=sample_name,
                          message="No transcripts found in FASTA file.")

    # Try external tools first
    tools = check_quantification_tools()
    if tools['salmon']:
        idx = build_salmon_index(transcriptome_fasta)
        if idx:
            out = tempfile.mkdtemp()
            result = _salmon_quant(reads_file, None, idx, out, sample_name, 1)
            if result:
                return result

    if tools['kallisto']:
        idx = kallisto_index(transcriptome_fasta)
        if idx:
            out = tempfile.mkdtemp()
            result = _kallisto_quant(reads_file, None, idx, out, sample_name, 1)
            if result:
                return result

    return _builtin_quantify(reads_file, transcripts, k=k, sample_name=sample_name)


def merge_quantification_results(results_list):
    """Merge per-sample TPM columns into one transcript x sample matrix.

    Raises:
        ValueError: when a result carries a TPM value that is not a number. The
            merged frame is the input to every downstream numeric operation, and a
            string or ``None`` used to be copied through unchanged — the column
            silently turned to object dtype and any later arithmetic failed far
            from the sample that produced the bad value.
    """
    if not results_list:
        return pd.DataFrame()
    frames = []
    for r in results_list:
        df = r.to_dataframe()
        if df.empty:
            continue
        bad = [(tid, val) for tid, val in zip(r.transcript_ids, df['tpm'])
               if isinstance(val, bool)
               or not isinstance(val, (int, float, np.integer, np.floating))]
        if bad:
            tid, val = bad[0]
            raise ValueError(
                f"Sample {r.sample_name!r} has a non-numeric TPM for transcript "
                f"{tid!r}: {val!r} ({type(val).__name__}).")
        df = df.rename(columns={'tpm': r.sample_name})
        df = df.set_index('transcript_id')
        frames.append(df[[r.sample_name]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).fillna(0)


def format_quant_report(result):
    # An external engine that gave no total read count has mapping_rate=None;
    # printing that as "0.0%" claimed a measurement that had not been made.
    if result.mapping_rate is None:
        rate_text = "not assessed (the engine reported no total read count)"
    else:
        rate_text = f"{result.mapping_rate:.1f}%"
    lines = [
        f"=== {result.tool.upper()} Quantification ===",
        f"Engine: {result.engine}",
        f"Sample: {result.sample_name}",
        f"Transcripts: {result.num_transcripts:,}",
        f"Mapped reads: {result.num_mapped_reads:,}",
        f"Mapping rate: {rate_text}",
    ]
    if result.message:
        lines.append(f"Note: {result.message}")
    return '\n'.join(lines)
