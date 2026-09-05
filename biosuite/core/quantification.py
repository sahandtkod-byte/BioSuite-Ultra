"""
RNA-seq transcript quantification with dual-mode execution.

Uses Salmon/Kallisto if installed, otherwise falls back to a pure Python
k-mer based pseudo-alignment quantifier.
"""
import os
import subprocess
import tempfile

from .utils import secure_temp_path, wrote_output
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QuantResult:
    tool: str
    sample_name: str
    num_transcripts: int = 0
    num_mapped_reads: int = 0
    mapping_rate: float = 0.0
    abundance_file: str = ""
    tpm_values: list = None
    num_reads_values: list = None
    transcript_ids: list = None
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
    from .utils import has_tool as _has_tool
    tools = {}
    for t in ('salmon', 'kallisto'):
        if not _has_tool(t):
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
        import warnings
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

    # Compute TPM
    transcript_ids = [name for name, _ in transcripts]
    rpk = np.zeros(len(transcript_ids))
    for i, tid in enumerate(transcript_ids):
        length_kb = lengths.get(tid, 1000) / 1000.0
        rpk[i] = counts.get(tid, 0) / length_kb if length_kb > 0 else 0

    scaling = rpk.sum() / 1e6 if rpk.sum() > 0 else 1
    tpm = rpk / scaling

    mapping_rate = mapped_reads / total_reads * 100 if total_reads > 0 else 0

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
        message=f"Using built-in k-mer quantifier ({mapped_reads}/{total_reads} reads mapped)"
    )


def _read_fasta(transcripts_file):
    """Read FASTA and return list of (id, sequence) tuples."""
    from .utils import read_fasta_simple
    return read_fasta_simple(transcripts_file)


# ── External Tool Wrappers ──────────────────────────────────────────────────

def _salmon_quant(reads_r1, reads_r2, index_dir, output_dir, sample_name, threads):
    cmd = ['salmon', 'quant', '-i', index_dir, '-l', 'A', '-r', reads_r1,
           '-o', os.path.join(output_dir, sample_name), '--threads', str(threads),
           '--validateMappings']
    if reads_r2:
        cmd.extend(['--paired', '-r', reads_r2])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            return None
        qf = os.path.join(output_dir, sample_name, 'quant.sf')
        if not os.path.exists(qf):
            return None
        df = pd.read_csv(qf, sep='\t')
        return QuantResult(tool='salmon', sample_name=sample_name,
                          num_transcripts=len(df), abundance_file=qf,
                          tpm_values=df['TPM'].tolist(),
                          num_reads_values=df['NumReads'].tolist(),
                          transcript_ids=df['Name'].tolist(),
                          engine='salmon', message="Using Salmon (external)")
    except Exception:
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
            return None
        af = os.path.join(output_dir, 'abundance.tsv')
        if not os.path.exists(af):
            return None
        df = pd.read_csv(af, sep='\t')
        return QuantResult(tool='kallisto', sample_name=sample_name,
                          num_transcripts=len(df), abundance_file=af,
                          tpm_values=df['tpm'].tolist(),
                          num_reads_values=df['est_count'].tolist(),
                          transcript_ids=df['target_id'].tolist(),
                          engine='kallisto', message="Using Kallisto (external)")
    except Exception:
        return None


# ── Index Building ──────────────────────────────────────────────────────────

def build_salmon_index(transcriptome_fasta, index_dir, threads=1):
    os.makedirs(index_dir, exist_ok=True)
    cmd = ['salmon', 'index', '-t', transcriptome_fasta, '-i', index_dir,
           '-k', '31', '--threads', str(threads)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        return index_dir if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def kallisto_index(transcriptome_fasta, index_file, threads=1):
    cmd = ['kallisto', 'index', '-i', index_file, '-k', '31',
           '--threads', str(threads), transcriptome_fasta]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        return index_file if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# ── Public API ──────────────────────────────────────────────────────────────

def salmon_quant(reads_r1, reads_r2=None, index_dir=None,
                 transcriptome_fasta=None, output_dir=None,
                 sample_name='sample', threads=1):
    if index_dir is None and transcriptome_fasta:
        index_dir = build_salmon_index(transcriptome_fasta,
                                       tempfile.mkdtemp(prefix='salmon_idx_'), threads)
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
    if index_file is None and transcriptome_fasta:
        index_file = kallisto_index(transcriptome_fasta,
                                    secure_temp_path(suffix='.idx'), threads)
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
        idx = build_salmon_index(transcriptome_fasta, tempfile.mkdtemp())
        if idx:
            out = tempfile.mkdtemp()
            result = _salmon_quant(reads_file, None, idx, out, sample_name, 1)
            if result:
                return result

    if tools['kallisto']:
        idx = kallisto_index(transcriptome_fasta, secure_temp_path(suffix='.idx'))
        if idx:
            out = tempfile.mkdtemp()
            result = _kallisto_quant(reads_file, None, idx, out, sample_name, 1)
            if result:
                return result

    return _builtin_quantify(reads_file, transcripts, k=k, sample_name=sample_name)


def merge_quantification_results(results_list):
    if not results_list:
        return pd.DataFrame()
    frames = []
    for r in results_list:
        df = r.to_dataframe()
        if df.empty:
            continue
        df = df.rename(columns={'tpm': r.sample_name})
        df = df.set_index('transcript_id')
        frames.append(df[[r.sample_name]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).fillna(0)


def format_quant_report(result):
    lines = [
        f"=== {result.tool.upper()} Quantification ===",
        f"Engine: {result.engine}",
        f"Sample: {result.sample_name}",
        f"Transcripts: {result.num_transcripts:,}",
        f"Mapped reads: {result.num_mapped_reads:,}",
        f"Mapping rate: {result.mapping_rate:.1f}%",
    ]
    if result.message:
        lines.append(f"Note: {result.message}")
    return '\n'.join(lines)
