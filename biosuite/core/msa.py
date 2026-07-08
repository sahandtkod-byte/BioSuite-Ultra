"""
Multiple Sequence Alignment with dual-mode execution.

Uses Clustal Omega/MUSCLE/MAFFT if installed, otherwise falls back to a
pure Python progressive alignment algorithm.
"""
import os
import subprocess
import tempfile
import numpy as np
from dataclasses import dataclass, field
from collections import Counter

try:
    from Bio import AlignIO, SeqIO
    HAS_BIO = True
except ImportError:
    HAS_BIO = False


@dataclass
class MSA:
    method: str
    sequences: list = field(default_factory=list)
    alignment_file: str = ""
    num_sequences: int = 0
    alignment_length: int = 0
    conservation: list = field(default_factory=list)
    engine: str = "builtin"
    message: str = ""

    @property
    def names(self):
        return [s[0] for s in self.sequences]

    @property
    def sequences_only(self):
        return [s[1] for s in self.sequences]


# ── External Tool Detection ─────────────────────────────────────────────────

from .utils import has_tool as _has_tool


def check_tools():
    return {
        'clustal_omega': _has_tool('clustalo'),
        'muscle': _has_tool('muscle'),
        'mafft': _has_tool('mafft'),
    }


def _is_nucleotide(seq):
    if not seq:
        return True
    sample = seq[:500].upper()
    nuc = set('ATCGN')
    return sum(1 for c in sample if c not in nuc) / max(len(sample), 1) < 0.1


def _write_fasta(sequences, filepath):
    with open(filepath, 'w') as f:
        for name, seq in sequences:
            f.write(f">{name}\n{seq}\n")


# ── Pure Python Progressive Alignment ───────────────────────────────────────

def _pairwise_distance(seqs, k=3):
    n = len(seqs)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _kmer_distance(seqs[i], seqs[j], k)
            mat[i][j] = mat[j][i] = d
    return mat


def _kmer_distance(s1, s2, k=3):
    if not s1 or not s2:
        return 1.0
    kmers1 = set()
    kmers2 = set()
    for i in range(len(s1) - k + 1):
        kmers1.add(s1[i:i + k])
    for i in range(len(s2) - k + 1):
        kmers2.add(s2[i:i + k])
    if not kmers1 or not kmers2:
        return 1.0
    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    return 1.0 - intersection / union if union > 0 else 1.0


def _upgma_tree(dist_matrix, n):
    clusters = {i: [i] for i in range(n)}
    merged = set()
    parent = {}
    heights = {i: 0.0 for i in range(n)}
    next_id = n
    mat = dist_matrix.copy()

    while len(clusters) - len(merged) > 1:
        min_val = float('inf')
        min_i, min_j = -1, -1
        active = [i for i in clusters if i not in merged]
        for i in active:
            for j in active:
                if i < j and mat[i][j] < min_val:
                    min_val = mat[i][j]
                    min_i, min_j = i, j

        new_cluster = clusters[min_i] + clusters[min_j]
        height = min_val / 2
        clusters[next_id] = new_cluster
        parent[min_i] = (next_id, height)
        parent[min_j] = (next_id, height)
        heights[next_id] = height

        # Create new larger matrix first
        new_mat = np.zeros((mat.shape[0] + 1, mat.shape[0] + 1))
        new_mat[:mat.shape[0], :mat.shape[0]] = mat

        for k_id in active:
            if k_id != min_i and k_id != min_j:
                d = (mat[min_i][k_id] * len(clusters[min_i]) +
                     mat[min_j][k_id] * len(clusters[min_j])) / (len(clusters[min_i]) + len(clusters[min_j]))
                new_mat[k_id, next_id] = d
                new_mat[next_id, k_id] = d

        mat = new_mat
        merged.add(min_i)
        merged.add(min_j)
        next_id += 1

    root = next_id - 1
    return clusters, parent, heights, root


def _build_guide_order(dist_matrix, seq_names):
    n = len(seq_names)
    clusters, parent, heights, root = _upgma_tree(dist_matrix, n)
    order = []

    def traverse(node):
        if node < n:
            order.append(node)
            return
        children = [c for c, (p, h) in parent.items() if p == node]
        for c in sorted(children, key=lambda x: heights.get(x, 0)):
            traverse(c)

    traverse(root)
    return order


def _align_two_profiles(seq_a, seq_b, match=1, mismatch=-1, gap=-2):
    """Align two sequences using dynamic programming (profile scoring)."""
    n, m = len(seq_a), len(seq_b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1) * gap
    dp[0, :] = np.arange(m + 1) * gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if seq_a[i-1] == seq_b[j-1] and seq_a[i-1] != '-' else mismatch
            if seq_a[i-1] == '-' or seq_b[j-1] == '-':
                s = gap
            dp[i][j] = max(dp[i-1][j-1] + s, dp[i-1][j] + gap, dp[i][j-1] + gap)

    # Traceback
    align_a, align_b = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = match if seq_a[i-1] == seq_b[j-1] and seq_a[i-1] != '-' else mismatch
            if seq_a[i-1] == '-' or seq_b[j-1] == '-':
                s = gap
            if dp[i][j] == dp[i-1][j-1] + s:
                align_a.append(seq_a[i-1])
                align_b.append(seq_b[j-1])
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i-1][j] + gap:
            align_a.append(seq_a[i-1])
            align_b.append('-')
            i -= 1
        else:
            align_a.append('-')
            align_b.append(seq_b[j-1])
            j -= 1

    return ''.join(reversed(align_a)), ''.join(reversed(align_b))


def _merge_alignments(aligned_a, aligned_b):
    """Merge two sets of aligned sequences using profile-profile scoring."""
    if not aligned_a:
        return aligned_b
    if not aligned_b:
        return aligned_a

    # Build profiles (frequency columns)
    profile_a = _make_profile(aligned_a)
    profile_b = _make_profile(aligned_b)
    la, lb = len(aligned_a[0]), len(aligned_b[0])

    # DP for profile-profile alignment
    gap_open, gap_ext = -8, -1
    dp = np.zeros((la + 1, lb + 1), dtype=np.int32)
    trace = np.zeros((la + 1, lb + 1), dtype=np.int8)

    for i in range(1, la + 1):
        dp[i][0] = dp[i-1][0] + gap_ext
        trace[i][0] = 1
    for j in range(1, lb + 1):
        dp[0][j] = dp[0][j-1] + gap_ext
        trace[0][j] = 2

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            diag = dp[i-1][j-1] + _profile_score_col(profile_a, i-1, profile_b, j-1)
            up = dp[i-1][j] + gap_ext
            left = dp[i][j-1] + gap_ext
            dp[i][j] = max(diag, up, left)
            trace[i][j] = np.argmax([diag, up, left])

    # Traceback
    cols_a, cols_b = [], []
    i, j = la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 0:
            cols_a.append(i - 1)
            cols_b.append(j - 1)
            i -= 1
            j -= 1
        elif i > 0 and trace[i][j] == 1:
            cols_a.append(i - 1)
            cols_b.append(-1)
            i -= 1
        else:
            cols_a.append(-1)
            cols_b.append(j - 1)
            j -= 1

    cols_a.reverse()
    cols_b.reverse()

    # Build merged alignment
    merged = []
    for seq in aligned_a:
        new_seq = []
        for ca in cols_a:
            new_seq.append(seq[ca] if ca >= 0 else '-')
        merged.append(''.join(new_seq))
    for seq in aligned_b:
        new_seq = []
        for cb in cols_b:
            new_seq.append(seq[cb] if cb >= 0 else '-')
        merged.append(''.join(new_seq))

    return merged


def _make_profile(aligned_seqs):
    """Build frequency profile from aligned sequences."""
    if not aligned_seqs:
        return {}
    length = len(aligned_seqs[0])
    chars = set('ACGTUNacgtun')
    profiles = []
    for i in range(length):
        col = [s[i] if i < len(s) else '-' for s in aligned_seqs]
        total = len(col)
        freq = {}
        for c in chars:
            freq[c] = col.count(c) / total
        freq['-'] = col.count('-') / total
        profiles.append(freq)
    return profiles


def _profile_score_col(profile_a, idx_a, profile_b, idx_b):
    """Score two profile columns against each other."""
    pa = profile_a[idx_a]
    pb = profile_b[idx_b]
    score = 0
    for c in set(list(pa.keys()) + list(pb.keys())):
        if c == '-':
            score += pa.get('-', 0) * pb.get('-', 0) * -2
        else:
            score += pa.get(c, 0) * pb.get(c, 0) * 1
    return int(score * 10)


def _progressive_msa(sequences):
    """Pure Python progressive multiple sequence alignment."""
    if len(sequences) <= 1:
        return sequences

    names = [s[0] for s in sequences]
    seqs = [s[1].upper() for s in sequences]
    n = len(seqs)

    if n == 2:
        a, b = _align_two_profiles(seqs[0], seqs[1])
        return [(names[0], a), (names[1], b)]

    dist_mat = _pairwise_distance(seqs)
    order = _build_guide_order(dist_mat, names)

    # Initialize alignments (one sequence each)
    aligned = {i: [seqs[i]] for i in range(n)}
    aligned_names = {i: [names[i]] for i in range(n)}

    # Progressive merge following guide tree order
    merged_already = set()
    for idx in order:
        if idx in merged_already:
            continue
        # Find closest unmerged neighbor
        best_j = -1
        best_dist = float('inf')
        for j in range(n):
            if j != idx and j not in merged_already:
                if dist_mat[idx][j] < best_dist:
                    best_dist = dist_mat[idx][j]
                    best_j = j
        if best_j < 0:
            continue

        merged_seqs = _merge_alignments(aligned[idx], aligned[best_j])
        merged_n = aligned_names[idx] + aligned_names[best_j]

        aligned[idx] = merged_seqs
        aligned_names[idx] = merged_n
        merged_already.add(best_j)

    # Collect results
    result_seqs = []
    for i in range(n):
        if i in aligned and aligned[i]:
            result_seqs.append((names[i], aligned[i][0]))

    return result_seqs


# ── External Tool Wrappers ──────────────────────────────────────────────────

def _run_clustal_omega(sequences):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        inp = f.name
    out = tempfile.mktemp(suffix='.fasta')
    try:
        _write_fasta(sequences, inp)
        bio_type = 'DNA' if _is_nucleotide(sequences[0][1]) else 'PROTEIN'
        cmd = ['clustalo', '-i', inp, '-o', out, '-t', bio_type,
               '--threads', '1', '--iterations', '1', '--force']
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and os.path.exists(out):
            return _load_bio_alignment(out, 'clustal_omega')
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        for p in [inp, out]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
    return None


def _run_muscle(sequences):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        inp = f.name
    out = tempfile.mktemp(suffix='.fasta')
    try:
        _write_fasta(sequences, inp)
        cmd = ['muscle', '-align', inp, '-output', out]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            cmd = ['muscle', '-in', inp, '-out', out]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and os.path.exists(out):
            return _load_bio_alignment(out, 'muscle')
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        for p in [inp, out]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
    return None


def _run_mafft(sequences):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        inp = f.name
    out = tempfile.mktemp(suffix='.fasta')
    try:
        _write_fasta(sequences, inp)
        cmd = ['mafft', '--auto', '--thread', '1', inp]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            with open(out, 'w') as f:
                f.write(r.stdout)
            return _load_bio_alignment(out, 'mafft')
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        for p in [inp, out]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
    return None


def _load_bio_alignment(filepath, method):
    if not HAS_BIO:
        return None
    try:
        alignment = AlignIO.read(filepath, 'fasta')
        seqs = [(rec.id, str(rec.seq)) for rec in alignment]
        conservation = compute_conservation(alignment)
        return MSA(method=method, sequences=seqs, alignment_file=filepath,
                   num_sequences=len(seqs),
                   alignment_length=alignment.get_alignment_length(),
                   conservation=conservation)
    except Exception:  # BioPython AlignIO can raise various errors
        return None


# ── Public API ──────────────────────────────────────────────────────────────

def auto_align(sequences, method='auto'):
    if not sequences or len(sequences) < 2:
        return MSA(method='none', message='Need at least 2 sequences.')

    tools = check_tools()
    if tools.get('clustal_omega'):
        result = _run_clustal_omega(sequences)
        if result:
            result.message = "Using Clustal Omega (external)"
            return result

    if tools.get('muscle'):
        result = _run_muscle(sequences)
        if result:
            result.message = "Using MUSCLE (external)"
            return result

    if tools.get('mafft'):
        result = _run_mafft(sequences)
        if result:
            result.message = "Using MAFFT (external)"
            return result

    # Pure Python fallback
    aligned_seqs = _progressive_msa(sequences)
    align_len = len(aligned_seqs[0][1]) if aligned_seqs else 0

    return MSA(
        method='builtin_progressive',
        sequences=aligned_seqs,
        num_sequences=len(aligned_seqs),
        alignment_length=align_len,
        conservation=[],
        message="Using built-in progressive alignment engine"
    )


def run_clustal_omega(sequences, **kwargs):
    return auto_align(sequences)

def run_muscle(sequences, **kwargs):
    return auto_align(sequences)

def run_mafft(sequences, **kwargs):
    return auto_align(sequences)


def compute_conservation(alignment):
    scores = []
    align_len = alignment.get_alignment_length()
    for i in range(align_len):
        column = alignment[:, i]
        chars = [c for c in column if c != '-']
        if not chars:
            scores.append(0.0)
            continue
        counts = Counter(chars)
        scores.append(counts.most_common(1)[0][1] / len(chars))
    return scores


def consensus_sequence(msa_result, threshold=0.5):
    if not msa_result or msa_result.num_sequences == 0:
        return ""
    seqs = msa_result.sequences_only
    align_len = msa_result.alignment_length
    consensus = []
    for i in range(align_len):
        col = [s[i] if i < len(s) else '-' for s in seqs]
        chars = [c for c in col if c != '-']
        if not chars:
            consensus.append('-')
            continue
        counts = Counter(chars)
        most_char, most_count = counts.most_common(1)[0]
        freq = most_count / len(chars)
        if freq >= threshold:
            consensus.append(most_char.lower() if freq < 1.0 else most_char)
        else:
            consensus.append('n')
    return ''.join(consensus)


def alignment_statistics(msa_result):
    if not msa_result:
        return {}
    conservation = msa_result.conservation
    gap_count = sum(s.count('-') for s in msa_result.sequences_only)
    total = msa_result.num_sequences * msa_result.alignment_length
    return {
        'num_sequences': msa_result.num_sequences,
        'alignment_length': msa_result.alignment_length,
        'gap_percentage': round(gap_count / total * 100, 2) if total else 0,
        'mean_conservation': round(float(np.mean(conservation)), 4) if conservation else 0,
        'highly_conserved': sum(1 for c in conservation if c >= 0.9) if conservation else 0,
        'variable': sum(1 for c in conservation if c < 0.5) if conservation else 0,
    }


def format_alignment(msa_result, max_width=80, show_conservation=True):
    if not msa_result or not msa_result.sequences:
        return "No alignment to display."
    lines = [f"Engine: {msa_result.message}", f"Method: {msa_result.method}",
             f"Sequences: {msa_result.num_sequences}",
             f"Length: {msa_result.alignment_length}", ""]
    max_name = max(len(n) for n, _ in msa_result.sequences)
    seq_width = max_width - max_name - 4
    for start in range(0, msa_result.alignment_length, seq_width):
        end = min(start + seq_width, msa_result.alignment_length)
        for name, seq in msa_result.sequences:
            lines.append(f"{name:>{max_name}}  {seq[start:end]}")
        if show_conservation and msa_result.conservation:
            bar = ''.join('|' if c >= 0.9 else ':' if c >= 0.7 else '.' if c >= 0.5 else ' '
                         for c in msa_result.conservation[start:end])
            lines.append(f"{'':>{max_name}}  {bar}")
        lines.append("")
    return '\n'.join(lines)


def read_fasta_for_msa(filepath):
    if not HAS_BIO:
        seqs = []
        name, buf = None, []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if name:
                        seqs.append((name, ''.join(buf)))
                    name = line[1:].split()[0]
                    buf = []
                elif line:
                    buf.append(line)
        if name:
            seqs.append((name, ''.join(buf)))
        return seqs
    return [(r.id, str(r.seq)) for r in SeqIO.parse(filepath, 'fasta')]
