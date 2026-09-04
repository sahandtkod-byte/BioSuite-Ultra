"""
Codon usage bias analysis and k-mer counting.
"""
from collections import Counter

import numpy as np

from .utils import GENETIC_CODE as CODON_TABLE


def codon_usage_table(sequence, frame=1):
    """Codon frequencies for a coding sequence.

    Returns:
        dict: ``codon_usage`` (percentage per codon), ``total_codons`` (the
        number of recognised codons actually found), ``amino_acids`` and
        ``codon_per_aa``.

    Note:
        ``total_codons`` previously reported ``1`` for a sequence containing no
        recognisable codons, because the divide-by-zero guard leaked into the
        reported count.
    """
    from .sequence import validate_nucleotide_sequence
    seq = validate_nucleotide_sequence(sequence, name="sequence").upper()
    if frame not in (1, 2, 3, -1, -2, -3):
        raise ValueError(f"frame must be one of 1, 2, 3, -1, -2, -3; got {frame!r}")
    if frame < 0:
        from .sequence import reverse_complement
        seq = reverse_complement(seq)
        frame = -frame
    start = frame - 1
    codons = [seq[i:i+3] for i in range(start, len(seq)-2, 3)]
    counts = Counter(c for c in codons if len(c) == 3 and c in CODON_TABLE)
    total = sum(counts.values())
    denominator = total or 1        # only guards the division
    usage = {c: round(n/denominator*100, 2) for c, n in sorted(counts.items())}
    aa_counts = Counter(CODON_TABLE.get(c, '?') for c in codons if len(c) == 3)
    codon_per_aa = {}
    for codon, count in counts.items():
        aa = CODON_TABLE[codon]
        if aa not in codon_per_aa:
            codon_per_aa[aa] = {}
        codon_per_aa[aa][codon] = round(count/denominator*100, 2)
    return {'codon_usage': usage, 'total_codons': total, 'amino_acids': dict(aa_counts), 'codon_per_aa': codon_per_aa}


def kmer_composition(sequence, k=3):
    """k-mer counts and frequencies for a nucleotide sequence."""
    from .sequence import validate_nucleotide_sequence
    seq = validate_nucleotide_sequence(sequence, name="sequence").upper()
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    counts = Counter(kmers)
    total = sum(counts.values()) or 1
    return {kmer: {'count': n, 'frequency': round(n/total, 6)} for kmer, n in counts.most_common()}


def sequence_complexity(sequence, window=20):
    """Sliding-window linguistic complexity of a nucleotide sequence."""
    from .sequence import validate_nucleotide_sequence
    seq = validate_nucleotide_sequence(sequence, name="sequence").upper()
    if not isinstance(window, int) or isinstance(window, bool) or window < 2:
        raise ValueError(f"window must be an integer >= 2, got {window!r}")
    complexities = []
    for i in range(0, len(seq) - window + 1, window // 2):
        chunk = seq[i:i+window]
        if not chunk:
            break
        unique_kmers = len(set(chunk[j:j+2] for j in range(len(chunk)-1)))
        max_possible = min(16, len(chunk)-1)
        complexity = unique_kmers / max_possible if max_possible > 0 else 0
        complexities.append({'position': i, 'complexity': round(complexity, 4)})
    avg = np.mean([c['complexity'] for c in complexities]) if complexities else 0
    return {'regions': complexities, 'average_complexity': round(float(avg), 4),
            # plain Python bool: np.bool_ breaks json.dumps on the API path
            'is_low_complexity': bool(avg < 0.3)}


def format_codon_usage(result):
    lines = ["=== Codon Usage Table ===", f"Total codons: {result['total_codons']}", ""]
    for aa, codons in sorted(result['codon_per_aa'].items()):
        lines.append(f"  {aa}:")
        for codon, pct in sorted(codons.items(), key=lambda x: -x[1]):
            bar = '█' * int(pct / 2)
            lines.append(f"    {codon}  {pct:5.1f}%  {bar}")
    return '\n'.join(lines)


def format_kmer_composition(result, top_n=20):
    lines = [f"=== K-mer Composition (top {top_n}) ==="]
    for kmer, data in list(result.items())[:top_n]:
        bar = '█' * int(data['frequency'] * 200)
        lines.append(f"  {kmer}  {data['count']:>6}  {data['frequency']:.4f}  {bar}")
    return '\n'.join(lines)
