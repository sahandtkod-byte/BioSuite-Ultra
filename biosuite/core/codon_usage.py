"""
Codon usage bias analysis and k-mer counting.
"""
import numpy as np
from collections import Counter
from dataclasses import dataclass

from .utils import GENETIC_CODE as CODON_TABLE


def codon_usage_table(sequence, frame=1):
    seq = sequence.upper()
    if frame < 0:
        from .sequence import reverse_complement
        seq = reverse_complement(seq)
        frame = -frame
    start = frame - 1
    codons = [seq[i:i+3] for i in range(start, len(seq)-2, 3)]
    counts = Counter(c for c in codons if len(c) == 3 and c in CODON_TABLE)
    total = sum(counts.values()) or 1
    usage = {c: round(n/total*100, 2) for c, n in sorted(counts.items())}
    aa_counts = Counter(CODON_TABLE.get(c, '?') for c in codons if len(c) == 3)
    codon_per_aa = {}
    for codon, count in counts.items():
        aa = CODON_TABLE[codon]
        if aa not in codon_per_aa:
            codon_per_aa[aa] = {}
        codon_per_aa[aa][codon] = round(count/total*100, 2)
    return {'codon_usage': usage, 'total_codons': total, 'amino_acids': dict(aa_counts), 'codon_per_aa': codon_per_aa}


def kmer_composition(sequence, k=3):
    seq = sequence.upper()
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    counts = Counter(kmers)
    total = sum(counts.values()) or 1
    return {kmer: {'count': n, 'frequency': round(n/total, 6)} for kmer, n in counts.most_common()}


def sequence_complexity(sequence, window=20):
    seq = sequence.upper()
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
            'is_low_complexity': avg < 0.3}


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
