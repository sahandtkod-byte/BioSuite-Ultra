"""
Sequence analysis: parsing, composition, and manipulation.

Provides I/O for standard bioinformatics file formats (FASTA, FASTQ, GenBank)
and core sequence operations including GC content calculation, reverse
complement, codon translation, and base composition statistics.

These functions operate on plain strings and do not require Biopython
(except for GenBank parsing).
"""
import os
import numpy as np

try:
    from Bio import SeqIO
    HAS_BIO = True
except ImportError:
    HAS_BIO = False


def read_fasta(filepath: str) -> list[tuple[str, str]] | None:
    """Parse a FASTA file into (header, sequence) tuples.

    Handles multi-line sequences and multiple records. The header
    includes everything after the '>' up to the first whitespace.

    Args:
        filepath: Path to a .fasta or .fa file.

    Returns:
        List of (header, sequence) tuples, or None if file not found/error.
    """
    sequences = []
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            name = None
            seq = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if name:
                        sequences.append((name, ''.join(seq)))
                    name = line[1:].strip()
                    seq = []
                else:
                    seq.append(line)
            if name:
                sequences.append((name, ''.join(seq)))
        return sequences
    except Exception as e:
        print(f"Error reading FASTA: {e}")
        return None


def read_fastq(filepath: str) -> list[tuple[str, str, str]] | None:
    """Parse a FASTQ file into (name, sequence, quality) tuples.

    Reads line-by-line (memory efficient for large files). Each FASTQ
    record has 4 lines: header (@), sequence, +, quality string.

    Quality scores are Phred+33 encoded (Sanger/Illumina 1.8+).
    To convert to numeric: score = ord(char) - 33.

    Args:
        filepath: Path to a .fastq or .fq file.

    Returns:
        List of (name, sequence, quality) tuples, or None on error.
    """
    sequences = []
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            while True:
                header = f.readline()
                if not header:
                    break
                header = header.strip()
                if not header:
                    continue
                seq = f.readline().strip()
                f.readline()  # + line
                qual = f.readline().strip()
                name = header[1:] if header.startswith('@') else header
                sequences.append((name, seq, qual))
        return sequences
    except Exception as e:
        print(f"Error reading FASTQ: {e}")
        return None


def read_genbank(filepath: str) -> list[tuple[str, str, list]] | None:
    """Parse a GenBank file into (id, sequence, features) tuples.

    Requires Biopython. Extracts sequence and all annotated features
    (CDS, gene, mRNA, etc.) with their types, locations, and qualifiers.

    Args:
        filepath: Path to a .gb or .genbank file.

    Returns:
        List of (record_id, sequence_string, features_list) tuples,
        or None if Biopython not installed or file not found.
    """
    if not HAS_BIO:
        print("Biopython not installed. Cannot read GenBank.")
        return None
    if not os.path.exists(filepath):
        return None
    try:
        records = []
        for record in SeqIO.parse(filepath, "genbank"):
            seq = str(record.seq)
            features = []
            for feat in record.features:
                features.append({
                    'type': feat.type,
                    'location': str(feat.location),
                    'qualifiers': feat.qualifiers
                })
            records.append((record.id, seq, features))
        return records
    except Exception as e:
        print(f"Error reading GenBank: {e}")
        return None


def gc_content(seq: str) -> float:
    """Calculate GC content as a percentage.

    GC content is the proportion of guanine (G) and cytosine (C) bases
    in a DNA sequence. It's a fundamental property used in:
    - Primer design (higher GC = higher melting temperature)
    - Identifying isochores in genomes
    - Taxonomic classification of organisms

    Args:
        seq: Nucleotide sequence string (case-insensitive).

    Returns:
        GC percentage (0.0 to 100.0). Returns 0.0 for empty sequences.
    """
    if not seq:
        return 0.0
    seq = seq.upper()
    g = seq.count('G')
    c = seq.count('C')
    return (g + c) / len(seq) * 100.0


def reverse_complement(seq: str) -> str:
    """Compute the reverse complement of a DNA sequence.

    Each base is replaced by its complement (A↔T, C↔G, N→N) and the
    result is reversed. Essential for:
    - Reading sequences from the opposite strand
    - Designing probes that bind to the complementary strand
    - Understanding gene orientation in genomic context

    Args:
        seq: DNA sequence string (case preserved in output).

    Returns:
        Reverse complemented sequence string.
    """
    comp = {
        'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
        'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'
    }
    return ''.join(comp.get(base, base) for base in reversed(seq))


def translate(seq: str, frame: int = 1, table: int = 1) -> str:
    """Translate a nucleotide sequence to protein using the standard genetic code.

    Reads codons (triplets) starting from the specified reading frame and
    maps each to its amino acid. Supports both positive and negative frames
    (negative frames reverse-complement first).

    Stop codons (TAA, TAG, TGA) are represented as '*'. Unknown codons
    (containing N or other ambiguous bases) are represented as 'X'.

    Args:
        seq: Nucleotide sequence string.
        frame: Reading frame (1, 2, 3 for forward; -1, -2, -3 for reverse).
        table: Genetic code table (currently only table 1 / standard is used).

    Returns:
        Translated protein string using one-letter amino acid codes.
    """
    genetic_code = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }
    if frame < 0:
        seq = reverse_complement(seq)
        frame = -frame
    start = frame - 1
    protein = []
    for i in range(start, len(seq) - 2, 3):
        codon = seq[i:i + 3].upper()
        protein.append(genetic_code.get(codon, 'X'))
    return ''.join(protein)


def sequence_stats(seq: str) -> dict:
    """Compute base composition statistics for a nucleotide sequence.

    Returns counts and percentages for each nucleotide (A, T, G, C, N)
    plus AT% and GC% content. Useful for quality assessment and
    compositional analysis of genomic data.

    Args:
        seq: Nucleotide sequence string (case-insensitive).

    Returns:
        Dict with keys: length, A, T, G, C, N, AT (%), GC (%), other (%).
        Empty sequence returns all zeros.
    """
    if not seq:
        return {'length': 0, 'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0,
                'AT': 0.0, 'GC': 0.0, 'other': 0.0}
    seq = seq.upper()
    length = len(seq)
    a = seq.count('A')
    t = seq.count('T')
    g = seq.count('G')
    c = seq.count('C')
    n = seq.count('N')
    return {
        'length': length,
        'A': a, 'T': t, 'G': g, 'C': c, 'N': n,
        'AT': (a + t) / length * 100,
        'GC': (g + c) / length * 100,
        'other': (length - a - t - g - c - n) / length * 100
    }


def quality_stats(qual_string: str) -> dict:
    """Compute Phred quality score statistics from a FASTQ quality string.

    Converts ASCII-encoded quality scores to numeric Phred scores using
    the Sanger/Illumina 1.8+ encoding (Phred+33). Higher scores indicate
    higher base-calling confidence.

    Quality score interpretation:
    - Q10 = 90% accuracy (1 in 10 error)
    - Q20 = 99% accuracy (1 in 100 error)
    - Q30 = 99.9% accuracy (1 in 1000 error)
    - Q40 = 99.99% accuracy (1 in 10000 error)

    Args:
        qual_string: ASCII quality string from FASTQ (Phred+33 encoded).

    Returns:
        Dict with: mean, min, max quality scores, and per-position scores.
    """
    scores = [ord(ch) - 33 for ch in qual_string]
    return {
        'mean': float(np.mean(scores)),
        'min': int(np.min(scores)) if scores else 0,
        'max': int(np.max(scores)) if scores else 0,
        'positions': list(range(1, len(scores) + 1)),
        'scores': scores
    }
