"""
Sequence analysis: parsing, composition, and manipulation.

Provides I/O for standard bioinformatics file formats (FASTA, FASTQ, GenBank)
and core sequence operations including GC content calculation, reverse
complement, codon translation, and base composition statistics.

These functions operate on plain strings and do not require Biopython
(except for GenBank parsing).
"""
from __future__ import annotations

import os

import numpy as np

from .log import get_logger
from .utils import smart_open

logger = get_logger(__name__)


try:
    from Bio import SeqIO
    HAS_BIO = True
except ImportError:
    HAS_BIO = False


def read_fasta(filepath: str) -> list[tuple[str, str]] | None:
    """Parse a FASTA file into (header, sequence) tuples.

    Handles multi-line sequences and multiple records.  The header is the whole
    text following '>' up to the end of the line; ``utils.read_fasta_simple`` is
    the variant that yields only the first token as an identifier.

    Compressed input (``.gz``/``.bz2``) is decompressed transparently and a
    UTF-8 BOM is consumed rather than swallowing the first record.

    Args:
        filepath: Path to a .fasta or .fa file.

    Returns:
        List of (header, sequence) tuples, or None if file not found/error.
    """
    sequences = []
    if not os.path.exists(filepath):
        return None
    try:
        with smart_open(filepath) as f:
            name = None
            seq = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if name is not None:
                        sequences.append((name, ''.join(seq)))
                    name = line[1:].strip()
                    seq = []
                else:
                    seq.append(line)
            if name is not None:
                sequences.append((name, ''.join(seq)))
        return sequences
    except Exception as e:
        logger.error(f"Error reading FASTA: {e}")
        return None


def read_fastq(filepath: str) -> list[tuple[str, str, str]] | None:
    """Parse a FASTQ file into (name, sequence, quality) tuples.

    Reads record-by-record (four lines each) and rejects records that violate the
    format instead of returning them as data: a header that does not start with
    '@', a separator line that is not '+', or a quality string whose length
    differs from the sequence.  Rejected records are counted and reported in one
    warning, so a truncated final record cannot masquerade as an empty read.

    Quality scores are Phred+33 encoded (Sanger/Illumina 1.8+); there is no
    Phred+64 detection, so a '+64' file will read as implausibly high quality.
    To convert to numeric: score = ord(char) - 33.

    Args:
        filepath: Path to a .fastq or .fq file (.gz and .bz2 accepted).

    Returns:
        List of (name, sequence, quality) tuples, or None on error.
    """
    sequences = []
    if not os.path.exists(filepath):
        return None
    rejected = []
    try:
        with smart_open(filepath) as f:
            record_start = 0
            while True:
                header = f.readline()
                if not header:
                    break
                header = header.strip()
                if not header:
                    continue
                record_start += 1
                seq = f.readline().strip()
                plus = f.readline().strip()
                qual = f.readline().strip()

                if not header.startswith('@'):
                    rejected.append(f"record {record_start}: header does not start with '@'")
                    continue
                if not plus.startswith('+'):
                    rejected.append(f"record {record_start}: separator line is {plus[:12]!r}, expected '+'")
                    continue
                if len(seq) != len(qual):
                    rejected.append(
                        f"record {record_start}: sequence length {len(seq)} != "
                        f"quality length {len(qual)}")
                    continue
                sequences.append((header[1:], seq, qual))
        if rejected:
            logger.warning(
                f"read_fastq: skipped {len(rejected)} malformed record(s) in {filepath}; "
                f"first: {rejected[0]}")
        return sequences
    except Exception as e:
        logger.error(f"Error reading FASTQ: {e}")
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
        logger.warning("Biopython not installed. Cannot read GenBank.")
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
        logger.error(f"Error reading GenBank: {e}")
        return None


#: IUPAC nucleotide alphabet plus gap/mask characters that legitimately appear
#: in FASTA records.  Anything else is not a nucleotide sequence.
IUPAC_NUCLEOTIDES = frozenset("ACGTUacgtu"      # unambiguous bases + RNA uracil
                              "RYSWKMBDHVNrysWkmbdhvn"  # IUPAC ambiguity codes
                              "-.*")            # gaps / alignment padding


def validate_nucleotide_sequence(seq, *, name: str = "sequence",
                                 allow_empty: bool = True) -> str:
    """Return *seq* as a string after checking it really is nucleotides.

    Silently accepting non-nucleotide input is how a pasted FASTA header or a
    protein sequence turned into a plausible-looking GC percentage.  Callers
    that legitimately handle arbitrary text should not use this helper.

    Args:
        seq: candidate sequence.
        name: parameter name to quote in error messages.
        allow_empty: whether the empty string is acceptable.

    Returns:
        The sequence with surrounding whitespace and internal newlines removed.

    Raises:
        TypeError: if *seq* is not a string.
        ValueError: if it is empty (and ``allow_empty`` is False) or contains
            characters outside the IUPAC nucleotide alphabet.
    """
    if seq is None or not isinstance(seq, str):
        raise TypeError(
            f"{name} must be a string of nucleotides, got {type(seq).__name__}")
    cleaned = "".join(seq.split())
    if not cleaned:
        if allow_empty:
            return ""
        raise ValueError(f"{name} must not be empty")
    invalid = sorted({c for c in cleaned if c not in IUPAC_NUCLEOTIDES})
    if invalid:
        preview = "".join(invalid[:8])
        raise ValueError(
            f"{name} contains characters that are not IUPAC nucleotides: "
            f"{preview!r}. Pass a DNA/RNA sequence (a FASTA header or protein "
            f"sequence is not one).")
    return cleaned


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

        Gap characters are excluded from the denominator; ambiguity codes are
        counted as non-GC.  ``'ACGT!@#'`` used to return 28.57 % by dividing
        by seven "bases".

    Raises:
        TypeError: if *seq* is not a string.
        ValueError: if *seq* contains non-nucleotide characters.
    """
    seq = validate_nucleotide_sequence(seq, name="seq")
    if not seq:
        return 0.0
    arr = np.array(list(seq.upper()), dtype='U1')
    counted = ~np.isin(arr, np.array(['-', '.', '*'], dtype='U1'))
    total = int(counted.sum())
    if total == 0:
        return 0.0
    gc = int(((arr == 'G') | (arr == 'C')) [counted].sum())
    return float(gc / total * 100.0)


def reverse_complement(seq: str) -> str:
    """Compute the reverse complement of a DNA sequence.

    Each base is replaced by its complement (A<->T, C<->G, N->N) and the
    result is reversed. Essential for:
    - Reading sequences from the opposite strand
    - Designing probes that bind to the complementary strand
    - Understanding gene orientation in genomic context

    Args:
        seq: DNA sequence string (case preserved in output).

    Returns:
        Reverse complemented sequence string.

    Raises:
        TypeError: if *seq* is not a string.
        ValueError: if *seq* contains non-nucleotide characters.  ``'XYZ123'``
            used to come back as ``'321ZYX'``, as if it had been complemented.
    """
    seq = validate_nucleotide_sequence(seq, name="seq")
    comp = str.maketrans('ACGTUNacgtunRYSWKMBDHVryswkmbdhv',
                         'TGCAANtgcaanYRSWMKVHDByrswmkvhdb')
    return seq.translate(comp)[::-1]


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
    from .utils import GENETIC_CODE
    seq = validate_nucleotide_sequence(seq, name="seq")
    if frame not in (1, 2, 3, -1, -2, -3):
        raise ValueError(f"frame must be one of 1, 2, 3, -1, -2, -3; got {frame!r}")
    if table != 1:
        raise ValueError(
            f"only the standard genetic code (table=1) is implemented, got {table!r}")
    if frame < 0:
        seq = reverse_complement(seq)
        frame = -frame
    start = frame - 1
    upper = seq.upper()
    codons = [upper[i:i + 3] for i in range(start, len(seq) - 2, 3)]
    return ''.join(GENETIC_CODE.get(c, 'X') for c in codons)


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

        ``other`` is what is neither A/T/G/C/N: the IUPAC ambiguity codes
        (R, Y, S, W, K, M, B, D, H, V) and the gap characters ``-``, ``.``,
        ``*``. It is not a bin for text that was never a sequence.

    Raises:
        TypeError: if *seq* is not a string. ``None`` included — a missing
            sequence used to come back as a full dict of zeros, a measurement of
            nothing.
        ValueError: if *seq* contains non-nucleotide characters, the rule
            :func:`gc_content` and :func:`reverse_complement` already enforce.
    """
    seq = validate_nucleotide_sequence(seq, name="seq")
    if not seq:
        return {'length': 0, 'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0,
                'AT': 0.0, 'GC': 0.0, 'other': 0.0}
    arr = np.array(list(seq.upper()), dtype='U1')
    length = len(arr)
    # Gaps and alignment padding are not bases, so they stay out of the
    # denominator exactly as gc_content() does them. Measured before: 'AC-GT.*N'
    # answered GC 25.0 here and 40.0 from gc_content() on the same string.
    gap = np.isin(arr, np.array(['-', '.', '*'], dtype='U1'))
    counted = int((~gap).sum())
    a = int((arr == 'A').sum())
    t = int((arr == 'T').sum())
    g = int((arr == 'G').sum())
    c = int((arr == 'C').sum())
    n = int((arr == 'N').sum())
    known = a + t + g + c + n
    pct = (lambda k: k / counted * 100 if counted else 0.0)
    return {
        'length': length,
        'A': a, 'T': t, 'G': g, 'C': c, 'N': n,
        'AT': pct(a + t),
        'GC': pct(g + c),
        'other': pct(counted - known)
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

    Raises:
        TypeError: if *qual_string* is not a string. It used to die inside
            ``ord()`` with a message about an int, or answer a dict of zeros for
            ``None``.
        ValueError: if a character is outside the Phred+33 range (``'!'`` to
            ``'~'``). A space decoded as Q-1 and a tab as Q-51, and the
            interpretation table in this docstring presents every score as a
            confidence, not as garbage.
    """
    if qual_string is None or not isinstance(qual_string, str):
        raise TypeError(
            f"qual_string must be a FASTQ quality string, got "
            f"{type(qual_string).__name__}")
    if not qual_string:
        return {'mean': 0.0, 'min': 0, 'max': 0, 'positions': [], 'scores': []}
    bad = sorted({ch for ch in qual_string if not 33 <= ord(ch) <= 126})
    if bad:
        preview = ", ".join(f"{ch!r} (Q{ord(ch) - 33})" for ch in bad[:5])
        raise ValueError(
            f"qual_string contains characters outside the Phred+33 range "
            f"('!' to '~'): {preview}. Decode the FASTQ before measuring it.")
    scores = np.array([ord(ch) - 33 for ch in qual_string], dtype=np.int32)
    return {
        'mean': float(np.mean(scores)),
        'min': int(np.min(scores)),
        'max': int(np.max(scores)),
        'positions': list(range(1, len(scores) + 1)),
        'scores': scores.tolist()
    }
