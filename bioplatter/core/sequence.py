"""Sequence analysis: FASTA, FASTQ, GenBank, GC%, quality, etc."""
import os
import numpy as np
import re

# Try to import Biopython for GenBank parsing
try:
    from Bio import SeqIO
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

def read_fasta(filepath):
    """Read FASTA file (name, sequence)."""
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

def read_fastq(filepath):
    """Read FASTQ file (name, sequence, quality)."""
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

def read_genbank(filepath):
    """Read GenBank file and return list of (name, sequence, features)."""
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

def gc_content(seq):
    if not seq: return 0.0
    seq = seq.upper()
    g = seq.count('G')
    c = seq.count('C')
    return (g + c) / len(seq) * 100.0

def reverse_complement(seq):
    comp = {'A':'T','T':'A','C':'G','G':'C','N':'N','a':'t','t':'a','c':'g','g':'c','n':'n'}
    return ''.join(comp.get(base, base) for base in reversed(seq))

def translate(seq, frame=1, table=1):
    """Simple translation using standard code."""
    genetic_code = {
        'TTT':'F','TTC':'F','TTA':'L','TTG':'L',
        'CTT':'L','CTC':'L','CTA':'L','CTG':'L',
        'ATT':'I','ATC':'I','ATA':'I','ATG':'M',
        'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
        'TCT':'S','TCC':'S','TCA':'S','TCG':'S',
        'CCT':'P','CCC':'P','CCA':'P','CCG':'P',
        'ACT':'T','ACC':'T','ACA':'T','ACG':'T',
        'GCT':'A','GCC':'A','GCA':'A','GCG':'A',
        'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*',
        'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
        'AAT':'N','AAC':'N','AAA':'K','AAG':'K',
        'GAT':'D','GAC':'D','GAA':'E','GAG':'E',
        'TGT':'C','TGC':'C','TGA':'*','TGG':'W',
        'CGT':'R','CGC':'R','CGA':'R','CGG':'R',
        'AGT':'S','AGC':'S','AGA':'R','AGG':'R',
        'GGT':'G','GGC':'G','GGA':'G','GGG':'G'
    }
    if frame < 0:
        seq = reverse_complement(seq)
        frame = -frame
    start = frame - 1
    protein = []
    for i in range(start, len(seq) - 2, 3):
        codon = seq[i:i+3].upper()
        protein.append(genetic_code.get(codon, 'X'))
    return ''.join(protein)

def sequence_stats(seq):
    """Return dict with length, counts, percentages."""
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
        'AT': (a+t)/length*100,
        'GC': (g+c)/length*100,
        'other': (length - a - t - g - c - n)/length*100
    }

def quality_stats(qual_string):
    """Compute mean quality score per position (Phred+33)."""
    scores = [ord(ch) - 33 for ch in qual_string]
    return {
        'mean': np.mean(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'positions': list(range(1, len(scores)+1)),
        'scores': scores
    }