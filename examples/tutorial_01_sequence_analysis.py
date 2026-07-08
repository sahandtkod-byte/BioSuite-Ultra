"""
BioSuite Tutorial 1: Sequence Analysis Fundamentals

This notebook demonstrates basic sequence analysis using BioSuite:
- Reading FASTA/FASTQ files
- Computing GC content
- Reverse complement
- Translation
- Sequence statistics
"""
# %%
# ## Setup

import sys
sys.path.insert(0, '..')

from biosuite.core.sequence import (
    read_fasta, read_fastq, gc_content, reverse_complement,
    translate, sequence_stats, quality_stats
)
from biosuite.plotting.plot_api import barplot

print("BioSuite Sequence Analysis Tutorial")
print("=" * 50)

# %%
# ## 1. GC Content Calculation

# Simple sequence
seq1 = "ATCGATCGATCGATCG"
gc = gc_content(seq1)
print(f"Sequence: {seq1}")
print(f"GC content: {gc:.2f}%")

# Real-world example: E. coli genome has ~50.8% GC
ecoli_seq = "ATGAAACGATTAGCGGCGGTACTGGAGGTC" * 100
print(f"\nE. coli-like sequence ({len(ecoli_seq)} bp)")
print(f"GC content: {gc_content(ecoli_seq):.2f}%")

# %%
# ## 2. Reverse Complement

dna = "ATCGATCG"
rc = reverse_complement(dna)
print(f"Original:  {dna}")
print(f"RevComp:   {rc}")
print(f"RevComp²:  {reverse_complement(rc)}")  # Should return original

# %%
# ## 3. Translation to Protein

# Start with ATG (Methionine)
coding_seq = "ATGAAATTTTGGGACTAA"
protein = translate(coding_seq)
print(f"DNA:    {coding_seq}")
print(f"Protein: {protein}")
print(f"Length:  {len(protein)} amino acids")

# Show all 6 reading frames
print("\nAll 6 reading frames:")
for frame in [1, 2, 3, -1, -2, -3]:
    prot = translate(coding_seq, frame=frame)
    print(f"  Frame {frame:+d}: {prot}")

# %%
# ## 4. Sequence Statistics

seq = "ATCGATCGATCGATCGNATCGATCG"
stats = sequence_stats(seq)
print(f"Sequence: {seq}")
print(f"Length:   {stats['length']}")
print(f"A: {stats['A']}, T: {stats['T']}, G: {stats['G']}, C: {stats['C']}, N: {stats['N']}")
print(f"AT%: {stats['AT']:.1f}%, GC%: {stats['GC']:.1f}%")

# %%
# ## 5. Quality Statistics (FASTQ)

qual_string = "IIIIIIIIIIIIIIIIIIII"  # High quality (Q40)
stats = quality_stats(qual_string)
print(f"Quality string: {qual_string}")
print(f"Mean quality: {stats['mean']:.1f}")
print(f"Min: {stats['min']}, Max: {stats['max']}")

# Low quality
qual_low = "!!!!!!!!!5555555555"
stats_low = quality_stats(qual_low)
print(f"\nLow quality string: {qual_low}")
print(f"Mean quality: {stats_low['mean']:.1f}")

# %%
# ## 6. Visualize Base Composition

import numpy as np

# Generate random sequences with different GC content
gc_contents = [20, 30, 40, 50, 60, 70, 80]
measured_gc = []

for target_gc in gc_contents:
    np.random.seed(42)
    n = 10000
    # Create sequence with target GC content
    gc_bases = int(n * target_gc / 100)
    at_bases = n - gc_bases
    seq = 'G' * (gc_bases // 2) + 'C' * (gc_bases // 2) + \
          'A' * (at_bases // 2) + 'T' * (at_bases // 2)
    # Shuffle
    seq_list = list(seq)
    np.random.shuffle(seq_list)
    seq = ''.join(seq_list)
    measured_gc.append(gc_content(seq))

# Plot
fig = barplot(
    categories=[f"{g}%" for g in gc_contents],
    values=measured_gc,
    title="GC Content Verification",
    ylabel="Measured GC%"
)
print("Base composition verification complete!")

# %%
# ## Summary
#
# This tutorial covered:
# 1. GC content calculation
# 2. Reverse complement
# 3. Translation (all 6 frames)
# 4. Sequence statistics
# 5. Quality score analysis
# 6. Base composition visualization
#
# Next: Tutorial 2 - Pairwise Alignment
