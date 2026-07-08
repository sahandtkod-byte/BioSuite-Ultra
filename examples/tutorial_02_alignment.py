"""
BioSuite Tutorial 2: Pairwise Sequence Alignment

This notebook demonstrates alignment algorithms:
- Needleman-Wunsch (global alignment)
- Smith-Waterman (local alignment)
- Alignment visualization
"""
# %%
# ## Setup

import sys
sys.path.insert(0, '..')

from biosuite.core.alignment import needleman_wunsch, smith_waterman
from biosuite.plotting.plot_api import scatter

print("BioSuite Alignment Tutorial")
print("=" * 50)

# %%
# ## 1. Needleman-Wunsch (Global Alignment)

# Two similar sequences
seq1 = "AGTACGCA"
seq2 = "TATGC"

aligned1, aligned2, score = needleman_wunsch(seq1, seq2)
print("Needman-Wunsch Global Alignment")
print(f"Seq1:      {seq1}")
print(f"Seq2:      {seq2}")
print(f"Aligned 1: {aligned1}")
print(f"Aligned 2: {aligned2}")
print(f"Score:     {score}")

# %%
# ## 2. Smith-Waterman (Local Alignment)

# Find the best matching region
seq1 = "GGATCGATCGATCG"
seq2 = "ATCGATCG"

aligned1, aligned2, score = smith_waterman(seq1, seq2)
print("\nSmith-Waterman Local Alignment")
print(f"Seq1:      {seq1}")
print(f"Seq2:      {seq2}")
print(f"Aligned 1: {aligned1}")
print(f"Aligned 2: {aligned2}")
print(f"Score:     {score}")

# %%
# ## 3. Compare Alignment Scores

# Test with different sequences
test_pairs = [
    ("ATCGATCG", "ATCGATCG", "Identical"),
    ("ATCGATCG", "ATCGATCC", "1 mismatch"),
    ("ATCGATCG", "ATCG", "Deletion"),
    ("ATCGATCG", "ATCGXXXX", "Gap insertion"),
    ("ATCGATCG", "GCTAGCTA", "Different"),
]

print("\nAlignment Score Comparison")
print(f"{'Pair':<20} {'NW Score':>10} {'SW Score':>10}")
print("-" * 45)

for seq1, seq2, desc in test_pairs:
    _, _, nw_score = needleman_wunsch(seq1, seq2)
    _, _, sw_score = smith_waterman(seq1, seq2)
    print(f"{desc:<20} {nw_score:>10} {sw_score:>10}")

# %%
# ## 4. Effect of Gap Penalty

seq1 = "AGTACGCA"
seq2 = "TATGC"

gap_penalties = [-1, -2, -3, -4, -5]
scores = []

for gap in gap_penalties:
    _, _, score = needleman_wunsch(seq1, seq2, gap=gap)
    scores.append(score)
    print(f"Gap penalty {gap}: Score = {score}")

# %%
# ## 5. Visualize Score Distribution

import numpy as np

# Generate random sequences and align
np.random.seed(42)
n_pairs = 50
scores = []

for _ in range(n_pairs):
    len1 = np.random.randint(10, 50)
    len2 = np.random.randint(10, 50)
    s1 = ''.join(np.random.choice(list('ACGT'), len1))
    s2 = ''.join(np.random.choice(list('ACGT'), len2))
    _, _, score = needleman_wunsch(s1, s2)
    scores.append(score)

# Plot score distribution
fig = scatter(
    x=list(range(n_pairs)),
    y=scores,
    title="Alignment Scores for Random Sequence Pairs",
    xlabel="Pair Index",
    ylabel="Alignment Score",
    show_regression=False
)
print(f"\nScore statistics:")
print(f"  Mean: {np.mean(scores):.2f}")
print(f"  Std:  {np.std(scores):.2f}")
print(f"  Min:  {min(scores)}")
print(f"  Max:  {max(scores)}")

# %%
# ## Summary
#
# This tutorial covered:
# 1. Needleman-Wunsch global alignment
# 2. Smith-Waterman local alignment
# 3. Score comparison
# 4. Gap penalty effects
# 5. Score distribution visualization
#
# Next: Tutorial 3 - Differential Expression Analysis
