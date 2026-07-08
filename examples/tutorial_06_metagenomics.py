"""
BioSuite Tutorial 6: Metagenomics Analysis

This notebook demonstrates metagenomics analysis:
- Taxonomic classification
- Diversity metrics
- 16S rRNA analysis
"""
# %%
# ## Setup

import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
from biosuite.core.metagenomics import (
    classify_16s_rna, format_16s_report,
    shannon_entropy, simpson_index, chao1_estimator,
    bray_curtis_distance, compute_alpha_diversity
)

print("BioSuite Metagenomics Tutorial")
print("=" * 50)

# %%
# ## 1. 16S rRNA Classification

# Sample 16S sequences (simplified)
sequences = [
    ("Ecoli_1", "TGGAGGAAGGTGGGGACGACGTCAGTATCGAATCTTGGATCAGGATCACCTCCGGA"),
    ("Staph_1", "AGCCATGCAGCACCTGTCTCAGCTTCCCGAAGGCACTATACGTAGATCGAAAGTTGAT"),
    ("Bacillus_1", "TGGAGAGTTTGATCATGGCTCAGATTGAACGCTGGCGGCAACCCTGATACAGGAT"),
    ("Lacto_1", "TGCGGTCGACCGTCTGGAAAGTCACCTTCTTTCCGGATCGAAAGTTGATGGCTCAT"),
]

result = classify_16s_rna(sequences)
print(format_16s_report(result))

# %%
# ## 2. Alpha Diversity Metrics

# Simulated abundance data
abundance = np.array([100, 50, 30, 20, 10, 5, 3, 2, 1])

print("\nAlpha Diversity Metrics:")
print(f"  Shannon entropy: {shannon_entropy(abundance):.4f}")
print(f"  Simpson index: {simpson_index(abundance):.4f}")
print(f"  Chao1 estimator: {chao1_estimator(abundance):.1f}")

# %%
# ## 3. Compare Diversity Across Samples

# Simulate multiple samples
np.random.seed(42)
samples = {
    'Healthy': np.random.poisson(50, 20),
    'Disease_A': np.random.poisson(30, 20),
    'Disease_B': np.random.poisson(20, 20),
}

print("\nDiversity comparison:")
print(f"{'Sample':<15} {'Shannon':>10} {'Simpson':>10} {'Richness':>10}")
print("-" * 50)

for name, counts in samples.items():
    shannon = shannon_entropy(counts)
    simpson = simpson_index(counts)
    richness = np.sum(counts > 0)
    print(f"{name:<15} {shannon:>10.4f} {simpson:>10.4f} {richness:>10}")

# %%
# ## 4. Beta Diversity (Bray-Curtis Distance)

sample1 = np.array([100, 50, 30, 20])
sample2 = np.array([80, 60, 25, 25])
sample3 = np.array([10, 10, 10, 10])

print("\nBeta Diversity (Bray-Curtis distance):")
print(f"  Sample1 vs Sample2: {bray_curtis_distance(sample1, sample2):.4f}")
print(f"  Sample1 vs Sample3: {bray_curtis_distance(sample1, sample3):.4f}")
print(f"  Sample2 vs Sample3: {bray_curtis_distance(sample2, sample3):.4f}")

# %%
# ## 5. Abundance Table

abundance_data = pd.DataFrame({
    'taxon': ['E. coli', 'S. aureus', 'B. subtilis', 'L. acidophilus', 'Other'],
    'Healthy': [40, 25, 20, 10, 5],
    'Disease_A': [60, 10, 15, 10, 5],
    'Disease_B': [80, 5, 5, 5, 5],
})

print("\nAbundance table:")
print(abundance_data.to_string(index=False))

# %%
# ## Summary
#
# This tutorial covered:
# 1. 16S rRNA taxonomic classification
# 2. Alpha diversity (Shannon, Simpson, Chao1)
# 3. Comparing diversity across samples
# 4. Beta diversity (Bray-Curtis distance)
# 5. Abundance table analysis
#
# Next: Tutorial 7 - Machine Learning for Biology
