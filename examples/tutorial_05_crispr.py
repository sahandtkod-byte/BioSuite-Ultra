"""
BioSuite Tutorial 5: CRISPR Guide RNA Design

This notebook demonstrates CRISPR guide RNA design:
- PAM site finding
- Guide scoring
- Multiple PAM types
"""
# %%
# ## Setup

import sys
sys.path.insert(0, '..')

from biosuite.core.crispr import design_guides, format_crispr_report, PAM_PATTERNS

print("BioSuite CRISPR Guide RNA Design Tutorial")
print("=" * 50)

# %%
# ## 1. Design Guides for SpCas9

target_seq = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGNGGATCGATCGATCGATCG"

result = design_guides(target_seq, pam_type='SpCas9', guide_length=20, max_guides=10)
print(format_crispr_report(result))

# %%
# ## 2. Compare Different PAM Types

print("\nComparing PAM types:")
print(f"{'PAM Type':<15} {'Pattern':<10} {'Guides Found':>12}")
print("-" * 40)

for pam_type, (pattern, _) in PAM_PATTERNS.items():
    result = design_guides(target_seq, pam_type=pam_type, max_guides=50)
    print(f"{pam_type:<15} {pattern:<10} {result.num_guides:>12}")

# %%
# ## 3. Guide Scoring Details

result = design_guides(target_seq, pam_type='SpCas9', max_guides=5)

print("\nTop 5 guides with detailed scores:")
print(f"{'#':<4} {'Guide':<24} {'PAM':<6} {'Strand':<7} {'Score':>6} {'GC%':>6}")
print("-" * 60)

for i, guide in enumerate(result.guides[:5]):
    print(f"{i+1:<4} {guide.sequence:<24} {guide.pam:<6} {guide.strand:<7} "
          f"{guide.score:>6.3f} {guide.gc_content:>5.1f}%")

# %%
# ## 4. Score Breakdown

if result.guides:
    guide = result.guides[0]
    print(f"\nScore breakdown for top guide:")
    print(f"  Sequence: {guide.sequence}")
    print(f"  GC content: {guide.gc_content:.1f}%")
    print(f"  On-target score: {guide.on_target_score:.3f}")
    print(f"  Off-target count: {guide.off_target_count}")

# %%
# ## 5. Design for Long Sequence

long_seq = "ATCGATCGATCG" * 100  # 1200 bp

result_long = design_guides(long_seq, pam_type='SpCas9', max_guides=20)
print(f"\nLong sequence ({len(long_seq)} bp):")
print(f"  Guides found: {result_long.num_guides}")
print(f"  Best score: {result_long.guides[0].score:.3f}" if result_long.guides else "  No guides found")

# %%
# ## Summary
#
# This tutorial covered:
# 1. Basic guide RNA design for SpCas9
# 2. Comparing different PAM types
# 3. Guide scoring details
# 4. Score breakdown
# 5. Designing for long sequences
#
# Next: Tutorial 6 - Metagenomics Analysis
