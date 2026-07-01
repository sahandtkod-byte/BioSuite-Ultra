"""
Next Generation Sequencing utilities: BAM/SAM, coverage, VCF.
"""
import os
import numpy as np
import pandas as pd

try:
    import pysam
    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False

def read_bam_header(bam_path):
    """Return header dictionary from BAM file."""
    if not HAS_PYSAM:
        print("pysam not installed. Install with: pip install pysam")
        return None
    if not os.path.exists(bam_path):
        print(f"BAM file not found: {bam_path}")
        return None
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            return bam.header
    except Exception as e:
        print(f"Error reading BAM: {e}")
        return None

def compute_coverage(bam_path, region=None):
    """Compute per-base coverage for a given region."""
    if not HAS_PYSAM:
        print("pysam required for coverage.")
        return None
    if not os.path.exists(bam_path):
        print(f"BAM file not found: {bam_path}")
        return None
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            if region:
                return np.sum(bam.count_coverage(region=region), axis=0)
            else:
                print("Whole genome coverage not implemented; use region.")
                return None
    except Exception as e:
        print(f"Coverage error: {e}")
        return None

def read_vcf(vcf_path):
    """Read VCF file and return DataFrame with CHROM, POS, REF, ALT, QUAL, INFO."""
    if not os.path.exists(vcf_path):
        print(f"VCF file not found: {vcf_path}")
        return None
    columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
    try:
        data = []
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 8:
                    data.append(parts[:8])
        if not data:
            print("No variant data found in VCF file.")
            return None
        df = pd.DataFrame(data, columns=columns)
        df['POS'] = df['POS'].astype(int)
        df['QUAL'] = pd.to_numeric(df['QUAL'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error reading VCF: {e}")
        return None

def manhattan_from_vcf(vcf_df, pvalue_col='QUAL', threshold=5e-8):
    """Convert VCF DataFrame to Manhattan plot data (chrom, pos, -log10(p))."""
    df = vcf_df.copy()
    df['chrom_num'] = df['CHROM'].str.replace('chr', '', case=False)
    df = df.sort_values(['chrom_num', 'POS'])
    pvals = pd.to_numeric(df['QUAL'], errors='coerce').fillna(0).clip(lower=0)
    df['neg_log10'] = -np.log10(pvals + 1e-300)
    return df[['CHROM', 'POS', 'neg_log10']]
