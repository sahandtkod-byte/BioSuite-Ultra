"""
Variant calling from BAM/SAM files with dual-mode execution.

Pure Python variant caller as default, FreeBayes as optional faster tool.
"""
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class Variant:
    chrom: str
    pos: int
    ref: str
    alt: str
    quality: float
    depth: int
    alt_count: int
    genotype: str
    variant_type: str  # SNP, INS, DEL, MNP
    filter_status: str = "PASS"


@dataclass
class VariantReport:
    tool: str
    engine: str
    total_variants: int = 0
    snps: int = 0
    indels: int = 0
    ti_tv_ratio: float = 0.0
    variants: list = field(default_factory=list)
    output_vcf: str = ""
    message: str = ""


from .utils import has_tool as _has_tool


def check_variant_tools():
    return {'freebayes': _has_tool('freebayes'), 'bcftools': _has_tool('bcftools')}


def _read_sam(sam_file):
    reads = []
    with open(sam_file) as f:
        for line in f:
            if line.startswith('@'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 11:
                continue
            flag = int(parts[1])
            if flag & 4:
                continue
            reads.append({
                'qname': parts[0],
                'rname': parts[2],
                'pos': int(parts[3]),
                'mapq': int(parts[4]),
                'cigar': parts[5],
                'seq': parts[9],
                'qual': parts[10],
            })
    return reads


def _pileup_reads(reads, min_depth=4, min_base_quality=20):
    piles = defaultdict(lambda: defaultdict(list))
    for r in reads:
        ref_pos = r['pos']
        read_seq = r['seq']
        cigar = r['cigar']
        read_offset = 0
        for num, op in _parse_cigar(cigar):
            if op == 'M':
                for i in range(num):
                    if read_offset < len(read_seq):
                        piles[r['rname']][ref_pos + i].append(read_seq[read_offset])
                    read_offset += 1
            elif op in ('I', 'S'):
                read_offset += num
            elif op == 'D':
                ref_pos += num
    return piles


def _parse_cigar(cigar_str):
    ops = []
    num = ''
    for c in cigar_str:
        if c.isdigit():
            num += c
        else:
            if num:
                ops.append((int(num), c))
            else:
                ops.append((1, c))
            num = ''
    return ops


def _call_variants_from_pileup(piles, min_depth=4, min_allele_freq=0.25):
    variants = []
    for chrom, positions in sorted(piles.items()):
        for pos, bases in sorted(positions.items()):
            if len(bases) < min_depth:
                continue
            counts = Counter(bases)
            ref_base = counts.most_common(1)[0][0]
            total = len(bases)

            for base, count in counts.most_common():
                if base == ref_base:
                    continue
                if count / total < min_allele_freq:
                    continue

                alt = base
                vtype = 'SNP'
                if len(ref_base) > 1 or len(alt) > 1:
                    vtype = 'MNP'

                gt = '0/1' if count / total < 0.75 else '1/1'
                qual = min(99, int(count / total * 60))

                variants.append(Variant(
                    chrom=chrom, pos=pos + 1, ref=ref_base, alt=alt,
                    quality=qual, depth=total, alt_count=count,
                    genotype=gt, variant_type=vtype
                ))

    return variants


def _builtin_call_variants(sam_file, output_vcf=None, min_depth=4):
    reads = _read_sam(sam_file)
    if not reads:
        return VariantReport(tool='builtin', engine='builtin', message="No mapped reads found.")

    piles = _pileup_reads(reads, min_depth=min_depth)
    variants = _call_variants_from_pileup(piles, min_depth=min_depth)

    snps = sum(1 for v in variants if v.variant_type == 'SNP')
    indels = sum(1 for v in variants if v.variant_type in ('INS', 'DEL'))
    ti_tv = _calculate_ti_ttv(variants)

    report = VariantReport(
        tool='builtin_pileup', engine='builtin',
        total_variants=len(variants), snps=snps, indels=indels,
        ti_tv_ratio=ti_tv, variants=variants,
        message=f"Built-in caller: {len(variants)} variants ({snps} SNPs, {indels} indels)"
    )

    if output_vcf:
        _write_vcf(variants, output_vcf)
        report.output_vcf = output_vcf

    return report


def _calculate_ti_ttv(variants):
    transitions = {'AG', 'GA', 'CT', 'TC'}
    ti = sum(1 for v in variants if f"{v.ref}{v.alt}" in transitions)
    ttv = len(variants) - ti
    return ti / ttv if ttv > 0 else 0


def _write_vcf(variants, output_file):
    with open(output_file, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=biosuite_builtin\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
        for v in variants:
            f.write(f"{v.chrom}\t{v.pos}\t.\t{v.ref}\t{v.alt}\t{v.quality}\t"
                    f"{v.filter_status}\tDP={v.depth};AC={v.alt_count}\tGT\t{v.genotype}\n")


# ── External Tool Wrappers ──────────────────────────────────────────────────

def _freebayes_call(bam_file, reference_file, output_vcf):
    cmd = ['freebayes', '-f', reference_file, bam_file]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode == 0:
            with open(output_vcf, 'w') as f:
                f.write(r.stdout)
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    return False


# ── Public API ──────────────────────────────────────────────────────────────

def call_variants(sam_bam_file, reference_file=None, output_vcf=None, min_depth=4):
    if not os.path.exists(sam_bam_file):
        return VariantReport(tool='none', engine='none', message=f"File not found: {sam_bam_file}")

    tools = check_variant_tools()

    if tools['freebayes'] and reference_file and os.path.exists(reference_file):
        if output_vcf is None:
            output_vcf = tempfile.mktemp(suffix='.vcf')
        if _freebayes_call(sam_bam_file, reference_file, output_vcf):
            return VariantReport(
                tool='freebayes', engine='freebayes', output_vcf=output_vcf,
                message="Using FreeBayes (external)"
            )

    if output_vcf is None:
        output_vcf = tempfile.mktemp(suffix='.vcf')
    return _builtin_call_variants(sam_bam_file, output_vcf, min_depth)


def filter_variants(variants, min_quality=20, min_depth=5, max_allele_freq=0.95):
    filtered = []
    for v in variants:
        if v.quality < min_quality:
            continue
        if v.depth < min_depth:
            continue
        af = v.alt_count / v.depth if v.depth > 0 else 0
        if af > max_allele_freq:
            continue
        filtered.append(v)
    return filtered


def variant_summary(report):
    lines = [
        "=== Variant Calling Report ===",
        f"Engine: {report.engine}",
        f"Total variants: {report.total_variants}",
        f"SNPs: {report.snps}",
        f"Indels: {report.indels}",
        f"Ti/Tv ratio: {report.ti_tv_ratio:.2f}",
    ]
    if report.message:
        lines.append(f"Note: {report.message}")
    return '\n'.join(lines)


# ── Structural Variant Detection ─────────────────────────────────────────────

@dataclass
class StructuralVariant:
    chrom: str
    start: int
    end: int
    sv_type: str  # DEL, INS, DUP, INV, BND
    size: int
    confidence: float
    support_reads: int = 0
    genotype: str = "0/1"


def detect_structural_variants(coverage_data, ref_coverage=None, chrom=None,
                                window_size=100, dup_threshold=1.5, del_threshold=0.5):
    """Detect structural variants from coverage depth analysis.

    Args:
        coverage_data: numpy array of read coverage.
        ref_coverage: reference coverage for normalization (or None for auto).
        chrom: chromosome name.
        window_size: window size for sliding analysis.
        dup_threshold: coverage ratio threshold for duplications.
        del_threshold: coverage ratio threshold for deletions.

    Returns:
        list of StructuralVariant objects.
    """
    if len(coverage_data) == 0:
        return []

    svs = []
    if ref_coverage is None or len(ref_coverage) == 0:
        ref_coverage = np.ones_like(coverage_data) * np.median(coverage_data)

    # Normalize
    median_cov = np.median(coverage_data[coverage_data > 0]) if np.any(coverage_data > 0) else 1
    median_ref = np.median(ref_coverage[ref_coverage > 0]) if np.any(ref_coverage > 0) else 1
    ratio = (coverage_data / max(median_cov, 1)) / (ref_coverage / max(median_ref, 1) + 1e-10)

    # Sliding window analysis
    for i in range(0, len(ratio) - window_size, window_size // 2):
        window = ratio[i:i + window_size]
        mean_ratio = np.mean(window)

        if mean_ratio < del_threshold:
            # Potential deletion
            svs.append(StructuralVariant(
                chrom=chrom or "chr1", start=i, end=i + window_size,
                sv_type="DEL", size=window_size,
                confidence=min(1.0, (del_threshold - mean_ratio) / del_threshold),
                support_reads=int(np.sum(coverage_data[i:i + window_size]))
            ))
        elif mean_ratio > dup_threshold:
            # Potential duplication
            svs.append(StructuralVariant(
                chrom=chrom or "chr1", start=i, end=i + window_size,
                sv_type="DUP", size=window_size,
                confidence=min(1.0, (mean_ratio - dup_threshold) / dup_threshold),
                support_reads=int(np.sum(coverage_data[i:i + window_size]))
            ))

    # Merge overlapping SVs
    svs = _merge_svs(svs)
    return svs


def _merge_svs(svs):
    """Merge overlapping structural variants."""
    if not svs:
        return []

    svs.sort(key=lambda x: (x.chrom, x.start, x.end))
    merged = [svs[0]]
    for sv in svs[1:]:
        last = merged[-1]
        if (sv.chrom == last.chrom and sv.sv_type == last.sv_type and
                sv.start <= last.end + 1000):
            # Merge
            merged[-1] = StructuralVariant(
                chrom=last.chrom, start=last.start, end=max(last.end, sv.end),
                sv_type=last.sv_type, size=max(last.end, sv.end) - last.start,
                confidence=max(last.confidence, sv.confidence),
                support_reads=last.support_reads + sv.support_reads
            )
        else:
            merged.append(sv)
    return merged


def detect_cnv(coverage_data, reference_coverage=None, chrom=None, window_size: int = 1000) -> pd.DataFrame:
    """Detect copy number variations from coverage data.

    Args:
        coverage_data: numpy array of read coverage.
        reference_coverage: reference coverage array.
        chrom: chromosome name.
        window_size: bin size for CNV analysis.

    Returns:
        DataFrame with chrom, start, end, cn (copy number), log2_ratio columns.
    """
    if len(coverage_data) == 0:
        return pd.DataFrame()

    if reference_coverage is None or len(reference_coverage) == 0:
        reference_coverage = np.ones_like(coverage_data) * np.median(coverage_data)

    # Bin the data
    n_bins = max(1, len(coverage_data) // window_size)
    results = []

    for i in range(n_bins):
        start = i * window_size
        end = min((i + 1) * window_size, len(coverage_data))
        cov = np.mean(coverage_data[start:end])
        ref = np.mean(reference_coverage[start:end]) if end <= len(reference_coverage) else cov

        if ref > 0:
            log2_ratio = np.log2((cov + 1) / (ref + 1))
            cn = max(0, round(2 * (cov / max(ref, 1))))
        else:
            log2_ratio = 0
            cn = 2

        results.append({
            "chrom": chrom or "chr1",
            "start": start,
            "end": end,
            "cn": cn,
            "log2_ratio": round(log2_ratio, 3),
            "coverage": round(cov, 2),
        })

    return pd.DataFrame(results)


def format_sv_report(svs):
    """Format structural variant results."""
    lines = ["=== Structural Variant Report ===", f"Total SVs: {len(svs)}"]
    type_counts = {}
    for sv in svs:
        type_counts[sv.sv_type] = type_counts.get(sv.sv_type, 0) + 1
    for sv_type, count in type_counts.items():
        lines.append(f"  {sv_type}: {count}")
    lines.append("\nDetails:")
    for sv in svs[:20]:
        lines.append(f"  {sv.chrom}:{sv.start}-{sv.end} {sv.sv_type} "
                     f"size={sv.size} conf={sv.confidence:.2f}")
    return "\n".join(lines)


def format_cnv_report(cnv_df):
    """Format CNV results."""
    if cnv_df.empty:
        return "No CNV data."
    lines = ["=== CNV Report ===", f"Total bins: {len(cnv_df)}"]
    cn_counts = cnv_df['cn'].value_counts().sort_index()
    for cn, count in cn_counts.items():
        label = {0: "DEL", 1: "DEL", 2: "Normal", 3: "DUP", 4: "DUP"}.get(cn, f"CN={cn}")
        lines.append(f"  CN={cn} ({label}): {count} bins")
    return "\n".join(lines)
