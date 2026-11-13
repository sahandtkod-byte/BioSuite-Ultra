"""
Variant calling from BAM/SAM files with dual-mode execution.

Pure Python variant caller as default, FreeBayes as optional faster tool.
"""
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binom


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
    # Per-ALT support for a multi-allelic record, in the order the alleles appear
    # in ``alt``.  ``alt_count`` alone cannot express a 1/2 site.
    allele_counts: dict = field(default_factory=dict)


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


from .utils import has_tool as _has_tool, secure_temp_path


def check_variant_tools():
    return {'freebayes': _has_tool('freebayes'), 'bcftools': _has_tool('bcftools')}


def _read_sam(sam_file, min_mapq=20):
    reads = []
    with open(sam_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith('@'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 11:
                continue
            flag = int(parts[1])
            # 0x4 unmapped, 0x100 secondary, 0x200 QC-fail, 0x400 duplicate,
            # 0x800 supplementary — all of them would inflate pileup depth with
            # bases no caller should trust.
            if flag & 0xF04:
                continue
            mapq = int(parts[4])
            if mapq < min_mapq:
                continue
            reads.append({
                'qname': parts[0],
                'rname': parts[2],
                'pos': int(parts[3]),
                'mapq': mapq,
                'cigar': parts[5],
                'seq': parts[9],
                'qual': parts[10],
            })
    return reads


def _pileup_reads(reads, min_base_quality=20):
    piles = defaultdict(lambda: defaultdict(list))
    for r in reads:
        ref_pos = r['pos']
        read_seq = r['seq'].upper()
        cigar = r['cigar']
        read_offset = 0
        for num, op in _parse_cigar(cigar):
            qual = r.get('qual') or '*'
            if op in ('M', '=', 'X'):  # '='/'X' also consume ref+read
                for _ in range(num):
                    if read_offset < len(read_seq):
                        base = read_seq[read_offset]
                        # Anything that is not A/C/G/T (N, or a character the
                        # aligner left in the sequence field) is not an allele.
                        # Honour the per-base quality filter when qualities
                        # are present ('*' means no qualities in the SAM).
                        if base in 'ACGT' and (
                                qual == '*' or read_offset >= len(qual) or
                                ord(qual[read_offset]) - 33 >= min_base_quality):
                            piles[r['rname']][ref_pos].append(base)
                    read_offset += 1
                    ref_pos += 1  # critical: advance or later ops drift
            elif op in ('I', 'S'):
                read_offset += num
            elif op in ('D', 'N'):  # 'N' = spliced intron: consumes reference
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


# Per-base sequencing error rate the built-in caller assumes when it turns pileup
# counts into a VCF QUAL: Q30 (one in a thousand) spread over the three bases the
# read could have carried instead of the called one.
BUILTIN_BASE_ERROR_RATE = 1e-3


def _allele_quality(count, depth, error_rate=BUILTIN_BASE_ERROR_RATE):
    """Phred-scale how unlikely ``count`` supporting reads are under no variant.

    QUAL used to be ``int(allele_fraction * 60)``, which ignored depth entirely:
    3 alt bases in 10 reads scored the same 18 as 90 in 300.  It also collided with
    ``filter_variants()``, whose default ``min_quality`` of 20 discarded every call
    the caller's own ``min_allele_freq`` of 0.25 could emit (QUAL 15).
    """
    per_alt_error = error_rate / 3.0
    # P(at least ``count`` errors at this position) if the reference is really
    # reference: the smaller it is, the stronger the evidence for the alt.
    p = float(binom.sf(count - 1, depth, per_alt_error))
    if p <= 0.0:
        return 99
    return min(99, int(round(-10.0 * np.log10(p))))


def _call_variants_from_pileup(piles, min_depth=4, min_allele_freq=0.25, reference=None):
    """Call variants from a pileup.

    One record is emitted per position, listing every ALT that clears
    ``min_allele_freq`` comma-separated in allele-frequency order, because VCF
    numbers ALT alleles 1..k within a single record: one record per ALT would each
    genotype the sample ``0/1`` at the same position, claiming a diploid carries
    three alleles at once.

    Args:
        piles: mapping of contig -> position -> list of sequenced bases.
        min_depth: minimum number of usable bases required to judge a position.
        min_allele_freq: minimum alternate-allele fraction to include an allele.
        reference: optional mapping of contig -> reference sequence.  Without it
            REF falls back to the majority sequenced base, which cannot tell
            reference from variant, so those calls are flagged ``NOREF`` and are
            filtered out by downstream tools instead of being silently believed.

    Returns:
        List of Variant.  A pileup only sees substituted bases, so every variant
        is a SNP; ``genotype`` is ``0/1``/``1/1`` for a single ALT, ``1/2`` for a
        two-allele site and ``./.`` when three or more ALTs clear the threshold,
        which a diploid cannot carry and a caller should not name.
    """
    variants = []
    for chrom, positions in sorted(piles.items()):
        contig = reference.get(chrom) if reference else None
        for pos, bases in sorted(positions.items()):
            if len(bases) < min_depth:
                continue
            counts = Counter(bases)
            if contig is not None and 1 <= pos <= len(contig):
                ref_base = contig[pos - 1]
                filter_status = 'PASS'
            else:
                ref_base = counts.most_common(1)[0][0]
                filter_status = 'NOREF'
            total = len(bases)

            alt_counts = [(base, count) for base, count in counts.most_common()
                          if base != ref_base and count / total >= min_allele_freq]
            if not alt_counts:
                continue

            alleles = [base for base, _ in alt_counts]
            support = sum(count for _, count in alt_counts)
            if len(alt_counts) == 1:
                fraction = alt_counts[0][1] / total
                gt = '1/1' if fraction >= 0.75 else '0/1'
            elif len(alt_counts) == 2:
                gt = '1/2'
            else:
                gt = './.'
            qual = max(_allele_quality(count, total) for _, count in alt_counts)

            variants.append(Variant(
                chrom=chrom, pos=pos, ref=ref_base, alt=",".join(alleles),
                quality=qual, depth=total, alt_count=support,
                genotype=gt,
                # The pileup this package builds itself only ever substitutes
                # single bases, so this is SNP in practice; a hand-built pileup
                # can carry longer alleles, and those are not SNPs.
                variant_type='SNP' if len(ref_base) == 1 and all(
                    len(a) == 1 for a in alleles) else 'MNP',
                filter_status=filter_status,
                allele_counts={base: count for base, count in alt_counts}
            ))

    return variants


def _load_reference(path):
    """Read a reference FASTA into {contig: sequence}.

    Contigs are keyed by the first header token, which is how SAM/RNAME names
    them.  Returns None when nothing usable could be read.
    """
    from .sequence import read_fasta
    records = read_fasta(path)
    if not records:
        return None
    reference = {}
    for header, seq in records:
        tokens = header.split()
        if tokens:
            reference[tokens[0]] = seq.upper()
    return reference or None


def _builtin_call_variants(sam_file, output_vcf=None, min_depth=4, reference_file=None):
    reads = _read_sam(sam_file)
    if not reads:
        return VariantReport(tool='builtin', engine='builtin', message="No mapped reads found.")

    reference = None
    if reference_file:
        reference = _load_reference(reference_file)
        if reference is None:
            return VariantReport(
                tool='builtin', engine='builtin',
                message=f"Reference FASTA could not be read: {reference_file}")

    piles = _pileup_reads(reads)
    variants = _call_variants_from_pileup(piles, min_depth=min_depth, reference=reference)

    snps = sum(1 for v in variants if v.variant_type == 'SNP')
    indels = sum(1 for v in variants if v.variant_type in ('INS', 'DEL'))
    ti_tv = _calculate_ti_ttv(variants)

    if reference:
        note = (f"Built-in caller: {len(variants)} variants ({snps} SNPs, {indels} indels). "
                f"REF read from {reference_file}. True INS/DEL calls are not emitted — "
                f"use freebayes for those.")
    else:
        note = (f"Built-in caller: {len(variants)} variants ({snps} SNPs, {indels} indels). "
                f"WARNING: no reference supplied, so REF is the majority sequenced allele "
                f"and every call is flagged NOREF — pass reference_file for usable output. "
                f"True INS/DEL calls are not emitted — use freebayes for those.")

    report = VariantReport(
        tool='builtin_pileup', engine='builtin',
        total_variants=len(variants), snps=snps, indels=indels,
        ti_tv_ratio=ti_tv, variants=variants,
        message=note
    )

    if output_vcf:
        _write_vcf(variants, output_vcf)
        report.output_vcf = output_vcf

    return report


def _calculate_ti_ttv(variants):
    """Ti/Tv over called alleles: a multi-allelic record scores one row per ALT."""
    transitions = {'AG', 'GA', 'CT', 'TC'}
    ti = ttv = 0
    for v in variants:
        for alt in _alt_alleles(v):
            if f"{v.ref}{alt}" in transitions:
                ti += 1
            else:
                ttv += 1
    # All-transitions sample has an infinite Ti/Tv — must not be reported
    # as 0.0 (that would look like a 100% transversion sample).
    if ttv == 0:
        return float('inf') if ti > 0 else 0.0
    return ti / ttv


def _alt_alleles(variant):
    """The ALT bases of a record, split the way VCF writes them."""
    if variant.allele_counts:
        return list(variant.allele_counts)
    return variant.alt.split(',')


def _ac_field(variant):
    """INFO/AC: allele counts, one value per ALT, as the VCF spec requires."""
    if variant.allele_counts:
        return ','.join(str(variant.allele_counts[alt]) for alt in variant.alt.split(','))
    return str(variant.alt_count)


def _af_field(variant):
    """INFO/AF: one allele fraction per ALT, so the QUAL cap loses nothing."""
    alleles = variant.alt.split(',')
    if variant.allele_counts:
        counts = [variant.allele_counts[a] for a in alleles]
    else:
        counts = [variant.alt_count] * len(alleles)
    return ','.join(f"{c / variant.depth:.4f}" if variant.depth else "0"
                    for c in counts)


def _write_vcf(variants, output_file):
    with open(output_file, 'w', encoding="utf-8") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=biosuite_builtin\n")
        f.write('##INFO=<ID=AC,Number=A,Type=Integer,Description='
                '"Alternate allele counts, one per ALT">\n')
        f.write('##INFO=<ID=AF,Number=A,Type=Float,Description='
                '"Alternate allele fractions, one per ALT">\n')
        f.write(f"##QUAL=<Reference: phred-scaled binomial evidence against a "
                f"{BUILTIN_BASE_ERROR_RATE:g} per-base error rate, capped at 99>\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
        for v in variants:
            f.write(f"{v.chrom}\t{v.pos}\t.\t{v.ref}\t{v.alt}\t{v.quality}\t"
                    f"{v.filter_status}\tDP={v.depth};AC={_ac_field(v)};AF={_af_field(v)}\t"
                    f"GT\t{v.genotype}\n")


# ── External Tool Wrappers ──────────────────────────────────────────────────

def _freebayes_call(bam_file, reference_file, output_vcf):
    cmd = ['freebayes', '-f', reference_file, bam_file]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode == 0:
            with open(output_vcf, 'w', encoding="utf-8") as f:
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
            output_vcf = secure_temp_path(suffix='.vcf')
        if _freebayes_call(sam_bam_file, reference_file, output_vcf):
            return VariantReport(
                tool='freebayes', engine='freebayes', output_vcf=output_vcf,
                message="Using FreeBayes (external)"
            )

    if output_vcf is None:
        output_vcf = secure_temp_path(suffix='.vcf')
    return _builtin_call_variants(sam_bam_file, output_vcf, min_depth, reference_file)


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
    if window_size < 1:
        # A zero-width window made np.mean([]) NaN, which failed every threshold
        # test, so the function returned an empty list after a screenful of
        # "Mean of empty slice" warnings.
        raise ValueError(f"window_size must be at least 1, got {window_size}.")

    if len(coverage_data) == 0:
        return []

    svs = []
    if ref_coverage is None or len(ref_coverage) == 0:
        ref_coverage = np.ones_like(coverage_data) * np.median(coverage_data)
    else:
        # Align lengths: broadcasting a shorter reference would raise.
        n = min(len(coverage_data), len(ref_coverage))
        coverage_data = coverage_data[:n]
        ref_coverage = ref_coverage[:n]

    # Normalize
    median_cov = np.median(coverage_data[coverage_data > 0]) if np.any(coverage_data > 0) else 1
    median_ref = np.median(ref_coverage[ref_coverage > 0]) if np.any(ref_coverage > 0) else 1
    ratio = (coverage_data / max(median_cov, 1)) / (ref_coverage / max(median_ref, 1) + 1e-10)

    # Sliding window analysis.  The final window must start at
    # len(ratio) - window_size inclusive, otherwise a variant in the last
    # window_size bases is never examined; a molecule shorter than one window is
    # still judged on the single partial window rather than ignored.
    step = max(window_size // 2, 1)
    last_start = max(len(ratio) - window_size, 0)
    length = len(ratio)
    for i in range(0, last_start + 1, step):
        window = ratio[i:i + window_size]
        mean_ratio = np.mean(window)
        end = min(i + window_size, length)

        if mean_ratio < del_threshold:
            # Potential deletion
            svs.append(StructuralVariant(
                chrom=chrom or "chr1", start=i, end=end,
                sv_type="DEL", size=end - i,
                confidence=min(1.0, (del_threshold - mean_ratio) / del_threshold),
                support_reads=int(np.sum(coverage_data[i:end]))
            ))
        elif mean_ratio > dup_threshold:
            # Potential duplication
            svs.append(StructuralVariant(
                chrom=chrom or "chr1", start=i, end=end,
                sv_type="DUP", size=end - i,
                confidence=min(1.0, (mean_ratio - dup_threshold) / dup_threshold),
                support_reads=int(np.sum(coverage_data[i:end]))
            ))

    # Merge overlapping SVs, then re-measure support over the merged span: the
    # windows step by half a window, so summing the members' support_reads counted
    # the bases in every overlap twice.
    svs = _merge_svs(svs)
    for sv in svs:
        sv.support_reads = int(np.sum(coverage_data[sv.start:sv.end]))
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
    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}.")

    if len(coverage_data) == 0:
        return pd.DataFrame()

    if reference_coverage is None or len(reference_coverage) == 0:
        reference_coverage = np.ones_like(coverage_data) * np.median(coverage_data)
    elif len(reference_coverage) != len(coverage_data):
        # Bins past the end of the reference used to substitute the sample's own
        # coverage for the missing comparison and then report CN=2, log2=0.0 —
        # a measured normal copy number for territory nothing was measured against.
        n = min(len(coverage_data), len(reference_coverage))
        coverage_data = coverage_data[:n]
        reference_coverage = reference_coverage[:n]

    # Bin the data — ceil division keeps the trailing partial bin
    # (floor division silently dropped the last len % window_size bases).
    n_bins = max(1, -(-len(coverage_data) // window_size))
    results = []

    for i in range(n_bins):
        start = i * window_size
        end = min((i + 1) * window_size, len(coverage_data))
        cov = np.mean(coverage_data[start:end])
        ref = np.mean(reference_coverage[start:end])

        if ref > 0:
            log2_ratio = np.log2((cov + 1) / (ref + 1))
            # Divide by ref itself, not max(ref, 1): on sub-1x reference data a
            # bin matching its reference coverage is 2 copies, and max(ref, 1)
            # reported it as CN=1 while log2_ratio in the same row said 0.0.
            cn = max(0, round(2 * cov / ref))
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
