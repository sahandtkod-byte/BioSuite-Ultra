"""Regression tests for variant_calling.py review fixes."""
import pytest
import numpy as np

from biosuite.core.variant_calling import (
    _read_sam, _pileup_reads, _calculate_ti_ttv, _call_variants_from_pileup,
    detect_cnv, detect_structural_variants, Variant,
)


def _sam_line(qname, flag, pos, cigar, seq, qual=None, mapq=60):
    q = qual if qual is not None else 'I' * len(seq)
    return f"{qname}\t{flag}\tchr1\t{pos}\t{mapq}\t{cigar}\t*\t0\t0\t{seq}\t{q}"


def _write_sam(tmp_path, lines):
    p = tmp_path / "x.sam"
    p.write_text("@HD\tVN:1.6\n" + "\n".join(lines) + "\n")
    return str(p)


def test_cigar_equals_x_counted_as_matches(tmp_path):
    # '=' and 'X' ops consume both ref and read — before the fix they were
    # silently ignored, so the variant vanished.
    lines = [_sam_line(f"r{i}", 0, 1, "5=1X5=", "ACGTGTACGTA") for i in range(10)]
    reads = _read_sam(_write_sam(tmp_path, lines))
    piles = _pileup_reads(reads, min_base_quality=0)
    # '=' or 'X' at ref positions 1..5 and 6..11; the 'X' sits at position 6.
    assert len(piles['chr1'][6]) == 10 and set(piles['chr1'][6]) == {'T'}


def test_cigar_n_advances_reference(tmp_path):
    # Spliced read 5M100N5M: second half maps to ref 106..110.
    line = _sam_line("sp", 0, 1, "5M100N5M", "ACGTATTTTT")
    reads = _read_sam(_write_sam(tmp_path, [line]))
    piles = _pileup_reads(reads, min_base_quality=0)
    assert piles['chr1'][106] == ['T'] and piles['chr1'][110] == ['T']
    assert not piles['chr1'][6], "pre-'N'-fix bug put the tail right after 5M"


def test_secondary_and_supplementary_filtered(tmp_path):
    lines = [
        _sam_line("prim", 0, 1, "10M", "ACGTACGTAC"),
        _sam_line("sec", 256, 1, "10M", "TTTTTTTTTT"),
        _sam_line("sup", 2048, 1, "10M", "GGGGGGGGGG"),
    ]
    reads = _read_sam(_write_sam(tmp_path, lines))
    assert len(reads) == 1


def test_base_quality_filter_applied(tmp_path):
    lines = []
    for i in range(10):
        seq = list("ACGTACGTAC")
        seq[3] = 'T'
        qual = list('I' * 10)
        qual[3] = '!' if i < 9 else 'I'  # 9 of 10 alt bases are low quality
        lines.append(_sam_line(f"r{i}", 0, 1, "10M", "".join(seq), "".join(qual)))
    reads = _read_sam(_write_sam(tmp_path, lines))
    piles = _pileup_reads(reads, min_base_quality=20)
    bases = piles['chr1'][4]
    assert bases.count('T') == 1  # before fix min_base_quality was ignored


def test_ti_tv_all_transitions_is_inf():
    variants = [
        Variant('c', 1, 'A', 'G', 60, 10, 5, '0/1', 'SNP'),
        Variant('c', 2, 'C', 'T', 60, 10, 5, '0/1', 'SNP'),
    ]
    assert _calculate_ti_ttv(variants) == float('inf')
    # sanity: mixed sample
    variants.append(Variant('c', 3, 'A', 'T', 60, 10, 5, '0/1', 'SNP'))
    assert _calculate_ti_ttv(variants) == pytest.approx(2.0)


def test_cnv_keeps_trailing_bin():
    cov = np.ones(5500) * 50.0
    df = detect_cnv(cov, window_size=1000)
    assert len(df) == 6 and df.iloc[-1]['end'] == 5500


def test_sv_shorter_reference_no_crash():
    cov = np.ones(500)
    ref = np.ones(300)  # shorter previously raised a broadcast error
    out = detect_structural_variants(cov, ref_coverage=ref)
    assert isinstance(out, list)


def test_pileup_variant_call_end_to_end(tmp_path):
    lines = []
    for i in range(10):
        seq = list("ACGTACGTAC")
        if i < 5:
            seq[0] = 'T'
        lines.append(_sam_line(f"r{i}", 0, 1, "10M", "".join(seq), 'I' * 10))
    reads = _read_sam(_write_sam(tmp_path, lines))
    piles = _pileup_reads(reads, min_base_quality=0)
    variants = _call_variants_from_pileup(piles, min_depth=5, min_allele_freq=0.25)
    assert len(variants) == 1
    v = variants[0]
    # VCF position stays at the SAM 1-based coordinate (fixes the old +1 shift);
    # with a tie the majority is the first-inserted base, 'T' here.
    assert v.pos == 1 and {v.ref, v.alt} == {'A', 'T'} and v.genotype == '0/1'
