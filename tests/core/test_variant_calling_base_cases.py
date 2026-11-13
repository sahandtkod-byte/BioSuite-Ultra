"""Audit-remediation regression tests — Phase 2 (biosuite/core/variant_calling.py)."""
import os
import tempfile

import numpy as np
import pytest

from biosuite.core.variant_calling import (
    _call_variants_from_pileup,
    _load_reference,
    _pileup_reads,
    _read_sam,
    _write_vcf,
    call_variants,
    detect_structural_variants,
)

REF_SEQ = "ACGTACGTACGT"          # 1-based pos 6 is 'C'
ALT_SEQ = "ACGTAGGTACGT"          # 'G' at position 6
LOW_QUAL = "IIIIIIIIIIII"         # Q40 everywhere


def sam_line(qname, flag, seq, mapq=60, pos=1):
    return (f"{qname}\t{flag}\tchr1\t{pos}\t{mapq}\t12M\t*\t0\t0\t{seq}\t{LOW_QUAL}")


def write_sam(tmp_path, lines, name="a.sam"):
    path = os.path.join(str(tmp_path), name)
    header = "HDR\tVN:1.6\tSO:coordinate\nSQ\tSN:chr1\tLN:12"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n" + "\n".join(lines) + "\n")
    return path


def write_ref(tmp_path, name="ref.fa"):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(">chr1\n" + REF_SEQ + "\n")
    return path


class TestStep21BaseNormalisation:
    """N6: bases were neither upper-cased nor filtered to ACGT."""

    def test_mixed_case_is_not_a_variant(self, tmp_path):
        lines = [sam_line(f"u{i}", 0, REF_SEQ) for i in range(4)]
        lines += [sam_line(f"l{i}", 0, REF_SEQ.lower()) for i in range(4)]
        sam = write_sam(tmp_path, lines)
        report = call_variants(sam, reference_file=write_ref(tmp_path),
                               output_vcf=os.path.join(str(tmp_path), "o.vcf"))
        assert report.variants == [], [(v.ref, v.alt) for v in report.variants]

    def test_ambiguous_base_is_not_counted_in_depth(self, tmp_path):
        lines = [sam_line(f"g{i}", 0, REF_SEQ) for i in range(6)]
        lines += [sam_line(f"n{i}", 0, "ACGTAN" + "GTACGT"[:6]) for i in range(2)]
        reads = _read_sam(write_sam(tmp_path, lines))
        piles = _pileup_reads(reads)
        # position 6 is where the two N-carrying reads are ambiguous
        assert set(piles["chr1"][6]) == {"C"}, piles["chr1"][6]
        assert len(piles["chr1"][6]) == 6

    def test_lower_case_variant_is_still_called(self, tmp_path):
        lines = [sam_line(f"r{i}", 0, ALT_SEQ.lower()) for i in range(6)]
        lines += [sam_line(f"w{i}", 0, REF_SEQ) for i in range(2)]
        reads = _read_sam(write_sam(tmp_path, lines))
        variants = _call_variants_from_pileup(
            _pileup_reads(reads), reference={"chr1": REF_SEQ})
        called = {v.pos: (v.ref, v.alt) for v in variants}
        assert called == {6: ("C", "G")}, called


class TestStep24SamFilters:
    """N8: duplicates, QC-fail reads and low MAPQ all inflated the pileup."""

    def test_duplicate_and_qc_fail_reads_are_dropped(self, tmp_path):
        lines = [sam_line(f"good{i}", 0, ALT_SEQ) for i in range(6)]
        lines += [sam_line(f"dup{i}", 1024, ALT_SEQ) for i in range(6)]
        assert len(_read_sam(write_sam(tmp_path, lines))) == 6

    def test_qc_fail_only_flag_is_dropped(self, tmp_path):
        lines = [sam_line(f"q{i}", 512, ALT_SEQ) for i in range(3)]
        lines += [sam_line(f"o{i}", 0, ALT_SEQ) for i in range(3)]
        assert len(_read_sam(write_sam(tmp_path, lines))) == 3

    def test_mapq_threshold_is_applied_and_configurable(self, tmp_path):
        lines = [sam_line(f"m{i}", 0, ALT_SEQ, mapq=10) for i in range(8)]
        sam = write_sam(tmp_path, lines)
        assert _read_sam(sam) == []
        assert len(_read_sam(sam, min_mapq=0)) == 8

    def test_depth_no_longer_counts_junk(self, tmp_path):
        lines = [sam_line(f"r{i}", 0, ALT_SEQ) for i in range(6)]
        lines += [sam_line(f"d{i}", 1024, ALT_SEQ) for i in range(10)]
        sam = write_sam(tmp_path, lines)
        report = call_variants(sam, reference_file=write_ref(tmp_path),
                               output_vcf=os.path.join(str(tmp_path), "o.vcf"))
        assert [v.depth for v in report.variants] == [6]


class TestStep22RealReference:
    """FIX-3: REF was the majority sequenced allele, so genotypes were meaningless."""

    def test_reference_loader_keys_by_contig(self, tmp_path):
        assert _load_reference(write_ref(tmp_path)) == {"chr1": REF_SEQ}

    def test_ref_comes_from_the_reference_not_the_reads(self, tmp_path):
        lines = [sam_line(f"v{i}", 0, ALT_SEQ) for i in range(6)]
        lines += [sam_line(f"w{i}", 0, REF_SEQ) for i in range(2)]
        report = call_variants(write_sam(tmp_path, lines),
                               reference_file=write_ref(tmp_path),
                               output_vcf=os.path.join(str(tmp_path), "o.vcf"))
        assert [(v.pos, v.ref, v.alt, v.genotype) for v in report.variants] == \
            [(6, "C", "G", "1/1")]
        assert report.variants[0].filter_status == "PASS"

    def test_heterozygous_call_keeps_reference_allele(self, tmp_path):
        lines = [sam_line(f"v{i}", 0, ALT_SEQ) for i in range(4)]
        lines += [sam_line(f"w{i}", 0, REF_SEQ) for i in range(4)]
        report = call_variants(write_sam(tmp_path, lines),
                               reference_file=write_ref(tmp_path),
                               output_vcf=os.path.join(str(tmp_path), "o.vcf"))
        assert [(v.ref, v.alt, v.genotype) for v in report.variants] == [("C", "G", "0/1")]

    def test_without_reference_every_call_is_flagged(self, tmp_path):
        # Same data as above: the majority allele used to be published as REF.
        lines = [sam_line(f"v{i}", 0, ALT_SEQ) for i in range(6)]
        lines += [sam_line(f"w{i}", 0, REF_SEQ) for i in range(2)]
        report = call_variants(write_sam(tmp_path, lines),
                               output_vcf=os.path.join(str(tmp_path), "o.vcf"))
        assert all(v.filter_status == "NOREF" for v in report.variants)
        assert "NOREF" in report.message
        # the inversion is exactly the historic bug: REF=G ALT=C
        assert [(v.ref, v.alt) for v in report.variants] == [("G", "C")]

    def test_unreadable_reference_is_reported_not_silently_ignored(self, tmp_path):
        sam = write_sam(tmp_path, [sam_line(f"v{i}", 0, ALT_SEQ) for i in range(6)])
        report = call_variants(sam, reference_file=os.path.join(str(tmp_path), "nope.fa"),
                               output_vcf=os.path.join(str(tmp_path), "o.vcf"))
        assert report.variants == []
        assert "could not be read" in report.message

    def test_vcf_filter_column_carries_the_flag(self, tmp_path):
        vcf = os.path.join(str(tmp_path), "out.vcf")
        lines = [sam_line(f"v{i}", 0, ALT_SEQ) for i in range(6)]
        lines += [sam_line(f"w{i}", 0, REF_SEQ) for i in range(2)]
        call_variants(write_sam(tmp_path, lines), output_vcf=vcf)
        body = [l for l in open(vcf, encoding="utf-8").read().splitlines() if not l.startswith("#")]
        assert body and all(cols[6] == "NOREF" for cols in (l.split("\t") for l in body)), body


class TestStep23WindowBound:
    """N7: the sliding window never reached the end of the molecule."""

    def test_terminal_deletion_reaches_the_end_of_the_molecule(self):
        # Windows overlap by half and _merge_svs joins them, so the reported span
        # starts earlier than the true deletion.  What the fix guarantees is that
        # the final window is examined at all: before it, no call ever reached 300.
        cov = np.array([30.0] * 200 + [0.0] * 100)
        svs = detect_structural_variants(cov, window_size=100, del_threshold=0.5)
        assert svs, "no structural variant reported at all"
        best = max(svs, key=lambda s: s.confidence)
        assert best.sv_type == "DEL"
        assert best.start <= 200, f"reported region misses the deletion start: {best.start}"
        assert best.end == 300, f"terminal window still not examined: {best.end}"
        assert best.confidence > 0.9

    def test_deletion_just_before_the_end_is_still_found(self):
        cov = np.array([30.0] * 240 + [0.0] * 60)      # ends 60 bp before 300
        svs = detect_structural_variants(cov, window_size=100, del_threshold=0.5)
        assert any(s.sv_type == "DEL" and s.end == 300 for s in svs), svs

    def test_contig_exactly_one_window_is_no_longer_invisible(self):
        svs = detect_structural_variants(np.zeros(100), window_size=100)
        assert svs and (svs[0].start, svs[0].end) == (0, 100)

    def test_end_never_exceeds_the_molecule(self):
        cov = np.array([30.0] * 45 + [0.0] * 45)     # 90 bp, shorter than the window
        for sv in detect_structural_variants(cov, window_size=100):
            assert 0 <= sv.start <= sv.end <= 90
            assert sv.size == sv.end - sv.start

    def test_normal_molecule_reports_nothing(self):
        assert detect_structural_variants(np.full(500, 30.0), window_size=100) == []
