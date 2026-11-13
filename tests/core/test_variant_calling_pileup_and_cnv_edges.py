"""Pileup bookkeeping, external-tool wiring and the copy-number / summary paths of
biosuite/core/variant_calling.py.

A CIGAR that consumes more reference than the read carries, a clipped or spliced
operation, a multi-allelic position and a freebayes subprocess are exactly where
an off-by-one silently becomes a wrong genotype, so every coordinate, count and
command line below was read off the running code.  pysam, freebayes and the
reference FASTA are never executed: the subprocess is stubbed and its argv is
asserted.
"""
import os
import subprocess

import numpy as np
import pytest

from biosuite.core import variant_calling as vc

REF_SEQ = "ACGTACGTACGT"          # 12 nt; 1-based position 6 is 'C'
ALT_SEQ = "ACGTAGGTACGT"          # 'G' at position 6


def sam_line(qname, flag, seq, mapq=60, pos=1, cigar="12M"):
    return f"{qname}\t{flag}\tchr1\t{pos}\t{mapq}\t{cigar}\t*\t0\t0\t{seq}\t{'I' * len(seq)}"


def write_sam(tmp_path, lines, name="a.sam"):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:12\n")
        handle.write("".join(line + "\n" for line in lines))
    return path


def write_ref(tmp_path, text=">chr1 desc\n" + REF_SEQ + "\n", name="ref.fa"):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def read(cigar, seq, qual=None, pos=1, rname="chr1"):
    return {"qname": "r", "rname": rname, "pos": pos, "mapq": 60,
            "cigar": cigar, "seq": seq, "qual": "I" * len(seq) if qual is None else qual}


def pile_dict(piles):
    return {chrom: {pos: sorted(bases) for pos, bases in sorted(positions.items())}
            for chrom, positions in piles.items()}


# ── SAM reading ──────────────────────────────────────────────────────────────

def test_a_cigar_operation_without_a_length_counts_as_one_base(tmp_path):
    assert vc._parse_cigar("M") == [(1, "M")]
    assert vc._parse_cigar("*5M") == [(1, "*"), (5, "M")]
    assert vc._parse_cigar("10M") == [(10, "M")]
    assert vc._parse_cigar("10") == []            # trailing digits, no operation
    assert vc._parse_cigar("") == []


def test_reads_without_a_usable_flag_or_quality_are_never_collected(tmp_path):
    lines = [sam_line("ok", 0, REF_SEQ),
             sam_line("dup", 0x400, ALT_SEQ),
             sam_line("secondary", 0x100, ALT_SEQ),
             sam_line("supp", 0x800, ALT_SEQ),
             sam_line("qcfail", 0x200, ALT_SEQ),
             sam_line("unmapped", 0x4, ALT_SEQ),
             sam_line("lowmapq", 0, ALT_SEQ, mapq=19),
             "name\t1\tchr1\t1"]                 # fewer than 11 columns
    reads = vc._read_sam(write_sam(tmp_path, lines))
    assert [r["qname"] for r in reads] == ["ok"]
    assert reads[0]["pos"] == 1 and reads[0]["cigar"] == "12M"


# ── pileup bookkeeping ───────────────────────────────────────────────────────

def test_a_cigar_longer_than_the_read_stops_piling_instead_of_wrapping():
    piles = vc._pileup_reads([read("12M", "ACGT")])
    assert pile_dict(piles) == {"chr1": {1: ["A"], 2: ["C"], 3: ["G"], 4: ["T"]}}


def test_clipped_and_inserted_read_bases_never_reach_the_pileup():
    piles = vc._pileup_reads([read("2S4M2I4M", "AAAACCCCGGGG")])
    assert pile_dict(piles) == {"chr1": {1: ["A"], 2: ["A"], 3: ["C"], 4: ["C"],
                                         5: ["G"], 6: ["G"], 7: ["G"], 8: ["G"]}}


@pytest.mark.parametrize("cigar,seq,expected", [
    # hard-clipped bases are absent from SEQ, so 'H' consumes neither strand
    ("3H5M", "ACGTT", {1: ["A"], 2: ["C"], 3: ["G"], 4: ["T"], 5: ["T"]}),
    # a gap in the read still advances the reference
    ("2M2D3M", "ACGGT", {1: ["A"], 2: ["C"], 5: ["G"], 6: ["G"], 7: ["T"]}),
    ("1N2M", "AC", {2: ["A"], 3: ["C"]}),   # a leading N shifts the whole read
])
def test_reference_and_read_offsets_stay_independent_for_every_operation(cigar, seq, expected):
    assert pile_dict(vc._pileup_reads([read(cigar, seq)])) == {"chr1": expected}


def test_an_unknown_cigar_operation_piles_nothing_at_all():
    assert vc._pileup_reads([read("*", "ACGT")]) == {}


def test_a_missing_or_short_quality_string_does_not_block_a_base():
    assert pile_dict(vc._pileup_reads([read("3M", "ACG", qual="*")])) == {
        "chr1": {1: ["A"], 2: ["C"], 3: ["G"]}}
    assert pile_dict(vc._pileup_reads([read("3M", "ACG", qual="II")])) == {
        "chr1": {1: ["A"], 2: ["C"], 3: ["G"]}}


# ── calling from a pileup ────────────────────────────────────────────────────

def test_a_position_below_the_minimum_depth_is_not_judged_at_all():
    assert vc._call_variants_from_pileup({"chr1": {5: ["A", "A", "C", "C"]}},
                                         min_depth=5) == []
    assert len(vc._call_variants_from_pileup({"chr1": {5: ["A", "A", "C", "C"]}},
                                             min_depth=4)) == 1


def test_an_allele_below_the_minimum_frequency_is_dropped_and_the_rest_are_kept():
    # 10 bases: A x4, G x4, C x2 -> A is the majority REF, C is only 20%
    piles = {"chr1": {10: ["A"] * 4 + ["C"] * 2 + ["G"] * 4}}
    called = vc._call_variants_from_pileup(piles, min_depth=4, min_allele_freq=0.25)
    assert [(v.ref, v.alt, v.depth, v.alt_count, v.quality, v.genotype)
            for v in called] == [("A", "G", 10, 4, 99, "0/1")]
    assert called[0].allele_counts == {"G": 4}


def test_a_multi_allelic_site_is_one_record_with_both_alt_alleles_and_a_1_2_call():
    """VCF numbers ALT alleles 1..k inside a single record.  Emitting one record
    per allele genotyped each of them 0/1 at the same position, which claims a
    diploid sample carries three alleles at one site."""
    piles = {"chr1": {4: ["T"] * 4 + ["G"] * 3 + ["C"] * 3}}
    called = vc._call_variants_from_pileup(piles, min_depth=4)
    assert len(called) == 1
    v = called[0]
    assert (v.ref, v.alt, v.alt_count, v.genotype, v.filter_status) == (
        "T", "G,C", 6, "1/2", "NOREF")
    assert v.allele_counts == {"G": 3, "C": 3}
    assert v.quality == 84


def test_a_third_alt_allele_above_the_threshold_leaves_the_genotype_unresolved():
    """A diploid cannot carry three alleles at one site, so the caller says './.'
    instead of picking a genotype it has no evidence for."""
    called = vc._call_variants_from_pileup(
        {"chr1": {4: ["T"] * 4 + ["G"] * 4 + ["C"] * 4 + ["A"] * 4}}, min_depth=4)
    assert [(v.alt, v.genotype, v.alt_count) for v in called] == [("G,C,A", "./.", 12)]


def test_a_pileup_of_one_allele_only_calls_nothing_because_ref_equals_alt():
    assert vc._call_variants_from_pileup({"chr1": {2: ["G"] * 5}}, min_depth=4) == []


def test_an_alternative_that_dominates_the_reads_is_called_homozygous():
    called = vc._call_variants_from_pileup(
        {"chr1": {7: ["A"] * 2 + ["G"] * 8}}, min_depth=4,
        reference={"chr1": "TTTTTTAGGGGGGGGG"})
    assert len(called) == 1
    variant = called[0]
    assert (variant.ref, variant.alt, variant.depth, variant.alt_count) == ("A", "G", 10, 8)
    assert (variant.genotype, variant.quality, variant.filter_status) == ("1/1", 99, "PASS")
    assert variant.variant_type == "SNP"


def test_a_minor_alternative_above_the_threshold_is_called_heterozygous():
    called = vc._call_variants_from_pileup(
        {"chr1": {7: ["A"] * 8 + ["G"] * 3}}, min_depth=4,
        reference={"chr1": "TTTTTTAGGGGGGG"})
    assert len(called) == 1
    variant = called[0]
    assert (variant.ref, variant.alt, variant.depth, variant.alt_count) == ("A", "G", 11, 3)
    assert (variant.genotype, variant.quality, variant.filter_status, variant.variant_type) == (
        "0/1", 82, "PASS", "SNP")


def test_the_quality_score_grows_with_depth_at_a_constant_allele_fraction():
    """QUAL used to be int(allele_fraction * 60): 3 alt bases in 10 reads scored
    the same 18 as 90 in 300, so depth carried no weight at all.  It is now a
    phred-scaled binomial tail against BUILTIN_BASE_ERROR_RATE, capped at 99."""
    assert vc.BUILTIN_BASE_ERROR_RATE == 1e-3
    assert vc._allele_quality(3, 10) == 84
    assert vc._allele_quality(30, 100) == 99
    assert vc._allele_quality(90, 300) == 99
    assert vc._allele_quality(1, 4) == 29, "one supporting read is the weakest case"
    assert vc._allele_quality(1, 20) == 22 < vc._allele_quality(1, 4), \
        "a lone alt base in a deeper pileup is the likelier error"
    assert vc._allele_quality(0, 10) == 0, "no supporting read is no evidence at all"
    assert vc._allele_quality(999, 1000) == 99, "the tail underflows to zero and caps"
    assert vc._allele_quality(30, 100) == 99, "reaches the cap through min(), not underflow"

    def quality(depth, alt):
        bases = ["A"] * (depth - alt) + ["G"] * alt
        called = vc._call_variants_from_pileup(
            {"c": {1: bases}}, min_depth=1, reference={"c": "A" * depth})
        return called[0].quality

    assert quality(4, 1) == 29 < quality(10, 3) == 84 < quality(100, 30) == 99


def test_the_written_vcf_carries_the_allele_fractions_the_cap_hides():
    """Capping QUAL at 99 flattens the strongest sites, so AF is written next to
    AC and stays informative for a multi-allelic record."""
    called = vc._call_variants_from_pileup(
        {"chr1": {4: ["T"] * 4 + ["G"] * 3 + ["C"] * 3}}, min_depth=4)
    assert vc._af_field(called[0]) == "0.3000,0.3000"
    assert vc._ac_field(called[0]) == "3,3"
    lone = vc.Variant("chr1", 9, "A", "G", 60, 10, 3, "0/1", "SNP")
    assert vc._af_field(lone) == "0.3000" and vc._ac_field(lone) == "3"
    empty = vc.Variant("chr1", 9, "A", "G", 60, 0, 0, "0/1", "SNP")
    assert vc._af_field(empty) == "0"


def test_a_call_at_the_minimum_allele_frequency_survives_the_default_filter():
    """min_allele_freq=0.25 used to emit QUAL=15, which the default min_quality
    of 20 in filter_variants() then discarded: the pipeline threw away the
    weakest calls its own caller was allowed to make."""
    bases = ["A"] * 15 + ["C"] * 5
    called = vc._call_variants_from_pileup({"c": {1: bases}}, min_depth=4,
                                           reference={"c": "A" * 20})
    assert [v.quality for v in called] == [99]
    assert [v.alt for v in vc.filter_variants(called)] == ["C"]


def test_a_multi_base_allele_is_typed_mnp_rather_than_snp():
    called = vc._call_variants_from_pileup({"chr1": {1: ["AT"] * 3 + ["CG"] * 1}}, min_depth=4)
    assert [(v.ref, v.alt, v.variant_type, v.quality) for v in called] == [
        ("AT", "CG", "MNP", 29)]


def test_a_position_past_the_end_of_the_reference_falls_back_to_noref():
    called = vc._call_variants_from_pileup({"chr1": {99: ["A"] * 3 + ["T"] * 4}},
                                           min_depth=4, reference={"chr1": "CCGG"})
    assert [(v.ref, v.alt, v.filter_status, v.genotype, v.quality) for v in called] == [
        ("T", "A", "NOREF", "0/1", 89)]


# ── reference loading ────────────────────────────────────────────────────────

def test_the_reference_is_keyed_by_the_first_header_token_and_upper_cased(tmp_path):
    ref = vc._load_reference(write_ref(tmp_path, ">chr1 some description\nacgtACGT\n>chr2\nGG\n"))
    assert ref == {"chr1": "ACGTACGT", "chr2": "GG"}


def test_a_fasta_without_a_usable_contig_name_is_no_reference_at_all(tmp_path):
    assert vc._load_reference(write_ref(tmp_path, ">   \nACGT\n")) is None
    assert vc._load_reference(write_ref(tmp_path, "")) is None


# ── built-in pipeline ────────────────────────────────────────────────────────

def test_a_sam_without_usable_reads_is_reported_instead_of_returning_an_empty_call(tmp_path):
    report = vc._builtin_call_variants(write_sam(tmp_path, []))
    assert (report.tool, report.engine) == ("builtin", "builtin")
    assert report.message == "No mapped reads found."
    assert report.variants == [] and report.output_vcf == ""


def test_a_run_without_an_output_path_writes_no_vcf_and_warns_about_the_missing_reference(
        tmp_path):
    sam = write_sam(tmp_path, [sam_line(f"r{i}", 0, ALT_SEQ) for i in range(4)]
                    + [sam_line(f"u{i}", 0, REF_SEQ) for i in range(4)])
    report = vc._builtin_call_variants(sam)
    assert report.output_vcf == ""
    assert report.total_variants == report.snps == 1 and report.indels == 0
    assert report.ti_tv_ratio == pytest.approx(0.0)
    assert [(v.pos, v.ref, v.alt, v.depth, v.alt_count, v.filter_status)
            for v in report.variants] == [(6, "G", "C", 8, 4, "NOREF")]
    assert report.message.startswith("Built-in caller: 1 variants (1 SNPs, 0 indels). WARNING:")
    assert "every call is flagged NOREF" in report.message


def test_a_run_with_a_reference_reads_ref_from_it_and_writes_the_vcf(tmp_path):
    sam = write_sam(tmp_path, [sam_line(f"r{i}", 0, ALT_SEQ) for i in range(4)]
                    + [sam_line(f"u{i}", 0, REF_SEQ) for i in range(4)])
    out = os.path.join(str(tmp_path), "calls.vcf")
    report = vc._builtin_call_variants(sam, out, reference_file=write_ref(tmp_path))
    assert report.output_vcf == out
    assert report.variants[0].filter_status == "PASS"
    assert report.message == (
        "Built-in caller: 1 variants (1 SNPs, 0 indels). REF read from "
        + write_ref(tmp_path) + ". True INS/DEL calls are not emitted — use freebayes for those.")
    body = open(out, encoding="utf-8").read().splitlines()
    assert body[0] == "##fileformat=VCFv4.2"
    assert body[1] == "##source=biosuite_builtin"
    assert body[2].startswith("##INFO=<ID=AC,Number=A,Type=Integer")
    assert body[3].startswith("##INFO=<ID=AF,Number=A,Type=Float")
    assert body[4].startswith("##QUAL=<Reference: phred-scaled binomial evidence")
    assert body[5].startswith("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE")
    assert body[6] == ("chr1\t6\t.\tC\tG\t99\tPASS\tDP=8;AC=4;AF=0.5000\tGT\t0/1")


def test_an_unreadable_reference_stops_the_builtin_caller_with_its_own_message(tmp_path):
    sam = write_sam(tmp_path, [sam_line("r0", 0, ALT_SEQ) for _ in range(4)])
    blank = write_ref(tmp_path, ">   \nACGT\n", name="blank.fa")
    report = vc._builtin_call_variants(sam, reference_file=blank)
    assert report.variants == [] and report.output_vcf == ""
    assert report.message == f"Reference FASTA could not be read: {blank}"


# ── call_variants and the external tool ─────────────────────────────────────

def test_a_missing_input_file_is_named_in_the_report_instead_of_raising(tmp_path):
    missing = os.path.join(str(tmp_path), "absent.sam")
    report = vc.call_variants(missing)
    assert (report.tool, report.engine) == ("none", "none")
    assert report.message == f"File not found: {missing}"


def test_check_variant_tools_probes_both_executables(monkeypatch):
    monkeypatch.setattr(vc, "_has_tool", lambda name: name == "freebayes")
    assert vc.check_variant_tools() == {"freebayes": True, "bcftools": False}


def test_the_freebayes_command_is_built_from_the_bam_and_the_reference_and_its_stdout_is_saved(
        tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append((cmd, capture_output, text, timeout))
        return subprocess.CompletedProcess(cmd, 0, stdout="##VCF from freebayes\n", stderr="")

    monkeypatch.setattr(vc.subprocess, "run", fake_run)
    out = os.path.join(str(tmp_path), "fb.vcf")
    assert vc._freebayes_call("a.bam", "ref.fa", out) is True
    assert calls == [(["freebayes", "-f", "ref.fa", "a.bam"], True, True, 7200)]
    assert open(out, encoding="utf-8").read() == "##VCF from freebayes\n"


@pytest.mark.parametrize("failure", ["nonzero", "missing", "timeout"])
def test_freebayes_failing_in_any_way_writes_nothing_and_says_false(tmp_path, monkeypatch,
                                                                    failure):
    out = os.path.join(str(tmp_path), "fb.vcf")
    if failure == "nonzero":
        def fake_run(cmd, capture_output, text, timeout):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="freebayes crashed")
    elif failure == "missing":
        def fake_run(cmd, capture_output, text, timeout):
            raise OSError("freebayes is not installed")
    else:
        def fake_run(cmd, capture_output, text, timeout):
            raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(vc.subprocess, "run", fake_run)
    assert vc._freebayes_call("a.bam", "ref.fa", out) is False
    assert not os.path.exists(out)


def test_a_working_freebayes_is_preferred_and_its_path_is_reported(tmp_path, monkeypatch):
    sam = write_sam(tmp_path, [sam_line(f"r{i}", 0, ALT_SEQ) for i in range(4)])
    ref = write_ref(tmp_path)
    monkeypatch.setattr(vc, "_has_tool", lambda name: name == "freebayes")
    used = []
    monkeypatch.setattr(vc, "_freebayes_call",
                        lambda bam, reference, output: used.append((bam, reference, output)) or True)
    report = vc.call_variants(sam, reference_file=ref, output_vcf=None)
    assert (report.tool, report.engine) == ("freebayes", "freebayes")
    assert report.message == "Using FreeBayes (external)"
    assert report.output_vcf == used[0][2]
    assert used == [(sam, ref, report.output_vcf)]
    assert report.output_vcf.endswith(".vcf")


def test_a_caller_supplied_output_path_reaches_freebayes_without_reserving_a_temp(
        tmp_path, monkeypatch):
    sam = write_sam(tmp_path, [sam_line(f"r{i}", 0, ALT_SEQ) for i in range(4)])
    ref = write_ref(tmp_path)
    given = os.path.join(str(tmp_path), "given.vcf")
    monkeypatch.setattr(vc, "_has_tool", lambda name: name == "freebayes")
    reserved = []
    monkeypatch.setattr(vc, "secure_temp_path", lambda suffix="": reserved.append(suffix))
    used = []
    monkeypatch.setattr(vc, "_freebayes_call",
                        lambda bam, reference, output: used.append(output) or True)

    report = vc.call_variants(sam, reference_file=ref, output_vcf=given)

    assert used == [given], "the caller's path must be the one the tool writes to"
    assert reserved == [], "no temporary path is needed when the caller supplied one"
    assert (report.tool, report.engine) == ("freebayes", "freebayes")
    assert report.output_vcf == given


def test_a_freebayes_run_that_fails_falls_through_to_the_builtin_caller(tmp_path, monkeypatch):
    sam = write_sam(tmp_path, [sam_line(f"r{i}", 0, ALT_SEQ) for i in range(4)]
                    + [sam_line(f"u{i}", 0, REF_SEQ) for i in range(4)])
    ref = write_ref(tmp_path)
    given = os.path.join(str(tmp_path), "fallthrough.vcf")
    monkeypatch.setattr(vc, "_has_tool", lambda name: name == "freebayes")
    monkeypatch.setattr(vc, "_freebayes_call", lambda *args: False)

    report = vc.call_variants(sam, reference_file=ref, output_vcf=given)

    # the reference is readable, so the built-in caller uses its pileup engine
    assert (report.tool, report.engine) == ("builtin_pileup", "builtin")
    assert report.output_vcf == given
    assert os.path.exists(given)
    assert report.total_variants == 1


def test_without_freebayes_the_temporary_vcf_path_is_reserved_before_the_builtin_runs(
        tmp_path, monkeypatch):
    sam = write_sam(tmp_path, [sam_line(f"r{i}", 0, ALT_SEQ) for i in range(4)]
                    + [sam_line(f"u{i}", 0, REF_SEQ) for i in range(4)])
    reserved = os.path.join(str(tmp_path), "reserved.vcf")
    seen = []
    monkeypatch.setattr(vc, "secure_temp_path",
                        lambda suffix="": seen.append(suffix) or reserved)
    monkeypatch.setattr(vc, "_has_tool", lambda name: False)
    report = vc.call_variants(sam)
    assert seen == [".vcf"]
    assert report.output_vcf == reserved
    assert os.path.exists(reserved) and report.total_variants == 1
    lines = open(reserved, encoding="utf-8").read().splitlines()
    assert [l for l in lines if not l.startswith("#")][0].startswith("chr1\t6\t.\t")


def test_a_reference_that_is_absent_from_disk_never_reaches_freebayes(tmp_path, monkeypatch):
    sam = write_sam(tmp_path, [sam_line(f"r{i}", 0, ALT_SEQ) for i in range(4)]
                    + [sam_line(f"u{i}", 0, REF_SEQ) for i in range(4)])
    absent = os.path.join(str(tmp_path), "no.fa")
    monkeypatch.setattr(vc, "_has_tool", lambda name: True)
    monkeypatch.setattr(vc, "_freebayes_call",
                        lambda *args: pytest.fail("freebayes must not run"))
    report = vc.call_variants(sam, reference_file=absent, output_vcf=None)
    assert report.tool == "builtin" and report.engine == "builtin"
    assert report.message == f"Reference FASTA could not be read: {absent}"
    assert report.variants == []


# ── filtering and summaries ──────────────────────────────────────────────────

def variant(pos, quality=60, depth=20, alt_count=10, vtype="SNP"):
    return vc.Variant("chr1", pos, "A", "C", quality, depth, alt_count, "0/1", vtype)


VARIANTS = [variant(1, quality=19),                 # fails the quality floor
            variant(2, depth=4, alt_count=2),       # fails the depth floor
            variant(3, alt_count=20),               # af 1.00, above max_allele_freq
            variant(4, alt_count=19),               # af 0.95, exactly at the limit
            variant(5, alt_count=5)]                # af 0.25


def test_the_filter_drops_low_quality_shallow_and_near_fixed_sites_independently():
    kept = vc.filter_variants(VARIANTS)
    assert [v.pos for v in kept] == [4, 5]
    assert kept[0].alt_count / kept[0].depth == pytest.approx(0.95)
    assert kept[1].alt_count / kept[1].depth == pytest.approx(0.25)


@pytest.mark.parametrize("kwargs,expected", [
    ({"min_quality": 0, "min_depth": 0, "max_allele_freq": 1.0}, [1, 2, 3, 4, 5]),
    ({"min_depth": 21}, []),
    ({"max_allele_freq": 0.3}, [5]),
    # One threshold loosened is not the whole filter loosened: the depth floor still
    # drops the 4-read site and the frequency ceiling still drops the fixed one.
    ({"min_quality": 19}, [1, 4, 5]),
    ({"min_quality": 20}, [4, 5]),
])
def test_every_filter_threshold_can_be_loosened_or_tightened(kwargs, expected):
    assert [v.pos for v in vc.filter_variants(VARIANTS, **kwargs)] == expected


def test_a_zero_depth_variant_is_never_treated_as_an_allele_of_frequency_one():
    zero = vc.Variant("chr1", 9, "A", "C", 99, 0, 0, "0/0", "SNP")
    assert [v.pos for v in vc.filter_variants([zero], min_depth=0)] == [9]


def test_the_summary_prints_every_counter_and_the_caller_note():
    report = vc.VariantReport(tool="builtin_pileup", engine="builtin", total_variants=3,
                              snps=2, indels=1, ti_tv_ratio=2.0, message="Built-in caller")
    assert vc.variant_summary(report) == (
        "=== Variant Calling Report ===\nEngine: builtin\nTotal variants: 3\n"
        "SNPs: 2\nIndels: 1\nTi/Tv ratio: 2.00\nNote: Built-in caller")


def test_the_summary_omits_the_note_line_when_the_caller_had_nothing_to_say():
    assert vc.variant_summary(vc.VariantReport(tool="none", engine="none")) == (
        "=== Variant Calling Report ===\nEngine: none\nTotal variants: 0\nSNPs: 0\n"
        "Indels: 0\nTi/Tv ratio: 0.00")


# ── structural variants and copy number ─────────────────────────────────────

def sv(chrom, start, end, sv_type="DEL", confidence=0.5, support=1):
    return vc.StructuralVariant(chrom, start, end, sv_type, end - start,
                                confidence, support)


def test_no_coverage_produces_no_structural_variants():
    assert vc.detect_structural_variants(np.array([])) == []
    assert vc.detect_structural_variants(np.zeros(0), ref_coverage=np.zeros(0)) == []


def test_a_deletion_window_is_reported_once_and_merged_across_half_step_windows():
    coverage = np.ones(300) * 10.0
    coverage[100:200] = 1.0
    svs = vc.detect_structural_variants(coverage, chrom="chr7", window_size=50)
    assert [(s.chrom, s.sv_type, s.start, s.end, s.size, s.support_reads) for s in svs] == [
        ("chr7", "DEL", 100, 200, 100, 100)]
    assert svs[0].confidence == pytest.approx(0.8)


def test_a_duplication_window_is_clamped_at_full_confidence():
    coverage = np.ones(300) * 10.0
    coverage[100:200] = 40.0
    svs = vc.detect_structural_variants(coverage, window_size=50)
    assert [(s.sv_type, s.start, s.end, s.size) for s in svs] == [("DUP", 75, 225, 150)]
    assert svs[0].confidence == pytest.approx(1.0)
    assert svs[0].support_reads == 4500, "the merged span holds 50x10 + 100x40 + 50x10 bases"


def test_different_event_types_are_never_merged_into_one_variant():
    coverage = np.ones(200) * 10.0
    coverage[0:50] = 40.0
    coverage[100:150] = 1.0
    svs = vc.detect_structural_variants(coverage, window_size=50)
    assert [(s.sv_type, s.start, s.end) for s in svs] == [("DUP", 0, 75), ("DEL", 100, 150)]


def test_overlapping_windows_of_the_same_event_are_merged_into_one_span():
    coverage = np.ones(300) * 10.0
    coverage[100:200] = 1.0
    loose = vc.detect_structural_variants(coverage, window_size=25, del_threshold=0.9)
    assert len(loose) == 1
    assert (loose[0].sv_type, loose[0].start, loose[0].end, loose[0].size) == (
        "DEL", 84, 217, 133)
    assert loose[0].confidence == pytest.approx((0.9 - 0.1) / 0.9), \
        "the deleted region sits at a tenth of baseline, so it is 8/9 of the way to zero"
    assert loose[0].support_reads == 430, \
        "16x10 + 100x1 + 17x10 over the merged span, not the sum of the overlapping windows"


def test_merged_windows_do_not_count_the_same_bases_twice():
    """The windows step by half a window, so adding the members' support counted
    every overlapped base twice: this fixture reported 8500 reads of support for a
    span holding 4500."""
    coverage = np.ones(300) * 10.0
    coverage[100:200] = 40.0
    merged = vc.detect_structural_variants(coverage, window_size=50)
    assert len(merged) == 1
    assert merged[0].support_reads == int(coverage[merged[0].start:merged[0].end].sum())


def test_variants_on_different_contigs_stay_separate_after_sorting():
    merged = vc._merge_svs([sv("chrB", 0, 10, confidence=0.6, support=2),
                            sv("chrA", 0, 10, support=1)])
    assert [(s.chrom, s.start, s.end, s.confidence, s.support_reads) for s in merged] == [
        ("chrA", 0, 10, 0.5, 1), ("chrB", 0, 10, 0.6, 2)]
    assert vc._merge_svs([]) == []


def test_a_gap_beyond_the_merge_distance_starts_a_new_variant():
    merged = vc._merge_svs([sv("chr1", 0, 10), sv("chr1", 5000, 5010)])
    assert [(s.start, s.end) for s in merged] == [(0, 10), (5000, 5010)]
    touching = vc._merge_svs([sv("chr1", 0, 10, support=1), sv("chr1", 40, 60, support=3)])
    assert [(s.start, s.end, s.size, s.support_reads) for s in touching] == [(0, 60, 60, 4)]


def test_the_sv_report_counts_types_and_lists_at_most_twenty_details():
    many = [sv("chr1", i, i + 10, "DEL" if (i // 10) % 2 == 0 else "DUP")
            for i in range(0, 220, 10)]
    lines = vc.format_sv_report(many).splitlines()
    assert lines[0] == "=== Structural Variant Report ===" and lines[1] == "Total SVs: 22"
    assert lines[2:4] == ["  DEL: 11", "  DUP: 11"]
    head = lines.index("Details:")
    listed = [line for line in lines[head + 1:] if line.startswith("  chr1:")]
    assert len(listed) == 20, "22 variants in, twenty rows of detail out"
    assert listed[0] == "  chr1:0-10 DEL size=10 conf=0.50"
    assert vc.format_sv_report([]).splitlines()[:2] == [
        "=== Structural Variant Report ===", "Total SVs: 0"]


def test_empty_coverage_yields_an_empty_cnv_frame_with_no_columns():
    frame = vc.detect_cnv(np.array([]))
    assert frame.empty and list(frame.columns) == []


def test_a_matching_reference_bin_is_two_copies_even_below_one_x():
    # every bin equals its reference (median 0.5x), so CN is 2 and log2 is 0
    frame = vc.detect_cnv(np.array([0.5] * 100), window_size=100)
    assert frame.to_dict(orient="records") == [
        {"chrom": "chr1", "start": 0, "end": 100, "cn": 2, "log2_ratio": 0.0,
         "coverage": 0.5}]


def test_a_given_reference_divides_the_copy_number_directly():
    frame = vc.detect_cnv(np.array([5.0] * 100), reference_coverage=np.array([2.5] * 100),
                          window_size=100)
    assert frame.iloc[0]["cn"] == 4
    assert frame.iloc[0]["log2_ratio"] == pytest.approx(0.778)


def test_a_zero_reference_coverage_is_read_as_a_normal_copy_number():
    frame = vc.detect_cnv(np.array([5.0] * 100), reference_coverage=np.zeros(100),
                          window_size=100)
    assert frame.to_dict(orient="records") == [
        {"chrom": "chr1", "start": 0, "end": 100, "cn": 2, "log2_ratio": 0,
         "coverage": 5.0}]


def test_a_reference_shorter_than_the_sample_stops_where_the_reference_stops():
    """Bins past the end of the reference used to substitute the sample's own
    coverage for the missing comparison and print CN=2, log2_ratio=0.0 — a measured
    normal copy number for territory nothing was measured against."""
    frame = vc.detect_cnv(np.array([5.0] * 30), reference_coverage=np.array([2.5] * 10),
                          window_size=10)
    assert [(row["start"], row["end"], row["cn"], row["log2_ratio"])
            for _, row in frame.iterrows()] == [(0, 10, 4, 0.778)]


@pytest.mark.parametrize("window_size", [0, -5])
def test_a_non_positive_window_size_is_refused_rather_than_warned(window_size):
    """window_size=0 made detect_structural_variants average empty slices and
    answer [] behind a screenful of RuntimeWarnings, and detect_cnv raised
    ZeroDivisionError from its own binning."""
    with pytest.raises(ValueError, match="window_size must be at least 1"):
        vc.detect_structural_variants(np.ones(10), window_size=window_size)
    with pytest.raises(ValueError, match="window_size must be at least 1"):
        vc.detect_cnv(np.ones(10), window_size=window_size)


def test_the_cnv_report_labels_copy_number_classes_and_names_the_unmapped_ones():
    frame = vc.detect_cnv(np.array([0.0, 0.0, 8.0, 8.0, 4.0, 4.0]), window_size=2)
    assert vc.format_cnv_report(frame) == (
        "=== CNV Report ===\nTotal bins: 3\n"
        "  CN=0 (DEL): 1 bins\n  CN=2 (Normal): 1 bins\n  CN=4 (DUP): 1 bins")
    odd = vc.detect_cnv(np.array([10.0] * 4), reference_coverage=np.array([1.0] * 4),
                        window_size=2)
    assert "CN=20 (CN=20): 2 bins" in vc.format_cnv_report(odd)
    assert vc.format_cnv_report(vc.detect_cnv(np.array([]))) == "No CNV data."
