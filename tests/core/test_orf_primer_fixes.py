"""Regression tests for orf_finder.py review fixes."""
import pytest

from biosuite.core.orf_finder import (
    find_orfs, find_restriction_sites, design_primers, _calculate_tm,
    ORF,
)
from biosuite.core.utils import reverse_complement_dna as rc


def test_restriction_reverse_strand_not_duplicated():
    seq = 'GAAGAC' + 'T' * 20 + 'GAAGAC' + 'T' * 20 + 'GTCTTC'
    sites = find_restriction_sites(seq, ['BbsI'])
    plus = [s for s in sites if s.strand == '+']
    minus = [s for s in sites if s.strand == '-']
    assert len(plus) == 2 and len(minus) == 1  # minus was 2 (loop bug)


def test_palindromic_enzyme_single_set():
    seq = 'G' * 5 + 'GAATTC' + 'A' * 10 + 'GAATTC'
    sites = find_restriction_sites(seq, ['EcoRI'])
    assert len(sites) == 2 and all(s.strand == '+' for s in sites)


def test_find_orfs_atg_mode_matches_old_behavior():
    seq = 'TTT' + 'ATG' + 'AAA' * 10 + 'TAA' + 'CCC'
    orfs = find_orfs(seq, min_length=1, include_start=True)
    assert len(orfs) == 1
    assert orfs[0].has_start_codon and orfs[0].has_stop_codon
    assert orfs[0].protein == 'M' + 'K' * 10


def test_find_orfs_stop_to_stop_default():
    # No ATG anywhere -> old code (param ignored) returned []; default
    # include_start=False must now report stop-to-stop segments
    # (one per translation frame, so multiple ORFs is expected).
    seq = 'TAA' + 'AAA' * 10 + 'TGA' + 'CCC' * 12 + 'TGA'
    orfs = find_orfs(seq, min_length=5)
    assert len(orfs) >= 1
    # full stop-to-stop segments (not starting with M) are now reported...
    assert any(o.protein.startswith('K' * 5) and not o.has_start_codon
               for o in orfs)
    # ...and a nested in-frame ATG also gets its own ORF entry
    assert any(o.has_start_codon for o in orfs)


def test_find_orfs_end_coordinate_includes_stop():
    seq = 'ATG' + 'GCT' * 5 + 'TAA'
    (orf,) = find_orfs(seq, min_length=1, include_start=True)
    assert orf.end == len(seq)  # stop codon included


def test_primers_anchor_to_flanks():
    # GC-balanced template so many candidates pass; ensure fwd near start,
    # rev near end — the old scoring pulled both to the middle.
    import random
    random.seed(5)
    flank_f = 'GCGCATGCGCATGCGCATGC' + 'ATGCAT' * 6
    mid = 'TATATATA' * 30
    flank_r = 'CGCGCGCGCGCGCGCGCGCG'
    template = flank_f + mid + flank_r
    fwd, rev = design_primers(template, 0, len(template),
                              primer_length=20, min_tm=50, max_tm=75,
                              gc_range=(25, 75))
    assert fwd is not None and rev is not None
    assert fwd.position < len(template) // 2
    assert rev.position > len(template) // 2
    assert fwd.strand == '+' and rev.strand == '-'


def test_reverse_primer_is_revcomp_of_template():
    import random
    random.seed(6)
    template = ''.join(random.choice('ACGT') for _ in range(200))
    template = 'GCGCGCGCGCGCGCGCGCGC' + template + 'GCGCGCGCGCGCGCGCGCGC'
    fwd, rev = design_primers(template, 0, len(template),
                              primer_length=20, min_tm=40, max_tm=90,
                              gc_range=(20, 100))
    if rev is not None:
        region_len = rev.length
        segment = template[rev.position:rev.position + region_len]
        assert rc(segment) == rev.sequence


def test_tm_wallace_short_primers():
    assert _calculate_tm("ATGC") == 2 * 2 + 4 * 2  # 2 AT + 2 GC -> 12
    long_tm = _calculate_tm("GCGCATGCGCATGCGCATGC")
    assert 55 <= long_tm <= 80
