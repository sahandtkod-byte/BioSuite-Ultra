"""Regression tests for the crispr.py review fixes.

1. Minus-strand guides must be reported 5'->3' in gRNA orientation
   (the previous code returned the forward-genome strand — unusable for
   ordering and inconsistent with the '+' branch).
2. ``position`` for '-' sites points at the forward start of the protospacer
   region (used to point at the PAM, off by len(PAM)).
3. 'reverse'-directed PAMs (Cas12a TTTV) place the protospacer downstream.
4. The public ``design_guides`` default ('auto') must never fall through to
   the un-parsable CRISPOR stub and return an empty guide list.
"""

import pytest

from biosuite.core.crispr import _find_pam_sites, design_guides
from biosuite.core.utils import reverse_complement_dna as rc


FWD_GUIDE = "ACGTACGTACGTACGTACGT"
MINUS_REGION = "GATTACAGATTACAGATTAC"


def _plus_seq():
    return "TTTT" * 10 + FWD_GUIDE + "AGG" + "CCCC" * 10


def _minus_seq():
    return "TTTT" * 10 + "CCA" + MINUS_REGION + "GGGG" * 10


def test_plus_strand_unchanged():
    sites = _find_pam_sites(_plus_seq(), 'NGG', 20)
    plus = [s for s in sites if s['strand'] == '+' and s['position'] == 40]
    assert plus and plus[0]['guide'] == FWD_GUIDE


def test_minus_strand_guide_is_grna_orientation():
    seq = _minus_seq()
    minus = [s for s in _find_pam_sites(seq, 'NGG', 20) if s['strand'] == '-']
    assert len(minus) == 1
    m = minus[0]
    assert m['guide'] == rc(MINUS_REGION)
    # guide + PAM must be contiguous on the PAM-bearing (reverse) strand
    assert m['guide'] + m['pam'] in rc(seq)


def test_minus_strand_position_is_protospacer_region_start():
    seq = _minus_seq()
    minus = [s for s in _find_pam_sites(seq, 'NGG', 20) if s['strand'] == '-']
    assert minus[0]['position'] == 43  # forward start of the 20 nt region


def test_cas12a_reverse_directed_downstream_protospacer():
    down = MINUS_REGION
    seq = "TTTT" * 10 + "TTTA" + down + "GGGG" * 10
    sites = _find_pam_sites(seq, 'TTTV', 20, direction='reverse')
    hit = [s for s in sites if s['strand'] == '+' and s['position'] == 44]
    assert hit and hit[0]['guide'] == down


def test_pam_at_sequence_boundary_not_dropped():
    # Reverse-directed PAM at position 0 must still yield a guide.
    down = MINUS_REGION
    seq = "TTTA" + down + "GGGG" * 10
    sites = _find_pam_sites(seq, 'TTTV', 20, direction='reverse')
    assert any(s['strand'] == '+' and s['position'] == 4 for s in sites)


def test_forward_guide_at_boundary_not_dropped():
    seq = FWD_GUIDE + "AGG" + "CCCC" * 10
    sites = _find_pam_sites(seq, 'NGG', 20)
    assert any(s['strand'] == '+' and s['position'] == 0 for s in sites)


def test_design_guides_never_returns_empty_via_stub():
    res = design_guides(_plus_seq())  # tool='auto' default
    assert res.engine == 'builtin'
    assert res.num_guides > 0
    assert any(g.strand == '+' for g in res.guides)


def test_design_guides_empty_sequence_message():
    res = design_guides("")
    assert res.engine == 'none'
    assert res.guides == []


def test_public_api_minus_guides_are_grna_orientation():
    seq = _minus_seq()
    res = design_guides(seq)
    minus = [g for g in res.guides if g.strand == '-']
    assert minus
    for g in minus:
        assert g.sequence + g.pam in rc(seq)
        assert seq[g.position:g.position + len(g.sequence)] == rc(g.sequence)


def test_sites_sorted_by_position():
    seq = _plus_seq() + _minus_seq()
    sites = _find_pam_sites(seq, 'NGG', 20)
    positions = [s['position'] for s in sites]
    assert positions == sorted(positions)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
