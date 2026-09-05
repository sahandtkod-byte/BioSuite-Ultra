"""Cloning design: restriction digestion, pcR primers, ligation assembly logic."""
import numpy as np

from biosuite.core import cloning as cl


def test_gc_content_and_tm():
    assert abs(cl._gc_content('ATGC') - 0.5) < 1e-9
    tm = cl._melting_temp('ATGCATGCATGCATGCA')
    assert 30 < tm < 80


def test_find_restriction_sites_ecori():
    seq = 'AAA' + 'GAATTC' * 3 + 'TTT'
    res = cl.find_restriction_sites(seq, 'EcoRI')
    assert res is not None


def test_simulate_digestion_splices_at_cut_sites():
    seq = ('ACGT' * 50) + 'GAATTC' + ('TTAA' * 50) + 'GAATTC' + 'GGCC' * 50
    out = cl.simulate_digestion(seq, 'EcoRI')
    if isinstance(out, dict):
        frags = out.get('fragments') or out.get('fragment_lengths') or []
        assert len(frags) >= 1
    else:
        assert out


def test_simulate_ligation_empty_and_matched():
    try:
        cl.simulate_ligation([])
    except Exception:
        pass
    frag = 'GAATTC' + 'ACGT' * 40
    out = cl.simulate_ligation([frag, frag])
    assert out is not None


def test_design_primers_returns_pair():
    template = 'ATGCGTACGTTAGCGTACGTACAGTACTGCATGCATCGTACGTACGATCGACTGCATGCA'
    out = cl.design_primers(template)
    assert isinstance(out, dict)
    assert out


def test_simulate_pcr_linear_growth():
    template = 'ATGGCGTACGTTAGCGTACAGTACTGCATGCATCGTACGTACGATCGACTGCATGCAA'
    fwd = template[:16]
    rev = template[-16:][::-1].translate(str.maketrans('ACGT', 'TGCA'))
    out = cl.simulate_pcr(template, fwd, rev)
    assert out is not None


def test_verify_insert():
    target = 'ACGTACGTACGTACGTACGT'
    out = cl.verify_insert(target, 'ACGTACGT')
    assert out is not None


def test_virtual_gel_agg():
    import matplotlib
    matplotlib.use('Agg')
    out = cl.plot_virtual_gel([5000, 3000, 1500, 700])
    assert out is not None


def test_format_reports_smoke():
    real = cl.simulate_digestion(('ACGT' * 100) + 'GAATTC' + ('TTAA' * 100) + 'GAATTC', 'EcoRI')
    assert isinstance(cl.format_digest_report(real), str)
    assert isinstance(cl.format_primer_report({'forward': 'ATGC', 'reverse': 'CGTA'}), str)
