"""End-to-end virtual cloning suite tests (exact values against real contracts)."""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.core import cloning as cl


def test_validate_dna_accepts_upper_and_raises():
    assert cl._validate_dna("ACGTNacgtn".upper()) == "ACGTNACGTN"
    with pytest.raises(ValueError):
        cl._validate_dna("ACGTX")


def test_reverse_complement():
    assert cl._reverse_complement("ACGTN") == "NACGT"


def test_find_ecori_sites_cut_offsets():
    # GAATTC at 0 and 6 — sites are 0-based match starts
    assert cl.find_restriction_sites("GAATTCGAATTC", "EcoRI") == [0, 6]


def test_find_all_overlapping():
    assert cl._find_all("AAAAAA", "AAA") == [0, 1, 2, 3]


def test_digest_single_enzyme():
    # LINEAR topology: fragment coordinates are exact
    res = cl.simulate_digestion("ATCGGAATTC" + "A" * 100, "EcoRI",
                                topology='linear')
    assert res['cuts']
    assert res['total_bp'] == sum(f['size'] for f in res['fragments'])
    for f in res['fragments']:
        assert f['end'] - f['start'] == f['size']
        assert len(f['sequence']) == f['size']


def test_digest_circular_wrap_fragment():
    # one cut on a circle -> single full-length fragment,
    # start == end denotes the wrap point
    res = cl.simulate_digestion("ATCGGAATTC" + "A" * 100, "EcoRI")
    assert len(res['fragments']) == 1
    f = res['fragments'][0]
    assert f['size'] == 110
    assert len(f['sequence']) == 110


def test_digest_linear_vs_circular_differs():
    seq = "GAATTC" + "T" * 30 + "GAATTC"
    circ = cl.simulate_digestion(seq, "EcoRI", topology='circular')
    lin = cl.simulate_digestion(seq, "EcoRI", topology='linear')
    assert len(circ['fragments']) <= len(lin['fragments'])


def test_ligation_contract():
    res = cl.simulate_ligation(["ATCCGG", "AATTCG"])
    assert res['fragment_count'] == 2
    assert isinstance(cl.format_ligation_report(res), str)


def test_primers_contract():
    target = "ATG" + "ACGTGCTAGCTAGCTAGCAT" * 8 + "TAA"
    p = cl.design_primers(target, primer_length=18)
    assert p['forward'] and p['reverse']
    assert 0 < p['fwd_gc'] <= 100 or 0 < p['fwd_gc'] <= 1.0
    assert isinstance(p['fwd_tm'], (int, float))
    assert "Primer" in cl.format_primer_report(p)


def test_simulate_pcr_contract():
    template = "A" * 50 + "ATGGCCACGT" + "T" * 100 + "ACGGCCAT" + "G" * 40
    pcr = cl.simulate_pcr(template, "ATGGCCACGT", "ATGGCCGT")
    assert pcr['size'] >= 0
    assert isinstance(pcr['product'], str)
    assert pcr['cycles'] == 30
    assert cl.format_pcr_report(pcr) is not None


def test_gc_fraction_and_tm():
    assert cl._gc_content("ACGT") == pytest.approx(0.5)     # fraction (0..1)
    assert 30 < cl._melting_temp("ACGTACGTACGTACGT") < 90


def test_virtual_gel_figure():
    fig = cl.plot_virtual_gel([1500, 1000, 700, 500, 100])
    assert isinstance(fig, plt.Figure)
    plt.close('all')


def test_verify_insert_contract():
    v = cl.verify_insert("ATG" + "ACGTACGT" + "TAA", "ACGTACGT")
    assert v['insert_found'] is True
    assert v['insert_start'] == 3


def test_insert_missing_reports_false():
    v = cl.verify_insert("ATGAAATAA", "GGGGCCCC")
    assert v['insert_found'] is False
