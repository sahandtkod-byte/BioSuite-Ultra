"""The text reports, and the fields they refuse to invent.

Each formatter used `.get(key, default)` for numbers the producing function
always returns, so a result dict that lost a key printed "Tm: 0.0 °C" or
"position N/A" as though the reaction had been measured.  They now index the
required fields, and only the genuinely optional 'warnings'/'overhangs' stay
optional.
"""
import pytest

from biosuite.core import cloning as cl

ONE_SITE = "ATCGGAATTC" + "A" * 100
PRIMER_TEMPLATE = "ATG" + "ACGTGCTAGCTAGCTAGCAT" * 8 + "TAA"
# A 60 nt target whose two primers are balanced, so the pair needs no advice.
CLEAN_TEMPLATE = "ATGCGTACGTTAGCGTACGTACAGTACTGCATGCATCGTACGTACGATCGACTGCATGCA"
PCR_TEMPLATE = "A" * 50 + "ATGGCCACGT" + "T" * 100 + "ACGGCCAT" + "G" * 40


def test_a_digest_report_prints_every_fragment_and_balances_the_total():
    text = cl.format_digest_report(cl.simulate_digestion(ONE_SITE, "EcoRI",
                                                         topology="linear"))
    assert text == "\n".join([
        "=" * 60,
        "  RESTRICTION DIGESTION REPORT",
        "=" * 60,
        "  Total input: 110 bp",
        "  Enzymes: EcoRI",
        "-" * 60,
        "  Enzyme overhangs:",
        "    EcoRI: 5prime",
        "-" * 60,
        "  Cut sites (1):",
        "    Position 5 (EcoRI, offset +1)",
        "-" * 60,
        "  Fragments: 2",
        "  #     Size (bp)",
        "  ———— ——————————",
        "  1           105",
        "  2             5",
        "  ———— ——————————",
        "  Total        110",
        "=" * 60,
    ])


def test_a_primer_report_prints_the_measured_pair():
    text = cl.format_primer_report(cl.design_primers(PRIMER_TEMPLATE, primer_length=18))
    assert text.splitlines() == [
        "=" * 60,
        "  PRIMER DESIGN REPORT",
        "=" * 60,
        "",
        "  Forward Primer: 5'-ATGACGTGCTAGCTAGCT-3'",
        "    Length: 18 nt   Tm: 48.0 °C   GC: 50.0%",
        "",
        "  Reverse Primer: 5'-TTAATGCTAGCTAGCTAG-3'",
        "    Length: 18 nt   Tm: 43.5 °C   GC: 38.9%",
        "",
        "  ΔTm: 4.5 °C",
        "  ✓ Good match",
        "",
        "  Warnings:",
        "    ⚠ Reverse primer GC content (38.9%) outside ideal range (40-60%).",
        "=" * 60,
    ]


def test_a_pcr_report_prints_the_positions_it_was_given():
    pcr = cl.simulate_pcr(PCR_TEMPLATE, "ATGGCCACGT", "ATGGCCGT")
    assert cl.format_pcr_report(pcr) == "\n".join([
        "=" * 60,
        "  PCR AMPLIFICATION REPORT",
        "=" * 60,
        "  Template length: N/A (see product size)",
        "  Product size: 118 bp",
        "  Cycles: 30",
        "  Estimated efficiency: 95%",
        "  Forward primer at: position 50",
        "  Reverse primer RC at: position 160",
        "-" * 60,
        "  ✓ No warnings",
        "=" * 60,
    ])


@pytest.mark.parametrize("key", ["fwd_tm", "rev_tm", "fwd_gc", "rev_gc"])
def test_a_primer_report_names_the_measurement_it_was_never_given(key):
    pair = {"forward": "ATGC", "reverse": "CGTA",
            "fwd_tm": 50.0, "rev_tm": 50.0, "fwd_gc": 50.0, "rev_gc": 50.0}
    del pair[key]
    with pytest.raises(KeyError) as exc:
        cl.format_primer_report(pair)
    assert str(exc.value) == f"'{key}'"


def test_a_primer_report_for_a_clean_pair_prints_no_warnings_section():
    text = cl.format_primer_report(cl.design_primers(CLEAN_TEMPLATE))
    assert "  ΔTm: 2.0 °C" in text
    assert "  ✓ Good match" in text
    assert "Warnings:" not in text
    assert "⚠" not in text


def test_a_pcr_report_refuses_a_result_without_binding_positions():
    with pytest.raises(KeyError) as exc:
        cl.format_pcr_report({"size": 10, "cycles": 30, "efficiency": 0.9})
    assert str(exc.value) == "'fwd_pos'"


def test_a_ligation_report_refuses_a_result_without_its_verification_flag():
    with pytest.raises(KeyError) as exc:
        cl.format_ligation_report({"fragment_count": 1, "size": 8,
                                   "is_circular": True, "efficiency": 0.95})
    assert str(exc.value) == "'verified'"


def test_a_ligation_report_still_treats_the_notes_list_as_optional():
    text = cl.format_ligation_report({"fragment_count": 1, "size": 8, "is_circular": True,
                                      "efficiency": 0.95, "verified": True})
    assert "  Fragments ligated: 1" in text
    assert "  Estimated efficiency: 95.0%" in text
    assert "  ✓ Insert verified in product" in text
    assert "Notes:" not in text
