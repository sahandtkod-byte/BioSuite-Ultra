"""Edge and failure paths of the cloning toolkit: rejected input, duplicate and
origin-landing cut positions, invalid ligated fragments, unfindable primers,
truncated inserts, gel lanes with nothing to draw and the report branches that
describe each of them.

These are the places where a coordinate can be off by one or a warning can be
computed and then thrown away, so every position, size and message below was
read off the running code rather than derived.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from biosuite.core import cloning as cl

# A template whose KpnI site (GGTACC) straddles the origin: it ends the string
# as "GGTAC" and completes with the leading "C", and its cut (offset 5) lands on
# index 0 of the circle.
ORIGIN_CUT_SEQ = "C" + "AT" * 8 + "GGTAC"          # 22 bp, KpnI cut_pos == 0
ORIGIN_CUT_SEQ2 = "C" + "GAATTC" + "AA" + "GGTAC"  # 14 bp, cuts at 0 and 2


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _random_template(length, seed):
    import random
    rnd = random.Random(seed)
    return "".join(rnd.choice("ACGT") for _ in range(length))


# ── input validation ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("call,name", [
    (lambda: cl._validate_dna("", "insert"), "insert"),
    (lambda: cl.simulate_digestion("", "EcoRI"), "digestion template"),
    (lambda: cl.find_restriction_sites("  ", "EcoRI"), "search sequence"),
    (lambda: cl.design_primers("", "10"), "primer template"),
    (lambda: cl.simulate_pcr("", "A", "T"), "PCR template"),
    (lambda: cl.verify_insert("", "ACGT"), "product"),
])
def test_every_entry_point_rejects_an_empty_sequence_by_name(call, name):
    with pytest.raises(ValueError) as exc:
        call()
    assert str(exc.value) == f"{name} is empty"


def test_rejected_characters_are_reported_sorted_and_once():
    with pytest.raises(ValueError) as exc:
        cl.find_restriction_sites("ACG!!XY9", "EcoRI")
    assert str(exc.value) == (
        "Invalid characters in search sequence: !9X. "
        "Only IUPAC nucleotide codes are accepted."
    )


def test_unknown_enzyme_names_are_refused_with_an_available_list():
    with pytest.raises(ValueError) as exc:
        cl.find_restriction_sites("ACGT", "NotAnEnzyme")
    assert str(exc.value).startswith(
        "Unknown enzyme: NotAnEnzyme. Available: AatII, Acc65I, AflII, ")


# ── restriction site search over the whole table ─────────────────────────────

def test_search_without_an_enzyme_returns_only_the_enzymes_that_actually_hit():
    res = cl.find_restriction_sites("AAAGAATTCGGATCCC")
    assert isinstance(res, dict)
    assert {k: v for k, v in res.items()
            if k in ("EcoRI", "ApoI", "BamHI", "MluCI")} == {
                "EcoRI": [3], "ApoI": [3], "BamHI": [9], "MluCI": [4]}
    assert all(positions for positions in res.values()), res
    assert "HindIII" not in res  # AAGCTT is absent from the sequence


# ── digestion: duplicate and origin-landing cut positions ────────────────────

@pytest.mark.parametrize("order,attributed_to", [(["ZraI", "AatII"], "ZraI"),
                                                 (["AatII", "ZraI"], "AatII")])
def test_two_isoschizomers_cutting_the_same_base_produce_one_cut(order, attributed_to):
    seq = "TTTT" + "GACGTC" + "A" * 20
    res = cl.simulate_digestion(seq, order, topology="linear")
    assert [c["cut_pos"] for c in res["cuts"]] == [5]
    assert res["cuts"][0]["enzyme"] == attributed_to
    assert res["cuts"][0]["site_start"] == 4
    # both enzymes are still reported as used, and both overhangs classified
    assert res["enzymes_used"] == order
    assert res["overhangs"] == {order[0]: "5prime", order[1]: "5prime"}
    assert res["sizes"] == [5, 25]


def test_a_linear_digest_never_yields_an_empty_fragment():
    seq = "TTTT" + "GACGTC" + "A" * 20
    res = cl.simulate_digestion(seq, ["AatII", "EcoRI"], topology="linear")
    assert [f["size"] for f in res["fragments"]] == [5, 25]
    assert all(f["end"] > f["start"] for f in res["fragments"])
    assert "".join(f["sequence"] for f in res["fragments"]) == seq.upper()
    assert sum(res["sizes"]) == res["total_bp"] == len(seq)


def test_a_site_that_straddles_the_origin_is_cut_exactly_at_position_zero():
    res = cl.simulate_digestion(ORIGIN_CUT_SEQ, "KpnI", topology="circular")
    assert [(c["site_start"], c["cut_pos"]) for c in res["cuts"]] == [(17, 0)]
    assert res["fragments"] == [{"size": 22, "sequence": ORIGIN_CUT_SEQ,
                                 "start": 0, "end": 0}]
    assert res["overhangs"] == {"KpnI": "3prime"}


def test_an_origin_cut_splits_the_circle_together_with_an_internal_cut():
    res = cl.simulate_digestion(ORIGIN_CUT_SEQ2, ["EcoRI", "KpnI"], topology="circular")
    assert [(c["enzyme"], c["cut_pos"]) for c in res["cuts"]] == [("KpnI", 0), ("EcoRI", 2)]
    assert [(f["start"], f["end"], f["size"]) for f in res["fragments"]] == [
        (0, 2, 2), (2, 0, 12)]
    assert [f["sequence"] for f in res["fragments"]] == ["CG", "AATTCAAGGTAC"]
    assert sum(res["sizes"]) == len(ORIGIN_CUT_SEQ2)


def test_the_same_molecule_read_linearly_loses_the_origin_site():
    circ = cl.simulate_digestion(ORIGIN_CUT_SEQ2, ["EcoRI", "KpnI"], topology="circular")
    lin = cl.simulate_digestion(ORIGIN_CUT_SEQ2, ["EcoRI", "KpnI"], topology="linear")
    assert [c["enzyme"] for c in lin["cuts"]] == ["EcoRI"]
    assert [(f["start"], f["end"], f["size"]) for f in lin["fragments"]] == [
        (0, 2, 2), (2, 14, 12)]


# ── ligation ─────────────────────────────────────────────────────────────────

def test_an_unusable_fragment_is_skipped_with_its_own_number_and_reason():
    res = cl.simulate_ligation(["ACGT", "!!bad!!", "TTTT"])
    assert res["product"] == "ACGTTTTT"
    assert res["fragment_count"] == 2
    assert res["warnings"] == [
        "Fragment 2 skipped: Invalid characters in fragment 2: !. "
        "Only IUPAC nucleotide codes are accepted."]


def test_a_ligation_of_only_invalid_fragments_says_both_were_rejected():
    res = cl.simulate_ligation(["!!", "@@"], circular=False)
    assert res["product"] == "" and res["size"] == 0 and res["fragment_count"] == 0
    assert res["is_circular"] is False
    assert res["efficiency"] == 0.0 and res["verified"] is False
    assert res["warnings"] == [
        "Fragment 1 skipped: Invalid characters in fragment 1: !. "
        "Only IUPAC nucleotide codes are accepted.",
        "Fragment 2 skipped: Invalid characters in fragment 2: @. "
        "Only IUPAC nucleotide codes are accepted.",
        "All fragments were invalid.",
    ]


@pytest.mark.parametrize("circular,expected", [(False, 0.90), (True, 0.75)])
def test_partial_complementarity_at_a_junction_adds_three_percent(circular, expected):
    # CCCC|GGGG share a 2-base complement at the junction: 0.95 - 0.08 (+0.15) + 0.03
    res = cl.simulate_ligation(["AAAACCCC", "GGGGAAAA"], circular=circular)
    assert res["efficiency"] == pytest.approx(expected)
    assert res["product"] == "AAAACCCCGGGGAAAA"


def test_a_junction_without_complementarity_gets_no_bonus():
    assert cl.simulate_ligation(["ATATAT", "GCGCGC"], circular=False)["efficiency"] == pytest.approx(0.87)


def test_an_insert_carried_by_the_product_is_verified_without_warnings():
    vector, insert = "TTTT" * 10, "ACGT" * 6
    res = cl.simulate_ligation([vector, insert], insert_sequence=insert,
                               vector_sequence=vector)
    assert res["verified"] is True and res["warnings"] == []


def test_an_insert_ligated_backwards_is_still_verified():
    vector, insert = "TTTT" * 10, "ACGT" * 6
    res = cl.simulate_ligation([vector, cl._reverse_complement(insert)],
                               insert_sequence=insert, vector_sequence=vector)
    assert res["verified"] is True and res["warnings"] == []


def test_an_insert_split_across_a_junction_is_verified_with_a_caution():
    vector, insert = "TTTT" * 10, "ACGT" * 6
    half = len(insert) // 2
    res = cl.simulate_ligation([vector + insert[:half], "CGCG" * 3, insert[half:] + vector],
                               insert_sequence=insert, vector_sequence=vector)
    assert res["verified"] is True
    assert res["warnings"] == ["Insert found but may span a junction — verify orientation."]
    assert insert not in res["product"]


def test_a_missing_insert_is_announced_as_a_failed_ligation():
    vector, insert = "TTTT" * 10, "ACGT" * 6
    res = cl.simulate_ligation([vector, "GGGG" * 6], insert_sequence=insert,
                               vector_sequence=vector)
    assert res["verified"] is False
    assert res["warnings"] == ["WARNING: Insert NOT detected in ligated product! "
                               "Ligation may have failed or insert was lost."]


def test_an_insert_of_at_most_nine_bases_is_never_searched_split_across_a_junction():
    # half = 4 is not > 4, so only the wholesale absence check runs
    vector = "TTTT" * 10
    res = cl.simulate_ligation([vector, "GGGG" * 6], insert_sequence="ACGTA",
                               vector_sequence=vector)
    assert res["verified"] is False
    assert res["warnings"] == ["WARNING: Insert NOT detected in ligated product! "
                               "Ligation may have failed or insert was lost."]


@pytest.mark.parametrize("bad,role", [("123", "insert"), ("!", "vector")])
def test_unusable_verification_sequences_disable_verification_but_nothing_else(bad, role):
    vector, insert = "TTTT" * 10, "ACGT" * 6
    kwargs = {"insert_sequence": bad if role == "insert" else insert,
              "vector_sequence": bad if role == "vector" else vector}
    res = cl.simulate_ligation([vector], **kwargs)
    assert res["product"] == vector and res["verified"] is False
    assert res["warnings"] == [
        f"Insert verification skipped: Invalid characters in {role}: {bad}. "
        "Only IUPAC nucleotide codes are accepted."]


# ── primer design ────────────────────────────────────────────────────────────

def test_a_template_of_only_ambiguous_bases_has_no_melting_temperature():
    assert cl._melting_temp("NNNN") == 0.0
    assert cl._gc_content("NNNN") == 0.0


def test_a_target_shorter_than_the_primer_is_refused():
    with pytest.raises(ValueError) as exc:
        cl.design_primers("ACGT", primer_length=20)
    assert str(exc.value) == ("Target sequence (4 bp) is shorter than primer length "
                              "(20 bp).")


def test_primers_at_the_extremes_of_gc_are_flagged_separately():
    low = cl.design_primers("ATATATATATATATATATATAT")
    assert low["forward"] == "ATATATATATATATATATAT"
    assert low["fwd_gc"] == 0.0 and low["rev_gc"] == 0.0
    assert low["warnings"][1:] == [
        "Forward primer GC content (0.0%) outside ideal range (40-60%).",
        "Reverse primer GC content (0.0%) outside ideal range (40-60%).",
    ]
    high = cl.design_primers("GCGCGCGCGCGCGCGCGCGCGC")
    assert high["fwd_gc"] == 100.0 and high["rev_gc"] == 100.0
    assert "Forward primer GC content (100.0%) outside ideal range (40-60%)." in high["warnings"]


def test_a_large_melting_temperature_gap_between_the_primers_is_warned_about():
    pair = cl.design_primers("ATATATATATATATATATATAT" + "GCGCGCGCGCGCGCGCGCGCGC")
    assert pair["forward"] == "ATATATATATATATATATAT"
    assert pair["reverse"] == "GCGCGCGCGCGCGCGCGCGC"
    assert pair["fwd_tm"] == pytest.approx(31.3)
    assert pair["rev_tm"] == pytest.approx(72.3)
    assert pair["warnings"][-1] == (
        "Tm difference (41.0°C) exceeds recommended 5°C threshold.")


def test_a_matched_primer_pair_reports_no_tm_warning():
    pair = cl.design_primers("ATGC" * 12)
    assert pair["fwd_tm"] == pytest.approx(pair["rev_tm"])
    assert not [w for w in pair["warnings"] if "Tm difference" in w]


# ── PCR ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cycles", [0, 51])
def test_an_unusual_cycle_count_is_noted_on_an_otherwise_valid_run(cycles):
    t = _random_template(60, 3)
    res = cl.simulate_pcr(t, t[:12], cl._reverse_complement(t[-12:]), cycles=cycles)
    assert res["warnings"] == [f"Unusual cycle count: {cycles}. Typical range is 25-35."]
    assert res["cycles"] == cycles


def test_a_forward_primer_whose_three_prime_end_is_present_reports_that_position():
    t = _random_template(60, 5)
    anchor = t[30:36]                      # the only place the 3' 6 bases occur
    primer = "TTTTTT" + anchor             # 12 nt, absent as a whole
    assert t.count(anchor) == 1 and primer not in t
    with pytest.raises(ValueError) as exc:
        cl.simulate_pcr(t, primer, cl._reverse_complement(t[-12:]))
    assert str(exc.value) == (
        f"Forward primer '{primer}' not found in template. "
        "Closest partial match (3' 6 bases) at position 24. "
        "Check primer sequence and template.")


def test_a_reverse_primer_binding_site_that_is_absent_reports_the_rc_and_nearest_match():
    t = _random_template(60, 5)
    rev_rc_anchor = t[20:26]
    rev_rc = "TTTTTT" + rev_rc_anchor
    rev = cl._reverse_complement(rev_rc)
    assert cl._reverse_complement(rev) == rev_rc
    assert t.count(rev_rc_anchor) == 1 and rev_rc not in t
    with pytest.raises(ValueError) as exc:
        cl.simulate_pcr(t, t[:12], rev)
    assert str(exc.value) == (
        f"Reverse primer binding site (RC: '{rev_rc}') not found in template."
        " Closest partial match (3' 6 bases) at position 14. "
        "Check primer sequence and template.")


def test_a_primer_too_short_to_hint_is_reported_without_a_partial_match():
    t = _random_template(60, 9)
    assert "AAAAAA" not in t
    with pytest.raises(ValueError) as exc:
        cl.simulate_pcr(t, "ACGTAC", cl._reverse_complement(t[-12:]))
    assert str(exc.value) == ("Forward primer 'ACGTAC' not found in template. "
                              "Check primer sequence and template.")


@pytest.mark.parametrize("primer,rc_of_primer,expected", [
    # > 6 nt, but even the critical 3' 6 bases are absent -> no position given
    ("TTTTTTTTTTTT", None,
     "Forward primer 'TTTTTTTTTTTT' not found in template. "
     "Check primer sequence and template."),
    # the RC site is exactly 6 nt -> too short to anchor a hint
    (None, "AAAAAA",
     "Reverse primer binding site (RC: 'AAAAAA') not found in template. "
     "Check primer sequence and template."),
    # > 6 nt with an absent 3' anchor -> no position given
    (None, "TTTTTTTTTT",
     "Reverse primer binding site (RC: 'TTTTTTTTTT') not found in template. "
     "Check primer sequence and template."),
])
def test_a_missing_primer_without_a_usable_three_prime_anchor_gives_no_position(primer,
                                                                                rc_of_primer,
                                                                                expected):
    t = _random_template(60, 9)
    assert "TTTTTT" not in t and "AAAAAA" not in t
    if primer is not None:
        args = (t, primer, cl._reverse_complement(t[-12:]))
    else:
        assert rc_of_primer[-6:] not in t
        args = (t, t[:12], cl._reverse_complement(rc_of_primer))
    with pytest.raises(ValueError) as exc:
        cl.simulate_pcr(*args)
    assert str(exc.value) == expected


def test_primers_that_bind_in_the_wrong_order_are_refused_with_both_positions():
    t = _random_template(60, 7)
    fwd = t[24:36]                              # binds downstream...
    rev = cl._reverse_complement(t[:12])        # ...while this RC site is at 0
    assert t.count(fwd) == 1 and t.count(cl._reverse_complement(rev)) == 1
    with pytest.raises(ValueError) as exc:
        cl.simulate_pcr(t, fwd, rev)
    assert str(exc.value) == (
        "Primer orientation error: forward primer binds at position 24 "
        "but reverse primer RC binds at position 0. Forward primer must bind "
        "upstream of the reverse primer binding site.")


@pytest.mark.parametrize("length,seed,efficiency", [
    (499, 1, 0.95), (500, 2, 0.90), (1999, 3, 0.90), (2000, 4, 0.80),
    (4999, 5, 0.80), (5000, 6, 0.65), (10001, 7, 0.65),
])
def test_amplification_efficiency_drops_with_amplicon_length(length, seed, efficiency):
    t = _random_template(length, seed)
    res = cl.simulate_pcr(t, t[:12], cl._reverse_complement(t[-12:]))
    assert (res["fwd_pos"], res["rev_pos"], res["size"]) == (0, length - 12, length)
    assert res["efficiency"] == pytest.approx(efficiency)
    assert res["product"] == t


def test_an_amplicon_over_ten_thousand_bases_is_called_out_in_the_report_data():
    t = _random_template(10001, 7)
    res = cl.simulate_pcr(t, t[:12], cl._reverse_complement(t[-12:]))
    assert res["warnings"] == [
        "Product is 10,001 bp. Very long amplicons may require optimization "
        "(touchdown PCR, DMSO, etc.)."]


# ── insert verification ──────────────────────────────────────────────────────

TRUNCATED_INSERT = "ACGTACGTACGTTTGCAATCGATCGATCGATC"   # 32 bp, halves of 16


@pytest.mark.parametrize("which", ["5p", "3p"])
def test_a_half_present_insert_is_reported_as_truncated_with_its_length(which):
    half = len(TRUNCATED_INSERT) // 2
    piece = TRUNCATED_INSERT[:half] if which == "5p" else TRUNCATED_INSERT[half:]
    res = cl.verify_insert("TTTT" + piece + "GGGG", TRUNCATED_INSERT)
    assert res["insert_found"] is False
    assert (res["insert_start"], res["insert_end"]) == (None, None)
    label = "5' half of the insert (first 16 bp)" if which == "5p" \
        else "3' half of the insert (last 16 bp)"
    assert res["warnings"] == ["Insert NOT found in the product sequence.",
                               f"Only the {label} was found. Insert may be truncated."]


def test_a_product_missing_every_trace_of_a_long_insert_gets_only_the_absence_warning():
    res = cl.verify_insert("T" * 40, TRUNCATED_INSERT)
    assert res["warnings"] == ["Insert NOT found in the product sequence."]


def test_an_insert_present_in_both_orientations_is_reported_as_ambiguous():
    rc = cl._reverse_complement(TRUNCATED_INSERT)
    res = cl.verify_insert(TRUNCATED_INSERT + rc, TRUNCATED_INSERT)
    assert res["insert_found"] is True and res["forward_orientation"] is True
    assert (res["insert_start"], res["insert_end"]) == (0, 32)
    assert res["warnings"] == [
        "Insert found in BOTH orientations. Verify by restriction mapping.",
        "The insert occurs 2 times in the product (at 0, 32). "
        "insert_start/insert_end span the first copy and any copies fused to it "
        "head-to-tail.",
    ]


# ── virtual gel ──────────────────────────────────────────────────────────────

def test_a_digest_result_list_of_dicts_is_plotted_as_one_lane():
    fig = cl.plot_virtual_gel(
        [{"size": 900, "sequence": "A", "start": 0, "end": 1},
         {"size": "bad", "sequence": "B"}, {"sequence": "no size"}],
        ladder=[900], lane_labels=["L1"])
    ax = fig.axes[0]
    assert len(ax.collections) == 4                      # glow + band for the 900s only
    assert [t.get_text() for t in ax.texts] == ["900", "900 bp", "L1", "Ladder"]


def test_nested_lists_are_one_lane_each_and_colour_names_select_the_lane_palette():
    fig = cl.plot_virtual_gel([[1000, 200], [3000]], ladder=[1000],
                              sample_colors=["cyan", "#ff0000"])
    ax = fig.axes[0]
    assert len(ax.collections) == 8                       # 2 bands x (glow + band) per lane
    assert [t.get_text() for t in ax.texts] == [
        "1000", "1000 bp", "200 bp", "Sample 1", "3000 bp", "Sample 2", "Ladder"]
    glows = [c.get_facecolor()[0] for c in ax.collections if c.get_zorder() == 2]
    assert len(glows) == 4                     # ladder, 2 x lane 1, lane 2
    assert tuple(round(v, 2) for v in glows[1]) == (0.05, 0.8, 0.8, 0.18)   # cyan lane
    assert tuple(round(v, 2) for v in glows[2]) == (0.05, 0.8, 0.8, 0.18)
    # a hex triplet is resolved by matplotlib instead of being ignored
    assert tuple(round(v, 2) for v in glows[3]) == (1.0, 0.0, 0.0, 0.18)


def test_labels_shorter_than_the_lanes_fall_back_to_numbered_names():
    fig = cl.plot_virtual_gel([[100], [200], [300]], ladder=[100], lane_labels=["only one"])
    labels = [t.get_text() for t in fig.axes[0].texts if t.get_text().startswith(("Sample", "only"))]
    assert labels == ["only one", "Sample 2", "Sample 3"]


def test_a_gel_with_no_usable_size_still_draws_lanes_without_bands():
    fig = cl.plot_virtual_gel([0, -5, "x"], ladder=[0, None, "y"])
    ax = fig.axes[0]
    assert len(ax.collections) == 0
    assert [t.get_text() for t in ax.texts] == ["Sample 1", "Ladder"]


def test_a_single_distinct_size_places_every_band_at_the_middle_of_the_gel():
    fig = cl.plot_virtual_gel([1000], ladder=[1000])
    ys = {round(c.get_paths()[0].vertices[:, 1].mean(), 2) for c in fig.axes[0].collections}
    assert ys == {0.48, 0.49}                 # frac 0.5 of gel_top 0.88 / gel_bottom 0.08


def test_band_placement_is_monotonic_in_fragment_size_within_a_lane():
    fig = cl.plot_virtual_gel([5000, 500], ladder=[5000, 500])
    ax = fig.axes[0]
    bands = [(round(c.get_paths()[0].vertices[:, 1].mean(), 2),
              round(c.get_paths()[0].vertices[:, 0].mean(), 2))
             for c in ax.collections if c.get_zorder() == 3]
    assert len(bands) == 4                          # 2 ladder bands + 2 sample bands
    assert bands[0][0] < bands[1][0], bands         # 5000 bp sits lower than 500 bp
    assert bands[2][0] < bands[3][0], bands         # same ordering in the sample lane
    assert all(x_ladder < x_sample for (_, x_ladder), (_, x_sample)
               in zip(bands[:2], bands[2:]))        # lane 1 is right of the ladder lane


def test_an_existing_axes_is_used_and_returned_as_the_same_figure():
    fig, ax = plt.subplots(figsize=(3, 3))
    returned = cl.plot_virtual_gel([40], ax=ax, ladder=[40, 80])
    assert returned is fig and len(returned.axes) == 1
    assert [t.get_text() for t in ax.texts] == ["40", "80", "40 bp", "Sample 1", "Ladder"]


# ── reports ──────────────────────────────────────────────────────────────────

def test_the_digest_report_lists_warnings_and_plain_integer_cut_positions():
    text = cl.format_digest_report({
        "total_bp": 100, "enzymes_used": ["EcoRI"], "overhangs": {},
        "warnings": ["cut geometry derived from unchecked table entries for: EcoRI"],
        "cuts": [7, 20], "sizes": [40, 30]})
    assert "  Enzyme overhangs:" not in text
    assert "  ! cut geometry derived from unchecked table entries for: EcoRI\n" in text
    assert "  Cut sites (2):\n    Position 7\n    Position 20\n" in text
    assert "  ⚠ WARNING: Fragment total (70) ≠ input (100)" in text


def test_the_digest_report_says_no_cut_sites_were_found():
    text = cl.format_digest_report({"total_bp": 100, "enzymes_used": [],
                                    "overhangs": {"EcoRI": "5prime"}, "warnings": [],
                                    "cuts": [], "sizes": [100]})
    assert "  Enzyme overhangs:\n    EcoRI: 5prime\n" in text
    assert "  No cut sites found." in text
    assert "!" not in text
    assert "  Total        100" in text


def test_the_digest_report_of_a_real_digest_is_balanced():
    res = cl.simulate_digestion("AAAGAATTC" + "C" * 30, "EcoRI", topology="linear")
    text = cl.format_digest_report(res)
    assert "  Cut sites (1):\n    Position 4 (EcoRI, offset +1)" in text
    assert "⚠ WARNING" not in text
    assert "  ! " not in text


def test_the_pcr_report_repeats_warnings_when_there_are_any():
    text = cl.format_pcr_report({"size": 500, "cycles": 0, "efficiency": 0.9,
                                 "fwd_pos": 1, "rev_pos": 400,
                                 "warnings": ["Unusual cycle count: 0. Typical range is 25-35."]})
    assert "  Warnings:\n    ⚠ Unusual cycle count: 0. Typical range is 25-35." in text
    assert "✓ No warnings" not in text


def test_the_ligation_report_marks_a_verified_insert_and_lists_notes():
    text = cl.format_ligation_report({"fragment_count": 2, "size": 40, "is_circular": True,
                                      "efficiency": 0.72, "verified": True,
                                      "warnings": ["Insert found but may span a junction"]})
    assert "  ✓ Insert verified in product" in text
    assert "  Notes:\n    ⚠ Insert found but may span a junction" in text
    assert "Topology: circular" in text


def test_the_ligation_report_stays_silent_about_an_unverified_empty_product():
    with_notes = cl.format_ligation_report({"fragment_count": 0, "size": 0,
                                            "is_circular": False, "efficiency": 0.0,
                                            "verified": False,
                                            "warnings": ["All fragments were invalid."]})
    assert "Insert verification" not in with_notes
    assert "  Notes:\n    ⚠ All fragments were invalid." in with_notes
    without = cl.format_ligation_report({"fragment_count": 3, "size": 30, "verified": False,
                                         "is_circular": True, "efficiency": 0.5})
    assert "  — Insert verification: not requested or no insert specified" in without
