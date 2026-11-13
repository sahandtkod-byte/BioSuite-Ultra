"""Audit-remediation regression tests — Phase 1 (biosuite/core/cloning.py + enzyme table).

One test per remediation step. Written to be independent of the pre-existing
suite so a reviewer can run this file alone and see exactly which fix is green.
"""
import itertools

import pytest

from biosuite.core.cloning import (
    _find_all,
    _reverse_complement,
    design_primers,
    find_restriction_sites,
    simulate_digestion,
    verify_insert,
)

IUPAC_DNA = "ATCGWSMKRYBDHVN"


class TestStep11ReverseComplement:
    """N5: _COMPLEMENT mapped every ambiguous code to itself."""

    def test_ambiguous_letters_complement_to_their_partner(self):
        assert _reverse_complement("AMCGT") == "ACGKT"
        assert _reverse_complement("RM") == "KY"
        assert _reverse_complement("BDHVN") == "NBDHV"

    def test_single_letter_pairs_against_iupac_definition(self):
        expected = {"A": "T", "T": "A", "C": "G", "G": "C",
                    "W": "W", "S": "S", "M": "K", "K": "M",
                    "R": "Y", "Y": "R", "B": "V", "V": "B",
                    "D": "H", "H": "D", "N": "N"}
        assert {c: _reverse_complement(c) for c in expected} == expected

    @pytest.mark.parametrize("a,b", list(itertools.product(IUPAC_DNA, repeat=2)))
    def test_matches_biopython_on_full_alphabet(self, a, b):
        from Bio.Seq import Seq
        seq = a + b
        assert _reverse_complement(seq) == str(Seq(seq).reverse_complement())

    def test_is_an_involution(self):
        for tri in itertools.product(IUPAC_DNA, repeat=3):
            seq = "".join(tri)
            assert _reverse_complement(_reverse_complement(seq)) == seq


class TestStep12IupacSiteSearch:
    """FIX-1: literal str.find meant no degenerate recognition site ever matched."""

    def test_ddeI_known_answer(self):
        # DdeI = CTNAG; CTGAG is one of its instantiations.
        assert find_restriction_sites("AAACTGAGCCC", enzyme="DdeI") == [3]

    def test_ecori_known_answer(self):
        assert find_restriction_sites("GGGAATTCCT", enzyme="EcoRI") == [2]
        assert find_restriction_sites("GAATTC", enzyme="EcoRI") == [0]

    def test_hinfI_known_answer(self):
        # HinfI = GANTC, and the sample base under N is an A here.
        assert find_restriction_sites("CGAATCG", enzyme="HinfI") == [1]

    def test_apoI_matches_ecori_site(self):
        # ApoI = RAATTY, and G/CC/G satisfy R/Y.
        assert find_restriction_sites("GAATTC", enzyme="ApoI") == [0]

    def test_no_false_positive(self):
        assert find_restriction_sites("AAAAAACCCCCC", enzyme="DdeI") == []

    def test_overlapping_sites_are_all_reported(self):
        # Regression against the audit report's own suggested fix: a bare
        # finditer over the expanded body silently drops overlaps.
        assert _find_all("ATATA", "ATA") == [0, 2]
        assert _find_all("AAAA", "AA") == [0, 1, 2]

    def test_lowercase_recognition_site_still_matches(self):
        assert _find_all("AAACTGAGCCC", "ctnag") == [3]

    def test_every_degenerate_enzyme_can_fire(self):
        from biosuite.core.utils import RESTRICTION_ENZYMES
        import random

        random.seed(7)
        seq = "".join(random.choice("ACGT") for _ in range(200_000))
        never = []
        for name, (site, _cut) in RESTRICTION_ENZYMES.items():
            if not set(site) & set("RYSWKMBDHVN"):
                continue
            if not _find_all(seq, site):
                never.append(name)
        # A 6-bp site with one N is expected roughly every 4^5 = 1024 bases, so
        # finding nothing in 200 kb means the expansion is still broken for it.
        assert not never, f"still unmatchable: {never}"


# Cut positions and overhang geometry checked against REBASE.  Every other entry
# in the table is an unchecked placeholder and must report itself as 'unverified'.
REBASE_VERIFIED = {
    # enzyme: (site, cut_position, overhang)
    "EcoRI":   ("GAATTC", 1, "5prime"),
    "BamHI":   ("GGATCC", 1, "5prime"),
    "HindIII": ("AAGCTT", 1, "5prime"),
    "SmaI":    ("CCCGGG", 3, "blunt"),
    "EcoRV":   ("GATATC", 3, "blunt"),
    "PstI":    ("CTGCAG", 5, "3prime"),
    "KpnI":    ("GGTACC", 5, "3prime"),
    "SacI":    ("GAGCTC", 5, "3prime"),
    "SphI":    ("GCATGC", 5, "3prime"),
}


class TestStep13CutGeometry:
    """N1 + N2: the cut-position column was a stub and every overhang said 5prime."""

    @pytest.mark.parametrize("enzyme", sorted(REBASE_VERIFIED))
    def test_table_matches_rebase(self, enzyme):
        from biosuite.core.utils import RESTRICTION_ENZYMES
        site, cut = RESTRICTION_ENZYMES[enzyme]
        want_site, want_cut, _want_type = REBASE_VERIFIED[enzyme]
        assert site == want_site
        assert cut == want_cut, f"{enzyme} cuts after base {want_cut}, table has {cut}"

    @pytest.mark.parametrize("enzyme", sorted(REBASE_VERIFIED))
    def test_overhang_type_is_verified(self, enzyme):
        from biosuite.core.utils import enzyme_cut_type
        assert enzyme_cut_type(enzyme) == REBASE_VERIFIED[enzyme][2]

    def test_blunt_enzymes_are_no_longer_claimed_as_5prime(self):
        # The old classifier reported '5prime' for all 169 enzymes.
        for enz in ("SmaI", "EcoRV"):
            site = REBASE_VERIFIED[enz][0]
            seq = "AAAA" + site + "AAAA"
            result = simulate_digestion(seq, enz, topology="linear")
            assert result["overhangs"][enz] == "blunt"

    def test_overhang_matches_the_derived_geometry(self):
        # overhang length = len(site) - 2*cut_position, so the label and the
        # number have to agree for every enzyme whose geometry we trust.
        for enz, (site, cut, kind) in REBASE_VERIFIED.items():
            overhang = len(site) - 2 * cut
            derived = "blunt" if overhang == 0 else ("5prime" if overhang > 0 else "3prime")
            assert derived == kind, f"{enz}: geometry and label disagree"

    def test_cut_position_is_site_start_plus_offset(self):
        # AAAA CTGCAG AAAA: the PstI site starts at index 4 and cuts after base 5,
        # i.e. at position 9, giving fragments of 9 bp and 4 bp on linear DNA.
        result = simulate_digestion("AAAACTGCAGAAA", "PstI", topology="linear")
        assert [c["site_start"] for c in result["cuts"]] == [4]
        assert [c["cut_pos"] for c in result["cuts"]] == [9]
        assert result["sizes"] == [9, 4]

    def test_unverified_enzyme_says_so(self):
        from biosuite.core.utils import enzyme_cut_type
        assert enzyme_cut_type("ZraI") == "unverified"
        result = simulate_digestion("AAAA" + "GACGTC" + "AAAA", "ZraI", topology="linear")
        assert result["warnings"], "an unchecked entry must surface a caveat"
        assert "ZraI" in result["warnings"][0]

    def test_verified_enzyme_produces_no_warning(self):
        result = simulate_digestion("AAAA" + "GAATTC" + "AAAA", "EcoRI", topology="linear")
        assert result["warnings"] == []


class TestStep14BothStrands:
    """N3: only the top strand was searched, so non-palindromic enzymes never fired."""

    def test_bsaI_lower_strand_known_answer(self):
        # BsaI = GGTCTC, which is not palindromic; GAGACC is its reverse complement
        # and sits at index 3 of this sequence.
        assert find_restriction_sites("AAAGAGACCTTT", enzyme="BsaI") == [3]

    def test_bsaI_upper_strand_still_found(self):
        assert find_restriction_sites("AAAGGTCTCTTT", enzyme="BsaI") == [3]

    def test_both_occurrences_reported_once_each(self):
        seq = "AAAGGTCTCTTT" + "C" * 10 + "GAGACC" + "TTT"   # 12 + 10 + 6 + 3
        assert find_restriction_sites(seq, enzyme="BsaI") == [3, 22]

    def test_palindromic_enzyme_is_not_double_counted(self):
        # EcoRI is self-complementary, so adding the rc search must change nothing.
        seq = "GGGAATTCCT" * 4
        assert find_restriction_sites(seq, enzyme="EcoRI") == [2, 12, 22, 32]

    def test_degenerate_palindrome_recognised_as_palindromic(self):
        # DdeI = CTNAG is its own reverse complement under the corrected table.
        from biosuite.core.cloning import _site_and_rc
        assert _site_and_rc("CTNAG") == ("CTNAG", "CTNAG")

    def test_strand_is_attributed(self):
        result = simulate_digestion("AAAGAGACCTTT", "BsaI", topology="linear")
        assert [c["strand"] for c in result["cuts"]] == ["-"]
        result = simulate_digestion("AAAGGTCTCTTT", "BsaI", topology="linear")
        assert [c["strand"] for c in result["cuts"]] == ["+"]

    def test_non_palindromic_enzyme_warns_about_geometry(self):
        result = simulate_digestion("AAAGAGACCTTT", "BsaI", topology="linear")
        assert any("BsaI" in w and "not modelled" in w for w in result["warnings"]), result["warnings"]

    def test_palindromic_enzyme_gets_no_asymmetry_warning(self):
        result = simulate_digestion("GGGAATTCCT", "EcoRI", topology="linear")
        assert not any("not modelled" in w for w in result["warnings"])

    def test_every_non_palindromic_enzyme_can_now_fire(self):
        from biosuite.core.cloning import _site_and_rc
        from biosuite.core.utils import RESTRICTION_ENZYMES
        import random

        random.seed(11)
        seq = "".join(random.choice("ACGT") for _ in range(300_000))
        asymmetric = [n for n, (s, _c) in RESTRICTION_ENZYMES.items()
                      if _site_and_rc(s)[0] != _site_and_rc(s)[1]]
        missed = [n for n, (s, _c) in RESTRICTION_ENZYMES.items()
                  if n in asymmetric and not find_restriction_sites(seq, enzyme=n)]
        assert len(asymmetric) >= 20, f"expected a sizeable non-palindromic set, got {len(asymmetric)}"
        assert not missed, f"still invisible on both strands: {missed}"


class TestStep15CircularOriginSites:
    """N4: a site straddling the origin of a plasmid was invisible."""

    ORIGIN_SPAN = "TC" + "A" * 10 + "GAAT"      # ends GAAT + starts TC = one EcoRI site

    def test_site_spanning_the_origin_is_cut_once(self):
        assert "GAATTC" not in self.ORIGIN_SPAN          # no internal site at all
        result = simulate_digestion(self.ORIGIN_SPAN, "EcoRI", topology="circular")
        assert len(result["cuts"]) == 1, result["cuts"]
        assert result["cuts"][0]["site_start"] == 12
        assert result["cuts"][0]["cut_pos"] == 13        # 12 + offset 1

    def test_one_cut_on_a_circle_gives_one_full_length_fragment(self):
        # A single cut linearises a plasmid: one fragment of the whole length.
        result = simulate_digestion(self.ORIGIN_SPAN, "EcoRI", topology="circular")
        assert result["sizes"] == [16]
        assert result["fragments"][0]["size"] == len(self.ORIGIN_SPAN)

    def test_fragment_sequence_is_the_rotation_from_the_cut(self):
        result = simulate_digestion(self.ORIGIN_SPAN, "EcoRI", topology="circular")
        expected = self.ORIGIN_SPAN[13:] + self.ORIGIN_SPAN[:13]
        assert result["fragments"][0]["sequence"] == expected
    def test_linear_molecule_with_the_same_sequence_is_not_cut(self):
        result = simulate_digestion(self.ORIGIN_SPAN, "EcoRI", topology="linear")
        assert result["cuts"] == []
        assert result["sizes"] == [16]

    def test_two_sites_on_a_circle_give_two_fragments(self):
        seq = "GAATTC" + "A" * 10 + "GAATTC" + "T" * 10   # 32 bp, two sites
        result = simulate_digestion(seq, "EcoRI", topology="circular")
        assert len(result["cuts"]) == 2
        assert result["sizes"] == [16, 16]
        assert sum(result["sizes"]) == 32

    @pytest.mark.parametrize("enzyme", ["EcoRI", "BamHI", "SmaI", "PstI", "HindIII"])
    def test_fragment_sizes_always_sum_to_the_molecule(self, enzyme):
        import random

        random.seed(3)
        site = {"EcoRI": "GAATTC", "BamHI": "GGATCC", "SmaI": "CCCGGG",
                "PstI": "CTGCAG", "HindIII": "AAGCTT"}[enzyme]
        for length in (40, 97, 250):
            seq = "".join(random.choice("ACGT") for _ in range(length))
            if site in seq or site in _reverse_complement(seq):
                continue                      # uninteresting, no origin case tested here
            circular = simulate_digestion(seq, enzyme, topology="circular")
            assert sum(circular["sizes"]) == len(seq) or not circular["cuts"], \
                f"{enzyme} at {length} bp lost bases: {circular['sizes']}"


class TestStep11bDownstreamConsumers:
    """design_primers and verify_insert consume the complement table; pin them."""

    AMBIGUOUS_END = "ATGCATGCATRYSWMKBDHVNACGT"      # 25 bp, last 20 hold ambiguity

    def test_reverse_primer_matches_biopython_on_ambiguous_template(self):
        from Bio.Seq import Seq

        designed = design_primers(self.AMBIGUOUS_END, primer_length=20)
        want = str(Seq(self.AMBIGUOUS_END.upper()[-20:]).reverse_complement())
        assert designed["reverse"] == want

    def test_reverse_primer_is_not_its_own_template_reversal(self):
        # The broken table made rc of an ambiguous stretch a near-palindrome, which
        # silently produced a primer that was the template read backwards.
        designed = design_primers(self.AMBIGUOUS_END, primer_length=20)
        assert designed["reverse"] != self.AMBIGUOUS_END.upper()[-20:][::-1]

    def test_reverse_oriented_insert_with_ambiguous_bases_is_detected(self):
        from Bio.Seq import Seq

        insert = "ACGTAMCGTGGATCCTTGCAGRNTAA"        # 25 bp, carries M, R and N
        # The product is built from Biopython's reverse complement, so this only
        # passes if cloning's own _reverse_complement agrees with it.
        product = "TTTT" + str(Seq(insert).reverse_complement()) + "CCCC"
        result = verify_insert(product, insert)
        assert result["insert_found"] is True, result["warnings"]
        assert result["forward_orientation"] is False

    def test_forward_oriented_insert_still_detected(self):
        insert = "ACGTAMCGTGGATCCTTGCAGRNTAA"
        result = verify_insert("TTTT" + insert + "CCCC", insert)
        assert result["insert_found"] is True
        assert result["forward_orientation"] is True





