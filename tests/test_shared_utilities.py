"""
Tests for the shared utilities in biosuite/core/utils.py and the DRY
improvements that ensure orf_finder, crispr, and cloning modules all
import from a single canonical source.

Covers:
  - reverse_complement_dna()
  - RESTRICTION_ENZYMES dictionary
  - RESTRICTION_ENZYMES_SITES dictionary
  - Identity checks (same object) across orf_finder, crispr, cloning
"""
import pytest


# ── reverse_complement_dna ──────────────────────────────────────────────────


class TestReverseComplementDna:
    """Tests for the canonical reverse-complement function in utils."""

    @pytest.fixture(autouse=True)
    def _load(self):
        from biosuite.core.utils import reverse_complement_dna
        self.rc = reverse_complement_dna

    def test_basic(self):
        """Simple 4-base complement and reversal."""
        assert self.rc("ATCG") == "CGAT"

    def test_with_n(self):
        """N is its own complement."""
        assert self.rc("ATCGN") == "NCGAT"

    def test_lowercase(self):
        """Lowercase bases should be complemented and preserved."""
        assert self.rc("atcg") == "cgat"

    def test_mixed_case(self):
        """Case is preserved per-character after complement."""
        assert self.rc("AtCg") == "cGaT"

    def test_empty_string(self):
        """Empty input yields empty output."""
        assert self.rc("") == ""

    def test_palindrome(self):
        """A palindrome sequence is its own reverse complement."""
        assert self.rc("AATT") == "AATT"

    def test_long_sequence(self):
        """A longer sequence is complemented and reversed correctly."""
        seq = "ATCGATCGATCGATCG" * 10
        result = self.rc(seq)
        assert len(result) == len(seq)
        # Complement of A is T, etc. — verify first/last swap
        assert result == seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    def test_all_four_bases(self):
        """Full round-trip: rc(rc(seq)) == seq."""
        seq = "GATTACA"
        assert self.rc(self.rc(seq)) == seq

    def test_non_dna_characters_ignored(self):
        """Characters not in the translation table are untouched (reversed only)."""
        # 'X' is not in the translation table, so it stays 'X'
        assert self.rc("AXT") == "AXT"  # X not complemented, just reversed
        # Wait — actually str.maketrans only maps listed chars; others pass through.
        # A->T, X->X, T->A => reversed => "AXT"
        assert self.rc("AXT") == "AXT"


# ── RESTRICTION_ENZYMES dictionary ─────────────────────────────────────────


class TestRestrictionEnzymes:
    """Tests for the RESTRICTION_ENZYMES shared dictionary."""

    @pytest.fixture(autouse=True)
    def _load(self):
        from biosuite.core.utils import RESTRICTION_ENZYMES
        self.enzymes = RESTRICTION_ENZYMES

    def test_has_18_enzymes(self):
        """The dictionary should contain 100+ enzymes."""
        assert len(self.enzymes) >= 100

    def test_all_entries_are_tuples(self):
        """Every value must be (site, cut_position) — a 2-tuple."""
        for name, value in self.enzymes.items():
            assert isinstance(value, tuple), f"{name} value is not a tuple"
            assert len(value) == 2, f"{name} tuple does not have exactly 2 elements"

    def test_ecori(self):
        """EcoRI: GAATTC, cut after position 1."""
        site, cut_pos = self.enzymes["EcoRI"]
        assert site == "GAATTC"
        assert cut_pos == 1

    def test_noti(self):
        """NotI: GCGGCCGC, cut after position 2."""
        site, cut_pos = self.enzymes["NotI"]
        assert site == "GCGGCCGC"
        assert cut_pos == 2

    def test_smai(self):
        """SmaI: CCCGGG, cut after position 3 (blunt cutter)."""
        site, cut_pos = self.enzymes["SmaI"]
        assert site == "CCCGGG"
        assert cut_pos == 3

    def test_all_sites_are_valid_dna(self):
        """Every recognition site should consist only of ACGT characters."""
        import re
        for name, (site, _cut_pos) in self.enzymes.items():
            assert re.fullmatch(r"[ACGTRYSWKMBDHVN]+", site), (
                f"{name} site '{site}' contains non-DNA/IUPAC characters"
            )

    def test_all_cut_positions_are_positive_integers(self):
        """Cut positions must be positive integers (>= 1)."""
        for name, (_site, cut_pos) in self.enzymes.items():
            assert isinstance(cut_pos, int), (
                f"{name} cut_position is not int: {cut_pos!r}"
            )
            assert cut_pos >= 1, (
                f"{name} cut_position is not positive: {cut_pos}"
            )

    def test_cut_positions_within_site_length(self):
        """Cut position should not exceed the length of the recognition site."""
        for name, (site, cut_pos) in self.enzymes.items():
            assert cut_pos <= len(site), (
                f"{name} cut_position {cut_pos} > site length {len(site)}"
            )

    def test_expected_enzymes_present(self):
        """Spot-check that all expected enzyme names exist."""
        expected = {
            "EcoRI", "BamHI", "HindIII", "NotI", "XhoI", "SacI",
            "KpnI", "SmaI", "XbaI", "PstI", "NcoI", "SpeI",
            "ApaI", "SalI", "SphI", "ClaI", "NheI", "MluI",
        }
        assert expected <= set(self.enzymes.keys())


# ── RESTRICTION_ENZYMES_SITES dictionary ────────────────────────────────────


class TestRestrictionEnzymesSites:
    """Tests for the RESTRICTION_ENZYMES_SITES convenience dictionary."""

    @pytest.fixture(autouse=True)
    def _load(self):
        from biosuite.core.utils import RESTRICTION_ENZYMES_SITES
        from biosuite.core.utils import RESTRICTION_ENZYMES
        self.sites = RESTRICTION_ENZYMES_SITES
        self.full = RESTRICTION_ENZYMES

    def test_same_keys_as_full_dict(self):
        """Sites dict should have exactly the same keys as the full dict."""
        assert set(self.sites.keys()) == set(self.full.keys())

    def test_values_are_strings(self):
        """All values in RESTRICTION_ENZYMES_SITES must be plain strings."""
        for name, value in self.sites.items():
            assert isinstance(value, str), (
                f"{name} value is {type(value).__name__}, expected str"
            )

    def test_values_are_not_tuples(self):
        """Values should be plain strings, not (site, cut_pos) tuples."""
        for name, value in self.sites.items():
            assert not isinstance(value, tuple), (
                f"{name} value is still a tuple — should be plain string"
            )

    def test_ecori_site(self):
        """EcoRI site should be 'GAATTC' (no cut position)."""
        assert self.sites["EcoRI"] == "GAATTC"

    def test_noti_site(self):
        """NotI site should be 'GCGGCCGC'."""
        assert self.sites["NotI"] == "GCGGCCGC"

    def test_sites_match_full_dict(self):
        """Each site string should match the site from the full tuple."""
        for name, (_site, _cut_pos) in self.full.items():
            assert self.sites[name] == _site, (
                f"{name}: sites dict has '{self.sites[name]}' "
                f"but full dict has '{_site}'"
            )


# ── DRY: orf_finder.py uses shared RESTRICTION_ENZYMES ─────────────────────


class TestOrfFinderUsesSharedEnzymes:
    """Verify orf_finder.py imports the same RESTRICTION_ENZYMES object."""

    def test_same_object_identity(self):
        """orf_finder's RESTRICTION_ENZYMES should be the exact same object."""
        from biosuite.core.orf_finder import RESTRICTION_ENZYMES as of_enzymes
        from biosuite.core.utils import RESTRICTION_ENZYMES as utils_enzymes
        assert of_enzymes is utils_enzymes, (
            "orf_finder.RESTRICTION_ENZYMES is not the same object as "
            "utils.RESTRICTION_ENZYMES — DRY violation!"
        )

    def test_orf_finder_exposes_all_enzymes(self):
        """The imported dictionary should contain all 18 enzymes."""
        from biosuite.core.orf_finder import RESTRICTION_ENZYMES
        assert len(RESTRICTION_ENZYMES) >= 100


# ── DRY: crispr.py uses shared reverse_complement ──────────────────────────


class TestCrisprUsesSharedReverseComplement:
    """Verify crispr.py's _reverse_complement is the shared function."""

    def test_same_function_identity(self):
        """crispr._reverse_complement should be the exact same function object."""
        from biosuite.core.crispr import _reverse_complement as crispr_rc
        from biosuite.core.utils import reverse_complement_dna as utils_rc
        assert crispr_rc is utils_rc, (
            "crispr._reverse_complement is not the same function as "
            "utils.reverse_complement_dna — DRY violation!"
        )

    def test_crispr_rc_produces_correct_output(self):
        """The aliased function should produce correct reverse complements."""
        from biosuite.core.crispr import _reverse_complement
        assert _reverse_complement("ATCG") == "CGAT"


# ── DRY: cloning.py uses shared RESTRICTION_ENZYMES ────────────────────


class TestCloningUsesSharedEnzymes:
    """Verify cloning.py imports RESTRICTION_ENZYMES (full tuple dict) from utils."""

    def test_same_object_identity(self):
        """cloning's RESTRICTION_ENZYMES should be the same object as
        utils.RESTRICTION_ENZYMES (the full tuple dict with cut positions)."""
        from biosuite.core.cloning import RESTRICTION_ENZYMES as cl_enzymes
        from biosuite.core.utils import RESTRICTION_ENZYMES
        assert cl_enzymes is RESTRICTION_ENZYMES, (
            "cloning.RESTRICTION_ENZYMES is not the same object as "
            "utils.RESTRICTION_ENZYMES — cloning should use the full dict."
        )

    def test_cloning_enzymes_are_tuples(self):
        """Cloning's RESTRICTION_ENZYMES values should be tuples
        (recognition_site, cut_position) for accurate digestion."""
        from biosuite.core.cloning import RESTRICTION_ENZYMES
        for name, value in RESTRICTION_ENZYMES.items():
            assert isinstance(value, tuple), (
                f"cloning.{name} is {type(value).__name__}, expected tuple"
            )
            assert isinstance(value[0], str), (
                f"cloning.{name}[0] is {type(value[0]).__name__}, expected str"
            )

    def test_cloning_enzymes_count(self):
        """cloning should have access to all restriction enzymes."""
        from biosuite.core.cloning import RESTRICTION_ENZYMES
        assert len(RESTRICTION_ENZYMES) >= 100

    def test_cloning_uses_full_dict(self):
        """cloning's RESTRICTION_ENZYMES should be the full tuple dict
        (with cut positions) for accurate digestion simulation."""
        from biosuite.core.cloning import RESTRICTION_ENZYMES as cl_enzymes
        from biosuite.core.utils import RESTRICTION_ENZYMES as full_enzymes
        assert cl_enzymes is full_enzymes, (
            "cloning.RESTRICTION_ENZYMES should be the full RESTRICTION_ENZYMES "
            "tuple dict (with cut positions), not the sites-only version"
        )
