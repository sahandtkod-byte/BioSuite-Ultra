"""Cross-implementation accuracy checks.

BioSuite's algorithms are compared with independent references — Biopython,
statsmodels, and a regex scan written here rather than reused from the module —
so that a shared bug cannot hide.  These are the audit's Phase 7 Accuracy Suite.
"""
import itertools
import random
import re

import pytest

from biosuite.core.alignment import needleman_wunsch, smith_waterman
from biosuite.core.cloning import find_restriction_sites
from biosuite.core.expression import _benjamini_hochberg
from biosuite.core.sequence import gc_content, reverse_complement, translate
from biosuite.core.utils import RESTRICTION_ENZYMES

Bio = pytest.importorskip("Bio")

IUPAC = "ACGTRYSWKMBDHVN"
IUPAC_REGEX = {
    "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]", "K": "[GT]", "M": "[AC]",
    "B": "[CGT]", "D": "[AGT]", "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}
COMPLEMENT = str.maketrans(IUPAC, "TGCAYRSWMKVHDBN")


def _sample_sequence(length, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


# ── reverse complement ───────────────────────────────────────────────────────

@pytest.mark.parametrize("length", [1, 2, 3])
def test_reverse_complement_matches_biopython_over_the_iupac_alphabet(length):
    for combo in itertools.product(IUPAC, repeat=length):
        seq = "".join(combo)
        assert reverse_complement(seq) == str(Bio.Seq.Seq(seq).reverse_complement()), seq


# ── translation ──────────────────────────────────────────────────────────────

def test_translate_matches_biopython_for_every_standard_codon():
    for codon in ("".join(c) for c in itertools.product("ACGT", repeat=3)):
        assert translate(codon) == str(Bio.Seq.Seq(codon).translate()), codon


def test_ambiguous_codons_are_conservative_by_documented_choice():
    """Biopython resolves ACN to T; BioSuite answers X for any codon it cannot
    read unambiguously.  The divergence is deliberate and recorded here so a
    future change cannot silently flip the convention either way."""
    assert translate("ACN") == "X"
    assert translate("NNN") == "X"


# ── GC content ───────────────────────────────────────────────────────────────

def test_gc_content_matches_the_reference_formula_including_ambiguous_bases():
    for seed in range(20):
        seq = _sample_sequence(60, seed) + "N" * (seed % 3)
        upper = seq.upper()
        expected = (upper.count("G") + upper.count("C")) / len(upper) * 100
        assert gc_content(seq) == pytest.approx(expected)


# ── multiple-testing correction ──────────────────────────────────────────────

def test_benjamini_hochberg_matches_statsmodels():
    multipletests = pytest.importorskip(
        "statsmodels.stats.multitest").multipletests
    for seed in (1, 2, 3):
        rng = random.Random(seed)
        pvals = [rng.random() for _ in range(150)]
        expected = multipletests(pvals, method="fdr_bh")[1]
        assert list(_benjamini_hochberg(pvals)) == pytest.approx(expected, abs=1e-12)


# ── pairwise alignment ───────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("scores", [(1, -1, -2), (2, -1, -5), (5, -4, -2)])
def test_alignment_scores_match_biopythons_pairwise_aligner(scores):
    match, mismatch, gap = scores
    for seed in range(5):
        first, second = _sample_sequence(60, seed), _sample_sequence(55, seed + 100)
        global_aligner = Bio.Align.PairwiseAligner(mode="global")
        local_aligner = Bio.Align.PairwiseAligner(mode="local")
        for aligner in (global_aligner, local_aligner):
            aligner.match_score, aligner.mismatch_score = match, mismatch
            aligner.open_gap_score = aligner.extend_gap_score = gap
        assert needleman_wunsch(first, second, match, mismatch, gap)[2] == \
            global_aligner.score(first, second)
        assert smith_waterman(first, second, match, mismatch, gap)[2] == \
            local_aligner.score(first, second)


# ── restriction enzymes ──────────────────────────────────────────────────────

def _scan_for_site(sequence, site):
    """Independent recognition-site scan: IUPAC-expanded, overlapping, both strands."""
    def pattern(code):
        return "".join(IUPAC_REGEX.get(char, char) for char in code)

    reverse = site.translate(COMPLEMENT)[::-1]
    found = [m.start() for m in re.finditer("(?=" + pattern(site) + ")", sequence)]
    if reverse != site:
        found += [m.start() for m in re.finditer("(?=" + pattern(reverse) + ")", sequence)]
    return sorted(found)


def test_recognition_positions_match_an_independent_scan():
    sequence = _sample_sequence(3000, seed=4242)
    for name in sorted(RESTRICTION_ENZYMES):
        site = RESTRICTION_ENZYMES[name][0].upper()
        assert find_restriction_sites(sequence, name) == _scan_for_site(sequence, site), name


def test_recognition_sequences_match_the_rebase_table_biopython_ships():
    """Fourteen of these sites were wrong before the Phase 7 comparison."""
    import Bio.Restriction as restriction

    shared = [(name, getattr(restriction, name)) for name in sorted(RESTRICTION_ENZYMES)
              if hasattr(restriction, name)]
    assert len(shared) > 100, "only %d enzymes could be compared" % len(shared)
    mismatched = {name: (RESTRICTION_ENZYMES[name][0].upper(), str(enzyme.site).upper())
                  for name, enzyme in shared
                  if RESTRICTION_ENZYMES[name][0].upper() != str(enzyme.site).upper()}
    assert not mismatched, mismatched
