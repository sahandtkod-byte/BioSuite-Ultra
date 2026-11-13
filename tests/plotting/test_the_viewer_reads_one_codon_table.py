"""The viewer reads the package's codon table, and one codon table only.

``biosuite.core.utils.GENETIC_CODE`` and the copy in this module held the same 64
codons, checked by nobody.  That is the arrangement in which one table gets fixed
and the figures keep lying.  The reverse-complement table was the same story and
was *not* identical: it knew only ACGT (see
``test_reverse_strand_work_reads_the_same_sequence.py``).
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from Bio.Data import CodonTable

from biosuite.core.utils import GENETIC_CODE as CORE_CODE
from biosuite.plotting import sequence_viewer as sv


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_there_is_one_codon_table_in_the_package():
    assert sv.GENETIC_CODE is CORE_CODE, "two dicts is two chances to disagree"


@pytest.mark.parametrize("codon", sorted(CodonTable.unambiguous_dna_by_id[1].forward_table))
def test_every_sense_codon_means_what_the_standard_table_says(codon):
    assert sv.GENETIC_CODE[codon] == CodonTable.unambiguous_dna_by_id[1].forward_table[codon]


def test_the_three_stops_are_stops_and_nothing_else_is():
    reference = CodonTable.unambiguous_dna_by_id[1]

    assert sorted(c for c, aa in sv.GENETIC_CODE.items() if aa == "*") == \
        sorted(reference.stop_codons) == ["TAA", "TAG", "TGA"]
    assert len(sv.GENETIC_CODE) == 64


def test_a_codon_the_table_does_not_hold_is_drawn_as_unknown_not_as_polar():
    """``NNN`` used to take the green of serine, threonine and asparagine."""
    fig = sv.draw_translation_view("ATGNNACCC", frame=1)

    boxes = {t.get_text(): tuple(np.round(p.get_facecolor()[:3], 3))
             for t, p in zip(fig.axes[1].texts, fig.axes[1].patches)}
    plt.close(fig)

    assert boxes["X"] == pytest.approx(
        matplotlib.colors.to_rgb(sv.AA_COLORS["unknown"]), abs=0.01), boxes
    assert boxes["X"] != boxes["P"], "proline and an unknown codon must not match"
    assert boxes["X"] != pytest.approx(
        matplotlib.colors.to_rgb(sv.AA_COLORS["polar"]), abs=0.01), boxes
    assert sv.AA_PROPERTIES["X"] == "unknown"


def test_the_frame_tracks_colour_an_unknown_codon_the_same_way():
    fig = sv.draw_sequence_view("ATGNNTAC", end=8)

    frame1 = fig.axes[2]
    colours = {t.get_text(): matplotlib.colors.to_rgb(t.get_color())
               for t in frame1.texts}
    plt.close(fig)

    assert colours["X"] == matplotlib.colors.to_rgb(sv.AA_COLORS["unknown"])
    assert colours["M"] != colours["X"]
