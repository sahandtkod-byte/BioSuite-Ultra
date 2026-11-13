"""Every entry point in the viewer checks its sequence the same way.

The five drawing functions each did ``sequence.upper().replace("U", "T")`` and
drew the result.  A pasted FASTA header, a protein string, or an empty record
from a failed file read therefore produced a figure — with eight singular
x-limit warnings in the sequence view, a mean of an empty slice in the GC plot,
and a division by the sequence length in the ORF map.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import sequence_viewer as sv

ENTRY_POINTS = [
    sv.draw_sequence_view, sv.draw_translation_view,
    sv.draw_gc_content_plot, sv.draw_orf_map, sv.create_sequence_overview,
]
IDS = [fn.__name__ for fn in ENTRY_POINTS]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("draw", ENTRY_POINTS, ids=IDS)
def test_nothing_to_draw_is_refused_by_name(draw):
    before = set(plt.get_fignums())

    with pytest.raises(ValueError, match="must not be empty"):
        draw("")

    assert set(plt.get_fignums()) == before, f"{draw.__name__} drew the empty case"


@pytest.mark.parametrize("draw", ENTRY_POINTS, ids=IDS)
def test_a_string_that_is_not_a_sequence_is_named_rather_than_coloured(draw):
    before = set(plt.get_fignums())

    with pytest.raises(ValueError, match="not IUPAC nucleotides"):
        draw("MSEQVK this is not DNA")

    assert set(plt.get_fignums()) == before, f"{draw.__name__} drew the junk case"


@pytest.mark.parametrize("draw", ENTRY_POINTS, ids=IDS)
def test_a_non_string_is_refused_before_any_attribute_look(draw):
    with pytest.raises(TypeError, match="must be a string of nucleotides"):
        draw(None)


def test_whitespace_from_a_pasted_record_is_part_of_the_sequence_not_a_letter():
    """FASTA lines arrive with newlines; they are not a letter the alphabet lacks."""
    fig = sv.draw_sequence_view("ATGAAAC\nCCGGG TT\nTA")

    assert [t.get_text() for t in fig.axes[1].texts] == list("ATGAAACCCGGGTTTA")
    plt.close(fig)


def test_rna_reaches_the_same_view_as_the_same_chain_in_dna():
    fig_rna = sv.draw_sequence_view("AUGAAACCCGGGUUU")
    fig_dna = sv.draw_sequence_view("ATGAAACCCGGGTTT")

    assert [t.get_text() for t in fig_rna.axes[1].texts] == \
        [t.get_text() for t in fig_dna.axes[1].texts], "U is read as T, not as nothing"
    plt.close(fig_rna); plt.close(fig_dna)
