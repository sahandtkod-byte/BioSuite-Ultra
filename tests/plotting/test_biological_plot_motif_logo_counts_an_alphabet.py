"""A sequence logo counts an alphabet, and says what it could not count.

``draw_motif_logo`` only knows ``ACGT``.  Lowercase input — the normal shape of a
consensus string out of a FASTA file — matched nothing, so every bar came out at
height ``0.0``: a logo that looks like "no information at any position" when the
real problem was the case of the letters.  Ambiguity codes and amino acids were
dropped the same way, so a column stopped short of 1.0 with no word said, and an
empty list died in ``max()`` with numpy's own complaint.
"""
import contextlib
import io
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import biological_plots as bp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _heights(ax, per_base=4):
    """The four bars of each base, in the order the drawer stacks A, C, G, T."""
    return [float(rect.get_height()) for rect in ax.patches]


def test_uppercase_input_still_stacks_to_one_per_position():
    ax = bp.draw_motif_logo(['ACGT', 'ACGT', 'TGCA'])

    heights = _heights(ax)
    assert heights[0] == pytest.approx(2 / 3), "A at position 0 in two of three"
    assert heights[3] == pytest.approx(1 / 3), "A at position 3 in one of three"
    first_column = heights[0] + heights[4] + heights[8] + heights[12]
    assert first_column == pytest.approx(1.0)


def test_lowercase_sequences_are_counted_as_the_same_letters():
    lower = bp.draw_motif_logo(['acgt', 'acgt', 'tgca'])
    upper = bp.draw_motif_logo(['ACGT', 'ACGT', 'TGCA'])

    assert _heights(lower) == pytest.approx(_heights(upper))
    assert _heights(lower)[0] == pytest.approx(2 / 3), "lowercase used to draw zeros"


def test_letters_outside_acgt_are_reported_not_silently_dropped(capsys):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        ax = bp.draw_motif_logo(['ANCG', 'ACGT', 'XYZ-'])

    text = out.getvalue()
    assert "4 letter(s) outside ACGT were not counted" in text, text
    assert "missing information, not low information" in text, text
    column_one = (_heights(ax)[0] + _heights(ax)[4] + _heights(ax)[8] + _heights(ax)[12])
    assert column_one == pytest.approx(2 / 3), "the N left that column short of 1.0"


def test_a_gap_is_not_a_letter_and_is_not_in_the_tally(capsys):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        bp.draw_motif_logo(['AC', 'ACGT', 'ACGT'])

    assert "outside ACGT" not in out.getvalue(), out.getvalue()


def test_no_sequences_at_all_is_refused_with_an_answer():
    with pytest.raises(ValueError, match=r'at least one sequence') as info:
        bp.draw_motif_logo([])

    assert 'max()' in str(info.value)


def test_a_single_sequence_still_draws_a_logo():
    ax = bp.draw_motif_logo(['ACGT'])

    heights = _heights(ax)
    assert heights[0] == pytest.approx(1.0)
    assert heights[4] == pytest.approx(0.0)


def test_the_drawer_returns_the_axes_it_was_given():
    fig, ax = plt.subplots()

    returned = bp.draw_motif_logo(['AC', 'AC'], ax=ax)

    assert returned is ax
    assert [t.get_text() for t in ax.texts] == []
    assert ax.get_xlabel() == 'Position' and ax.get_ylabel() == 'Frequency'
