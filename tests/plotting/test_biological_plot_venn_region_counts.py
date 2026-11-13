"""Region counts a Venn drawer can actually draw.

``draw_venn2`` / ``draw_venn3`` are public (they are in ``__all__``, and
``plot_api.venn`` calls them).  Called directly, a two-element list died on
``subsets[2]`` with a bare ``IndexError: list index out of range``, a negative
count was printed inside the circle, and a fractional one came out as ``10.0``
where the figure claims a number of members — the demo collected sizes with
``float``, so a terminal session typed ``10,15,4`` and got three decimals on the
picture.  The circles themselves are always the same radius, which is a
simplification the title now states instead of hiding.
"""
import contextlib
import io
import logging
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


def _numbers(ax):
    """Region labels only: '4.5' counts as a number, 'A' does not."""
    kept = []
    for text in ax.texts:
        try:
            float(text.get_text())
        except ValueError:
            continue
        kept.append(text.get_text())
    return kept


def test_two_set_counts_are_printed_exactly_as_given():
    ax = bp.draw_venn2([50, 40, 15], set_labels=('X', 'Y'))

    assert _numbers(ax) == ['50', '40', '15']
    assert len(ax.patches) == 2, "two circles"
    assert {'X', 'Y'} <= {t.get_text() for t in ax.texts}


def test_the_title_says_the_areas_are_not_proportional():
    """Positive control on the honest label: the geometry is fixed, so a reader
    must not infer |A| from the area of circle A."""
    for n_sets, sizes, labels in [(2, [50, 40, 15], ('X', 'Y')),
                                  (3, [10, 12, 8, 4, 3, 2, 1], ('A', 'B', 'C'))]:
        drawer = bp.draw_venn2 if n_sets == 2 else bp.draw_venn3
        ax = drawer(sizes, set_labels=labels)

        assert 'not proportional' in ax.get_title(), ax.get_title()
        assert {p.get_radius() for p in ax.patches} == {1.0}, "the radii never move"


def test_a_whole_number_is_printed_without_a_trailing_decimal():
    ax = bp.draw_venn2([10.0, 15.0, 4.0], set_labels=('A', 'B'))

    assert _numbers(ax) == ['10', '15', '4'], "10.0 is not a count of anything"


def test_a_fractional_region_is_kept_and_warned_about(caplog):
    with caplog.at_level(logging.WARNING, logger='biosuite.plotting.biological_plots'):
        ax = bp.draw_venn3([1, 2, 3, 4.5, 5, 6, 2], set_labels=('A', 'B', 'C'))

    warned = [r for r in caplog.records if 'fractional region counts' in r.getMessage()]
    assert len(warned) == 1, [r.getMessage() for r in caplog.records]
    assert warned[0].name == 'biosuite.plotting.biological_plots'
    assert '4.5' in _numbers(ax), _numbers(ax)


@pytest.mark.parametrize("sizes", [[10, 20], [1, 2, 3, 4], [], (5, 6, 7, 8, 9, 10)])
def test_a_region_list_of_the_wrong_length_names_the_shape_it_needed(sizes):
    with pytest.raises(ValueError, match=r"draw_venn2\(\) needs 3 region counts") as info:
        bp.draw_venn2(sizes)

    text = str(info.value)
    assert "[onlyA, onlyB, both]" in text, text
    assert str(len(sizes)) in text, text


def test_the_three_set_drawer_names_its_own_seven_regions():
    with pytest.raises(ValueError, match=r"draw_venn3\(\) needs 7 region counts") as info:
        bp.draw_venn3([1, 2, 3, 4, 5, 6])

    assert "[A, B, C, AB, AC, BC, ABC]" in str(info.value)


def test_labels_have_to_match_the_number_of_sets():
    with pytest.raises(ValueError, match=r"draw_venn2\(\) needs 2 set labels"):
        bp.draw_venn2([1, 2, 3], set_labels=('A', 'B', 'C'))


@pytest.mark.parametrize("value", [-5, -0.5])
def test_a_region_cannot_hold_a_negative_number_of_members(value):
    with pytest.raises(ValueError, match=r"region 0 is -") as info:
        bp.draw_venn2([value, 10, 2])

    assert "negative number of members" in str(info.value)


def test_a_region_that_is_not_a_number_is_refused():
    with pytest.raises(ValueError, match=r"region 1 must be a count, got 'many'"):
        bp.draw_venn2([10, 'many', 2])

    with pytest.raises(ValueError, match=r"region 0 must be a count"):
        bp.draw_venn3([True, 2, 3, 4, 5, 6, 1])


def test_a_triple_overlap_cannot_be_larger_than_a_pair():
    with pytest.raises(ValueError, match=r"ABC=7") as info:
        bp.draw_venn3([10, 12, 8, 4, 5, 6, 7], set_labels=('A', 'B', 'C'))

    text = str(info.value)
    assert "AB=4, AC=5, BC=6" in text, text
    assert "no pair can be smaller than the triple" in text, text


def test_a_legal_three_set_figure_prints_all_seven_counts():
    ax = bp.draw_venn3([10, 12, 8, 4, 5, 6, 2], set_labels=('A', 'B', 'C'))

    assert _numbers(ax) == ['10', '12', '8', '4', '5', '6', '2']
    assert len(ax.patches) == 3


def test_the_demo_collects_sizes_as_whole_numbers(tmp_path, monkeypatch):
    """Typed '10,15,4' must not reach the picture as 10.0/15.0/4.0."""
    import builtins
    import io

    captured = {}

    def spy(default_name, save_format, dpi, pdf=None, story=None):
        fig = plt.gcf()
        captured['numbers'] = [t.get_text() for t in fig.axes[0].texts
                               if t.get_text().replace('-', '').isdigit()]
        plt.close(fig)
        return None

    def ask(prompt=""):
        if "Number of sets" in prompt:
            return '2'
        if "sizes" in prompt:
            return '10,15,4'
        return ''

    monkeypatch.setattr(bp, 'ask_save_plot', spy)
    monkeypatch.setattr(builtins, 'input', ask)
    monkeypatch.setitem(bp.config, 'quiet', False)
    with contextlib.redirect_stdout(io.StringIO()):
        bp.venn_diagram()

    assert captured['numbers'] == ['10', '15', '4'], captured['numbers']
