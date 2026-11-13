"""The ipywidgets classes: the interface they build and what each button does.

``SequenceAnalyzer``, ``AlignmentViewer`` and ``PlotExplorer`` build their widgets
inside ``show()``, so the only way to reach a button's handler is to display the
interface, find the widgets in the tree it hands to ``display()`` and click them the
way ipywidgets itself does.  ``Button.click()`` is ipywidgets' own programmatic
trigger, so the callback runs through the widget's real dispatcher rather than a
recorded stand-in.

Everything here needs ipywidgets; ``biosuite.notebook`` degrades to magics-only when
it is absent, which is the point of the module's ``HAS_WIDGETS`` guard.
"""
import matplotlib.pyplot as plt
import pytest

import biosuite.notebook as nbmod

ipywidgets = pytest.importorskip("ipywidgets")

#: Button label -> the fragment its handler must print for ``ATCGATCG``.
ANALYZER_BUTTONS = {
    "GC Content": "GC content: 50.00%",
    "Reverse Comp": "Reverse complement:\nCGATCGAT",
    "Translate": "Protein:\nID",
    "Full Stats": "length: 8",
}


@pytest.fixture
def displayed(monkeypatch):
    """Capture what a widget's ``show()`` hands to IPython's ``display()``."""
    shown = []
    monkeypatch.setattr(nbmod, "display",
                        lambda *objects, **kwargs: shown.extend(objects))
    return shown


def _find(widget, kind, **attributes):
    """Depth-first search for a widget of ``kind`` with the given attribute values."""
    if isinstance(widget, kind) and all(getattr(widget, key) == value
                                        for key, value in attributes.items()):
        return widget
    for child in getattr(widget, "children", ()) or ():
        found = _find(child, kind, **attributes)
        if found is not None:
            return found
    return None


def _click(displayed, description):
    button = _find(displayed[0], ipywidgets.Button, description=description)
    assert button is not None, f"no button labelled {description!r} was displayed"
    button.click()


# ── SequenceAnalyzer ─────────────────────────────────────────────────────────

def test_analyzer_displays_a_sequence_box_and_four_buttons(displayed):
    nbmod.SequenceAnalyzer().show()

    tree = displayed[0]
    assert isinstance(tree, ipywidgets.VBox)
    textarea = _find(tree, ipywidgets.Textarea)
    assert textarea is not None
    assert textarea.description == 'Sequence:'

    labels = []

    def collect(widget):
        if isinstance(widget, ipywidgets.Button):
            labels.append(widget.description)
        for child in getattr(widget, "children", ()) or ():
            collect(child)

    collect(tree)
    assert set(labels) == set(ANALYZER_BUTTONS)


@pytest.mark.parametrize("description,expected", sorted(ANALYZER_BUTTONS.items()))
def test_each_analyzer_button_reports_its_own_result(displayed, description, expected,
                                                     capsys):
    nbmod.SequenceAnalyzer().show()

    _find(displayed[0], ipywidgets.Textarea).value = "atcgatcg"
    _click(displayed, description)

    assert expected in capsys.readouterr().out


@pytest.mark.parametrize("description", sorted(ANALYZER_BUTTONS))
def test_analyzer_buttons_ask_for_a_sequence_instead_of_failing(displayed, description,
                                                                 capsys):
    nbmod.SequenceAnalyzer().show()

    _find(displayed[0], ipywidgets.Textarea).value = "   "
    _click(displayed, description)

    assert "Please enter a sequence" in capsys.readouterr().out


def test_analyzer_methods_compute_without_any_widgets():
    """The methods are the API a notebook user calls directly, no kernel needed."""
    analyzer = nbmod.SequenceAnalyzer()

    assert analyzer.gc_content("ATCGATCG") == pytest.approx(50.0)
    assert analyzer.reverse_complement("ATCG") == "CGAT"
    assert analyzer.translate("ATGAAATTTTAA") == "MKF*"


def test_analyzer_stats_method_returns_the_core_dictionary():
    result = nbmod.SequenceAnalyzer().sequence_stats("ATCGATCG")

    assert result["length"] == 8
    assert result["GC"] == 50.0


# ── AlignmentViewer ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("description", ["Needleman-Wunsch", "Smith-Waterman"])
def test_alignment_buttons_print_a_score_and_both_rows(displayed, description, capsys):
    nbmod.AlignmentViewer().show()

    box = displayed[0]
    _find(box, ipywidgets.Textarea, description="Seq 1:").value = "ATCGATCG"
    _find(box, ipywidgets.Textarea, description="Seq 2:").value = "ATCAATCG"

    _click(displayed, description)

    out = capsys.readouterr().out
    assert "Score:" in out
    assert out.count("Seq1: ") == 1 and out.count("Seq2: ") == 1


@pytest.mark.parametrize("description", ["Needleman-Wunsch", "Smith-Waterman"])
def test_alignment_buttons_ask_for_both_sequences(displayed, description, capsys):
    nbmod.AlignmentViewer().show()

    _find(displayed[0], ipywidgets.Textarea, description="Seq 1:").value = "ATCG"
    _click(displayed, description)

    assert "Please enter both sequences" in capsys.readouterr().out


# ── PlotExplorer ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("plot_type", ["Volcano", "PCA", "Scatter", "Heatmap",
                                       "Boxplot"])
def test_plot_explorer_builds_one_matplotlib_figure_per_plot_type(displayed, plot_type):
    nbmod.PlotExplorer().show()

    before = set(plt.get_fignums())
    _find(displayed[0], ipywidgets.Dropdown).value = plot_type
    _click(displayed, "Generate Plot")

    assert len(displayed) == 2, "the explorer displayed the figure itself"
    assert isinstance(displayed[1], plt.Figure)
    assert set(plt.get_fignums()) - before, "the figure was never opened"
    plt.close("all")


def test_plot_explorer_switches_to_plotly_when_interactive_is_checked(displayed):
    plotly_figure = pytest.importorskip("plotly.graph_objs").Figure

    nbmod.PlotExplorer().show()

    _find(displayed[0], ipywidgets.Checkbox).value = True
    _click(displayed, "Generate Plot")

    assert isinstance(displayed[1], plotly_figure)
    assert len(displayed[1].data) >= 1  # the selected plot really drew a trace


def test_plot_explorer_offers_every_plot_it_can_draw(displayed):
    nbmod.PlotExplorer().show()

    options = _find(displayed[0], ipywidgets.Dropdown).options

    assert set(options) == {"Volcano", "PCA", "Scatter", "Heatmap", "Boxplot"}


def test_plot_explorer_says_when_it_cannot_draw_the_selected_type(displayed, capsys):
    """An option with no branch behind it drew nothing and said nothing; adding a
    fifth plot type to the list without the code is exactly that mistake."""
    nbmod.PlotExplorer().show()

    dropdown = _find(displayed[0], ipywidgets.Dropdown)
    dropdown.options = (*dropdown.options, "Circos")
    dropdown.value = "Circos"
    _click(displayed, "Generate Plot")

    assert "Cannot draw plot type 'Circos'" in capsys.readouterr().out
    assert len(displayed) == 1, "an undrawable type must not display a figure"


def test_plot_explorer_leaves_the_global_rng_alone(displayed):
    """Five branches reseeded the process RNG with ``np.random.seed(42)``; the
    demo arrays now come from a local generator."""
    import numpy as np

    nbmod.PlotExplorer().show()
    before = np.random.get_state()

    _click(displayed, "Generate Plot")

    after = np.random.get_state()
    assert after[0] == before[0]
    assert np.array_equal(after[1], before[1])
    assert after[2:] == before[2:]
    plt.close("all")
