"""A gene-order strip draws one block per gene, in the order it was given.

The blocks are laid out left to right, the names sit inside them, and a caller's
axes is used as-is.  A colour list shorter than the gene list is cycled rather
than silently dropped.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest

from biosuite.plotting.synteny import plot_gene_order


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_one_block_per_gene_in_the_order_given():
    fig = plot_gene_order(["C", "A", "B"], genome_name="Test genome")
    ax = fig.axes[0]

    assert [p.get_x() for p in ax.patches] == [0, 1, 2]
    assert [t.get_text() for t in ax.texts] == ["C", "A", "B"]
    assert ax.get_title() == "Test genome"


def test_a_supplied_axes_holds_the_whole_strip():
    outer, ax = plt.subplots()

    fig = plot_gene_order(["A", "B"], ax=ax)

    assert fig is outer
    assert len(outer.axes) == 1
    assert len(ax.patches) == 2


def test_a_short_colour_list_is_cycled_over_the_genes():
    fig = plot_gene_order(["A", "B", "C"], colors=["#ff0000", "#0000ff"])
    faces = [p.get_facecolor() for p in fig.axes[0].patches]

    assert faces[0] == pytest.approx(mcolors.to_rgba("#ff0000"))
    assert faces[1] == pytest.approx(mcolors.to_rgba("#0000ff"))
    assert faces[2] == pytest.approx(mcolors.to_rgba("#ff0000"))
