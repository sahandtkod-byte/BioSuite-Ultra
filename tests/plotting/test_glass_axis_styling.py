"""``apply_glass_ax``: the rounded border it draws, and the axes it must skip."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from biosuite.core.utils import apply_glass_ax


def _borders(ax):
    return sum(isinstance(p, FancyBboxPatch) for p in ax.patches)


def test_glass_border_is_drawn_on_a_cartesian_axes():
    """Positive control: the helper really does add one rounded box."""
    fig, ax = plt.subplots()
    apply_glass_ax(ax)

    assert _borders(ax) == 1


def test_glass_border_is_skipped_on_polar_axes():
    """A polar axes has no 'top' spine, so the border cannot be built.

    Regression: the ``KeyError('top')`` used to escape here and kill
    ``circos_plot`` after it had already drawn the plot.
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    apply_glass_ax(ax)

    assert _borders(ax) == 0
