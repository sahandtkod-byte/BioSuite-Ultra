"""An alignment this wide cannot be drawn one cell per base.

``alignment_viewer`` sized its figure from the data — ``figsize=(max_len*0.3,
n*0.4)`` — so a 2,000-column alignment asked matplotlib for a **600 x 0.8 inch**
canvas and drew 4,001 rectangles on it.  Nothing said the figure had become
unreadable; the user got a PNG that was either unopenable or a smear, and the
height shrank with the number of sequences instead of growing.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import biological_plots as bp

CAP_WIDTH = 42.0
CAP_HEIGHT = 30.0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.fixture
def drive_alignment(monkeypatch):
    import builtins
    import contextlib
    import io

    def run(sequences):
        remaining = list(sequences) + ['']
        captured = {}

        def ask(prompt=""):
            if prompt.strip():
                if "Use default alignment" in prompt:
                    return 'n'
                return ''
            return remaining.pop(0) if remaining else ''

        def spy(default_name, save_format, dpi, pdf=None, story=None):
            fig = plt.gcf()
            captured['figsize'] = tuple(np.round(fig.get_size_inches(), 2))
            captured['rectangles'] = sum(1 for p in fig.axes[0].patches
                                         if type(p).__name__ == 'Rectangle')
            captured['name'] = default_name
            plt.close(fig)
            return None

        monkeypatch.setattr(builtins, 'input', ask)
        monkeypatch.setattr(bp, 'ask_save_plot', spy)
        monkeypatch.setitem(bp.config, 'quiet', False)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            bp.alignment_viewer()
        return captured, printed.getvalue()

    return run


def test_a_normal_alignment_keeps_the_size_the_cells_deserve(drive_alignment):
    captured, out = drive_alignment(['ATCGATCG', 'ATCGATCG'])

    assert captured['figsize'] == (2.4, 0.8), captured['figsize']
    assert 'capped' not in out, out


def test_a_wide_alignment_is_capped_and_the_cap_is_named(drive_alignment):
    wide = ['A' * 1500]

    captured, out = drive_alignment(wide)

    assert captured['figsize'] == (CAP_WIDTH, 0.4), captured['figsize']
    assert "1500 columns need 450.0 x 0.4 inches" in out, out
    assert "the figure is capped at 42.0 x 0.4 inches" in out, out
    assert "Subset the alignment" in out, out


def test_a_tall_alignment_is_capped_too(drive_alignment):
    many = ['ACGT'] * 100

    captured, out = drive_alignment(many)

    assert captured['figsize'] == (1.2, CAP_HEIGHT), captured['figsize']
    assert "100 sequences x 4 columns need 1.2 x 40.0 inches" in out, out
    assert "capped at 1.2 x 30.0 inches" in out, out


def test_both_dimensions_capped_at_once(drive_alignment):
    captured, out = drive_alignment(['A' * 200] * 100)

    assert captured['figsize'] == (CAP_WIDTH, CAP_HEIGHT), captured['figsize']
    assert "100 sequences x 200 columns need 60.0 x 40.0 inches" in out, out


def test_the_cells_are_still_drawn_one_per_base(drive_alignment):
    captured, _out = drive_alignment(['ACG', 'ACT'])

    assert captured['rectangles'] == 6, "two sequences x three bases"
    assert captured['name'] == 'alignment'
