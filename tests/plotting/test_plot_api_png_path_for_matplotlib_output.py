"""The PNG that stands in for a matplotlib figure asked to write HTML.

``_save_mpl_figure`` did ``str(output_html).replace('.html', '.png')``.  A
``report.pdf`` request therefore came back as a real PDF under a name that said
PDF — the docstring promises a PNG — and a path with ``.html`` twice rewrote
both halves, so ``out.html/run.html`` landed as ``out.png/run.png`` in a
directory that no longer existed.  A name with no extension was passed to
``savefig`` unchanged, which is the one shape matplotlib refuses.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import plot_api as pa


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


@pytest.mark.parametrize('requested,expected', [
    ('report.html', 'report.png'),
    ('report.HTML', 'report.png'),
    ('report.htm', 'report.png'),
    ('report.pdf', 'report.pdf.png'),
    ('out.html/run.html', 'out.html/run.png'),
    ('plain', 'plain.png'),
    ('archive.tar.html', 'archive.tar.png'),
])
def test_the_requested_stem_becomes_a_png_name(requested, expected):
    assert pa._png_path(requested) == expected


def test_a_html_suffix_inside_a_directory_name_survives(tmp_path):
    nested = tmp_path / 'snap.html'
    nested.mkdir()
    target = nested / 'figure.html'
    fig, ax = plt.subplots()
    ax.plot([1.0, 2.0], [3.0, 4.0])

    pa._save_mpl_figure(fig, str(target))

    assert [p.name for p in nested.iterdir()] == ['figure.png']
    assert (nested / 'figure.png').read_bytes()[:4] == b'\x89PNG'


def test_a_pdf_name_gets_a_png_beside_it_not_a_pdf(tmp_path):
    target = tmp_path / 'figure.pdf'
    fig, ax = plt.subplots()
    ax.plot([1.0, 2.0], [3.0, 4.0])

    pa._save_mpl_figure(fig, str(target))

    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ['figure.pdf.png'], "the docstring promises a PNG, so a PNG is written"
    assert not target.exists(), "nothing is written under the PDF name"
    assert (tmp_path / 'figure.pdf.png').read_bytes()[:4] == b'\x89PNG'


def test_no_path_means_no_file(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1.0], [1.0])

    pa._save_mpl_figure(fig, None)
    pa._save_mpl_figure(fig, '')

    assert list(tmp_path.iterdir()) == []


def test_a_plot_with_an_output_argument_leaves_exactly_one_png(tmp_path):
    """The end-to-end contract every plot shares: HTML in, PNG out."""
    target = tmp_path / 'volcano.html'

    pa.volcano([2.0, -2.5], [0.001, 0.4], output_html=str(target))

    assert [p.name for p in tmp_path.iterdir()] == ['volcano.png']
    assert (tmp_path / 'volcano.png').stat().st_size > 1000
