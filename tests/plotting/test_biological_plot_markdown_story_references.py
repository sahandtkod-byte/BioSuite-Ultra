"""The plot story only promises images that are in the folder.

``generate_markdown_story`` wrote ``![{name}]({name}.png)`` for every name it was
given — a name whose PNG the batch export had *failed* to write (the exporter
prints ``FAILED to save`` and carries on) still got a heading and a link, so the
report advertised plots that did not exist.  And a name containing a square
bracket ended the alt text early: ``x](y`` came out as ``![x](y](x](y.png)``,
which no renderer can read.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import biological_plots as bp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_every_plot_is_listed_with_its_own_heading(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for name in ('volcano', 'pca'):
        (tmp_path / f"{name}.png").write_bytes(b'\x89PNG\r\n\x1a\n')

    bp.generate_markdown_story(['volcano', 'pca'], output_path=str(tmp_path / 's.md'))

    text = (tmp_path / 's.md').read_text(encoding='utf-8')
    assert '![volcano](volcano.png)' in text, text
    assert '![pca](pca.png)' in text, text
    assert '## Volcano' in text and '## Pca' in text, text
    assert "This report contains 2 exported plots." in text, text
    assert 'not in this folder' not in capsys.readouterr().out


def test_a_plot_that_was_never_written_is_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'volcano.png').write_bytes(b'\x89PNG\r\n\x1a\n')

    bp.generate_markdown_story(['volcano', 'dot_plot'],
                              output_path=str(tmp_path / 's.md'))

    out = capsys.readouterr().out
    assert "1 of 2 referenced images are not in this folder: dot_plot.png" in out, out
    text = (tmp_path / 's.md').read_text(encoding='utf-8')
    assert '![dot_plot](dot_plot.png)' in text, "listed anyway, by design"


def test_a_name_with_brackets_cannot_break_the_link(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    bp.generate_markdown_story(['x](y', 'a[b'], output_path=str(tmp_path / 's.md'))

    text = (tmp_path / 's.md').read_text(encoding='utf-8')
    assert r'![x\](y](<x](y.png>)' in text, repr(text)
    assert r'![a\[b](<a[b.png>)' in text, repr(text)
    assert '![x](y](' not in text, repr(text)
    assert "2 of 2 referenced images are not in this folder" in capsys.readouterr().out


def test_no_plots_at_all_is_a_story_about_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    bp.generate_markdown_story([], output_path=str(tmp_path / 'empty.md'))

    text = (tmp_path / 'empty.md').read_text(encoding='utf-8')
    assert "This report contains 0 exported plots." in text, text
    assert '![' not in text, text
    assert '### ' not in text, text
    assert 'not in this folder' not in capsys.readouterr().out


def test_the_story_is_written_in_utf_8(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    bp.generate_markdown_story(['بیان_ژن'], output_path=str(tmp_path / 's.md'))

    text = (tmp_path / 's.md').read_text(encoding='utf-8')
    assert '![بیان_ژن](بیان_ژن.png)' in text, text
    assert '### بیان ژن' in text, text
