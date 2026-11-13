"""The combined HTML report: escaped, declared, and small enough to send.

``export_all_interactive`` wrote ``f"<h2>{name}</h2>"`` straight from the caller's
dict key, so a plot named ``<img src=x onerror=alert(1)>`` was markup, not text —
the same injection that ``workflow/report.py`` had.  It declared no character
encoding, so a Persian title reached the file as UTF-8 bytes the browser was free
to decode as Latin-1.  And each figure inlined its own copy of Plotly.js:
measured 4,854,499 bytes for one figure and 9,708,715 for two.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import plot_api as pa

pytest.importorskip('plotly.graph_objects')


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _one_figure():
    return pa.barplot(['x'], [1.0], interactive=True)


def _write(tmp_path, names):
    path = tmp_path / 'report.html'
    written = pa.export_all_interactive({name: _one_figure() for name in names},
                                        output_path=str(path))
    return path, written


def test_a_plot_name_is_text_and_never_markup(tmp_path):
    hostile = '<img src=x onerror=alert(1)>'

    path, _ = _write(tmp_path, [hostile, 'clean'])
    html = path.read_text(encoding='utf-8')

    assert f'<h2>{hostile}</h2>' not in html
    assert '<h2>&lt;img src=x onerror=alert(1)&gt;</h2>' in html
    assert '<h2>clean</h2>' in html
    assert html.count('<h2>') == 2, "one heading per figure, whatever the name says"


def test_the_page_declares_the_encoding_the_file_is_written_in(tmp_path):
    path, _ = _write(tmp_path, ['عنوان بیان ژن'])
    raw = path.read_bytes()

    assert b'<meta charset="utf-8">' in raw[:300]
    assert 'عنوان بیان ژن' in raw.decode('utf-8'), "the name round-trips as itself"
    assert raw.decode('utf-8').count('عنوان بیان ژن') == 1


def test_plotly_js_is_inlined_once_however_many_figures_there_are(tmp_path):
    single, _ = _write(tmp_path, ['one'])
    three, _ = _write(tmp_path, ['one', 'two', 'three'])
    small, big = single.stat().st_size, three.stat().st_size

    assert big < small * 1.4, (
        f"three figures weighed {big} against {small} for one: the library is "
        f"being embedded per figure again")
    html = three.read_text(encoding='utf-8')
    # 'Plotly.newPlot' also appears inside the inlined library itself, so the
    # per-figure marker is the div each figure is drawn into.
    assert html.count('plotly-graph-div') == 3


def test_the_report_needs_no_network_to_open(tmp_path):
    path, _ = _write(tmp_path, ['one', 'two'])
    html = path.read_text(encoding='utf-8')

    assert '<script src="https://cdn.plot.ly' not in html, \
        "a CDN reference means the page is blank without internet"


def test_the_written_path_is_returned(tmp_path):
    path, returned = _write(tmp_path, ['one'])

    assert returned == str(path)
    assert path.exists()


def test_no_figures_is_refused_and_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match=r'was given no figures'):
        pa.export_all_interactive({}, output_path=str(tmp_path / 'none.html'))

    assert list(tmp_path.iterdir()) == []


def test_a_value_that_is_not_a_plotly_figure_is_named_by_key(tmp_path):
    with pytest.raises(ValueError, match='not Plotly figures') as info:
        pa.export_all_interactive({'good': _one_figure(),
                                   'bad': plt.figure()},
                                  output_path=str(tmp_path / 'x.html'))

    assert 'bad' in str(info.value)
    assert 'interactive=True' in str(info.value)


def test_without_plotly_nothing_is_written_and_the_caller_is_told(tmp_path, monkeypatch,
                                                                  caplog, capsys):
    monkeypatch.setattr(pa, 'HAS_PLOTLY', False)
    report = tmp_path / 'report.html'

    with caplog.at_level('WARNING'):
        returned = pa.export_all_interactive({'Volcano': object()}, output_path=str(report))

    assert returned is None, "None now means refused, and a written path means saved"
    assert not report.exists()
    assert 'Plotly not installed. Cannot generate interactive report.' in \
        capsys.readouterr().out
    records = [r for r in caplog.records if 'Plotly not installed' in r.getMessage()]
    assert len(records) == 1 and records[0].name == 'biosuite.plotting.plot_api'


def test_a_mapping_that_is_not_a_dict_is_refused_as_a_type(tmp_path):
    with pytest.raises(TypeError, match=r'name -> Plotly figure, got list'):
        pa.export_all_interactive([(_one_figure(), )], output_path=str(tmp_path / 'y.html'))
