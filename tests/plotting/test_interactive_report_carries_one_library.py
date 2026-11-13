"""The combined report carries plotly.js once, escapes its titles and says where it went.

Three plots used to make a 14,563,191-byte file — ``plotly.io.to_html`` embedded
the whole library once per figure (the same three figures cost 4,870,142 bytes
through ``plot_api.export_all_interactive``).  A plot name went into an ``<h2>``
verbatim, so ``<script>alert(1)</script>`` reached the file as live markup, and
the function returned ``None`` whether or not it had written anything.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import interactive_plots as ip


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _figure():
    return ip.interactive_scatter([1.0, 2.0], [3.0, 4.0])


def _report_size(path):
    return path.stat().st_size


def test_three_plots_do_not_cost_three_copies_of_the_library(tmp_path):
    one = tmp_path / "one.html"
    three = tmp_path / "three.html"

    ip.export_interactive_report({"a": _figure()}, str(one))
    ip.export_interactive_report({"a": _figure(), "b": _figure(), "c": _figure()},
                                 str(three))

    # The library dominates the file: three copies would be about three times
    # the size of one report, one copy stays near it.
    assert _report_size(three) < 1.5 * _report_size(one), (
        _report_size(one), _report_size(three))


def test_the_report_is_self_contained_for_an_offline_reader(tmp_path):
    out = tmp_path / "report.html"
    ip.export_interactive_report({"a": _figure()}, str(out))

    text = out.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in text, "the library is inline, not fetched"
    # The library's own source mentions cdn.plot.ly as a config default, so the
    # check has to be for a script tag that would fetch it, not a substring.
    assert '<script src="https://' not in text, "an offline reader must need no CDN"


def test_a_plot_name_is_escaped_into_the_report(tmp_path):
    out = tmp_path / "report.html"

    ip.export_interactive_report({'<script>alert(1)</script>': _figure()}, str(out))

    text = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in text, "the name reached the HTML as markup"
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text


def test_the_report_returns_the_path_it_wrote(tmp_path):
    out = tmp_path / "report.html"

    returned = ip.export_interactive_report({"a": _figure()}, str(out))

    assert returned == str(out)
    assert out.exists()


@pytest.mark.parametrize("plots, reason", [
    ({}, "at least one figure"),
    ({"bad": object()}, "plot 'bad' is not a plotly Figure"),
])
def test_a_report_it_cannot_write_is_refused_before_the_file_exists(plots, reason, tmp_path):
    out = tmp_path / "report.html"

    with pytest.raises(ValueError, match=reason):
        ip.export_interactive_report(plots, str(out))

    assert not out.exists(), "a refused report left a file behind"
