"""The Interactive Plots tab: demo and CSV data, and the eleven plot types it offers.

``_inter_generate`` hands the data to ``biosuite.plotting.plot_api`` for the newer
types and to ``biosuite.plotting.interactive_plots`` for the legacy ones, writing an
HTML file the user opens in a browser.  Each test drives the handler for one combo
entry and then checks the file that was actually written (or the message that was
shown when nothing was), so the entries the tab advertises are the entries it has.
"""
import numpy as np
import pandas as pd
import pytest

from biosuite.plotting import interactive_plots

COMBO_TYPES = ['Scatter', 'Bar', 'Heatmap', 'Volcano', 'Line', 'Pie',
               'PCA', 'Manhattan', 'Boxplot', 'Violin', 'QQ-Plot']

# Only these read the loaded table; the rest draw BioSuite's own arrays and must
# say so in the box instead of announcing a save over data the user never gave.
FROM_TABLE = {'Scatter', 'Bar', 'Heatmap'}

CSV = pd.DataFrame({"sample": ["s1", "s2", "s3"],
                    "value": [1.5, 2.5, 3.5],
                    "other": [10.0, 20.0, 30.0]})


@pytest.fixture
def interactive(ui):
    ui.app._inter_data = None
    ui.clear("inter_info")
    return ui


@pytest.fixture
def demo_data(interactive):
    interactive.app._inter_demo()
    return interactive


def _demo_arrays():
    """The arrays the demo builds, generated the same way."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(100), rng.standard_normal(100),
            rng.choice(['A', 'B', 'C'], 100))


def test_the_demo_data_is_the_points_and_groups_the_box_announces(demo_data):
    x, y, label = _demo_arrays()

    data = demo_data.app._inter_data
    assert set(data) == {"x", "y", "label"}
    assert np.array_equal(data["x"], x)
    assert np.array_equal(data["y"], y)
    assert np.array_equal(data["label"], label)
    assert "Demo data loaded: 100 points, 3 groups" in demo_data.text("inter_info")


def test_the_demo_leaves_the_global_rng_alone(interactive):
    before = np.random.get_state()

    interactive.app._inter_demo()

    after = np.random.get_state()
    assert after[0] == before[0]
    assert np.array_equal(after[1], before[1])
    assert after[2:] == before[2:]


def test_loading_a_csv_shows_the_columns_and_shape_of_the_file(interactive, tmp_path):
    path = tmp_path / "counts.csv"
    path.write_text("sample,value,other\ns1,1.5,10.0\ns2,2.5,20.0\ns3,3.5,30.0\n",
                    encoding="utf-8")
    interactive.open_path = str(path)

    interactive.app._inter_load_csv()

    info = interactive.text("inter_info")
    assert f"Loaded: {path}" in info
    assert f"Columns: {list(CSV.columns)}" in info, info
    assert f"Shape: {CSV.shape}" in info, info
    assert interactive.app._inter_data.equals(CSV)


def test_a_cancelled_load_dialog_leaves_the_data_alone(interactive):
    interactive.app._inter_data = None
    interactive.open_path = ""

    interactive.app._inter_load_csv()

    assert interactive.app._inter_data is None
    assert interactive.text("inter_info").strip() == ""


def test_a_csv_that_cannot_be_read_is_reported(interactive, tmp_path):
    interactive.app._inter_data = None
    missing = tmp_path / "absent.csv"
    interactive.open_path = str(missing)

    interactive.app._inter_load_csv()

    errors = interactive.messages_of("error")
    assert len(errors) == 1 and errors[0].startswith("Could not load CSV:")
    assert "absent.csv" in errors[0], errors[0]
    assert interactive.app._inter_data is None


def test_generating_without_data_asks_for_some(interactive, tmp_path):
    interactive.app._inter_data = None

    interactive.app._inter_generate()

    assert interactive.messages_of("warning") == ["Load CSV or use Demo Data first."]
    assert not (tmp_path / "interactive_output.html").exists()


@pytest.mark.parametrize("plot_type", COMBO_TYPES)
def test_every_plot_type_the_combo_offers_writes_its_html(demo_data, tmp_path, plot_type):
    demo_data.app.inter_type.set(plot_type)

    demo_data.app._inter_generate()

    written = tmp_path / "interactive_output.html"
    assert written.exists(), f"{plot_type} claimed to save but wrote nothing"
    assert written.stat().st_size > 1000, f"{plot_type} wrote an empty document"
    text = demo_data.text("inter_info")
    assert "Saved: interactive_output.html" in text, text
    assert f"Plot type: {plot_type.lower()}" in text, text
    assert demo_data.messages_of("success") ==             ["Interactive plot saved to:\ninteractive_output.html"]
    assert "does not read the loaded table" in text, \
        f"{plot_type} drew demo data without saying so"


@pytest.mark.parametrize("plot_type", ['Scatter', 'Bar', 'Heatmap'])
def test_a_loaded_csv_is_offered_to_the_legacy_plot_types(interactive, tmp_path, plot_type):
    path = tmp_path / "counts.csv"
    CSV.to_csv(path, index_label="sample")
    interactive.open_path = str(path)
    interactive.app._inter_load_csv()
    interactive.app.inter_type.set(plot_type)

    interactive.app._inter_generate()

    assert (tmp_path / "interactive_output.html").exists()
    assert interactive.messages_of("warning") == []
    assert interactive.messages_of("error") == []
    assert "Sample data" not in interactive.text("inter_info"),             f"{plot_type} ignored the table it was given"


def test_the_plot_type_choice_cannot_be_typed_over(interactive):
    assert interactive.app.inter_type.cget("state") == "readonly"


def test_a_plot_type_that_is_not_offered_is_refused(demo_data, tmp_path):
    """A typed value used to fall through to a scatter, so the box announced a
    scatter while the picker said something else."""
    demo_data.set("inter_info", "earlier info")
    demo_data.app.inter_type.set("Radar")

    demo_data.app._inter_generate()

    assert demo_data.messages_of("error") == [
        "Unknown plot type: 'Radar'. Choose Scatter, Bar, Heatmap, Volcano, "
        "Line, Pie, PCA, Manhattan, Boxplot, Violin or QQ-Plot."]
    assert demo_data.messages_of("success") == []
    assert demo_data.text("inter_info").strip() == "earlier info"
    assert not (tmp_path / "interactive_output.html").exists()


def test_a_scatter_needs_two_numeric_columns(interactive, tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("value,note\n1.0,a\n2.0,b\n", encoding="utf-8")
    interactive.open_path = str(path)
    interactive.app._inter_load_csv()
    interactive.app.inter_type.set('Scatter')

    interactive.app._inter_generate()

    assert interactive.messages_of("warning") ==             ["A scatter needs at least two numeric columns."]
    assert not (tmp_path / "interactive_output.html").exists(),             "the tab claimed a save it never made"


def test_a_loaded_csv_with_a_sample_plot_type_is_flagged(interactive, tmp_path):
    """A volcano drawn over a loaded table still uses BioSuite's own arrays, so
    the note must say so even though the data source is a DataFrame."""
    path = tmp_path / "counts.csv"
    CSV.to_csv(path, index_label="sample")
    interactive.open_path = str(path)
    interactive.app._inter_load_csv()
    interactive.app.inter_type.set('Volcano')

    interactive.app._inter_generate()

    assert (tmp_path / "interactive_output.html").exists()
    assert "does not read the loaded table" in interactive.text("inter_info")


def test_a_failing_plotter_is_reported_with_its_reason(demo_data, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("plotly is not installed")

    monkeypatch.setattr(interactive_plots, "interactive_scatter", broken)
    demo_data.app.inter_type.set('Scatter')

    demo_data.app._inter_generate()

    assert demo_data.messages_of("error") == ["plotly is not installed"]
    assert demo_data.messages_of("success") == []


@pytest.mark.parametrize("plot_type,entry", [
    ('Scatter', '_gui_interactive_scatter'),
    ('Bar', '_gui_interactive_bar'),
    ('Heatmap', '_gui_interactive_heatmap'),
    ('Volcano', '_gui_interactive_volcano'),
    ('Line', '_gui_interactive_line'),
    ('Pie', '_gui_interactive_pie'),
])
def test_the_sidebar_entries_show_the_frame_and_draw_their_plot(
        interactive, tmp_path, plot_type, entry):
    getattr(interactive.app, entry)()

    assert interactive.app._current_frame == "interactive"
    assert interactive.app.inter_type.get() == plot_type
    assert (tmp_path / "interactive_output.html").exists(),             f"{entry} drew nothing"
    assert f"Plot type: {plot_type.lower()}" in interactive.text("inter_info")
