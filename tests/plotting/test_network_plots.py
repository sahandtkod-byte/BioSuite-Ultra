"""Network plots: graph builders, statistics against hand-computed values, rendering,
and the whole module as it loads when networkx is not installed."""
import importlib.util
import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pytest

from biosuite.plotting.network_plots import (
    create_metabolic_network,
    create_ppi_network,
    create_regulatory_network,
    network_statistics,
    plot_degree_distribution,
    plot_network,
)

nx = pytest.importorskip("networkx")

PPI = [("A", "B", 0.9), ("B", "C", 0.7), ("C", "D", 0.5)]
REGULATORY = [("TF1", "GeneA", "activation"), ("TF1", "GeneB", "repression"),
              ("GeneA", "GeneB", "activation")]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def without_networkx():
    """``network_plots`` executed a second time with the import of networkx blocked."""
    path = (pathlib.Path(__file__).resolve().parents[2]
            / "biosuite" / "plotting" / "network_plots.py")
    blocked = {"networkx": None}
    saved = {k: sys.modules.get(k) for k in blocked}
    sys.modules.update(blocked)
    try:
        spec = importlib.util.spec_from_file_location(
            "biosuite.plotting.network_plots_without_networkx", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


def test_create_ppi_network_stores_interaction_weights():
    graph = create_ppi_network(PPI)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert not graph.is_directed()
    assert graph["A"]["B"]["weight"] == pytest.approx(0.9)


def test_create_ppi_network_from_single_interaction():
    graph = create_ppi_network([("X", "Y", 1.0)])
    assert sorted(graph.nodes) == ["X", "Y"]
    assert graph.number_of_edges() == 1


def test_create_regulatory_network_keeps_effect_per_edge():
    graph = create_regulatory_network(REGULATORY)
    assert graph.is_directed()
    assert graph.number_of_nodes() == 3
    assert graph["TF1"]["GeneA"]["effect"] == "activation"
    assert graph["TF1"]["GeneB"]["effect"] == "repression"


def test_create_regulatory_network_from_no_edges():
    graph = create_regulatory_network([])
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_create_metabolic_network_directs_reactions_by_enzyme():
    graph = create_metabolic_network([("Glucose", "G6P", "HK"), ("G6P", "F6P", "PGI")])
    assert graph.is_directed()
    assert sorted(graph.nodes) == ["F6P", "G6P", "Glucose"]
    assert graph["Glucose"]["G6P"]["enzyme"] == "HK"


def test_network_statistics_match_values_computed_from_the_edges():
    stats = network_statistics(create_ppi_network(PPI))
    # A 4-node path: 3 edges, density 2E/(N(N-1)) = 0.5, mean degree 2E/N = 1.5.
    assert stats["nodes"] == 4
    assert stats["edges"] == 3
    assert stats["density"] == pytest.approx(0.5)
    assert stats["avg_degree"] == pytest.approx(1.5)
    assert stats["is_connected"] is True
    assert stats["clustering_coefficient"] == pytest.approx(0.0)


def test_network_statistics_reports_directed_clustering():
    """A directed triangle is not the undirected one: NetworkX's directed
    (Fagiolo) coefficient is 0.5, while the undirected projection is 1.0."""
    graph = create_regulatory_network(REGULATORY)

    stats = network_statistics(graph)

    assert stats["nodes"] == 3
    assert stats["clustering_coefficient"] == pytest.approx(0.5)
    assert nx.average_clustering(graph.to_undirected()) == pytest.approx(1.0)
    assert stats["is_connected"] is True


@pytest.mark.parametrize("graph", [None, nx.Graph()])
def test_network_statistics_of_missing_graph_is_empty(graph):
    assert network_statistics(graph) == {}


def test_plot_network_draws_every_node_and_edge():
    graph = create_ppi_network(PPI)
    fig = plot_network(graph, title="PPI Test")
    ax = fig.axes[0]
    # One collection for the nodes plus one per drawn edge, then one label each.
    assert len(ax.collections) == 1 + graph.number_of_edges()
    assert {t.get_text() for t in ax.texts} >= set(graph.nodes)
    assert ax.get_title() == "PPI Test"


@pytest.mark.parametrize("graph", [None, nx.Graph()])
def test_plot_network_without_data_shows_placeholder(graph):
    fig = plot_network(graph, title="Empty Case")
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["No network data"]
    assert ax.get_title() == "Empty Case"
    assert len(ax.collections) == 0
    assert ax.axison is False, "the empty panel keeps no ticks or spines"


def test_plot_network_colours_repression_edges_differently():
    """Directed edges are arrow patches, so the styles live on the patches."""
    from matplotlib.patches import FancyArrowPatch

    graph = create_regulatory_network(REGULATORY)
    fig = plot_network(graph)
    arrows = [p for p in fig.axes[0].patches if isinstance(p, FancyArrowPatch)]
    assert len(arrows) == graph.number_of_edges()
    assert {p.get_linestyle() for p in arrows} == {"-", "--"}

    dashed = [p for p in arrows if p.get_linestyle() == "--"]
    solid = [p for p in arrows if p.get_linestyle() == "-"]
    assert len(dashed) == 1 and len(solid) == 2
    # Repression is drawn in red, activation in green.
    assert dashed[0].get_facecolor()[0] > 0.5 and dashed[0].get_facecolor()[1] < 0.5
    assert solid[0].get_facecolor()[1] > solid[0].get_facecolor()[0]


def test_plot_degree_distribution_bars_reflect_the_graph():
    graph = create_ppi_network(PPI)
    degrees = [degree for _, degree in graph.degree()]
    expected, _bins = np.histogram(degrees, bins=max(5, len(set(degrees))), density=True)

    fig = plot_degree_distribution(graph)
    ax = fig.axes[0]
    bars = [patch for patch in ax.patches if isinstance(patch, mpatches.Rectangle)]

    assert [bar.get_height() for bar in bars] == pytest.approx(expected)
    assert ax.get_ylabel() == "Density", "a density histogram is not a frequency"
    assert ax.get_title() == "Degree Distribution"
    assert ax.get_yscale() == "log"


def test_plot_degree_distribution_of_empty_graph_shows_placeholder():
    fig = plot_degree_distribution(nx.Graph())
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["No data"]
    assert [patch for patch in ax.patches if isinstance(patch, mpatches.Rectangle)] == []
    assert ax.get_title() == "Degree Distribution"
    assert ax.axison is False


def test_plot_degree_distribution_without_a_graph_shows_the_same_placeholder():
    fig = plot_degree_distribution(None)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["No data"]
    assert ax.get_title() == "Degree Distribution"


def test_node_names_become_the_plotted_labels():
    graph = create_ppi_network(PPI, node_names={"A": "Protein A"})

    assert graph.nodes["A"]["label"] == "Protein A"
    fig = plot_network(graph)
    labelled = {text.get_text() for text in fig.axes[0].texts}
    assert "Protein A" in labelled
    assert "A" not in labelled


def test_a_node_name_for_a_node_the_network_lacks_is_refused():
    with pytest.raises(ValueError, match=r"node_names names \['Z'\]"):
        create_ppi_network(PPI, node_names={"Z": "missing"})


def test_node_names_must_be_a_mapping():
    with pytest.raises(ValueError, match="must be a mapping of node IDs"):
        create_ppi_network(PPI, node_names=["A", "B"])


def test_the_nodes_take_the_colour_the_caller_asked_for():
    fig = plot_network(create_ppi_network(PPI), node_color="#ff0000")

    faces = fig.axes[0].collections[0].get_facecolor()
    assert {tuple(round(float(value), 3) for value in face[:3]) for face in faces} == {
        (1.0, 0.0, 0.0)}


def test_more_than_fifty_nodes_hide_the_labels_and_say_so():
    fig = plot_network(nx.path_graph(60), show_labels=True, title="Chain")

    ax = fig.axes[0]
    assert [text for text in ax.texts if text.get_text()] == []
    assert ax.get_title() == "Chain (labels hidden for 60 nodes)"


def test_fifty_nodes_still_get_their_labels():
    fig = plot_network(nx.path_graph(50), show_labels=True, title="Chain")

    ax = fig.axes[0]
    assert len([text for text in ax.texts if text.get_text()]) == 50
    assert ax.get_title() == "Chain"


def test_an_effect_outside_the_two_documented_kinds_is_refused():
    with pytest.raises(ValueError, match="has effect 'activating'"):
        create_regulatory_network([("A", "B", "activating")])


def test_a_repeated_interaction_keeps_the_last_weight():
    """Documented in ``create_ppi_network``: ``nx.Graph`` overwrites the pair."""
    graph = create_ppi_network([("A", "B", 0.9), ("A", "B", 0.1)])

    assert graph["A"]["B"]["weight"] == pytest.approx(0.1)


def test_plot_network_labels_only_when_the_caller_asks_for_it():
    graph = create_ppi_network(PPI)

    fig = plot_network(graph, show_labels=False)

    assert [t.get_text() for t in fig.axes[0].texts] == [], "no node was named on the canvas"
    assert len(fig.axes[0].collections) == 1 + graph.number_of_edges(), "the nodes and edges still drew"


def test_plot_network_draws_on_an_axes_the_caller_owns():
    outer, ax = plt.subplots()

    fig = plot_network(create_ppi_network(PPI), ax=ax)

    assert fig is outer, "the caller's figure comes back, not a fresh one"
    assert len(outer.axes) == 1


def test_plot_degree_distribution_draws_on_an_axes_the_caller_owns():
    outer, ax = plt.subplots()

    fig = plot_degree_distribution(create_ppi_network(PPI), ax=ax)

    assert fig is outer
    assert len(outer.axes) == 1
    assert ax.get_title() == "Degree Distribution"


def test_the_module_loads_without_networkx_and_the_builders_say_so(without_networkx):
    assert without_networkx.HAS_NX is False


def test_graph_builders_return_nothing_without_networkx(without_networkx):
    assert without_networkx.create_ppi_network(PPI) is None
    assert without_networkx.create_regulatory_network(REGULATORY) is None
    assert without_networkx.create_metabolic_network([("a", "b", "e")]) is None


def test_plotting_still_works_without_networkx_and_says_there_is_no_graph(without_networkx):
    fig = without_networkx.plot_network(None, title="Offline")

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["No network data"]
    assert ax.get_title() == "Offline"
    assert len(ax.collections) == 0
