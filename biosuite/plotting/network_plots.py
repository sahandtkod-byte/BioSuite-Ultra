"""
Network visualization for PPI, gene regulatory, and metabolic networks.
Uses networkx for graph construction and matplotlib for rendering.
"""
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

# Public surface. ``biosuite/plotting/__init__.py`` star-imports this
# module, and without this list it re-exports the module's own imports
# (numpy, pandas, os, plotly, ...) as package-level names too.
__all__ = [
    'create_ppi_network',
    'create_regulatory_network',
    'create_metabolic_network',
    'plot_network',
    'plot_degree_distribution',
    'network_statistics',
]

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from ..core.utils import apply_glass_ax, config

#: Beyond this many nodes every label is an unreadable smear, so the panel says
#: that it hid them instead of drawing a blank canvas.
_MAX_LABELLED_NODES = 50

_REGULATORY_EFFECTS = ('activation', 'repression')


def _placeholder(ax, title, message):
    """Draw the empty-panel state both plot functions share."""
    ax.text(0.5, 0.5, message, ha='center', va='center',
            transform=ax.transAxes, color='gray')
    ax.set_title(title)
    ax.axis('off')
    apply_glass_ax(ax)


def create_ppi_network(interactions, node_names=None):
    """Create a protein-protein interaction network.

    Args:
        interactions: list of (protein_a, protein_b, weight) tuples.  A pair
            that appears twice keeps the last weight, as ``nx.Graph`` does.
        node_names: optional dict mapping IDs to display names.  Every key must
            be a node in the network; the names are stored as each node's
            ``label`` attribute and used by ``plot_network``.

    Raises:
        ValueError: if ``node_names`` names a node the interactions do not have.
    """
    if not HAS_NX:
        return None
    G = nx.Graph()
    for a, b, w in interactions:
        G.add_edge(a, b, weight=w)
    if node_names:
        if not isinstance(node_names, Mapping):
            raise ValueError(
                "node_names must be a mapping of node IDs to labels, "
                f"got {type(node_names).__name__}")
        unknown = sorted(set(node_names) - set(G.nodes))
        if unknown:
            raise ValueError(
                f"node_names names {unknown}, which the interactions do not "
                "produce; remove them or add the interactions")
        for node, label in node_names.items():
            G.nodes[node]['label'] = label
    return G


def create_regulatory_network(edges):
    """Create a gene regulatory network with activation/repression.

    Args:
        edges: list of (regulator, target, 'activation'|'repression') tuples.

    Raises:
        ValueError: if an edge's effect is not one of the two documented kinds.
    """
    if not HAS_NX:
        return None
    G = nx.DiGraph()
    for reg, tgt, etype in edges:
        if etype not in _REGULATORY_EFFECTS:
            raise ValueError(
                f"regulatory edge {reg!r} -> {tgt!r} has effect {etype!r}; "
                f"use one of {_REGULATORY_EFFECTS}")
        G.add_edge(reg, tgt, effect=etype)
    return G


def create_metabolic_network(reactions):
    """Create a metabolic network from reaction data.

    Args:
        reactions: list of (substrate, product, enzyme) tuples.
    """
    if not HAS_NX:
        return None
    G = nx.DiGraph()
    for sub, prod, enzyme in reactions:
        G.add_edge(sub, prod, enzyme=enzyme)
    return G


def plot_network(G, title="Network", node_color='#00ff88', edge_color='gray',
                 node_size=300, figsize=(10, 8), show_labels=True, ax=None):
    """Plot a network graph using matplotlib.

    Args:
        G: networkx graph.
        title: plot title.
        node_color: node fill color, or one color per node.
        edge_color: color of an edge with no activation/repression effect.
        node_size: node size.
        figsize: figure size when ``ax`` is None.
        show_labels: name the nodes.  Above 50 nodes the names are hidden and
            the title records the cutoff.
        ax: optional axes; the caller keeps ownership of its figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if not HAS_NX or G is None or len(G) == 0:
        _placeholder(ax, title, "No network data")
        return fig

    if show_labels and len(G) > _MAX_LABELLED_NODES:
        title = f"{title} (labels hidden for {len(G)} nodes)"

    pos = nx.spring_layout(G, k=2/np.sqrt(len(G)), seed=42)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color=node_color, alpha=0.85)

    edge_styles = []
    for _u, _v, data in G.edges(data=True):
        if data.get('effect') == 'activation':
            edge_styles.append(('green', '-', 1.5))
        elif data.get('effect') == 'repression':
            edge_styles.append(('red', '--', 1.5))
        else:
            edge_styles.append((edge_color, '-', 1.0))

    for (u, v), (color, style, width) in zip(G.edges(), edge_styles):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                               edge_color=color, style=style, width=width, alpha=0.6)

    if show_labels and len(G) <= _MAX_LABELLED_NODES:
        labels = {}
        for n in G.nodes():
            label = str(G.nodes[n].get('label', n))
            labels[n] = label if len(label) < 12 else label[:10] + '..'
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='white', ax=ax)

    ax.set_title(title, fontsize=14, color=config.get('text', '#00ff99'))
    ax.axis('off')
    apply_glass_ax(ax)
    return fig


def plot_degree_distribution(G, ax=None):
    """Plot the degree distribution of a network as a density histogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    if G is None or len(G) == 0:
        _placeholder(ax, 'Degree Distribution', "No data")
        return fig

    degrees = [d for _, d in G.degree()]
    hist, bins = np.histogram(degrees, bins=max(5, len(set(degrees))), density=True)
    ax.bar(bins[:-1], hist, width=np.diff(bins)[0]*0.8, color='#00ff88', alpha=0.7, edgecolor='#00cc6a')
    ax.set_xlabel('Degree', color=config.get('text', '#00ff99'))
    ax.set_ylabel('Density', color=config.get('text', '#00ff99'))
    ax.set_title('Degree Distribution')
    ax.set_yscale('log')
    apply_glass_ax(ax)
    return fig


def network_statistics(G):
    """Summarise a graph.

    ``is_connected`` means ``nx.is_connected`` for an undirected graph and
    weak connectivity for a directed one, matching NetworkX's own distinction.
    ``clustering_coefficient`` is NetworkX's average clustering: the directed
    (Fagiolo) coefficient for a directed graph, not an assumed zero.
    ``avg_degree`` counts both ends of every edge, so a directed graph's
    in-degree and out-degree are both included.

    Returns an empty dict for a missing or empty graph.
    """
    if G is None or len(G) == 0:
        return {}
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': round(nx.density(G), 4),
        'avg_degree': round(2 * G.number_of_edges() / max(G.number_of_nodes(), 1), 2),
        'is_connected': nx.is_connected(G) if not G.is_directed() else nx.is_weakly_connected(G),
        'clustering_coefficient': round(nx.average_clustering(G), 4),
    }
