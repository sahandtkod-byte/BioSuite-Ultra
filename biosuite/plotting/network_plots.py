"""
Network visualization for PPI, gene regulatory, and metabolic networks.
Uses networkx for graph construction and matplotlib for rendering.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from ..core.utils import apply_glass_ax, config


def create_ppi_network(interactions, node_names=None):
    """Create a protein-protein interaction network.

    Args:
        interactions: list of (protein_a, protein_b, weight) tuples.
        node_names: optional dict mapping IDs to display names.
    """
    if not HAS_NX:
        return None
    G = nx.Graph()
    for a, b, w in interactions:
        G.add_edge(a, b, weight=w)
    return G


def create_regulatory_network(edges):
    """Create a gene regulatory network with activation/repression.

    Args:
        edges: list of (regulator, target, 'activation'|'repression') tuples.
    """
    if not HAS_NX:
        return None
    G = nx.DiGraph()
    for reg, tgt, etype in edges:
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
        node_color: node fill color.
        node_size: node size.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if not HAS_NX or G is None or len(G) == 0:
        ax.text(0.5, 0.5, "No network data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

    pos = nx.spring_layout(G, k=2/np.sqrt(len(G)), seed=42)

    node_colors = []
    degrees = dict(G.degree())
    for node in G.nodes():
        node_colors.append(degrees[node])

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color=node_colors, cmap=plt.cm.viridis, alpha=0.85)

    edge_styles = []
    for u, v, data in G.edges(data=True):
        if data.get('effect') == 'activation':
            edge_styles.append(('green', '-', 1.5))
        elif data.get('effect') == 'repression':
            edge_styles.append(('red', '--', 1.5))
        else:
            edge_styles.append((edge_color, '-', 1.0))

    for (u, v), (color, style, width) in zip(G.edges(), edge_styles):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                               edge_color=color, style=style, width=width, alpha=0.6)

    if show_labels and len(G) <= 50:
        labels = {}
        for n in G.nodes():
            labels[n] = n if not isinstance(n, str) or len(n) < 12 else n[:10] + '..'
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='white', ax=ax)

    ax.set_title(title, fontsize=14, color=config.get('text', '#00ff99'))
    ax.axis('off')
    apply_glass_ax(ax)
    return fig


def plot_degree_distribution(G, ax=None):
    """Plot degree distribution of a network."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    degrees = [d for _, d in G.degree()]
    if not degrees:
        ax.text(0.5, 0.5, "No data", ha='center', transform=ax.transAxes)
        return fig

    hist, bins = np.histogram(degrees, bins=max(5, len(set(degrees))), density=True)
    ax.bar(bins[:-1], hist, width=np.diff(bins)[0]*0.8, color='#00ff88', alpha=0.7, edgecolor='#00cc6a')
    ax.set_xlabel('Degree', color=config.get('text', '#00ff99'))
    ax.set_ylabel('Frequency', color=config.get('text', '#00ff99'))
    ax.set_title('Degree Distribution')
    ax.set_yscale('log')
    apply_glass_ax(ax)
    return fig


def network_statistics(G):
    if G is None or len(G) == 0:
        return {}
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': round(nx.density(G), 4),
        'avg_degree': round(2 * G.number_of_edges() / max(G.number_of_nodes(), 1), 2),
        'is_connected': nx.is_connected(G) if not G.is_directed() else nx.is_weakly_connected(G),
        'clustering_coefficient': round(nx.average_clustering(G), 4) if not G.is_directed() else 0,
    }
