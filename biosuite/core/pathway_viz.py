"""
Pathway visualization — draw KEGG/Reactome-style pathway maps
with gene expression overlays. Pure Python/matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from collections import OrderedDict


class PathwayNode:
    """Node in a pathway map."""

    def __init__(self, node_id, name, x=0, y=0, node_type="gene", width=1.0, height=0.5):
        self.node_id = node_id
        self.name = name
        self.x = x
        self.y = y
        self.node_type = node_type
        self.width = width
        self.height = height
        self.expression = None
        self.color = None

    def set_expression(self, value, vmin=-2, vmax=2):
        self.expression = value
        norm = np.clip((value - vmin) / (vmax - vmin), 0, 1)
        if norm < 0.5:
            # Blue (downregulated)
            r = int(30 + norm * 2 * 200)
            g = int(80 + norm * 2 * 100)
            b = int(220 - norm * 2 * 50)
        else:
            # Red (upregulated)
            r = int(230)
            g = int(230 - (norm - 0.5) * 2 * 200)
            b = int(170 - (norm - 0.5) * 2 * 150)
        self.color = f'#{r:02x}{g:02x}{b:02x}'


class PathwayEdge:
    """Edge between pathway nodes."""

    def __init__(self, source, target, edge_type="activation", label=""):
        self.source = source
        self.target = target
        self.edge_type = edge_type
        self.label = label


class PathwayMap:
    """A complete pathway map."""

    def __init__(self, name="pathway"):
        self.name = name
        self.nodes = OrderedDict()
        self.edges = []
        self.title = name

    def add_node(self, node_id, name, x=0, y=0, **kwargs):
        node = PathwayNode(node_id, name, x, y, **kwargs)
        self.nodes[node_id] = node
        return self

    def add_edge(self, source, target, edge_type="activation", label=""):
        self.edges.append(PathwayEdge(source, target, edge_type, label))
        return self

    def set_expression(self, expr_dict, vmin=-2, vmax=2):
        for node_id, value in expr_dict.items():
            if node_id in self.nodes:
                self.nodes[node_id].set_expression(value, vmin, vmax)

    def layout_grid(self, n_cols=3, spacing_x=3, spacing_y=2):
        """Arrange nodes in a grid."""
        for i, (nid, node) in enumerate(self.nodes.items()):
            row = i // n_cols
            col = i % n_cols
            node.x = col * spacing_x
            node.y = -row * spacing_y

    def layout_linear(self, spacing=3):
        """Arrange nodes in a line."""
        for i, (nid, node) in enumerate(self.nodes.items()):
            node.x = i * spacing
            node.y = 0


def draw_pathway(pathway_map, title=None, figsize=(12, 8), ax=None,
                 show_labels=True, node_colors=True, cmap_center=0):
    """Draw a pathway map.

    Args:
        pathway_map: PathwayMap instance.
        title: plot title.
        figsize: figure size.
        ax: optional axes.
        show_labels: show node names.
        node_colors: color nodes by expression.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Draw edges first
    for edge in pathway_map.edges:
        src = pathway_map.nodes.get(edge.source)
        tgt = pathway_map.nodes.get(edge.target)
        if not src or not tgt:
            continue

        style = '-' if edge.edge_type == 'activation' else '--'
        color = 'green' if edge.edge_type == 'activation' else 'red'
        if edge.edge_type == 'inhibition':
            color = 'red'

        ax.annotate('', xy=(tgt.x, tgt.y), xytext=(src.x, src.y),
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                     connectionstyle='arc3,rad=0.1', linestyle=style))

        if edge.label:
            mx = (src.x + tgt.x) / 2
            my = (src.y + tgt.y) / 2
            ax.text(mx, my + 0.2, edge.label, fontsize=7, ha='center', color='gray')

    # Draw nodes
    for nid, node in pathway_map.nodes.items():
        color = node.color if (node_colors and node.color) else '#4ecdc4'
        rect = FancyBboxPatch((node.x - node.width / 2, node.y - node.height / 2),
                               node.width, node.height,
                               boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        if show_labels:
            label = node.name[:15] if len(node.name) > 15 else node.name
            ax.text(node.x, node.y, label, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='black')

        if node.expression is not None:
            ax.text(node.x, node.y - node.height / 2 - 0.15,
                    f'{node.expression:.1f}', ha='center', va='top', fontsize=6,
                    color='gray')

    # Auto-scale
    if pathway_map.nodes:
        xs = [n.x for n in pathway_map.nodes.values()]
        ys = [n.y for n in pathway_map.nodes.values()]
        ax.set_xlim(min(xs) - 2, max(xs) + 2)
        ax.set_ylim(min(ys) - 2, max(ys) + 2)

    ax.set_aspect('equal')
    ax.set_title(title or pathway_map.title, fontsize=14)
    ax.axis('off')
    return fig


def create_kegg_style_pathway():
    """Create a sample KEGG-style MAPK signaling pathway."""
    pm = PathwayMap("MAPK Signaling Pathway")
    pm.add_node("EGF", "EGF", 0, 0)
    pm.add_node("EGFR", "EGFR", 0, -2)
    pm.add_node("GRB2", "GRB2", 0, -4)
    pm.add_node("SOS", "SOS", 0, -6)
    pm.add_node("RAS", "RAS", 0, -8)
    pm.add_node("RAF", "RAF", 0, -10)
    pm.add_node("MEK", "MEK", 0, -12)
    pm.add_node("ERK", "ERK", 0, -14)
    pm.add_node("MYC", "c-Myc", 3, -14)
    pm.add_node("FOS", "c-Fos", 3, -12)

    pm.add_edge("EGF", "EGFR", "activation")
    pm.add_edge("EGFR", "GRB2", "activation")
    pm.add_edge("GRB2", "SOS", "activation")
    pm.add_edge("SOS", "RAS", "activation")
    pm.add_edge("RAS", "RAF", "activation")
    pm.add_edge("RAF", "MEK", "activation")
    pm.add_edge("MEK", "ERK", "activation")
    pm.add_edge("ERK", "MYC", "activation")
    pm.add_edge("ERK", "FOS", "activation")

    return pm


def create_custom_pathway(gene_names, connections=None):
    """Create a custom pathway from gene names.

    Args:
        gene_names: list of gene names.
        connections: list of (source_idx, target_idx) tuples, or None for linear.

    Returns:
        PathwayMap instance.
    """
    pm = PathwayMap("Custom Pathway")
    for i, gene in enumerate(gene_names):
        pm.add_node(f"g{i}", gene, x=i * 3, y=0)

    if connections:
        for src, tgt in connections:
            pm.add_edge(f"g{src}", f"g{tgt}")
    else:
        for i in range(len(gene_names) - 1):
            pm.add_edge(f"g{i}", f"g{i+1}")

    return pm


def format_pathway_report(pathway_map):
    """Format pathway info as text."""
    lines = [f"Pathway: {pathway_map.name}",
             f"Nodes: {len(pathway_map.nodes)}",
             f"Edges: {len(pathway_map.edges)}", ""]
    for nid, node in pathway_map.nodes.items():
        expr = f" (expr={node.expression:.2f})" if node.expression is not None else ""
        lines.append(f"  {node.name}{expr}")
    lines.append("\nConnections:")
    for edge in pathway_map.edges:
        src = pathway_map.nodes.get(edge.source)
        tgt = pathway_map.nodes.get(edge.target)
        if src and tgt:
            lines.append(f"  {src.name} --{edge.edge_type}--> {tgt.name}")
    return "\n".join(lines)
