"""
Pathway visualization — draw KEGG/Reactome-style pathway maps
with gene expression overlays. Pure Python/matplotlib.
"""
from collections import OrderedDict

import numpy as np

from .log import get_logger

logger = get_logger(__name__)

# down-regulated / unchanged / up-regulated.  The scale is interpolated between
# these anchors so that two adjacent values never land on opposite sides of a
# visible band edge, and so that "no change" is a neutral grey rather than a
# saturated hue a reader would take for mild up-regulation.
EXPRESSION_ANCHORS = ((30, 80, 220), (235, 235, 235), (230, 30, 20))
DEFAULT_NODE_COLOR = '#4ecdc4'


def _scale_value(value, label):
    """A finite number for the expression scale, or a named refusal."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{label} must be a number, got {type(value).__name__}: {value!r}")
    if not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return float(value)


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
        """Record an expression value and the colour the scale gives it.

        The bounds have to be two different finite numbers: equal bounds divided
        by zero, and reversed ones painted the opposite of what the values mean.
        """
        low = _scale_value(vmin, "vmin")
        high = _scale_value(vmax, "vmax")
        if high <= low:
            raise ValueError(f"the colour scale needs vmax above vmin, got vmin={low!r}, "
                             f"vmax={high!r}")
        number = _scale_value(value, "expression value")
        self.expression = number
        norm = min(1.0, max(0.0, (number - low) / (high - low)))
        # Piecewise-linear blue -> grey -> red, continuous across the midpoint.
        half = len(EXPRESSION_ANCHORS) - 1
        segment = min(int(norm * half), half - 1)
        local = norm * half - segment
        start = np.array(EXPRESSION_ANCHORS[segment], dtype=float)
        end = np.array(EXPRESSION_ANCHORS[segment + 1], dtype=float)
        r, g, b = (start + (end - start) * local).round().astype(int)
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

    def unresolvable_edges(self):
        """Edges naming a node that is not in this map.

        They cannot be drawn, and they are not counted as drawn: a report that
        said 'Edges: 3' above a list of one connection left the reader to discover
        the difference by counting.
        """
        return [edge for edge in self.edges
                if edge.source not in self.nodes or edge.target not in self.nodes]

    def set_expression(self, expr_dict, vmin=-2, vmax=2):
        for node_id, value in expr_dict.items():
            if node_id in self.nodes:
                self.nodes[node_id].set_expression(value, vmin, vmax)

    def layout_grid(self, n_cols=3, spacing_x=3, spacing_y=2):
        """Arrange nodes in a grid.

        ``n_cols`` is a whole number of at least 1: zero divided by zero, and a
        fraction stacked every node into one column while reporting success.
        """
        if isinstance(n_cols, bool) or not isinstance(n_cols, int) or n_cols < 1:
            raise ValueError(f"n_cols must be a whole number of at least 1, got {n_cols!r}")
        for i, (_nid, node) in enumerate(self.nodes.items()):
            row = i // n_cols
            col = i % n_cols
            node.x = col * spacing_x
            node.y = -row * spacing_y

    def layout_linear(self, spacing=3):
        """Arrange nodes in a line."""
        for i, (_nid, node) in enumerate(self.nodes.items()):
            node.x = i * spacing
            node.y = 0


def draw_pathway(pathway_map, title=None, figsize=(12, 8), ax=None,
                 show_labels=True, node_colors=True):
    """Draw a pathway map.

    Args:
        pathway_map: PathwayMap instance.
        title: plot title.
        figsize: figure size.
        ax: optional axes.
        show_labels: show node names.
        node_colors: color nodes by expression.
    """
    import matplotlib.pyplot as plt

    from matplotlib.patches import FancyBboxPatch

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    skipped = pathway_map.unresolvable_edges()
    if skipped:
        logger.warning("%s: %d edge(s) name a node that is not in the map and were "
                       "not drawn: %s", pathway_map.name, len(skipped),
                       ", ".join(f"{e.source}->{e.target}" for e in skipped[:5]))

    # Draw edges first
    for edge in pathway_map.edges:
        src = pathway_map.nodes.get(edge.source)
        tgt = pathway_map.nodes.get(edge.target)
        if not src or not tgt:
            continue

        style = '-' if edge.edge_type == 'activation' else '--'
        color = 'green' if edge.edge_type == 'activation' else 'red'

        ax.annotate('', xy=(tgt.x, tgt.y), xytext=(src.x, src.y),
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                     connectionstyle='arc3,rad=0.1', linestyle=style))

        if edge.label:
            mx = (src.x + tgt.x) / 2
            my = (src.y + tgt.y) / 2
            ax.text(mx, my + 0.2, edge.label, fontsize=7, ha='center', color='gray')

    # Draw nodes
    for _nid, node in pathway_map.nodes.items():
        color = node.color if (node_colors and node.color) else DEFAULT_NODE_COLOR
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

    Raises:
        ValueError: when a connection indexes outside the gene list.  An index that
        names no gene used to become an edge between two node ids that do not exist,
        which the drawing silently left out and the report still counted as an edge.
    """
    pm = PathwayMap("Custom Pathway")
    names = list(gene_names)
    for i, gene in enumerate(names):
        pm.add_node(f"g{i}", gene, x=i * 3, y=0)

    if connections:
        for src, tgt in connections:
            for index, label in ((src, "source"), (tgt, "target")):
                if not isinstance(index, int) or isinstance(index, bool) \
                        or not 0 <= index < len(names):
                    raise ValueError(
                        f"connection ({src}, {tgt}) has a {label} index of {index!r}, "
                        f"which is not a gene position in a list of {len(names)} names")
            pm.add_edge(f"g{src}", f"g{tgt}")
    else:
        for i in range(len(names) - 1):
            pm.add_edge(f"g{i}", f"g{i+1}")

    return pm


def format_pathway_report(pathway_map):
    """Format pathway info as text.

    Edges that name a node the map does not hold are counted apart, because
    `draw_pathway` leaves them out: an 'Edges' line that includes them would be a
    number the picture does not show.
    """
    skipped = pathway_map.unresolvable_edges()
    edge_line = f"Edges: {len(pathway_map.edges)}"
    if skipped:
        edge_line += f" ({len(skipped)} not drawn: their endpoints are not in the map)"
    lines = [f"Pathway: {pathway_map.name}",
             f"Nodes: {len(pathway_map.nodes)}",
             edge_line, ""]
    for _nid, node in pathway_map.nodes.items():
        expr = f" (expr={node.expression:.2f})" if node.expression is not None else ""
        lines.append(f"  {node.name}{expr}")
    lines.append("\nConnections:")
    drawn = 0
    for edge in pathway_map.edges:
        src = pathway_map.nodes.get(edge.source)
        tgt = pathway_map.nodes.get(edge.target)
        if src and tgt:
            lines.append(f"  {src.name} --{edge.edge_type}--> {tgt.name}")
            drawn += 1
    if not drawn:
        lines.append("  (none: no edge has both of its nodes in this map)")
    return "\n".join(lines)
