"""Smoke tests for pathway_viz.py (pure-matplotlib module)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from biosuite.core.pathway_viz import (
    PathwayMap, draw_pathway, create_kegg_style_pathway,
    create_custom_pathway, format_pathway_report,
)


def test_expression_color_mapping():
    pm = PathwayMap('t')
    pm.add_node('a', 'A', 0, 0)
    pm.add_node('b', 'B', 3, 0)
    pm.set_expression({'a': -2.0, 'b': 2.0})
    assert pm.nodes['a'].color.startswith('#') and len(pm.nodes['a'].color) == 7
    assert pm.nodes['a'].color != pm.nodes['b'].color  # opposite extremes


def test_draw_kegg_sample_no_crash():
    pm = create_kegg_style_pathway()
    fig = draw_pathway(pm, title='x')
    assert fig is not None
    plt.close(fig)


def test_layout_grid_and_linear():
    pm = create_custom_pathway([f'G{i}' for i in range(7)])
    pm.layout_grid(n_cols=3)
    xs = [n.x for n in pm.nodes.values()]
    ys = [n.y for n in pm.nodes.values()]
    assert len(set(xs)) == 3 and min(ys) == -4  # 7 nodes, 3 cols -> rows 0,1,2
    pm.layout_linear(spacing=2)
    assert [n.x for n in pm.nodes.values()] == [0, 2, 4, 6, 8, 10, 12]


def test_report_and_edges_with_missing_node():
    pm = PathwayMap('r')
    pm.add_node('a', 'A')
    pm.add_edge('a', 'ghost', 'activation')  # unresolvable target
    txt = format_pathway_report(pm)
    assert 'Nodes: 1' in txt
    fig = draw_pathway(pm)
    plt.close(fig)


def test_expression_extremes_valid_hex():
    from biosuite.core.pathway_viz import PathwayNode
    for v in (-10, -1, 0, 1, 10):
        n = PathwayNode('x', 'x')
        n.set_expression(v, vmin=-2, vmax=2)
        c = n.color
        int(c[1:], 16)  # parses -> valid hex color
