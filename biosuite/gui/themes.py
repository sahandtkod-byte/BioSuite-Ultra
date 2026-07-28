"""
Theme definitions, plot categories, and font constants for BioSuite GUI.
"""

THEMES = {
    'dark-green': {
        'name': 'Dark-Green-Cyber', 'ctk_mode': 'dark',
        'bg': '#0a0f0a', 'bg_secondary': '#0d170d', 'card': '#111c11',
        'card_hover': '#162216', 'accent': '#00ff88', 'accent_dim': '#00cc6a',
        'accent_glow': '#00cc6a', 'text': '#e0ffe8', 'text_dim': '#6b9b7a',
        'text_muted': '#3d6b4a', 'border': '#1a3a1a', 'border_light': '#2a5a2a',
        'sidebar_bg': '#060d06', 'sidebar_text': '#a0d0a0', 'sidebar_hover': '#0f1f0f',
        'sidebar_active': '#00ff88', 'sidebar_active_text': '#000000',
        'danger': '#ff4444', 'success': '#00ff88', 'input_bg': '#0a150a',
        'scrollbar': '#1a3a1a', 'header_accent': '#00ff88',
        'overlay': '#000000', 'dialog_bg': '#0d1a0d', 'dialog_border': '#00cc6a',
    },
    'dark-purple': {
        'name': 'Dark-Purple-Cyber', 'ctk_mode': 'dark',
        'bg': '#0a0a12', 'bg_secondary': '#0f0f1a', 'card': '#13132a',
        'card_hover': '#1a1a35', 'accent': '#b44aff', 'accent_dim': '#9933e6',
        'accent_glow': '#9933e6', 'text': '#e8e0ff', 'text_dim': '#8a7aaa',
        'text_muted': '#5a4a7a', 'border': '#2a1a3a', 'border_light': '#3a2a5a',
        'sidebar_bg': '#08080f', 'sidebar_text': '#b0a0d0', 'sidebar_hover': '#150f22',
        'sidebar_active': '#b44aff', 'sidebar_active_text': '#ffffff',
        'danger': '#ff4466', 'success': '#44ffaa', 'input_bg': '#0a0a18',
        'scrollbar': '#2a1a3a', 'header_accent': '#d080ff',
        'overlay': '#000000', 'dialog_bg': '#120f1f', 'dialog_border': '#9933e6',
    },
    'light-blue': {
        'name': 'Light-Blue-Cyber', 'ctk_mode': 'light',
        'bg': '#f0f4fa', 'bg_secondary': '#e8eef8', 'card': '#ffffff',
        'card_hover': '#f5f8ff', 'accent': '#2563eb', 'accent_dim': '#1d4ed8',
        'accent_glow': '#1d4ed8', 'text': '#0f172a', 'text_dim': '#64748b',
        'text_muted': '#94a3b8', 'border': '#e2e8f0', 'border_light': '#cbd5e1',
        'sidebar_bg': '#0f172a', 'sidebar_text': '#94a3b8', 'sidebar_hover': '#1e293b',
        'sidebar_active': '#3b82f6', 'sidebar_active_text': '#ffffff',
        'danger': '#dc2626', 'success': '#16a34a', 'input_bg': '#f8fafc',
        'scrollbar': '#cbd5e1', 'header_accent': '#2563eb',
        'overlay': '#0f172a', 'dialog_bg': '#ffffff', 'dialog_border': '#2563eb',
    },
}

PLOT_CATEGORIES = {
    'Advanced Biological': [
        ('Volcano Plot', 'volcano'), ('PCA Plot', 'pca'),
        ('Manhattan Plot', 'manhattan'), ('MA Plot', 'ma'), ('Venn Diagram', 'venn'),
    ],
    'Basic Biological': [
        ('Barplot', 'barplot'), ('Boxplot', 'boxplot'), ('Heatmap', 'heatmap'),
        ('Scatter Plot', 'scatter'), ('Time Series', 'timeseries'),
    ],
    'Mathematical': [
        ('Sine', 'sine'), ('Cosine', 'cosine'), ('Linear', 'linear'),
        ('Quadratic', 'quadratic'), ('Cubic', 'cubic'), ('Exponential', 'exponential'),
        ('Logistic', 'logistic'),
    ],
    'Specialized': [
        ('GSEA Plot', 'gsea'), ('Motif Logo', 'motif'), ('Sankey Diagram', 'sankey'),
    ],
    'Additional': [
        ('QQ-plot', 'qq'), ('Clustered Heatmap', 'clustered_heatmap'),
        ('Circos Plot', 'circos'), ('Alignment Viewer', 'alignment'), ('UMAP Plot', 'umap'),
    ],
    'New Plots': [
        ('Violin Plot', 'violin'), ('Raincloud Plot', 'raincloud'),
        ('Ridge Plot', 'ridge'), ('Dot Plot', 'dotplot'),
    ],
    'Genomics': [
        ('UpSet Plot', 'upset'), ('Genome Browser', 'genome_browser'), ('Synteny Dotplot', 'synteny'),
    ],
    'Sequence': [
        ('Sequence Logo', 'seq_logo'), ('Conservation Bar', 'conservation_bar'),
    ],
    'Interactive': [
        ('Interactive Scatter', 'interactive_scatter'), ('Interactive Bar', 'interactive_bar'),
        ('Interactive Heatmap', 'interactive_heatmap'), ('Interactive Volcano', 'interactive_volcano'),
        ('Interactive Line', 'interactive_line'), ('Interactive Pie', 'interactive_pie'),
    ],
}


def _build_plot_funcs():
    """Build plot_id -> callable mapping. Called once after heavy imports available."""
    funcs = {}

    def _noop():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Under construction', ha='center', va='center', fontsize=16, color='gray')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # --- plot_api functions (all take interactive=False by default) ---
    try:
        from biosuite.plotting import plot_api
        funcs['volcano'] = lambda: plot_api.volcano(
            __import__('numpy').random.randn(200),
            __import__('numpy').uniform(0.001, 1, 200))
        funcs['pca'] = lambda: plot_api.pca(__import__('numpy').random.randn(30, 10))
        funcs['manhattan'] = lambda: _manhattan_demo()
        funcs['ma'] = lambda: plot_api.ma(
            __import__('numpy').random.uniform(2, 12, 300),
            __import__('numpy').random.randn(300))
        funcs['barplot'] = lambda: plot_api.barplot(
            ['Gene A', 'Gene B', 'Gene C', 'Gene D', 'Gene E'],
            [23, 45, 12, 67, 34])
        funcs['boxplot'] = lambda: plot_api.boxplot({
            'Control': __import__('numpy').random.randn(30).tolist(),
            'Treatment': (__import__('numpy').random.randn(30) + 1).tolist(),
            'Recovery': (__import__('numpy').random.randn(30) + 0.5).tolist()})
        funcs['heatmap'] = lambda: plot_api.heatmap(__import__('numpy').random.rand(10, 8))
        funcs['scatter'] = lambda: _scatter_demo()
        funcs['violin'] = lambda: plot_api.violin({
            'Group A': __import__('numpy').random.randn(40).tolist(),
            'Group B': (__import__('numpy').random.randn(40) + 1.5).tolist()})
    except Exception:
        pass

    # --- biological_plots functions ---
    try:
        from biosuite.plotting import biological_plots as bp
        funcs['venn'] = lambda: bp.draw_venn2([10, 15, 5], ('Set A', 'Set B'))
        funcs['timeseries'] = lambda: _timeseries_demo()
        funcs['qq'] = lambda: _qq_demo()
        funcs['gsea'] = lambda: _gsea_demo()
        funcs['motif'] = lambda: bp.draw_motif_logo(
            ['ACGTACGT', 'ACGACGT', 'ACGTTCGT', 'ACGAACGT', 'ACGTCCGT'])
        funcs['circos'] = lambda: _noop()
        funcs['ridge'] = lambda: _noop()
        funcs['raincloud'] = lambda: _noop()
        funcs['dotplot'] = lambda: _noop()
        funcs['seq_logo'] = lambda: bp.draw_motif_logo(
            ['ACGTACGT', 'ACGACGT', 'ACGTTCGT', 'ACGAACGT'])
        funcs['conservation_bar'] = lambda: _conservation_demo()
    except Exception:
        pass

    # --- specialized_plots ---
    try:
        from biosuite.plotting import specialized_plots as sp
        funcs['sankey'] = lambda: sp.sankey_diagram(
            {'Source A': 100, 'Source B': 200},
            {'Target X': 150, 'Target Y': 150})
    except Exception:
        pass

    # --- math_plots ---
    try:
        from biosuite.plotting import math_plots as mp
        funcs['sine'] = lambda: mp.sine_plot()
        funcs['cosine'] = lambda: mp.cosine_plot()
        funcs['linear'] = lambda: mp.linear_plot()
        funcs['quadratic'] = lambda: mp.quadratic_plot()
        funcs['cubic'] = lambda: mp.cubic_plot()
        funcs['exponential'] = lambda: mp.exponential_plot()
        funcs['logistic'] = lambda: mp.logistic_plot()
    except Exception:
        pass

    # --- network_plots ---
    try:
        from biosuite.plotting import network_plots as np_mod
        funcs['clustered_heatmap'] = lambda: _clustered_heatmap_demo()
    except Exception:
        pass

    # --- upset_plots ---
    try:
        from biosuite.plotting import upset_plots as up
        funcs['upset'] = lambda: up.plot_upset(up.compute_upset_matrix([
            {'A', 'B', 'C'}, {'B', 'C'}, {'A', 'C'}, {'A', 'B'}, {'B'}]))
    except Exception:
        pass

    # --- genome_browser ---
    try:
        from biosuite.plotting import genome_browser as gb
        funcs['genome_browser'] = lambda: gb.plot_genome_tracks([])
    except Exception:
        pass

    # --- synteny ---
    try:
        from biosuite.plotting import synteny as syn
        funcs['synteny'] = lambda: syn.plot_synteny_dotplot(
            __import__('numpy').random.rand(10, 10))
    except Exception:
        pass

    # --- interactive_plots ---
    try:
        from biosuite.plotting import interactive_plots as ip
        funcs['interactive_scatter'] = lambda: ip.interactive_scatter(
            __import__('numpy').random.randn(100),
            __import__('numpy').random.randn(100))
        funcs['interactive_bar'] = lambda: ip.interactive_bar(
            ['A', 'B', 'C', 'D'], [25, 40, 30, 55])
        funcs['interactive_heatmap'] = lambda: ip.interactive_heatmap(
            __import__('numpy').random.rand(8, 8))
        funcs['interactive_volcano'] = lambda: ip.interactive_volcano(
            __import__('numpy').random.randn(200),
            __import__('numpy').uniform(0.001, 1, 200))
        funcs['interactive_line'] = lambda: _interactive_line_demo()
        funcs['interactive_pie'] = lambda: ip.interactive_pie(
            {'Category A': 35, 'Category B': 25, 'Category C': 20, 'Category D': 20})
    except Exception:
        pass

    # --- plasmid_map ---
    try:
        from biosuite.plotting import plasmid_map as pm
        funcs['plasmid'] = lambda: pm.create_sample_plasmid()
    except Exception:
        pass

    # --- sequence_viewer ---
    try:
        from biosuite.plotting import sequence_viewer as sv
        funcs['alignment_viewer'] = lambda: sv.draw_sequence_view(
            'ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG')
    except Exception:
        pass

    return funcs


def _manhattan_demo():
    import numpy as np
    from biosuite.plotting.plot_api import manhattan
    np.random.seed(42)
    chroms = []
    pos = []
    for c in range(1, 23):
        chroms.extend([f'chr{c}'] * 30)
        pos.extend(sorted(np.random.randint(1, 250_000_000, 30)))
    pvals = np.random.uniform(0.001, 1, len(chroms))
    pvals[:10] = np.random.uniform(1e-10, 1e-5, 10)
    return manhattan(chroms, pos, pvals)


def _scatter_demo():
    import numpy as np
    from biosuite.plotting.plot_api import scatter
    np.random.seed(42)
    x = np.random.randn(100)
    y = x * 2 + np.random.randn(100) * 0.5
    return scatter(x, y, show_regression=True)


def _timeseries_demo():
    import numpy as np
    import matplotlib.pyplot as plt
    from biosuite.plotting.style import apply_style, style_ax, get_colors, get_figsize
    apply_style()
    fig, ax = plt.subplots(figsize=get_figsize())
    t = np.linspace(0, 24, 200)
    colors = get_colors(3)
    ax.plot(t, 20 + 5*np.sin(t/4), label='Temperature', color=colors[0], linewidth=1.5)
    ax.plot(t, 60 + 10*np.sin(t/4 + 1), label='Humidity', color=colors[1], linewidth=1.5)
    ax.plot(t, 1000 + 200*np.sin(t/4 + 2), label='Pressure', color=colors[2], linewidth=1.5)
    style_ax(ax, title='Time Series', xlabel='Time (hours)', ylabel='Value')
    ax.legend()
    return fig


def _qq_demo():
    import numpy as np
    import matplotlib.pyplot as plt
    from biosuite.plotting.style import apply_style, style_ax, COLORS, get_figsize
    apply_style()
    np.random.seed(42)
    data = np.random.randn(100)
    sorted_data = np.sort(data)
    theoretical = np.sort(np.random.randn(100))
    fig, ax = plt.subplots(figsize=get_figsize())
    ax.scatter(theoretical, sorted_data, s=30, alpha=0.7, color=COLORS["primary"],
               edgecolors="white", linewidths=0.3)
    lims = [min(theoretical.min(), sorted_data.min()) - 0.5,
            max(theoretical.max(), sorted_data.max()) + 0.5]
    ax.plot(lims, lims, '--', color=COLORS["regression"], linewidth=1.5, label='y=x')
    style_ax(ax, title='Q-Q Plot', xlabel='Theoretical Quantiles', ylabel='Sample Quantiles')
    ax.legend()
    return fig


def _gsea_demo():
    import numpy as np
    import matplotlib.pyplot as plt
    from biosuite.plotting.style import apply_style, style_ax, COLORS, get_figsize
    apply_style()
    np.random.seed(42)
    n = 200
    es = np.cumsum(np.random.randn(n) * 0.05)
    fig, ax = plt.subplots(figsize=get_figsize())
    ax.plot(es, color=COLORS["primary"], linewidth=1.5)
    ax.fill_between(range(n), es, alpha=0.2, color=COLORS["primary"])
    ax.axhline(0, color='gray', linewidth=0.5)
    peak_idx = np.argmax(np.abs(es))
    ax.axvline(peak_idx, color=COLORS["significant"], linestyle='--', alpha=0.7)
    style_ax(ax, title='GSEA Enrichment Score', xlabel='Gene Rank', ylabel='Enrichment Score')
    return fig


def _conservation_demo():
    import numpy as np
    import matplotlib.pyplot as plt
    from biosuite.plotting.style import apply_style, style_ax, COLORS, get_figsize
    apply_style()
    np.random.seed(42)
    positions = np.arange(1, 51)
    scores = np.random.uniform(0.5, 1.0, 50)
    scores[10:15] = np.random.uniform(0.1, 0.3, 5)
    fig, ax = plt.subplots(figsize=get_figsize())
    ax.bar(positions, scores, color=[COLORS["primary"] if s > 0.5 else COLORS["significant"] for s in scores],
           width=0.8, edgecolor='white', linewidth=0.3)
    style_ax(ax, title='Conservation Score', xlabel='Position', ylabel='Score')
    ax.set_ylim(0, 1.1)
    return fig


def _clustered_heatmap_demo():
    import numpy as np
    import matplotlib.pyplot as plt
    from biosuite.plotting.style import apply_style, style_ax, get_figsize
    apply_style()
    np.random.seed(42)
    try:
        import seaborn as sns
        data = np.random.rand(15, 12)
        fig = sns.clustermap(data, cmap='viridis', figsize=get_figsize(),
                            linewidths=0.5, linecolor='white',
                            dendrogram_ratio=0.15, cbar_pos=(0.02, 0.8, 0.03, 0.15))
        fig.fig.suptitle('Clustered Heatmap', y=1.02, fontsize=14, fontweight='bold')
        return fig.fig
    except ImportError:
        fig, ax = plt.subplots(figsize=get_figsize())
        ax.imshow(np.random.rand(15, 12), cmap='viridis', aspect='auto')
        style_ax(ax, title='Clustered Heatmap')
        return fig


def _interactive_line_demo():
    import numpy as np
    from biosuite.plotting.interactive_plots import interactive_line
    t = np.linspace(0, 10, 100)
    return interactive_line(
        {'sin': np.sin(t).tolist(), 'cos': np.cos(t).tolist()},
        x=t.tolist(), title='Interactive Line Plot')


# Build at import time
PLOT_FUNCS = _build_plot_funcs()


# ─── Font Constants ───────────────────────────────────────────────────────────

FONT_FAMILY = 'Segoe UI'
FONT_MONO = 'Consolas'
FONT_TITLE = (FONT_FAMILY, 22, 'bold')
FONT_HEADING = (FONT_FAMILY, 16, 'bold')
FONT_SUBHEADING = (FONT_FAMILY, 13, 'bold')
FONT_BODY = (FONT_FAMILY, 12)
FONT_SMALL = (FONT_FAMILY, 10)
FONT_SIDEBAR = (FONT_FAMILY, 12)
FONT_CODE = (FONT_MONO, 11)
FONT_BUTTON = (FONT_FAMILY, 12, 'bold')
