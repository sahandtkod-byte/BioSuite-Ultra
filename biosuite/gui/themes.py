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
    """Build plot_id -> callable mapping. Each returns a matplotlib Figure."""
    import builtins
    import numpy as np
    funcs = {}

    def _get_fig(result):
        """Extract matplotlib Figure from various return types."""
        if result is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14, color='gray')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
            return fig
        import matplotlib.figure
        if isinstance(result, matplotlib.figure.Figure):
            return result
        import matplotlib.axes
        if isinstance(result, matplotlib.axes.Axes):
            return result.get_figure()
        # Plotly figure — render to matplotlib
        if hasattr(result, 'write_image'):
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.text(0.5, 0.5, 'Interactive plot (use in browser)',
                        ha='center', va='center', fontsize=12, color='gray')
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
                return fig
            except Exception:
                pass
        return result

    def _safe(func):
        """Wrap function: monkey-patch input() to return defaults, return Figure."""
        def wrapper():
            _orig_input = builtins.input
            builtins.input = lambda *a, **k: 'n'  # default to "don't save"
            try:
                result = func()
                return _get_fig(result)
            finally:
                builtins.input = _orig_input
        return wrapper

    # === plot_api functions ===
    try:
        from biosuite.plotting import plot_api as api

        def _volcano():
            fc = np.random.randn(200)
            pv = np.random.uniform(0.001, 1, 200)
            pv[:15] = np.random.uniform(1e-8, 0.01, 15)
            return api.volcano(fc, pv)

        def _pca():
            return api.pca(np.random.randn(30, 10))

        def _ma():
            mean_expr = np.random.uniform(2, 12, 300)
            log_fc = np.random.randn(300)
            return api.ma(mean_expr, log_fc)

        def _barplot():
            return api.barplot(['Gene A', 'Gene B', 'Gene C', 'Gene D', 'Gene E'],
                              [23, 45, 12, 67, 34])

        def _boxplot():
            return api.boxplot({
                'Control': np.random.randn(30).tolist(),
                'Treatment': (np.random.randn(30) + 1).tolist(),
                'Recovery': (np.random.randn(30) + 0.5).tolist()})

        def _heatmap():
            return api.heatmap(np.random.rand(10, 8))

        def _scatter():
            x = np.random.randn(100)
            y = x * 2 + np.random.randn(100) * 0.5
            return api.scatter(x, y, show_regression=True)

        def _violin():
            return api.violin({
                'Group A': np.random.randn(40).tolist(),
                'Group B': (np.random.randn(40) + 1.5).tolist()})

        funcs['volcano'] = _safe(_volcano)
        funcs['pca'] = _safe(_pca)
        funcs['ma'] = _safe(_ma)
        funcs['barplot'] = _safe(_barplot)
        funcs['boxplot'] = _safe(_boxplot)
        funcs['heatmap'] = _safe(_heatmap)
        funcs['scatter'] = _safe(_scatter)
        funcs['violin'] = _safe(_violin)
    except Exception:
        pass

    # === biological_plots functions ===
    try:
        from biosuite.plotting import biological_plots as bp

        def _venn():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6))
            bp.draw_venn2([10, 15, 5], ('Set A', 'Set B'), ax=ax)
            return fig

        def _motif():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            bp.draw_motif_logo(['ACGTACGT', 'ACGACGT', 'ACGTTCGT', 'ACGAACGT', 'ACGTCCGT'], ax=ax)
            return fig

        def _seq_logo():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            bp.draw_motif_logo(['ACGTACGT', 'ACGACGT', 'ACGTTCGT', 'ACGAACGT'], ax=ax)
            return fig

        funcs['venn'] = _safe(_venn)
        funcs['motif'] = _safe(_motif)
        funcs['seq_logo'] = _safe(_seq_logo)
    except Exception:
        pass

    # === specialized_plots ===
    try:
        from biosuite.plotting import specialized_plots as sp

        def _sankey():
            return sp.sankey_diagram()

        funcs['sankey'] = _safe(_sankey)
    except Exception:
        pass

    # === math_plots (wrap to skip input()) ===
    try:
        from biosuite.plotting import math_plots as mp
        for name, func in [('sine', mp.sine_plot), ('cosine', mp.cosine_plot),
                           ('linear', mp.linear_plot), ('quadratic', mp.quadratic_plot),
                           ('cubic', mp.cubic_plot), ('exponential', mp.exponential_plot),
                           ('logistic', mp.logistic_plot)]:
            funcs[name] = _safe(func)
    except Exception:
        pass

    # === upset_plots ===
    try:
        from biosuite.plotting import upset_plots as up

        def _upset():
            sets_dict = {'A': {'g1','g2','g3'}, 'B': {'g2','g3','g4'},
                         'C': {'g1','g3','g5'}, 'D': {'g2','g4','g6'}}
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 7))
            up.plot_upset(sets_dict, ax=ax)
            return fig

        funcs['upset'] = _safe(_upset)
    except Exception:
        pass

    # === genome_browser ===
    try:
        from biosuite.plotting import genome_browser as gb

        def _genome_browser():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.text(0.5, 0.5, 'Genome Browser\n(Load a BED/VCF file to visualize)',
                    ha='center', va='center', fontsize=14, color='gray',
                    transform=ax.transAxes)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
            return fig

        funcs['genome_browser'] = _safe(_genome_browser)
    except Exception:
        pass

    # === synteny ===
    try:
        from biosuite.plotting import synteny as syn

        def _synteny():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            data = np.random.rand(10, 10)
            ax.imshow(data, cmap='coolwarm', aspect='auto')
            ax.set_title('Synteny Dotplot')
            ax.set_xlabel('Genome 1')
            ax.set_ylabel('Genome 2')
            return fig

        funcs['synteny'] = _safe(_synteny)
    except Exception:
        pass

    # === interactive_plots (return plotly → render placeholder) ===
    try:
        from biosuite.plotting import interactive_plots as ip

        def _interactive_scatter():
            x = np.random.randn(100)
            y = x * 2 + np.random.randn(100) * 0.5
            return ip.interactive_scatter(x, y)

        def _interactive_bar():
            return ip.interactive_bar(['A', 'B', 'C', 'D'], [25, 40, 30, 55])

        def _interactive_heatmap():
            return ip.interactive_heatmap(np.random.rand(8, 8))

        def _interactive_volcano():
            fc = np.random.randn(200)
            pv = np.random.uniform(0.001, 1, 200)
            pv[:15] = np.random.uniform(1e-8, 0.01, 15)
            return ip.interactive_volcano(fc, pv)

        def _interactive_line():
            t = list(np.linspace(0, 10, 100))
            ys = [list(np.sin(t)), list(np.cos(t))]
            return ip.interactive_line(t, ys, names=['sin', 'cos'], title='Interactive Line')

        def _interactive_pie():
            return ip.interactive_pie(['A', 'B', 'C', 'D'], [35, 25, 20, 20])

        funcs['interactive_scatter'] = _safe(_interactive_scatter)
        funcs['interactive_bar'] = _safe(_interactive_bar)
        funcs['interactive_heatmap'] = _safe(_interactive_heatmap)
        funcs['interactive_volcano'] = _safe(_interactive_volcano)
        funcs['interactive_line'] = _safe(_interactive_line)
        funcs['interactive_pie'] = _safe(_interactive_pie)
    except Exception:
        pass

    # === plasmid_map ===
    try:
        from biosuite.plotting import plasmid_map as pm

        def _plasmid():
            result = pm.create_sample_plasmid()
            if hasattr(result, 'fig'):
                return result.fig
            if hasattr(result, 'savefig'):
                return result
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'Plasmid Map', ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
            return fig

        funcs['plasmid'] = _safe(_plasmid)
    except Exception:
        pass

    # === sequence_viewer ===
    try:
        from biosuite.plotting import sequence_viewer as sv

        def _alignment_viewer():
            return sv.draw_sequence_view('ATCGATCGATCGATCGATCGATCGATCGATCGATCG')

        funcs['alignment_viewer'] = _safe(_alignment_viewer)
    except Exception:
        pass

    # === conservation ===
    try:
        from biosuite.plotting import conservation_plots as cp

        def _conservation_bar():
            sequences = ['ACGTACGT', 'ACGACGT', 'ACGTTCGT', 'ACGAACGT', 'ACGTCCGT']
            return cp.plot_conservation_bar(sequences)

        funcs['conservation_bar'] = _safe(_conservation_bar)
    except Exception:
        pass

    # === specialized no-ops ===
    def _noop():
        import matplotlib.pyplot as plt
        from biosuite.plotting.style import apply_style, COLORS, get_figsize
        apply_style()
        fig, ax = plt.subplots(figsize=get_figsize())
        ax.text(0.5, 0.5, 'Coming soon', ha='center', va='center',
                fontsize=16, color=COLORS["not_sig"], transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        return fig

    for name in ['circos', 'ridge', 'raincloud', 'dotplot', 'umap']:
        if name not in funcs:
            funcs[name] = _safe(_noop)

    # === demos for remaining ===
    def _timeseries_demo():
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

    def _manhattan_demo():
        chroms = []
        pos = []
        for c in range(1, 23):
            chroms.extend([f'chr{c}'] * 30)
            pos.extend(sorted(np.random.randint(1, 250_000_000, 30).tolist()))
        pvals = np.random.uniform(0.001, 1, len(chroms))
        pvals[:10] = np.random.uniform(1e-10, 1e-5, 10)
        from biosuite.plotting.plot_api import manhattan
        return manhattan(chroms, pos, pvals.tolist())

    def _qq_demo():
        import matplotlib.pyplot as plt
        from biosuite.plotting.style import apply_style, style_ax, COLORS, get_figsize
        apply_style()
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
        import matplotlib.pyplot as plt
        from biosuite.plotting.style import apply_style, style_ax, COLORS, get_figsize
        apply_style()
        n = 200
        es = np.cumsum(np.random.randn(n) * 0.05)
        fig, ax = plt.subplots(figsize=get_figsize())
        ax.plot(es, color=COLORS["primary"], linewidth=1.5)
        ax.fill_between(range(n), es, alpha=0.2, color=COLORS["primary"])
        ax.axhline(0, color='gray', linewidth=0.5)
        style_ax(ax, title='GSEA Enrichment Score', xlabel='Gene Rank', ylabel='Enrichment Score')
        return fig

    def _clustered_heatmap_demo():
        import matplotlib.pyplot as plt
        from biosuite.plotting.style import apply_style, style_ax, get_figsize
        apply_style()
        fig, ax = plt.subplots(figsize=get_figsize())
        ax.imshow(np.random.rand(15, 12), cmap='viridis', aspect='auto')
        style_ax(ax, title='Heatmap (clustered)')
        return fig

    funcs['timeseries'] = _safe(_timeseries_demo)
    funcs['manhattan'] = _safe(_manhattan_demo)
    funcs['qq'] = _safe(_qq_demo)
    funcs['gsea'] = _safe(_gsea_demo)
    funcs['clustered_heatmap'] = _safe(_clustered_heatmap_demo)

    return funcs


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
