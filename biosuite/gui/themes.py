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
        ('Quadratic', 'quadratic'), ('Cubic', 'cubic'),
        ('Exponential', 'exponential'), ('Logistic', 'logistic'),
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
        ('UpSet Plot', 'upset'), ('Genome Browser', 'genome_browser'),
        ('Synteny Dotplot', 'synteny'),
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

PLOT_FUNCS = {}  # Populated by _build_plot_funcs after heavy imports load

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
