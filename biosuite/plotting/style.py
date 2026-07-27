"""
BioSuite Ultra — Centralized Plot Style System.

Professional, publication-ready styling for all matplotlib and plotly plots.
Consistent colors, fonts, sizes, and grid across all plot types.

Usage:
    from biosuite.plotting.style import apply_style, COLORS, SIZES
    apply_style()  # Apply global style
"""
import os

# ── Color Palette (publication-quality, colorblind-friendly) ──────────

COLORS = {
    # Primary palette (Tableau-10 inspired, colorblind-safe)
    "primary":     "#4C72B0",  # Steel blue
    "secondary":   "#DD8452",  # Sandy orange
    "tertiary":    "#55A868",  # Muted green
    "quaternary":  "#C44E52",  # Muted red
    "fifth":       "#8172B3",  # Muted purple
    "sixth":       "#937860",  # Tan
    "seventh":     "#DA8BC3",  # Pink
    "eighth":      "#8C8C8C",  # Gray
    # Semantic colors
    "significant":  "#C44E52",  # Red for significant points
    "not_sig":      "#8C8C8C",  # Gray for non-significant
    "up":           "#C44E52",  # Red for upregulated
    "down":         "#4C72B0",  # Blue for downregulated
    "neutral":      "#8C8C8C",  # Gray for neutral
    "threshold":    "#8C8C8C",  # Gray for threshold lines
    "regression":   "#C44E52",  # Red for regression lines
    "highlight":    "#DD8452",  # Orange for highlights
}

# Color cycle for multi-series plots
COLOR_CYCLE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
               "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

# ── Font Configuration ───────────────────────────────────────────────

FONTS = {
    "family":       "DejaVu Sans",   # Cross-platform, always available
    "title_size":   14,
    "subtitle_size": 12,
    "label_size":   11,
    "tick_size":    10,
    "legend_size":  10,
    "annotation_size": 9,
}

# ── Size Configuration ───────────────────────────────────────────────

SIZES = {
    "fig_width":    8.0,
    "fig_height":   6.0,
    "dpi":          150,
    "scatter_s":    40,       # Default scatter point size
    "scatter_alpha": 0.7,
    "line_width":   1.5,
    "grid_alpha":   0.3,
    "threshold_alpha": 0.6,
    "marker_edge":  0.5,      # Edge width for scatter markers
}

# ── Matplotlib rcParams ──────────────────────────────────────────────

MPL_RC_PARAMS = {
    # Figure
    "figure.figsize":       (SIZES["fig_width"], SIZES["fig_height"]),
    "figure.dpi":           SIZES["dpi"],
    "figure.facecolor":     "white",
    "figure.edgecolor":     "#E0E0E0",
    "figure.titlesize":     FONTS["title_size"],
    "figure.titleweight":   "bold",
    # Axes
    "axes.facecolor":       "#FAFAFA",
    "axes.edgecolor":       "#CCCCCC",
    "axes.linewidth":       0.8,
    "axes.titlesize":       FONTS["title_size"],
    "axes.titleweight":     "bold",
    "axes.titlepad":        10,
    "axes.labelsize":       FONTS["label_size"],
    "axes.labelweight":     "normal",
    "axes.labelpad":        8,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    # Grid
    "axes.grid":            True,
    "axes.grid.axis":       "y",
    "grid.alpha":           SIZES["grid_alpha"],
    "grid.color":           "#E0E0E0",
    "grid.linewidth":       0.5,
    "grid.linestyle":       "--",
    # Ticks
    "xtick.labelsize":      FONTS["tick_size"],
    "ytick.labelsize":      FONTS["tick_size"],
    "xtick.color":          "#555555",
    "ytick.color":          "#555555",
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.size":     4,
    "ytick.major.size":     4,
    # Legend
    "legend.fontsize":      FONTS["legend_size"],
    "legend.frameon":       True,
    "legend.framealpha":    0.9,
    "legend.edgecolor":     "#CCCCCC",
    "legend.fancybox":      True,
    "legend.shadow":        False,
    # Font
    "font.family":          "sans-serif",
    "font.sans-serif":      [FONTS["family"], "Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            FONTS["label_size"],
    # Lines
    "lines.linewidth":      SIZES["line_width"],
    "lines.markersize":     6,
    # Savefig
    "savefig.dpi":          SIZES["dpi"],
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",
    "savefig.edgecolor":    "white",
    "savefig.pad_inches":   0.1,
    # Patches (bars, boxes)
    "patch.linewidth":      0.5,
    "patch.edgecolor":      "#CCCCCC",
}


# ── Plotly Configuration ─────────────────────────────────────────────

PLOTLY_LAYOUT = {
    "template": "plotly_white",
    "font": {"family": FONTS["family"], "size": 13, "color": "#333333"},
    "title": {"font": {"size": 16, "color": "#222222"}, "x": 0.5, "xanchor": "center"},
    "xaxis": {"gridcolor": "#E8E8E8", "zerolinecolor": "#CCCCCC", "showgrid": True},
    "yaxis": {"gridcolor": "#E8E8E8", "zerolinecolor": "#CCCCCC", "showgrid": True},
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
}


# ── Apply Functions ──────────────────────────────────────────────────

def apply_style(style="default"):
    """Apply BioSuite plot style to matplotlib.

    Args:
        style: 'default' (white background), 'dark' (dark theme),
               or 'publication' (minimal, print-ready).
    """
    import matplotlib.pyplot as plt

    if style == "dark":
        dark_params = {
            "figure.facecolor":  "#1E1E1E",
            "axes.facecolor":    "#2D2D2D",
            "axes.edgecolor":    "#555555",
            "axes.labelcolor":   "#E0E0E0",
            "axes.grid.color":   "#444444",
            "text.color":        "#E0E0E0",
            "xtick.color":       "#AAAAAA",
            "ytick.color":       "#AAAAAA",
            "savefig.facecolor": "#1E1E1E",
            "grid.alpha":        0.2,
        }
        params = {**MPL_RC_PARAMS, **dark_params}
    elif style == "publication":
        params = {
            **MPL_RC_PARAMS,
            "axes.grid":       False,
            "axes.linewidth":  1.0,
            "axes.titlesize":  12,
            "axes.labelsize":  10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.size":       10,
            "figure.dpi":      300,
            "savefig.dpi":     300,
        }
    else:
        params = MPL_RC_PARAMS

    plt.rcParams.update(params)


def get_colors(n=None):
    """Get color palette.

    Args:
        n: Number of colors needed. If None, returns full cycle.

    Returns:
        List of color hex strings.
    """
    if n is None:
        return COLOR_CYCLE[:]
    # Repeat cycle if more colors needed
    return [COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in range(n)]


def get_figsize(width=None, height=None):
    """Get standard figure size."""
    return (width or SIZES["fig_width"], height or SIZES["fig_height"])


def get_dpi():
    """Get DPI from config or default."""
    try:
        from biosuite.core.utils import config
        return config.get("default_dpi", SIZES["dpi"])
    except Exception:
        return SIZES["dpi"]


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to an axes object.

    Args:
        ax: matplotlib Axes object.
        title: plot title.
        xlabel: x-axis label.
        ylabel: y-axis label.
    """
    if title:
        ax.set_title(title, fontsize=FONTS["title_size"], fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONTS["label_size"], labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONTS["label_size"], labelpad=8)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    # Grid
    ax.grid(True, axis="y", alpha=SIZES["grid_alpha"], linestyle="--", color="#E0E0E0")
    ax.set_axisbelow(True)

    return ax


def style_legend(ax, **kwargs):
    """Apply consistent legend styling."""
    defaults = {
        "frameon": True,
        "framealpha": 0.9,
        "edgecolor": "#CCCCCC",
        "fancybox": True,
        "shadow": False,
        "fontsize": FONTS["legend_size"],
    }
    defaults.update(kwargs)
    ax.legend(**defaults)
    return ax


# ── Backward Compatibility ───────────────────────────────────────────

# Old code that imports MPL_STYLE will still work
MPL_STYLE = {k: v for k, v in MPL_RC_PARAMS.items()
             if k in ("figure.facecolor", "axes.facecolor", "text.color",
                      "axes.labelcolor", "xtick.color", "ytick.color")}

__all__ = [
    "COLORS", "COLOR_CYCLE", "FONTS", "SIZES",
    "MPL_RC_PARAMS", "MPL_STYLE", "PLOTLY_LAYOUT",
    "apply_style", "get_colors", "get_figsize", "get_dpi",
    "style_ax", "style_legend",
]
