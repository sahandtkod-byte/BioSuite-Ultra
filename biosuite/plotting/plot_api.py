"""
Programmatic plot API with interactive=True support.

Provides clean function interfaces for all plot types that can be called
from Python code, Jupyter notebooks, or the REST API. Each function returns
a matplotlib figure or Plotly figure when interactive=True.

Usage:
    from biosuite.plotting.plot_api import volcano, pca, manhattan

    # Static matplotlib (default)
    fig = volcano(log2fc=fc, pvalues=pvals)

    # Interactive Plotly
    fig = volcano(log2fc=fc, pvalues=pvals, interactive=True)
    fig.show()  # In Jupyter
    fig.write_html("volcano.html")  # Export
"""
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import matplotlib.pyplot as plt
import matplotlib

from biosuite.plotting.style import (
    COLORS, COLOR_CYCLE, FONTS, SIZES, PLOTLY_LAYOUT,
    apply_style, get_colors, get_figsize, get_dpi,
    style_ax, style_legend,
)


# ── Theme helpers ────────────────────────────────────────────────────────────

PLOTLY_TEMPLATE = 'plotly_white'

# Backward compat — old code importing MPL_STYLE still works
from biosuite.plotting.style import MPL_STYLE

def _apply_mpl_style():
    """Apply BioSuite plot style. Called at start of each plot function."""
    apply_style("default")


def _new_fig(ax_kwargs=None):
    """Create a new figure with standard styling."""
    apply_style("default")
    fig, ax = plt.subplots(figsize=get_figsize())
    style_ax(ax, **(ax_kwargs or {}))
    return fig, ax


# ── Volcano Plot ─────────────────────────────────────────────────────────────

def volcano(log2fc, pvalues, gene_names=None, fc_thresh=1.0, p_thresh=0.05,
            title="Volcano Plot", interactive=False, output_html=None):
    """Volcano plot showing differential expression.

    Args:
        log2fc: array of log2 fold changes.
        pvalues: array of p-values.
        gene_names: optional gene names for hover.
        fc_thresh: fold-change threshold.
        p_thresh: p-value threshold.
        title: plot title.
        interactive: if True, return Plotly figure.
        output_html: if set, save as HTML.

    Returns:
        matplotlib Figure or Plotly Figure.
    """
    log2fc = np.asarray(log2fc, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    neg_log10 = -np.log10(pvalues + 1e-300)
    sig = (np.abs(log2fc) >= fc_thresh) & (pvalues < p_thresh)

    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=log2fc[~sig], y=neg_log10[~sig], mode='markers',
            name='Not significant', marker=dict(size=5, color=COLORS["not_sig"], opacity=0.5)
        ))
        labels = []
        for i in range(len(log2fc)):
            if sig[i]:
                lbl = gene_names[i] if gene_names and i < len(gene_names) else f"Gene {i}"
                labels.append(f"{lbl}<br>FC={log2fc[i]:.2f}, p={pvalues[i]:.2e}")
            else:
                labels.append("")
        fig.add_trace(go.Scatter(
            x=log2fc[sig], y=neg_log10[sig], mode='markers',
            name='Significant', marker=dict(size=7, color=COLORS["significant"]),
            text=np.array(labels)[sig].tolist(), hoverinfo='text'
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title=title,
                          xaxis_title='Log2 Fold Change',
                          yaxis_title='-log10(p-value)')
        fig.add_hline(y=-np.log10(p_thresh), line_dash="dash", line_color=COLORS["threshold"], opacity=0.6)
        fig.add_vline(x=-fc_thresh, line_dash="dash", line_color=COLORS["threshold"], opacity=0.6)
        fig.add_vline(x=fc_thresh, line_dash="dash", line_color=COLORS["threshold"], opacity=0.6)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=get_figsize())
        ax.scatter(log2fc[~sig], neg_log10[~sig], s=SIZES["scatter_s"]*0.5, alpha=0.4,
                   label='Not sig', color=COLORS["not_sig"], edgecolors="white", linewidths=0.3)
        ax.scatter(log2fc[sig], neg_log10[sig], s=SIZES["scatter_s"], alpha=0.8,
                   label='Significant', color=COLORS["significant"], edgecolors="white", linewidths=0.3)
        ax.axhline(-np.log10(p_thresh), linestyle='--', color=COLORS["threshold"],
                   alpha=SIZES["threshold_alpha"], linewidth=1)
        ax.axvline(-fc_thresh, linestyle='--', color=COLORS["threshold"],
                   alpha=SIZES["threshold_alpha"], linewidth=1)
        ax.axvline(fc_thresh, linestyle='--', color=COLORS["threshold"],
                   alpha=SIZES["threshold_alpha"], linewidth=1)
        style_ax(ax, title=title, xlabel='Log2 Fold Change', ylabel='-log10(P-value)')
        style_legend(ax)
        if output_html:
            fig.savefig(output_html.replace('.html', '.png'), dpi=get_dpi(), bbox_inches='tight')
        return fig


# ── PCA Plot ─────────────────────────────────────────────────────────────────

def pca(data, labels=None, group_col=None, n_components=2,
        title="PCA Plot", interactive=False, output_html=None):
    """Principal Component Analysis scatter plot.

    Args:
        data: numpy array (n_samples x n_features) or DataFrame.
        labels: sample labels for hover.
        group_col: column name for coloring (if DataFrame).
        n_components: number of PCs.
        title: plot title.
        interactive: if True, return Plotly figure.
        output_html: if set, save as HTML.

    Returns:
        matplotlib Figure or Plotly Figure.
    """
    from sklearn.decomposition import PCA

    if isinstance(data, pd.DataFrame):
        if group_col and group_col in data.columns:
            groups = data[group_col].astype(str).values
        else:
            groups = None
        numeric_data = data.select_dtypes(include=[np.number]).values
    else:
        numeric_data = np.asarray(data, dtype=float)
        groups = labels

    pca_model = PCA(n_components=n_components)
    coords = pca_model.fit_transform(numeric_data)
    var_exp = pca_model.explained_variance_ratio_

    x_label = f"PC1 ({var_exp[0]*100:.1f}%)"
    y_label = f"PC2 ({var_exp[1]*100:.1f}%)" if n_components > 1 else "PC2"

    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        if groups is not None:
            colors = get_colors(len(np.unique(groups)))
            for i, group in enumerate(np.unique(groups)):
                mask = groups == group
                fig.add_trace(go.Scatter(
                    x=coords[mask, 0], y=coords[mask, 1] if n_components > 1 else [0]*sum(mask),
                    mode='markers', name=str(group), marker=dict(size=9, color=colors[i]),
                    text=labels[mask] if labels is not None else None,
                    hoverinfo='text+x+y'
                ))
        else:
            fig.add_trace(go.Scatter(
                x=coords[:, 0], y=coords[:, 1] if n_components > 1 else [0]*len(coords),
                mode='markers', marker=dict(size=9, color=COLORS["primary"])
            ))
        fig.update_layout(**PLOTLY_LAYOUT, title=title, xaxis_title=x_label, yaxis_title=y_label)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=get_figsize())
        if groups is not None:
            colors = get_colors(len(np.unique(groups)))
            for i, group in enumerate(np.unique(groups)):
                mask = groups == group
                ax.scatter(coords[mask, 0], coords[mask, 1] if n_components > 1 else [0]*sum(mask),
                          label=group, s=SIZES["scatter_s"], alpha=SIZES["scatter_alpha"],
                          color=colors[i], edgecolors="white", linewidths=0.3)
            style_legend(ax)
        else:
            ax.scatter(coords[:, 0], coords[:, 1] if n_components > 1 else [0]*len(coords),
                      color=COLORS["primary"], s=SIZES["scatter_s"], alpha=SIZES["scatter_alpha"],
                      edgecolors="white", linewidths=0.3)
        style_ax(ax, title=title, xlabel=x_label, ylabel=y_label)
        return fig


# ── Manhattan Plot ───────────────────────────────────────────────────────────

def manhattan(chromosomes, positions, pvalues, threshold=5e-8,
              title="Manhattan Plot", interactive=False, output_html=None):
    """Manhattan plot for GWAS results.

    Args:
        chromosomes: array of chromosome names.
        positions: array of genomic positions.
        pvalues: array of p-values.
        threshold: genome-wide significance threshold.
        title: plot title.
        interactive: if True, return Plotly figure.
        output_html: if set, save as HTML.

    Returns:
        matplotlib Figure or Plotly Figure.
    """
    chromosomes = np.asarray(chromosomes)
    positions = np.asarray(positions, dtype=int)
    pvalues = np.asarray(pvalues, dtype=float)
    neg_log10 = -np.log10(pvalues + 1e-300)

    unique_chroms = list(dict.fromkeys(chromosomes))
    chrom_offsets = {}
    offset = 0
    for chrom in unique_chroms:
        chrom_offsets[chrom] = offset
        offset += max(positions[chromosomes == chrom]) + 1000000

    cumpos = np.array([positions[i] + chrom_offsets[chromosomes[i]] for i in range(len(positions))])

    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        colors = get_colors(len(unique_chroms))
        for i, chrom in enumerate(unique_chroms):
            mask = chromosomes == chrom
            fig.add_trace(go.Scatter(
                x=cumpos[mask], y=neg_log10[mask], mode='markers', name=str(chrom),
                marker=dict(size=4, color=colors[i]),
                text=[f"{chrom}:{int(p)}" for p in positions[mask]], hoverinfo='text+y'
            ))
        fig.add_hline(y=-np.log10(threshold), line_dash="dash", line_color=COLORS["significant"], opacity=0.7)
        fig.update_layout(**PLOTLY_LAYOUT, title=title, xaxis_title="Chromosome", yaxis_title="-log10(p-value)")
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = get_colors(len(unique_chroms))
        for i, chrom in enumerate(unique_chroms):
            mask = chromosomes == chrom
            ax.scatter(cumpos[mask], neg_log10[mask], s=SIZES["scatter_s"]*0.3, alpha=0.6, color=colors[i])
        ax.axhline(-np.log10(threshold), linestyle='--', color=COLORS["significant"], alpha=0.7, linewidth=1)
        style_ax(ax, title=title, xlabel='Chromosome', ylabel='-log10(p)')
        return fig


# ── MA Plot ──────────────────────────────────────────────────────────────────

def ma(mean_expression, log_fc, sig=None, gene_names=None,
       fc_thresh=1.0, title="MA Plot", interactive=False, output_html=None):
    """MA plot showing fold change vs expression level.

    Args:
        mean_expression: mean expression values (log scale).
        log_fc: log2 fold changes.
        sig: boolean array of significance (auto-computed if None).
        gene_names: optional gene names for hover.
        fc_thresh: fold-change threshold for significance.
        title: plot title.
        interactive: if True, return Plotly figure.
        output_html: if set, save as HTML.

    Returns:
        matplotlib Figure or Plotly Figure.
    """
    mean_expression = np.asarray(mean_expression, dtype=float)
    log_fc = np.asarray(log_fc, dtype=float)
    if sig is None:
        sig = np.abs(log_fc) >= fc_thresh

    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mean_expression[~sig], y=log_fc[~sig], mode='markers',
            name='Not significant', marker=dict(size=5, color='gray', opacity=0.5)
        ))
        labels = []
        for i in range(len(log_fc)):
            if sig[i]:
                lbl = gene_names[i] if gene_names and i < len(gene_names) else f"Gene {i}"
                labels.append(f"{lbl}<br>FC={log_fc[i]:.2f}")
            else:
                labels.append("")
        fig.add_trace(go.Scatter(
            x=mean_expression[sig], y=log_fc[sig], mode='markers',
            name='Significant', marker=dict(size=7, color=COLORS["primary"]),
            text=np.array(labels)[sig].tolist(), hoverinfo='text'
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title=title, xaxis_title='Mean Expression (log)',
                          yaxis_title='Log2 Fold Change')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=get_figsize())
        ax.scatter(mean_expression[~sig], log_fc[~sig], s=SIZES["scatter_s"]*0.4, alpha=0.4, color=COLORS["not_sig"])
        ax.scatter(mean_expression[sig], log_fc[sig], s=SIZES["scatter_s"]*0.7, alpha=0.8, color=COLORS["primary"])
        ax.axhline(0, color=COLORS["threshold"], linewidth=0.8, linestyle='--')
        style_ax(ax, title=title, xlabel='Mean Expression (log)', ylabel='Log2 Fold Change')
        return fig


# ── Heatmap ──────────────────────────────────────────────────────────────────

def heatmap(data, row_labels=None, col_labels=None, title="Heatmap",
            cmap="viridis", interactive=False, output_html=None):
    """Heatmap with optional hover values.

    Args:
        data: 2D numpy array or DataFrame.
        row_labels: row names.
        col_labels: column names.
        title: plot title.
        cmap: matplotlib colormap name.
        interactive: if True, return Plotly figure.
        output_html: if set, save as HTML.

    Returns:
        matplotlib Figure or Plotly Figure.
    """
    if isinstance(data, pd.DataFrame):
        if row_labels is None:
            row_labels = data.index.tolist()
        if col_labels is None:
            col_labels = data.columns.tolist()
        data = data.values

    data = np.asarray(data, dtype=float)

    if interactive and HAS_PLOTLY:
        fig = go.Figure(data=go.Heatmap(
            z=data, x=col_labels, y=row_labels, colorscale='Viridis',
            hovertemplate='Row: %{y}<br>Col: %{x}<br>Value: %{z:.3f}<extra></extra>'
        ))
        fig.update_layout(title=title, template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(max(8, len(col_labels or []) * 0.8),
                                        max(6, len(row_labels or []) * 0.4)))
        im = ax.imshow(data, aspect='auto', cmap=cmap)
        if row_labels:
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=FONTS["tick_size"])
        if col_labels:
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=FONTS["tick_size"])
        plt.colorbar(im, ax=ax, shrink=0.8)
        style_ax(ax, title=title)
        return fig


# ── Boxplot ──────────────────────────────────────────────────────────────────

def boxplot(data_dict, title="Boxplot", ylabel="Value",
            interactive=False, output_html=None):
    """Boxplot from dict of group_name -> values.

    Args:
        data_dict: dict mapping group names to value arrays.
        title: plot title.
        ylabel: y-axis label.
        interactive: if True, return Plotly figure.
        output_html: if set, save as HTML.

    Returns:
        matplotlib Figure or Plotly Figure.
    """
    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        for name, values in data_dict.items():
            fig.add_trace(go.Box(y=values, name=name, boxmean='sd'))
        fig.update_layout(title=title, yaxis_title=ylabel, template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=get_figsize())
        bp = ax.boxplot(data_dict.values(), tick_labels=data_dict.keys(),
                        patch_artist=True, widths=0.6)
        colors = get_colors(len(data_dict))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('#CCCCCC')
        for median in bp['medians']:
            median.set_color('#333333')
            median.set_linewidth(1.5)
        style_ax(ax, title=title, ylabel=ylabel)
        return fig


# ── Scatter with Regression ──────────────────────────────────────────────────

def scatter(x, y, labels=None, title="Scatter Plot", xlabel="X", ylabel="Y",
            show_regression=True, interactive=False, output_html=None):
    """Scatter plot with optional regression line.

    Args:
        x, y: numeric arrays.
        labels: point labels for hover.
        title: plot title.
        xlabel, ylabel: axis labels.
        show_regression: add regression line.
        interactive: if True, return Plotly figure.
        output_html: if set, save as HTML.

    Returns:
        matplotlib Figure or Plotly Figure.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        hover = labels if labels else [f"x={xi:.2f}, y={yi:.2f}" for xi, yi in zip(x, y)]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers', text=hover, hoverinfo='text',
            marker=dict(size=8, color=COLORS["primary"])
        ))
        if show_regression:
            from scipy import stats as sp_stats
            slope, intercept, r, p, se = sp_stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            fig.add_trace(go.Scatter(
                x=x_line, y=slope * x_line + intercept, mode='lines',
                name=f'R²={r**2:.3f}', line=dict(color='red', dash='dash')
            ))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=get_figsize())
        ax.scatter(x, y, s=SIZES["scatter_s"], color=COLORS["primary"],
                   edgecolors='white', linewidths=0.3, alpha=SIZES["scatter_alpha"])
        if show_regression:
            from scipy import stats as sp_stats
            slope, intercept, r, p, se = sp_stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, '--', linewidth=1.5,
                   color=COLORS["regression"], label=f'R²={r**2:.3f}, p={p:.2e}')
            ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return fig


# ── Barplot ──────────────────────────────────────────────────────────────────

def barplot(categories, values, errors=None, title="Barplot", ylabel="Value",
            interactive=False, output_html=None):
    """Bar chart with optional error bars.

    Args:
        categories: category names.
        values: values for each category.
        errors: optional error bar values.
        title: plot title.
        ylabel: y-axis label.
        interactive: if True, return Plotly figure.
        output_html: if set, save as HTML.

    Returns:
        matplotlib Figure or Plotly Figure.
    """
    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        error_y = dict(type='data', array=errors) if errors is not None else None
        fig.add_trace(go.Bar(
            x=categories, y=values, error_y=error_y,
            marker_color=COLORS["primary"], marker_line_color='white', marker_line_width=1.5
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title=title, yaxis_title=ylabel)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=get_figsize())
        colors = get_colors(len(categories))
        bars = ax.bar(categories, values, color=colors, edgecolor='white', linewidth=0.5, width=0.7)
        if errors is not None:
            ax.errorbar(range(len(categories)), values, yerr=errors, fmt='none',
                       c=COLORS["threshold"], capsize=4, capthick=1, elinewidth=1)
        style_ax(ax, title=title, ylabel=ylabel)
        return fig


# ── Violin Plot ──────────────────────────────────────────────────────────────

def violin(data_dict, title="Violin Plot", ylabel="Value",
           interactive=False, output_html=None):
    """Violin plot from dict of group_name -> values.

    Args:
        data_dict: dict mapping group names to value arrays.
        title: plot title.
        ylabel: y-axis label.
        interactive: if True, return Plotly figure.
        output_html: if set, save as HTML.

    Returns:
        matplotlib Figure or Plotly Figure.
    """
    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        for name, values in data_dict.items():
            fig.add_trace(go.Violin(y=values, name=name, box_visible=True, meanline_visible=True))
        fig.update_layout(**PLOTLY_LAYOUT, title=title, yaxis_title=ylabel)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=get_figsize())
        parts = ax.violinplot(data_dict.values(), showmeans=True, showmedians=True, showextrema=False)
        colors = get_colors(len(data_dict))
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        parts['cmeans'].set_color(COLORS["primary"])
        parts['cmedians'].set_color(COLORS["significant"])
        ax.set_xticks(range(1, len(data_dict) + 1))
        ax.set_xticklabels(data_dict.keys())
        style_ax(ax, title=title, ylabel=ylabel)
        return fig


# ── Time Series ──────────────────────────────────────────────────────────────

def timeseries(x, ys, names=None, title="Time Series", xlabel="Time", ylabel="Value",
               interactive=False, output_html=None):
    """Multi-line time series plot.

    Args:
        x: time points.
        ys: list of value arrays.
        names: line names.
        title: plot title.
        xlabel, ylabel: axis labels.
        interactive: if True, return Plotly figure.
        output_html: if set, save as HTML.

    Returns:
        matplotlib Figure or Plotly Figure.
    """
    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        for i, y in enumerate(ys):
            name = names[i] if names and i < len(names) else f"Series {i+1}"
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=name))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, y in enumerate(ys):
            name = names[i] if names and i < len(names) else f"Series {i+1}"
            ax.plot(x, y, marker='o', label=name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        return fig


# ── Q-Q Plot ─────────────────────────────────────────────────────────────────

def qqplot(pvalues, title="Q-Q Plot", interactive=False, output_html=None):
    """Q-Q plot for normality testing.

    Args:
        pvalues: array of p-values.
        title: plot title.
        interactive: if True, return Plotly figure.
        output_html: if set, save as HTML.

    Returns:
        matplotlib Figure or Plotly Figure.
    """
    from scipy import stats as sp_stats
    sorted_p = np.sort(np.asarray(pvalues, dtype=float))
    n = len(sorted_p)
    expected = sp_stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    observed = sp_stats.norm.ppf(np.clip(sorted_p, 1e-300, 1 - 1e-300))

    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=expected, y=observed, mode='markers',
            marker=dict(size=6, color='#00ff88'),
            text=[f"p={p:.2e}" for p in sorted_p], hoverinfo='text+x'
        ))
        min_val = min(expected.min(), observed.min())
        max_val = max(expected.max(), observed.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Expected', line=dict(dash='dash', color='red')
        ))
        fig.update_layout(title=title, xaxis_title='Expected', yaxis_title='Observed',
                          template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(expected, observed, s=10, color='#00ff88')
        lims = [min(expected.min(), observed.min()), max(expected.max(), observed.max())]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel('Expected')
        ax.set_ylabel('Observed')
        ax.set_title(title)
        return fig


# ── Venn Diagram ─────────────────────────────────────────────────────────────

def venn(set_sizes, set_names=None, title="Venn Diagram",
         interactive=False, output_html=None):
    """Venn diagram for 2-3 sets.

    Args:
        set_sizes: list of [A, B] or [A, B, C, AB, AC, BC, ABC] sizes.
        set_names: list of set names.
        title: plot title.
        interactive: if True, return Plotly sunburst.
        output_html: if set, save as HTML.

    Returns:
        matplotlib Figure or Plotly Figure.
    """
    if set_names is None:
        set_names = [f"Set {i+1}" for i in range(len(set_sizes))]

    if interactive and HAS_PLOTLY:
        fig = go.Figure(go.Sunburst(
            labels=set_names + ["Intersection"],
            parents=[''] * len(set_names) + [set_names[0]],
            values=set_sizes + [min(set_sizes) if len(set_sizes) == 2 else set_sizes[-1]],
            branchvalues='total'
        ))
        fig.update_layout(title=title, template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(7, 7))
        if len(set_sizes) == 2:
            from matplotlib.patches import Circle
            c1 = Circle((-0.3, 0), 0.8, fc='lightblue', ec='black', alpha=0.5)
            c2 = Circle((0.3, 0), 0.8, fc='lightcoral', ec='black', alpha=0.5)
            ax.add_patch(c1)
            ax.add_patch(c2)
            ax.text(-0.7, 0, str(set_sizes[0]), ha='center', fontsize=14)
            ax.text(0.7, 0, str(set_sizes[1]), ha='center', fontsize=14)
            ax.text(0, 0, str(set_sizes[2] if len(set_sizes) > 2 else 0), ha='center', fontsize=14)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title)
        return fig


# ── Batch Export ─────────────────────────────────────────────────────────────

def export_all_interactive(plots_dict, output_path="interactive_report.html"):
    """Combine multiple Plotly figures into one HTML report.

    Args:
        plots_dict: dict mapping name -> Plotly Figure.
        output_path: output HTML path.
    """
    if not HAS_PLOTLY:
        print("Plotly not installed. Cannot generate interactive report.")
        return

    import plotly.io as pio
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><title>BioSuite Interactive Report</title>",
        "<style>body{background:#0a0f0a;color:#e0ffe8;font-family:sans-serif;padding:20px;}"
        "h1{color:#00ff88;} h2{color:#00cc66;margin-top:30px;}</style></head><body>",
        "<h1>BioSuite Interactive Report</h1>"
    ]

    for name, fig in plots_dict.items():
        html_parts.append(f"<h2>{name}</h2>")
        html_parts.append(pio.to_html(fig, full_html=False))

    html_parts.append("</body></html>")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    print(f"Interactive report saved: {output_path}")
