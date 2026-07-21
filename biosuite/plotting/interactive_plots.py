"""
Interactive Plotly-based plots with hover, zoom, and HTML export.
Graceful fallback to matplotlib if plotly is not installed.
"""
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import matplotlib.pyplot as plt
import pandas as pd


def interactive_scatter(x, y, labels=None, color_col=None, title="Scatter Plot",
                        x_label="X", y_label="Y", output_html=None, figsize=(9, 6)):
    """Interactive scatter plot with hover info.

    Args:
        x, y: numeric arrays.
        labels: list of label strings for hover.
        color_col: categorical array for coloring.
        title: plot title.
        output_html: if set, save as interactive HTML.
        figsize: fallback matplotlib size.
    """
    if HAS_PLOTLY:
        fig = go.Figure()
        if color_col is not None:
            categories = list(set(color_col))
            for cat in categories:
                mask = [c == cat for c in color_col]
                hover = [labels[i] if labels else f"x={x[i]:.2f}, y={y[i]:.2f}"
                         for i in range(len(x)) if mask[i]]
                fig.add_trace(go.Scatter(
                    x=np.array(x)[mask], y=np.array(y)[mask],
                    mode='markers', name=str(cat),
                    text=hover, hoverinfo='text'
                ))
        else:
            hover = [labels[i] if labels else f"x={x[i]:.2f}, y={y[i]:.2f}"
                     for i in range(len(x))]
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers',
                text=hover, hoverinfo='text',
                marker=dict(size=8, color='#00ff88')
            ))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                          template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
            print(f"Saved: {output_html}")
        return fig
    else:
        return _fallback_scatter(x, y, labels, color_col, title, x_label, y_label, figsize)


def _fallback_scatter(x, y, labels, color_col, title, x_label, y_label, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    if color_col is not None:
        categories = list(set(color_col))
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        for cat, color in zip(categories, colors):
            mask = [c == cat for c in color_col]
            ax.scatter(np.array(x)[mask], np.array(y)[mask], label=str(cat),
                       color=color, s=40, edgecolors='black')
        ax.legend()
    else:
        ax.scatter(x, y, s=40, color='#00ff88', edgecolors='black')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title + " (static fallback)")
    return fig


def interactive_bar(categories, values, errors=None, title="Bar Chart",
                    x_label="Category", y_label="Value", output_html=None,
                    figsize=(9, 6)):
    """Interactive bar chart."""
    if HAS_PLOTLY:
        fig = go.Figure()
        error_y = dict(type='data', array=errors) if errors is not None else None
        fig.add_trace(go.Bar(
            x=categories, y=values, error_y=error_y,
            marker_color='#00ff88', marker_line_color='white', marker_line_width=1.5
        ))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                          template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))
        ax.bar(categories, values, color=colors, edgecolor='black')
        if errors is not None:
            ax.errorbar(range(len(categories)), values, yerr=errors, fmt='none',
                        c='black', capsize=5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title + " (static fallback)")
        return fig


def interactive_heatmap(data, row_labels=None, col_labels=None, title="Heatmap",
                        colorscale='Viridis', output_html=None, figsize=(9, 7)):
    """Interactive heatmap with hover values."""
    if HAS_PLOTLY:
        fig = go.Figure(data=go.Heatmap(
            z=data, x=col_labels, y=row_labels, colorscale=colorscale,
            hovertemplate='Row: %{y}<br>Col: %{x}<br>Value: %{z:.3f}<extra></extra>'
        ))
        fig.update_layout(title=title, template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data, aspect='auto', cmap='viridis')
        if row_labels:
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
        if col_labels:
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha='right')
        plt.colorbar(im, ax=ax)
        ax.set_title(title + " (static fallback)")
        return fig


def interactive_volcano(lfc, pvals, gene_names=None, fc_thresh=1.0, p_thresh=0.05,
                        title="Volcano Plot", output_html=None, figsize=(9, 7)):
    """Interactive volcano plot with gene name hover."""
    neg_log10 = -np.log10(np.array(pvals, dtype=float) + 1e-300)
    sig = (np.abs(lfc) >= fc_thresh) & (np.array(pvals) < p_thresh)

    if HAS_PLOTLY:
        fig = go.Figure()
        # Non-significant
        fig.add_trace(go.Scatter(
            x=np.array(lfc)[~sig], y=neg_log10[~sig],
            mode='markers', name='Not significant',
            marker=dict(size=5, color='gray', opacity=0.5),
            text=[f"Not sig" for _ in range(sum(~sig))], hoverinfo='text'
        ))
        # Significant
        labels = []
        for i in range(len(lfc)):
            if sig[i]:
                lbl = gene_names[i] if gene_names and i < len(gene_names) else f"Gene {i}"
                labels.append(f"{lbl}<br>FC={lfc[i]:.2f}, p={pvals[i]:.2e}")
            else:
                labels.append("")
        fig.add_trace(go.Scatter(
            x=np.array(lfc)[sig], y=neg_log10[sig],
            mode='markers', name='Significant',
            marker=dict(size=7, color='red'),
            text=np.array(labels)[sig].tolist(), hoverinfo='text'
        ))
        fig.update_layout(title=title, xaxis_title='Log2 Fold Change',
                          yaxis_title='-log10(p-value)', template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        return _fallback_volcano(lfc, neg_log10, sig, title, figsize)


def _fallback_volcano(lfc, neg_log10, sig, title, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    lfc_arr = np.asarray(lfc)
    ax.scatter(lfc_arr[~sig], neg_log10[~sig], s=8, alpha=0.4, color='gray')
    ax.scatter(lfc_arr[sig], neg_log10[sig], s=15, alpha=0.7, color='red')
    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title(title + " (static fallback)")
    return fig


def interactive_line(x, ys, names=None, title="Line Plot",
                     x_label="X", y_label="Y", output_html=None, figsize=(9, 6)):
    """Interactive multi-line plot."""
    if HAS_PLOTLY:
        fig = go.Figure()
        for i, y in enumerate(ys):
            name = names[i] if names and i < len(names) else f"Series {i+1}"
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=name))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                          template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        for i, y in enumerate(ys):
            name = names[i] if names and i < len(names) else f"Series {i+1}"
            ax.plot(x, y, marker='o', label=name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title + " (static fallback)")
        ax.legend()
        return fig


def interactive_3d_scatter(x, y, z, labels=None, color_col=None,
                           title="3D Scatter", output_html=None):
    """Interactive 3D scatter plot."""
    if not HAS_PLOTLY:
        return None

    fig = go.Figure()
    if color_col is not None:
        categories = list(set(color_col))
        for cat in categories:
            mask = [c == cat for c in color_col]
            fig.add_trace(go.Scatter3d(
                x=np.array(x)[mask], y=np.array(y)[mask], z=np.array(z)[mask],
                mode='markers', name=str(cat),
                text=[labels[i] if labels else "" for i in range(len(x)) if mask[i]],
                hoverinfo='text'
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(size=4, color='#00ff88'),
            text=labels, hoverinfo='text'
        ))
    fig.update_layout(title=title, template='plotly_dark')
    if output_html:
        fig.write_html(output_html)
    return fig


def interactive_pie(labels, values, title="Pie Chart", output_html=None):
    """Interactive pie/donut chart."""
    if HAS_PLOTLY:
        fig = go.Figure(data=go.Pie(
            labels=labels, values=values, hole=0.3,
            textinfo='label+percent', marker=dict(colors=plt.cm.Set3.colors)
        ))
        fig.update_layout(title=title, template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title(title + " (static fallback)")
        return fig


def export_interactive_report(plots_dict, output_path="report.html"):
    """Combine multiple interactive plots into a single HTML report.

    Args:
        plots_dict: dict mapping plot_name -> plotly Figure.
        output_path: output HTML path.
    """
    if not HAS_PLOTLY:
        print("Plotly not installed. Cannot generate interactive report.")
        return

    from plotly.subplots import make_subplots
    import plotly.io as pio

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><title>BioSuite Interactive Report</title>",
        "<style>body{background:#1a1a2e;color:#eee;font-family:sans-serif;padding:20px;}"
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


# ── Boxplot ──────────────────────────────────────────────────────────────────

def interactive_boxplot(data_dict, title="Boxplot", y_label="Value",
                        output_html=None, figsize=(9, 6)):
    """Interactive boxplot with outlier hover.

    Args:
        data_dict: dict mapping group_name -> list of values.
    """
    if HAS_PLOTLY:
        fig = go.Figure()
        for name, values in data_dict.items():
            fig.add_trace(go.Box(y=values, name=name, boxmean='sd',
                                 marker_color='#00ff88'))
        fig.update_layout(title=title, yaxis_title=y_label, template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(data_dict.values(), tick_labels=data_dict.keys())
        ax.set_ylabel(y_label)
        ax.set_title(title + " (static fallback)")
        return fig


# ── PCA ──────────────────────────────────────────────────────────────────────

def interactive_pca(coords, variance_explained=None, labels=None,
                    color_col=None, title="PCA Plot", output_html=None,
                    figsize=(9, 7)):
    """Interactive PCA scatter with variance explained in axis labels."""
    x_label = "PC1"
    y_label = "PC2"
    if variance_explained and len(variance_explained) >= 2:
        x_label = f"PC1 ({variance_explained[0]*100:.1f}%)"
        y_label = f"PC2 ({variance_explained[1]*100:.1f}%)"

    if HAS_PLOTLY:
        fig = go.Figure()
        if color_col is not None:
            categories = list(set(color_col))
            for cat in categories:
                mask = [c == cat for c in color_col]
                hover = [labels[i] if labels else f"PC1={coords[i,0]:.2f}, PC2={coords[i,1]:.2f}"
                         for i in range(len(coords)) if mask[i]]
                fig.add_trace(go.Scatter(
                    x=np.array(coords[:, 0])[mask], y=np.array(coords[:, 1])[mask],
                    mode='markers', name=str(cat), text=hover, hoverinfo='text',
                    marker=dict(size=10)
                ))
        else:
            hover = [labels[i] if labels else f"PC1={c[0]:.2f}, PC2={c[1]:.2f}"
                     for i, c in enumerate(coords)]
            fig.add_trace(go.Scatter(
                x=coords[:, 0], y=coords[:, 1], mode='markers',
                text=hover, hoverinfo='text',
                marker=dict(size=10, color='#00ff88')
            ))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                          template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if color_col is not None:
            categories = list(set(color_col))
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
            for cat, color in zip(categories, colors):
                mask = [c == cat for c in color_col]
                ax.scatter(coords[mask, 0], coords[mask, 1], label=str(cat), color=color)
            ax.legend()
        else:
            ax.scatter(coords[:, 0], coords[:, 1], color='#00ff88')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title + " (static fallback)")
        return fig


# ── Manhattan Plot ───────────────────────────────────────────────────────────

def interactive_manhattan(chromosomes, positions, neg_log10_pvals,
                          threshold=None, title="Manhattan Plot",
                          output_html=None, figsize=(12, 5)):
    """Interactive Manhattan plot with SNP hover."""
    if HAS_PLOTLY:
        colors = ['#00ff88', '#00cc66']
        fig = go.Figure()
        unique_chroms = list(dict.fromkeys(chromosomes))
        for i, chrom in enumerate(unique_chroms):
            mask = [c == chrom for c in chromosomes]
            x_vals = np.array(positions)[mask]
            y_vals = np.array(neg_log10_pvals)[mask]
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='markers', name=str(chrom),
                marker=dict(size=4, color=colors[i % 2]),
                text=[f"{chrom}:{int(x)}" for x in x_vals],
                hoverinfo='text+y'
            ))
        if threshold:
            fig.add_hline(y=-np.log10(threshold), line_dash="dash",
                         line_color="red", annotation_text=f"p={threshold}")
        fig.update_layout(title=title, xaxis_title="Position",
                          yaxis_title="-log10(p-value)", template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        unique_chroms = list(dict.fromkeys(chromosomes))
        for i, chrom in enumerate(unique_chroms):
            mask = [c == chrom for c in chromosomes]
            color = colors_hex[i % 2] if 'colors_hex' in dir() else '#00ff88'
            ax.scatter(np.array(positions)[mask], np.array(neg_log10_pvals)[mask],
                       s=4, alpha=0.6, color=['#00ff88', '#00cc66'][i % 2])
        ax.set_xlabel("Position")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(title + " (static fallback)")
        return fig


# ── MA Plot ──────────────────────────────────────────────────────────────────

def interactive_ma(mean_expression, log_fc, sig=None, gene_names=None,
                  title="MA Plot", fc_thresh=1.0, output_html=None, figsize=(9, 7)):
    """Interactive MA plot showing fold change vs expression."""
    if sig is None:
        sig = np.abs(log_fc) >= fc_thresh

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.array(mean_expression)[~sig], y=np.array(log_fc)[~sig],
            mode='markers', name='Not significant',
            marker=dict(size=5, color='gray', opacity=0.5)
        ))
        labels = []
        for i in range(len(log_fc)):
            if sig[i]:
                lbl = gene_names[i] if gene_names and i < len(gene_names) else f"Gene {i}"
                labels.append(f"{lbl}<br>FC={log_fc[i]:.2f}")
            else:
                labels.append("")
        fig.add_trace(go.Scatter(
            x=np.array(mean_expression)[sig], y=np.array(log_fc)[sig],
            mode='markers', name='Significant',
            marker=dict(size=7, color='#00ff88'),
            text=np.array(labels)[sig].tolist(), hoverinfo='text'
        ))
        fig.update_layout(title=title, xaxis_title='Mean Expression (log)',
                          yaxis_title='Log2 Fold Change', template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(mean_expression[~sig], log_fc[~sig], s=8, alpha=0.4, color='gray')
        ax.scatter(mean_expression[sig], log_fc[sig], s=15, alpha=0.7, color='#00ff88')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel("Mean Expression (log)")
        ax.set_ylabel("Log2 Fold Change")
        ax.set_title(title + " (static fallback)")
        return fig


# ── QQ Plot ──────────────────────────────────────────────────────────────────

def interactive_qq(pvalues, title="Q-Q Plot", output_html=None, figsize=(7, 7)):
    """Interactive Q-Q plot for normality testing."""
    from scipy import stats as sp_stats
    sorted_p = np.sort(np.array(pvalues))
    n = len(sorted_p)
    expected = sp_stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    observed = sp_stats.norm.ppf(sorted_p)

    if HAS_PLOTLY:
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
        fig.update_layout(title=title, xaxis_title='Expected',
                          yaxis_title='Observed', template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(expected, observed, s=10, color='#00ff88')
        ax.plot([min(expected.min(), observed.min()), max(expected.max(), observed.max())],
                [min(expected.min(), observed.min()), max(expected.max(), observed.max())],
                'r--', linewidth=1)
        ax.set_xlabel("Expected")
        ax.set_ylabel("Observed")
        ax.set_title(title + " (static fallback)")
        return fig


# ── Violin Plot ──────────────────────────────────────────────────────────────

def interactive_violin(data_dict, title="Violin Plot", y_label="Value",
                       output_html=None, figsize=(9, 6)):
    """Interactive violin plot."""
    if HAS_PLOTLY:
        fig = go.Figure()
        for name, values in data_dict.items():
            fig.add_trace(go.Violin(y=values, name=name, box_visible=True,
                                    meanline_visible=True))
        fig.update_layout(title=title, yaxis_title=y_label, template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        parts = ax.violinplot(list(data_dict.values()), showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(data_dict) + 1))
        ax.set_xticklabels(data_dict.keys())
        ax.set_ylabel(y_label)
        ax.set_title(title + " (static fallback)")
        return fig


# ── Dot Plot (Single-Cell) ───────────────────────────────────────────────────

def interactive_dotplot(categories, genes, values, sizes=None,
                        title="Dot Plot", output_html=None):
    """Interactive dot plot for single-cell marker genes."""
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=categories, y=genes, mode='markers',
            marker=dict(
                size=sizes if sizes is not None else [10] * len(genes),
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Expression")
            ),
            text=[f"{cat}: {gene}<br>Expr: {v:.2f}<br>%: {s:.0f}%"
                  for cat, gene, v, s in zip(categories, genes, values,
                                              sizes if sizes else [10]*len(genes))],
            hoverinfo='text'
        ))
        fig.update_layout(title=title, template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(max(8, len(categories)), max(6, len(genes) * 0.4)))
        sizes_arr = np.array(sizes) if sizes else np.ones(len(genes)) * 50
        colors_arr = np.array(values)
        scatter = ax.scatter(
            [categories.index(c) if isinstance(c, str) else c for c in categories],
            range(len(genes)),
            s=sizes_arr, c=colors_arr, cmap='viridis', alpha=0.7
        )
        plt.colorbar(scatter, ax=ax, label="Expression")
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes)
        ax.set_title(title + " (static fallback)")
        return fig


# ── UpSet Plot (as Sunburst) ────────────────────────────────────────────────

def interactive_upset_sunburst(set_sizes, intersections, title="UpSet (Sunburst)",
                               output_html=None):
    """Interactive sunburst alternative to UpSet plots."""
    if HAS_PLOTLY:
        labels = list(intersections.keys()) + list(set_sizes.keys())
        parents = [''] * len(intersections) + [''] * len(set_sizes)
        values = list(intersections.values()) + list(set_sizes.values())
        fig = go.Figure(go.Sunburst(
            labels=labels, parents=parents, values=values,
            branchvalues='total', marker=dict(colors=plt.cm.Set3.colors[:len(labels)])
        ))
        fig.update_layout(title=title, template='plotly_dark')
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(set_sizes.values(), labels=set_sizes.keys(), autopct='%1.1f%%')
        ax.set_title(title + " (static fallback)")
        return fig
