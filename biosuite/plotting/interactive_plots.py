"""
Interactive Plotly-based plots with hover, zoom, and HTML export.
Graceful fallback to matplotlib if plotly is not installed.

Every public function here takes its arrays from a caller who is often a GUI
table or a parsed file, so each one checks the lengths and the emptiness of what
it was given before it draws.  The two backends are held to the same claim: a
figure that shows a threshold in plotly shows it in the fallback too, and an
input one backend refuses is refused by both.
"""
import html

import numpy as np

# The package's own validators: an empty array and a length mismatch are the two
# mistakes every one of these functions used to absorb silently (an empty plotly
# figure, or numpy's own IndexError from three frames down the stack).
from .plot_api import _nonempty, _same_length

# Public surface. ``biosuite/plotting/__init__.py`` star-imports this
# module, and without this list it re-exports the module's own imports
# (numpy, pandas, os, plotly, ...) as package-level names too.
__all__ = [
    'interactive_scatter',
    'interactive_bar',
    'interactive_heatmap',
    'interactive_volcano',
    'interactive_line',
    'interactive_3d_scatter',
    'interactive_pie',
    'export_interactive_report',
    'interactive_boxplot',
    'interactive_pca',
    'interactive_manhattan',
    'interactive_ma',
    'interactive_qq',
    'interactive_violin',
    'interactive_dotplot',
    'interactive_upset_sunburst',
]

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import matplotlib.pyplot as plt


def _categories(values):
    """The categories in the order the caller gave them.

    Three plots built their colour-column groups with ``list(set(...))``, so the
    legend order, the trace order and the colour assignment were whatever the hash
    table produced: the same six labels came out in two different orders in two
    runs (measured), and a figure the README promises is reproducible was not.
    """
    return list(dict.fromkeys(values))


def _write_html(fig, output_html):
    """Write one standalone interactive figure and say where it went.

    Twelve functions each had their own ``fig.write_html(output_html)`` and only
    one of them said anything about it, so a caller could not tell a saved plot
    from an unsaved one.  A standalone file must carry its own copy of plotly.js
    (this machine is often offline), which is why the library is inlined here and
    only once per *report*.
    """
    if not output_html:
        return
    fig.write_html(output_html)
    print(f"Saved: {output_html}")


def interactive_scatter(x, y, labels=None, color_col=None, title="Scatter Plot",
                        x_label="X", y_label="Y", output_html=None, figsize=(9, 6)):
    """Interactive scatter plot with hover info.

    Args:
        x, y: numeric arrays of the same length.
        labels: hover label per point, one per point of *x*.
        color_col: categorical array of the same length; one trace per class.
        title: plot title.
        output_html: if set, save as interactive HTML.
        figsize: fallback matplotlib size.

    Raises:
        ValueError: if the arrays disagree in length, or *x* is empty.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    _same_length(x=x, y=y)
    _nonempty(x, "x", "interactive_scatter")
    if labels is not None:
        _same_length(x=x, labels=labels)
    if color_col is not None:
        _same_length(x=x, color_col=color_col)

    if HAS_PLOTLY:
        fig = go.Figure()
        if color_col is not None:
            for cat in _categories(color_col):
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
        _write_html(fig, output_html)
        return fig
    else:
        return _fallback_scatter(x, y, labels, color_col, title, x_label, y_label, figsize)


def _fallback_scatter(x, y, labels, color_col, title, x_label, y_label, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    if color_col is not None:
        categories = _categories(color_col)
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
    """Interactive bar chart.

    Raises:
        ValueError: if *values*/*errors* do not line up with *categories*, or
            there is nothing to draw.
    """
    _same_length(categories=categories, values=values)
    _nonempty(np.asarray(values), "values", "interactive_bar")
    if errors is not None:
        _same_length(categories=categories, errors=errors)

    if HAS_PLOTLY:
        fig = go.Figure()
        error_y = dict(type='data', array=errors) if errors is not None else None
        fig.add_trace(go.Bar(
            x=categories, y=values, error_y=error_y,
            marker_color='#00ff88', marker_line_color='white', marker_line_width=1.5
        ))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                          template='plotly_dark')
        _write_html(fig, output_html)
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
    """Interactive heatmap with hover values.

    Raises:
        ValueError: if *data* is not a non-empty 2-D matrix, or a label axis does
            not have one label per row/column.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"interactive_heatmap() needs a 2-D matrix, got a "
                         f"{data.ndim}-D array with shape {data.shape}")
    _nonempty(data, "data", "interactive_heatmap")
    if row_labels is not None and len(row_labels) != data.shape[0]:
        raise ValueError(f"interactive_heatmap() got {len(row_labels)} row labels "
                         f"for {data.shape[0]} rows; each row needs one label")
    if col_labels is not None and len(col_labels) != data.shape[1]:
        raise ValueError(f"interactive_heatmap() got {len(col_labels)} column "
                         f"labels for {data.shape[1]} columns; each column needs "
                         "one label")

    if HAS_PLOTLY:
        fig = go.Figure(data=go.Heatmap(
            z=data, x=col_labels, y=row_labels, colorscale=colorscale,
            hovertemplate='Row: %{y}<br>Col: %{x}<br>Value: %{z:.3f}<extra></extra>'
        ))
        fig.update_layout(title=title, template='plotly_dark')
        _write_html(fig, output_html)
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
    """Interactive volcano plot with gene name hover.

    Raises:
        ValueError: if *lfc*/*pvals* disagree in length, are empty, or a threshold
            cannot be one (a negative fold-change floor, a p-value outside 0..1).
    """
    lfc = np.asarray(lfc, dtype=float)
    pvals = np.asarray(pvals, dtype=float)
    _same_length(lfc=lfc, pvals=pvals)
    _nonempty(lfc, "lfc", "interactive_volcano")
    if gene_names is not None:
        _same_length(lfc=lfc, gene_names=gene_names)
    if fc_thresh < 0:
        raise ValueError(f"fc_thresh must be a fold change of 0 or more, got "
                         f"{fc_thresh}: a negative floor marks every gene "
                         "significant")
    if not 0 < p_thresh <= 1:
        raise ValueError(f"p_thresh must be a p-value above 0 and at most 1, got "
                         f"{p_thresh!r}")

    neg_log10 = -np.log10(pvals + 1e-300)
    sig = (np.abs(lfc) >= fc_thresh) & (pvals < p_thresh)

    if HAS_PLOTLY:
        fig = go.Figure()
        # Non-significant
        fig.add_trace(go.Scatter(
            x=lfc[~sig], y=neg_log10[~sig],
            mode='markers', name='Not significant',
            marker=dict(size=5, color='gray', opacity=0.5),
            text=["Not sig"] * int(np.sum(~sig)), hoverinfo='text'
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
            x=lfc[sig], y=neg_log10[sig],
            mode='markers', name='Significant',
            marker=dict(size=7, color='red'),
            text=np.array(labels)[sig].tolist(), hoverinfo='text'
        ))
        fig.update_layout(title=title, xaxis_title='Log2 Fold Change',
                          yaxis_title='-log10(p-value)', template='plotly_dark')
        _write_html(fig, output_html)
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
    """Interactive multi-line plot.

    Raises:
        ValueError: if any series is not as long as *x*, or *x* is empty.
    """
    _nonempty(np.asarray(x), "x", "interactive_line")
    for i, y in enumerate(ys):
        _same_length(**{"x": x, f"ys[{i}]": y})

    if HAS_PLOTLY:
        fig = go.Figure()
        for i, y in enumerate(ys):
            name = names[i] if names and i < len(names) else f"Series {i+1}"
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=name))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                          template='plotly_dark')
        _write_html(fig, output_html)
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
    """Interactive 3D scatter plot.

    Matplotlib has no 3-D stand-in in this module, so without plotly this
    returns ``None`` — a contract the fallback suite pins.  The inputs are
    checked either way, so a caller learns about mismatched arrays whether or
    not the library is installed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    _same_length(x=x, y=y, z=z)
    _nonempty(x, "x", "interactive_3d_scatter")
    if labels is not None:
        _same_length(x=x, labels=labels)
    if color_col is not None:
        _same_length(x=x, color_col=color_col)

    if not HAS_PLOTLY:
        return None

    fig = go.Figure()
    if color_col is not None:
        for cat in _categories(color_col):
            mask = [c == cat for c in color_col]
            fig.add_trace(go.Scatter3d(
                x=x[mask], y=y[mask], z=z[mask],
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
    _write_html(fig, output_html)
    return fig


def interactive_pie(labels, values, title="Pie Chart", output_html=None):
    """Interactive pie/donut chart.

    Raises:
        ValueError: if the labels and values do not line up, or the values add up
            to nothing (every wedge would be undefined).
    """
    _same_length(labels=labels, values=values)
    _nonempty(np.asarray(values), "values", "interactive_pie")
    total = float(np.sum(np.asarray(values, dtype=float)))
    if not total > 0:
        raise ValueError(f"interactive_pie() needs values that add up to more "
                         f"than zero; these add up to {total:g}, so no wedge has "
                         "a share of anything")

    if HAS_PLOTLY:
        fig = go.Figure(data=go.Pie(
            labels=labels, values=values, hole=0.3,
            textinfo='label+percent', marker=dict(colors=plt.cm.Set3.colors)
        ))
        fig.update_layout(title=title, template='plotly_dark')
        _write_html(fig, output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title(title + " (static fallback)")
        return fig


def export_interactive_report(plots_dict, output_path="report.html"):
    """Combine multiple interactive plots into a single HTML report.

    The report carries one copy of plotly.js for the whole file: the old version
    let ``plotly.io.to_html`` embed the library per figure, which measured
    14,563,191 bytes for three plots against 4,870,142 for the same three from
    ``plot_api.export_all_interactive``.  Plot names are HTML-escaped — a name is
    user data, and ``<script>`` in one reached the file verbatim.

    Args:
        plots_dict: dict mapping plot_name -> plotly Figure.
        output_path: output HTML path.

    Returns:
        The path the report was written to.

    Raises:
        ValueError: if there is no figure to write, or a value is not a Figure.
    """
    if not HAS_PLOTLY:
        print("Plotly not installed. Cannot generate interactive report.")
        return None

    import plotly.io as pio

    if not plots_dict:
        raise ValueError("export_interactive_report() needs at least one figure; "
                         "an empty report is not a result")

    html_parts = [
        "<!DOCTYPE html>",
        "<html><head><title>BioSuite Interactive Report</title>",
        "<style>body{background:#1a1a2e;color:#eee;font-family:sans-serif;padding:20px;}"
        "h1{color:#00ff88;} h2{color:#00cc66;margin-top:30px;}</style></head><body>",
        "<h1>BioSuite Interactive Report</h1>"
    ]

    for i, (name, fig) in enumerate(plots_dict.items()):
        if not isinstance(fig, go.Figure):
            raise ValueError(f"plot {name!r} is not a plotly Figure (it is a "
                             f"{type(fig).__name__}); the report writes figures only")
        html_parts.append(f"<h2>{html.escape(str(name))}</h2>")
        html_parts.append(pio.to_html(
            fig, full_html=False,
            include_plotlyjs='inline' if i == 0 else False))

    html_parts.append("</body></html>")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    print(f"Interactive report saved: {output_path}")
    return output_path


# ── Boxplot ──────────────────────────────────────────────────────────────────

def interactive_boxplot(data_dict, title="Boxplot", y_label="Value",
                        output_html=None, figsize=(9, 6)):
    """Interactive boxplot with outlier hover.

    Args:
        data_dict: dict mapping group_name -> list of values, at least one value
            per group.
    """
    if not data_dict:
        raise ValueError("interactive_boxplot() needs at least one group; got an "
                         "empty mapping, which draws an empty box")
    for group, values in data_dict.items():
        if len(values) == 0:
            raise ValueError(f"group {group!r} has no values; a boxplot of "
                             "nothing has no median to draw")

    if HAS_PLOTLY:
        fig = go.Figure()
        for name, values in data_dict.items():
            fig.add_trace(go.Box(y=values, name=name, boxmean='sd',
                                 marker_color='#00ff88'))
        fig.update_layout(title=title, yaxis_title=y_label, template='plotly_dark')
        _write_html(fig, output_html)
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
    """Interactive PCA scatter with variance explained in axis labels.

    Raises:
        ValueError: if *coords* is not a non-empty table of at least two
            components, or *labels*/*color_col* do not have one entry per sample.
            A caller passing the list-of-lists a PCA function naturally returns
            used to get ``TypeError: list indices must be integers``.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"interactive_pca() needs a 2-D table with at least two "
                         f"components per sample, got shape {coords.shape}")
    _nonempty(coords, "coords", "interactive_pca")
    if labels is not None:
        _same_length(coords=coords, labels=labels)
    if color_col is not None:
        _same_length(coords=coords, color_col=color_col)

    x_label = "PC1"
    y_label = "PC2"
    if variance_explained is not None and len(variance_explained) >= 2:
        x_label = f"PC1 ({variance_explained[0]*100:.1f}%)"
        y_label = f"PC2 ({variance_explained[1]*100:.1f}%)"

    if HAS_PLOTLY:
        fig = go.Figure()
        if color_col is not None:
            for cat in _categories(color_col):
                mask = [c == cat for c in color_col]
                hover = [labels[i] if labels else f"PC1={coords[i,0]:.2f}, PC2={coords[i,1]:.2f}"
                         for i in range(len(coords)) if mask[i]]
                fig.add_trace(go.Scatter(
                    x=coords[mask, 0], y=coords[mask, 1],
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
        _write_html(fig, output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        if color_col is not None:
            categories = _categories(color_col)
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
            for cat, color in zip(categories, colors):
                mask = np.array([c == cat for c in color_col])
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
    """Interactive Manhattan plot with SNP hover.

    The significance line is drawn by both backends: the matplotlib fallback used
    to drop it, so the same function showed a threshold in the GUI and none in a
    saved PNG.

    Raises:
        ValueError: if the three columns disagree in length, are empty, or
            *threshold* is not a p-value between 0 and 1.
    """
    _same_length(chromosomes=chromosomes, positions=positions,
                 neg_log10_pvals=neg_log10_pvals)
    _nonempty(np.asarray(neg_log10_pvals), "neg_log10_pvals", "interactive_manhattan")
    threshold_line = None
    if threshold is not None:
        if not 0 < threshold < 1:
            raise ValueError(f"threshold is a p-value and must be between 0 and 1 "
                             f"exclusive, got {threshold!r}: -log10 of it would be "
                             "undefined or negative")
        threshold_line = -np.log10(threshold)

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
        if threshold_line is not None:
            fig.add_hline(y=threshold_line, line_dash="dash",
                         line_color="red", annotation_text=f"p={threshold}")
        fig.update_layout(title=title, xaxis_title="Position",
                          yaxis_title="-log10(p-value)", template='plotly_dark')
        _write_html(fig, output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        unique_chroms = list(dict.fromkeys(chromosomes))
        for i, chrom in enumerate(unique_chroms):
            mask = [c == chrom for c in chromosomes]
            ax.scatter(np.array(positions)[mask], np.array(neg_log10_pvals)[mask],
                       s=4, alpha=0.6, color=['#00ff88', '#00cc66'][i % 2])
        if threshold_line is not None:
            ax.axhline(threshold_line, color="red", linestyle="--",
                       label=f"p={threshold}")
            ax.legend()
        ax.set_xlabel("Position")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(title + " (static fallback)")
        return fig


# ── MA Plot ──────────────────────────────────────────────────────────────────

def interactive_ma(mean_expression, log_fc, sig=None, gene_names=None,
                  title="MA Plot", fc_thresh=1.0, output_html=None, figsize=(9, 7)):
    """Interactive MA plot showing fold change vs expression.

    The zero line is drawn by both backends: the fallback had it and the plotly
    figure did not, so the same call produced two different claims about where
    "no change" is.

    Raises:
        ValueError: if the columns disagree in length, are empty, or *sig* /
            *fc_thresh* cannot describe a significance split.
    """
    mean_expression = np.asarray(mean_expression, dtype=float)
    log_fc = np.asarray(log_fc, dtype=float)
    _same_length(mean_expression=mean_expression, log_fc=log_fc)
    _nonempty(mean_expression, "mean_expression", "interactive_ma")
    if gene_names is not None:
        _same_length(log_fc=log_fc, gene_names=gene_names)
    if fc_thresh < 0:
        raise ValueError(f"fc_thresh must be a fold change of 0 or more, got "
                         f"{fc_thresh}")
    if sig is None:
        sig = np.abs(log_fc) >= fc_thresh
    else:
        sig = np.asarray(sig, dtype=bool)
        _same_length(log_fc=log_fc, sig=sig)

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mean_expression[~sig], y=log_fc[~sig],
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
            x=mean_expression[sig], y=log_fc[sig],
            mode='markers', name='Significant',
            marker=dict(size=7, color='#00ff88'),
            text=np.array(labels)[sig].tolist(), hoverinfo='text'
        ))
        fig.add_hline(y=0, line_color="black", line_width=0.5)
        fig.update_layout(title=title, xaxis_title='Mean Expression (log)',
                          yaxis_title='Log2 Fold Change', template='plotly_dark')
        _write_html(fig, output_html)
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
    """Interactive Q-Q plot for normality testing.

    The p-values are checked the way ``plot_api.qqplot`` checks them: an empty
    list used to die in numpy (``zero-size array to reduction``) and a value
    outside 0..1 was crushed against ±8.2 by ``np.clip`` — an impossible
    probability drawn as the most extreme point.

    Raises:
        ValueError: if there are no p-values, or one of them is not a probability.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    _nonempty(pvalues, "pvalues", "interactive_qq")
    outside = ~np.isfinite(pvalues) | (pvalues < 0) | (pvalues > 1)
    if outside.any():
        raise ValueError(
            f"interactive_qq() got {int(outside.sum())} of {pvalues.size} p-values "
            f"outside [0, 1] (NaN and infinity count as outside; for example "
            f"{pvalues[outside][0]!r}). A probability cannot leave that range; "
            "clipping it would draw the impossible value as the most extreme "
            "point on the plot.")

    from scipy import stats as sp_stats
    sorted_p = np.sort(pvalues)
    n = len(sorted_p)
    expected = sp_stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    observed = sp_stats.norm.ppf(np.clip(sorted_p, 1e-16, 1 - 1e-16))

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
        _write_html(fig, output_html)
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
    """Interactive violin plot.

    Raises:
        ValueError: if there is no group, or a group has no values (``violinplot``
            died on ``zero-size array to reduction`` and the plotly path drew an
            empty figure for the same input).
    """
    if not data_dict:
        raise ValueError("interactive_violin() needs at least one group; got an "
                         "empty mapping, which draws an empty violin")
    for group, values in data_dict.items():
        if len(values) == 0:
            raise ValueError(f"group {group!r} has no values; a density of "
                             "nothing cannot be estimated")

    if HAS_PLOTLY:
        fig = go.Figure()
        for name, values in data_dict.items():
            fig.add_trace(go.Violin(y=values, name=name, box_visible=True,
                                    meanline_visible=True))
        fig.update_layout(title=title, yaxis_title=y_label, template='plotly_dark')
        _write_html(fig, output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.violinplot(list(data_dict.values()), showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(data_dict) + 1))
        ax.set_xticklabels(data_dict.keys())
        ax.set_ylabel(y_label)
        ax.set_title(title + " (static fallback)")
        return fig


# ── Dot Plot (Single-Cell) ───────────────────────────────────────────────────

def interactive_dotplot(categories, genes, values, sizes=None,
                        title="Dot Plot", output_html=None):
    """Interactive dot plot for single-cell marker genes.

    Args:
        categories: cell type / cluster per dot.
        genes: gene per dot.
        values: expression per dot, same length as *genes*.
        sizes: percentage of expressing cells per dot (same length), or ``None``
            for a uniform dot.

    The four columns are zipped for the hover text, so a caller who passed three
    categories, two genes and one value used to get a trace with three x values,
    two y values and two hover strings — missing cells with nothing said.

    Raises:
        ValueError: if the columns disagree in length, or there is nothing to draw.
    """
    _same_length(categories=categories, genes=genes, values=values)
    _nonempty(np.asarray(values), "values", "interactive_dotplot")
    if sizes is not None:
        _same_length(genes=genes, sizes=sizes)

    if HAS_PLOTLY:
        fig = go.Figure()
        _sizes = list(sizes) if sizes is not None else [10] * len(genes)
        fig.add_trace(go.Scatter(
            x=categories, y=genes, mode='markers',
            marker=dict(
                size=_sizes,
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Expression")
            ),
            text=[f"{cat}: {gene}<br>Expr: {v:.2f}<br>%: {s:.0f}%"
                  for cat, gene, v, s in zip(categories, genes, values, _sizes)],
            hoverinfo='text'
        ))
        fig.update_layout(title=title, template='plotly_dark')
        _write_html(fig, output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(max(8, len(categories)), max(6, len(genes) * 0.4)))
        sizes_arr = np.array(sizes) if sizes is not None else np.ones(len(genes)) * 50
        colors_arr = np.array(values)
        scatter = ax.scatter(
            [categories.index(c) if isinstance(c, str) else c for c in categories],
            range(len(genes)),
            s=sizes_arr, c=colors_arr, cmap='viridis', alpha=0.7
        )
        plt.colorbar(scatter, ax=ax, label="Expression")
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes)
        if all(isinstance(c, str) for c in categories):
            categories_in_order = _categories(categories)
            ax.set_xticks(range(len(categories_in_order)))
            ax.set_xticklabels(categories_in_order, rotation=45, ha='right')
        ax.set_title(title + " (static fallback)")
        return fig


# ── UpSet Plot (as Sunburst) ────────────────────────────────────────────────

def interactive_upset_sunburst(set_sizes, intersections, title="UpSet (Sunburst)",
                               output_html=None):
    """Interactive sunburst alternative to UpSet plots.

    Raises:
        ValueError: if either mapping is empty — the plot is the two of them.
    """
    if not set_sizes or not intersections:
        raise ValueError("interactive_upset_sunburst() needs both the set sizes "
                         "and the intersections; one of them is empty")

    if HAS_PLOTLY:
        labels = list(intersections.keys()) + list(set_sizes.keys())
        parents = [''] * len(intersections) + [''] * len(set_sizes)
        values = list(intersections.values()) + list(set_sizes.values())
        fig = go.Figure(go.Sunburst(
            labels=labels, parents=parents, values=values,
            branchvalues='total', marker=dict(colors=plt.cm.Set3.colors[:len(labels)])
        ))
        fig.update_layout(title=title, template='plotly_dark')
        _write_html(fig, output_html)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(set_sizes.values(), labels=set_sizes.keys(), autopct='%1.1f%%')
        ax.set_title(title + " (static fallback: set sizes only, "
                            "the intersections are not drawn)")
        return fig
