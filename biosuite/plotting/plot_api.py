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
import html
import numbers

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import matplotlib
import matplotlib.pyplot as plt

from ..core.log import get_logger

logger = get_logger(__name__)

#: The public plotting surface.  interactive_plots reuses the private validation
#: helpers directly, and the style constants below are module configuration, so
#: neither belongs in a star import of this API.
__all__ = [
    "volcano", "pca", "manhattan", "ma", "heatmap", "boxplot", "scatter",
    "barplot", "violin", "timeseries", "qqplot", "venn", "export_all_interactive",
]

# ── Theme helpers ────────────────────────────────────────────────────────────

PLOTLY_TEMPLATE = 'plotly_dark'
MPL_STYLE = {
    'figure.facecolor': '#121212',
    'axes.facecolor': '#1a1a1a',
    'text.color': '#00ff99',
    'axes.labelcolor': '#00ff99',
    'xtick.color': '#00ff99',
    'ytick.color': '#00ff99',
}


def _apply_mpl_style():
    for k, v in MPL_STYLE.items():
        plt.rcParams[k] = v


def _plotly_available(interactive):
    """Whether the interactive branch may run — and a warning when it was asked
    for and cannot.

    Every plot below answers with a matplotlib figure when Plotly is not
    installed, which is the documented degradation; what was missing was the
    sentence saying so.  The caller who wrote ``fig.write_html(...)`` on the next
    line got an ``AttributeError`` with no clue, and the GUI above these
    functions promises an HTML export.
    """
    if interactive and not HAS_PLOTLY:
        logger.warning(
            "interactive=True was asked for but Plotly is not installed, so a "
            "matplotlib figure is returned instead: it has no write_html() and "
            "cannot be shown in a browser. Install plotly, or save the figure "
            "with output_html= (which writes a PNG).")
        return False
    return True


def _png_path(output_html):
    """The PNG that stands in for a matplotlib figure asked to write HTML.

    ``str.replace('.html', '.png')`` rewrote every occurrence in the path, so
    ``report.html/run.html`` lost both, and a ``.pdf`` or ``.svg`` name came back
    as that format rather than the PNG the docstring promises. Only the trailing
    extension is touched now.
    """
    stem = str(output_html)
    if stem.lower().endswith('.html'):
        stem = stem[:-len('.html')]
    elif stem.lower().endswith('.htm'):
        stem = stem[:-len('.htm')]
    return stem + '.png'


def _save_mpl_figure(fig, output_html):
    """Write a matplotlib figure when an ``output_html`` path was asked for.

    Every plot takes ``output_html``, and every one of them except ``volcano``
    quietly returned without writing anything when the Plotly backend was not
    being used — the caller got no file and no reason.  matplotlib cannot export
    HTML, so the figure lands as a PNG under the requested stem.
    """
    if output_html:
        fig.savefig(_png_path(output_html), dpi=150, bbox_inches='tight')


# ── Input guards ─────────────────────────────────────────────────────────────

def _groups(data_dict, plot, minimum=1):
    """Check a ``group_name -> values`` mapping before anything is drawn.

    An empty mapping used to reach matplotlib, which answered with its own
    "Dimensions of labels and X must be compatible", and to Plotly, which
    returned a figure with no traces at all — an empty chart that the caller
    could only read as "no group has any values".
    """
    if not isinstance(data_dict, dict):
        raise TypeError(
            f"{plot}() needs a dict of group name -> values, got "
            f"{type(data_dict).__name__}")
    if not data_dict:
        raise ValueError(
            f"{plot}() was given no groups; an empty chart cannot be told apart "
            f"from a measurement that found nothing, so it is refused")
    thin = {name: len(values) for name, values in data_dict.items()
            if len(values) < minimum}
    if thin:
        parts = ', '.join(f"{name}={count}" for name, count in list(thin.items())[:3])
        raise ValueError(
            f"{plot}() needs at least {minimum} value(s) per group; {len(thin)} "
            f"group(s) have fewer ({parts})")
    return data_dict


def _plotly_colorscale(cmap):
    """A Plotly colorscale for a matplotlib colormap *name*.

    ``heatmap(cmap=)`` is documented as a matplotlib colormap name and the
    static branch accepts every one of them, but the interactive branch handed
    the string straight to Plotly's ``colorscale``: 'viridis' and 'magma' happen
    to exist in both lists, while 'coolwarm' and 'seismic' died with
    "Invalid value of type 'builtins.str' received for the 'colorscale'
    property".  Sampling the colormap works for every name matplotlib knows; a
    name only Plotly knows is passed through untouched.
    """
    if isinstance(cmap, (list, tuple)) and cmap and isinstance(cmap[0], (list, tuple)):
        return cmap                       # already a [[fraction, colour], ...] scale
    name = getattr(cmap, 'name', None) or str(cmap)
    colormap = None
    for candidate in (name, name.lower()):
        try:
            colormap = matplotlib.colormaps.get_cmap(candidate)
            break
        except ValueError:
            continue
    if colormap is None:
        return name  # a name only Plotly knows; let it resolve there
    stops = np.linspace(0.0, 1.0, 10)
    scale = []
    for position in stops:
        red, green, blue = (int(round(255 * channel)) for channel in colormap(position)[:3])
        scale.append((float(position), f'#{red:02x}{green:02x}{blue:02x}'))
    return scale


def _vector(values, label, dtype=float):
    """A 1-D numpy array from whatever the caller handed over, or a clear error.

    Lists are converted because a boolean mask cannot index a plain list:
    ``labels == 'a'`` is the scalar ``False``, and ``coords[False, 0]`` is an
    empty selection — the figure came back with a legend and no points at all.
    """
    arr = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(
            f"{label} must be a 1-D sequence of {arr.size} values, got a "
            f"{arr.ndim}-D array of shape {arr.shape}")
    return arr


def _nonempty(arr, label, plot):
    if arr.size == 0:
        raise ValueError(
            f"{plot}() needs at least one value in {label}; got none. An empty "
            f"figure is not a result — it is the shape a failed run takes.")
    return arr


def _same_length(**named):
    """Refuse arrays whose lengths disagree, naming each one."""
    lengths = {k: len(v) for k, v in named.items()}
    if len(set(lengths.values())) > 1:
        parts = ', '.join(f"{k}={v}" for k, v in lengths.items())
        raise ValueError(
            f"{parts}: every column of a plot must describe the same number of "
            f"entries")
    return next(iter(named.values()))


def _probability(value, label, plot):
    """A p-value cutoff: strictly above 0 and no more than 1.

    ``p_thresh=0`` made the significance line ``-log10(0) = +inf`` (matplotlib
    draws nothing) and a negative one turned it into ``nan``, both silently —
    the figure looked like a volcano with no threshold in it.
    """
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f"{plot}() needs {label} to be a finite number, got {value!r}")
    if not 0 < float(value) <= 1:
        raise ValueError(
            f"{plot}() needs {label} to be a probability above 0 and at most 1, "
            f"got {value!r}; -log10({value!r}) is "
            f"{'infinity' if value == 0 else 'not a real number' if value < 0 else 'negative'}")
    return float(value)


def _probabilities(arr, label, plot):
    """Validate a whole array of p-values, naming what is out of range."""
    bad = arr[(arr < 0) | (arr > 1) | ~np.isfinite(arr)]
    if bad.size:
        shown = ', '.join(repr(float(v)) for v in bad[:3])
        raise ValueError(
            f"{plot}() got {bad.size} of {arr.size} {label} outside [0, 1] "
            f"(NaN and infinity count as outside; for example {shown}). A "
            f"probability cannot leave that range; clipping it would draw the "
            f"impossible value as the most extreme point on the plot.")
    return arr


def _boolean_mask(values, n, plot):
    """A boolean selection of exactly *n* entries, from a list or an array.

    0/1 integers are accepted because that is what a caller reads out of a
    results table; anything else is refused rather than coerced by numpy's rule
    that every non-empty string is true.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(
            f"{plot}() needs sig as a 1-D boolean mask, got a {arr.ndim}-D array "
            f"of shape {arr.shape}")
    if len(arr) != n:
        raise ValueError(
            f"{plot}() was given {n} genes and a sig mask of {len(arr)} entries; "
            f"the mask picks points out, so it needs one entry per gene")
    if arr.dtype == np.bool_:
        return arr
    offenders = [v for v in arr.tolist() if v not in (0, 1)]
    if offenders:
        raise ValueError(
            f"{plot}() needs sig to hold booleans or 0/1; got {len(offenders)} "
            f"other value(s), for example {offenders[0]!r}")
    return arr.astype(bool)


def _count(value, label, plot, minimum=0):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise ValueError(f"{plot}() needs {label} to be a whole number, got {value!r}")
    if value < minimum:
        raise ValueError(f"{plot}() needs {label} >= {minimum}, got {int(value)}")
    return int(value)


def _nonnegative(value, label, plot):
    """A fold-change cutoff: any finite number from 0 up.

    ``fc_thresh=-1`` made ``abs(log2fc) >= -1`` true for every gene, so the whole
    plot was coloured 'significant' and the two guide lines crossed at the wrong
    side of the axis.
    """
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f"{plot}() needs {label} to be a finite number, got {value!r}")
    if value < 0:
        raise ValueError(
            f"{plot}() needs {label} >= 0, got {value!r}; a negative fold-change "
            f"cutoff classifies every gene as significant")
    return float(value)


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
        output_html: if set, save the figure — HTML from Plotly, a PNG of
            the same name from matplotlib.

    Returns:
        matplotlib Figure or Plotly Figure.

    Raises:
        ValueError: if the two arrays describe different numbers of genes, if a
            p-value lies outside [0, 1], if a p-value cutoff is not a probability,
            or if the fold-change cutoff is negative.
    """
    log2fc = _nonempty(_vector(log2fc, 'log2fc'), 'log2fc', 'volcano')
    pvalues = _vector(pvalues, 'pvalues')
    _same_length(log2fc=log2fc, pvalues=pvalues)
    _probabilities(pvalues, 'pvalues', 'volcano')
    fc_thresh = _nonnegative(fc_thresh, 'fc_thresh', 'volcano')
    p_thresh = _probability(p_thresh, 'p_thresh', 'volcano')
    if gene_names is not None and len(gene_names) != log2fc.size:
        raise ValueError(
            f"volcano() was given {log2fc.size} genes and {len(gene_names)} names; "
            f"hover labels have to line up with the points they name")
    neg_log10 = -np.log10(pvalues + 1e-300)
    sig = (np.abs(log2fc) >= fc_thresh) & (pvalues < p_thresh)

    if interactive and _plotly_available(interactive):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=log2fc[~sig], y=neg_log10[~sig], mode='markers',
            name='Not significant', marker=dict(size=5, color='gray', opacity=0.5)
        ))
        labels = []
        for i in range(len(log2fc)):
            if sig[i]:
                lbl = gene_names[i] if gene_names is not None else f"Gene {i}"
                labels.append(f"{lbl}<br>FC={log2fc[i]:.2f}, p={pvalues[i]:.2e}")
            else:
                labels.append("")
        fig.add_trace(go.Scatter(
            x=log2fc[sig], y=neg_log10[sig], mode='markers',
            name='Significant', marker=dict(size=7, color='red'),
            text=np.array(labels)[sig].tolist(), hoverinfo='text'
        ))
        fig.update_layout(title=title, xaxis_title='Log2 Fold Change',
                          yaxis_title='-log10(p-value)', template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(log2fc[~sig], neg_log10[~sig], s=10, alpha=0.5, label='Not sig', color='gray')
        ax.scatter(log2fc[sig], neg_log10[sig], s=20, alpha=0.7, label='Significant', color='red')
        ax.axhline(-np.log10(p_thresh), linestyle='--', color='white', alpha=0.5)
        ax.axvline(-fc_thresh, linestyle='--', alpha=0.5)
        ax.axvline(fc_thresh, linestyle='--', alpha=0.5)
        ax.set_xlabel('Log2 Fold Change')
        ax.set_ylabel('-log10(P-value)')
        ax.set_title(title)
        ax.legend()
        _save_mpl_figure(fig, output_html)
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
        output_html: if set, save the figure — HTML from Plotly, a PNG of
            the same name from matplotlib.

    Returns:
        matplotlib Figure or Plotly Figure.

    Raises:
        ValueError: if there is nothing to decompose, if ``group_col`` names a
            column the frame does not have, or if ``labels`` does not describe
            one entry per sample.  A list of labels used to be compared to a
            single group name, which is the scalar ``False``, and every group
            then selected zero samples: the figure had a legend and no points.
    """
    from sklearn.decomposition import PCA

    n_components = _count(n_components, 'n_components', 'pca', minimum=1)
    if isinstance(data, pd.DataFrame):
        if group_col and group_col not in data.columns:
            raise ValueError(
                f"pca() was asked to group by {group_col!r}, which is not a column "
                f"of this frame ({', '.join(map(str, data.columns))}); an unknown "
                f"column name used to draw an ungrouped cloud instead")
        if group_col:
            groups = data[group_col].astype(str).values
        else:
            groups = None
        numeric_data = data.select_dtypes(include=[np.number]).values
        if numeric_data.shape[1] == 0:
            raise ValueError(
                f"pca() found no numeric columns in a frame of {len(data.columns)}; "
                f"select_dtypes() kept none of them, so there is nothing to decompose")
    else:
        numeric_data = np.asarray(data, dtype=float)
        if numeric_data.ndim != 2:
            raise ValueError(
                f"pca() needs a (samples x features) matrix, got a {numeric_data.ndim}-D "
                f"array of shape {numeric_data.shape}")
        if labels is None:
            groups = None
        else:
            # np.asarray: a plain list cannot be boolean-indexed, and `groups == g`
            # on one is the scalar False that silently selects nothing.
            groups = np.asarray(labels)
    if numeric_data.shape[0] == 0:
        raise ValueError("pca() needs at least one sample row; got an empty matrix")
    if groups is not None and len(groups) != numeric_data.shape[0]:
        raise ValueError(
            f"pca() was given {numeric_data.shape[0]} samples and {len(groups)} "
            f"labels; every label has to sit on the sample it names")
    if labels is not None and len(labels) != numeric_data.shape[0]:
        # The frame path groups by a column, so `groups` never carried the labels;
        # a short list would have died inside numpy when the hover text was built.
        raise ValueError(
            f"pca() was given {numeric_data.shape[0]} samples and {len(labels)} "
            f"labels; every label has to sit on the sample it names")
    if n_components > min(numeric_data.shape):
        raise ValueError(
            f"pca() cannot take {n_components} components from a "
            f"{numeric_data.shape[0]} x {numeric_data.shape[1]} matrix; the most "
            f"that can exist is {min(numeric_data.shape)}")

    pca_model = PCA(n_components=n_components)
    coords = pca_model.fit_transform(numeric_data)
    var_exp = pca_model.explained_variance_ratio_

    x_label = f"PC1 ({var_exp[0]*100:.1f}%)"
    y_label = f"PC2 ({var_exp[1]*100:.1f}%)" if n_components > 1 else "PC2"

    if interactive and _plotly_available(interactive):
        fig = go.Figure()
        if groups is not None:
            for group in np.unique(groups):
                mask = groups == group
                fig.add_trace(go.Scatter(
                    x=coords[mask, 0], y=coords[mask, 1] if n_components > 1 else [0]*sum(mask),
                    mode='markers', name=str(group),
                    text=np.asarray(labels)[mask].tolist() if labels is not None else None,
                    hoverinfo='text+x+y'
                ))
        else:
            fig.add_trace(go.Scatter(
                x=coords[:, 0], y=coords[:, 1] if n_components > 1 else [0]*len(coords),
                mode='markers', marker=dict(size=10, color='#00ff88')
            ))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(7, 6))
        if groups is not None:
            for group in np.unique(groups):
                mask = groups == group
                ax.scatter(coords[mask, 0], coords[mask, 1] if n_components > 1 else [0]*sum(mask),
                          label=group, s=70)
            ax.legend()
        else:
            ax.scatter(coords[:, 0], coords[:, 1] if n_components > 1 else [0]*len(coords),
                      color='#00ff88', s=70)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        _save_mpl_figure(fig, output_html)
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
        output_html: if set, save the figure — HTML from Plotly, a PNG of
            the same name from matplotlib.

    Returns:
        matplotlib Figure or Plotly Figure.

    Raises:
        ValueError: if the three arrays describe different numbers of variants, if
            a p-value is outside [0, 1], if a position is not a whole number or is
            negative, or if the threshold is not a probability.  Float positions
            used to be truncated by ``dtype=int``, so 1.9 Mb was plotted as 1 Mb.
    """
    chromosomes = _nonempty(_vector(chromosomes, 'chromosomes', dtype=None),
                            'chromosomes', 'manhattan')
    positions = _vector(positions, 'positions', dtype=float)
    pvalues = _vector(pvalues, 'pvalues')
    _same_length(chromosomes=chromosomes, positions=positions, pvalues=pvalues)
    _probabilities(pvalues, 'pvalues', 'manhattan')
    threshold = _probability(threshold, 'threshold', 'manhattan')
    fractional = positions[positions != np.floor(positions)]
    if fractional.size:
        raise ValueError(
            f"manhattan() got {fractional.size} position(s) that are not whole "
            f"numbers (for example {float(fractional[0])!r}); a base-pair coordinate "
            f"is a count, and truncating it moves the point to the left")
    negative = positions[positions < 0]
    if negative.size:
        raise ValueError(
            f"manhattan() got {negative.size} negative position(s) (for example "
            f"{float(negative[0])!r}); the chromosome offset arithmetic cannot "
            f"place a point before the start of its chromosome")
    positions = positions.astype(int)
    neg_log10 = -np.log10(pvalues + 1e-300)

    unique_chroms = list(dict.fromkeys(chromosomes))
    chrom_offsets = {}
    offset = 0
    for chrom in unique_chroms:
        chrom_offsets[chrom] = offset
        offset += max(positions[chromosomes == chrom]) + 1000000

    cumpos = np.array([positions[i] + chrom_offsets[chromosomes[i]] for i in range(len(positions))])

    if interactive and _plotly_available(interactive):
        fig = go.Figure()
        colors = ['#00ff88', '#00cc66']
        for i, chrom in enumerate(unique_chroms):
            mask = chromosomes == chrom
            fig.add_trace(go.Scatter(
                x=cumpos[mask], y=neg_log10[mask], mode='markers', name=str(chrom),
                marker=dict(size=4, color=colors[i % 2]),
                text=[f"{chrom}:{int(p)}" for p in positions[mask]], hoverinfo='text+y'
            ))
        fig.add_hline(y=-np.log10(threshold), line_dash="dash", line_color="red")
        fig.update_layout(title=title, xaxis_title="Chromosome", yaxis_title="-log10(p-value)",
                          template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, chrom in enumerate(unique_chroms):
            mask = chromosomes == chrom
            ax.scatter(cumpos[mask], neg_log10[mask], s=5, color=['#00ff88', '#00cc66'][i % 2])
        ax.axhline(-np.log10(threshold), linestyle='--', color='red', alpha=0.7)
        ax.set_xlabel('Chromosome')
        ax.set_ylabel('-log10(p)')
        ax.set_title(title)
        _save_mpl_figure(fig, output_html)
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
        output_html: if set, save the figure — HTML from Plotly, a PNG of
            the same name from matplotlib.

    Returns:
        matplotlib Figure or Plotly Figure.

    Raises:
        ValueError: if the two arrays describe different numbers of genes, or a
            supplied ``sig`` is not one boolean per gene.  A Python list for
            ``sig`` used to die on ``~sig`` with "bad operand type for unary ~:
            'list'", which is a numpy complaint about a caller's reasonable input.
    """
    mean_expression = _nonempty(_vector(mean_expression, 'mean_expression'),
                                'mean_expression', 'ma')
    log_fc = _vector(log_fc, 'log_fc')
    _same_length(mean_expression=mean_expression, log_fc=log_fc)
    if gene_names is not None and len(gene_names) != log_fc.size:
        raise ValueError(
            f"ma() was given {log_fc.size} genes and {len(gene_names)} names; "
            f"hover labels have to line up with the points they name")
    if sig is None:
        fc_thresh = _nonnegative(fc_thresh, 'fc_thresh', 'ma')
        sig = np.abs(log_fc) >= fc_thresh
    else:
        sig = _boolean_mask(sig, len(log_fc), 'ma')

    if interactive and _plotly_available(interactive):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mean_expression[~sig], y=log_fc[~sig], mode='markers',
            name='Not significant', marker=dict(size=5, color='gray', opacity=0.5)
        ))
        labels = []
        for i in range(len(log_fc)):
            if sig[i]:
                lbl = gene_names[i] if gene_names is not None else f"Gene {i}"
                labels.append(f"{lbl}<br>FC={log_fc[i]:.2f}")
            else:
                labels.append("")
        fig.add_trace(go.Scatter(
            x=mean_expression[sig], y=log_fc[sig], mode='markers',
            name='Significant', marker=dict(size=7, color='#00ff88'),
            text=np.array(labels)[sig].tolist(), hoverinfo='text'
        ))
        fig.update_layout(title=title, xaxis_title='Mean Expression (log)',
                          yaxis_title='Log2 Fold Change', template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(mean_expression[~sig], log_fc[~sig], s=8, alpha=0.4, color='gray')
        ax.scatter(mean_expression[sig], log_fc[sig], s=15, alpha=0.7, color='#00ff88')
        # The theme paints the axes near-black (#1a1a1a); a black zero line was
        # invisible on it, which is the same as not drawing the reference at all.
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Mean Expression (log)')
        ax.set_ylabel('Log2 Fold Change')
        ax.set_title(title)
        _save_mpl_figure(fig, output_html)
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
        output_html: if set, save the figure — HTML from Plotly, a PNG of
            the same name from matplotlib.

    Returns:
        matplotlib Figure or Plotly Figure.

    Raises:
        ValueError: if the data is not a 2-D grid with at least one cell, or a
            label list does not have one entry per row or per column.
    """
    if isinstance(data, pd.DataFrame):
        if row_labels is None:
            row_labels = data.index.tolist()
        if col_labels is None:
            col_labels = data.columns.tolist()
        data = data.values

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"heatmap() needs a 2-D grid, got a {data.ndim}-D array of shape "
            f"{data.shape}; a flat list has no rows to name")
    if 0 in data.shape:
        raise ValueError(
            f"heatmap() got a grid with shape {data.shape}; there is no cell to "
            f"colour, so there is no figure to read")
    if row_labels is not None and len(row_labels) != data.shape[0]:
        raise ValueError(
            f"heatmap() has {data.shape[0]} rows and {len(row_labels)} row labels; "
            f"an extra name lands on the wrong row and a missing one shifts every "
            f"label below it")
    if col_labels is not None and len(col_labels) != data.shape[1]:
        raise ValueError(
            f"heatmap() has {data.shape[1]} columns and {len(col_labels)} column "
            f"labels; every name has to sit above the column it names")

    if interactive and _plotly_available(interactive):
        fig = go.Figure(data=go.Heatmap(
            z=data, x=col_labels, y=row_labels, colorscale=_plotly_colorscale(cmap),
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
            ax.set_yticklabels(row_labels, fontsize=8)
        if col_labels:
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        _save_mpl_figure(fig, output_html)
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
        output_html: if set, save the figure — HTML from Plotly, a PNG of
            the same name from matplotlib.

    Returns:
        matplotlib Figure or Plotly Figure.

    Raises:
        ValueError: if there is no group to draw, or a group has no values.
    """
    _groups(data_dict, 'boxplot', minimum=1)

    if interactive and _plotly_available(interactive):
        fig = go.Figure()
        for name, values in data_dict.items():
            fig.add_trace(go.Box(y=values, name=name, boxmean='sd'))
        fig.update_layout(title=title, yaxis_title=ylabel, template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(data_dict.values(), tick_labels=data_dict.keys())
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _save_mpl_figure(fig, output_html)
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
        output_html: if set, save the figure — HTML from Plotly, a PNG of
            the same name from matplotlib.

    Returns:
        matplotlib Figure or Plotly Figure.

    Raises:
        ValueError: if the arrays are empty or of different lengths, if a point
            is not finite, or if a fit was asked for from data that cannot give
            one.  One point used to answer with a legend reading
            ``R²=nan, p=nan`` — a printed measurement that is not a number.
    """
    x = _nonempty(_vector(x, 'x'), 'x', 'scatter')
    y = _vector(y, 'y')
    _same_length(x=x, y=y)
    if labels is not None and len(labels) != x.size:
        raise ValueError(
            f"scatter() was given {x.size} points and {len(labels)} labels; a "
            f"hover label has to sit on the point it names")
    not_finite = ~(np.isfinite(x) & np.isfinite(y))
    if not_finite.any():
        raise ValueError(
            f"scatter() got {int(not_finite.sum())} of {x.size} points that are "
            f"not finite numbers (first at index {int(np.flatnonzero(not_finite)[0])}); "
            f"they would silently drop out of the fit and the axis range")

    fit = None
    if show_regression:
        from scipy import stats as sp_stats
        if x.size < 2:
            raise ValueError(
                f"scatter() cannot fit a line to {x.size} point; a slope needs "
                f"at least two. Pass show_regression=False to plot the point alone")
        if np.unique(x).size < 2:
            raise ValueError(
                f"scatter() cannot fit a line when every x is {float(x[0])!r}: "
                f"the slope is infinite, and scipy answers with NaN — the legend "
                f"would print R²=nan as if it were a result. Pass "
                f"show_regression=False, or plot the groups another way")
        fit = sp_stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = fit.slope * x_line + fit.intercept

    if interactive and _plotly_available(interactive):
        fig = go.Figure()
        hover = labels if labels else [f"x={xi:.2f}, y={yi:.2f}" for xi, yi in zip(x, y)]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers', text=hover, hoverinfo='text',
            marker=dict(size=8, color='#00ff88')
        ))
        if fit is not None:
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode='lines',
                name=f'R²={fit.rvalue**2:.3f}', line=dict(color='red', dash='dash')
            ))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(x, y, s=40, color='#00ff88', edgecolors='black', alpha=0.7)
        if fit is not None:
            ax.plot(x_line, y_line, 'r--', linewidth=1.5,
                   label=f'R²={fit.rvalue**2:.3f}, p={fit.pvalue:.2e}')
            ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _save_mpl_figure(fig, output_html)
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
        output_html: if set, save the figure — HTML from Plotly, a PNG of
            the same name from matplotlib.

    Returns:
        matplotlib Figure or Plotly Figure.

    Raises:
        ValueError: if the categories and values disagree about how many bars
            there are, if an error list has a different length again, or if an
            error is negative.
    """
    categories = _nonempty(_vector(categories, 'categories', dtype=None),
                           'categories', 'barplot')
    values = _vector(values, 'values')
    _same_length(categories=categories, values=values)
    if errors is not None:
        errors = _vector(errors, 'errors')
        _same_length(categories=categories, errors=errors)
        negative = errors[errors < 0]
        if negative.size:
            raise ValueError(
                f"barplot() got {negative.size} negative error value(s) (for "
                f"example {float(negative[0])!r}); an uncertainty is a length, and "
                f"a negative one draws the cap through the axis")

    if interactive and _plotly_available(interactive):
        fig = go.Figure()
        error_y = dict(type='data', array=errors) if errors is not None else None
        fig.add_trace(go.Bar(
            x=categories, y=values, error_y=error_y,
            marker_color='#00ff88', marker_line_color='white', marker_line_width=1.5
        ))
        fig.update_layout(title=title, yaxis_title=ylabel, template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))
        ax.bar(categories, values, color=colors, edgecolor='black')
        if errors is not None:
            ax.errorbar(range(len(categories)), values, yerr=errors, fmt='none', c='black', capsize=5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _save_mpl_figure(fig, output_html)
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
        output_html: if set, save the figure — HTML from Plotly, a PNG of
            the same name from matplotlib.

    Returns:
        matplotlib Figure or Plotly Figure.

    Raises:
        ValueError: if there is no group to draw, or a group has fewer than two
            values — a violin is a density estimate, and one point has no shape.
    """
    _groups(data_dict, 'violin', minimum=2)

    if interactive and _plotly_available(interactive):
        fig = go.Figure()
        for name, values in data_dict.items():
            fig.add_trace(go.Violin(y=values, name=name, box_visible=True, meanline_visible=True))
        fig.update_layout(title=title, yaxis_title=ylabel, template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.violinplot(list(data_dict.values()), showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(data_dict) + 1))
        ax.set_xticklabels(data_dict.keys())
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _save_mpl_figure(fig, output_html)
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
        output_html: if set, save the figure — HTML from Plotly, a PNG of
            the same name from matplotlib.

    Returns:
        matplotlib Figure or Plotly Figure.

    Raises:
        ValueError: if ``ys`` is a flat list of numbers rather than a list of
            series, or a series has a different number of points than ``x``.  A
            flat list used to draw one constant line per number, because
            ``ax.plot(x, 3.0)`` is legal matplotlib.
    """
    x = _nonempty(_vector(x, 'x'), 'x', 'timeseries')
    flat = ("timeseries() needs a list of series, one per line, but got a flat "
            "sequence of {n} plain numbers — wrap it in a list to draw one line")
    if isinstance(ys, np.ndarray) and ys.ndim < 2:
        raise ValueError(flat.format(n=ys.size))
    if isinstance(ys, np.ndarray):
        raw = list(ys)
    elif ys is None or not len(ys):
        raw = []
    elif all(isinstance(v, numbers.Real) for v in ys):
        raise ValueError(flat.format(n=len(ys)))
    else:
        raw = list(ys)
    if not raw:
        raise ValueError("timeseries() was given no series to draw")
    if names is not None and len(names) > len(raw):
        raise ValueError(
            f"timeseries() got {len(raw)} series and {len(names)} names; the extra "
            f"names would be dropped without the caller hearing about it")
    series = []
    for i, values in enumerate(raw):
        arr = _nonempty(_vector(values, f'ys[{i}]'), f'ys[{i}]', 'timeseries')
        if arr.size != x.size:
            raise ValueError(
                f"timeseries() has {x.size} time points and {arr.size} values in "
                f"series {i}; a line cannot end before its x axis does")
        series.append(arr)
    if names is not None and len(names) < len(series):
        logger.warning(
            "timeseries() got %d series and %d names; the series at index %d "
            "onward take an auto name (Series N).",
            len(series), len(names), len(names))

    if interactive and _plotly_available(interactive):
        fig = go.Figure()
        for i, y in enumerate(series):
            name = names[i] if names and i < len(names) else f"Series {i+1}"
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=name))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, y in enumerate(series):
            name = names[i] if names and i < len(names) else f"Series {i+1}"
            ax.plot(x, y, marker='o', label=name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        _save_mpl_figure(fig, output_html)
        return fig


# ── Q-Q Plot ─────────────────────────────────────────────────────────────────

def qqplot(pvalues, title="Q-Q Plot", interactive=False, output_html=None):
    """Q-Q plot for normality testing.

    Args:
        pvalues: array of p-values.
        title: plot title.
        interactive: if True, return Plotly figure.
        output_html: if set, save the figure — HTML from Plotly, a PNG of
            the same name from matplotlib.

    Returns:
        matplotlib Figure or Plotly Figure.

    Raises:
        ValueError: if there is no p-value, or one lies outside [0, 1].  Out of
            range values used to be clipped to the nearest legal probability and
            drawn as the most extreme point on the plot: a p of -0.5 arrived at
            the same height as p = 1e-16.
    """
    from scipy import stats as sp_stats
    sorted_p = np.sort(_nonempty(_vector(pvalues, 'pvalues'), 'pvalues', 'qqplot'))
    _probabilities(sorted_p, 'pvalues', 'qqplot')
    n = len(sorted_p)
    expected = sp_stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    # clip to float64-resolvable epsilon: 1 - 1e-300 == 1.0 exactly,
    # so the old bound gave norm.ppf(1.0) = +inf for p=1 (or 0 -> -inf)
    observed = sp_stats.norm.ppf(np.clip(sorted_p, 1e-16, 1 - 1e-16))

    if interactive and _plotly_available(interactive):
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
        _save_mpl_figure(fig, output_html)
        return fig


# ── Venn Diagram ─────────────────────────────────────────────────────────────

def venn(set_sizes, set_names=None, title="Venn Diagram",
         interactive=False, output_html=None):
    """Venn diagram for two or three sets.

    Args:
        set_sizes: the region sizes in the order the drawing helpers use them —
            ``[only A, only B, both]`` for two sets, ``[A, B, C, AB, AC, BC, ABC]``
            for three.
        set_names: two or three set names (default ``Set 1``, ``Set 2``, ``Set 3``).
        title: plot title.
        interactive: if True, return a Plotly sunburst of the sets and their
            intersection (the pairwise regions only exist in the 3-set figure).
        output_html: if set, save the figure.

    Returns:
        matplotlib Figure or Plotly Figure.

    Raises:
        ValueError: if the sizes are neither 3 nor 7 values, if the names do not
            match the number of sets, if a size is not a non-negative whole
            number, or if the triple overlap is larger than a pairwise one.
            A wrong length used to draw a figure with no circles and no numbers
            in it at all.
    """
    sizes = list(set_sizes)
    if len(sizes) == 3:
        n_sets = 2
    elif len(sizes) == 7:
        n_sets = 3
    else:
        raise ValueError(
            "venn() takes the 3 region sizes of two sets [onlyA, onlyB, both] or "
            f"the 7 of three sets [A, B, C, AB, AC, BC, ABC], not {len(sizes)} values")
    region_names = (('only A', 'only B', 'both') if n_sets == 2 else
                    ('only A', 'only B', 'only C', 'AB', 'AC', 'BC', 'ABC'))
    for i, value in enumerate(sizes):
        _count(value, f"set_sizes[{i}] ({region_names[i]})", 'venn', minimum=0)
    if set_names is None:
        set_names = [f"Set {i+1}" for i in range(n_sets)]
    elif len(set_names) != n_sets:
        raise ValueError(f"{n_sets} sets need {n_sets} names, got {len(set_names)}")
    both = sizes[2] if n_sets == 2 else sizes[-1]
    if n_sets == 3 and both > min(sizes[3:6]):
        raise ValueError(
            f"venn() was given ABC={both} but the pairwise regions "
            f"AB={sizes[3]}, AC={sizes[4]}, BC={sizes[5]}; every member of the "
            f"triple overlap is also a member of each pair, so no pair can be "
            f"smaller than the triple")

    if interactive and _plotly_available(interactive):
        # ``branchvalues='total'`` reads a parent's own value as its TOTAL, so the
        # region sizes the caller passed (which are the parts that do NOT overlap)
        # cannot go there directly: with [50, 40, 15] the wedge named X was drawn
        # 50 wide while |X| is 65, and for three sets the child (ABC) could be
        # larger than the parent it sat in.
        if n_sets == 2:
            totals = [sizes[0] + both, sizes[1] + both]
        else:
            totals = [sizes[0] + sizes[3] + sizes[4] + both,
                      sizes[1] + sizes[3] + sizes[5] + both,
                      sizes[2] + sizes[4] + sizes[5] + both]
        fig = go.Figure(go.Sunburst(
            labels=set_names + ["Intersection"],
            parents=[''] * n_sets + [set_names[0]],
            values=totals + [both],
            branchvalues='total'
        ))
        fig.update_layout(title=title, template=PLOTLY_TEMPLATE)
        if output_html:
            fig.write_html(output_html)
        return fig
    else:
        from .biological_plots import draw_venn2, draw_venn3

        _apply_mpl_style()
        fig, ax = plt.subplots(figsize=(7, 7))
        if n_sets == 2:
            draw_venn2(sizes, set_labels=set_names, ax=ax)
        else:
            draw_venn3(sizes, set_labels=set_names, ax=ax)
        ax.set_title(title)
        _save_mpl_figure(fig, output_html)
        return fig


# ── Batch Export ─────────────────────────────────────────────────────────────

def export_all_interactive(plots_dict, output_path="interactive_report.html"):
    """Combine multiple Plotly figures into one HTML report.

    Args:
        plots_dict: dict mapping name -> Plotly Figure.
        output_path: output HTML path.

    Returns:
        The path written, or ``None`` when Plotly is missing and nothing was
        written.  Both outcomes used to answer ``None``, so a caller could not
        tell a saved report from a refused one without parsing stdout.

    Raises:
        ValueError: if there is nothing to put in the report, or a value is not a
            Plotly figure.
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly not installed. Cannot generate interactive report.")
        print("Plotly not installed. Cannot generate interactive report.")
        return None

    import plotly.io as pio

    if not isinstance(plots_dict, dict):
        raise TypeError(
            f"export_all_interactive() needs a dict of name -> Plotly figure, got "
            f"{type(plots_dict).__name__}")
    if not plots_dict:
        raise ValueError(
            "export_all_interactive() was given no figures; an empty report reads "
            "like a run that produced nothing rather than a call that asked for "
            "nothing")
    not_figures = [name for name, fig in plots_dict.items() if not isinstance(fig, go.Figure)]
    if not_figures:
        raise ValueError(
            f"export_all_interactive() got {len(not_figures)} value(s) that are not "
            f"Plotly figures ({', '.join(map(str, not_figures[:3]))}); a matplotlib "
            f"Figure cannot be inlined here — build it with interactive=True")

    html_parts = [
        "<!DOCTYPE html>",
        '<html><head><meta charset="utf-8">'
        "<title>BioSuite Interactive Report</title>",
        "<style>body{background:#0a0f0a;color:#e0ffe8;font-family:sans-serif;padding:20px;}"
        "h1{color:#00ff88;} h2{color:#00cc66;margin-top:30px;}</style></head><body>",
        "<h1>BioSuite Interactive Report</h1>"
    ]

    # The names are data, not markup: a plot called '<h1>x</h1><img onerror=…>'
    # used to be written straight into the page and executed by the browser.
    # Plotly.js is inlined once, by the first figure — measured at 4.85 MB per
    # figure otherwise, so a twelve-plot report arrived at ~58 MB of the same
    # script, and the file still renders with no network.
    for i, (name, fig) in enumerate(plots_dict.items()):
        html_parts.append(f"<h2>{html.escape(str(name), quote=True)}</h2>")
        html_parts.append(pio.to_html(fig, full_html=False,
                                      include_plotlyjs='inline' if i == 0 else False))

    html_parts.append("</body></html>")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    print(f"Interactive report saved: {output_path}")
    return output_path
