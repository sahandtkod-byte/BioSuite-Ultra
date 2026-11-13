"""Specialized plots: GSEA, Motif Logo, Sankey Diagram, UMAP."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.utils import (
    apply_glass_ax,
    ask_save_plot,
    config,
    load_dataframe_safe,
    maybe_downsample,
)
from ._demo_guard import demo_guard as _demo

# Import internal drawing functions from biological_plots (for motif and sankey)
from .biological_plots import draw_motif_logo, draw_sankey

# ---------- Global for UMAP ----------
HAS_UMAP = False
# Public surface. ``biosuite/plotting/__init__.py`` star-imports this
# module, and without this list it re-exports the module's own imports
# (numpy, pandas, os, plotly, ...) as package-level names too.
__all__ = [
    'HAS_UMAP',
    'gsea_plot',
    'motif_logo',
    'sankey_diagram',
    'umap_plot',
]

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    pass


def _running_sum(scores):
    """The cumulative score a GSEA-style plot draws, and its normaliser.

    Dividing by the sum of the positive scores is what makes the curve an
    enrichment score rather than a raw fold-change sum — and it is undefined
    when the list has no positive score at all.  That case used to divide by
    ``1e-10`` and produce a curve in the 1e12 range whose own title read
    ``max ES = -1257302210.934`` (measured); an all-negative ranking is refused
    by name instead.
    """
    positive_total = float(np.sum(scores[scores > 0]))
    if not positive_total > 0:
        raise ValueError(
            f"the ranked list has no positive score (largest is {scores.max():.4g}), "
            "so there is nothing to enrich; a GSEA running sum needs at least one "
            "gene ranked above zero")
    return np.cumsum(scores) / positive_total


def gsea_plot(pdf=None):
    print("\n--- GSEA Running Sum Plot ---")
    with _demo('gsea_plot'):
        use_file = input("Load ranked list from file? (y/n): ").strip().lower()
        scores = None
        if use_file == 'y':
            path = input("File path (CSV with 'score' column): ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                score_col = input("Score column (e.g., log2FC): ").strip()
                if score_col in df.columns:
                    scores = pd.to_numeric(df[score_col], errors='coerce').dropna().values
                else:
                    print(f"  Column {score_col!r} is not in the file "
                          f"({', '.join(map(str, df.columns))}); drawing the "
                          "simulated ranked list instead.")
        if scores is None or len(scores) == 0:
            if use_file == 'y':
                print("  No score column was usable; drawing a simulated ranked "
                      "list instead.")
            rng = np.random.default_rng(42)
            scores = rng.normal(0, 1, 1000)
            scores[:50] = rng.normal(2, 0.5, 50)
        order = np.argsort(-scores)
        running_sum = _running_sum(np.asarray(scores, dtype=float)[order])
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(running_sum, color='blue', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='--')
        ax.fill_between(range(len(running_sum)), 0, running_sum, where=(running_sum>0), color='red', alpha=0.3)
        ax.fill_between(range(len(running_sum)), running_sum, 0, where=(running_sum<0), color='green', alpha=0.3)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Enrichment Score (ES)')
        ax.set_title(f'GSEA Running Sum (max ES = {np.max(running_sum):.3f})')
        apply_glass_ax(ax)
        ask_save_plot('gsea', config['save_format'], config['default_dpi'], pdf)


def motif_logo(pdf=None):
    print("\n--- Motif Logo (internal) ---")
    with _demo('motif_logo'):
        use_default = input("Use default alignment? (y/n): ").strip().lower()
        seqs = []
        if use_default == 'n':
            print("Enter sequences (ACGT only, one per line, empty line to finish):")
            while True:
                try:
                    line = input().strip().upper()
                except EOFError:
                    break
                if not line:
                    break
                seqs.append(line)
        if not seqs:
            seqs = ["AAGT", "AAGT", "AAGT", "AAGT", "CAGT", "CAGT", "AACT", "AACT", "ACGT", "ATGT"]
        fig, ax = plt.subplots(figsize=(10,4))
        draw_motif_logo(seqs, ax=ax)
        apply_glass_ax(ax)
        ask_save_plot('motif_logo', config['save_format'], config['default_dpi'], pdf)


def sankey_diagram(pdf=None):
    print("\n--- Sankey Diagram (internal) ---")
    with _demo('sankey_diagram'):
        use_default = input("Use default data? (y/n): ").strip().lower()
        if use_default == 'n':
            labels = input("Node labels (comma-sep): ").split(',')
            sources = list(map(int, input("Source indices (comma-sep): ").split(',')))
            targets = list(map(int, input("Target indices (comma-sep): ").split(',')))
            values = list(map(float, input("Values (comma-sep): ").split(',')))
        else:
            labels = ["Stimulus", "Receptor", "Membrane", "Cytoplasm", "Nucleus", "Apoptosis"]
            sources = [0, 1, 1, 2, 3, 4]
            targets = [1, 2, 3, 4, 5, 5]
            values = [100, 80, 20, 60, 40, 100]
        fig, ax = plt.subplots(figsize=(10,6))
        draw_sankey(labels, sources, targets, values, ax=ax)
        apply_glass_ax(ax)
        ask_save_plot('sankey', config['save_format'], config['default_dpi'], pdf)


def umap_plot(pdf=None):
    print("\n--- UMAP Plot (Dimensionality Reduction) ---")
    if not HAS_UMAP:
        print("UMAP not installed. Install with: pip install umap-learn")
        print("Alternatively, use PCA plot (option 2).")
        return
    with _demo('umap_plot'):
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.empty:
                    print("No numeric columns found.")
                    return
                data = numeric_df.values
                if data.shape[0] > config['downsample_threshold']:
                    # Even stride, not a random draw: the embedding is a claim
                    # about the table, and two runs of the same file used to
                    # draw two different samples of it (cluster 28's lesson,
                    # still here until this cluster).
                    positions = list(range(data.shape[0]))
                    idx, _ = maybe_downsample(positions, positions,
                                              config['downsample_threshold'])
                    print(f"  Drawing {len(idx):,} of {data.shape[0]:,} samples "
                          f"(even stride, first and last kept).")
                    data = data[idx]
                    df = df.iloc[idx]
                group_col = input("Group column (optional): ").strip()
                groups = df[group_col].astype(str).values if group_col and group_col in df.columns else ['Sample']*data.shape[0]
                reducer = umap.UMAP(n_components=2, random_state=42)
                embedding = reducer.fit_transform(data)
                fig, ax = plt.subplots(figsize=(8,6))
                scatter = ax.scatter(embedding[:,0], embedding[:,1], c=pd.Categorical(groups).codes, cmap='tab10', alpha=0.7)
                if len(np.unique(groups)) <= 10:
                    legend = ax.legend(*scatter.legend_elements(), title="Groups")
                    ax.add_artist(legend)
                ax.set_title('UMAP Projection')
                apply_glass_ax(ax)
                ask_save_plot('umap', config['save_format'], config['default_dpi'], pdf)
                return
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 20))
        groups = ['Group1']*50 + ['Group2']*50
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)
        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(embedding[:,0], embedding[:,1], c=pd.Categorical(groups).codes, cmap='tab10', alpha=0.7)
        ax.set_title('UMAP (default data)')
        apply_glass_ax(ax)
        ask_save_plot('umap', config['save_format'], config['default_dpi'], pdf)
