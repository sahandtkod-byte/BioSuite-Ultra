"""Specialized plots: GSEA, Motif Logo, Sankey Diagram, UMAP."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from ..core.utils import (config, session, autosave_session, safe_float_input, safe_int_input,
                          load_dataframe_safe, maybe_downsample, apply_glass_ax, ask_save_plot)
# Import internal drawing functions from biological_plots (for motif and sankey)
from .biological_plots import draw_motif_logo, draw_sankey

# ---------- Global for UMAP ----------
HAS_UMAP = False
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    pass

def gsea_plot(pdf=None):
    print("\n--- GSEA Running Sum Plot ---")
    try:
        use_file = input("Load ranked list from file? (y/n): ").strip().lower()
        scores = None
        if use_file == 'y':
            path = input("File path (CSV with 'score' column): ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                score_col = input("Score column (e.g., log2FC): ").strip()
                if score_col in df.columns:
                    scores = pd.to_numeric(df[score_col], errors='coerce').dropna().values
        if scores is None or len(scores) == 0:
            np.random.seed(42)
            scores = np.random.normal(0, 1, 1000)
            scores[:50] = np.random.normal(2, 0.5, 50)
        order = np.argsort(-scores)
        sorted_scores = scores[order]
        running_sum = np.cumsum(sorted_scores) / (np.sum(sorted_scores[sorted_scores > 0]) + 1e-10)
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
        plt.close('all')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

def motif_logo(pdf=None):
    print("\n--- Motif Logo (internal) ---")
    try:
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
        plt.close('all')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

def sankey_diagram(pdf=None):
    print("\n--- Sankey Diagram (internal) ---")
    try:
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
        plt.close('all')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

def umap_plot(pdf=None):
    print("\n--- UMAP Plot (Dimensionality Reduction) ---")
    if not HAS_UMAP:
        print("UMAP not installed. Install with: pip install umap-learn")
        print("Alternatively, use PCA plot (option 2).")
        return
    try:
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
                    idx = np.random.choice(data.shape[0], config['downsample_threshold'], replace=False)
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
                plt.close('all')
                plt.close()
                return
        np.random.seed(42)
        data = np.random.randn(100, 20)
        groups = ['Group1']*50 + ['Group2']*50
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)
        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(embedding[:,0], embedding[:,1], c=pd.Categorical(groups).codes, cmap='tab10', alpha=0.7)
        ax.set_title('UMAP (default data)')
        apply_glass_ax(ax)
        ask_save_plot('umap', config['save_format'], config['default_dpi'], pdf)
        plt.close('all')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")