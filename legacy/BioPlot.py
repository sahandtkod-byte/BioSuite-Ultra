
"""
Bio-Platter Pro v11.0 
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
import signal
import math
import re
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Wedge, Circle, Arc, Rectangle, ConnectionPatch
from matplotlib.path import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ---------- Global flag for Ctrl+C ----------
_interrupted = False

def signal_handler(sig, frame):
    global _interrupted
    _interrupted = True
    print("\n⚠️ Operation interrupted by user. Returning to menu...")

signal.signal(signal.SIGINT, signal_handler)

# ---------- Optional large file support and UMAP ----------
try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    import feather
    HAS_FEATHER = True
except ImportError:
    HAS_FEATHER = False

try:
    import tables
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs): return iterable

# ---------- Tkinter GUI ----------
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk, simpledialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

# ---------- Configuration & Session ----------
DEFAULT_CONFIG = {
    "theme": "light",
    "default_dpi": 180,
    "save_format": "png",
    "interactive": False,
    "downsample_threshold": 5000,
    "quiet": False
}
CONFIG_FILE = "bioplatter_config.json"
SESSION_FILE = "bioplatter_session.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except:
        pass

def load_session():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_session(session):
    try:
        with open(SESSION_FILE, 'w') as f:
            json.dump(session, f, indent=2)
    except:
        pass

def autosave_session():
    save_session(session)

config = load_config()
session = load_session()
_interaction_counter = 0

# ---------- Theme (with fallback) ----------
def set_theme(choice):
    if choice == 'light':
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                plt.style.use('default')
        plt.rcParams['figure.facecolor'] = '#e6f7fa'
        plt.rcParams['axes.facecolor'] = '#f0fcfd'
        plt.rcParams['text.color'] = '#006064'
        plt.rcParams['axes.labelcolor'] = '#006064'
        plt.rcParams['xtick.color'] = '#006064'
        plt.rcParams['ytick.color'] = '#006064'
        plt.rcParams['axes.edgecolor'] = '#80ced6'
        plt.rcParams['grid.color'] = '#b2ebf2'
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['font.size'] = 11
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['axes.titlesize'] = 13
    else:
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#121212'
        plt.rcParams['axes.facecolor'] = '#1a1a1a'
        plt.rcParams['text.color'] = '#00ff99'
        plt.rcParams['axes.labelcolor'] = '#00ff99'
        plt.rcParams['xtick.color'] = '#00ff99'
        plt.rcParams['ytick.color'] = '#00ff99'
        plt.rcParams['axes.edgecolor'] = '#2c2c2c'
        plt.rcParams['grid.color'] = '#333333'
        plt.rcParams['grid.alpha'] = 0.4
        plt.rcParams['font.size'] = 11
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['axes.titlesize'] = 13

# ---------- Safe input helpers ----------
def safe_float_input(prompt, default, key=None):
    global _interaction_counter
    if config.get('quiet', False):
        return default
    if key and key in session:
        default = session[key]
    try:
        val = input(prompt).strip()
        if val == '':
            val = default
        else:
            val = float(val)
        if key:
            session[key] = val
            _interaction_counter += 1
            if _interaction_counter % 10 == 0:
                autosave_session()
        return val
    except:
        print(f"⚠️ Invalid. Using default: {default}")
        return default

def safe_int_input(prompt, default, key=None):
    global _interaction_counter
    if config.get('quiet', False):
        return default
    if key and key in session:
        default = session[key]
    try:
        val = input(prompt).strip()
        if val == '':
            val = default
        else:
            val = int(val)
        if key:
            session[key] = val
            _interaction_counter += 1
            if _interaction_counter % 10 == 0:
                autosave_session()
        return val
    except:
        print(f"⚠️ Invalid. Using default: {default}")
        return default

def safe_list_input(prompt, cast=float, key=None):
    global _interaction_counter
    if config.get('quiet', False):
        return None
    if key and key in session:
        default_str = ",".join(str(x) for x in session[key])
        prompt = f"{prompt} (last: {default_str}): "
    else:
        prompt = f"{prompt}: "
    try:
        raw = input(prompt).strip()
        if not raw:
            if key and key in session:
                return session[key]
            return None
        items = [cast(x.strip()) for x in raw.split(',')]
        if key:
            session[key] = items
            _interaction_counter += 1
            if _interaction_counter % 10 == 0:
                autosave_session()
        return items
    except:
        print("⚠️ Invalid list. Using default.")
        return None

def load_dataframe_safe(filepath):
    """Load CSV/Excel/JSON/TXT/Parquet/Feather/HDF5 with progress for large files."""
    global _interrupted
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    try:
        ext = os.path.splitext(filepath)[1].lower()
        file_size = os.path.getsize(filepath)
        if HAS_TQDM and file_size > 50_000_000:
            print(f"Loading large file ({file_size/1e6:.1f} MB)...")

        if ext == '.csv':
            if file_size > 100_000_000 and not config.get('quiet', False):
                sample = input("Large CSV (>100MB). Sample rows? (y/n): ").strip().lower()
                if sample == 'y':
                    n_rows = safe_int_input("Number of rows to sample (default 50000): ", 50000)
                    df = pd.read_csv(filepath, nrows=n_rows)
                else:
                    df = pd.read_csv(filepath)
            else:
                df = pd.read_csv(filepath)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath)
        elif ext == '.json':
            df = pd.read_json(filepath)
        elif ext == '.txt':
            df = pd.read_csv(filepath, sep=None, engine='python')
        elif ext == '.parquet' and HAS_PARQUET:
            df = pd.read_parquet(filepath)
        elif ext == '.feather' and HAS_FEATHER:
            df = feather.read_dataframe(filepath)
        elif ext in ['.h5', '.hdf5'] and HAS_HDF5:
            df = pd.read_hdf(filepath)
        else:
            print(f"Unsupported format or missing library: {ext}")
            return None
        if _interrupted:
            _interrupted = False
            return None
        if df.empty:
            print("⚠️ File is empty.")
            return None
        if not config.get('quiet', False):
            show_stats = input("Show data summary (describe)? (y/n): ").strip().lower()
            if show_stats == 'y':
                print(df.describe())
        return df
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def maybe_downsample(x, y, max_points=None):
    if max_points is None:
        max_points = config.get('downsample_threshold', 5000)
    if len(x) <= max_points:
        return x, y
    indices = np.random.choice(len(x), max_points, replace=False)
    print(f"⚠️ Downsampled from {len(x)} to {max_points} points.")
    return x[indices], y[indices]

def apply_glass_ax(ax):
    try:
        rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                              transform=ax.transAxes, clip_on=False,
                              facecolor='none', edgecolor=ax.spines['top'].get_edgecolor(),
                              linewidth=0.8, alpha=0.3)
        ax.add_patch(rect)
    except:
        pass

def ask_save_plot(default_name, pdf=None, story=None):
    if pdf is not None:
        try:
            plt.tight_layout()
            pdf.savefig()
            print(f"   ➕ Added to PDF: {default_name}")
            if story is not None:
                story.append(f"![{default_name}]({default_name}.png)")
        except:
            pass
        return
    save = input("💾 Save this plot? (y/n): ").strip().lower()
    if save == 'y':
        name = input(f"   Filename (default {default_name}.{config['save_format']}): ").strip()
        if not name:
            name = default_name
        try:
            plt.tight_layout()
            plt.savefig(f"{name}.{config['save_format']}", dpi=config['default_dpi'],
                        bbox_inches='tight', facecolor=plt.rcParams['figure.facecolor'])
            print(f"   ✅ Saved as {name}.{config['save_format']}")
        except:
            print("⚠️ Save failed.")

# ---------- Statistical Reports (enhanced) ----------
def report_boxplot_stats(data, group_col, value_col):
    stats_dict = {}
    for group in data[group_col].unique():
        vals = data[data[group_col]==group][value_col].dropna()
        if len(vals) > 0:
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            outliers = vals[(vals < q1 - 1.5*iqr) | (vals > q3 + 1.5*iqr)]
            stats_dict[group] = {
                'median': vals.median(),
                'iqr': iqr,
                'outliers': len(outliers)
            }
    print("\n📊 Boxplot Statistics:")
    for group, st in stats_dict.items():
        print(f"   {group}: median={st['median']:.2f}, IQR={st['iqr']:.2f}, outliers={st['outliers']}")
    return stats_dict

def report_scatter_stats(x, y):
    if len(x) < 2:
        print("Insufficient data for correlation.")
        return
    corr, pval = stats.pearsonr(x, y)
    print(f"\n📊 Scatter Statistics: Pearson r = {corr:.3f}, p-value = {pval:.4f}")
    return {'r': corr, 'p': pval}

def report_volcano_stats(lfc, pvals, fc_thresh, p_thresh):
    sig = (np.abs(lfc) >= fc_thresh) & (pvals < p_thresh)
    up = sig & (lfc > 0)
    down = sig & (lfc < 0)
    print(f"\n📊 Volcano Statistics:")
    print(f"   Up-regulated genes: {np.sum(up)}")
    print(f"   Down-regulated genes: {np.sum(down)}")
    print(f"   Total significant: {np.sum(sig)}")
    return {'up': int(np.sum(up)), 'down': int(np.sum(down))}

def report_pca_stats(pca):
    print(f"\n📊 PCA Statistics:")
    for i, var in enumerate(pca.explained_variance_ratio_[:2]):
        print(f"   PC{i+1} explains {var*100:.2f}% of variance")
    return {'var1': pca.explained_variance_ratio_[0], 'var2': pca.explained_variance_ratio_[1]}

def report_manhattan_stats(df, threshold=5e-8):
    sig = df[df['p'] > -np.log10(threshold)]
    if not sig.empty:
        top = sig.loc[sig['p'].idxmax()]
        print(f"\n📊 Manhattan Statistics:")
        print(f"   Significant SNPs: {len(sig)}")
        print(f"   Top SNP: {top['chrom']}:{int(top['pos'])} with -log10(p)={top['p']:.2f}")
    else:
        print("\n📊 No SNPs reached genome-wide significance.")

# ---------- Statistical tests helpers ----------
def add_ttest_to_boxplot(data, group_col, value_col, ax):
    groups = data[group_col].unique()
    if len(groups) == 2:
        g1 = data[data[group_col]==groups[0]][value_col].dropna()
        g2 = data[data[group_col]==groups[1]][value_col].dropna()
        _, pval = stats.ttest_ind(g1, g2)
        # Add annotation
        y_max = max(data[value_col].max(), data[value_col].max()*1.05)
        ax.text(0.5, y_max, f't-test p = {pval:.4f}', ha='center', transform=ax.transAxes, fontsize=9)
    else:
        # ANOVA
        groups_data = [data[data[group_col]==g][value_col].dropna().values for g in groups]
        _, pval = stats.f_oneway(*groups_data)
        ax.text(0.5, data[value_col].max()*1.05, f'ANOVA p = {pval:.4f}', ha='center', transform=ax.transAxes, fontsize=9)

def add_regression_eq(x, y, ax):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    eq = f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}, p = {p_value:.4f}'
    ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

# ---------- Internal plot implementations (no external libs) ----------
def draw_venn2(subsets, set_labels=('A', 'B'), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    r = 1.0
    d = 0.5
    c1 = (-d, 0)
    c2 = (d, 0)
    circle1 = Circle(c1, r, fc='lightblue', ec='black', alpha=0.5)
    circle2 = Circle(c2, r, fc='lightcoral', ec='black', alpha=0.5)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.text(c1[0]-0.7*r, 0, str(subsets[0]), ha='center', va='center', fontsize=14)
    ax.text(c2[0]+0.7*r, 0, str(subsets[1]), ha='center', va='center', fontsize=14)
    ax.text(0, 0, str(subsets[2]), ha='center', va='center', fontsize=14)
    ax.text(c1[0]-0.9*r, -1.2*r, set_labels[0], ha='center', va='center', fontsize=12)
    ax.text(c2[0]+0.9*r, -1.2*r, set_labels[1], ha='center', va='center', fontsize=12)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Venn Diagram (2 sets)')
    return ax

def draw_venn3(subsets, set_labels=('A', 'B', 'C'), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    r = 1.0
    angles = np.radians([0, 120, 240])
    centers = [(r*np.cos(a), r*np.sin(a)) for a in angles]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for i, (cx, cy) in enumerate(centers):
        circle = Circle((cx, cy), r, fc=colors[i], ec='black', alpha=0.5)
        ax.add_patch(circle)
    ab_pos = (centers[0][0]/2 + centers[1][0]/2, centers[0][1]/2 + centers[1][1]/2)
    ac_pos = (centers[0][0]/2 + centers[2][0]/2, centers[0][1]/2 + centers[2][1]/2)
    bc_pos = (centers[1][0]/2 + centers[2][0]/2, centers[1][1]/2 + centers[2][1]/2)
    abc_pos = (sum(c[0] for c in centers)/3, sum(c[1] for c in centers)/3)
    positions = [centers[0], centers[1], centers[2], ab_pos, ac_pos, bc_pos, abc_pos]
    for i, pos in enumerate(positions):
        ax.text(pos[0], pos[1], str(subsets[i]), ha='center', va='center', fontsize=12)
    ax.text(centers[0][0]-1.2*r, centers[0][1]-0.2, set_labels[0], ha='center', va='center', fontsize=12)
    ax.text(centers[1][0]+1.2*r, centers[1][1]-0.2, set_labels[1], ha='center', va='center', fontsize=12)
    ax.text(centers[2][0], centers[2][1]+1.2*r, set_labels[2], ha='center', va='center', fontsize=12)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Venn Diagram (3 sets)')
    return ax

def draw_motif_logo(sequences, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,4))
    max_len = max(len(s) for s in sequences)
    padded = [s.ljust(max_len, '-') for s in sequences]
    chars = ['A', 'C', 'G', 'T']
    counts = {c: np.zeros(max_len) for c in chars}
    for seq in padded:
        for i, ch in enumerate(seq):
            if ch in counts:
                counts[ch][i] += 1
    total = len(sequences)
    positions = np.arange(max_len)
    bar_width = 0.8
    bottom = np.zeros(max_len)
    colors_logo = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
    for ch in chars:
        freq = counts[ch] / total
        ax.bar(positions, freq, width=bar_width, bottom=bottom, color=colors_logo[ch], edgecolor='black', alpha=0.8, label=ch)
        bottom += freq
    ax.set_xlabel('Position')
    ax.set_ylabel('Frequency')
    ax.set_title('Sequence Logo (internal)')
    ax.legend()
    return ax

def draw_sankey(labels, sources, targets, values, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    max_val = max(values)
    unique_nodes = list(set(labels))
    source_x = 0.2
    target_x = 0.8
    source_nodes = set(labels[s] for s in sources)
    target_nodes = set(labels[t] for t in targets)
    all_source_nodes = list(source_nodes)
    all_target_nodes = list(target_nodes)
    for i, node in enumerate(all_source_nodes):
        y = (i+1)/(len(all_source_nodes)+1)
        rect = Rectangle((source_x-0.05, y-0.03), 0.1, 0.06, facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(source_x+0.05, y, node, ha='left', va='center', fontsize=8)
    for i, node in enumerate(all_target_nodes):
        y = (i+1)/(len(all_target_nodes)+1)
        rect = Rectangle((target_x-0.05, y-0.03), 0.1, 0.06, facecolor='lightgreen', edgecolor='black')
        ax.add_patch(rect)
        ax.text(target_x+0.05, y, node, ha='left', va='center', fontsize=8)
    for s_idx, t_idx, val in zip(sources, targets, values):
        src_node = labels[s_idx]
        tgt_node = labels[t_idx]
        if src_node in all_source_nodes:
            src_y = (all_source_nodes.index(src_node)+1)/(len(all_source_nodes)+1)
        else:
            continue
        if tgt_node in all_target_nodes:
            tgt_y = (all_target_nodes.index(tgt_node)+1)/(len(all_target_nodes)+1)
        else:
            continue
        lw = 2 + 8 * val / max_val
        patch = ConnectionPatch(xyA=(source_x, src_y), xyB=(target_x, tgt_y), coordsA='data', coordsB='data',
                                arrowstyle='-', connectionstyle='arc3,rad=0.2', color='gray', alpha=0.6, linewidth=lw)
        ax.add_patch(patch)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Sankey Diagram (internal)')
    return ax

# ---------- Existing Biological Plots (with improvements) ----------
def volcano_plot(pdf=None):
    print("\n--- Volcano Plot ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        lfc, pvals = None, None
        if use_file == 'y':
            last_path = session.get('last_volcano_path', '')
            path = input(f"File path [{last_path}]: ").strip()
            if not path and last_path:
                path = last_path
            if path:
                session['last_volcano_path'] = path
                autosave_session()
                df = load_dataframe_safe(path)
                if df is not None:
                    lfc_col = input("Log2FC column: ").strip()
                    p_col = input("P-value column: ").strip()
                    if lfc_col in df.columns and p_col in df.columns:
                        lfc = pd.to_numeric(df[lfc_col], errors='coerce').dropna().values
                        pvals = pd.to_numeric(df[p_col], errors='coerce').dropna().values
                        if len(lfc) > config['downsample_threshold']:
                            idx = np.random.choice(len(lfc), config['downsample_threshold'], replace=False)
                            lfc = lfc[idx]
                            pvals = pvals[idx]
        if lfc is None or len(lfc) == 0:
            np.random.seed(42)
            lfc = np.random.normal(0, 1.5, 500)
            pvals = np.random.uniform(0, 1, 500)
            pvals[:30] = np.random.uniform(1e-6, 0.05, 30)
        fc_thresh = safe_float_input("Fold-change threshold (default 1.0): ", 1.0, key='volcano_fc')
        p_thresh = safe_float_input("P-value threshold (default 0.05): ", 0.05, key='volcano_p')
        # Add FDR line (Benjamini-Hochberg approximate line)
        neg_log10 = -np.log10(pvals + 1e-300)
        sig = (np.abs(lfc) >= fc_thresh) & (pvals < p_thresh)
        up = sig & (lfc > 0)
        down = sig & (lfc < 0)
        report_volcano_stats(lfc, pvals, fc_thresh, p_thresh)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(lfc[~sig], neg_log10[~sig], s=10, alpha=0.5, label='Not sig', color='gray')
        ax.scatter(lfc[down], neg_log10[down], s=20, alpha=0.7, label='Down', color='blue')
        ax.scatter(lfc[up], neg_log10[up], s=20, alpha=0.7, label='Up', color='red')
        ax.axhline(-np.log10(p_thresh), linestyle='--', color='white' if config['theme']=='dark' else 'black', alpha=0.5)
        ax.axvline(-fc_thresh, linestyle='--', alpha=0.5)
        ax.axvline(fc_thresh, linestyle='--', alpha=0.5)
        # Add FDR approximate curve (Benjamini-Hochberg inspired)
        sorted_p = np.sort(pvals)
        n = len(sorted_p)
        fdr_line = -np.log10(sorted_p * n / (np.arange(1, n+1) * 0.05))  # approximate
        ax.plot(np.percentile(lfc, 50) + np.linspace(-2,2,len(fdr_line)), fdr_line, 'r:', label='FDR approx', alpha=0.6)
        ax.set_xlabel('Log₂ Fold Change')
        ax.set_ylabel('-log₁₀(P‑value)')
        ax.set_title('Volcano Plot')
        ax.legend(framealpha=0.7)
        apply_glass_ax(ax)
        ask_save_plot('volcano', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def pca_plot(pdf=None):
    print("\n--- PCA Plot ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            last_path = session.get('last_pca_path', '')
            path = input(f"File path [{last_path}]: ").strip()
            if not path and last_path:
                path = last_path
            if path:
                session['last_pca_path'] = path
                autosave_session()
                df = load_dataframe_safe(path)
                if df is not None:
                    numeric_df = df.select_dtypes(include=[np.number])
                    if numeric_df.shape[1] >= 2:
                        data = numeric_df.values
                        if data.shape[0] > config['downsample_threshold']:
                            idx = np.random.choice(data.shape[0], config['downsample_threshold'], replace=False)
                            data = data[idx]
                            df = df.iloc[idx]
                        group_col = input("Group column (optional): ").strip()
                        groups = df[group_col].astype(str).values if group_col and group_col in df else ['Sample']*data.shape[0]
                        pca = PCA(n_components=2)
                        pc = pca.fit_transform(data)
                        report_pca_stats(pca)
                        df_pca = pd.DataFrame({'PC1': pc[:,0], 'PC2': pc[:,1], 'Group': groups})
                        fig, ax = plt.subplots(figsize=(7,6))
                        sns.scatterplot(x='PC1', y='PC2', hue='Group', data=df_pca, s=70, palette='Set1', ax=ax)
                        ax.set_title(f'PCA (var: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})')
                        apply_glass_ax(ax)
                        ask_save_plot('pca', pdf)
                        plt.show()
                        return
        np.random.seed(42)
        data = np.random.randn(30, 100)
        groups = ['Ctrl']*15 + ['Treat']*15
        pca = PCA(n_components=2)
        pc = pca.fit_transform(data)
        report_pca_stats(pca)
        df_pca = pd.DataFrame({'PC1': pc[:,0], 'PC2': pc[:,1], 'Group': groups})
        fig, ax = plt.subplots(figsize=(7,6))
        sns.scatterplot(x='PC1', y='PC2', hue='Group', data=df_pca, s=70, palette='Set1', ax=ax)
        ax.set_title(f'PCA (var: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})')
        apply_glass_ax(ax)
        ask_save_plot('pca', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def manhattan_plot(pdf=None):
    print("\n--- Manhattan Plot ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        df = None
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
        if df is None:
            np.random.seed(42)
            n = 1000
            chroms = [f'Chr{i}' for i in range(1,6)]
            df = pd.DataFrame({'chrom': np.random.choice(chroms, n),
                               'pos': np.random.randint(1, 1_000_000, n),
                               'p': -np.log10(np.random.uniform(1e-8, 0.5, n))})
        else:
            chrom_col = input("Chromosome column: ").strip()
            pos_col = input("Position column: ").strip()
            p_col = input("P-value column: ").strip()
            if chrom_col in df.columns and pos_col in df.columns and p_col in df.columns:
                df = df[[chrom_col, pos_col, p_col]].dropna()
                df['chrom'] = df[chrom_col].astype(str)
                df['pos'] = pd.to_numeric(df[pos_col], errors='coerce')
                df['p'] = -np.log10(pd.to_numeric(df[p_col], errors='coerce') + 1e-300)
                df = df.dropna()
            else:
                print("Invalid columns. Using defaults.")
                return manhattan_plot(pdf)
        if df.empty:
            print("No valid data.")
            return
        report_manhattan_stats(df, threshold=5e-8)
        df = df.sort_values(['chrom', 'pos'])
        df['cumpos'] = df.groupby('chrom')['pos'].transform(lambda x: x.cumsum())
        offsets = df.groupby('chrom')['cumpos'].max().cumsum().shift(1).fillna(0)
        df['cumpos'] += df['chrom'].map(offsets.to_dict())
        fig, ax = plt.subplots(figsize=(12,5))
        colors = ['#2c7bb6', '#abd9e9'] if config['theme']=='light' else ['#00ff99', '#33cc66']
        for i, (chr_name, grp) in enumerate(df.groupby('chrom')):
            ax.scatter(grp['cumpos'], grp['p'], s=5, color=colors[i%2], label=chr_name)
        ax.axhline(-np.log10(5e-8), linestyle='--', color='red', alpha=0.7, label='Genome-wide sig')
        ax.set_xlabel('Chromosome')
        ax.set_ylabel('-log₁₀(p)')
        ax.set_title('Manhattan Plot')
        ax.set_xticks(offsets + df.groupby('chrom')['pos'].max().values / 2)
        ax.set_xticklabels(offsets.index)
        ax.legend(loc='upper right', ncol=2, framealpha=0.5)
        apply_glass_ax(ax)
        ask_save_plot('manhattan', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def ma_plot(pdf=None):
    print("\n--- MA Plot ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                col1 = input("First expression column: ").strip()
                col2 = input("Second expression column: ").strip()
                if col1 in df.columns and col2 in df.columns:
                    A = (np.log2(df[col1].astype(float)+1) + np.log2(df[col2].astype(float)+1)) / 2
                    M = np.log2(df[col2].astype(float)+1) - np.log2(df[col1].astype(float)+1)
                else:
                    print("Invalid columns. Using defaults.")
                    use_file = 'n'
        if use_file != 'y':
            np.random.seed(42)
            M = np.random.normal(0, 1, 500)
            A = np.random.normal(8, 2, 500)
        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(A, M, s=10, alpha=0.6, color='steelblue')
        ax.axhline(0, linestyle='--', color='red')
        ax.axhline(1, linestyle=':', color='gray')
        ax.axhline(-1, linestyle=':', color='gray')
        ax.set_xlabel('A (mean log intensity)')
        ax.set_ylabel('M (log₂ fold change)')
        ax.set_title('MA Plot')
        apply_glass_ax(ax)
        ask_save_plot('maplot', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def venn_diagram(pdf=None):
    print("\n--- Venn Diagram (internal) ---")
    try:
        n_sets = input("Number of sets (2 or 3): ").strip()
        if n_sets == '2':
            sizes = safe_list_input("Enter sizes a,b,ab (e.g., 10,15,4): ", float)
            if sizes is None or len(sizes) != 3:
                sizes = [10,15,4]
            fig, ax = plt.subplots(figsize=(6,6))
            draw_venn2(sizes, set_labels=('Set A', 'Set B'), ax=ax)
        elif n_sets == '3':
            sizes = safe_list_input("Enter sizes a,b,c,ab,ac,bc,abc: ", float)
            if sizes is None or len(sizes) != 7:
                sizes = [8,8,8,3,3,3,1]
            fig, ax = plt.subplots(figsize=(6,6))
            draw_venn3(sizes, set_labels=('A', 'B', 'C'), ax=ax)
        else:
            print("Only 2 or 3 sets supported.")
            return
        apply_glass_ax(ax)
        ask_save_plot('venn', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def barplot_custom(pdf=None):
    print("\n--- Barplot ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        groups, values, errors = None, None, None
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                x_col = input("Category column: ").strip()
                y_col = input("Value column: ").strip()
                err_col = input("Error column (optional): ").strip()
                if x_col in df.columns and y_col in df.columns:
                    groups = df[x_col].astype(str).tolist()
                    values = pd.to_numeric(df[y_col], errors='coerce').dropna().values
                    if err_col and err_col in df.columns:
                        errors = pd.to_numeric(df[err_col], errors='coerce').dropna().values
        if groups is None:
            groups = ['Ctrl', 'TrA', 'TrB']
            values = np.array([5.2, 8.7, 3.4])
            errors = np.array([0.5, 0.8, 0.4])
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x=groups, y=values, palette='viridis', edgecolor='black', ax=ax)
        if errors is not None and len(errors) == len(values):
            ax.errorbar(x=range(len(groups)), y=values, yerr=errors, fmt='none', c='black', capsize=5)
        ax.set_ylabel('Expression')
        ax.set_title('Barplot')
        apply_glass_ax(ax)
        ask_save_plot('barplot', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def boxplot_custom(pdf=None):
    print("\n--- Boxplot ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                value_col = input("Value column: ").strip()
                group_col = input("Group column: ").strip()
                if value_col in df.columns and group_col in df.columns:
                    df_plot = df[[group_col, value_col]].dropna()
                    fig, ax = plt.subplots(figsize=(6,5))
                    sns.boxplot(x=group_col, y=value_col, data=df_plot, palette='Set2', ax=ax)
                    sns.stripplot(x=group_col, y=value_col, data=df_plot, color='black', alpha=0.6, size=4, ax=ax)
                    ax.set_title('Boxplot')
                    apply_glass_ax(ax)
                    report_boxplot_stats(df_plot, group_col, value_col)
                    # Add statistical test if two groups
                    if len(df_plot[group_col].unique()) >= 2:
                        add_ttest_to_boxplot(df_plot, group_col, value_col, ax)
                    ask_save_plot('boxplot', pdf)
                    plt.show()
                    return
        ctrl = np.random.normal(5,1,30)
        treat = np.random.normal(7.5,1.2,30)
        df_plot = pd.DataFrame({'Group': ['Ctrl']*30 + ['Treat']*30, 'Value': np.concatenate([ctrl, treat])})
        fig, ax = plt.subplots(figsize=(6,5))
        sns.boxplot(x='Group', y='Value', data=df_plot, palette='Set2', ax=ax)
        sns.stripplot(x='Group', y='Value', data=df_plot, color='black', alpha=0.6, size=4, ax=ax)
        ax.set_title('Boxplot')
        apply_glass_ax(ax)
        report_boxplot_stats(df_plot, 'Group', 'Value')
        add_ttest_to_boxplot(df_plot, 'Group', 'Value', ax)
        ask_save_plot('boxplot', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def heatmap_custom(pdf=None):
    print("\n--- Heatmap ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr = numeric_df.corr()
                    fig, ax = plt.subplots(figsize=(7,6))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                    ax.set_title('Correlation Heatmap')
                    apply_glass_ax(ax)
                    ask_save_plot('heatmap', pdf)
                    plt.show()
                    return
        corr = np.array([[1,0.8,0.2,0.1],[0.8,1,0.3,0.2],[0.2,0.3,1,0.7],[0.1,0.2,0.7,1]])
        labels = ['Gene_A','Gene_B','Gene_C','Gene_D']
        fig, ax = plt.subplots(figsize=(7,6))
        sns.heatmap(pd.DataFrame(corr, index=labels, columns=labels), annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Gene Correlation')
        apply_glass_ax(ax)
        ask_save_plot('heatmap', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def scatter_custom(pdf=None):
    print("\n--- Scatter Plot ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        x, y = None, None
        corr_type = input("Correlation type (pearson/spearman) [pearson]: ").strip().lower() or 'pearson'
        if use_file == 'y':
            last_path = session.get('last_scatter_path', '')
            path = input(f"File path [{last_path}]: ").strip()
            if not path and last_path:
                path = last_path
            if path:
                session['last_scatter_path'] = path
                autosave_session()
                df = load_dataframe_safe(path)
                if df is not None:
                    x_col = input("X column: ").strip()
                    y_col = input("Y column: ").strip()
                    if x_col in df.columns and y_col in df.columns:
                        x = pd.to_numeric(df[x_col], errors='coerce').dropna().values
                        y = pd.to_numeric(df[y_col], errors='coerce').dropna().values
                        x, y = maybe_downsample(x, y)
        if x is None or len(x) < 2:
            np.random.seed(42)
            x = np.linspace(0,10,20)
            y = 2.5*x + np.random.normal(0,1.5,20)
        fig, ax = plt.subplots(figsize=(7,5))
        sns.regplot(x=x, y=y, ci=None, scatter_kws={'s':60, 'edgecolor':'black'}, line_kws={'color':'red','ls':'--'}, ax=ax)
        ax.set_title('Scatter with Regression')
        apply_glass_ax(ax)
        # Compute appropriate correlation
        if corr_type == 'spearman':
            corr, pval = stats.spearmanr(x, y)
            print(f"\n📊 Scatter Statistics: Spearman ρ = {corr:.3f}, p-value = {pval:.4f}")
            eq = f'Spearman ρ = {corr:.3f}, p = {pval:.4f}'
            ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            print(f"\n📊 Scatter Statistics: Pearson r = {r_value:.3f}, p-value = {p_value:.4f}")
            eq = f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}, p = {p_value:.4f}'
            ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        ask_save_plot('scatter', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def timeseries_plot(pdf=None):
    print("\n--- Time Series ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                time_col = input("Time column: ").strip()
                val_col = input("Value column: ").strip()
                if time_col in df.columns and val_col in df.columns:
                    times = pd.to_numeric(df[time_col], errors='coerce').dropna().values
                    values = pd.to_numeric(df[val_col], errors='coerce').dropna().values
                    if len(times) == len(values):
                        fig, ax = plt.subplots(figsize=(8,5))
                        sns.lineplot(x=times, y=values, marker='o', color='green', ax=ax)
                        ax.set_title('Time Series')
                        apply_glass_ax(ax)
                        ask_save_plot('timeseries', pdf)
                        plt.show()
                        return
        times = np.arange(0,24,2)
        values = np.exp(-0.1*times) + np.random.normal(0,0.05,len(times))
        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(x=times, y=values, marker='o', color='green', ax=ax)
        ax.set_title('Time Series')
        apply_glass_ax(ax)
        ask_save_plot('timeseries', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def qq_plot(pdf=None):
    print("\n--- QQ-plot (Normality Check) ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        data = None
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                col = input("Column to test: ").strip()
                if col in df.columns:
                    data = pd.to_numeric(df[col], errors='coerce').dropna().values
        if data is None or len(data) == 0:
            np.random.seed(42)
            data = np.random.normal(0, 1, 200)
        if len(data) > config['downsample_threshold']:
            data = np.random.choice(data, config['downsample_threshold'], replace=False)
        fig, ax = plt.subplots(figsize=(6,6))
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title('QQ-plot (Normal Distribution)')
        apply_glass_ax(ax)
        ask_save_plot('qqplot', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def clustered_heatmap(pdf=None):
    print("\n--- Clustered Heatmap with Dendrograms ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        data = None
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.empty:
                    print("No numeric columns found.")
                    return
                data = numeric_df
        if data is None:
            np.random.seed(42)
            data = pd.DataFrame(np.random.rand(10, 10), columns=[f'Var{i}' for i in range(10)])
        if data.shape[0] > 2000:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            try:
                pc = pca.fit_transform(data)
                bins = np.percentile(pc.flatten(), np.linspace(0,100,11))
                indices = []
                for i in range(len(bins)-1):
                    mask = (pc.flatten() >= bins[i]) & (pc.flatten() < bins[i+1])
                    idx = np.where(mask)[0]
                    if len(idx) > 0:
                        n_sample = max(1, int(2000 * len(idx) / data.shape[0]))
                        indices.extend(np.random.choice(idx, min(n_sample, len(idx)), replace=False))
                data = data.iloc[indices]
            except:
                data = data.sample(2000)
            print(f"⚠️ Downsampled to {data.shape[0]} rows for heatmap.")
        g = sns.clustermap(data, cmap='coolwarm', standard_scale=1,
                           figsize=(10, 8), dendrogram_ratio=0.2,
                           cbar_pos=(0.02, 0.8, 0.03, 0.18))
        g.ax_heatmap.set_title('Clustered Heatmap')
        ask_save_plot('clustered_heatmap', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def circos_plot(pdf=None):
    print("\n--- Simple Circos Plot (internal) ---")
    try:
        use_default = input("Use default data? (y/n): ").strip().lower()
        if use_default == 'n':
            sectors = {}
            n = safe_int_input("Number of sectors: ", 3)
            for i in range(n):
                name = input(f"Sector {i+1} name: ").strip()
                size = safe_float_input(f"Size of {name}: ", 10)
                sectors[name] = size
            links = []
            n_links = safe_int_input("Number of links: ", 2)
            for i in range(n_links):
                s1 = input(f"Link {i+1} from sector: ").strip()
                s2 = input(f"          to sector: ").strip()
                links.append((s1, s2))
        else:
            sectors = {"Gene_A": 10, "Gene_B": 8, "Gene_C": 12, "Gene_D": 6}
            links = [("Gene_A", "Gene_B"), ("Gene_B", "Gene_C"), ("Gene_C", "Gene_D"), ("Gene_D", "Gene_A")]
        fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': 'polar'})
        total = sum(sectors.values())
        start = 0
        colors = plt.cm.tab20(np.linspace(0,1,len(sectors)))
        for (name, size), color in zip(sectors.items(), colors):
            end = start + 2*np.pi * size/total
            ax.bar(x=start, height=0.5, width=end-start, bottom=0.2,
                   color=color, edgecolor='black', alpha=0.7, align='edge')
            ax.text(start + (end-start)/2, 0.8, name, ha='center', va='center', fontsize=8)
            start = end
        sector_names = list(sectors.keys())
        for (s1, s2) in links:
            if s1 not in sector_names or s2 not in sector_names:
                continue
            i1 = sector_names.index(s1)
            i2 = sector_names.index(s2)
            start1 = sum(list(sectors.values())[:i1])/total * 2*np.pi
            end1 = start1 + list(sectors.values())[i1]/total * 2*np.pi
            start2 = sum(list(sectors.values())[:i2])/total * 2*np.pi
            end2 = start2 + list(sectors.values())[i2]/total * 2*np.pi
            theta1 = (start1 + end1)/2
            theta2 = (start2 + end2)/2
            rad = np.linspace(theta1, theta2, 100)
            r = np.linspace(1.0, 1.0, 100)
            ax.plot(rad, r, color='red', alpha=0.6, linewidth=1)
        ax.set_ylim(0,1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Simple Circos Plot")
        apply_glass_ax(ax)
        ask_save_plot('circos', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def alignment_viewer(pdf=None):
    print("\n--- Alignment Viewer ---")
    try:
        use_default = input("Use default alignment? (y/n): ").strip().lower()
        sequences = []
        if use_default == 'n':
            print("Enter sequences (one per line, empty line to finish):")
            while True:
                seq = input().strip().upper()
                if not seq:
                    break
                sequences.append(seq)
        if not sequences:
            sequences = ["ATCGATCG", "ATCGATCG", "ATCGGTCG", "ATCGATAG", "ATCGATGG"]
        max_len = max(len(s) for s in sequences)
        aligned = [list(s.ljust(max_len, '-')) for s in sequences]
        fig, ax = plt.subplots(figsize=(max_len*0.3, len(sequences)*0.4))
        ax.set_xlim(-0.5, max_len-0.5)
        ax.set_ylim(-0.5, len(sequences)-0.5)
        for i, row in enumerate(aligned):
            for j, base in enumerate(row):
                color = {'A':'lightgreen', 'T':'lightcoral', 'C':'lightblue', 'G':'lightyellow', '-':'lightgray'}.get(base, 'white')
                rect = plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
                ax.text(j, i, base, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.set_xticks(range(max_len))
        ax.set_yticks(range(len(sequences)))
        ax.set_xticklabels([f"{i+1}" for i in range(max_len)])
        ax.set_yticklabels([f"Seq{i+1}" for i in range(len(sequences))])
        ax.set_title("Alignment Viewer")
        apply_glass_ax(ax)
        ask_save_plot('alignment', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

# ---------- New Plots: Violin, Raincloud, Ridge, Dot ----------
def violin_plot(pdf=None):
    print("\n--- Violin Plot (Enhanced Distribution) ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                value_col = input("Value column: ").strip()
                group_col = input("Group column: ").strip()
                if value_col in df.columns and group_col in df.columns:
                    df_plot = df[[group_col, value_col]].dropna()
                    fig, ax = plt.subplots(figsize=(8,6))
                    sns.violinplot(x=group_col, y=value_col, data=df_plot, inner='quartile', palette='muted')
                    sns.stripplot(x=group_col, y=value_col, data=df_plot, color='black', alpha=0.5, size=3)
                    ax.set_title('Violin Plot with Quartiles')
                    if len(df_plot[group_col].unique()) >= 2:
                        add_ttest_to_boxplot(df_plot, group_col, value_col, ax)
                    apply_glass_ax(ax)
                    ask_save_plot('violin', pdf)
                    plt.show()
                    return
        # Default data
        ctrl = np.random.normal(5,1,50)
        treat = np.random.normal(7.5,1.2,50)
        df_plot = pd.DataFrame({'Group': ['Ctrl']*50 + ['Treat']*50, 'Value': np.concatenate([ctrl, treat])})
        fig, ax = plt.subplots(figsize=(8,6))
        sns.violinplot(x='Group', y='Value', data=df_plot, inner='quartile', palette='muted')
        sns.stripplot(x='Group', y='Value', data=df_plot, color='black', alpha=0.5, size=3)
        ax.set_title('Violin Plot (Demo)')
        add_ttest_to_boxplot(df_plot, 'Group', 'Value', ax)
        apply_glass_ax(ax)
        ask_save_plot('violin', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def raincloud_plot(pdf=None):
    print("\n--- Raincloud Plot (Box+Violin+Swarm) ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                value_col = input("Value column: ").strip()
                group_col = input("Group column: ").strip()
                if value_col in df.columns and group_col in df.columns:
                    df_plot = df[[group_col, value_col]].dropna()
                    fig, ax = plt.subplots(figsize=(8,6))
                    # Half violin
                    sns.violinplot(x=group_col, y=value_col, data=df_plot, inner=None, palette='muted', alpha=0.5)
                    # Boxplot inside
                    sns.boxplot(x=group_col, y=value_col, data=df_plot, width=0.2, boxprops=dict(alpha=0.5), showfliers=False)
                    # Swarm plot
                    sns.swarmplot(x=group_col, y=value_col, data=df_plot, color='black', alpha=0.6, size=4)
                    ax.set_title('Raincloud Plot')
                    if len(df_plot[group_col].unique()) >= 2:
                        add_ttest_to_boxplot(df_plot, group_col, value_col, ax)
                    apply_glass_ax(ax)
                    ask_save_plot('raincloud', pdf)
                    plt.show()
                    return
        # Default data
        ctrl = np.random.normal(5,1,50)
        treat = np.random.normal(7.5,1.2,50)
        df_plot = pd.DataFrame({'Group': ['Ctrl']*50 + ['Treat']*50, 'Value': np.concatenate([ctrl, treat])})
        fig, ax = plt.subplots(figsize=(8,6))
        sns.violinplot(x='Group', y='Value', data=df_plot, inner=None, palette='muted', alpha=0.5)
        sns.boxplot(x='Group', y='Value', data=df_plot, width=0.2, boxprops=dict(alpha=0.5), showfliers=False)
        sns.swarmplot(x='Group', y='Value', data=df_plot, color='black', alpha=0.6, size=4)
        ax.set_title('Raincloud Plot (Demo)')
        add_ttest_to_boxplot(df_plot, 'Group', 'Value', ax)
        apply_glass_ax(ax)
        ask_save_plot('raincloud', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def ridge_plot(pdf=None):
    print("\n--- Ridge Plot (Overlapping KDEs) ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                value_col = input("Value column: ").strip()
                group_col = input("Group column: ").strip()
                if value_col in df.columns and group_col in df.columns:
                    df_plot = df[[group_col, value_col]].dropna()
                    groups = df_plot[group_col].unique()
                    fig, ax = plt.subplots(figsize=(8,6))
                    for i, grp in enumerate(groups):
                        subset = df_plot[df_plot[group_col] == grp][value_col].dropna()
                        if len(subset) == 0:
                            continue
                        density = stats.gaussian_kde(subset)
                        xs = np.linspace(subset.min(), subset.max(), 200)
                        ys = density(xs) + i
                        ax.fill_between(xs, ys, i, alpha=0.5, label=grp)
                    ax.set_xlabel(value_col)
                    ax.set_ylabel('Density shift')
                    ax.set_title('Ridge Plot')
                    apply_glass_ax(ax)
                    ask_save_plot('ridge', pdf)
                    plt.show()
                    return
        # Default data: two groups
        np.random.seed(42)
        g1 = np.random.normal(5,1,200)
        g2 = np.random.normal(7,1.2,200)
        groups = ['Group1']*200 + ['Group2']*200
        values = np.concatenate([g1,g2])
        df_plot = pd.DataFrame({'Group': groups, 'Value': values})
        fig, ax = plt.subplots(figsize=(8,6))
        uniq = df_plot['Group'].unique()
        for i, grp in enumerate(uniq):
            subset = df_plot[df_plot['Group']==grp]['Value'].dropna()
            density = stats.gaussian_kde(subset)
            xs = np.linspace(subset.min(), subset.max(), 200)
            ys = density(xs) + i
            ax.fill_between(xs, ys, i, alpha=0.5, label=grp)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density shift')
        ax.set_title('Ridge Plot (Demo)')
        apply_glass_ax(ax)
        ask_save_plot('ridge', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def dot_plot(pdf=None):
    print("\n--- Dot Plot (Single-cell style) ---")
    try:
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                # Expecting columns: gene, cluster, pct_exp, avg_exp
                gene_col = input("Gene column: ").strip()
                cluster_col = input("Cluster column: ").strip()
                pct_col = input("Percent expressed column: ").strip()
                exp_col = input("Average expression column: ").strip()
                if all(c in df.columns for c in [gene_col, cluster_col, pct_col, exp_col]):
                    # Create pivot-like plot
                    genes = df[gene_col].unique()
                    clusters = df[cluster_col].unique()
                    pivot_pct = df.pivot(index=gene_col, columns=cluster_col, values=pct_col).fillna(0)
                    pivot_exp = df.pivot(index=gene_col, columns=cluster_col, values=exp_col).fillna(0)
                    fig, ax = plt.subplots(figsize=(len(clusters)*0.8, len(genes)*0.4))
                    for i, gene in enumerate(genes):
                        for j, cl in enumerate(clusters):
                            pct = pivot_pct.loc[gene, cl]
                            exp = pivot_exp.loc[gene, cl]
                            size = 20 + (pct * 80)  # size proportional to percent
                            color = exp
                            ax.scatter(j, i, s=size, c=color, cmap='viridis', vmin=0, vmax=max(pivot_exp.max()))
                    ax.set_yticks(range(len(genes)))
                    ax.set_yticklabels(genes)
                    ax.set_xticks(range(len(clusters)))
                    ax.set_xticklabels(clusters)
                    ax.set_title('Dot Plot')
                    plt.colorbar(ax.collections[0], ax=ax, label='Avg Expression')
                    apply_glass_ax(ax)
                    ask_save_plot('dotplot', pdf)
                    plt.show()
                    return
        print("Using default demo data.")
        # Create default demo data
        genes = ['GeneA', 'GeneB', 'GeneC']
        clusters = ['Cluster1', 'Cluster2', 'Cluster3']
        data = []
        for g in genes:
            for c in clusters:
                pct = np.random.uniform(0,1)
                exp = np.random.uniform(0,3)
                data.append([g,c,pct,exp])
        df = pd.DataFrame(data, columns=['gene','cluster','pct_exp','avg_exp'])
        genes = df['gene'].unique()
        clusters = df['cluster'].unique()
        pivot_pct = df.pivot(index='gene', columns='cluster', values='pct_exp').fillna(0)
        pivot_exp = df.pivot(index='gene', columns='cluster', values='avg_exp').fillna(0)
        fig, ax = plt.subplots(figsize=(len(clusters)*0.8, len(genes)*0.4))
        for i, gene in enumerate(genes):
            for j, cl in enumerate(clusters):
                pct = pivot_pct.loc[gene, cl]
                exp = pivot_exp.loc[gene, cl]
                size = 20 + (pct * 80)
                ax.scatter(j, i, s=size, c=exp, cmap='viridis', vmin=0, vmax=max(pivot_exp.max()))
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes)
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels(clusters)
        ax.set_title('Dot Plot (Demo)')
        plt.colorbar(ax.collections[0], ax=ax, label='Avg Expression')
        apply_glass_ax(ax)
        ask_save_plot('dotplot', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")
# ---------- Mathematical Functions (with session saving) ----------
def sine_plot(pdf=None):
    print("\n--- Sine: y = A * sin(B*x + C) ---")
    try:
        A = safe_float_input("Amplitude A (default 1): ", 1, key='sine_A')
        B = safe_float_input("Frequency B (default 1): ", 1, key='sine_B')
        C = safe_float_input("Phase C (default 0): ", 0, key='sine_C')
        x = np.linspace(0, 2*np.pi, 300)
        y = A * np.sin(B*x + C)
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='blue', linewidth=2, ax=ax)
        ax.set_title(f'Sine: y = {A}·sin({B}x + {C})')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('sine', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def cosine_plot(pdf=None):
    print("\n--- Cosine: y = A * cos(B*x + C) ---")
    try:
        A = safe_float_input("Amplitude A (default 1): ", 1, key='cosine_A')
        B = safe_float_input("Frequency B (default 1): ", 1, key='cosine_B')
        C = safe_float_input("Phase C (default 0): ", 0, key='cosine_C')
        x = np.linspace(0, 2*np.pi, 300)
        y = A * np.cos(B*x + C)
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='green', linewidth=2, ax=ax)
        ax.set_title(f'Cosine: y = {A}·cos({B}x + {C})')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('cosine', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def linear_plot(pdf=None):
    print("\n--- Linear: y = a*x + b ---")
    try:
        a = safe_float_input("Slope a (default 2): ", 2, key='linear_a')
        b = safe_float_input("Intercept b (default 1): ", 1, key='linear_b')
        x = np.linspace(-5, 5, 100)
        y = a*x + b
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='purple', linewidth=2, ax=ax)
        ax.set_title(f'Linear: y = {a}x + {b}')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('linear', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def quadratic_plot(pdf=None):
    print("\n--- Quadratic: y = a*x² + b*x + c ---")
    try:
        a = safe_float_input("a (default 1): ", 1, key='quad_a')
        b = safe_float_input("b (default -3): ", -3, key='quad_b')
        c = safe_float_input("c (default 2): ", 2, key='quad_c')
        x = np.linspace(-5, 5, 100)
        y = a*x**2 + b*x + c
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='red', linewidth=2, ax=ax)
        ax.set_title(f'Quadratic: y = {a}x² + {b}x + {c}')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('quadratic', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def cubic_plot(pdf=None):
    print("\n--- Cubic: y = a*x³ + b*x² + c*x + d ---")
    try:
        a = safe_float_input("a (default 1): ", 1, key='cubic_a')
        b = safe_float_input("b (default -2): ", -2, key='cubic_b')
        c = safe_float_input("c (default 1): ", 1, key='cubic_c')
        d = safe_float_input("d (default 0): ", 0, key='cubic_d')
        x = np.linspace(-4, 4, 100)
        y = a*x**3 + b*x**2 + c*x + d
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='orange', linewidth=2, ax=ax)
        ax.set_title(f'Cubic: y = {a}x³ + {b}x² + {c}x + {d}')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('cubic', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def exponential_plot(pdf=None):
    print("\n--- Exponential: y = a * exp(b*x) + c ---")
    try:
        a = safe_float_input("Scale a (default 1): ", 1, key='exp_a')
        b = safe_float_input("Rate b (default 0.5): ", 0.5, key='exp_b')
        c = safe_float_input("Offset c (default 0): ", 0, key='exp_c')
        x = np.linspace(0, 5, 100)
        y = a * np.exp(b*x) + c
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='teal', linewidth=2, ax=ax)
        ax.set_title(f'Exponential: y = {a}·exp({b}x) + {c}')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('exponential', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def logistic_plot(pdf=None):
    print("\n--- Logistic: y = L / (1 + exp(-k*(x - x0))) ---")
    try:
        L = safe_float_input("Carrying capacity L (default 1): ", 1, key='log_L')
        k = safe_float_input("Growth rate k (default 1): ", 1, key='log_k')
        x0 = safe_float_input("Midpoint x0 (default 0): ", 0, key='log_x0')
        x = np.linspace(-6, 6, 200)
        y = L / (1 + np.exp(-k*(x - x0)))
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='darkred', linewidth=2, ax=ax)
        ax.set_title(f'Logistic: y = {L}/(1+exp(-{k}(x-{x0})))')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('logistic', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

# ---------- Specialized Plots ----------
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
        ask_save_plot('gsea', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

def motif_logo(pdf=None):
    print("\n--- Motif Logo (internal) ---")
    try:
        use_default = input("Use default alignment? (y/n): ").strip().lower()
        seqs = []
        if use_default == 'n':
            print("Enter sequences (ACGT only, one per line, empty line to finish):")
            while True:
                line = input().strip().upper()
                if not line:
                    break
                seqs.append(line)
        if not seqs:
            seqs = ["AAGT", "AAGT", "AAGT", "AAGT", "CAGT", "CAGT", "AACT", "AACT", "ACGT", "ATGT"]
        fig, ax = plt.subplots(figsize=(10,4))
        draw_motif_logo(seqs, ax=ax)
        apply_glass_ax(ax)
        ask_save_plot('motif_logo', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

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
        ask_save_plot('sankey', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

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
                groups = df[group_col].astype(str).values if group_col and group_col in df else ['Sample']*data.shape[0]
                reducer = umap.UMAP(n_components=2, random_state=42)
                embedding = reducer.fit_transform(data)
                fig, ax = plt.subplots(figsize=(8,6))
                scatter = ax.scatter(embedding[:,0], embedding[:,1], c=pd.Categorical(groups).codes, cmap='tab10', alpha=0.7)
                if len(np.unique(groups)) <= 10:
                    legend = ax.legend(*scatter.legend_elements(), title="Groups")
                    ax.add_artist(legend)
                ax.set_title('UMAP Projection')
                apply_glass_ax(ax)
                ask_save_plot('umap', pdf)
                plt.show()
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
        ask_save_plot('umap', pdf)
        plt.show()
    except Exception as e:
        print(f"❌ Error: {e}")

# ---------- Export Utilities ----------
def export_all_to_folder(folder_name="bioplatter_export"):
    global _interrupted
    print(f"\n📁 Exporting all plots to folder: {folder_name}")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    original_format = config['save_format']
    config['save_format'] = 'png'
    original_quiet = config.get('quiet', False)
    config['quiet'] = True
    plot_funcs = [
        volcano_plot, pca_plot, manhattan_plot, ma_plot, venn_diagram,
        barplot_custom, boxplot_custom, heatmap_custom, scatter_custom, timeseries_plot,
        sine_plot, cosine_plot, linear_plot, quadratic_plot, cubic_plot,
        exponential_plot, logistic_plot, gsea_plot, motif_logo, sankey_diagram,
        qq_plot, clustered_heatmap, circos_plot, alignment_viewer, umap_plot,
        violin_plot, raincloud_plot, ridge_plot, dot_plot
    ]
    original_cwd = os.getcwd()
    os.chdir(folder_name)
    def default_input(prompt):
        if "Load data from file?" in prompt or "Show data summary" in prompt:
            return 'n'
        if "Number of sets" in prompt:
            return '2'
        if "Use default" in prompt:
            return 'y'
        if "Save as HTML" in prompt:
            return 'n'
        if "Enter sequences" in prompt:
            return ''
        if "Switch to" in prompt:
            return ''
        if "Correlation type" in prompt:
            return 'pearson'
        return ''
    import builtins
    original_input = builtins.input
    builtins.input = default_input
    global ask_save_plot
    original_ask_save_plot = ask_save_plot
    def auto_save_plot(default_name, pdf=None, story=None):
        try:
            plt.tight_layout()
            plt.savefig(f"{default_name}.png", dpi=config['default_dpi'],
                        bbox_inches='tight', facecolor=plt.rcParams['figure.facecolor'])
            print(f"   ✅ Auto‑saved: {default_name}.png")
        except:
            pass
    ask_save_plot = auto_save_plot
    try:
        for func in tqdm(plot_funcs, desc="Exporting to folder"):
            if _interrupted:
                print("\n⚠️ Export interrupted by user.")
                break
            try:
                func(pdf=None)
                plt.close()
            except Exception as e:
                print(f"⚠️ {func.__name__} failed: {e}")
    finally:
        builtins.input = original_input
        ask_save_plot = original_ask_save_plot
        os.chdir(original_cwd)
        config['save_format'] = original_format
        config['quiet'] = original_quiet
    print(f"✅ All plots saved to folder: {folder_name}")

def generate_markdown_story(plots_list, output_file="story.md"):
    with open(output_file, 'w') as f:
        f.write("# Bio-Platter Pro Analysis Report\n\n")
        f.write("This report was automatically generated.\n\n")
        for i, plot_name in enumerate(plots_list, 1):
            f.write(f"## {i}. {plot_name}\n")
            f.write(f"![{plot_name}]({plot_name}.png)\n\n")
            f.write(f"*Description:* This plot shows {plot_name} analysis.\n\n")
    print(f"📝 Markdown story saved as {output_file}")

# ---------- Help Function ----------
def show_help():
    help_text = """
    📖 Bio‑Platter Pro v11.0 – Interactive Help
    ============================================
    Requirements: pip install matplotlib seaborn numpy pandas scipy scikit-learn tqdm pyarrow feather-format tables umap-learn

    1  Volcano Plot       : log2FC vs -log10(p-value) with FDR line
    2  PCA Plot           : Principal Component Analysis
    3  Manhattan Plot     : GWAS results across chromosomes
    4  MA Plot            : M (log2FC) vs A (mean intensity)
    5  Venn Diagram       : 2 or 3 set intersections (internal)
    6  Barplot            : Grouped bar chart with error bars
    7  Boxplot            : Distribution with swarm + t-test/ANOVA p-value
    8  Heatmap            : Correlation matrix heatmap
    9  Scatter Plot       : XY scatter with regression (Pearson/Spearman)
    10 Time Series        : Line plot over time
    11-17 Mathematical    : Sine, Cosine, Linear, Quadratic, Cubic, Exponential, Logistic
    18 GSEA Plot          : Running enrichment score for ranked gene lists
    19 Motif Logo         : Sequence logo for DNA motifs (internal)
    20 Sankey Diagram     : Flow diagram (internal)
    21 QQ-plot            : Normality check
    22 Clustered Heatmap  : Heatmap with dendrograms
    23 Circos Plot        : Simple circular genomic links
    24 Alignment Viewer   : Colored sequence alignment
    25 UMAP Plot          : Uniform Manifold Approximation (needs umap-learn)
    --- New Plots ---
    26 Violin Plot        : Enhanced distribution with quartiles
    27 Raincloud Plot     : Box+Violin+Swarm combination
    28 Ridge Plot         : Overlapping KDEs for multiple groups
    29 Dot Plot           : Single-cell style dot plot (size = % expressed, color = avg exp)
    ---
    🛠️ Utilities:
    30 Change Theme       : Switch light/dark
    31 PDF Batch          : Export all plots to a single PDF
    32 Folder Export      : Save all plots as PNG in a folder
    33 Markdown Story     : Generate report from exported images
    34 Launch GUI         : Open Tkinter window (if available)
    0  Exit
    """
    print(help_text)

# ---------- CLI Menu ----------
def print_menu(theme_mode):
    pre = '\033[96m' if theme_mode == 'light' else '\033[92m'
    bold = '\033[1m'
    reset = '\033[0m'
    print(pre + bold + "\n" + "▰"*64)
    print("       🧬 BIO‑PLATTER PRO v11.0  |  Ultimate Independent Suite")
    print("▰"*64 + reset)
    print(pre + "\n 🌿 ADVANCED BIOLOGICAL")
    print("   1. Volcano Plot      2. PCA Plot         3. Manhattan Plot")
    print("   4. MA Plot           5. Venn Diagram")
    print(pre + "\n 📊 BASIC BIOLOGICAL")
    print("   6. Barplot           7. Boxplot          8. Heatmap")
    print("   9. Scatter          10. Time Series")
    print(pre + "\n 🧮 MATHEMATICAL")
    print("  11. Sine             12. Cosine          13. Linear")
    print("  14. Quadratic        15. Cubic           16. Exponential")
    print("  17. Logistic")
    print(pre + "\n 🧪 SPECIALIZED")
    print("  18. GSEA Plot        19. Motif Logo      20. Sankey Diagram")
    print(pre + "\n 🆕 ADDITIONAL PLOTS")
    print("  21. QQ-plot          22. Clustered Heatmap")
    print("  23. Circos Plot      24. Alignment Viewer")
    print("  25. UMAP Plot (optional)")
    print(pre + "\n 🌟 NEW PLOTS (v11.0)")
    print("  26. Violin Plot      27. Raincloud Plot")
    print("  28. Ridge Plot       29. Dot Plot")
    print(pre + "\n 🛠️  UTILITIES")
    print("  30. Change Theme     31. Export all to PDF (batch)")
    print("  32. Export all to Folder   33. Generate Markdown Story")
    print("  34. Launch GUI (Tkinter)")
    print("   0. Exit")
    print(pre + "▰"*64 + reset)

def batch_export_to_pdf(pdf_filename="bioplatter_report.pdf"):
    print("\n📄 Batch exporting all plots to PDF...")
    plot_funcs = [
        volcano_plot, pca_plot, manhattan_plot, ma_plot, venn_diagram,
        barplot_custom, boxplot_custom, heatmap_custom, scatter_custom, timeseries_plot,
        sine_plot, cosine_plot, linear_plot, quadratic_plot, cubic_plot,
        exponential_plot, logistic_plot, gsea_plot, motif_logo, sankey_diagram,
        qq_plot, clustered_heatmap, circos_plot, alignment_viewer, umap_plot,
        violin_plot, raincloud_plot, ridge_plot, dot_plot
    ]
    def default_input(prompt):
        if "Load data from file?" in prompt or "Show data summary" in prompt:
            return 'n'
        if "Number of sets" in prompt:
            return '2'
        if "Use default" in prompt:
            return 'y'
        if "Save as HTML" in prompt:
            return 'n'
        if "Correlation type" in prompt:
            return 'pearson'
        return ''
    import builtins
    original_input = builtins.input
    builtins.input = default_input
    try:
        with PdfPages(pdf_filename) as pdf:
            if HAS_TQDM:
                iterator = tqdm(plot_funcs, desc="Generating PDF")
            else:
                iterator = plot_funcs
            for func in iterator:
                try:
                    func(pdf=pdf)
                    plt.close()
                except:
                    pass
        print(f"✅ PDF saved: {pdf_filename}")
    except Exception as e:
        print(f"❌ Batch failed: {e}")
    finally:
        builtins.input = original_input

# ---------- GUI (Tkinter) – clean version, no duplicate functions ----------
if HAS_TK:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
    import builtins

    def center_window(win, width, height):
        win.update_idletasks()
        x = (win.winfo_screenwidth() // 2) - (width // 2)
        y = (win.winfo_screenheight() // 2) - (height // 2)
        win.geometry(f'{width}x{height}+{x}+{y}')

    def get_math_formula(plot_id):
        formulas = {
            'sine': 'y = A · sin(B·x + C)',
            'cosine': 'y = A · cos(B·x + C)',
            'linear': 'y = a·x + b',
            'quadratic': 'y = a·x² + b·x + c',
            'cubic': 'y = a·x³ + b·x² + c·x + d',
            'exponential': 'y = a · exp(b·x) + c',
            'logistic': 'y = L / (1 + exp(-k·(x - x₀)))'
        }
        return formulas.get(plot_id, "")

    def show_math_dialog(parent, prompt, default, formula):
        dlg = tk.Toplevel(parent)
        dlg.title("Parameter Input")
        center_window(dlg, 520, 240)
        dlg.configure(bg='#1e1e2e')
        dlg.transient(parent)
        dlg.grab_set()
        if formula:
            tk.Label(dlg, text=formula, font=('Segoe UI', 12, 'bold'), bg='#1e1e2e', fg='#00ff99').pack(pady=15)
        tk.Label(dlg, text=prompt.strip(), bg='#1e1e2e', fg='white', wraplength=450).pack(pady=5)
        entry = tk.Entry(dlg, bg='#353545', fg='white', font=('Segoe UI', 10), width=30)
        entry.pack(pady=10)
        if default: entry.insert(0, default)
        entry.focus_set()
        result = ""
        def ok():
            nonlocal result
            result = entry.get().strip()
            dlg.destroy()
        def cancel():
            nonlocal result
            result = ""
            dlg.destroy()
        btn_frame = tk.Frame(dlg, bg='#1e1e2e')
        btn_frame.pack(pady=15)
        tk.Button(btn_frame, text="OK", command=ok, bg='#00ff99', fg='black', padx=15).pack(side='left', padx=10)
        tk.Button(btn_frame, text="Cancel", command=cancel, bg='#e74c3c', fg='white', padx=15).pack(side='left', padx=10)
        parent.wait_window(dlg)
        return result

    def run_gui():
        if not HAS_TK:
            print("Tkinter not available. Please install python3-tk.")
            return
        root = tk.Tk()
        root.title("Bio-Platter Pro v11.0 - Graphical Interface")
        root.geometry("950x720")
        root.configure(bg='#1e1e2e')
        center_window(root, 950, 720)
        root.minsize(800, 600)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', background='#3c3c5c', foreground='white')
        style.configure('TLabel', background='#1e1e2e', foreground='white')

        header = tk.Frame(root, bg='#2a2a3e', height=90)
        header.pack(fill='x')
        header.pack_propagate(False)
        tk.Label(header, text='🧬 BIO-PLATTER PRO v11.0', font=('Segoe UI', 20, 'bold'), bg='#2a2a3e', fg='#00ff99').pack(pady=10)
        tk.Label(header, text='Ultimate Independent Scientific Plotting Suite', font=('Segoe UI', 10), bg='#2a2a3e', fg='#aaaaaa').pack()

        main = tk.Frame(root, bg='#1e1e2e')
        main.pack(fill='both', expand=True, padx=15, pady=10)

        left = tk.Frame(main, bg='#252535', width=240)
        left.pack(side='left', fill='y', padx=(0,10))
        left.pack_propagate(False)
        tk.Label(left, text='🔍 Search', font=('Segoe UI', 10, 'bold'), bg='#252535', fg='#00ff99').pack(pady=(10,5))
        search_entry = tk.Entry(left, bg='#353545', fg='white', relief='flat')
        search_entry.pack(padx=10, pady=(0,10), fill='x')
        tk.Label(left, text='📂 Categories', font=('Segoe UI', 10, 'bold'), bg='#252535', fg='#00ff99').pack(pady=(5,5))
        cat_box = tk.Listbox(left, bg='#353545', fg='white', relief='flat', height=8, selectbackground='#00ff99', selectforeground='#1e1e2e')
        cat_box.pack(padx=10, pady=(0,10), fill='both', expand=True)
        categories = ['All Plots', 'Advanced Biological', 'Basic Biological', 'Mathematical', 'Specialized', 'Additional', 'New Plots']
        for c in categories: cat_box.insert('end', c)
        cat_box.selection_set(0)

        right = tk.Frame(main, bg='#1e1e2e')
        right.pack(side='right', fill='both', expand=True)
        count_label = tk.Label(right, text='📊 Available: 0', bg='#1e1e2e', fg='#aaaaaa', anchor='w')
        count_label.pack(fill='x')
        list_frame = tk.Frame(right, bg='#1e1e2e')
        list_frame.pack(fill='both', expand=True)
        scroll = tk.Scrollbar(list_frame)
        scroll.pack(side='right', fill='y')
        plot_list = tk.Listbox(list_frame, yscrollcommand=scroll.set, bg='#2a2a3e', fg='#e0e0e0',
                               selectbackground='#00ff99', selectforeground='#1e1e2e',
                               font=('Consolas', 10), relief='flat')
        plot_list.pack(side='left', fill='both', expand=True)
        scroll.config(command=plot_list.yview)

        plots_data = [
            ('🌋 Volcano Plot', 'volcano', 'Advanced Biological'),
            ('📊 PCA Plot', 'pca', 'Advanced Biological'),
            ('🧬 Manhattan Plot', 'manhattan', 'Advanced Biological'),
            ('📈 MA Plot', 'ma', 'Advanced Biological'),
            ('🔄 Venn Diagram', 'venn', 'Advanced Biological'),
            ('📊 Barplot', 'barplot', 'Basic Biological'),
            ('📦 Boxplot', 'boxplot', 'Basic Biological'),
            ('🔥 Heatmap', 'heatmap', 'Basic Biological'),
            ('⚫ Scatter Plot', 'scatter', 'Basic Biological'),
            ('📉 Time Series', 'timeseries', 'Basic Biological'),
            ('📐 Sine', 'sine', 'Mathematical'),
            ('📐 Cosine', 'cosine', 'Mathematical'),
            ('📐 Linear', 'linear', 'Mathematical'),
            ('📐 Quadratic', 'quadratic', 'Mathematical'),
            ('📐 Cubic', 'cubic', 'Mathematical'),
            ('📐 Exponential', 'exponential', 'Mathematical'),
            ('📐 Logistic', 'logistic', 'Mathematical'),
            ('🧬 GSEA Plot', 'gsea', 'Specialized'),
            ('🧬 Motif Logo', 'motif', 'Specialized'),
            ('🧬 Sankey Diagram', 'sankey', 'Specialized'),
            ('📊 QQ-plot', 'qq', 'Additional'),
            ('🔥 Clustered Heatmap', 'clustered_heatmap', 'Additional'),
            ('🎯 Circos Plot', 'circos', 'Additional'),
            ('🔬 Alignment Viewer', 'alignment', 'Additional'),
            ('🗺️ UMAP Plot', 'umap', 'Additional'),
            ('🎻 Violin Plot', 'violin', 'New Plots'),
            ('☁️ Raincloud Plot', 'raincloud', 'New Plots'),
            ('🏔️ Ridge Plot', 'ridge', 'New Plots'),
            ('🔘 Dot Plot', 'dotplot', 'New Plots')
        ]
        for name, _, _ in plots_data: plot_list.insert('end', name)

        plot_funcs = {
            'volcano': volcano_plot, 'pca': pca_plot, 'manhattan': manhattan_plot,
            'ma': ma_plot, 'venn': venn_diagram, 'barplot': barplot_custom,
            'boxplot': boxplot_custom, 'heatmap': heatmap_custom, 'scatter': scatter_custom,
            'timeseries': timeseries_plot, 'sine': sine_plot, 'cosine': cosine_plot,
            'linear': linear_plot, 'quadratic': quadratic_plot, 'cubic': cubic_plot,
            'exponential': exponential_plot, 'logistic': logistic_plot, 'gsea': gsea_plot,
            'motif': motif_logo, 'sankey': sankey_diagram, 'qq': qq_plot,
            'clustered_heatmap': clustered_heatmap, 'circos': circos_plot,
            'alignment': alignment_viewer, 'umap': umap_plot,
            'violin': violin_plot, 'raincloud': raincloud_plot, 'ridge': ridge_plot, 'dotplot': dot_plot
        }

        def update_list():
            term = search_entry.get().strip().lower()
            sel_idx = cat_box.curselection()
            sel_cat = cat_box.get(sel_idx[0]) if sel_idx else 'All Plots'
            plot_list.delete(0, 'end')
            cnt = 0
            for name, pid, cat in plots_data:
                if (sel_cat == 'All Plots' or cat == sel_cat) and (term == '' or term in name.lower()):
                    plot_list.insert('end', name); cnt += 1
            count_label.config(text=f'📊 Available: {cnt}')
        search_entry.bind('<KeyRelease>', lambda e: update_list())
        cat_box.bind('<<ListboxSelect>>', lambda e: update_list())
        update_list()

        def run_plot_with_gui(plot_id):
            original_input = builtins.input
            _last_df = None
            def gui_input(prompt):
                nonlocal _last_df
                if 'Load data from file?' in prompt:
                    return 'y' if messagebox.askyesno("Confirm", prompt) else 'n'
                if 'Show data summary' in prompt or 'Use default' in prompt:
                    return 'y' if messagebox.askyesno("Confirm", prompt) else 'n'
                if 'Save this plot?' in prompt or 'Save as HTML' in prompt:
                    return 'y' if messagebox.askyesno("Save", prompt) else 'n'
                if 'Correlation type' in prompt:
                    res = simpledialog.askstring("Correlation", "Enter 'pearson' or 'spearman'", initialvalue='pearson')
                    return res if res else 'pearson'
                if 'File path' in prompt:
                    path = filedialog.askopenfilename(title=prompt, filetypes=[("CSV","*.csv"),("Excel","*.xlsx")])
                    if path:
                        try:
                            if path.endswith('.csv'): _last_df = pd.read_csv(path)
                            else: _last_df = pd.read_excel(path)
                        except Exception as e:
                            messagebox.showerror("Error", str(e))
                    return path if path else ''
                if 'column' in prompt.lower():
                    if _last_df is not None:
                        cols = list(_last_df.columns)
                        dlg = tk.Toplevel(root)
                        dlg.title("Select Column")
                        center_window(dlg, 450, 180)
                        dlg.configure(bg='#1e1e2e')
                        dlg.transient(root)
                        dlg.grab_set()
                        tk.Label(dlg, text=prompt, bg='#1e1e2e', fg='white').pack(pady=15)
                        combo = ttk.Combobox(dlg, values=cols, state='readonly', width=35)
                        combo.pack(pady=10)
                        result = ""
                        def ok():
                            nonlocal result
                            result = combo.get()
                            dlg.destroy()
                        tk.Button(dlg, text="OK", command=ok, bg='#00ff99', fg='black').pack(pady=10)
                        root.wait_window(dlg)
                        return result
                    else:
                        return simpledialog.askstring("Column Name", prompt)
                default = ''
                import re
                match = re.search(r'\(default ([^)]+)\)', prompt)
                if match: default = match.group(1)
                formula = get_math_formula(plot_id)
                return show_math_dialog(root, prompt, default, formula)
            builtins.input = gui_input
            try:
                func = plot_funcs.get(plot_id)
                if func: func()
                else: messagebox.showerror("Error", f"Plot '{plot_id}' not found.")
            except Exception as e:
                messagebox.showerror("Plot Error", str(e))
            finally:
                builtins.input = original_input

        def on_generate():
            sel = plot_list.curselection()
            if not sel:
                messagebox.showwarning("No Selection", "Please select a plot.")
                return
            plot_name = plot_list.get(sel[0])
            plot_id = next(p[1] for p in plots_data if p[0] == plot_name)
            run_plot_with_gui(plot_id)

        plot_list.bind('<Double-Button-1>', lambda e: on_generate())

        btn_frame = tk.Frame(right, bg='#1e1e2e')
        btn_frame.pack(pady=15)
        tk.Button(btn_frame, text='🎨 Generate Plot', command=on_generate, bg='#00ff99', fg='#1e1e2e', font=('Segoe UI',11,'bold'), padx=20).pack(side='left', padx=5)
        tk.Button(btn_frame, text='❓ Help', command=lambda: show_help_dialog(root), bg='#3c3c5c', fg='white', padx=20).pack(side='left', padx=5)
        tk.Button(btn_frame, text='📁 Export All', command=lambda: export_all_dialog(root), bg='#e67e22', fg='white', padx=20).pack(side='left', padx=5)
        tk.Button(btn_frame, text='✖ Exit', command=root.destroy, bg='#e74c3c', fg='white', padx=20).pack(side='left', padx=5)

        status = tk.Frame(root, bg='#2a2a3e', height=35)
        status.pack(fill='x', side='bottom')
        status.pack_propagate(False)
        libs = f"Libraries: tqdm {'✓' if HAS_TQDM else '✗'}  PyArrow {'✓' if HAS_PARQUET else '✗'}  Feather {'✓' if HAS_FEATHER else '✗'}  HDF5 {'✓' if HAS_HDF5 else '✗'}  UMAP {'✓' if HAS_UMAP else '✗'}"
        tk.Label(status, text=libs, bg='#2a2a3e', fg='#aaaaaa', font=('Segoe UI',8)).pack(pady=8)

        root.bind('<Return>', lambda e: on_generate())
        root.bind('<Escape>', lambda e: root.destroy())
        root.mainloop()

    def show_help_dialog(parent):
        win = tk.Toplevel(parent)
        win.title("Bio-Platter Pro v11.0 - Help")
        center_window(win, 800, 600)
        win.configure(bg='#1e1e2e')
        text = tk.Text(win, bg='#2a2a3e', fg='#e0e0e0', wrap='word', font=('Consolas', 9))
        text.pack(fill='both', expand=True, padx=10, pady=10)
        help_str = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    BIO-PLATTER PRO v11.0 - ULTIMATE HELP GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT: Install requirements: pip install matplotlib seaborn numpy pandas scipy scikit-learn tqdm pyarrow feather-format tables umap-learn

All plots are fully graphical. You will be asked for parameters via popup dialogs.
No terminal input is required.

📊 PLOT TYPES (30 plots total)
─────────────────────────────────────────────────────────────────────────────
1.  Volcano Plot       : log2FC vs -log10(p-value) with FDR line
2.  PCA Plot           : Principal Component Analysis (2D projection)
3.  Manhattan Plot     : GWAS results across chromosomes
4.  MA Plot            : M (log2FC) vs A (mean intensity)
5.  Venn Diagram       : 2 or 3 set intersections (self-contained)
6.  Barplot            : Grouped bar chart with error bars
7.  Boxplot            : Distribution with swarm + t-test/ANOVA p-value
8.  Heatmap            : Correlation matrix heatmap
9.  Scatter Plot       : XY scatter with regression (Pearson/Spearman)
10. Time Series        : Line plot over time
11. Sine               : y = A·sin(B·x + C)
12. Cosine             : y = A·cos(B·x + C)
13. Linear             : y = a·x + b
14. Quadratic          : y = a·x² + b·x + c
15. Cubic              : y = a·x³ + b·x² + c·x + d
16. Exponential        : y = a·exp(b·x) + c
17. Logistic           : y = L/(1+exp(-k·(x-x₀)))
18. GSEA Plot          : Running enrichment score for ranked gene lists
19. Motif Logo         : Sequence logo (self-contained)
20. Sankey Diagram     : Flow diagram (self-contained)
21. QQ-plot            : Normality check
22. Clustered Heatmap  : Heatmap with dendrograms
23. Circos Plot        : Circular genomic links
24. Alignment Viewer   : Colored sequence alignment
25. UMAP Plot          : Advanced dimensionality reduction (needs umap-learn)
--- NEW in v11.0 ---
26. Violin Plot        : Enhanced distribution with quartiles + statistical test
27. Raincloud Plot     : Combination of box, violin, and swarm
28. Ridge Plot         : Overlapping KDEs for multiple groups
29. Dot Plot           : Single-cell style dot plot (size = % expressed, color = avg exp)

HOW TO USE:
─────────────────────────────────────────────────────────────────────────────
1. Double-click a plot or press "Generate Plot"
2. Answer the popup questions:
   - Yes/No: click Yes or No
   - File: click Browse and select CSV/Excel
   - Column: choose from dropdown (after loading file)
   - Coefficients: enter number (leave empty for default)
3. The plot will appear in a new window.

💡 TIPS:
─────────────────────────────────────────────────────────────────────────────
- Leave input empty to use default value.
- For biological plots, answer "No" to file question to use demo data.
- Press Escape to exit the main window.
- Export All to Folder saves all 30 plots as PNG.

No internet connection required. All plots are self-contained.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        text.insert('1.0', help_str)
        text.config(state='disabled')
        tk.Button(win, text="Close", command=win.destroy, bg='#3c3c5c', fg='white', padx=15).pack(pady=10)

    def export_all_dialog(parent):
        folder = filedialog.askdirectory(title="Select Export Folder")
        if folder:
            try:
                export_all_to_folder(folder)
                messagebox.showinfo("Export Complete", f"All plots saved to:\n{folder}")
            except Exception as e:
                messagebox.showerror("Export Failed", str(e))

# ---------- Main Function ----------
def main():
    global _interaction_counter
    parser = argparse.ArgumentParser(description="Bio-Platter Pro v11.0 - Ultimate Independent Suite")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--pdf", type=str, default="bioplatter_report.pdf")
    parser.add_argument("--config", action="store_true")
    parser.add_argument("--svg", action="store_true")
    parser.add_argument("--pdf-output", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--export-folder", type=str)
    parser.add_argument("--story", action="store_true")
    args = parser.parse_args()

    if args.config:
        new_theme = input("Theme (light/dark): ").strip()
        if new_theme in ['light','dark']:
            config['theme'] = new_theme
        new_dpi = input(f"DPI [{config['default_dpi']}]: ").strip()
        if new_dpi.isdigit():
            config['default_dpi'] = int(new_dpi)
        config['quiet'] = args.quiet or False
        save_config(config)
        print("Config saved.")
        return

    if args.gui:
        if HAS_TK:
            run_gui()
        else:
            print("Tkinter not available. Please install python3-tk.")
        return

    if args.export_folder:
        export_all_to_folder(args.export_folder)
        if args.story:
            plot_names = ["volcano", "pca", "manhattan", "maplot", "venn",
                         "barplot", "boxplot", "heatmap", "scatter", "timeseries",
                         "sine", "cosine", "linear", "quadratic", "cubic", "exponential", "logistic",
                         "gsea", "motif_logo", "sankey",
                         "qqplot", "clustered_heatmap", "circos", "alignment", "umap",
                         "violin", "raincloud", "ridge", "dotplot"]
            generate_markdown_story(plot_names, os.path.join(args.export_folder, "story.md"))
        return

    if args.quiet:
        config['quiet'] = True

    if args.svg:
        config['save_format'] = 'svg'
    elif args.pdf_output:
        config['save_format'] = 'pdf'
    else:
        config['save_format'] = 'png'
    if args.interactive:
        config['interactive'] = True

    set_theme(config['theme'])
    print(f"✅ Theme: {config['theme']}, Format: {config['save_format']}, Interactive: {config['interactive']}, Quiet: {config['quiet']}\n")

    if args.batch:
        batch_export_to_pdf(args.pdf)
        return

    # Interactive menu loop with Ctrl+C and 'h' help
    while True:
        try:
            print_menu(config['theme'])
            choice = input("🔹 Select (0-34, h for help): ").strip().lower()
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == 'h' or choice == 'help':
                show_help()
            elif choice == '1': volcano_plot()
            elif choice == '2': pca_plot()
            elif choice == '3': manhattan_plot()
            elif choice == '4': ma_plot()
            elif choice == '5': venn_diagram()
            elif choice == '6': barplot_custom()
            elif choice == '7': boxplot_custom()
            elif choice == '8': heatmap_custom()
            elif choice == '9': scatter_custom()
            elif choice == '10': timeseries_plot()
            elif choice == '11': sine_plot()
            elif choice == '12': cosine_plot()
            elif choice == '13': linear_plot()
            elif choice == '14': quadratic_plot()
            elif choice == '15': cubic_plot()
            elif choice == '16': exponential_plot()
            elif choice == '17': logistic_plot()
            elif choice == '18': gsea_plot()
            elif choice == '19': motif_logo()
            elif choice == '20': sankey_diagram()
            elif choice == '21': qq_plot()
            elif choice == '22': clustered_heatmap()
            elif choice == '23': circos_plot()
            elif choice == '24': alignment_viewer()
            elif choice == '25': umap_plot()
            elif choice == '26': violin_plot()
            elif choice == '27': raincloud_plot()
            elif choice == '28': ridge_plot()
            elif choice == '29': dot_plot()
            elif choice == '30':
                new_theme = input("Switch to (light/dark): ").strip()
                if new_theme in ['light','dark']:
                    config['theme'] = new_theme
                    save_config(config)
                    set_theme(new_theme)
                    print(f"Theme changed to {new_theme}.")
            elif choice == '31':
                pdf_file = input("PDF filename [bioplatter_report.pdf]: ").strip()
                if not pdf_file:
                    pdf_file = "bioplatter_report.pdf"
                batch_export_to_pdf(pdf_file)
            elif choice == '32':
                folder = input("Folder name [bioplatter_export]: ").strip()
                if not folder:
                    folder = "bioplatter_export"
                export_all_to_folder(folder)
            elif choice == '33':
                folder = input("Folder containing images [bioplatter_export]: ").strip()
                if not folder:
                    folder = "bioplatter_export"
                plot_names = ["volcano", "pca", "manhattan", "maplot", "venn",
                             "barplot", "boxplot", "heatmap", "scatter", "timeseries",
                             "sine", "cosine", "linear", "quadratic", "cubic", "exponential", "logistic",
                             "gsea", "motif_logo", "sankey",
                             "qqplot", "clustered_heatmap", "circos", "alignment", "umap",
                             "violin", "raincloud", "ridge", "dotplot"]
                generate_markdown_story(plot_names, os.path.join(folder, "story.md"))
            elif choice == '34':
                if HAS_TK:
                    run_gui()
                else:
                    print("Tkinter not available.")
            else:
                print("❌ Invalid choice. Type 'h' for help.")
            _interaction_counter += 1
            if _interaction_counter % 5 == 0:
                autosave_session()
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user. Returning to menu...")
            continue
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue

if __name__ == "__main__":
    main()
