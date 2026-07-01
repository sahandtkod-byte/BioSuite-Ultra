"""Utility functions: config, session, safe input, file loading, theme, helpers."""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def get_app_dir():
    """Get the application directory, works with PyInstaller."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APP_DIR = get_app_dir()

def center_window(win, width, height):
    win.update_idletasks()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry(f'{width}x{height}+{x}+{y}')

_interrupted = False
_interaction_counter = 0

DEFAULT_CONFIG = {
    "theme": "dark-green",
    "default_dpi": 180,
    "save_format": "png",
    "interactive": False,
    "downsample_threshold": 5000,
    "quiet": False
}

CONFIG_FILE = os.path.join(APP_DIR, "bioplatter_config.json")
SESSION_FILE = os.path.join(APP_DIR, "bioplatter_session.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

def load_session():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_session(session_data):
    try:
        with open(SESSION_FILE, 'w') as f:
            json.dump(session_data, f, indent=2)
    except Exception:
        pass

def autosave_session(session_data=None):
    if session_data is None:
        session_data = session
    save_session(session_data)

config = load_config()
session = load_session()

def set_theme(choice):
    is_dark = choice in ('dark', 'dark-green', 'dark-purple')
    if not is_dark:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            try:
                plt.style.use('seaborn-whitegrid')
            except Exception:
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
        except Exception:
            try:
                plt.style.use('seaborn-darkgrid')
            except Exception:
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
                autosave_session(session)
        return val
    except Exception:
        print(f"Invalid. Using default: {default}")
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
                autosave_session(session)
        return val
    except Exception:
        print(f"Invalid. Using default: {default}")
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
                autosave_session(session)
        return items
    except Exception:
        print("Invalid list. Using default.")
        return None

def load_dataframe_safe(filepath):
    global _interrupted
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath)
        elif ext == '.json':
            df = pd.read_json(filepath)
        elif ext == '.txt':
            df = pd.read_csv(filepath, sep=None, engine='python')
        elif ext == '.parquet':
            df = pd.read_parquet(filepath)
        elif ext in ['.h5', '.hdf5']:
            df = pd.read_hdf(filepath)
        else:
            df = pd.read_csv(filepath, sep=None, engine='python')
        if df.empty:
            print("File is empty.")
            return None
        if not config.get('quiet', False):
            try:
                show_stats = input("Show data summary (describe)? (y/n): ").strip().lower()
                if show_stats == 'y':
                    print(df.describe())
            except EOFError:
                pass
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def maybe_downsample(x, y, max_points=None):
    if max_points is None:
        max_points = config.get('downsample_threshold', 5000)
    if len(x) <= max_points:
        return x, y
    indices = np.random.choice(len(x), max_points, replace=False)
    print(f"Downsampled from {len(x)} to {max_points} points.")
    return x[indices], y[indices]

def apply_glass_ax(ax):
    try:
        rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                              transform=ax.transAxes, clip_on=False,
                              facecolor='none', edgecolor=ax.spines['top'].get_edgecolor(),
                              linewidth=0.8, alpha=0.3)
        ax.add_patch(rect)
    except Exception:
        pass

def ask_save_plot(default_name, save_format, dpi, pdf=None, story=None):
    if pdf is not None:
        try:
            plt.tight_layout()
            pdf.savefig()
            print(f"   Added to PDF: {default_name}")
            if story is not None:
                story.append(f"![{default_name}]({default_name}.png)")
        except Exception:
            pass
        return
    try:
        save = input("Save this plot? (y/n): ").strip().lower()
    except EOFError:
        save = 'n'
    if save == 'y':
        name = input(f"   Filename (default {default_name}.{save_format}): ").strip()
        if not name:
            name = default_name
        try:
            plt.tight_layout()
            plt.savefig(f"{name}.{save_format}", dpi=dpi,
                        bbox_inches='tight', facecolor=plt.rcParams['figure.facecolor'])
            print(f"   Saved as {name}.{save_format}")
        except Exception:
            print("Save failed.")

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
    print("\nBoxplot Statistics:")
    for group, st in stats_dict.items():
        print(f"   {group}: median={st['median']:.2f}, IQR={st['iqr']:.2f}, outliers={st['outliers']}")
    return stats_dict

def report_scatter_stats(x, y):
    from scipy import stats as sp_stats
    if len(x) < 2:
        print("Insufficient data for correlation.")
        return
    corr, pval = sp_stats.pearsonr(x, y)
    print(f"\nScatter Statistics: Pearson r = {corr:.3f}, p-value = {pval:.4f}")
    return {'r': corr, 'p': pval}

def report_volcano_stats(lfc, pvals, fc_thresh, p_thresh):
    sig = (np.abs(lfc) >= fc_thresh) & (pvals < p_thresh)
    up = sig & (lfc > 0)
    down = sig & (lfc < 0)
    print(f"\nVolcano Statistics:")
    print(f"   Up-regulated genes: {np.sum(up)}")
    print(f"   Down-regulated genes: {np.sum(down)}")
    print(f"   Total significant: {np.sum(sig)}")
    return {'up': int(np.sum(up)), 'down': int(np.sum(down))}

def report_pca_stats(pca):
    print(f"\nPCA Statistics:")
    for i, var in enumerate(pca.explained_variance_ratio_[:2]):
        print(f"   PC{i+1} explains {var*100:.2f}% of variance")
    return {'var1': pca.explained_variance_ratio_[0], 'var2': pca.explained_variance_ratio_[1]}

def report_manhattan_stats(df, threshold=5e-8):
    if 'p' not in df.columns:
        return
    sig = df[df['p'] > -np.log10(threshold)]
    if not sig.empty:
        top = sig.loc[sig['p'].idxmax()]
        print(f"\nManhattan Statistics:")
        print(f"   Significant SNPs: {len(sig)}")
        print(f"   Top SNP: {top['chrom']}:{int(top['pos'])} with -log10(p)={top['p']:.2f}")
    else:
        print("\nNo SNPs reached genome-wide significance.")

def add_ttest_to_boxplot(data, group_col, value_col, ax):
    from scipy import stats as sp_stats
    groups = data[group_col].unique()
    if len(groups) == 2:
        g1 = data[data[group_col]==groups[0]][value_col].dropna()
        g2 = data[data[group_col]==groups[1]][value_col].dropna()
        _, pval = sp_stats.ttest_ind(g1, g2)
        y_max = data[value_col].max() * 1.05
        ax.text(0.5, y_max, f't-test p = {pval:.4f}', ha='center',
                transform=ax.transAxes, fontsize=9)
    else:
        groups_data = [data[data[group_col]==g][value_col].dropna().values for g in groups]
        _, pval = sp_stats.f_oneway(*groups_data)
        ax.text(0.5, data[value_col].max()*1.05, f'ANOVA p = {pval:.4f}', ha='center',
                transform=ax.transAxes, fontsize=9)

def add_regression_eq(x, y, ax):
    from scipy import stats as sp_stats
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
    eq = f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}, p = {p_value:.4f}'
    ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7))
