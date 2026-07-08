"""Utility functions: config, session, safe input, file loading, theme, helpers."""
import os
import sys
import json
import subprocess
import warnings
import threading
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Re-export validators ────────────────────────────────────────────────────
from biosuite.core.validators import (
    validate_sequence,
    validate_file_extension,
    validate_dataframe_columns,
    validate_range,
    retry_on_error,
    safe_execute,
    InputValidator,
    default_validator,
)


class PerformanceWarning(UserWarning):
    """Warning emitted when a pure-Python fallback is used instead of an external tool."""
    pass


class FormatWarning(UserWarning):
    """Warning for file format issues or deprecated usage."""
    pass


# ── Threading Utilities ──────────────────────────────────────────────────────

def run_in_background(func, *args, callback=None, **kwargs):
    """Run a function in a background thread with optional callback.

    Args:
        func: function to run.
        *args: positional arguments.
        callback: function to call with result when done (called in main thread).
        **kwargs: keyword arguments.

    Returns:
        threading.Thread object.
    """
    def _wrapper():
        try:
            result = func(*args, **kwargs)
            if callback:
                callback(result)
        except Exception as e:
            if callback:
                callback(None)
            print(f"Background task error: {e}")

    thread = threading.Thread(target=_wrapper, daemon=True)
    thread.start()
    return thread


def run_with_progress(func, progress_callback=None, *args, **kwargs):
    """Run a function with progress updates.

    Args:
        func: function to run (must accept progress_callback kwarg).
        progress_callback: function to call with progress (0.0 to 1.0).
        *args: positional arguments.
        **kwargs: keyword arguments.

    Returns:
        Result of func.
    """
    return func(*args, progress_callback=progress_callback, **kwargs)


class CachedResult:
    """Cache function results to avoid recomputation."""

    def __init__(self, func, maxsize=128, ttl: float = None):
        """Initialize CachedResult.

        Args:
            func: Function to cache.
            maxsize: Maximum number of cached entries.
            ttl: Time-to-live in seconds. Entries older than this are evicted
                 on access. None means no expiration.
        """
        self.func = func
        self.cache = {}
        self.maxsize = maxsize
        self.access_order = []
        self.ttl = ttl  # seconds, or None for no expiration

    def __call__(self, *args, **kwargs):
        """Call the wrapped function, using cache if available."""
        key = str(args) + str(kwargs)
        now = time.monotonic()
        if key in self.cache:
            result, created_at = self.cache[key]
            if self.ttl is not None and (now - created_at) > self.ttl:
                # Entry expired — evict it
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
            else:
                return result
        # Auto-clean all expired entries on access
        self._evict_expired(now)
        result = self.func(*args, **kwargs)
        if len(self.cache) >= self.maxsize:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        self.cache[key] = (result, now)
        self.access_order.append(key)
        return result

    def is_expired(self, key: str) -> bool:
        """Check if a specific cache entry has expired.

        Args:
            key: The cache key to check.

        Returns:
            True if the entry doesn't exist or has expired; False otherwise.
        """
        if key not in self.cache:
            return True
        if self.ttl is None:
            return False
        _, created_at = self.cache[key]
        return (time.monotonic() - created_at) > self.ttl

    def _evict_expired(self, now: float = None):
        """Remove all expired entries from the cache."""
        if self.ttl is None or not self.cache:
            return
        if now is None:
            now = time.monotonic()
        expired_keys = [
            key for key, (_, created_at) in self.cache.items()
            if (now - created_at) > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)

    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.access_order.clear()

    def __len__(self):
        return len(self.cache)

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
    "quiet": False,
    "api_keys": {
        "ncbi_email": "",
        "ncbi_api_key": "",
        "uniprot_email": "",
        "kegg_email": "",
        "alphafold_email": "",
    }
}

CONFIG_FILE = os.path.join(APP_DIR, "biosuite_config.json")
SESSION_FILE = os.path.join(APP_DIR, "biosuite_session.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except (json.JSONDecodeError, OSError):
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass

def load_session():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

def save_session(session_data):
    try:
        with open(SESSION_FILE, 'w') as f:
            json.dump(session_data, f, indent=2)
    except OSError:
        pass

def autosave_session(session_data=None):
    if session_data is None:
        session_data = session
    save_session(session_data)

config = load_config()
session = load_session()


# ── API Key Management ──────────────────────────────────────────────────────

def get_api_key(service):
    """Get API key for a service (ncbi, uniprot, kegg, alphafold, etc.)."""
    keys = config.get('api_keys', {})
    return keys.get(service, '')

def set_api_key(service, key):
    """Save API key for a service."""
    if 'api_keys' not in config:
        config['api_keys'] = {}
    config['api_keys'][service] = key
    save_config(config)

def prompt_api_key(service, description=""):
    """Prompt user to enter API key if not already set."""
    existing = get_api_key(service)
    if existing:
        return existing
    if config.get('quiet', False):
        return ''
    print(f"\n  API key required for: {service}")
    if description:
        print(f"  {description}")
    key = input(f"  Enter API key (or blank to skip): ").strip()
    if key:
        set_api_key(service, key)
        print(f"  API key saved for {service}.")
    return key

def set_theme(choice):
    is_dark = choice in ('dark', 'dark-green', 'dark-purple')
    if not is_dark:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
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
        except OSError:
            try:
                plt.style.use('seaborn-darkgrid')
            except OSError:
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
    except (ValueError, TypeError):
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
    except (ValueError, TypeError):
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
    except (ValueError, TypeError):
        print("Invalid list. Using default.")
        return None

def load_dataframe_safe(filepath):
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
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def load_dataframe_safe_interactive(filepath):
    """Load dataframe and optionally show summary stats (interactive use only)."""
    df = load_dataframe_safe(filepath)
    if df is not None and not config.get('quiet', False):
        try:
            show_stats = input("Show data summary (describe)? (y/n): ").strip().lower()
            if show_stats == 'y':
                print(df.describe())
        except EOFError:
            pass
    return df

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
    except (AttributeError, TypeError, ValueError):
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
        except OSError:
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
        y_max = data[value_col].max()
        ax.text(0.5, 1.02, f't-test p = {pval:.4f}', ha='center',
                transform=ax.transAxes, fontsize=9)
    else:
        groups_data = [data[data[group_col]==g][value_col].dropna().values for g in groups]
        _, pval = sp_stats.f_oneway(*groups_data)
        ax.text(0.5, 1.02, f'ANOVA p = {pval:.4f}', ha='center',
                transform=ax.transAxes, fontsize=9)

def add_regression_eq(x, y, ax):
    from scipy import stats as sp_stats
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
    eq = f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}, p = {p_value:.4f}'
    ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7))


# ── Shared Constants & Helpers ───────────────────────────────────────────────

GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

STOP_CODONS = {'TAA', 'TAG', 'TGA'}


def has_tool(name):
    """Check if an external command-line tool is available."""
    try:
        r = subprocess.run([name, '--version'], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def read_fasta_simple(filepath):
    """Read a FASTA file into (header, sequence) tuples without Biopython.

    This is the shared fallback reader used by modules that don't need
    the full feature set of sequence.read_fasta (which supports GenBank etc.).
    """
    sequences = []
    name, buf = None, []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name:
                    sequences.append((name, ''.join(buf)))
                name = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line)
    if name:
        sequences.append((name, ''.join(buf)))
    return sequences

# ── Shared Bioinformatics Utilities ────────────────────────────────────────────

def reverse_complement_dna(seq: str) -> str:
    """Compute the reverse complement of a DNA sequence.

    This is the canonical implementation shared across all modules.
    Each base is replaced by its complement (A<->T, C<->G, N->N) and the
    result is reversed.

    Args:
        seq: DNA sequence string (case preserved in output).

    Returns:
        Reverse complemented sequence string.
    """
    comp = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
    return seq.translate(comp)[::-1]


# ── Restriction Enzyme Database ────────────────────────────────────────────────
# Comprehensive database of restriction enzymes with recognition sites
# and cut positions. Used by orf_finder.py, cloning.py, and other modules.
# Format: enzyme_name -> (recognition_site, cut_position_from_start_of_site)

RESTRICTION_ENZYMES = {
    # ── A ──────────────────────────────────────────────────────────────────
    'AatII':     ('GACGTC', 1),
    'Acc65I':    ('GGTACC', 1),
    'AflII':     ('CTTAAG', 1),
    'AflIII':    ('ACRYGT', 1),
    'AgeI':      ('ACCGGT', 1),
    'AluI':      ('AGCT', 2),
    'ApaI':      ('GGGCCC', 1),
    'ApaLI':     ('GTGCAC', 1),
    'ApoI':      ('RAATTY', 1),
    'AscI':      ('GGCGCGCC', 2),
    'AvaI':      ('CYCGRG', 1),
    'AvaII':     ('GGWCC', 1),
    'AvrII':     ('CCTAGG', 1),
    # ── B ──────────────────────────────────────────────────────────────────
    'BaeI':      ('ACGRAC', 1),
    'BamHI':     ('GGATCC', 1),
    'BanI':      ('GGYRCC', 1),
    'BanII':     ('GRAGCY', 1),
    'BbsI':      ('GAAGAC', 2),
    'BcgI':      ('CGANNNNNNTGC', 2),
    'BciVI':     ('GTATCC', 1),
    'BclI':      ('TGATCA', 1),
    'BfaI':      ('CTAG', 1),
    'BfuAI':     ('ACCTGC', 1),
    'BglI':      ('GCCNNNNNGGC', 2),
    'BglII':     ('AGATCT', 1),
    'BmtI':      ('GCTAGC', 1),
    'BsaI':      ('GGTCTC', 1),
    'BsaHI':     ('GRCGYC', 1),
    'BsaWI':     ('WCCGGW', 1),
    'BseRI':     ('GAGGAG', 1),
    'BsgI':      ('GTGCAG', 1),
    'BsiWI':     ('CGTACG', 1),
    'BslI':      ('CCNNNNNNNGG', 2),
    'BsmAI':     ('GTCTC', 1),
    'BsmFI':     ('GGGAC', 1),
    'BsoBI':     ('CYCGRG', 1),
    'Bsp1286I':  ('GDGCHC', 1),
    'BspEI':     ('TCCGGA', 1),
    'BspHI':     ('TCATGA', 1),
    'BspMI':     ('ACCTGC', 1),
    'BsrFI':     ('RCCGGY', 1),
    'BsrGI':     ('TGTACA', 1),
    'BstBI':     ('TTCGAA', 1),
    'BstEI':     ('GGTNACC', 1),
    'BstNI':     ('CCWGG', 1),
    'BstUI':     ('CGCG', 2),
    'BstXI':     ('CCANNNNNNTGG', 2),
    'BstZ17I':   ('GTATAC', 1),
    'BsuRI':     ('GGCC', 2),
    'BtgI':      ('CCRGAG', 1),
    'BtgZI':     ('GCGACG', 1),
    'BtsI':      ('GCAGTG', 1),
    'BveI':      ('ACCTGC', 1),
    # ── C ──────────────────────────────────────────────────────────────────
    'Cac8I':     ('GCNNGC', 1),
    'ClaI':      ('ATCGAT', 1),
    'CviAII':    ('CATG', 1),
    'CviJI':     ('RGYRCA', 1),
    # ── D ──────────────────────────────────────────────────────────────────
    'DdeI':      ('CTNAG', 1),
    'DpnI':      ('GATC', 1),
    'DraI':      ('TTTAAA', 1),
    'DraIII':    ('CACGTG', 1),
    'DrdI':      ('GACNNNNNGTC', 1),
    # ── E ──────────────────────────────────────────────────────────────────
    'EaeI':      ('YGGCCR', 1),
    'EagI':      ('CGGCCG', 1),
    'EarI':      ('GCTCTTC', 1),
    'EciI':      ('TGGCCA', 1),
    'EcoNI':     ('CCTNNNNNAGG', 1),
    'EcoO109I':  ('RGGNCCY', 1),
    'EcoRI':     ('GAATTC', 1),
    'EcoRV':     ('GATATC', 1),
    'EcoT22I':   ('ATGCAT', 1),
    # ── F ──────────────────────────────────────────────────────────────────
    'FatI':      ('CATG', 1),
    'FauI':      ('CCCGC', 2),
    'Fnu4HI':    ('GCNGC', 1),
    'FokI':      ('GGATG', 1),
    'FseI':      ('GGCCGGCC', 2),
    'FspI':      ('TGCGCA', 1),
    # ── H ──────────────────────────────────────────────────────────────────
    'HaeII':     ('RGCGCY', 1),
    'HaeIII':    ('GGCC', 2),
    'HgaI':      ('GACGC', 1),
    'HhaI':      ('GCGC', 1),
    'HindIII':   ('AAGCTT', 1),
    'HinfI':     ('GANTC', 1),
    'HpaI':      ('GTTAAC', 1),
    'HpaII':     ('CCGG', 1),
    'Hpy188I':   ('TCNGA', 1),
    'Hpy188III': ('TCNNGA', 1),
    'HpyAV':     ('ACCTT', 1),
    'HpyCH4III': ('ACGT', 1),
    'HpyCH4IV':  ('ACGT', 1),
    # ── K ──────────────────────────────────────────────────────────────────
    'KasI':      ('GGCGCC', 1),
    'KpnI':      ('GGTACC', 1),
    # ── M ──────────────────────────────────────────────────────────────────
    'MboI':      ('GATC', 1),
    'MboII':     ('GAAGA', 1),
    'MfeI':      ('CAATTG', 1),
    'MluI':      ('ACGCGT', 1),
    'MluCI':     ('AATT', 1),
    'MmeI':      ('TCCRAC', 1),
    'MnlI':      ('CCTC', 1),
    'MseI':      ('TTAA', 1),
    'MslI':      ('CAYNNNNRTG', 1),
    'MspA1I':    ('CMGCAG', 1),
    'MspI':      ('CCGG', 1),
    'MwoI':      ('GCNNNNNNNGC', 1),
    # ── N ──────────────────────────────────────────────────────────────────
    'NaeI':      ('GCCGGC', 1),
    'NarI':      ('GGCGCC', 1),
    'NcoI':      ('CCATGG', 1),
    'NdeI':      ('CATATG', 1),
    'NgoMIV':    ('GCCGGC', 1),
    'NheI':      ('GCTAGC', 1),
    'NlaIII':    ('CATG', 1),
    'NlaIV':     ('GGNNCC', 1),
    'NotI':      ('GCGGCCGC', 2),
    'NruI':      ('TCGCGA', 1),
    'NsiI':      ('ATGCAT', 1),
    'NspI':      ('RCATGY', 1),
    'NspBIII':   ('CMGCKG', 1),
    # ── P ──────────────────────────────────────────────────────────────────
    'PacI':      ('TTAATTAA', 2),
    'PaeR7I':    ('CTCGAG', 1),
    'PciI':      ('ACATGT', 1),
    'PmeI':      ('GTTTAAAC', 2),
    'PmlI':      ('CACGTG', 1),
    'PpuMI':     ('RGGWCCY', 1),
    'PstI':      ('CTGCAG', 1),
    'PvuI':      ('CGATCG', 1),
    'PvuII':     ('CAGCTG', 1),
    # ── R ──────────────────────────────────────────────────────────────────
    'RsaI':      ('GTAC', 2),
    'RsrII':     ('CGGACCG', 1),
    # ── S ──────────────────────────────────────────────────────────────────
    'SacI':      ('GAGCTC', 1),
    'SacII':     ('CCGCGG', 1),
    'SalI':      ('GTCGAC', 1),
    'SapI':      ('GCTCTTC', 1),
    'Sau3AI':    ('GATC', 1),
    'Sau96I':    ('GGNCC', 1),
    'SbfI':      ('CCTGCAGG', 2),
    'ScaI':      ('AGTACT', 1),
    'ScrFI':     ('CCNGG', 1),
    'SexAI':     ('ACCWGGT', 1),
    'SfaNI':     ('GCATC', 1),
    'SfiI':      ('GGCCNNNNNGGCC', 2),
    'SfoI':      ('GGCGCC', 1),
    'SgrAI':     ('CRCCGGYG', 1),
    'SmaI':      ('CCCGGG', 3),
    'SmlI':      ('CTYRAG', 1),
    'SnaBI':     ('TACGTA', 1),
    'SpeI':      ('ACTAGT', 1),
    'SphI':      ('GCATGC', 1),
    'SplI':      ('CGTACG', 1),
    'SrfI':      ('GCCCGGGC', 1),
    'Sse8387I':  ('CCTGCAGG', 1),
    'SspI':      ('AATATT', 1),
    'StuI':      ('AGGCCT', 1),
    'StyI':      ('CCWWGG', 1),
    'StyD4I':    ('CCNGG', 1),
    # ── T ──────────────────────────────────────────────────────────────────
    'TaiI':      ('ACGT', 1),
    'TaqI':      ('TCGA', 1),
    'TfiI':      ('GAWTC', 1),
    'TseI':      ('GCWGC', 1),
    'Tsp45I':    ('GTSAC', 1),
    'Tsp509I':   ('AATT', 1),
    'TsrI':      ('GGNCC', 1),
    'Tth111I':   ('GACNNNGTC', 1),
    # ── X ──────────────────────────────────────────────────────────────────
    'XbaI':      ('TCTAGA', 1),
    'XcmI':      ('CCANNNNNNTGG', 2),
    'XhoI':      ('CTCGAG', 1),
    'XmaI':      ('CCCGGG', 1),
    'XmaCI':     ('CCGG', 1),
    'XmnI':      ('GAANNNNTTC', 1),
    # ── Z ──────────────────────────────────────────────────────────────────
    'ZraI':      ('GACGTC', 1),
}

# Convenience: enzyme sites only (no cut positions) for modules that only
# need recognition sequences (e.g. cloning.py digestion simulation).
RESTRICTION_ENZYMES_SITES = {name: site for name, (site, _) in RESTRICTION_ENZYMES.items()}

