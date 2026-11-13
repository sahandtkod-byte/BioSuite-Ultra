"""All biological plots: Volcano, PCA, Manhattan, MA, Venn, Barplot, Boxplot, Heatmap, Scatter, Timeseries, QQ, ClusteredHeatmap, Circos, Alignment, Violin, Raincloud, Ridge, Dot."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import (
    Circle,
    ConnectionPatch,
    Rectangle,
)
from scipy import stats as sp_stats

# Public surface. ``biosuite/plotting/__init__.py`` star-imports this
# module, and without this list it re-exports the module's own imports
# (numpy, pandas, os, plotly, ...) as package-level names too.
__all__ = [
    'draw_venn2',
    'draw_venn3',
    'draw_motif_logo',
    'draw_sankey',
    'volcano_plot',
    'pca_plot',
    'manhattan_plot',
    'ma_plot',
    'venn_diagram',
    'barplot_custom',
    'boxplot_custom',
    'heatmap_custom',
    'scatter_custom',
    'timeseries_plot',
    'qq_plot',
    'clustered_heatmap',
    'circos_plot',
    'alignment_viewer',
    'violin_plot',
    'raincloud_plot',
    'ridge_plot',
    'dot_plot',
    'export_all_to_folder',
    'generate_markdown_story',
    'batch_export_to_pdf',
]

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# Every demo runs inside one shared guard: report a failure, name it in the log,
# and release the figures it opened.  ``specialized_plots`` runs its four demos
# through the same object rather than keeping a second, leakier copy.
from ._demo_guard import demo_guard as _demo

from ..core.log import get_logger
from ..core.utils import (
    add_ttest_to_boxplot,
    apply_glass_ax,
    ask_save_plot,
    autosave_session,
    config,
    load_dataframe_safe,
    maybe_downsample,
    report_boxplot_stats,
    report_manhattan_stats,
    report_pca_stats,
    report_volcano_stats,
    safe_float_input,
    safe_int_input,
    safe_list_input,
    session,
)

logger = get_logger(__name__)


def _checked_regions(subsets, set_labels, n_sets):
    """Region counts and names, validated before anything is drawn.

    Called straight (not through venn_diagram) these used to die on
    ``subsets[2]`` with a bare IndexError for a two-value list, print
    ``10.0`` for a count, and happily draw ``-5`` inside a circle.
    """
    values = list(subsets)
    expected = 3 if n_sets == 2 else 7
    if len(values) != expected:
        raise ValueError(
            f"draw_venn{n_sets}() needs {expected} region counts, got {len(values)} "
            f"({values[:8]}); the regions are "
            + ('[onlyA, onlyB, both]' if n_sets == 2 else
               '[A, B, C, AB, AC, BC, ABC]'))
    if len(set_labels) != n_sets:
        raise ValueError(
            f"draw_venn{n_sets}() needs {n_sets} set labels, got {len(set_labels)} "
            f"({list(set_labels)})")
    counts = []
    for i, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer, float)):
            raise ValueError(
                f"draw_venn{n_sets}() region {i} must be a count, got {value!r}")
        if value < 0:
            raise ValueError(
                f"draw_venn{n_sets}() region {i} is {value!r}; a set cannot hold a "
                f"negative number of members")
        counts.append(int(value) if float(value).is_integer() else float(value))
    if any(float(v) != int(float(v)) for v in values):
        logger.warning("draw_venn%d() received fractional region counts (%s); they "
                       "are drawn as given.", n_sets, values)
    return counts, tuple(set_labels)


def draw_venn2(subsets, set_labels=('A', 'B'), ax=None):
    subsets, set_labels = _checked_regions(subsets, set_labels, 2)
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
    ax.set_title('Venn Diagram (2 sets; schematic, circle areas are not proportional)')
    return ax

def draw_venn3(subsets, set_labels=('A', 'B', 'C'), ax=None):
    subsets, set_labels = _checked_regions(subsets, set_labels, 3)
    if subsets[6] > min(subsets[3:6]):
        raise ValueError(
            f"draw_venn3() was given ABC={subsets[6]} but the pairwise regions "
            f"AB={subsets[3]}, AC={subsets[4]}, BC={subsets[5]}; every member of the "
            f"triple overlap is also a member of each pair, so no pair can be "
            f"smaller than the triple")
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
    ax.set_title('Venn Diagram (3 sets; schematic, circle areas are not proportional)')
    return ax

def draw_motif_logo(sequences, ax=None):
    if not len(sequences):
        raise ValueError(
            "draw_motif_logo() needs at least one sequence; max() of an empty "
            "sequence length has no answer")
    # The counter only knows ACGT, so a lowercase input used to draw a logo with
    # every bar at zero height and no word said about it.
    sequences = [str(s).upper() for s in sequences]
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
    other = sum(1 for seq in padded for ch in seq if ch not in counts and ch != '-')
    if other:
        print(f"   {other} letter(s) outside ACGT were not counted (gaps '-' aside); "
              "a column that stops short of 1.0 is missing information, not low "
              "information.")
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
    if len(sources) != len(targets) or len(sources) != len(values):
        raise ValueError(
            f"draw_sankey() needs one source, target and value per link, got "
            f"{len(sources)} sources, {len(targets)} targets and {len(values)} values")
    if len(labels) == 0:
        raise ValueError("draw_sankey() has no node labels to name a link by")
    out_of_range = [int(i) for i in list(sources) + list(targets)
                    if not 0 <= int(i) < len(labels)]
    if out_of_range:
        raise ValueError(
            f"draw_sankey() was given {len(labels)} labels (indices 0..{len(labels) - 1}) "
            f"but {len(out_of_range)} link end(s) outside them, e.g. index "
            f"{out_of_range[0]}")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))
    if not len(values):
        # max() of nothing used to raise ValueError from inside the drawing.
        ax.set_title('Sankey Diagram (no flows to draw)')
        ax.axis('off')
        return ax
    max_val = max(values)
    if max_val == 0:
        print("   draw_sankey() every flow is zero; links are drawn at their minimum "
              "width instead of scaled.")
    source_x = 0.2
    target_x = 0.8
    source_nodes = list(dict.fromkeys(labels[s] for s in sources))
    target_nodes = list(dict.fromkeys(labels[t] for t in targets))
    for i, node in enumerate(source_nodes):
        y = (i+1)/(len(source_nodes)+1)
        rect = Rectangle((source_x-0.05, y-0.03), 0.1, 0.06, facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(source_x+0.05, y, node, ha='left', va='center', fontsize=8)
    for i, node in enumerate(target_nodes):
        y = (i+1)/(len(target_nodes)+1)
        rect = Rectangle((target_x-0.05, y-0.03), 0.1, 0.06, facecolor='lightgreen', edgecolor='black')
        ax.add_patch(rect)
        ax.text(target_x+0.05, y, node, ha='left', va='center', fontsize=8)
    for s_idx, t_idx, val in zip(sources, targets, values):
        src_node = labels[s_idx]
        tgt_node = labels[t_idx]
        # Both node lists were built from these very indices above, so a link can
        # never be missing from the list its own endpoint went into.
        src_y = (source_nodes.index(src_node)+1)/(len(source_nodes)+1)
        tgt_y = (target_nodes.index(tgt_node)+1)/(len(target_nodes)+1)
        lw = 2 + (8 * val / max_val if max_val else 0)
        patch = ConnectionPatch(xyA=(source_x, src_y), xyB=(target_x, tgt_y), coordsA='data', coordsB='data',
                                arrowstyle='-', connectionstyle='arc3,rad=0.2', color='gray', alpha=0.6, linewidth=lw)
        ax.add_patch(patch)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Sankey Diagram (internal)')
    return ax

def volcano_plot(pdf=None):
    print("\n--- Volcano Plot ---")
    with _demo('volcano_plot'):
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
                        # One row per gene: dropping each column on its own would
                        # pair gene i's fold change with gene i+1's p-value.
                        pairs = df[[lfc_col, p_col]].apply(
                            pd.to_numeric, errors='coerce').dropna()
                        lfc = pairs[lfc_col].to_numpy()
                        pvals = pairs[p_col].to_numpy()
                        if len(lfc) == 0:
                            print(f"   {path} has no numeric rows in "
                                  f"{lfc_col!r} / {p_col!r}; nothing to plot.")
                    else:
                        missing = [c for c in (lfc_col, p_col) if c not in df.columns]
                        print(f"   {path} has no column(s) {', '.join(repr(c) for c in missing)}; "
                              f"its columns are {', '.join(map(str, df.columns))}.")
        if lfc is None or len(lfc) == 0:
            print("   Drawing the built-in simulated data set (500 genes) instead — "
                  "these points are not from your file.")
            np.random.seed(42)
            lfc = np.random.normal(0, 1.5, 500)
            pvals = np.random.uniform(0, 1, 500)
            pvals[:30] = np.random.uniform(1e-6, 0.05, 30)
        fc_thresh = safe_float_input("Fold-change threshold (default 1.0): ", 1.0, key='volcano_fc')
        p_thresh = safe_float_input("P-value threshold (default 0.05): ", 0.05, key='volcano_p')
        # The counts describe the whole table; only the drawing is thinned, and
        # maybe_downsample is the deterministic sampler every other plot uses.
        # Before, an unseeded np.random.choice ran *before* the report, so the
        # same file printed 79, then 85, then 83 significant genes.
        report_volcano_stats(lfc, pvals, fc_thresh, p_thresh)
        lfc, pvals = maybe_downsample(lfc, pvals)
        neg_log10 = -np.log10(pvals + 1e-300)
        sig = (np.abs(lfc) >= fc_thresh) & (pvals < p_thresh)
        up = sig & (lfc > 0)
        down = sig & (lfc < 0)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(lfc[~sig], neg_log10[~sig], s=10, alpha=0.5, label='Not sig', color='gray')
        ax.scatter(lfc[down], neg_log10[down], s=20, alpha=0.7, label='Down', color='blue')
        ax.scatter(lfc[up], neg_log10[up], s=20, alpha=0.7, label='Up', color='red')
        ax.axhline(-np.log10(p_thresh), linestyle='--', color='white' if config['theme']=='dark' else 'black', alpha=0.5)
        ax.axvline(-fc_thresh, linestyle='--', alpha=0.5)
        ax.axvline(fc_thresh, linestyle='--', alpha=0.5)
        ax.set_xlabel('Log2 Fold Change')
        ax.set_ylabel('-log10(P-value)')
        ax.set_title('Volcano Plot')
        ax.legend(framealpha=0.7)
        apply_glass_ax(ax)
        ask_save_plot('volcano', config['save_format'], config['default_dpi'], pdf)

def pca_plot(pdf=None):
    from sklearn.decomposition import PCA
    print("\n--- PCA Plot ---")
    with _demo('pca_plot'):
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
                    if numeric_df.shape[1] < 2:
                        print(f"   {path} has {numeric_df.shape[1]} numeric column(s); "
                              "PCA needs at least two features.")
                    elif numeric_df.shape[0] < 2:
                        print(f"   {path} has {numeric_df.shape[0]} sample row(s); PCA "
                              "needs at least two samples.")
                    else:
                        data = numeric_df.values
                        group_col = input("Group column (optional): ").strip()
                        if group_col and group_col in df.columns:
                            groups = df[group_col].astype(str).to_numpy()
                        else:
                            if group_col:
                                print(f"   {group_col!r} is not a column of this table "
                                      f"({', '.join(map(str, df.columns))}); every sample "
                                      "is drawn in one group called 'Sample'.")
                            groups = np.array(['Sample'] * data.shape[0])
                        # The decomposition runs on every sample, so the variance
                        # percentages describe the file; only the points drawn are
                        # thinned, and the stride is the deterministic one
                        # utils.maybe_downsample uses (an unseeded
                        # np.random.choice made a different picture every run).
                        pca = PCA(n_components=2)
                        pc = pca.fit_transform(data)
                        report_pca_stats(pca)
                        limit = config['downsample_threshold']
                        if pc.shape[0] > limit:
                            keep = np.unique(np.linspace(0, pc.shape[0] - 1, limit)
                                             .round().astype(int))
                            print(f"   Drawing {len(keep)} of {pc.shape[0]} samples "
                                  "(the decomposition used all of them).")
                            pc, groups = pc[keep], groups[keep]
                        df_pca = pd.DataFrame({'PC1': pc[:,0], 'PC2': pc[:,1], 'Group': groups})
                        fig, ax = plt.subplots(figsize=(7,6))
                        sns.scatterplot(x='PC1', y='PC2', hue='Group', data=df_pca, s=70, palette='Set1', ax=ax)
                        ax.set_title(f'PCA (var: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})')
                        apply_glass_ax(ax)
                        ask_save_plot('pca', config['save_format'], config['default_dpi'], pdf)
                        return
        print("   Drawing the built-in simulated data set (30 samples x 100 "
              "features, two groups) — this is not your file.")
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
        ask_save_plot('pca', config['save_format'], config['default_dpi'], pdf)

def manhattan_plot(pdf=None):
    print("\n--- Manhattan Plot ---")
    with _demo('manhattan_plot'):
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
                df = None
                np.random.seed(42)
                n = 1000
                chroms = [f'Chr{i}' for i in range(1,6)]
                df = pd.DataFrame({'chrom': np.random.choice(chroms, n),
                                   'pos': np.random.randint(1, 1_000_000, n),
                                   'p': -np.log10(np.random.uniform(1e-8, 0.5, n))})
        if df.empty:
            print("No valid data.")
            return
        report_manhattan_stats(df, threshold=5e-8)
        df = df.sort_values(['chrom', 'pos'])
        chrom_max = df.groupby('chrom')['pos'].max()
        offsets = chrom_max.cumsum().shift(1).fillna(0)
        df['cumpos'] = df['pos'] + df['chrom'].map(offsets.to_dict())
        fig, ax = plt.subplots(figsize=(12,5))
        colors = ['#2c7bb6', '#abd9e9'] if config['theme']=='light' else ['#00ff99', '#33cc66']
        for i, (chr_name, grp) in enumerate(df.groupby('chrom')):
            ax.scatter(grp['cumpos'], grp['p'], s=5, color=colors[i%2], label=chr_name)
        ax.axhline(-np.log10(5e-8), linestyle='--', color='red', alpha=0.7, label='Genome-wide sig')
        ax.set_xlabel('Chromosome')
        ax.set_ylabel('-log10(p)')
        ax.set_title('Manhattan Plot')
        ax.set_xticks(offsets + chrom_max.values / 2)
        ax.set_xticklabels(offsets.index)
        ax.legend(loc='upper right', ncol=2, framealpha=0.5)
        apply_glass_ax(ax)
        ask_save_plot('manhattan', config['save_format'], config['default_dpi'], pdf)

def ma_plot(pdf=None):
    print("\n--- MA Plot ---")
    with _demo('ma_plot'):
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
            else:
                # Without this the synthetic-data branch below is skipped and the
                # plot line dies on unbound A and M, leaving its figure open.
                print("Could not read the file. Using defaults.")
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
        ax.set_ylabel('M (log2 fold change)')
        ax.set_title('MA Plot')
        apply_glass_ax(ax)
        ask_save_plot('maplot', config['save_format'], config['default_dpi'], pdf)

def venn_diagram(pdf=None):
    print("\n--- Venn Diagram (internal) ---")
    with _demo('venn_diagram'):
        n_sets = input("Number of sets (2 or 3): ").strip()
        if n_sets == '2':
            sizes = safe_list_input("Enter sizes a,b,ab (e.g., 10,15,4): ", int)
            if sizes is None or len(sizes) != 3:
                sizes = [10,15,4]
            fig, ax = plt.subplots(figsize=(6,6))
            draw_venn2(sizes, set_labels=('Set A', 'Set B'), ax=ax)
        elif n_sets == '3':
            sizes = safe_list_input("Enter sizes a,b,c,ab,ac,bc,abc: ", int)
            if sizes is None or len(sizes) != 7:
                sizes = [8,8,8,3,3,3,1]
            fig, ax = plt.subplots(figsize=(6,6))
            draw_venn3(sizes, set_labels=('A', 'B', 'C'), ax=ax)
        else:
            print("Only 2 or 3 sets supported.")
            return
        apply_glass_ax(ax)
        ask_save_plot('venn', config['save_format'], config['default_dpi'], pdf)

def barplot_custom(pdf=None):
    print("\n--- Barplot ---")
    with _demo('barplot_custom'):
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
                    # One row per bar: dropping the value column on its own would
                    # shift every label under the wrong height.
                    has_errors = bool(err_col) and err_col in df.columns
                    wanted = [x_col, y_col] + ([err_col] if has_errors else [])
                    rows = df[wanted].copy()
                    rows[y_col] = pd.to_numeric(rows[y_col], errors='coerce')
                    if has_errors:
                        rows[err_col] = pd.to_numeric(rows[err_col], errors='coerce')
                    rows = rows.dropna(subset=[y_col])
                    groups = rows[x_col].astype(str).tolist()
                    values = rows[y_col].to_numpy()
                    if has_errors:
                        errors = rows[err_col].to_numpy()
                        blank = int(np.isnan(errors).sum())
                        if blank:
                            print(f"   {blank} of {len(errors)} values in {err_col!r} "
                                  "are not a number; those caps are absent.")
                    elif err_col:
                        print(f"   {err_col!r} is not a column of this table; the bars "
                              "are drawn without error caps.")
        if not groups:
            print("   Drawing the built-in demo bars (Ctrl/TrA/TrB) — these bars are "
                  "not from your file.")
            groups = ['Ctrl', 'TrA', 'TrB']
            values = np.array([5.2, 8.7, 3.4])
            errors = np.array([0.5, 0.8, 0.4])
        fig, ax = plt.subplots(figsize=(6,4))
        # seaborn aggregates repeated categories into one bar holding their mean,
        # while the caller's error column belongs to the rows.  Caps drawn at row
        # positions on aggregated bars describe bars that do not exist, so they are
        # only added when one row really is one bar.
        unique_groups = list(dict.fromkeys(str(g) for g in groups))
        duplicates = len(unique_groups) != len(groups)
        sns.barplot(x=groups, y=values, hue=groups, palette='viridis',
                    edgecolor='black', legend=False, ax=ax)
        if errors is not None:
            if duplicates:
                print(f"   {len(groups)} rows collapse into {len(unique_groups)} "
                      "categories, so each bar is a mean; the error column belongs to "
                      "the rows and is not drawn. seaborn's own confidence interval is "
                      "shown instead.")
            else:
                ax.errorbar(x=range(len(unique_groups)), y=values, yerr=errors,
                            fmt='none', c='black', capsize=5)
        ax.set_ylabel('Expression')
        ax.set_title('Barplot')
        apply_glass_ax(ax)
        ask_save_plot('barplot', config['save_format'], config['default_dpi'], pdf)

def boxplot_custom(pdf=None):
    print("\n--- Boxplot ---")
    with _demo('boxplot_custom'):
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
                    sns.boxplot(x=group_col, y=value_col, data=df_plot, hue=group_col,
                                palette='Set2', legend=False, ax=ax)
                    sns.stripplot(x=group_col, y=value_col, data=df_plot, color='black', alpha=0.6, size=4, ax=ax)
                    ax.set_title('Boxplot')
                    apply_glass_ax(ax)
                    report_boxplot_stats(df_plot, group_col, value_col)
                    if len(df_plot[group_col].unique()) >= 2:
                        add_ttest_to_boxplot(df_plot, group_col, value_col, ax)
                    ask_save_plot('boxplot', config['save_format'], config['default_dpi'], pdf)
                    return
        print("   Drawing the built-in demo groups (Ctrl/Treat, 30 each) — this is "
              "not your file.")
        ctrl = np.random.normal(5,1,30)
        treat = np.random.normal(7.5,1.2,30)
        df_plot = pd.DataFrame({'Group': ['Ctrl']*30 + ['Treat']*30, 'Value': np.concatenate([ctrl, treat])})
        fig, ax = plt.subplots(figsize=(6,5))
        sns.boxplot(x='Group', y='Value', data=df_plot, hue='Group', palette='Set2',
                    legend=False, ax=ax)
        sns.stripplot(x='Group', y='Value', data=df_plot, color='black', alpha=0.6, size=4, ax=ax)
        ax.set_title('Boxplot')
        apply_glass_ax(ax)
        report_boxplot_stats(df_plot, 'Group', 'Value')
        add_ttest_to_boxplot(df_plot, 'Group', 'Value', ax)
        ask_save_plot('boxplot', config['save_format'], config['default_dpi'], pdf)

def heatmap_custom(pdf=None):
    print("\n--- Heatmap ---")
    with _demo('heatmap_custom'):
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr = numeric_df.corr()
                    # A column that never varies has no correlation with anything:
                    # pandas writes NaN and seaborn leaves the cell blank, which
                    # reads as "r = 0" to whoever is looking at it.
                    blank = int(np.triu(corr.isna().to_numpy(), 1).sum())
                    # corr() spreads a NaN across the dead column's whole row, so
                    # reading the culprits back off corr would name every column it
                    # touched.  The zero-variance columns come from the data.
                    constant = [str(name) for name in numeric_df.columns
                                if numeric_df[name].nunique(dropna=False) <= 1]
                    if blank or constant:
                        print(f"   {blank} column pair(s) have no correlation because a "
                              f"column never varies"
                              f"{': ' + ', '.join(constant) if constant else ''}. Those "
                              "cells are blank — undefined, not zero.")
                    fig, ax = plt.subplots(figsize=(7,6))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                    ax.set_title('Correlation Heatmap')
                    apply_glass_ax(ax)
                    ask_save_plot('heatmap', config['save_format'], config['default_dpi'], pdf)
                    return
                print("That file has no numeric columns to correlate; drawing the demo matrix.")
        corr = np.array([[1,0.8,0.2,0.1],[0.8,1,0.3,0.2],[0.2,0.3,1,0.7],[0.1,0.2,0.7,1]])
        labels = ['Gene_A','Gene_B','Gene_C','Gene_D']
        fig, ax = plt.subplots(figsize=(7,6))
        sns.heatmap(pd.DataFrame(corr, index=labels, columns=labels), annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Gene Correlation')
        apply_glass_ax(ax)
        ask_save_plot('heatmap', config['save_format'], config['default_dpi'], pdf)

def scatter_custom(pdf=None):
    print("\n--- Scatter Plot ---")
    with _demo('scatter_custom'):
        use_file = input("Load data from file? (y/n): ").strip().lower()
        x, y = None, None
        corr_type = 'pearson'
        if use_file == 'y':
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
                        # One point per row: dropping each column on its own would
                        # pair x from one row with y from the next.
                        pairs = df[[x_col, y_col]].apply(
                            pd.to_numeric, errors='coerce').dropna()
                        x, y = maybe_downsample(pairs[x_col].to_numpy(),
                                                pairs[y_col].to_numpy())
                        if len(x) < 2:
                            print(f"   {path} gives fewer than two usable "
                                  f"{x_col!r}/{y_col!r} pairs; a line cannot be fitted.")
                    else:
                        missing = [c for c in (x_col, y_col) if c not in df.columns]
                        print(f"   {path} has no column(s) "
                              f"{', '.join(repr(c) for c in missing)}; its columns are "
                              f"{', '.join(map(str, df.columns))}.")
        if x is None or len(x) < 2:
            print("   Drawing the built-in simulated points (20 of them) instead — "
                  "this is not your file.")
            np.random.seed(42)
            x = np.linspace(0,10,20)
            y = 2.5*x + np.random.normal(0,1.5,20)
        # The statistic is worked out before any figure exists.  It used to be
        # computed after regplot had already drawn, so a column of identical x
        # values died inside linregress with an empty figure open, printed one
        # "Error:" line and left the points unpainted.
        distinct_x = len(np.unique(x))
        fit = None
        if corr_type == 'spearman':
            if distinct_x < 2 or len(np.unique(y)) < 2:
                print("   Spearman's rho needs two distinct values in both columns; "
                      "the points are drawn without a statistic.")
            else:
                result = sp_stats.spearmanr(x, y)
                fit = ('spearman', float(result.statistic), float(result.pvalue))
        elif distinct_x < 2:
            print(f"   Every x value is {float(x[0])!r}, so no line can be fitted; the "
                  "points are drawn without one.")
        else:
            slope, intercept, r_value, p_value, _stderr = sp_stats.linregress(x, y)
            fit = ('pearson', slope, intercept, r_value, p_value)

        fig, ax = plt.subplots(figsize=(7,5))
        if fit is None:
            sns.scatterplot(x=x, y=y, s=60, edgecolor='black', ax=ax)
            ax.set_title('Scatter (no line fitted)')
        else:
            sns.regplot(x=x, y=y, ci=None, scatter_kws={'s':60, 'edgecolor':'black'},
                        line_kws={'color':'red', 'ls':'--'}, ax=ax)
            ax.set_title('Scatter with Regression')
        apply_glass_ax(ax)
        if fit is not None and fit[0] == 'spearman':
            rho, pval = fit[1], fit[2]
            print(f"\nScatter Statistics: Spearman rho = {rho:.3f}, p-value = {pval:.4f}")
            eq = f'Spearman rho = {rho:.3f}, p = {pval:.4f}'
            ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        elif fit is not None:
            _slope, _intercept, r_value, p_value = fit[1], fit[2], fit[3], fit[4]
            print(f"\nScatter Statistics: Pearson r = {r_value:.3f}, p-value = {p_value:.4f}")
            eq = (f'y = {fit[1]:.3f}x + {fit[2]:.3f}\n'
                  f'R² = {r_value ** 2:.3f}, p = {p_value:.4f}')
            ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        ask_save_plot('scatter', config['save_format'], config['default_dpi'], pdf)

def timeseries_plot(pdf=None):
    print("\n--- Time Series ---")
    with _demo('timeseries_plot'):
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                time_col = input("Time column: ").strip()
                val_col = input("Value column: ").strip()
                if time_col in df.columns and val_col in df.columns:
                    # One point per row, and a line needs two of them: dropping
                    # each column on its own used to leave the two arrays a
                    # different length, which silently plotted the demo curve
                    # over the user's file.
                    pairs = df[[time_col, val_col]].apply(
                        pd.to_numeric, errors='coerce').dropna()
                    times = pairs[time_col].to_numpy()
                    values = pairs[val_col].to_numpy()
                    if len(times) >= 2:
                        fig, ax = plt.subplots(figsize=(8,5))
                        sns.lineplot(x=times, y=values, marker='o', color='green', ax=ax)
                        ax.set_title('Time Series')
                        apply_glass_ax(ax)
                        ask_save_plot('timeseries', config['save_format'], config['default_dpi'], pdf)
                        return
                    print("File has fewer than two complete time/value pairs. "
                          "Using defaults.")
        times = np.arange(0,24,2)
        values = np.exp(-0.1*times) + np.random.normal(0,0.05,len(times))
        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(x=times, y=values, marker='o', color='green', ax=ax)
        ax.set_title('Time Series')
        apply_glass_ax(ax)
        ask_save_plot('timeseries', config['save_format'], config['default_dpi'], pdf)

def qq_plot(pdf=None):
    print("\n--- QQ-plot (Normality Check) ---")
    with _demo('qq_plot'):
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
            print("   No usable column arrived, so the QQ-plot shows 200 simulated "
                  "standard-normal draws — not your data.")
            np.random.seed(42)
            data = np.random.normal(0, 1, 200)
        if len(data) > config['downsample_threshold']:
            data = np.random.choice(data, config['downsample_threshold'], replace=False)
        fig, ax = plt.subplots(figsize=(6,6))
        sp_stats.probplot(data, dist="norm", plot=ax)
        ax.set_title('QQ-plot (Normal Distribution)')
        apply_glass_ax(ax)
        ask_save_plot('qqplot', config['save_format'], config['default_dpi'], pdf)

def clustered_heatmap(pdf=None):
    print("\n--- Clustered Heatmap with Dendrograms ---")
    with _demo('clustered_heatmap'):
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
            data = data.sample(2000)
            print(f"Downsampled to {data.shape[0]} rows for heatmap.")
        g = sns.clustermap(data, cmap='coolwarm', standard_scale=1,
                           figsize=(10, 8), dendrogram_ratio=0.2,
                           cbar_pos=(0.02, 0.8, 0.03, 0.18))
        g.ax_heatmap.set_title('Clustered Heatmap')
        ask_save_plot('clustered_heatmap', config['save_format'], config['default_dpi'], pdf)

def circos_plot(pdf=None):
    print("\n--- Simple Circos Plot (internal) ---")
    with _demo('circos_plot'):
        use_default = input("Use default data? (y/n): ").strip().lower()
        if use_default == 'n':
            sectors = {}
            n = safe_int_input("Number of sectors: ", 3)
            for i in range(n):
                name = input(f"Sector {i+1} name: ").strip()
                size = safe_float_input(f"Size of {name}: ", 10)
                if name in sectors:
                    print(f"   Sector name {name!r} was given twice; the earlier size "
                          f"is replaced, so {len(sectors)} of the {n} sectors asked "
                          "for survive.")
                sectors[name] = size
            links = []
            n_links = safe_int_input("Number of links: ", 2)
            for i in range(n_links):
                s1 = input(f"Link {i+1} from sector: ").strip()
                s2 = input("          to sector: ").strip()
                links.append((s1, s2))
        else:
            sectors = {"Gene_A": 10, "Gene_B": 8, "Gene_C": 12, "Gene_D": 6}
            links = [("Gene_A", "Gene_B"), ("Gene_B", "Gene_C"), ("Gene_C", "Gene_D"), ("Gene_D", "Gene_A")]
        total = sum(sectors.values())
        if not sectors or total <= 0:
            print(f"   Circos needs a positive total angle budget; the sectors given "
                  f"sum to {total!r}"
                  f"{', so nothing is drawn' if not sectors else ''}.")
            return
        fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': 'polar'})
        start = 0
        colors = plt.cm.tab20(np.linspace(0,1,len(sectors)))
        sector_names = list(sectors.keys())
        for (name, size), color in zip(sectors.items(), colors):
            end = start + 2*np.pi * size/total
            ax.bar(x=start, height=0.5, width=end-start, bottom=0.2,
                   color=color, edgecolor='black', alpha=0.7, align='edge')
            ax.text(start + (end-start)/2, 0.8, name, ha='center', va='center', fontsize=8)
            start = end
        dropped = []
        for (s1, s2) in links:
            if s1 not in sector_names or s2 not in sector_names:
                dropped.append((s1, s2))
                continue
            i1 = sector_names.index(s1)
            i2 = sector_names.index(s2)
            sizes = list(sectors.values())
            start1 = sum(sizes[:i1])/total * 2*np.pi
            end1 = start1 + sizes[i1]/total * 2*np.pi
            start2 = sum(sizes[:i2])/total * 2*np.pi
            end2 = start2 + sizes[i2]/total * 2*np.pi
            theta1 = (start1 + end1)/2
            theta2 = (start2 + end2)/2
            rad = np.linspace(theta1, theta2, 100)
            r = np.linspace(1.0, 1.0, 100)
            ax.plot(rad, r, color='red', alpha=0.6, linewidth=1)
        if dropped:
            print(f"   {len(dropped)} link(s) named a sector that does not exist and "
                  f"were not drawn: {', '.join(f'{a}->{b}' for a, b in dropped[:5])}.")
        ax.set_ylim(0,1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Simple Circos Plot")
        apply_glass_ax(ax)
        ask_save_plot('circos', config['save_format'], config['default_dpi'], pdf)

def alignment_viewer(pdf=None):
    print("\n--- Alignment Viewer ---")
    with _demo('alignment_viewer'):
        use_default = input("Use default alignment? (y/n): ").strip().lower()
        sequences = []
        if use_default == 'n':
            print("Enter sequences (one per line, empty line to finish):")
            while True:
                try:
                    seq = input().strip().upper()
                except EOFError:
                    break
                if not seq:
                    break
                sequences.append(seq)
        if not sequences:
            sequences = ["ATCGATCG", "ATCGATCG", "ATCGGTCG", "ATCGATAG", "ATCGATGG"]
        max_len = max(len(s) for s in sequences)
        aligned = [list(s.ljust(max_len, '-')) for s in sequences]
        # One cell per base at 0.3in per column made a 2000-column alignment a
        # 600-inch-wide figure; the canvas is capped and the cap is said out loud.
        width = min(max_len * 0.3, 42.0)
        height = min(len(sequences) * 0.4, 30.0)
        if max_len * 0.3 > width or len(sequences) * 0.4 > height:
            print(f"   {len(sequences)} sequences x {max_len} columns need "
                  f"{round(max_len * 0.3, 1)} x {round(len(sequences) * 0.4, 1)} inches "
                  f"at one cell per base; the figure is capped at "
                  f"{width} x {height} inches, so the cells shrink. Subset the "
                  "alignment to read it comfortably.")
        fig, ax = plt.subplots(figsize=(width, height))
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
        ask_save_plot('alignment', config['save_format'], config['default_dpi'], pdf)

def violin_plot(pdf=None):
    print("\n--- Violin Plot (Enhanced Distribution) ---")
    with _demo('violin_plot'):
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
                    sns.violinplot(x=group_col, y=value_col, data=df_plot,
                                   inner='quartile', hue=group_col, palette='muted',
                                   legend=False, ax=ax)
                    sns.stripplot(x=group_col, y=value_col, data=df_plot, color='black', alpha=0.5, size=3)
                    ax.set_title('Violin Plot with Quartiles')
                    if len(df_plot[group_col].unique()) >= 2:
                        add_ttest_to_boxplot(df_plot, group_col, value_col, ax)
                    apply_glass_ax(ax)
                    ask_save_plot('violin', config['save_format'], config['default_dpi'], pdf)
                    return
        print("   Drawing the built-in demo groups (Ctrl/Treat, 50 each) — this is "
              "not your file.")
        ctrl = np.random.normal(5,1,50)
        treat = np.random.normal(7.5,1.2,50)
        df_plot = pd.DataFrame({'Group': ['Ctrl']*50 + ['Treat']*50, 'Value': np.concatenate([ctrl, treat])})
        fig, ax = plt.subplots(figsize=(8,6))
        sns.violinplot(x='Group', y='Value', data=df_plot, inner='quartile',
                       hue='Group', palette='muted', legend=False, ax=ax)
        sns.stripplot(x='Group', y='Value', data=df_plot, color='black', alpha=0.5, size=3)
        ax.set_title('Violin Plot (Demo)')
        add_ttest_to_boxplot(df_plot, 'Group', 'Value', ax)
        apply_glass_ax(ax)
        ask_save_plot('violin', config['save_format'], config['default_dpi'], pdf)

def raincloud_plot(pdf=None):
    print("\n--- Raincloud Plot (Box+Violin+Swarm) ---")
    with _demo('raincloud_plot'):
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
                    sns.violinplot(x=group_col, y=value_col, data=df_plot, inner=None,
                                   hue=group_col, palette='muted', alpha=0.5,
                                   legend=False, ax=ax)
                    sns.boxplot(x=group_col, y=value_col, data=df_plot, width=0.2, boxprops=dict(alpha=0.5), showfliers=False)
                    sns.swarmplot(x=group_col, y=value_col, data=df_plot, color='black', alpha=0.6, size=4)
                    ax.set_title('Raincloud Plot')
                    if len(df_plot[group_col].unique()) >= 2:
                        add_ttest_to_boxplot(df_plot, group_col, value_col, ax)
                    apply_glass_ax(ax)
                    ask_save_plot('raincloud', config['save_format'], config['default_dpi'], pdf)
                    return
        print("   Drawing the built-in demo groups (Ctrl/Treat, 50 each) — this is "
              "not your file.")
        ctrl = np.random.normal(5,1,50)
        treat = np.random.normal(7.5,1.2,50)
        df_plot = pd.DataFrame({'Group': ['Ctrl']*50 + ['Treat']*50, 'Value': np.concatenate([ctrl, treat])})
        fig, ax = plt.subplots(figsize=(8,6))
        sns.violinplot(x='Group', y='Value', data=df_plot, inner=None, hue='Group',
                       palette='muted', alpha=0.5, legend=False, ax=ax)
        sns.boxplot(x='Group', y='Value', data=df_plot, width=0.2, boxprops=dict(alpha=0.5), showfliers=False)
        sns.swarmplot(x='Group', y='Value', data=df_plot, color='black', alpha=0.6, size=4)
        ax.set_title('Raincloud Plot (Demo)')
        add_ttest_to_boxplot(df_plot, 'Group', 'Value', ax)
        apply_glass_ax(ax)
        ask_save_plot('raincloud', config['save_format'], config['default_dpi'], pdf)

def ridge_plot(pdf=None):
    print("\n--- Ridge Plot (Overlapping KDEs) ---")
    with _demo('ridge_plot'):
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                value_col = input("Value column: ").strip()
                group_col = input("Group column: ").strip()
                if value_col in df.columns and group_col in df.columns:
                    df_plot = df[[group_col, value_col]].dropna()
                    # A kernel density estimate needs two points.  One group with a
                    # single value used to abort the whole plot from inside
                    # gaussian_kde ("'dataset' input should have multiple elements").
                    counts = df_plot.groupby(group_col)[value_col].transform('size')
                    drawable = df_plot[counts >= 2]
                    skipped = sorted(set(df_plot[group_col]) - set(drawable[group_col]),
                                     key=str)
                    if skipped:
                        print("  Groups with a single value, no density drawn: "
                              f"{', '.join(str(g) for g in skipped)}")
                    if drawable.empty:
                        print("No group has the two values a density needs. "
                              "Using defaults.")
                    else:
                        fig, ax = plt.subplots(figsize=(8,6))
                        for i, grp in enumerate(drawable[group_col].unique()):
                            subset = drawable[drawable[group_col] == grp][value_col]
                            density = sp_stats.gaussian_kde(subset)
                            xs = np.linspace(subset.min(), subset.max(), 200)
                            ys = density(xs) + i
                            ax.fill_between(xs, ys, i, alpha=0.5, label=grp)
                        ax.set_xlabel(value_col)
                        ax.set_ylabel('Density shift')
                        ax.set_title('Ridge Plot')
                        apply_glass_ax(ax)
                        ask_save_plot('ridge', config['save_format'], config['default_dpi'], pdf)
                        return
        print("   Drawing the built-in demo ridges (two groups of 200) — this is not "
              "your file.")
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
            density = sp_stats.gaussian_kde(subset)
            xs = np.linspace(subset.min(), subset.max(), 200)
            ys = density(xs) + i
            ax.fill_between(xs, ys, i, alpha=0.5, label=grp)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density shift')
        ax.set_title('Ridge Plot (Demo)')
        apply_glass_ax(ax)
        ask_save_plot('ridge', config['save_format'], config['default_dpi'], pdf)

def dot_plot(pdf=None):
    print("\n--- Dot Plot (Single-cell style) ---")
    with _demo('dot_plot'):
        use_file = input("Load data from file? (y/n): ").strip().lower()
        if use_file == 'y':
            path = input("File path: ").strip()
            df = load_dataframe_safe(path)
            if df is not None:
                gene_col = input("Gene column: ").strip()
                cluster_col = input("Cluster column: ").strip()
                pct_col = input("Percent expressed column: ").strip()
                exp_col = input("Average expression column: ").strip()
                if all(c in df.columns for c in [gene_col, cluster_col, pct_col, exp_col]):
                    cells = df.groupby([gene_col, cluster_col]).size()
                    repeats = int((cells > 1).sum())
                    if repeats:
                        print(f"   {repeats} gene/cluster cell(s) hold more than one "
                              "row; they are averaged. Pass one row per gene and "
                              "cluster if you meant something else.")
                    genes = df[gene_col].unique()
                    clusters = df[cluster_col].unique()
                    pivot_pct = df.pivot_table(index=gene_col, columns=cluster_col,
                                               values=pct_col, aggfunc='mean').fillna(0)
                    pivot_exp = df.pivot_table(index=gene_col, columns=cluster_col,
                                               values=exp_col, aggfunc='mean').fillna(0)
                    if float(pivot_pct.to_numpy().max(initial=0.0)) > 1.0:
                        # size = 20 + pct*80, so a 0-100 column becomes balloons.
                        print("   The percent-expressed column is on a 0-100 scale; "
                              "marker area is computed from a 0-1 fraction, so it is "
                              "divided by 100 for this plot.")
                        pivot_pct = pivot_pct / 100.0
                    fig, ax = plt.subplots(figsize=(len(clusters)*0.8, len(genes)*0.4))
                    for i, gene in enumerate(genes):
                        for j, cl in enumerate(clusters):
                            pct = pivot_pct.loc[gene, cl]
                            exp = pivot_exp.loc[gene, cl]
                            size = 20 + (pct * 80)
                            color = exp
                            ax.scatter(j, i, s=size, c=color, cmap='viridis', vmin=0, vmax=max(pivot_exp.max()))
                    ax.set_yticks(range(len(genes)))
                    ax.set_yticklabels(genes)
                    ax.set_xticks(range(len(clusters)))
                    ax.set_xticklabels(clusters)
                    ax.set_title('Dot Plot')
                    plt.colorbar(ax.collections[0], ax=ax, label='Avg Expression')
                    apply_glass_ax(ax)
                    ask_save_plot('dotplot', config['save_format'], config['default_dpi'], pdf)
                    return
        print("Using default demo data.")
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
        ask_save_plot('dotplot', config['save_format'], config['default_dpi'], pdf)

def _batch_plot_functions():
    """Every demo the batch exporters run, in export order.

    Both exporters used to carry their own copy of this list and their own
    answer map, and they drifted: the PDF export answered the "Number of sets"
    question with an empty string, so Venn quietly skipped itself out of the
    report while the folder export still wrote its PNG.

    ``math_plots`` and ``specialized_plots`` are imported here rather than at
    module scope because ``specialized_plots`` pulls drawing helpers out of
    this module, which would make the top-level import circular.
    """
    from .math_plots import (
        cosine_plot,
        cubic_plot,
        exponential_plot,
        linear_plot,
        logistic_plot,
        quadratic_plot,
        sine_plot,
    )
    from .specialized_plots import gsea_plot, motif_logo, sankey_diagram, umap_plot

    return [
        volcano_plot, pca_plot, manhattan_plot, ma_plot, venn_diagram,
        barplot_custom, boxplot_custom, heatmap_custom, scatter_custom, timeseries_plot,
        sine_plot, cosine_plot, linear_plot, quadratic_plot, cubic_plot,
        exponential_plot, logistic_plot, gsea_plot, motif_logo, sankey_diagram,
        qq_plot, clustered_heatmap, circos_plot, alignment_viewer, umap_plot,
        violin_plot, raincloud_plot, ridge_plot, dot_plot
    ]


def _batch_default_input(prompt):
    """Answer a demo's questions the way a batch run means them: defaults only.

    The batch saves through ``auto_save_plot`` below and never loads a file, so the
    questions a file or a saver would ask (correlation type, one sequence per line,
    "Save this plot?") cannot reach this responder; anything else it is handed is
    answered with the empty line that means "take the default".
    """
    if "Load data from file?" in prompt:
        return 'n'
    if "Number of sets" in prompt:
        return '2'
    if "Use default" in prompt or "Use default alignment" in prompt:
        return 'y'
    return ''


def export_all_to_folder(folder_name="biosuite_export"):
    import builtins

    print(f"\nExporting all plots to folder: {folder_name}")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    original_format = config['save_format']
    original_quiet = config.get('quiet', False)
    config['save_format'] = 'png'
    config['quiet'] = True

    plot_funcs = _batch_plot_functions()
    original_cwd = os.getcwd()
    os.chdir(folder_name)

    original_input = builtins.input
    builtins.input = _batch_default_input

    _orig_ask_save = ask_save_plot
    def auto_save_plot(default_name, save_format=None, dpi=None, pdf=None, story=None):
        # Replaces utils.ask_save_plot for the batch, so it owes the same release.
        fig = plt.gcf()
        try:
            try:
                plt.tight_layout()
                plt.savefig(f"{default_name}.png", dpi=config['default_dpi'],
                            bbox_inches='tight', facecolor=plt.rcParams['figure.facecolor'])
                print(f"   Auto-saved: {default_name}.png")
            except OSError as exc:
                # Printing nothing at all left the user believing the batch export
                # had produced a file for every plot.
                print(f"   FAILED to save {default_name}.png: {exc}")
        finally:
            plt.close(fig)

    # Patch the auto-saver into EVERY module that IMPORTS ask_save_plot
    # (math_plots / specialized_plots imported the symbol into their own
    # namespaces; patching only this module silently skipped 16 of the
    # 29 plots — sine/cosine/logistic/... were never written to disk).
    from . import math_plots as _mp
    from . import specialized_plots as _sp
    _orig_map = {}
    for _mod in (_mp, _sp):
        _orig_map[_mod] = _mod.ask_save_plot
        _mod.ask_save_plot = auto_save_plot
    import biosuite.plotting.biological_plots as bp_mod
    bp_mod.ask_save_plot = auto_save_plot

    try:
        iterator = tqdm(plot_funcs, desc="Exporting to folder") if HAS_TQDM else plot_funcs
        for func in iterator:
            try:
                func(pdf=None)
            except Exception as e:
                print(f"Warning: {func.__name__} failed: {e}")
    finally:
        builtins.input = original_input
        bp_mod.ask_save_plot = _orig_ask_save
        for _mod, _fn in _orig_map.items():
            _mod.ask_save_plot = _fn
        config['save_format'] = original_format
        config['quiet'] = original_quiet
        os.chdir(original_cwd)

def generate_markdown_story(plot_names, output_path="story.md"):
    import datetime
    lines = [
        "# BioSuite – Plot Story Report",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        f"This report contains {len(plot_names)} exported plots.",
        "",
        "## Plots",
        ""
    ]
    missing = []
    for name in plot_names:
        img_path = f"{name}.png"
        # A square bracket in the name would end the alt text early and turn the
        # line into broken markup, and a name whose PNG was never written made the
        # story advertise a plot the export failed to produce.
        safe = str(name).replace('[', '\\[').replace(']', '\\]')
        lines.append(f"### {str(name).replace('_', ' ').title()}")
        # <…> is the CommonMark form of a link destination, and the only one that
        # survives a file name holding ']' or '(' — without it the first ']' closed
        # the link and the line stopped being markup.  Ordinary names are left as
        # they were, so a normal story still reads ![volcano](volcano.png).
        special = any(ch in img_path for ch in '[]() ')
        destination = f"<{img_path}>" if special else img_path
        lines.append(f"![{safe}]({destination})")
        lines.append("")
        if not os.path.exists(img_path):
            missing.append(img_path)
    if missing:
        print(f"   {len(missing)} of {len(plot_names)} referenced images are not in "
              f"this folder: {', '.join(missing[:5])}. The story links to them "
              "anyway; open it from the folder that holds them.")
    with open(output_path, 'w', encoding="utf-8") as f:
        f.write('\n'.join(lines))
    print(f"Story report saved to {output_path}")

def batch_export_to_pdf(pdf_path="biosuite_report.pdf"):
    import builtins

    print(f"\nBatch exporting all plots to {pdf_path}...")
    plot_funcs = _batch_plot_functions()
    original_input = builtins.input
    builtins.input = _batch_default_input
    original_quiet = config.get('quiet', False)
    config['quiet'] = True
    try:
        with PdfPages(pdf_path) as pdf:
            for func in plot_funcs:
                try:
                    func(pdf=pdf)
                except Exception as e:
                    print(f"Warning: {func.__name__} failed: {e}")
        print(f"PDF saved: {pdf_path}")
    finally:
        config['quiet'] = original_quiet
        builtins.input = original_input




