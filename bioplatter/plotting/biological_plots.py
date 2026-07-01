"""All biological plots: Volcano, PCA, Manhattan, MA, Venn, Barplot, Boxplot, Heatmap, Scatter, Timeseries, QQ, ClusteredHeatmap, Circos, Alignment, Violin, Raincloud, Ridge, Dot."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from matplotlib.patches import FancyBboxPatch, Wedge, Circle, Arc, Rectangle, ConnectionPatch
from matplotlib.path import Path
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from ..core.utils import (config, session, autosave_session, safe_float_input, safe_int_input,
                          safe_list_input, load_dataframe_safe, maybe_downsample, apply_glass_ax,
                          ask_save_plot, report_boxplot_stats, report_scatter_stats,
                          report_volcano_stats, report_pca_stats, report_manhattan_stats,
                          add_ttest_to_boxplot, add_regression_eq, _interrupted)

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
        if src_node in source_nodes:
            src_y = (source_nodes.index(src_node)+1)/(len(source_nodes)+1)
        else:
            continue
        if tgt_node in target_nodes:
            tgt_y = (target_nodes.index(tgt_node)+1)/(len(target_nodes)+1)
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
        ax.set_xlabel('Log2 Fold Change')
        ax.set_ylabel('-log10(P-value)')
        ax.set_title('Volcano Plot')
        ax.legend(framealpha=0.7)
        apply_glass_ax(ax)
        ask_save_plot('volcano', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

def pca_plot(pdf=None):
    from sklearn.decomposition import PCA
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
                        groups = df[group_col].astype(str).values if group_col and group_col in df.columns else ['Sample']*data.shape[0]
                        pca = PCA(n_components=2)
                        pc = pca.fit_transform(data)
                        report_pca_stats(pca)
                        df_pca = pd.DataFrame({'PC1': pc[:,0], 'PC2': pc[:,1], 'Group': groups})
                        fig, ax = plt.subplots(figsize=(7,6))
                        sns.scatterplot(x='PC1', y='PC2', hue='Group', data=df_pca, s=70, palette='Set1', ax=ax)
                        ax.set_title(f'PCA (var: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})')
                        apply_glass_ax(ax)
                        ask_save_plot('pca', config['save_format'], config['default_dpi'], pdf)
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
        ask_save_plot('pca', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
        ax.set_ylabel('M (log2 fold change)')
        ax.set_title('MA Plot')
        apply_glass_ax(ax)
        ask_save_plot('maplot', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
        ask_save_plot('venn', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
        ask_save_plot('barplot', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
                    if len(df_plot[group_col].unique()) >= 2:
                        add_ttest_to_boxplot(df_plot, group_col, value_col, ax)
                    ask_save_plot('boxplot', config['save_format'], config['default_dpi'], pdf)
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
        ask_save_plot('boxplot', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
                    ask_save_plot('heatmap', config['save_format'], config['default_dpi'], pdf)
                    plt.show()
                    return
        corr = np.array([[1,0.8,0.2,0.1],[0.8,1,0.3,0.2],[0.2,0.3,1,0.7],[0.1,0.2,0.7,1]])
        labels = ['Gene_A','Gene_B','Gene_C','Gene_D']
        fig, ax = plt.subplots(figsize=(7,6))
        sns.heatmap(pd.DataFrame(corr, index=labels, columns=labels), annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Gene Correlation')
        apply_glass_ax(ax)
        ask_save_plot('heatmap', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

def scatter_custom(pdf=None):
    print("\n--- Scatter Plot ---")
    try:
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
        if corr_type == 'spearman':
            corr, pval = sp_stats.spearmanr(x, y)
            print(f"\nScatter Statistics: Spearman rho = {corr:.3f}, p-value = {pval:.4f}")
            eq = f'Spearman rho = {corr:.3f}, p = {pval:.4f}'
            ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        else:
            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
            print(f"\nScatter Statistics: Pearson r = {r_value:.3f}, p-value = {p_value:.4f}")
            eq = f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}, p = {p_value:.4f}'
            ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        ask_save_plot('scatter', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
                        ask_save_plot('timeseries', config['save_format'], config['default_dpi'], pdf)
                        plt.show()
                        return
        times = np.arange(0,24,2)
        values = np.exp(-0.1*times) + np.random.normal(0,0.05,len(times))
        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(x=times, y=values, marker='o', color='green', ax=ax)
        ax.set_title('Time Series')
        apply_glass_ax(ax)
        ask_save_plot('timeseries', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
        sp_stats.probplot(data, dist="norm", plot=ax)
        ax.set_title('QQ-plot (Normal Distribution)')
        apply_glass_ax(ax)
        ask_save_plot('qqplot', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
            data = data.sample(2000)
            print(f"Downsampled to {data.shape[0]} rows for heatmap.")
        g = sns.clustermap(data, cmap='coolwarm', standard_scale=1,
                           figsize=(10, 8), dendrogram_ratio=0.2,
                           cbar_pos=(0.02, 0.8, 0.03, 0.18))
        g.ax_heatmap.set_title('Clustered Heatmap')
        ask_save_plot('clustered_heatmap', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
        sector_names = list(sectors.keys())
        for (name, size), color in zip(sectors.items(), colors):
            end = start + 2*np.pi * size/total
            ax.bar(x=start, height=0.5, width=end-start, bottom=0.2,
                   color=color, edgecolor='black', alpha=0.7, align='edge')
            ax.text(start + (end-start)/2, 0.8, name, ha='center', va='center', fontsize=8)
            start = end
        for (s1, s2) in links:
            if s1 not in sector_names or s2 not in sector_names:
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
        ax.set_ylim(0,1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Simple Circos Plot")
        apply_glass_ax(ax)
        ask_save_plot('circos', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

def alignment_viewer(pdf=None):
    print("\n--- Alignment Viewer ---")
    try:
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
        ask_save_plot('alignment', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
                    ask_save_plot('violin', config['save_format'], config['default_dpi'], pdf)
                    plt.show()
                    return
        ctrl = np.random.normal(5,1,50)
        treat = np.random.normal(7.5,1.2,50)
        df_plot = pd.DataFrame({'Group': ['Ctrl']*50 + ['Treat']*50, 'Value': np.concatenate([ctrl, treat])})
        fig, ax = plt.subplots(figsize=(8,6))
        sns.violinplot(x='Group', y='Value', data=df_plot, inner='quartile', palette='muted')
        sns.stripplot(x='Group', y='Value', data=df_plot, color='black', alpha=0.5, size=3)
        ax.set_title('Violin Plot (Demo)')
        add_ttest_to_boxplot(df_plot, 'Group', 'Value', ax)
        apply_glass_ax(ax)
        ask_save_plot('violin', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
                    sns.violinplot(x=group_col, y=value_col, data=df_plot, inner=None, palette='muted', alpha=0.5)
                    sns.boxplot(x=group_col, y=value_col, data=df_plot, width=0.2, boxprops=dict(alpha=0.5), showfliers=False)
                    sns.swarmplot(x=group_col, y=value_col, data=df_plot, color='black', alpha=0.6, size=4)
                    ax.set_title('Raincloud Plot')
                    if len(df_plot[group_col].unique()) >= 2:
                        add_ttest_to_boxplot(df_plot, group_col, value_col, ax)
                    apply_glass_ax(ax)
                    ask_save_plot('raincloud', config['save_format'], config['default_dpi'], pdf)
                    plt.show()
                    return
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
        ask_save_plot('raincloud', config['save_format'], config['default_dpi'], pdf)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

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
                        density = sp_stats.gaussian_kde(subset)
                        xs = np.linspace(subset.min(), subset.max(), 200)
                        ys = density(xs) + i
                        ax.fill_between(xs, ys, i, alpha=0.5, label=grp)
                    ax.set_xlabel(value_col)
                    ax.set_ylabel('Density shift')
                    ax.set_title('Ridge Plot')
                    apply_glass_ax(ax)
                    ask_save_plot('ridge', config['save_format'], config['default_dpi'], pdf)
                    plt.show()
                    return
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
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

def dot_plot(pdf=None):
    print("\n--- Dot Plot (Single-cell style) ---")
    try:
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
                    genes = df[gene_col].unique()
                    clusters = df[cluster_col].unique()
                    pivot_pct = df.pivot(index=gene_col, columns=cluster_col, values=pct_col).fillna(0)
                    pivot_exp = df.pivot(index=gene_col, columns=cluster_col, values=exp_col).fillna(0)
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
                    plt.show()
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
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

def export_all_to_folder(folder_name="bioplatter_export"):
    import builtins
    from .math_plots import (sine_plot, cosine_plot, linear_plot,
                              quadratic_plot, cubic_plot, exponential_plot, logistic_plot)
    from .specialized_plots import gsea_plot, motif_logo, sankey_diagram, umap_plot
    print(f"\nExporting all plots to folder: {folder_name}")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    original_format = config['save_format']
    original_quiet = config.get('quiet', False)
    config['save_format'] = 'png'
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
        if "Use default" in prompt or "Use default alignment" in prompt:
            return 'y'
        if "Save as HTML" in prompt or "Save this plot?" in prompt:
            return 'n'
        if "Enter sequences" in prompt:
            return ''
        if "Switch to" in prompt:
            return ''
        if "Correlation type" in prompt:
            return 'pearson'
        return ''

    original_input = builtins.input
    builtins.input = default_input

    _orig_ask_save = ask_save_plot
    def auto_save_plot(default_name, save_format=None, dpi=None, pdf=None, story=None):
        try:
            plt.tight_layout()
            plt.savefig(f"{default_name}.png", dpi=config['default_dpi'],
                        bbox_inches='tight', facecolor=plt.rcParams['figure.facecolor'])
            print(f"   Auto-saved: {default_name}.png")
        except Exception:
            pass
        plt.close()

    import bioplatter.plotting.biological_plots as bp_mod
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
    for name in plot_names:
        img_path = f"{name}.png"
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append(f"![{name}]({img_path})")
        lines.append("")
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Story report saved to {output_path}")

def batch_export_to_pdf(pdf_path="bioplatter_report.pdf"):
    from .math_plots import (sine_plot, cosine_plot, linear_plot,
                              quadratic_plot, cubic_plot, exponential_plot, logistic_plot)
    from .specialized_plots import gsea_plot, motif_logo, sankey_diagram, umap_plot
    import builtins
    print(f"\nBatch exporting all plots to {pdf_path}...")
    plot_funcs = [
        volcano_plot, pca_plot, manhattan_plot, ma_plot, venn_diagram,
        barplot_custom, boxplot_custom, heatmap_custom, scatter_custom, timeseries_plot,
        sine_plot, cosine_plot, linear_plot, quadratic_plot, cubic_plot,
        exponential_plot, logistic_plot, gsea_plot, motif_logo, sankey_diagram,
        qq_plot, clustered_heatmap, circos_plot, alignment_viewer, umap_plot,
        violin_plot, raincloud_plot, ridge_plot, dot_plot
    ]
    original_input = builtins.input
    def default_input(prompt):
        return ''
    builtins.input = default_input
    original_quiet = config.get('quiet', False)
    config['quiet'] = True
    try:
        with PdfPages(pdf_path) as pdf:
            for func in plot_funcs:
                try:
                    func(pdf=pdf)
                    plt.close()
                except Exception as e:
                    print(f"Warning: {func.__name__} failed: {e}")
        print(f"PDF saved: {pdf_path}")
    finally:
        config['quiet'] = original_quiet
        builtins.input = original_input

from .math_plots import (sine_plot, cosine_plot, linear_plot,
                          quadratic_plot, cubic_plot, exponential_plot, logistic_plot)
from .specialized_plots import gsea_plot, motif_logo, sankey_diagram, umap_plot



