#!/usr/bin/env python3
"""
BioSuite – Integrated Bioinformatic Platform Entry Point.

Usage:
    python run.py                        Launch interactive CLI menu
    python run.py --gui                  Launch modern graphical interface
    python run.py --batch --pdf report.pdf   Batch export all plots to PDF
    python run.py --export-folder myplots    Export all plots to a folder
    python run.py --config                Edit configuration
    python run.py --benchmark             Run performance benchmarks
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bioplatter.core.utils import config, save_config, set_theme, load_session
from bioplatter.plotting.biological_plots import export_all_to_folder, generate_markdown_story, batch_export_to_pdf
from bioplatter.cli.menu import main_cli


def run_benchmark():
    """Run performance benchmarks for core algorithms.

    Tests alignment and differential expression performance at various
    input sizes to demonstrate the speedup from vectorized implementations.
    """
    import numpy as np
    import pandas as pd
    from bioplatter.core.alignment import needleman_wunsch, smith_waterman
    from bioplatter.core.expression import differential_expression

    print("\n" + "=" * 60)
    print("  BioSuite Performance Benchmark")
    print("=" * 60)

    # ── Alignment Benchmark ──
    print("\n--- Needleman-Wunsch (Global Alignment) ---")
    print(f"{'Length':>8}  {'Time (ms)':>10}  {'Score':>8}")
    print("-" * 35)
    for length in [50, 100, 200, 500, 1000]:
        np.random.seed(42)
        s1 = ''.join(np.random.choice(list('ACGT'), length))
        s2 = ''.join(np.random.choice(list('ACGT'), length))
        start = time.perf_counter()
        _, _, score = needleman_wunsch(s1, s2)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{length:>8}  {elapsed:>9.2f}  {score:>8}")

    print("\n--- Smith-Waterman (Local Alignment) ---")
    print(f"{'Length':>8}  {'Time (ms)':>10}  {'Score':>8}")
    print("-" * 35)
    for length in [50, 100, 200, 500, 1000]:
        np.random.seed(42)
        s1 = ''.join(np.random.choice(list('ACGT'), length))
        s2 = ''.join(np.random.choice(list('ACGT'), length))
        start = time.perf_counter()
        _, _, score = smith_waterman(s1, s2)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{length:>8}  {elapsed:>9.2f}  {score:>8}")

    # ── Differential Expression Benchmark ──
    print("\n--- Differential Expression (Vectorized t-test) ---")
    print(f"{'Genes':>8}  {'Samples':>8}  {'Time (ms)':>10}")
    print("-" * 35)
    for n_genes in [100, 500, 1000, 5000, 10000]:
        np.random.seed(42)
        n_samples = 6
        data = {
            'gene': [f'G{i}' for i in range(n_genes)],
            **{f'S{i}': np.random.randint(100, 10000, n_genes) for i in range(n_samples)}
        }
        df = pd.DataFrame(data)
        conditions = ['ctrl'] * 3 + ['treat'] * 3
        start = time.perf_counter()
        result = differential_expression(df, conditions)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{n_genes:>8}  {n_samples:>8}  {elapsed:>9.2f}")

    print("\n" + "=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BioSuite – Bioinformatics Platform")
    parser.add_argument("--gui", action="store_true", help="Launch modern graphical interface")
    parser.add_argument("--batch", action="store_true", help="Batch export all plots to PDF")
    parser.add_argument("--pdf", type=str, default="bioplatter_report.pdf", help="PDF filename for batch")
    parser.add_argument("--export-folder", type=str, help="Export all plots to a folder")
    parser.add_argument("--story", action="store_true", help="Generate markdown story after export")
    parser.add_argument("--config", action="store_true", help="Edit configuration")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--quiet", action="store_true", help="Run quietly (no prompts)")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        return

    if args.config:
        print("Themes: dark-green, dark-purple, light-blue")
        new_theme = input(f"Theme [{config['theme']}]: ").strip()
        if new_theme in ['dark-green', 'dark-purple', 'light-blue']:
            config['theme'] = new_theme
        new_dpi = input(f"DPI [{config['default_dpi']}]: ").strip()
        if new_dpi.isdigit():
            config['default_dpi'] = int(new_dpi)
        config['quiet'] = args.quiet
        save_config(config)
        print("Config saved.")
        return

    if args.quiet:
        config['quiet'] = True

    set_theme(config['theme'])

    if args.gui:
        from bioplatter.gui.main_window import BioSuiteApp
        app = BioSuiteApp()
        app.mainloop()
    elif args.export_folder:
        export_all_to_folder(args.export_folder)
        if args.story:
            plot_names = ["volcano", "pca", "manhattan", "maplot", "venn",
                         "barplot", "boxplot", "heatmap", "scatter", "timeseries",
                         "sine", "cosine", "linear", "quadratic", "cubic", "exponential", "logistic",
                         "gsea", "motif_logo", "sankey",
                         "qqplot", "clustered_heatmap", "circos", "alignment", "umap",
                         "violin", "raincloud", "ridge", "dotplot"]
            generate_markdown_story(plot_names, os.path.join(args.export_folder, "story.md"))
    elif args.batch:
        batch_export_to_pdf(args.pdf)
    else:
        main_cli()


if __name__ == "__main__":
    main()
