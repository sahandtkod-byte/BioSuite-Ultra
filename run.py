#!/usr/bin/env python3
"""
BioSuite – Integrated Bioinformatic Platform Entry Point.
Usage:
    python run.py              (interactive CLI)
    python run.py --gui        (launch modern GUI)
    python run.py --batch --pdf report.pdf   (batch export)
    python run.py --export-folder myplots    (export all to folder)
    python run.py --config      (edit config)
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bioplatter.core.utils import config, save_config, set_theme, load_session
from bioplatter.plotting.biological_plots import export_all_to_folder, generate_markdown_story, batch_export_to_pdf
from bioplatter.cli.menu import main_cli

def main():
    parser = argparse.ArgumentParser(description="BioSuite – Bioinformatics Platform")
    parser.add_argument("--gui", action="store_true", help="Launch modern graphical interface")
    parser.add_argument("--batch", action="store_true", help="Batch export all plots to PDF")
    parser.add_argument("--pdf", type=str, default="bioplatter_report.pdf", help="PDF filename for batch")
    parser.add_argument("--export-folder", type=str, help="Export all plots to a folder")
    parser.add_argument("--story", action="store_true", help="Generate markdown story after export")
    parser.add_argument("--config", action="store_true", help="Edit configuration")
    parser.add_argument("--quiet", action="store_true", help="Run quietly (no prompts)")
    args = parser.parse_args()

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
