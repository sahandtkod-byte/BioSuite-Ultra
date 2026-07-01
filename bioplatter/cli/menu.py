"""
Command-line interactive menu for BioSuite.
Uses the modularized plot functions from bioplatter.plotting.
"""
import sys
import os
import matplotlib.pyplot as plt
from ..core.utils import config, set_theme, save_config
from ..plotting.biological_plots import (
    volcano_plot, pca_plot, manhattan_plot, ma_plot, venn_diagram,
    barplot_custom, boxplot_custom, heatmap_custom, scatter_custom, timeseries_plot,
    qq_plot, clustered_heatmap, circos_plot, alignment_viewer,
    violin_plot, raincloud_plot, ridge_plot, dot_plot,
    export_all_to_folder, generate_markdown_story, batch_export_to_pdf
)
from ..plotting.math_plots import sine_plot, cosine_plot, linear_plot, quadratic_plot, cubic_plot, exponential_plot, logistic_plot
from ..plotting.specialized_plots import gsea_plot, motif_logo, sankey_diagram, umap_plot

def show_help():
    help_text = """
    BioSuite CLI Help
    =================
    1-29 : Select plot
    30   : Change theme
    31   : Export all to PDF
    32   : Export all to folder
    33   : Generate markdown story
    34   : Launch GUI
    0    : Exit
    """
    print(help_text)

def print_menu():
    pre = '\033[96m' if config['theme'] == 'light' else '\033[92m'
    bold = '\033[1m'
    reset = '\033[0m'
    print(pre + bold + "\n" + "="*64)
    print("       BIO-SUITE PRO v2.0 (Integrated)  |  CLI Mode")
    print("="*64 + reset)
    print(pre + "\n ADVANCED BIOLOGICAL")
    print("   1. Volcano Plot      2. PCA Plot         3. Manhattan Plot")
    print("   4. MA Plot           5. Venn Diagram")
    print(pre + "\n BASIC BIOLOGICAL")
    print("   6. Barplot           7. Boxplot          8. Heatmap")
    print("   9. Scatter          10. Time Series")
    print(pre + "\n MATHEMATICAL")
    print("  11. Sine             12. Cosine          13. Linear")
    print("  14. Quadratic        15. Cubic           16. Exponential")
    print("  17. Logistic")
    print(pre + "\n SPECIALIZED")
    print("  18. GSEA Plot        19. Motif Logo      20. Sankey Diagram")
    print(pre + "\n ADDITIONAL PLOTS")
    print("  21. QQ-plot          22. Clustered Heatmap")
    print("  23. Circos Plot      24. Alignment Viewer")
    print("  25. UMAP Plot (optional)")
    print(pre + "\n NEW PLOTS")
    print("  26. Violin Plot      27. Raincloud Plot")
    print("  28. Ridge Plot       29. Dot Plot")
    print(pre + "\n UTILITIES")
    print("  30. Change Theme     31. Export all to PDF (batch)")
    print("  32. Export all to Folder   33. Generate Markdown Story")
    print("  34. Launch GUI")
    print("  35. Pairwise Alignment (Needleman-Wunsch)")
    print("  36. Pairwise Alignment (Smith-Waterman)")
    print("  37. Build Phylogenetic Tree (from aligned FASTA)")
    print("  38. Load VCF & Manhattan Plot")
    print("  39. Differential Expression (from count matrix)")
    print("   0. Exit")
    print(pre + "="*64 + reset)

def main_cli():
    """Run the command-line interactive menu."""
    while True:
        print_menu()
        choice = input("Select (0-39, h for help): ").strip().lower()
        if choice == '0':
            print("Goodbye!")
            break
        elif choice == 'h':
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
            if not pdf_file: pdf_file = "bioplatter_report.pdf"
            batch_export_to_pdf(pdf_file)
        elif choice == '32':
            folder = input("Folder name [bioplatter_export]: ").strip()
            if not folder: folder = "bioplatter_export"
            export_all_to_folder(folder)
        elif choice == '33':
            folder = input("Folder containing images [bioplatter_export]: ").strip()
            if not folder: folder = "bioplatter_export"
            plot_names = ["volcano", "pca", "manhattan", "maplot", "venn",
                         "barplot", "boxplot", "heatmap", "scatter", "timeseries",
                         "sine", "cosine", "linear", "quadratic", "cubic", "exponential", "logistic",
                         "gsea", "motif_logo", "sankey",
                         "qqplot", "clustered_heatmap", "circos", "alignment", "umap",
                         "violin", "raincloud", "ridge", "dotplot"]
            generate_markdown_story(plot_names, os.path.join(folder, "story.md"))
        elif choice == '34':
            from ..gui.main_window import BioSuiteApp
            app = BioSuiteApp()
            app.mainloop()
        elif choice == '35':
            s1 = input("Sequence 1: ").strip()
            s2 = input("Sequence 2: ").strip()
            from ..core.alignment import needleman_wunsch
            a1, a2, sc = needleman_wunsch(s1, s2)
            print(f"Score: {sc}\n{a1}\n{a2}")
        elif choice == '36':
            s1 = input("Sequence 1: ").strip()
            s2 = input("Sequence 2: ").strip()
            from ..core.alignment import smith_waterman
            a1, a2, sc = smith_waterman(s1, s2)
            print(f"Score: {sc}\n{a1}\n{a2}")
        elif choice == '37':
            filepath = input("FASTA file with aligned sequences: ").strip()
            from ..core.sequence import read_fasta
            from ..core.phylogeny import distance_matrix, upgma_tree, plot_phylogenetic_tree
            seqs = read_fasta(filepath)
            if seqs is None:
                print("Could not read file.")
                continue
            names = [n for n,_ in seqs]
            sequences = [s for _,s in seqs]
            dist = distance_matrix(sequences)
            link = upgma_tree(dist, names)
            plot_phylogenetic_tree(link, names)
            plt.show()
        elif choice == '38':
            filepath = input("VCF file path: ").strip()
            from ..core.ngs import read_vcf, manhattan_from_vcf
            df = read_vcf(filepath)
            if df is not None:
                mh = manhattan_from_vcf(df)
                plt.scatter(mh['POS'], mh['neg_log10'], s=2)
                plt.show()
        elif choice == '39':
            filepath = input("Count matrix file: ").strip()
            cond = input("Conditions (comma-sep): ").strip().split(',')
            from ..core.expression import read_counts_matrix, differential_expression
            counts = read_counts_matrix(filepath)
            if counts is not None:
                res = differential_expression(counts, cond)
                print(res.head(20))
        else:
            print("Invalid choice. Type 'h' for help.")
