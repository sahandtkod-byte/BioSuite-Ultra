"""
BioSuite Ultra — Professional CLI Interface

Supports both interactive menu mode and non-interactive command mode:
    biosuite                    # Interactive menu
    biosuite --cmd gc ATCGATCG  # Non-interactive
    biosuite --cmd translate ATCGATCG --frame 1
"""
import sys
import os
import argparse
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
from ..plotting.upset_plots import plot_upset, compute_upset_matrix, compute_set_statistics, plot_set_sizes
from ..plotting.genome_browser import plot_genome_tracks, parse_bed, parse_vcf, create_coverage_from_bam, create_bed_track, create_variant_track
from ..plotting.conservation_plots import plot_sequence_logo, plot_conservation_bar, plot_logo_with_conservation, compute_conservation_scores, plot_motif_enrichment
from ..plotting.synteny import plot_dotplot, plot_synteny_dotplot, compute_synteny_score, plot_synteny
from ..core.workflow.pipeline import Pipeline, format_pipeline_report
from ..core.workflow.batch import BatchProcessor, format_batch_report
from ..core.workflow.report import create_report, generate_pipeline_report, generate_batch_report
from ..core.go_browser import GOBrowser, format_go_results
from ..core.pathway_viz import PathwayMap, draw_pathway, create_kegg_style_pathway, format_pathway_report
from ..core.gwas import run_gwas, detect_lead_snps, generate_gwas_data, format_gwas_report
from ..core.epitope import (predict_t_cell_epitopes, predict_b_cell_epitopes,
                              predict_linear_epitopes, format_epitope_report)


G = '\033[92m'  # Green
C = '\033[96m'  # Cyan
Y = '\033[93m'  # Yellow
W = '\033[97m'  # White
B = '\033[1m'   # Bold
D = '\033[2m'   # Dim
R = '\033[0m'   # Reset
M = '\033[95m'  # Magenta


# ── Command Registry for Non-Interactive Mode ─────────────────────────────────
COMMAND_REGISTRY = {
    'gc': {'func': 'biosuite.core.sequence:gc_content', 'desc': 'Calculate GC content', 'args': ['sequence']},
    'revcomp': {'func': 'biosuite.core.sequence:reverse_complement', 'desc': 'Reverse complement', 'args': ['sequence']},
    'translate': {'func': 'biosuite.core.sequence:translate', 'desc': 'Translate DNA to protein', 'args': ['sequence'], 'kwargs': {'frame': 1}},
    'stats': {'func': 'biosuite.core.sequence:sequence_stats', 'desc': 'Sequence composition stats', 'args': ['sequence']},
    'nw': {'func': 'biosuite.core.alignment:needleman_wunsch', 'desc': 'Needleman-Wunsch alignment', 'args': ['seq1', 'seq2']},
    'sw': {'func': 'biosuite.core.alignment:smith_waterman', 'desc': 'Smith-Waterman alignment', 'args': ['seq1', 'seq2']},
    'blast': {'func': 'biosuite.core.blast:run_blast', 'desc': 'BLAST sequence search', 'args': ['query', 'database']},
    'crispr': {'func': 'biosuite.core.crispr:design_guides', 'desc': 'CRISPR guide design', 'args': ['sequence']},
    'epitope': {'func': 'biosuite.core.epitope:predict_t_cell_epitopes', 'desc': 'T-cell epitope prediction', 'args': ['sequence']},
    'gwas': {'func': 'biosuite.core.gwas:run_gwas', 'desc': 'Run GWAS analysis', 'args': ['datafile']},
    'hwe': {'func': 'biosuite.core.popgen:hardy_weinberg_test', 'desc': 'Hardy-Weinberg test', 'args': ['AA', 'Aa', 'aa']},
    'volcano': {'func': 'biosuite.plotting.plot_api:volcano', 'desc': 'Volcano plot', 'args': ['lfc_file', 'pval_file']},
    'pca': {'func': 'biosuite.plotting.plot_api:pca', 'desc': 'PCA plot', 'args': ['datafile']},
    'manhattan': {'func': 'biosuite.plotting.plot_api:manhattan', 'desc': 'Manhattan plot', 'args': ['datafile']},
    'digest': {'func': 'biosuite.core.cloning:simulate_digestion', 'desc': 'Restriction digestion', 'args': ['sequence', 'enzyme']},
    'primers': {'func': 'biosuite.core.cloning:design_primers', 'desc': 'Primer design', 'args': ['sequence']},
    'gui': {'func': '_launch_gui', 'desc': 'Launch GUI application', 'args': []},
    'api': {'func': '_launch_api', 'desc': 'Launch REST API server', 'args': [], 'kwargs': {'port': 8000}},
    'theme': {'func': '_change_theme', 'desc': 'Change theme', 'args': ['theme_name']},
}


def _import_func(func_path):
    """Import a function from 'module:function' path."""
    module_path, func_name = func_path.rsplit(':', 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def _execute_command(cmd_name, args):
    """Execute a registered command with arguments."""
    if cmd_name not in COMMAND_REGISTRY:
        print(f"Unknown command: {cmd_name}")
        print(f"Available commands: {', '.join(sorted(COMMAND_REGISTRY.keys()))}")
        return

    entry = COMMAND_REGISTRY[cmd_name]

    # Handle special commands
    if entry['func'] == '_launch_gui':
        _launch_gui()
        return
    elif entry['func'] == '_launch_api':
        _launch_api(args.port if hasattr(args, 'port') else 8000)
        return
    elif entry['func'] == '_change_theme':
        theme = args.theme_name if hasattr(args, 'theme_name') else 'dark-green'
        set_theme(theme)
        print(f"Theme set to: {theme}")
        return

    # Dynamic import and call
    func = _import_func(entry['func'])
    func_args = [getattr(args, a) for a in entry['args'] if hasattr(args, a)]
    func_kwargs = {}
    for k, v in entry.get('kwargs', {}).items():
        if hasattr(args, k):
            func_kwargs[k] = getattr(args, k)
        else:
            func_kwargs[k] = v

    result = func(*func_args, **func_kwargs)
    if result is not None:
        if hasattr(result, 'to_string'):
            print(result.to_string())
        elif isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print(result)


def _launch_gui():
    """Launch the GUI application."""
    from ..gui.main_window import BioSuiteApp
    app = BioSuiteApp()
    app.mainloop()


def _launch_api(port=8000):
    """Launch the REST API server."""
    try:
        import uvicorn
        from ..api import app
        print(f"Starting API server on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ImportError:
        print("Install uvicorn: pip install uvicorn")


def _change_theme(theme_name):
    """Change the application theme."""
    set_theme(theme_name)
    config['theme'] = theme_name
    save_config(config)
    print(f"Theme changed to: {theme_name}")


def _build_parser():
    """Build argparse parser for non-interactive mode."""
    parser = argparse.ArgumentParser(
        prog='biosuite',
        description='BioSuite Ultra — Bioinformatics Platform v4.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  biosuite                          Interactive menu
  biosuite gc ATCGATCG              GC content
  biosuite translate ATCGATCG       Translate DNA
  biosuite nw AGTACGCA TATGC        Alignment
  biosuite gui                      Launch GUI
  biosuite api --port 8000          Launch API server
  biosuite theme dark-purple        Change theme

Available commands:
  """ + ', '.join(sorted(COMMAND_REGISTRY.keys()))
    )
    parser.add_argument('command', nargs='?', default=None,
                        help='Command to execute (omit for interactive menu)')
    parser.add_argument('cmd_args', nargs='*',
                        help='Arguments for the command')
    parser.add_argument('--gui', action='store_true',
                        help='Launch GUI application')
    parser.add_argument('--api', action='store_true',
                        help='Launch REST API server')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port for API server (default: 8000)')
    parser.add_argument('--theme', type=str, default=None,
                        help='Theme: dark-green, dark-purple, light-blue')
    parser.add_argument('--version', action='version', version='BioSuite Ultra 4.2.4')
    return parser


def _header():
    print()
    print(f"{G}{B}{'═'*68}{R}")
    print(f"{G}{B}   ██████╗ ██╗   ██╗██╗███████╗███╗   ██╗ ██████╗███████╗{R}")
    print(f"{G}{B}   ██╔══██╗╚██╗ ██╔╝██║██╔════╝████╗  ██║██╔════╝██╔════╝{R}")
    print(f"{G}{B}   ██████╔╝ ╚████╔╝ ██║█████╗  ██╔██╗ ██║██║     █████╗  {R}")
    print(f"{G}{B}   ██╔══██╗  ╚██╔╝  ██║██╔══╝  ██║╚██╗██║██║     ██╔══╝  {R}")
    print(f"{G}{B}   ██████╔╝   ██║   ██║███████╗██║ ╚████║╚██████╗███████╗{R}")
    print(f"{G}{B}   ╚═════╝    ╚═╝   ╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝{R}")
    print(f"{D}   Bioinformatic Platform  ·  Ultra v4.2.2  ·  Pure Python{R}")
    print(f"{G}{B}{'═'*68}{R}")


def _section(title, color=C):
    print()
    print(f"  {color}{B}┌─ {title} {'─' * (58 - len(title))}┐{R}")
    print(f"  {color}│{R}")


def _end_section():
    print(f"  {C}└{'─'*62}┘{R}")


def _item(num, name, desc="", color=W):
    if desc:
        print(f"  {C}│{R}  {B}{G}{num:>3}{R}  {color}{name:<28}{R} {D}{desc}{R}")
    else:
        print(f"  {C}│{R}  {B}{G}{num:>3}{R}  {color}{name}{R}")


def _col2(n1, n2, c1="", c2="", d1="", d2=""):
    p1 = f"{n1:<26}" + (f" {D}{d1}{R}" if d1 else "")
    p2 = f"{n2:<26}" + (f" {D}{d2}{R}" if d2 else "")
    print(f"  {C}│{R}  {G}{n1:>3}{R} {c1}  {G}{n2:>3}{R} {c2}")


def print_menu():
    _header()

    _section("VISUALIZATION  ·  Statistical & Biological Plots")
    print(f"  {C}│{R}  {B}{G} 1{R} Volcano Plot          {B}{G} 2{R} PCA Plot             {B}{G} 3{R} Manhattan Plot")
    print(f"  {C}│{R}  {B}{G} 4{R} MA Plot               {B}{G} 5{R} Venn Diagram         {B}{G} 6{R} Barplot")
    print(f"  {C}│{R}  {B}{G} 7{R} Boxplot               {B}{G} 8{R} Correlation Heatmap  {B}{G} 9{R} Scatter + Regression")
    print(f"  {C}│{R}  {B}{G}10{R} Time Series           {B}{G}11{R} Violin Plot          {B}{G}12{R} Raincloud Plot")
    print(f"  {C}│{R}  {B}{G}13{R} Ridge Plot            {B}{G}14{R} Dot Plot (SC)        {B}{G}15{R} QQ-Plot (Normality)")
    print(f"  {C}│{R}  {B}{G}16{R} Clustered Heatmap     {B}{G}17{R} Circos Plot          {B}{G}18{R} Alignment Viewer")
    _end_section()

    _section("SEQUENCE ANALYSIS  ·  I/O, Composition & Translation", M)
    print(f"  {C}│{R}  {B}{G}20{R} BLAST Sequence Search     {B}{G}21{R} Multiple Seq Alignment")
    print(f"  {C}│{R}  {B}{G}22{R} Needleman-Wunsch (Global) {B}{G}23{R} Smith-Waterman (Local)")
    print(f"  {C}│{R}  {B}{G}24{R} Sequence Statistics        {B}{G}25{R} Reverse Complement")
    print(f"  {C}│{R}  {B}{G}26{R} Translate (6 frames)      {B}{G}27{R} GC% Calculator")
    print(f"  {C}│{R}  {B}{G}28{R} Read FASTA/FASTQ/GenBank")
    _end_section()

    _section("TRANSCRIPTOMICS  ·  RNA-seq & Expression Analysis", Y)
    print(f"  {C}│{R}  {B}{G}30{R} FASTQ Quality Trimming     {B}{G}31{R} RNA-seq Quantification")
    print(f"  {C}│{R}  {B}{G}32{R} Differential Expression    {B}{G}33{R} CPM Normalization")
    print(f"  {C}│{R}  {B}{G}34{R} TPM Normalization          {B}{G}35{R} GO / KEGG Enrichment")
    _end_section()

    _section("GENOMICS & NGS  ·  Read Mapping, Variants & Assembly", G)
    print(f"  {C}│{R}  {B}{G}40{R} Short Read Alignment (BWA)  {B}{G}41{R} Variant Calling (FreeBayes)")
    print(f"  {C}│{R}  {B}{G}42{R} ChIP-seq Peak Calling       {B}{G}43{R} Genome Assembly (SPAdes)")
    print(f"  {C}│{R}  {B}{G}44{R} Phylogenetic Tree (NJ/UPGMA) {B}{G}45{R} ML Phylogeny (RAxML)")
    print(f"  {C}│{R}  {B}{G}46{R} Bayesian Phylogeny (MrBayes) {B}{G}47{R} Manhattan from VCF")
    _end_section()

    _section("SINGLE-CELL & PROTEOMICS  ·  scRNA-seq & Structures", M)
    print(f"  {C}│{R}  {B}{G}50{R} Single-Cell RNA-seq (Scanpy)  {B}{G}51{R} Protein Structure (PDB)")
    print(f"  {C}│{R}  {B}{G}52{R} Structure Prediction (ESM)     {B}{G}53{R} Molecular Docking (Vina)")
    _end_section()

    _section("SPECIALIZED TOOLS  ·  Metagenomics, CRISPR, ML & More", Y)
    print(f"  {C}│{R}  {B}{G}55{R} Metagenomics Pipeline       {B}{G}56{R} CRISPR Guide RNA Design")
    print(f"  {C}│{R}  {B}{G}57{R} Flux Balance Analysis       {B}{G}58{R} Population Genetics")
    print(f"  {C}│{R}  {B}{G}59{R} Epigenomics (Methylation)   {B}{G}60{R} Metabolomics Analysis")
    print(f"  {C}│{R}  {B}{G}61{R} Molecular Dynamics (MD)     {B}{G}62{R} Machine Learning (RF/SVM)")
    _end_section()

    _section("ADVANCED VISUALIZATION  ·  UpSet, Genome Browser, Logos, Synteny", M)
    print(f"  {C}│{R}  {B}{G}77{R} UpSet Plot (Multi-set)     {B}{G}78{R} Genome Browser Viewer")
    print(f"  {C}│{R}  {B}{G}79{R} Interactive Scatter         {B}{G}80{R} Sequence Logo")
    print(f"  {C}│{R}  {B}{G}81{R} Synteny Dotplot             ")
    _end_section()

    _section("SEQUENCE TOOLS  ·  ORF, Primers, Restriction Enzymes", C)
    print(f"  {C}│{R}  {B}{G}63{R} ORF Finder                  {B}{G}64{R} Restriction Enzyme Mapper")
    print(f"  {C}│{R}  {B}{G}65{R} Primer Design               {B}{G}66{R} K-mer Counter")
    _end_section()

    _section("DATABASE SEARCH  ·  NCBI, UniProt, PDB, KEGG", M)
    print(f"  {C}│{R}  {B}{G}67{R} NCBI Entrez Search          {B}{G}68{R} UniProt Protein Search")
    print(f"  {C}│{R}  {B}{G}69{R} PDB Structure Search        {B}{G}70{R} KEGG Pathway Search")
    print(f"  {C}│{R}  {B}{G}71{R} Ensembl Gene Search         {B}{G}72{R} Search All Databases")
    _end_section()

    _section("FILE FORMATS  ·  BED, GFF, Newick, Stockholm", G)
    print(f"  {C}│{R}  {B}{G}73{R} Parse BED File              {B}{G}74{R} Parse GFF/GTF File")
    print(f"  {C}│{R}  {B}{G}75{R} Parse Newick Tree           {B}{G}76{R} Parse Stockholm Alignment")
    _end_section()

    _section("ADVANCED PLOTS  ·  Mathematical & Specialized", C)
    print(f"  {C}│{R}  {B}{G}101{R} GSEA Running Sum     {B}{G}102{R} Sequence Logo     {B}{G}103{R} Sankey Diagram")
    print(f"  {C}│{R}  {B}{G}104{R} UMAP Projection      {B}{G}105{R} Sine              {B}{G}106{R} Cosine")
    print(f"  {C}│{R}  {B}{G}107{R} Linear              {B}{G}108{R} Quadratic         {B}{G}109{R} Cubic")
    print(f"  {C}│{R}  {B}{G}110{R} Exponential          {B}{G}111{R} Logistic")
    _end_section()

    _section("ANALYSIS  ·  Codon Usage, Survival, Network", Y)
    print(f"  {C}│{R}  {B}{G}120{R} Codon Usage Table       {B}{G}121{R} K-mer Composition")
    print(f"  {C}│{R}  {B}{G}122{R} Sequence Complexity    {B}{G}123{R} Survival Analysis (KM)")
    print(f"  {C}│{R}  {B}{G}124{R} Network Visualization   {B}{G}125{R} Network Statistics")
    _end_section()

    _section("WORKFLOW & DOMAIN  ·  Pipeline, GWAS, Epitope, GO", M)
    print(f"  {C}│{R}  {B}{G}92{R} Pipeline Builder          {B}{G}93{R} Batch Processor")
    print(f"  {C}│{R}  {B}{G}94{R} HTML Report Generator     {B}{G}95{R} GO Browser")
    print(f"  {C}│{R}  {B}{G}96{R} Pathway Visualization     {B}{G}97{R} GWAS Analysis")
    print(f"  {C}│{R}  {B}{G}98{R} Epitope Prediction        ")
    _end_section()

    _section("GENOMICS TOOLS  ·  16S, SV/CNV, BigWig", Y)
    print(f"  {C}│{R}  {B}{G} a{R} 16S rRNA Pipeline         {B}{G} b{R} SV / CNV Detection")
    print(f"  {C}│{R}  {B}{G} c{R} BigWig Reader             ")
    _end_section()

    _section("MOLECULAR CLONING  ·  Plasmid Maps, Restriction Digest, PCR", G)
    print(f"  {C}│{R}  {B}{G} l{R} Plasmid Map Viewer       {B}{G} m{R} Restriction Digest")
    print(f"  {C}│{R}  {B}{G} n{R} Virtual Gel Electrophoresis {B}{G} o{R} PCR Simulation")
    print(f"  {C}│{R}  {B}{G} p{R} Sequence Viewer          {B}{G} q{R} Ligation Simulator")
    _end_section()

    _section("PHASE 6  ·  Interactive Plots, Provenance, Advanced DE", M)
    print(f"  {C}│{R}  {B}{G} d{R} Interactive Plot (Plotly) {B}{G} e{R} Provenance Tracker")
    print(f"  {C}│{R}  {B}{G} f{R} Plugin Manager            {B}{G} g{R} File Format Detector")
    print(f"  {C}│{R}  {B}{G} h{R} DESeq2 Normalization      {B}{G} i{R} VST Transformation")
    print(f"  {C}│{R}  {B}{G} j{R} Negative Binomial DE      {B}{G} k{R} Universal File Reader")
    _end_section()

    _section("SYSTEM  ·  Configuration & Export", W)
    print(f"  {C}│{R}  {B}{G}89{R} Launch GUI             {B}{G}90{R} Launch API Server")
    print(f"  {C}│{R}  {B}{G}99{R} Change Theme           {B}{G} 0{R} Exit")
    _end_section()
    print()


def main_cli(argv=None):
    """Main CLI entry point. Supports interactive menu and non-interactive commands."""
    parser = _build_parser()

    # Non-interactive mode: parse arguments
    if argv is not None or len(sys.argv) > 1:
        args = parser.parse_args(argv)
        if args.theme:
            _change_theme(args.theme)
        if args.gui:
            _launch_gui()
            return
        if args.api:
            _launch_api(args.port)
            return
        if args.command:
            # Create a namespace with cmd_args as individual attrs
            cmd_args = argparse.Namespace()
            for i, a in enumerate(args.cmd_args):
                setattr(cmd_args, f"arg{i}", a)
            # Also set them by name if command has defined args
            if args.command in COMMAND_REGISTRY:
                entry = COMMAND_REGISTRY[args.command]
                for i, arg_name in enumerate(entry.get("args", [])):
                    if i < len(args.cmd_args):
                        setattr(cmd_args, arg_name, args.cmd_args[i])
            _execute_command(args.command, cmd_args)
        return

    # Interactive mode
    while True:
        print_menu()
        choice = input(f"  {G}{B}▸ Select option:{R} ").strip().lower()

        if choice == '0':
            print(f"\n  {G}Goodbye! {D}Thank you for using BioSuite Ultra.{R}\n")
            break

        # ── Visualization (1-18) ──
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
        elif choice == '11': violin_plot()
        elif choice == '12': raincloud_plot()
        elif choice == '13': ridge_plot()
        elif choice == '14': dot_plot()
        elif choice == '15': qq_plot()
        elif choice == '16': clustered_heatmap()
        elif choice == '17': circos_plot()
        elif choice == '18': alignment_viewer()

        # ── Sequence Analysis (20-28) ──
        elif choice == '20':
            from ..core.blast import run_blast, format_blast_result
            q = input("  Query FASTA: ").strip()
            db = input("  Database FASTA: ").strip()
            if q and db:
                result = run_blast(q, db)
                print(format_blast_result(result))

        elif choice == '21':
            from ..core.msa import auto_align, format_alignment, consensus_sequence, read_fasta_for_msa
            fp = input("  FASTA file: ").strip()
            seqs = read_fasta_for_msa(fp)
            if seqs:
                result = auto_align(seqs)
                print(format_alignment(result))
                print(f"  Consensus: {consensus_sequence(result)}")

        elif choice == '22':
            s1 = input("  Sequence 1: ").strip()
            s2 = input("  Sequence 2: ").strip()
            from ..core.alignment import needleman_wunsch
            a1, a2, sc = needleman_wunsch(s1, s2)
            print(f"  Score: {sc}\n  {a1}\n  {a2}")

        elif choice == '23':
            s1 = input("  Sequence 1: ").strip()
            s2 = input("  Sequence 2: ").strip()
            from ..core.alignment import smith_waterman
            a1, a2, sc = smith_waterman(s1, s2)
            print(f"  Score: {sc}\n  {a1}\n  {a2}")

        elif choice == '24':
            from ..core.sequence import sequence_stats
            seq = input("  DNA sequence: ").strip().upper()
            for k, v in sequence_stats(seq).items():
                print(f"    {k}: {v}")

        elif choice == '25':
            from ..core.sequence import reverse_complement
            print(f"  Result: {reverse_complement(input('  DNA: ').strip())}")

        elif choice == '26':
            from ..core.sequence import translate
            seq = input("  DNA: ").strip()
            frame = int(input("  Frame (1-3, -1 to -3) [1]: ") or "1")
            print(f"  Protein: {translate(seq, frame=frame)}")

        elif choice == '27':
            from ..core.sequence import gc_content, sequence_stats
            seq = input("  DNA: ").strip()
            print(f"  GC%: {gc_content(seq):.2f}%")
            s = sequence_stats(seq)
            print(f"  A:{s['A']} T:{s['T']} G:{s['G']} C:{s['C']} N:{s['N']}")

        elif choice == '28':
            from ..core.sequence import read_fasta
            fp = input("  FASTA file: ").strip()
            seqs = read_fasta(fp)
            if seqs:
                for name, seq in seqs:
                    print(f"  >{name} ({len(seq)} bp)")

        # ── Transcriptomics (30-35) ──
        elif choice == '30':
            from ..core.trimming import trim_fastq, format_trim_report
            infile = input("  Input FASTQ: ").strip()
            out = input("  Output [auto]: ").strip() or None
            qual = int(input("  Quality [20]: ") or "20")
            ml = int(input("  Min length [36]: ") or "36")
            print(format_trim_report(trim_fastq(infile, out, qual, ml)))

        elif choice == '31':
            from ..core.quantification import quantify_reads, format_quant_report
            reads = input("  Reads FASTQ: ").strip()
            trans = input("  Transcriptome FASTA: ").strip()
            sample = input("  Sample name [s1]: ").strip() or "s1"
            print(format_quant_report(quantify_reads(reads, trans, sample)))

        elif choice == '32':
            from ..core.expression import read_counts_matrix, differential_expression
            fp = input("  Count matrix: ").strip()
            cond = input("  Conditions (comma-sep): ").strip().split(',')
            counts = read_counts_matrix(fp)
            if counts is not None:
                print(differential_expression(counts, cond).head(20).to_string())

        elif choice == '33':
            from ..core.expression import read_counts_matrix, cpm_normalization
            counts = read_counts_matrix(input("  Count matrix: ").strip())
            if counts is not None:
                print(cpm_normalization(counts).head(10).to_string())

        elif choice == '34':
            from ..core.expression import read_counts_matrix, tpm_normalization
            import numpy as np
            counts = read_counts_matrix(input("  Count matrix: ").strip())
            if counts is not None:
                lens = np.array([float(x) for x in input("  Gene lengths (kb, comma-sep): ").split(',')])
                print(tpm_normalization(counts, lens).head(10).to_string())

        elif choice == '35':
            from ..core.enrichment import run_ora, run_gsea, format_enrichment_report
            genes = [g.strip() for g in input("  Gene IDs: ").split(',') if g.strip()]
            m = input("  Method (ORA/GSEA) [ORA]: ").strip().upper() or "ORA"
            result = run_ora(genes) if m == 'ORA' else run_gsea(genes)
            print(format_enrichment_report(result))

        # ── Genomics & NGS (40-47) ──
        elif choice == '40':
            from ..core.read_aligner import align_reads, format_alignment_report
            print(format_alignment_report(align_reads(input("  Reference: ").strip(), input("  Reads: ").strip())))

        elif choice == '41':
            from ..core.variant_calling import call_variants, variant_summary
            print(variant_summary(call_variants(input("  SAM/BAM: ").strip())))

        elif choice == '42':
            from ..core.peak_calling import call_peaks, format_peak_report
            print(format_peak_report(call_peaks(input("  ChIP-seq BAM/SAM: ").strip())))

        elif choice == '43':
            from ..core.assembly import assemble, format_assembly_report
            print(format_assembly_report(assemble(input("  Reads FASTQ: ").strip())))

        elif choice == '44':
            from ..core.msa import read_fasta_for_msa
            from ..core.phylogeny import distance_matrix, upgma_tree, plot_phylogenetic_tree
            seqs = read_fasta_for_msa(input("  Aligned FASTA: ").strip())
            if seqs:
                names = [n for n, _ in seqs]
                dist = distance_matrix([s for _, s in seqs])
                plot_phylogenetic_tree(upgma_tree(dist, names), names)
                plt.show()

        elif choice == '45':
            from ..core.ml_phylogeny import build_tree, format_phylo_report
            print(format_phylo_report(build_tree(input("  Aligned FASTA: ").strip(),
                                                  bootstrap=int(input("  Bootstrap [100]: ") or "100"))))

        elif choice == '46':
            from ..core.bayesian_phylogeny import run_bayesian, format_bayesian_report
            print(format_bayesian_report(run_bayesian(input("  Aligned FASTA: ").strip())))

        elif choice == '47':
            from ..core.ngs import read_vcf, manhattan_from_vcf
            df = read_vcf(input("  VCF file: ").strip())
            if df is not None:
                mh = manhattan_from_vcf(df)
                plt.scatter(mh['POS'], mh['neg_log10'], s=2)
                plt.title("Manhattan Plot from VCF")
                plt.show()

        # ── Single-Cell & Proteomics (50-53) ──
        elif choice == '50':
            from ..core.single_cell import load_count_matrix, run_full_pipeline, format_sc_report
            adata, err = load_count_matrix(input("  Count matrix (h5ad/csv/10x): ").strip())
            if err:
                print(f"  Error: {err}")
            else:
                print(format_sc_report(run_full_pipeline(adata)[1]))

        elif choice == '51':
            from ..core.structure import full_analysis, format_structure_report
            pdb = input("  PDB ID or file: ").strip()
            import os
            info = full_analysis(filepath=pdb) if os.path.exists(pdb) else full_analysis(pdb_id=pdb)
            print(format_structure_report(info))

        elif choice == '52':
            from ..core.structure_prediction import predict_structure, format_prediction_report
            seq = input("  Protein seq or UniProt ID: ").strip()
            result = predict_structure(sequence=seq) if len(seq) > 10 else predict_structure(uniprot_id=seq)
            print(format_prediction_report(result))

        elif choice == '53':
            from ..core.docking import dock, format_docking_report
            print(format_docking_report(dock(input("  Receptor PDB: ").strip(), input("  Ligand PDB: ").strip())))

        # ── Specialized (55-62) ──
        elif choice == '55':
            from ..core.metagenomics import classify_reads, format_metagenomics_report
            print(format_metagenomics_report(classify_reads(input("  Reads FASTQ: ").strip())))

        elif choice == '56':
            from ..core.crispr import design_guides, format_crispr_report
            seq = input("  Target DNA: ").strip()
            pam = input("  PAM type [SpCas9]: ").strip() or "SpCas9"
            print(format_crispr_report(design_guides(seq, pam_type=pam)))

        elif choice == '57':
            from ..core.metabolism import run_fba, format_flux_report
            model = input("  SBML file (blank=demo): ").strip()
            print(format_flux_report(run_fba(model_file=model if model else None)))

        elif choice == '58':
            import numpy as np
            from ..core.popgen import full_analysis, format_popgen_report
            print("  Enter rows (0/1/2 genotypes, comma-sep):")
            rows = []
            while True:
                row = input("  > ").strip()
                if not row: break
                rows.append([int(x) for x in row.split(',')])
            if rows:
                print(format_popgen_report(full_analysis(np.array(rows))))

        elif choice == '59':
            from ..core.epigenomics import parse_bisulfite_bed, calculate_methylation_levels, format_epigenomics_report
            print(format_epigenomics_report(calculate_methylation_levels(parse_bisulfite_bed(input("  BED file: ").strip()))))

        elif choice == '60':
            from ..core.metabolomics import detect_peaks
            import numpy as np
            arr = np.array([float(x) for x in input("  Intensities (comma-sep): ").split(',')])
            features = detect_peaks(arr)
            print(f"  Detected {len(features)} peaks")

        elif choice == '61':
            from ..core.md_simulation import run_simulation, format_md_report
            print(format_md_report(run_simulation(input("  PDB file: ").strip(),
                                                   steps=int(input("  Steps [1000]: ") or "1000"))))

        elif choice == '62':
            from ..core.bio_ml import train_random_forest, train_svm, format_ml_report
            import numpy as np
            print("  Enter feature rows (comma-sep):")
            rows = []
            while True:
                row = input("  > ").strip()
                if not row: break
                rows.append([float(x) for x in row.split(',')])
            if rows:
                X = np.array(rows)
                y = [l.strip() for l in input("  Labels: ").split(',')]
                m = input("  Model (RF/SVM) [RF]: ").strip().upper() or "RF"
                print(format_ml_report(train_svm(X, y) if m == 'SVM' else train_random_forest(X, y)))

        # ── Advanced Plots (101-111) ──
        elif choice == '101': gsea_plot()
        elif choice == '102': motif_logo()
        elif choice == '103': sankey_diagram()
        elif choice == '104': umap_plot()
        elif choice == '105': sine_plot()
        elif choice == '106': cosine_plot()
        elif choice == '107': linear_plot()
        elif choice == '108': quadratic_plot()
        elif choice == '109': cubic_plot()
        elif choice == '110': exponential_plot()
        elif choice == '111': logistic_plot()

        # ── Analysis (81-86) ──
        elif choice == '81':
            from ..core.codon_usage import codon_usage_table, format_codon_usage
            seq = input("  DNA sequence: ").strip().upper()
            result = codon_usage_table(seq)
            print(format_codon_usage(result))

        elif choice == '82':
            from ..core.codon_usage import kmer_composition, format_kmer_composition
            seq = input("  Sequence: ").strip().upper()
            k = int(input("  K-mer size [3]: ") or "3")
            result = kmer_composition(seq, k=k)
            print(format_kmer_composition(result))

        elif choice == '83':
            from ..core.codon_usage import sequence_complexity
            seq = input("  Sequence: ").strip().upper()
            result = sequence_complexity(seq)
            print(f"  Average complexity: {result['average_complexity']}")
            print(f"  Low complexity: {result['is_low_complexity']}")

        elif choice == '84':
            from ..core.survival import kaplan_meier, log_rank_test, format_km_result
            import numpy as np
            print("  Enter survival times (comma-sep):")
            times = np.array([float(x) for x in input("  Times: ").split(',')])
            print("  Enter event indicators (1=event, 0=censored, comma-sep):")
            events = np.array([int(x) for x in input("  Events: ").split(',')])
            result = kaplan_meier(times, events)
            print(format_km_result(result))

        elif choice == '85':
            from ..plotting.network_plots import plot_network, create_ppi_network, network_statistics
            print("  Enter interactions (protein_a,protein_b,weight — one per line, empty to finish):")
            interactions = []
            while True:
                line = input("  > ").strip()
                if not line: break
                parts = line.split(',')
                interactions.append((parts[0].strip(), parts[1].strip(), float(parts[2]) if len(parts) > 2 else 1.0))
            if interactions:
                graph = create_ppi_network(interactions)
                stats = network_statistics(graph)
                print(f"  Nodes: {stats['nodes']}, Edges: {stats['edges']}, Density: {stats['density']}")
                plot_network(graph, title="PPI Network")
                import matplotlib.pyplot as plt
                plt.show()

        elif choice == '86':
            from ..plotting.network_plots import network_statistics, create_ppi_network
            print("  Enter interactions (protein_a,protein_b,weight — one per line, empty to finish):")
            interactions = []
            while True:
                line = input("  > ").strip()
                if not line: break
                parts = line.split(',')
                interactions.append((parts[0].strip(), parts[1].strip(), float(parts[2]) if len(parts) > 2 else 1.0))
            if interactions:
                graph = create_ppi_network(interactions)
                stats = network_statistics(graph)
                for k, v in stats.items():
                    print(f"  {k}: {v}")

        # ── Advanced Visualization (77-81) ──
        elif choice == '77':
            print("  Enter set elements (one set per line, comma-sep, empty to finish):")
            sets_dict = {}
            while True:
                line = input("  Set name,elements: ").strip()
                if not line: break
                parts = line.split(',', 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    elements = set(x.strip() for x in parts[1].split(','))
                    sets_dict[name] = elements
            if sets_dict:
                fig = plot_upset(sets_dict, title="UpSet Plot")
                plt.show()
                plt.close()
            else:
                print("  No sets entered.")

        elif choice == '78':
            from ..plotting.genome_browser import create_bed_track, create_variant_track
            print("  Track types: coverage, bed, variant")
            tracks = []
            while True:
                ttype = input("  Track type (empty to finish): ").strip()
                if not ttype: break
                if ttype == 'bed':
                    fp = input("  BED file: ").strip()
                    if fp and os.path.exists(fp):
                        tracks.append(create_bed_track(fp))
                elif ttype == 'variant':
                    fp = input("  VCF file: ").strip()
                    if fp and os.path.exists(fp):
                        tracks.append(create_variant_track(fp))
                elif ttype == 'coverage':
                    fp = input("  BAM/SAM file: ").strip()
                    if fp and os.path.exists(fp):
                        tracks.append(create_coverage_from_bam(fp))
            if tracks:
                fig = plot_genome_tracks(tracks, title="Genome Browser")
                plt.show()
                plt.close()
            else:
                print("  No tracks added.")

        elif choice == '79':
            from ..plotting.interactive_plots import interactive_scatter
            import numpy as np
            use_file = input("  Load from file? (y/n): ").strip().lower()
            if use_file == 'y':
                fp = input("  CSV file: ").strip()
                if fp and os.path.exists(fp):
                    import pandas as pd
                    df = pd.read_csv(fp)
                    x_col = input("  X column: ").strip()
                    y_col = input("  Y column: ").strip()
                    if x_col in df.columns and y_col in df.columns:
                        x = pd.to_numeric(df[x_col], errors='coerce').dropna().values
                        y = pd.to_numeric(df[y_col], errors='coerce').dropna().values
                        fig = interactive_scatter(x, y, title="Interactive Scatter",
                                                  output_html="scatter_interactive.html")
                        print("  Saved: scatter_interactive.html")
                    else:
                        print("  Invalid columns.")
                else:
                    print("  File not found.")
            else:
                np.random.seed(42)
                x = np.random.randn(100)
                y = x * 2 + np.random.randn(100)
                fig = interactive_scatter(x, y, title="Interactive Scatter (demo)",
                                          output_html="scatter_interactive.html")
                print("  Saved: scatter_interactive.html")

        elif choice == '80':
            print("  Enter aligned sequences (one per line, empty to finish):")
            seqs = []
            while True:
                seq = input("  > ").strip().upper()
                if not seq: break
                seqs.append(seq)
            if not seqs:
                seqs = ["ACGTACGT", "ACGAACGT", "ACGTACGA", "ACGTACGT"]
            fig = plot_logo_with_conservation(seqs)
            plt.show()
            plt.close()

        elif choice == '81':
            print("  Enter gene orders (comma-sep, one genome per line):")
            g1 = [g.strip() for g in input("  Genome 1: ").split(',') if g.strip()]
            g2 = [g.strip() for g in input("  Genome 2: ").split(',') if g.strip()]
            if not g1 or not g2:
                g1 = ['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE']
                g2 = ['GeneA', 'GeneC', 'GeneB', 'GeneE', 'GeneD']
            score, pairs = compute_synteny_score(g1, g2)
            print(f"  Synteny score: {score:.3f}")
            fig = plot_synteny_dotplot(g1, g2, title="Synteny Dotplot")
            plt.show()
            plt.close()

        # ── Sequence Tools (63-66) ──
        elif choice == '63':
            from ..core.orf_finder import find_orfs, format_orf_results
            seq = input("  DNA sequence: ").strip().upper()
            min_len = int(input("  Min protein length [30]: ") or "30")
            orfs = find_orfs(seq, min_length=min_len)
            print(format_orf_results(orfs))

        elif choice == '64':
            from ..core.orf_finder import find_restriction_sites, format_restriction_sites
            seq = input("  DNA sequence: ").strip().upper()
            enzymes = input("  Enzymes (comma-sep, or blank for all): ").strip()
            enzyme_list = [e.strip() for e in enzymes.split(',') if e.strip()] or None
            sites = find_restriction_sites(seq, enzymes=enzyme_list)
            print(format_restriction_sites(sites))

        elif choice == '65':
            from ..core.orf_finder import design_primers, format_primers
            seq = input("  Template DNA: ").strip().upper()
            start = int(input("  Amplicon start [0]: ") or "0")
            end = int(input("  Amplicon end [end]: ") or "0") or None
            fwd, rev = design_primers(seq, amplicon_start=start, amplicon_end=end)
            print(format_primers(fwd, rev))

        elif choice == '66':
            from collections import Counter
            seq = input("  Sequence: ").strip().upper()
            k = int(input("  K-mer size [3]: ") or "3")
            kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
            counts = Counter(kmers)
            print(f"\n  Top 20 {k}-mers:")
            for kmer, count in counts.most_common(20):
                print(f"    {kmer}: {count}")

        # ── Database Search (67-72) ──
        elif choice == '67':
            from ..core.databases import search_ncbi, format_search_results
            query = input("  Search query: ").strip()
            results = search_ncbi(query)
            print(format_search_results(results))

        elif choice == '68':
            from ..core.databases import search_uniprot, format_search_results
            query = input("  Protein/gene name: ").strip()
            results = search_uniprot(query)
            print(format_search_results(results))

        elif choice == '69':
            from ..core.databases import search_pdb, format_search_results
            query = input("  Protein name or keyword: ").strip()
            results = search_pdb(query)
            print(format_search_results(results))

        elif choice == '70':
            from ..core.databases import search_kegg, format_search_results
            query = input("  Pathway/compound name: ").strip()
            results = search_kegg(query)
            print(format_search_results(results))

        elif choice == '71':
            from ..core.databases import search_ensembl, format_search_results
            gene = input("  Gene symbol: ").strip()
            species = input("  Species [human]: ").strip() or "human"
            results = search_ensembl(gene, species=species)
            print(format_search_results(results))

        elif choice == '72':
            from ..core.databases import search_all, format_search_results
            query = input("  Search query: ").strip()
            results = search_all(query)
            print(format_search_results(results))

        # ── File Formats (73-76) ──
        elif choice == '73':
            from ..core.file_formats import parse_bed, format_bed_summary, bed_to_dataframe
            fp = input("  BED file path: ").strip()
            records = parse_bed(fp)
            print(format_bed_summary(records))
            df = bed_to_dataframe(records)
            print(df.head(10).to_string())

        elif choice == '74':
            from ..core.file_formats import parse_gff, format_gff_summary, gff_to_dataframe
            fp = input("  GFF/GTF file path: ").strip()
            records = parse_gff(fp)
            print(format_gff_summary(records))

        elif choice == '75':
            from ..core.file_formats import parse_newick, tree_to_ascii
            newick = input("  Newick string: ").strip()
            tree = parse_newick(newick)
            for line in tree_to_ascii(tree):
                print(f"  {line}")

        elif choice == '76':
            from ..core.file_formats import parse_stockholm
            fp = input("  Stockholm file path: ").strip()
            data = parse_stockholm(fp)
            print(f"  Sequences: {len(data['alignment'])}")
            for name, seq in list(data['alignment'].items())[:5]:
                print(f"    {name}: {seq[:50]}{'...' if len(seq)>50 else ''}")

        # ── Workflow & Domain (92-98) ──
        elif choice == '92':
            print("  Pipeline Builder — chain steps together")
            print("  Add steps (name:function_name, empty to finish):")
            steps = []
            while True:
                line = input("  > ").strip()
                if not line: break
                parts = line.split(':', 1)
                if len(parts) == 2:
                    steps.append({"name": parts[0].strip(), "func": eval(parts[1].strip())})
            if steps:
                p = Pipeline("cli_pipeline")
                p.add_steps(steps)
                p.run()
                print(format_pipeline_report(p))

        elif choice == '93':
            print("  Batch Processor — run analysis on multiple samples")
            func_name = input("  Function (e.g., 'lambda x: x.upper()'): ").strip()
            try:
                func = eval(func_name)
            except (SyntaxError, NameError, TypeError, ValueError):
                print("  Invalid function."); continue
            samples = input("  Sample IDs (comma-sep): ").strip().split(',')
            samples = [s.strip() for s in samples if s.strip()]
            if samples:
                bp = BatchProcessor("cli_batch")
                bp.add_samples(samples, func)
                bp.run(max_workers=4)
                print(format_batch_report(bp))

        elif choice == '94':
            title = input("  Report title [BioSuite Report]: ").strip() or "BioSuite Report"
            report = create_report(title)
            report.add_text("This report was generated by BioSuite Ultra.")
            report.add_stats({"Modules": 48, "Tests": 203})
            path = input("  Output file [report.html]: ").strip() or "report.html"
            report.save(path)
            print(f"  Report saved to {path}")

        elif choice == '95':
            go = GOBrowser()
            print(f"  Loaded {len(go.terms)} GO terms (built-in subset)")
            query = input("  Search query: ").strip()
            if query:
                results = go.search(query)
                print(format_go_results(results))
            else:
                ns = input("  Namespace (BP/MF/CC) [BP]: ").strip().upper() or "BP"
                terms = go.get_namespace_terms(ns)
                print(f"  {ns} terms: {len(terms)}")
                for t in terms[:20]:
                    print(f"    {t.go_id} {t.name}")

        elif choice == '96':
            pm = create_kegg_style_pathway()
            print(format_pathway_report(pm))
            fig = draw_pathway(pm)
            plt.show()
            plt.close()

        elif choice == '97':
            print("  GWAS Analysis")
            use_demo = input("  Use demo data? (y/n) [y]: ").strip().lower() or 'y'
            if use_demo == 'y':
                data = generate_gwas_data(n_snps=2000)
            else:
                fp = input("  CSV file with SNP data: ").strip()
                import pandas as pd
                data = pd.read_csv(fp) if fp and os.path.exists(fp) else generate_gwas_data(2000)
            results = run_gwas(data)
            leads = detect_lead_snps(results)
            print(format_gwas_report(results, leads))

        elif choice == '98':
            seq = input("  Protein sequence: ").strip().upper()
            if not seq:
                seq = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS"
            mhc = input("  HLA type [A0201]: ").strip() or "A0201"
            tc = predict_t_cell_epitopes(seq, mhc_type=mhc)
            bc = predict_b_cell_epitopes(seq)
            print(format_epitope_report(tc, bc, "input protein"))

        # ── System (99) ──
        elif choice == '99':
            print("  Themes: 1=Dark-Green  2=Dark-Purple  3=Light-Blue")
            t = input("  Select: ").strip()
            tm = {'1': 'dark-green', '2': 'dark-purple', '3': 'light-blue'}.get(t)
            if tm:
                config['theme'] = tm
                save_config(config)
                set_theme(tm)
                print(f"  Theme → {tm}")

        # ── Genomics Tools (a, b, c) ──
        elif choice == 'a':
            from ..core.metagenomics import classify_16s_rna, format_16s_report
            print("  16S rRNA Classification Pipeline")
            print("  Enter sequences (name:sequence, one per line, empty to finish):")
            seqs = []
            while True:
                line = input("  > ").strip()
                if not line: break
                if ':' in line:
                    name, seq = line.split(':', 1)
                    seqs.append((name.strip(), seq.strip().upper()))
            if not seqs:
                # Demo
                seqs = [("Ecoli_16S", "TGGAGGAAGGTGGGGACGACGTCAGTATCGAATCTTGGATCAGGATCACCTCCGGA"),
                         ("Staph_16S", "AGCCATGCAGCACCTGTCTCAGCTTCCCGAAGGCACTATACGTAGATCGAAAGTTGAT")]
                print("  Using demo data...")
            result = classify_16s_rna(seqs)
            print(format_16s_report(result))

        elif choice == 'b':
            import numpy as np
            from ..core.variant_calling import (detect_structural_variants, detect_cnv,
                                                 format_sv_report, format_cnv_report)
            print("  SV / CNV Detection from Coverage Data")
            use_demo = input("  Use demo data? (y/n) [y]: ").strip().lower() or 'y'
            if use_demo == 'y':
                np.random.seed(42)
                cov = np.random.poisson(30, 10000).astype(float)
                cov[3000:4000] *= 0.3  # Deletion
                cov[6000:7000] *= 2.5  # Duplication
                ref = np.ones(10000) * 30
            else:
                fp = input("  Coverage file (CSV, one value per line): ").strip()
                if fp and os.path.exists(fp):
                    cov = np.loadtxt(fp, delimiter=',')
                    ref = np.ones_like(cov) * np.median(cov)
                else:
                    print("  File not found. Using demo data.")
                    np.random.seed(42)
                    cov = np.random.poisson(30, 10000).astype(float)
                    ref = np.ones(10000) * 30
            svs = detect_structural_variants(cov, ref)
            print(format_sv_report(svs))
            cnv = detect_cnv(cov, ref)
            print(f"\n{format_cnv_report(cnv)}")

        elif choice == 'c':
            import numpy as np
            from ..core.file_formats import read_bigwig, bigwig_summary, format_bigwig_summary
            print("  BigWig Reader")
            fp = input("  BigWig file path: ").strip()
            if fp and os.path.exists(fp):
                data = read_bigwig(fp)
                if "error" in data:
                    print(f"  Error: {data['error']}")
                elif "chroms" in data:
                    print(f"  Chromosomes: {len(data['chroms'])}")
                    for chrom, values in list(data['chroms'].items())[:5]:
                        print(f"    {chrom}: {len(values)} values")
                else:
                    summary = bigwig_summary(fp)
                    print(format_bigwig_summary(summary))
            else:
                print("  File not found.")

        # ── Molecular Cloning (l-q) ──
        elif choice == 'l':
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from ..plotting.plasmid_map import create_sample_plasmid, draw_plasmid, format_plasmid_report
            pm = create_sample_plasmid()
            print(format_plasmid_report(pm))
            fig = draw_plasmid(pm)
            plt.show()
            plt.close()

        elif choice == 'm':
            from ..core.cloning import find_restriction_sites, simulate_digestion, format_digest_report
            seq = input("  DNA sequence: ").strip().upper()
            enzymes = input("  Enzymes (comma-sep, e.g. EcoRI,BamHI): ").strip().split(',')
            enzymes = [e.strip() for e in enzymes if e.strip()]
            if seq and enzymes:
                result = simulate_digestion(seq, enzymes)
                print(format_digest_report(result))
            else:
                print("  Invalid input.")

        elif choice == 'n':
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from ..core.cloning import simulate_digestion, plot_virtual_gel
            seq = input("  DNA sequence: ").strip().upper()
            enzymes = input("  Enzymes (comma-sep): ").strip().split(',')
            enzymes = [e.strip() for e in enzymes if e.strip()]
            if seq and enzymes:
                result = simulate_digestion(seq, enzymes)
                fig = plot_virtual_gel(result['fragments'])
                plt.show()
                plt.close()
            else:
                print("  Invalid input.")

        elif choice == 'o':
            from ..core.cloning import design_primers, simulate_pcr, format_primer_report
            template = input("  Template sequence: ").strip().upper()
            start = int(input("  Amplicon start: ") or "0")
            end = int(input("  Amplicon end: ") or str(len(template)))
            if template:
                fwd, rev = design_primers(template, start, end)
                print(format_primer_report(fwd, rev))
                if fwd and rev:
                    amplicon = simulate_pcr(template, fwd, rev)
                    print(f"  Amplicon length: {len(amplicon)} bp")
            else:
                print("  No sequence provided.")

        elif choice == 'p':
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from ..plotting.sequence_viewer import draw_sequence_view
            seq = input("  DNA sequence (or FASTA path): ").strip()
            import os
            if os.path.exists(seq):
                from ..core.sequence import read_fasta
                records = read_fasta(seq)
                if records:
                    seq = records[0][1]
            fig = draw_sequence_view(seq.upper())
            plt.show()
            plt.close()

        elif choice == 'q':
            from ..core.cloning import simulate_ligation, format_digest_report
            print("  Enter fragment sizes (comma-separated):")
            sizes = input("  Sizes: ").strip()
            if sizes:
                frag_list = [int(s.strip()) for s in sizes.split(',') if s.strip().isdigit()]
                if frag_list:
                    result = simulate_ligation(frag_list)
                    print(f"  Ligated product: {result.get('total_bp', 0)} bp")
                    print(f"  Circular: {result.get('circular', False)}")
                else:
                    print("  Invalid sizes.")
            else:
                print("  No fragments provided.")

        elif choice == '89':
            from ..gui.main_window import BioSuiteApp
            app = BioSuiteApp()
            app.mainloop()
            print(f"\n  {G}GUI closed.{R} {D}CLI ready.{R}")
            import time
            time.sleep(0.3)

        # ── Phase 6 Features (d-k) ──
        elif choice == 'd':
            from ..plotting.plot_api import volcano, pca, manhattan, scatter, boxplot, heatmap
            import numpy as np
            print("  Interactive Plot Generator (Plotly)")
            print("  Plot types: 1=Volcano  2=PCA  3=Manhattan  4=Scatter  5=Boxplot  6=Heatmap")
            ptype = input("  Select plot type [1]: ").strip() or "1"
            np.random.seed(42)
            if ptype == '1':
                fc = np.random.normal(0, 1.5, 500)
                pvals = np.random.uniform(0, 1, 500)
                pvals[:30] = np.random.uniform(1e-6, 0.05, 30)
                fig = volcano(fc, pvals, interactive=True, title="Interactive Volcano")
            elif ptype == '2':
                data = np.random.randn(30, 50)
                fig = pca(data, labels=['Ctrl']*15 + ['Treat']*15, interactive=True)
            elif ptype == '3':
                chroms = np.random.choice(['chr1', 'chr2', 'chr3'], 200)
                positions = np.random.randint(1, 1000000, 200)
                pvals = np.random.uniform(0, 1, 200)
                fig = manhattan(chroms, positions, pvals, interactive=True)
            elif ptype == '4':
                x = np.random.randn(100)
                y = x * 2 + np.random.randn(100)
                fig = scatter(x, y, interactive=True)
            elif ptype == '5':
                data = {'Ctrl': np.random.randn(30).tolist(), 'Treat': (np.random.randn(30) + 1).tolist()}
                fig = boxplot(data, interactive=True)
            elif ptype == '6':
                data = np.random.randn(10, 8)
                fig = heatmap(data, interactive=True)
            else:
                print("  Invalid plot type."); continue
            outfile = input("  Output HTML [plot.html]: ").strip() or "plot.html"
            fig.write_html(outfile)
            print(f"  Saved: {outfile}")

        elif choice == 'e':
            from ..core.provenance import ProvenanceTracker
            print("  Provenance Tracker")
            print("  Actions: 1=Start session  2=Record step  3=View history  4=Export HTML  5=Export JSON")
            action = input("  Select action [1]: ").strip() or "1"
            if action == '1':
                db = input("  Database name [analysis.db]: ").strip() or "analysis.db"
                tracker = ProvenanceTracker(db)
                print(f"  Session started: {tracker.session_id}")
            elif action == '2':
                db = input("  Database file [analysis.db]: ").strip() or "analysis.db"
                tracker = ProvenanceTracker(db)
                module = input("  Module name: ").strip()
                func = input("  Function name: ").strip()
                params = input("  Parameters (JSON): ").strip()
                result = input("  Result summary: ").strip()
                import json
                try:
                    params_dict = json.loads(params) if params else {}
                except json.JSONDecodeError:
                    params_dict = {"raw": params}
                tracker.record(module, func, params_dict, result)
                print("  Step recorded.")
            elif action == '3':
                db = input("  Database file [analysis.db]: ").strip() or "analysis.db"
                tracker = ProvenanceTracker(db)
                print(tracker.summary())
            elif action == '4':
                db = input("  Database file [analysis.db]: ").strip() or "analysis.db"
                tracker = ProvenanceTracker(db)
                outfile = input("  Output HTML [provenance.html]: ").strip() or "provenance.html"
                tracker.export_html(outfile)
                print(f"  Saved: {outfile}")
            elif action == '5':
                db = input("  Database file [analysis.db]: ").strip() or "analysis.db"
                tracker = ProvenanceTracker(db)
                outfile = input("  Output JSON [provenance.json]: ").strip() or "provenance.json"
                tracker.export_json(outfile)
                print(f"  Saved: {outfile}")

        elif choice == 'f':
            from ..core.plugin import PluginManager
            pm = PluginManager()
            print("  Plugin Manager")
            print("  Actions: 1=Discover  2=List  3=Create template")
            action = input("  Select action [1]: ").strip() or "1"
            if action == '1':
                discovered = pm.discover()
                print(f"  Discovered {len(discovered)} plugin(s)")
            elif action == '2':
                pm.discover()
                pm.list_plugins()
            elif action == '3':
                name = input("  Plugin name: ").strip()
                if name:
                    pm.create_plugin_template(name, '.')
                    print(f"  Created: biosuite-plugin-{name}/")

        elif choice == 'g':
            from ..core.file_formats import detect_file_format, read_file, format_file_summary
            fp = input("  File path: ").strip()
            if fp and os.path.exists(fp):
                fmt = detect_file_format(fp)
                print(f"  Detected format: {fmt}")
                result = read_file(fp)
                print(format_file_summary(result))
            else:
                print("  File not found.")

        elif choice == 'h':
            from ..core.expression import read_counts_matrix, deseq2_normalization
            fp = input("  Count matrix: ").strip()
            counts = read_counts_matrix(fp)
            if counts is not None:
                normalized = deseq2_normalization(counts)
                print("  DESeq2-normalized data:")
                print(normalized.head(10).to_string())

        elif choice == 'i':
            from ..core.expression import read_counts_matrix, variance_stabilizing_transformation
            fp = input("  Count matrix: ").strip()
            counts = read_counts_matrix(fp)
            if counts is not None:
                vst = variance_stabilizing_transformation(counts)
                print("  VST-transformed data:")
                print(vst.head(10).to_string())

        elif choice == 'j':
            from ..core.expression import read_counts_matrix, differential_expression
            fp = input("  Count matrix: ").strip()
            cond = input("  Conditions (comma-sep): ").strip().split(',')
            counts = read_counts_matrix(fp)
            if counts is not None:
                result = differential_expression(counts, cond, method='nb')
                print("  Differential Expression (Negative Binomial):")
                print(result.head(20).to_string())

        elif choice == 'k':
            from ..core.file_formats import detect_file_format, read_file, format_file_summary
            fp = input("  File path: ").strip()
            if fp and os.path.exists(fp):
                fmt = detect_file_format(fp)
                print(f"  Detected format: {fmt}")
                result = read_file(fp)
                print(format_file_summary(result))
                if 'records' in result and result['records']:
                    print(f"\n  First 5 records:")
                    for rec in result['records'][:5]:
                        print(f"    {rec}")
            else:
                print("  File not found.")

        elif choice == '90':
            port = input("  Port [8000]: ").strip() or "8000"
            print(f"\n  {G}Starting BioSuite API Server on port {port}...{R}")
            print(f"  API Docs: http://localhost:{port}/docs")
            print(f"  Press Ctrl+C to stop\n")
            try:
                import uvicorn
                from ..api import app
                uvicorn.run(app, host="0.0.0.0", port=int(port))
            except KeyboardInterrupt:
                print(f"\n  {G}API server stopped.{R}")
            except ImportError:
                print(f"  {Y}Install uvicorn: pip install uvicorn{R}")

        else:
            print(f"  {Y}Invalid option. Type 0 to exit.{R}")


def main_cli_gui():
    """Entry point for biosuite-gui command — launches GUI directly."""
    from ..gui.main_window import BioSuiteApp
    app = BioSuiteApp()
    app.mainloop()
