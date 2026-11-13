"""
BioSuite Ultra — Professional CLI Interface

Supports both interactive menu mode and non-interactive command mode:
    biosuite                    # Interactive menu
    biosuite gc ATCGATCG        # Non-interactive
    biosuite nw AGTACGCA TATGC  # Commands take positional arguments
"""
import argparse
import os
import re
import sys


def _enable_utf8_stdio():
    """Make CLI output safe on Windows classic consoles.

    Windows' legacy cmd defaults to cp1252; any box-drawing character in
    the banner (e.g. '█') then raises UnicodeEncodeError and kills run.py
    at startup. Reconfigure stdio to UTF-8 with replace-fallback; no-ops
    on streams without ``reconfigure`` (pytest capture, very old runtimes).
    """
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):  # fmt: skip
            pass


from .. import __version__
from ..core.epitope import (
    format_epitope_report,
    predict_b_cell_epitopes,
    predict_t_cell_epitopes,
)
from ..core.go_browser import GOBrowser, format_go_results
from ..core.gwas import (
    detect_lead_snps,
    format_gwas_report,
    generate_gwas_data,
    run_gwas,
)
from ..core.log import get_logger
from ..core.utils import config, save_config, set_theme
from ..core.workflow.batch import BatchProcessor, format_batch_report
from ..core.workflow.pipeline import Pipeline, format_pipeline_report
from ..core.workflow.report import (
    create_report,
)

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
    'search': {'func': 'biosuite.cli.menu:search_menu', 'desc': 'Search the interactive menu', 'args': ['query']},
    'translate': {'func': 'biosuite.core.sequence:translate', 'desc': 'Translate DNA to protein', 'args': ['sequence'], 'kwargs': {'frame': 1}},
    'stats': {'func': 'biosuite.core.sequence:sequence_stats', 'desc': 'Sequence composition stats', 'args': ['sequence']},
    'nw': {'func': 'biosuite.cli.menu:_cli_align', 'desc': 'Needleman-Wunsch alignment', 'args': ['seq1', 'seq2'], 'kwargs': {'method': 'nw'}},
    'sw': {'func': 'biosuite.cli.menu:_cli_align', 'desc': 'Smith-Waterman alignment', 'args': ['seq1', 'seq2'], 'kwargs': {'method': 'sw'}},
    'blast': {'func': 'biosuite.cli.menu:_cli_blast', 'desc': 'BLAST sequence search', 'args': ['query', 'database']},
    'crispr': {'func': 'biosuite.core.crispr:design_guides', 'desc': 'CRISPR guide design', 'args': ['sequence'], 'report': 'biosuite.core.crispr:format_crispr_report'},
    'epitope': {'func': 'biosuite.cli.menu:_cli_epitope', 'desc': 'T-cell and B-cell epitope prediction', 'args': ['sequence']},
    'gwas': {'func': 'biosuite.cli.menu:_cli_gwas', 'desc': 'Run GWAS analysis from a CSV/TSV of SNPs', 'args': ['datafile']},
    'hwe': {'func': 'biosuite.cli.menu:_cli_hardy_weinberg', 'desc': 'Hardy-Weinberg test (AA Aa aa counts)', 'args': ['AA', 'Aa', 'aa']},
    'volcano': {'func': 'biosuite.cli.menu:_cli_volcano', 'desc': 'Volcano plot from two numeric columns', 'args': ['lfc_file', 'pval_file']},
    'pca': {'func': 'biosuite.cli.menu:_cli_pca', 'desc': 'PCA plot of a numeric matrix', 'args': ['datafile']},
    'manhattan': {'func': 'biosuite.cli.menu:_cli_manhattan', 'desc': 'Manhattan plot (chrom, pos, p_value)', 'args': ['datafile']},
    'digest': {'func': 'biosuite.core.cloning:simulate_digestion', 'desc': 'Restriction digestion', 'args': ['sequence', 'enzyme'], 'report': 'biosuite.core.cloning:format_digest_report'},
    'primers': {'func': 'biosuite.core.cloning:design_primers', 'desc': 'Primer design', 'args': ['sequence'], 'report': 'biosuite.core.cloning:format_primer_report'},
    'gui': {'func': '_launch_gui', 'desc': 'Launch GUI application', 'args': []},
    'api': {'func': '_launch_api', 'desc': 'Launch REST API server', 'args': [], 'kwargs': {'port': 8000}},
    'theme': {'func': '_change_theme', 'desc': 'Change theme', 'args': ['theme_name']},
}


logger = get_logger(__name__)


def _import_func(func_path):
    """Import a callable from a ``module:function`` path.

    The name after the colon may be dotted (``builtins:str.upper``), because
    :data:`SAFE_FUNCTIONS` lists aliases exactly that way and a resolver that could
    not read its own table would fail the moment a caller resolved an alias through
    it instead of through :func:`resolve_safe_callable`.
    """
    module_path, func_name = func_path.rsplit(':', 1)
    import importlib
    target = importlib.import_module(module_path)
    for part in func_name.split('.'):
        target = getattr(target, part)
    return target


#: Short aliases the interactive pipeline/batch builders accept in addition to
#: fully qualified ``module:function`` paths.  Everything reachable from the
#: menu must be an explicitly whitelisted, importable BioSuite callable — the
#: builders used to ``eval()`` whatever the user typed, which executed
#: arbitrary Python supplied through piped or scripted stdin.
SAFE_FUNCTIONS = {
    'gc_content': 'biosuite.core.sequence:gc_content',
    'reverse_complement': 'biosuite.core.sequence:reverse_complement',
    'revcomp': 'biosuite.core.sequence:reverse_complement',
    'translate': 'biosuite.core.sequence:translate',
    'sequence_stats': 'biosuite.core.sequence:sequence_stats',
    'find_orfs': 'biosuite.core.orf_finder:find_orfs',
    'upper': 'builtins:str.upper',
    'lower': 'builtins:str.lower',
    'length': 'builtins:len',
}

_SAFE_BUILTINS = {'str.upper': str.upper, 'str.lower': str.lower, 'len': len}


def resolve_safe_callable(name):
    """Resolve *name* to a callable without executing user-supplied code.

    Accepts either a key of :data:`SAFE_FUNCTIONS` or a ``module:function``
    path inside the ``biosuite`` package.  Anything else raises ``ValueError``;
    there is deliberately no code-evaluation path.
    """
    name = (name or '').strip()
    if not name:
        raise ValueError("No function given.")
    target = SAFE_FUNCTIONS.get(name, name)
    if target.startswith('builtins:'):
        key = target.split(':', 1)[1]
        if key not in _SAFE_BUILTINS:
            raise ValueError(f"Unsupported builtin: {key}")
        return _SAFE_BUILTINS[key]
    if ':' not in target:
        raise ValueError(
            f"Unknown function {name!r}. Use one of "
            f"{', '.join(sorted(SAFE_FUNCTIONS))} or a "
            "'biosuite.module:function' path.")
    module_path = target.split(':', 1)[0]
    if module_path != 'biosuite' and not module_path.startswith('biosuite.'):
        raise ValueError("Only functions inside the 'biosuite' package may be used here.")
    try:
        func = _import_func(target)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot resolve {target!r}: {exc}") from exc
    if not callable(func):
        raise ValueError(f"{target} is not callable.")
    return func


def _cli_hardy_weinberg(AA, Aa, aa):
    """CLI adapter: three genotype counts -> hardy_weinberg_test's dict argument."""
    from ..core.popgen import hardy_weinberg_test
    try:
        counts = {'AA': int(AA), 'Aa': int(Aa), 'aa': int(aa)}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Genotype counts must be integers: {exc}") from exc
    return hardy_weinberg_test(counts)


# ── Command-line adapters ────────────────────────────────────────────────────
#
# A registered command is called with the strings the user typed, so every entry
# that needs more than one straight call lives here instead.  Before these existed
# the registry pointed ``biosuite gwas`` at ``run_gwas`` directly: the path string
# reached ``DataFrame.iterrows`` and the command answered
# "Error: AttributeError: 'str' object has no attribute 'iterrows'" (exit 1), and
# ``volcano``/``pca``/``manhattan`` failed the same way.  ``blast`` was worse: it
# reported "Query not found" inside the repr of a result object and exited 0.

def _require_file(path, what="File"):
    """Raise the error :func:`_execute_command` renders as a one-line usage message."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def _read_table(path):
    """Read a CSV or TSV with a header row — the shape every tabular command takes."""
    import pandas as pd

    _require_file(path, "Data file")
    return pd.read_csv(path, sep=None, engine="python")


def _read_numbers(path):
    """One number per line (or per first field), ignoring header and comment lines."""
    import numpy as np

    _require_file(path, "Data file")
    values = []
    skipped = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                values.append(float(line.replace(",", " ").split()[0]))
            except ValueError:
                skipped += 1
    if skipped:
        print(f"Warning: skipped {skipped} line(s) that are not numbers in {path}",
              file=sys.stderr)
    if not values:
        raise ValueError(f"No numbers found in {path}")
    return np.array(values)


def _save_figure(fig, output):
    """Write the rendered figure out and hand it back to pyplot."""
    import matplotlib.pyplot as plt

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"Saved: {output}"


#: One provenance tracker per database file, kept for the life of the process.
_PROVENANCE_TRACKERS = {}


def _provenance_tracker(db_name):
    """Return the tracker for *db_name*, reusing the one this session already opened.

    Option 'e' used to build a fresh ``ProvenanceTracker`` for every action, and each
    new tracker starts a new session — so a step recorded by "Record step" was
    already invisible to "View history", and the export was always empty.
    """
    from ..core.provenance import ProvenanceTracker

    key = os.path.abspath(db_name)
    tracker = _PROVENANCE_TRACKERS.get(key)
    if tracker is None:
        tracker = ProvenanceTracker(db_name)
        _PROVENANCE_TRACKERS[key] = tracker
    return tracker


def _cli_align(seq1, seq2, method="nw"):
    """Pairwise alignment as two aligned lines and a score, never as a tuple repr."""
    from ..core.alignment import needleman_wunsch, smith_waterman

    align = needleman_wunsch if method == "nw" else smith_waterman
    a1, a2, score = align(seq1, seq2)
    return f"  Score: {score}\n  {a1}\n  {a2}"


def _cli_blast(query, database):
    from ..core.blast import format_blast_result, run_blast

    _require_file(query, "Query FASTA")
    _require_file(database, "Database FASTA")
    return format_blast_result(run_blast(query, database))


def _cli_epitope(sequence):
    from ..core.epitope import (
        format_epitope_report,
        predict_b_cell_epitopes,
        predict_t_cell_epitopes,
    )

    return format_epitope_report(predict_t_cell_epitopes(sequence),
                                 predict_b_cell_epitopes(sequence), "command line")


def _cli_gwas(datafile):
    from ..core.gwas import detect_lead_snps, format_gwas_report, run_gwas

    data = _read_table(datafile)
    results = run_gwas(data)
    return format_gwas_report(results, detect_lead_snps(results))


def _cli_volcano(lfc_file, pval_file, output="volcano.png"):
    from ..plotting.plot_api import volcano

    log2fc = _read_numbers(lfc_file)
    pvalues = _read_numbers(pval_file)
    if len(log2fc) != len(pvalues):
        raise ValueError(
            f"{lfc_file} has {len(log2fc)} values but {pval_file} has {len(pvalues)}; "
            f"the two files need one value per gene")
    return _save_figure(volcano(log2fc, pvalues), output)


def _cli_pca(datafile, output="pca.png"):
    from ..plotting.plot_api import pca

    table = _read_table(datafile).select_dtypes("number")
    if table.empty:
        raise ValueError(f"{datafile} has no numeric columns to project")
    return _save_figure(pca(table.values), output)


def _cli_manhattan(datafile, output="manhattan.png"):
    from ..plotting.plot_api import manhattan

    table = _read_table(datafile)
    missing = [c for c in ("chrom", "pos", "p_value") if c not in table.columns]
    if missing:
        raise ValueError(f"{datafile} is missing column(s): {', '.join(missing)}")
    return _save_figure(manhattan(table["chrom"], table["pos"], table["p_value"]),
                        output)


def _execute_command(cmd_name, args):
    """Execute a registered command with arguments.

    Returns a process exit code: 0 on success, 2 for a usage error (missing or
    invalid arguments) and 1 for a failure inside the analysis.  User errors
    print a one-line message — never a Python traceback.
    """
    if cmd_name not in COMMAND_REGISTRY:
        print(f"Unknown command: {cmd_name}")
        print(f"Available commands: {', '.join(sorted(COMMAND_REGISTRY.keys()))}")
        return 2

    entry = COMMAND_REGISTRY[cmd_name]

    # Arity check before dispatch: calling the core function with too few
    # arguments used to surface as a raw TypeError traceback.
    required = list(entry.get('args', []))
    supplied = [a for a in required if hasattr(args, a)]
    if len(supplied) < len(required):
        missing = [a for a in required if not hasattr(args, a)]
        print(f"Usage: biosuite {cmd_name} " + " ".join(f"<{a}>" for a in required))
        print(f"Missing argument(s): {', '.join(missing)}")
        return 2
    extra = [getattr(args, f"arg{i}") for i in range(len(required), 64)
             if hasattr(args, f"arg{i}")]
    if extra:
        print(f"Usage: biosuite {cmd_name} " + " ".join(f"<{a}>" for a in required))
        print(f"Too many arguments: {len(required) + len(extra)} given, "
              f"{len(required)} expected.")
        return 2

    # Handle special commands
    if entry['func'] == '_launch_gui':
        _launch_gui()
        return 0
    elif entry['func'] == '_launch_api':
        _launch_api(args.port if hasattr(args, 'port') else 8000)
        return 0
    elif entry['func'] == '_change_theme':
        # An inline copy of this used to call set_theme() and print, never writing
        # the config — so ``biosuite theme dark-purple`` was forgotten on the next
        # start while reporting success.
        theme = args.theme_name if hasattr(args, 'theme_name') else 'dark-green'
        return 0 if _change_theme(theme) else 2

    # Dynamic import and call
    func = _import_func(entry['func'])
    func_args = [getattr(args, a) for a in entry['args'] if hasattr(args, a)]
    func_kwargs = {}
    for k, v in entry.get('kwargs', {}).items():
        if hasattr(args, k):
            func_kwargs[k] = getattr(args, k)
        else:
            func_kwargs[k] = v

    try:
        result = func(*func_args, **func_kwargs)
    except (ValueError, TypeError) as exc:
        logger.debug("Command %s failed", cmd_name, exc_info=True)
        print(f"Error: {exc}")
        return 2
    except FileNotFoundError as exc:
        print(f"File not found: {exc.filename or exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        logger.debug("Command %s failed", cmd_name, exc_info=True)
        print(f"Error: {type(exc).__name__}: {exc}")
        print("Set BIOSUITE_DEBUG=1 for a full traceback.")
        if os.environ.get("BIOSUITE_DEBUG"):
            raise
        return 1

    if result is not None:
        if entry.get('report'):
            # Some core functions answer a dataclass or a dict whose repr is not what
            # a person should read; their module already has a formatter.
            result = _import_func(entry['report'])(result)
        if hasattr(result, 'to_string'):
            print(result.to_string())
        elif isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print(result)
    return 0


def _launch_gui():
    """Launch the GUI application."""
    from ..gui.main_window import BioSuiteApp
    app = BioSuiteApp()
    app.mainloop()


def _launch_api(port=8000):
    """Launch the REST API server (loopback by default).

    Binding 0.0.0.0 unconditionally exposed a development server to the whole
    network; the interface is now opt-in through BIOSUITE_API_HOST, and the
    same production-configuration gate as ``biosuite.api.server`` applies.
    """
    try:
        port = int(port)
    except (TypeError, ValueError):
        print(f"Invalid port: {port!r}. Use an integer between 1 and 65535.")
        return
    if not 1 <= port <= 65535:
        print(f"Invalid port: {port}. Use an integer between 1 and 65535.")
        return
    try:
        import uvicorn

        from ..api import app
        from ..api.server import ensure_safe_to_serve
    except ImportError:
        print("Install the API extra: pip install 'biosuite-ultra[api]'")
        return
    try:
        ensure_safe_to_serve()
    except SystemExit as exc:
        print(str(exc))
        return
    host = os.environ.get("BIOSUITE_API_HOST", "127.0.0.1")
    print(f"Starting API server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)


#: The themes the application actually ships.  ``set_theme`` treats anything that is
#: not one of its dark names as light, so without this list ``biosuite theme
#: dark-greene`` silently switched the whole app to a light theme and reported
#: success.
THEME_CHOICES = {'1': 'dark-green', '2': 'dark-purple', '3': 'light-blue'}
VALID_THEMES = tuple(dict.fromkeys(THEME_CHOICES.values()))


def _change_theme(theme_name):
    """Apply and save a theme.  Returns False when the name is not one of ours."""
    if theme_name not in VALID_THEMES:
        print(f"Unknown theme {theme_name!r}. "
              f"Choose one of: {', '.join(VALID_THEMES)}.")
        return False
    set_theme(theme_name)
    config['theme'] = theme_name
    save_config(config)
    print(f"Theme changed to: {theme_name}")
    return True


def _build_parser():
    """Build argparse parser for non-interactive mode."""
    parser = argparse.ArgumentParser(
        prog='biosuite',
        description=f'BioSuite Ultra — Bioinformatics Platform v{__version__}',
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
    parser.add_argument('--version', action='version',
                        version=f'BioSuite Ultra {__version__}')
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
    print(f"{D}   Bioinformatic Platform  ·  Ultra v{__version__}  ·  Pure Python{R}")
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


def _col2(n1, n2, c1="", c2="", _d1="", _d2=""):
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
    print(f"  {C}│{R}  {B}{G}59{R} Epigenomics (Meth/DMR/ATAC) {B}{G}60{R} Metabolomics Analysis")
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

    _section("ADVANCED TOOLS  ·  Interactive Plots, Provenance & File I/O", M)
    print(f"  {C}│{R}  {B}{G} d{R} Interactive Plot (Plotly) {B}{G} e{R} Provenance Tracker")
    print(f"  {C}│{R}  {B}{G} f{R} Plugin Manager            {B}{G} g{R} File Format Detector")
    print(f"  {C}│{R}  {B}{G} h{R} DESeq2 Normalization      {B}{G} i{R} VST Transformation")
    print(f"  {C}│{R}  {B}{G} j{R} Negative Binomial DE      {B}{G} k{R} Universal File Reader")
    _end_section()

    _section("SYSTEM  ·  Configuration & Export", W)
    print(f"  {C}│{R}  {B}{G}89{R} Launch GUI             {B}{G}90{R} Launch API Server")
    print(f"  {C}│{R}  {B}{G}99{R} Change Theme           {B}{G} 0{R} Exit")
    print(f"  {C}│{R}  {B}{G} s{R} Search Menu")
    _end_section()
    print()


_MENU_PART_RE = re.compile(r"^([a-z0-9]{1,3})\s+(.+)$")


def search_menu(query):
    """Print every menu item whose number, name or section matches ``query``.

    Matching is case-insensitive.  The items are parsed from the text
    :func:`print_menu` generates, so the search can never drift from the menu
    that is actually displayed.  Prints the matches grouped by section and
    returns nothing; the print is the result.
    """
    import contextlib
    import io

    query = query.strip().lower()
    if not query:
        print("  Usage: search <keyword> — e.g. search blast")
        return

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_menu()

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    section = ""
    matches = []
    for raw_line in buf.getvalue().splitlines():
        line = ansi.sub("", raw_line)
        if "┌─" in line:
            section = line.split("┌─", 1)[1].split("·")[0].rstrip("─ ").strip()
            continue
        if "│" not in line:
            continue
        body = line.split("│", 1)[1]
        for part in re.split(r"\s{2,}", body.strip()):
            m = _MENU_PART_RE.match(part.strip())
            if not m:
                continue
            number, name = m.group(1), m.group(2)
            if query in f"{number} {name} {section}".lower():
                matches.append((section, number, name))

    if not matches:
        print(f"  No menu items match {query!r}.")
        return

    print(f"  Menu search for {query!r}:")
    shown = set()
    for match_section, number, name in matches:
        if match_section not in shown:
            print(f"    ── {match_section}")
            shown.add(match_section)
        print(f"    {B}{G}{number}{R}  {name}")


def main_cli(argv=None):
    """Main CLI entry point. Supports interactive menu and non-interactive commands."""
    _enable_utf8_stdio()

    parser = _build_parser()

    # Non-interactive mode: parse arguments
    if argv is not None or len(sys.argv) > 1:
        args = parser.parse_args(argv)
        if args.theme:
            if not _change_theme(args.theme):
                return 2
        if args.gui:
            _launch_gui()
            return 0
        if args.api:
            _launch_api(args.port)
            return 0
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
            return _execute_command(args.command, cmd_args)
        return 0

    # Interactive mode. run.py no longer applies the theme up front, so
    # lightweight operations never load the plotting stack; the menu applies
    # it here, before any demo can draw.
    set_theme(config['theme'])
    while True:
        print_menu()
        try:
            choice = input(f"  {G}{B}▸ Select option:{R} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D / Ctrl-C at the prompt is a normal way to leave a CLI;
            # it used to escape as an EOFError traceback.
            print(f"\n  {G}Goodbye! {D}Thank you for using BioSuite Ultra.{R}\n")
            return 0

        if choice == '0':
            print(f"\n  {G}Goodbye! {D}Thank you for using BioSuite Ultra.{R}\n")
            break

        try:
            _handle_menu_choice(choice)
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {Y}Cancelled — returning to the menu.{R}")
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — a menu action must never kill the session
            logger.debug("Menu option %s failed", choice, exc_info=True)
            print(f"  {Y}{type(exc).__name__}: {exc}{R}")
            print(f"  {D}The action failed; the session is still running. "
                  f"Set BIOSUITE_DEBUG=1 for a full traceback.{R}")
            if os.environ.get("BIOSUITE_DEBUG"):
                import traceback
                traceback.print_exc()
    return 0


def _handle_menu_choice(choice):
    """Execute a single interactive menu choice.

    Raised exceptions propagate to :func:`main_cli`, which prints a one-line
    diagnostic and returns to the menu instead of terminating the session.
    """
    # ── Visualization (1-18) ──
    if choice == '1':
        from ..plotting.biological_plots import volcano_plot
        volcano_plot()
    elif choice == '2':
        from ..plotting.biological_plots import pca_plot
        pca_plot()
    elif choice == '3':
        from ..plotting.biological_plots import manhattan_plot
        manhattan_plot()
    elif choice == '4':
        from ..plotting.biological_plots import ma_plot
        ma_plot()
    elif choice == '5':
        from ..plotting.biological_plots import venn_diagram
        venn_diagram()
    elif choice == '6':
        from ..plotting.biological_plots import barplot_custom
        barplot_custom()
    elif choice == '7':
        from ..plotting.biological_plots import boxplot_custom
        boxplot_custom()
    elif choice == '8':
        from ..plotting.biological_plots import heatmap_custom
        heatmap_custom()
    elif choice == '9':
        from ..plotting.biological_plots import scatter_custom
        scatter_custom()
    elif choice == '10':
        from ..plotting.biological_plots import timeseries_plot
        timeseries_plot()
    elif choice == '11':
        from ..plotting.biological_plots import violin_plot
        violin_plot()
    elif choice == '12':
        from ..plotting.biological_plots import raincloud_plot
        raincloud_plot()
    elif choice == '13':
        from ..plotting.biological_plots import ridge_plot
        ridge_plot()
    elif choice == '14':
        from ..plotting.biological_plots import dot_plot
        dot_plot()
    elif choice == '15':
        from ..plotting.biological_plots import qq_plot
        qq_plot()
    elif choice == '16':
        from ..plotting.biological_plots import clustered_heatmap
        clustered_heatmap()
    elif choice == '17':
        from ..plotting.biological_plots import circos_plot
        circos_plot()
    elif choice == '18':
        from ..plotting.biological_plots import alignment_viewer
        alignment_viewer()

    # ── Sequence Analysis (20-28) ──
    elif choice == '20':
        from ..core.blast import format_blast_result, run_blast
        q = input("  Query FASTA: ").strip()
        db = input("  Database FASTA: ").strip()
        if q and db:
            result = run_blast(q, db)
            print(format_blast_result(result))
        else:
            print("  Both a query FASTA and a database FASTA are required.")

    elif choice == '21':
        from ..core.msa import (
            auto_align,
            consensus_sequence,
            format_alignment,
            read_fasta_for_msa,
        )
        fp = input("  FASTA file: ").strip()
        seqs = read_fasta_for_msa(fp)
        if seqs:
            result = auto_align(seqs)
            print(format_alignment(result))
            print(f"  Consensus: {consensus_sequence(result)}")
        else:
            print("  No sequences read from that file.")

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
        else:
            print("  No sequences read from that file.")

    # ── Transcriptomics (30-35) ──
    elif choice == '30':
        from ..core.trimming import format_trim_report, trim_fastq
        infile = input("  Input FASTQ: ").strip()
        out = input("  Output [auto]: ").strip() or None
        qual = int(input("  Quality [20]: ") or "20")
        ml = int(input("  Min length [36]: ") or "36")
        print(format_trim_report(trim_fastq(infile, out, qual, ml)))

    elif choice == '31':
        from ..core.quantification import format_quant_report, quantify_reads
        reads = input("  Reads FASTQ: ").strip()
        trans = input("  Transcriptome FASTA: ").strip()
        sample = input("  Sample name [s1]: ").strip() or "s1"
        print(format_quant_report(quantify_reads(reads, trans, sample)))

    elif choice == '32':
        from ..core.expression import differential_expression, read_counts_matrix
        fp = input("  Count matrix: ").strip()
        cond = input("  Conditions (comma-sep): ").strip().split(',')
        counts = read_counts_matrix(fp)
        if counts is not None:
            print(differential_expression(counts, cond).head(20).to_string())
        else:
            print("  No count matrix was read.")

    elif choice == '33':
        from ..core.expression import cpm_normalization, read_counts_matrix
        counts = read_counts_matrix(input("  Count matrix: ").strip())
        if counts is not None:
            print(cpm_normalization(counts).head(10).to_string())
        else:
            print("  No count matrix was read.")

    elif choice == '34':
        import numpy as np

        from ..core.expression import read_counts_matrix, tpm_normalization
        counts = read_counts_matrix(input("  Count matrix: ").strip())
        if counts is not None:
            lens = np.array([float(x) for x in input("  Gene lengths (kb, comma-sep): ").split(',')])
            print(tpm_normalization(counts, lens).head(10).to_string())
        else:
            print("  No count matrix was read.")

    elif choice == '35':
        from ..core.enrichment import (
            format_enrichment_report,
            run_gsea,
            run_kegg_enrichment,
            run_ora,
        )
        genes = [g.strip() for g in input("  Gene IDs: ").split(',') if g.strip()]
        m = input("  Method (ORA/GSEA/KEGG) [ORA]: ").strip().upper() or "ORA"
        if m not in ('ORA', 'GSEA', 'KEGG'):
            print(f"  Unknown method {m!r}: use ORA, GSEA or KEGG.")
            return
        if m == 'ORA':
            result = run_ora(genes)
        elif m == 'GSEA':
            result = run_gsea(genes)
        else:
            result = run_kegg_enrichment(genes)
        print(format_enrichment_report(result))

    # ── Genomics & NGS (40-47) ──
    elif choice == '40':
        from ..core.read_aligner import align_reads, format_alignment_report
        print(format_alignment_report(align_reads(input("  Reference: ").strip(), input("  Reads: ").strip())))

    elif choice == '41':
        from ..core.variant_calling import call_variants, filter_variants, variant_summary
        report = call_variants(input("  SAM/BAM: ").strip())
        print(variant_summary(report))
        flt = input("  Filter min_quality,min_depth,max_af (blank=keep all): ").strip()
        if flt:
            try:
                quality, depth, max_af = (float(x) for x in flt.split(','))
            except ValueError:
                print("  Filter needs three numbers: min_quality,min_depth,max_af.")
                return
            kept = filter_variants(report.variants, min_quality=quality,
                                   min_depth=depth, max_allele_freq=max_af)
            print(f"  Kept {len(kept)} of {len(report.variants)} variant(s) after filtering.")

    elif choice == '42':
        from ..core.peak_calling import (
            annotate_peaks_with_genes,
            call_peaks,
            format_peak_report,
        )
        report = call_peaks(input("  ChIP-seq BAM/SAM: ").strip())
        print(format_peak_report(report))
        bed = input("  Gene BED to annotate peaks (blank=skip): ").strip()
        if bed:
            if not report.peaks:
                print("  No peaks to annotate.")
                return
            try:
                annotate_peaks_with_genes(report.peaks, bed)
            except (OSError, ValueError) as e:
                print(f"  Could not read the gene BED: {e}")
                return
            for p in report.peaks[:20]:
                print(f"  {p.chrom}:{p.start}-{p.end} -> {p.name}")
            if len(report.peaks) > 20:
                print(f"  ... and {len(report.peaks) - 20} more.")

    elif choice == '43':
        from ..core.assembly import assemble, format_assembly_report
        print(format_assembly_report(assemble(input("  Reads FASTQ: ").strip())))

    elif choice == '44':
        from ..core.msa import read_fasta_for_msa
        from ..core.phylogeny import (
            distance_matrix,
            plot_phylogenetic_tree,
            upgma_tree,
        )
        import matplotlib.pyplot as plt

        seqs = read_fasta_for_msa(input("  Aligned FASTA: ").strip())
        if not seqs:
            print("  No sequences read from that file.")
        else:
            names = [n for n, _ in seqs]
            dist = distance_matrix([s for _, s in seqs])
            tree = upgma_tree(dist)
            plot_phylogenetic_tree(tree, names)
            print(f"  UPGMA tree from {len(names)} sequences "
                  f"({', '.join(names)}).")
            plt.show()
            plt.close()

    elif choice == '45':
        from ..core.ml_phylogeny import build_tree, format_phylo_report
        print(format_phylo_report(build_tree(input("  Aligned FASTA: ").strip(),
                                              bootstrap=int(input("  Bootstrap [100]: ") or "100"))))

    elif choice == '46':
        from ..core.bayesian_phylogeny import format_bayesian_report, run_bayesian
        print(format_bayesian_report(run_bayesian(input("  Aligned FASTA: ").strip())))

    elif choice == '47':
        from ..core.ngs import manhattan_from_vcf, read_vcf
        import matplotlib.pyplot as plt

        df = read_vcf(input("  VCF file: ").strip())
        if df is None:
            print("  No VCF was read.")
        else:
            import numpy as np

            mh = manhattan_from_vcf(df)
            plt.scatter(mh['POS'], mh['neg_log10'], s=2)
            plt.title("Manhattan Plot from VCF")
            print(f"  Plotting {len(mh)} variants; "
                  f"{int((mh['neg_log10'] > -np.log10(5e-8)).sum())} below p=5e-8.")
            plt.show()
            plt.close()

    # ── Single-Cell & Proteomics (50-53) ──
    elif choice == '50':
        from ..core.single_cell import (
            format_sc_report,
            load_count_matrix,
            run_full_pipeline,
        )
        adata, err = load_count_matrix(input("  Count matrix (h5ad/csv/10x): ").strip())
        if err:
            print(f"  Error: {err}")
        else:
            print(format_sc_report(run_full_pipeline(adata)[1]))

    elif choice == '51':
        from ..core.structure import format_structure_report, full_analysis
        pdb = input("  PDB ID or file: ").strip()
        info = full_analysis(filepath=pdb) if os.path.exists(pdb) else full_analysis(pdb_id=pdb)
        print(format_structure_report(info))

    elif choice == '52':
        from ..core.structure_prediction import (
            format_prediction_report,
            predict_structure,
        )
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
        from ..core.metabolism import (
            check_metabolism_tools,
            format_flux_report,
            knockout_analysis,
            run_fba,
        )
        model = input("  SBML file (blank=demo): ").strip()
        print(format_flux_report(run_fba(model_file=model if model else None)))
        genes = input("  Knockout gene IDs (comma-sep, blank=skip): ").strip()
        ids = [g.strip() for g in genes.split(',') if g.strip()]
        if ids:
            results = knockout_analysis(model if model else None, ids)
            if not results:
                if check_metabolism_tools()['cobra']:
                    print("  Knockout analysis returned no results.")
                else:
                    print("  No gene was tested: gene knockouts need COBRApy, "
                          "which is not installed.")
            else:
                for r in results:
                    flux = "n/a" if r.knockout_flux is None else f"{r.knockout_flux:.4f}"
                    print(f"  {r.gene}: knockout flux={flux}  essential={r.essential}")

    elif choice == '58':
        import numpy as np

        from ..core.popgen import format_popgen_report, full_analysis
        print("  Enter rows (0/1/2 genotypes, comma-sep):")
        rows = []
        while True:
            row = input("  > ").strip()
            if not row: break
            rows.append([int(x) for x in row.split(',')])
        if rows:
            print(format_popgen_report(full_analysis(np.array(rows))))

    elif choice == '59':
        from ..core.epigenomics import (
            atac_peak_stats,
            calculate_methylation_levels,
            find_dmrs,
            format_epigenomics_report,
            parse_atac_peaks,
            parse_bisulfite_bed,
        )
        mode = input("  Mode (METH/DMR/ATAC) [METH]: ").strip().upper() or "METH"
        if mode == 'METH':
            print(format_epigenomics_report(calculate_methylation_levels(parse_bisulfite_bed(input("  BED file: ").strip()))))
        elif mode == 'DMR':
            try:
                sites1 = parse_bisulfite_bed(input("  Group 1 BED file: ").strip())
                sites2 = parse_bisulfite_bed(input("  Group 2 BED file: ").strip())
            except OSError as e:
                print(f"  Could not read a BED file: {e}")
                return
            dmrs = find_dmrs(sites1, sites2)
            print(f"  DMRs found: {len(dmrs)}")
            for d in dmrs[:20]:
                print(f"  {d['chrom']}:{d['pos']}  delta={d['delta_methylation']:+.4f}  "
                      f"p={d['p_value']:.6g}")
            if len(dmrs) > 20:
                print(f"  ... and {len(dmrs) - 20} more.")
        elif mode == 'ATAC':
            try:
                peaks = parse_atac_peaks(input("  ATAC peak BED file: ").strip())
            except OSError as e:
                print(f"  Could not read the peak file: {e}")
                return
            stats = atac_peak_stats(peaks)
            if not stats:
                print("  No peaks parsed from that file.")
                return
            print(f"  Peaks: {stats['total_peaks']}  Total bp: {stats['total_bp']}")
            print(f"  Mean length: {stats['mean_length']:.1f}  "
                  f"Median length: {stats['median_length']:.1f}")
            print(f"  Chromosomes: {stats['chromosomes']}")
        else:
            print(f"  Unknown mode {mode!r}: use METH, DMR or ATAC.")

    elif choice == '60':
        import numpy as np

        from ..core.metabolomics import detect_peaks
        arr = np.array([float(x) for x in input("  Intensities (comma-sep): ").split(',')])
        features = detect_peaks(arr)
        print(f"  Detected {len(features)} peaks")

    elif choice == '61':
        from ..core.md_simulation import format_md_report, run_simulation
        print(format_md_report(run_simulation(input("  PDB file: ").strip(),
                                               steps=int(input("  Steps [1000]: ") or "1000"))))

    elif choice == '62':
        import numpy as np

        from ..core.bio_ml import format_ml_report, train_random_forest, train_svm
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
    elif choice == '101':
        from ..plotting.specialized_plots import gsea_plot
        gsea_plot()
    elif choice == '102':
        from ..plotting.specialized_plots import motif_logo
        motif_logo()
    elif choice == '103':
        from ..plotting.specialized_plots import sankey_diagram
        sankey_diagram()
    elif choice == '104':
        from ..plotting.specialized_plots import umap_plot
        umap_plot()
    elif choice == '105':
        from ..plotting.math_plots import sine_plot
        sine_plot()
    elif choice == '106':
        from ..plotting.math_plots import cosine_plot
        cosine_plot()
    elif choice == '107':
        from ..plotting.math_plots import linear_plot
        linear_plot()
    elif choice == '108':
        from ..plotting.math_plots import quadratic_plot
        quadratic_plot()
    elif choice == '109':
        from ..plotting.math_plots import cubic_plot
        cubic_plot()
    elif choice == '110':
        from ..plotting.math_plots import exponential_plot
        exponential_plot()
    elif choice == '111':
        from ..plotting.math_plots import logistic_plot
        logistic_plot()

    # ── Analysis (120-125) ──  (renumbered from 81-86: the display
    # labels these 120-125 and option 81 already means Synteny Dotplot —
    # the old shadowing made Synteny unreachable and 120-125 "Invalid".)
    elif choice == '120':
        from ..core.codon_usage import codon_usage_table, format_codon_usage
        seq = input("  DNA sequence: ").strip().upper()
        result = codon_usage_table(seq)
        print(format_codon_usage(result))

    elif choice == '121':
        from ..core.codon_usage import format_kmer_composition, kmer_composition
        seq = input("  Sequence: ").strip().upper()
        k = int(input("  K-mer size [3]: ") or "3")
        result = kmer_composition(seq, k=k)
        print(format_kmer_composition(result))

    elif choice == '122':
        from ..core.codon_usage import sequence_complexity
        seq = input("  Sequence: ").strip().upper()
        result = sequence_complexity(seq)
        print(f"  Average complexity: {result['average_complexity']}")
        print(f"  Low complexity: {result['is_low_complexity']}")

    elif choice == '123':
        import numpy as np

        from ..core.survival import format_km_result, kaplan_meier
        print("  Enter survival times (comma-sep):")
        times = np.array([float(x) for x in input("  Times: ").split(',')])
        print("  Enter event indicators (1=event, 0=censored, comma-sep):")
        events = np.array([int(x) for x in input("  Events: ").split(',')])
        result = kaplan_meier(times, events)
        print(format_km_result(result))

    elif choice == '124':
        import matplotlib.pyplot as plt

        from ..plotting.network_plots import (
            create_ppi_network,
            network_statistics,
            plot_network,
        )
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
            plt.show()
            plt.close()

    elif choice == '125':
        from ..plotting.network_plots import create_ppi_network, network_statistics
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
            else:
                print(f"  Ignored line {line!r}: expected 'name,element,element'.")
        if sets_dict:
            import matplotlib.pyplot as plt

            from ..plotting.upset_plots import plot_upset
            fig = plot_upset(sets_dict, title="UpSet Plot")
            plt.show()
            plt.close()
        else:
            print("  No sets entered.")

    elif choice == '78':
        import matplotlib.pyplot as plt

        from ..plotting.genome_browser import (
            create_bed_track,
            create_coverage_from_bam,
            create_variant_track,
            plot_genome_tracks,
        )
        print("  Track types: coverage, bed, variant")
        tracks = []
        while True:
            ttype = input("  Track type (empty to finish): ").strip()
            if not ttype: break
            if ttype == 'bed':
                fp = input("  BED file: ").strip()
                if fp and os.path.exists(fp):
                    tracks.append(create_bed_track(fp))
                else:
                    print("  BED file not found; track skipped.")
            elif ttype == 'variant':
                fp = input("  VCF file: ").strip()
                if fp and os.path.exists(fp):
                    tracks.append(create_variant_track(fp))
                else:
                    print("  VCF file not found; track skipped.")
            elif ttype == 'coverage':
                fp = input("  BAM/SAM file: ").strip()
                if fp and os.path.exists(fp):
                    tracks.append(create_coverage_from_bam(fp))
                else:
                    print("  BAM/SAM file not found; track skipped.")
            else:
                print(f"  Unknown track type {ttype!r}; expected bed, variant or coverage.")
        if tracks:
            fig = plot_genome_tracks(tracks, title="Genome Browser")
            plt.show()
            plt.close()
        else:
            print("  No tracks added.")

    elif choice == '79':
        import numpy as np

        from ..plotting.interactive_plots import interactive_scatter
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
            rng = np.random.default_rng(42)
            x = rng.standard_normal(100)
            y = x * 2 + rng.standard_normal(100)
            fig = interactive_scatter(x, y, title="Interactive Scatter (demo)",
                                      output_html="scatter_interactive.html")
            print("  Saved: scatter_interactive.html")

    elif choice == '80':
        import matplotlib.pyplot as plt

        from ..plotting.conservation_plots import plot_logo_with_conservation
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
        import matplotlib.pyplot as plt

        from ..plotting.synteny import compute_synteny_score, plot_synteny_dotplot
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
        from ..core.orf_finder import (
            find_restriction_sites,
            format_restriction_sites,
        )
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
        from ..core.databases import format_search_results, search_ncbi
        query = input("  Search query: ").strip()
        results = search_ncbi(query)
        print(format_search_results(results))

    elif choice == '68':
        from ..core.databases import format_search_results, search_uniprot
        query = input("  Protein/gene name: ").strip()
        results = search_uniprot(query)
        print(format_search_results(results))

    elif choice == '69':
        from ..core.databases import format_search_results, search_pdb
        query = input("  Protein name or keyword: ").strip()
        results = search_pdb(query)
        print(format_search_results(results))

    elif choice == '70':
        from ..core.databases import format_search_results, search_kegg
        query = input("  Pathway/compound name: ").strip()
        results = search_kegg(query)
        print(format_search_results(results))

    elif choice == '71':
        from ..core.databases import format_search_results, search_ensembl
        gene = input("  Gene symbol: ").strip()
        species = input("  Species [human]: ").strip() or "human"
        results = search_ensembl(gene, species=species)
        print(format_search_results(results))

    elif choice == '72':
        from ..core.databases import format_search_results, search_all
        query = input("  Search query: ").strip()
        results = search_all(query)
        print(format_search_results(results))

    # ── File Formats (73-76) ──
    elif choice == '73':
        from ..core.file_formats import (
            bed_to_dataframe,
            format_bed_summary,
            parse_bed,
        )
        fp = input("  BED file path: ").strip()
        records = parse_bed(fp)
        print(format_bed_summary(records))
        df = bed_to_dataframe(records)
        print(df.head(10).to_string())

    elif choice == '74':
        from ..core.file_formats import (
            format_gff_summary,
            parse_gff,
        )
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
        print("  Add steps as 'label=function', empty line to finish.")
        print(f"  Known functions: {', '.join(sorted(SAFE_FUNCTIONS))}")
        print("  (or a fully qualified 'biosuite.module:function' path)")
        steps = []
        while True:
            line = input("  > ").strip()
            if not line: break
            # 'label=function' is unambiguous for 'module:function' targets;
            # the legacy 'label:function' form is still accepted.
            parts = line.split('=', 1) if '=' in line else line.split(':', 1)
            if len(parts) == 2:
                name = parts[0].strip()
                try:
                    func = resolve_safe_callable(parts[1].strip())
                except (ValueError, ImportError, AttributeError) as exc:
                    print(f"  {Y}{exc}{R}")
                    continue
                # ``PipelineStep`` already carries per-step args, but the builder never
                # asked for any — so every step of every pipeline died with
                # "gc_content() missing 1 required positional argument: 'seq'".
                arg_text = input(f"  Arguments for {name} (comma-sep, empty = none): ")
                args = [a.strip() for a in arg_text.split(",") if a.strip()]
                steps.append({"name": name, "func": func, "args": args})
            else:
                print(f"  {Y}Not a step: {line!r}. Use 'label=function'.{R}")
        if steps:
            p = Pipeline("cli_pipeline")
            p.add_steps(steps)
            p.run()
            print(format_pipeline_report(p))
        else:
            print("  No steps added — nothing to run.")

    elif choice == '93':
        print("  Batch Processor — run analysis on multiple samples")
        print(f"  Known functions: {', '.join(sorted(SAFE_FUNCTIONS))}")
        func_name = input("  Function (name or 'biosuite.module:function'): ").strip()
        try:
            func = resolve_safe_callable(func_name)
        except (ValueError, ImportError, AttributeError) as exc:
            print(f"  {Y}{exc}{R}")
            return
        # Each value typed here is passed to the function as its own job, so the
        # prompt says that instead of implying they are just labels.
        values = input("  Values to analyse (comma-sep; each is passed to the function): ")
        samples = [s.strip() for s in values.split(',') if s.strip()]
        if samples:
            bp = BatchProcessor("cli_batch")
            bp.add_samples(samples, func)
            bp.run(max_workers=4)
            print(format_batch_report(bp))
        else:
            print("  No values given — nothing to run.")

    elif choice == '94':
        from ..core.utils import count_analysis_modules, count_test_files
        title = input("  Report title [BioSuite Report]: ").strip() or "BioSuite Report"
        report = create_report(title)
        report.add_text("This report was generated by BioSuite Ultra.")
        report.add_stats({"Modules": count_analysis_modules(),
                          "Test files": count_test_files()})
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
        import matplotlib.pyplot as plt

        from ..core.pathway_viz import (
            create_kegg_style_pathway,
            draw_pathway,
            format_pathway_report,
        )
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
            if not fp:
                print("  No file given — nothing to analyse.")
                return
            if not os.path.isfile(fp):
                print(f"  File not found: {fp}")
                return
            # A missing path used to fall back to 2,000 simulated SNPs without a
            # word, so the report a user read was not the report they asked for.
            data = pd.read_csv(fp)
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
        tm = THEME_CHOICES.get(t)
        if tm:
            _change_theme(tm)
        else:
            print(f"  Not a theme choice: {t!r}. Type 1, 2 or 3.")

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
            else:
                print(f"  Ignored {line!r}: expected 'name:sequence'.")
        if not seqs:
            # Demo
            seqs = [("Ecoli_16S", "TGGAGGAAGGTGGGGACGACGTCAGTATCGAATCTTGGATCAGGATCACCTCCGGA"),
                     ("Staph_16S", "AGCCATGCAGCACCTGTCTCAGCTTCCCGAAGGCACTATACGTAGATCGAAAGTTGAT")]
            print("  Using demo data...")
        result = classify_16s_rna(seqs)
        print(format_16s_report(result))

    elif choice == 'b':
        import numpy as np

        from ..core.variant_calling import (
            detect_cnv,
            detect_structural_variants,
            format_cnv_report,
            format_sv_report,
        )
        print("  SV / CNV Detection from Coverage Data")
        use_demo = input("  Use demo data? (y/n) [y]: ").strip().lower() or 'y'
        if use_demo == 'y':
            rng = np.random.default_rng(42)
            cov = rng.poisson(30, 10000).astype(float)
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
                rng = np.random.default_rng(42)
                cov = rng.poisson(30, 10000).astype(float)
                ref = np.ones(10000) * 30
        svs = detect_structural_variants(cov, ref)
        print(format_sv_report(svs))
        cnv = detect_cnv(cov, ref)
        print(f"\n{format_cnv_report(cnv)}")

    elif choice == 'c':
        import numpy as np

        from ..core.file_formats import (
            bigwig_summary,
            format_bigwig_summary,
            read_bigwig,
        )
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
        import matplotlib.pyplot as plt

        from ..plotting.plasmid_map import (
            create_sample_plasmid,
            draw_plasmid,
            format_plasmid_report,
        )
        pm = create_sample_plasmid()
        print(format_plasmid_report(pm))
        fig = draw_plasmid(pm)
        plt.show()
        plt.close()

    elif choice == 'm':
        from ..core.cloning import (
            find_restriction_sites,
            format_digest_report,
            simulate_digestion,
        )
        seq = input("  DNA sequence: ").strip().upper()
        enzymes = input("  Enzymes (comma-sep, e.g. EcoRI,BamHI): ").strip().split(',')
        enzymes = [e.strip() for e in enzymes if e.strip()]
        if seq and enzymes:
            result = simulate_digestion(seq, enzymes)
            print(format_digest_report(result))
        else:
            print("  Invalid input.")

    elif choice == 'n':
        import matplotlib.pyplot as plt

        from ..core.cloning import plot_virtual_gel, simulate_digestion
        seq = input("  DNA sequence: ").strip().upper()
        enzymes = input("  Enzymes (comma-sep): ").strip().split(',')
        enzymes = [e.strip() for e in enzymes if e.strip()]
        if seq and enzymes:
            result = simulate_digestion(seq, enzymes)
            sizes = [f['size'] for f in result['fragments']]
            print(f"  Virtual gel of {len(sizes)} fragment(s): "
                  f"{', '.join(str(s) for s in sizes)} bp.")
            plot_virtual_gel(result['fragments'])
            plt.show()
            plt.close()
        else:
            print("  Invalid input.")

    elif choice == 'o':
        from ..core.cloning import (
            design_primers,
            format_pcr_report,
            format_primer_report,
            simulate_pcr,
        )
        template = input("  Template sequence: ").strip().upper()
        length = int(input("  Primer length [20]: ") or "20")
        if not template:
            print("  No sequence provided.")
        else:
            # ``design_primers(target_seq, primer_length)`` answers a dict, so the
            # old ``fwd, rev = design_primers(template, start, end)`` both passed
            # three positional arguments (TypeError for every template) and
            # unpacked a dict into its key names.
            primers = design_primers(template, primer_length=length)
            print(format_primer_report(primers))
            amplicon = simulate_pcr(template, primers["forward"], primers["reverse"])
            print(format_pcr_report(amplicon))

    elif choice == 'p':
        import matplotlib.pyplot as plt

        from ..plotting.sequence_viewer import draw_sequence_view
        text = input("  DNA sequence (or FASTA path): ").strip()
        seq = text
        if os.path.exists(text):
            from ..core.sequence import read_fasta
            records = read_fasta(text)
            seq = records[0][1] if records else ""
        if not seq:
            print("  No sequence to draw.")
        else:
            print(f"  Sequence view of {len(seq)} bp.")
            draw_sequence_view(seq.upper())
            plt.show()
            plt.close()

    elif choice == 'q':
        from ..core.cloning import format_ligation_report, simulate_ligation
        # ``simulate_ligation`` ligates fragment *sequences* and reports
        # ``size``/``product``; this option asked for integer lengths, fed them in,
        # and read a key the result does not have — so it either raised
        # ``AttributeError: 'int' object has no attribute 'upper'`` or printed
        # "Ligated product: 0 bp" for a product it had never measured.
        print("  Enter fragment sequences (one per line, empty to finish):")
        fragments = []
        while True:
            line = input("  > ").strip().upper()
            if not line:
                break
            fragments.append(line)
        if not fragments:
            print("  No fragments provided.")
        else:
            print(format_ligation_report(simulate_ligation(fragments)))

    elif choice == '89':
        _launch_gui()
        print(f"\n  {G}GUI closed.{R} {D}CLI ready.{R}")

    # ── Advanced Tools (d-k) ──
    elif choice == 'd':
        import numpy as np

        from ..plotting.plot_api import (
            boxplot,
            heatmap,
            manhattan,
            pca,
            scatter,
            volcano,
        )
        print("  Interactive Plot Generator (Plotly)")
        print("  Plot types: 1=Volcano  2=PCA  3=Manhattan  4=Scatter  5=Boxplot  6=Heatmap")
        ptype = input("  Select plot type [1]: ").strip() or "1"
        rng = np.random.default_rng(42)
        if ptype == '1':
            fc = rng.normal(0, 1.5, 500)
            pvals = rng.uniform(0, 1, 500)
            pvals[:30] = rng.uniform(1e-6, 0.05, 30)
            fig = volcano(fc, pvals, interactive=True, title="Interactive Volcano")
        elif ptype == '2':
            data = rng.standard_normal((30, 50))
            fig = pca(data, labels=['Ctrl']*15 + ['Treat']*15, interactive=True)
        elif ptype == '3':
            chroms = rng.choice(['chr1', 'chr2', 'chr3'], 200)
            positions = rng.integers(1, 1000000, 200)
            pvals = rng.uniform(0, 1, 200)
            fig = manhattan(chroms, positions, pvals, interactive=True)
        elif ptype == '4':
            x = rng.standard_normal(100)
            y = x * 2 + rng.standard_normal(100)
            fig = scatter(x, y, interactive=True)
        elif ptype == '5':
            data = {'Ctrl': rng.standard_normal(30).tolist(),
                    'Treat': (rng.standard_normal(30) + 1).tolist()}
            fig = boxplot(data, interactive=True)
        elif ptype == '6':
            data = rng.standard_normal((10, 8))
            fig = heatmap(data, interactive=True)
        else:
            print("  Invalid plot type."); return
        outfile = input("  Output HTML [plot.html]: ").strip() or "plot.html"
        fig.write_html(outfile)
        print(f"  Saved: {outfile}")

    elif choice == 'e':
        print("  Provenance Tracker")
        print("  Actions: 1=Start session  2=Record step  3=View history  4=Export HTML  5=Export JSON")
        action = input("  Select action [1]: ").strip() or "1"
        db = input("  Database file [analysis.db]: ").strip() or "analysis.db"
        tracker = _provenance_tracker(db)
        if action == '1':
            print(f"  Session started: {tracker.session_id}")
        elif action == '2':
            module = input("  Module name: ").strip()
            func = input("  Function name: ").strip()
            params = input("  Parameters (JSON): ").strip()
            result = input("  Result summary: ").strip()
            import json
            try:
                params_dict = json.loads(params) if params else {}
                if not isinstance(params_dict, dict):
                    params_dict = {"value": params_dict}
            except json.JSONDecodeError:
                params_dict = {"raw": params}
            tracker.record(module, func, params_dict, result)
            print(f"  Step recorded. {D}({tracker.session_id}, "
                  f"{len(tracker.get_steps())} in this session){R}")
        elif action == '3':
            print(tracker.summary())
        elif action == '4':
            outfile = input("  Output HTML [provenance.html]: ").strip() or "provenance.html"
            tracker.export_html(outfile)
            print(f"  Saved: {outfile}")
        elif action == '5':
            outfile = input("  Output JSON [provenance.json]: ").strip() or "provenance.json"
            tracker.export_json(outfile)
            print(f"  Saved: {outfile}")
        else:
            print(f"  Not an action: {action!r}. Type 1, 2, 3, 4 or 5.")

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
            else:
                print("  No plugin name given.")
        else:
            print(f"  Unknown action {action!r}.")

    elif choice == 'g':
        from ..core.file_formats import (
            detect_file_format,
            format_file_summary,
            read_file,
        )
        fp = input("  File path: ").strip()
        if fp and os.path.exists(fp):
            fmt = detect_file_format(fp)
            print(f"  Detected format: {fmt}")
            result = read_file(fp)
            print(format_file_summary(result))
        else:
            print("  File not found.")

    elif choice == 'h':
        from ..core.expression import deseq2_normalization, read_counts_matrix
        fp = input("  Count matrix: ").strip()
        counts = read_counts_matrix(fp)
        if counts is not None:
            normalized = deseq2_normalization(counts)
            print("  DESeq2-normalized data:")
            print(normalized.head(10).to_string())
        else:
            print("  No count matrix was read.")

    elif choice == 'i':
        from ..core.expression import (
            read_counts_matrix,
            variance_stabilizing_transformation,
        )
        fp = input("  Count matrix: ").strip()
        counts = read_counts_matrix(fp)
        if counts is not None:
            vst = variance_stabilizing_transformation(counts)
            print("  VST-transformed data:")
            print(vst.head(10).to_string())
        else:
            print("  No count matrix was read.")

    elif choice == 'j':
        from ..core.expression import differential_expression, read_counts_matrix
        fp = input("  Count matrix: ").strip()
        cond = input("  Conditions (comma-sep): ").strip().split(',')
        counts = read_counts_matrix(fp)
        if counts is not None:
            result = differential_expression(counts, cond, method='nb')
            print("  Differential Expression (Negative Binomial):")
            print(result.head(20).to_string())
        else:
            print("  No count matrix was read.")

    elif choice == 'k':
        from ..core.file_formats import (
            detect_file_format,
            format_file_summary,
            read_file,
        )
        fp = input("  File path: ").strip()
        if fp and os.path.exists(fp):
            fmt = detect_file_format(fp)
            print(f"  Detected format: {fmt}")
            result = read_file(fp)
            print(format_file_summary(result))
            if 'records' in result and result['records']:
                print("\n  First 5 records:")
                for rec in result['records'][:5]:
                    print(f"    {rec}")
            else:
                print("  The file holds no records to list.")
        else:
            print("  File not found.")

    elif choice == 's':
        query = input(f"  {B}Search the menu for:{R} ").strip()
        search_menu(query)

    elif choice == '90':
        port = input("  Port [8000]: ").strip() or "8000"
        print(f"  API Docs: http://localhost:{port}/docs")
        print("  Press Ctrl+C to stop\n")
        try:
            _launch_api(port)
        except KeyboardInterrupt:
            print(f"\n  {G}API server stopped.{R}")

    else:
        print(f"  {Y}Invalid option. Type 0 to exit.{R}")


def main_cli_gui():
    """Entry point for biosuite-gui command — launches GUI directly."""
    from ..gui.main_window import BioSuiteApp
    app = BioSuiteApp()
    app.mainloop()
