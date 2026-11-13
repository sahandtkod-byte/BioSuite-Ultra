"""
Jupyter notebook integration for BioSuite.

Provides IPython magic commands and ipywidgets for interactive analysis
in Jupyter notebooks and Google Colab.

Usage in Jupyter:
    %pip install biosuite-ultra
    %load_ext biosuite.notebook

    # Quick analysis
    %biosuite gc ATCGATCGATCG

    # Interactive widgets
    from biosuite.notebook import SequenceAnalyzer
    analyzer = SequenceAnalyzer()
    analyzer.show()
"""
try:
    from IPython.core.magic import Magics, line_magic, magics_class
    # ``display`` belongs to IPython, not to ipywidgets: %biostats calls it, and
    # with only IPython installed the name was never bound, so the magic printed
    # the shape of the frame and then died with NameError.
    from IPython.display import clear_output, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

try:
    import ipywidgets as widgets
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

#: The always-available surface plus whichever optional group is installed, so a
#: star import on a reduced install never demands a name that does not exist.
__all__ = [
    "HAS_IPYTHON",
    "HAS_WIDGETS",
    "quick_gc",
    "quick_translate",
    "quick_align",
    "quick_blast",
]
if HAS_IPYTHON:
    __all__ += ["BioSuiteMagics", "load_ipython_extension"]
if HAS_WIDGETS:
    __all__ += ["SequenceAnalyzer", "AlignmentViewer", "PlotExplorer"]

#: Optional exports mapped to the dependency group that provides them, so an
#: explicit import on a reduced install fails with the fix, not a bare name error.
_OPTIONAL_EXPORTS = {
    "BioSuiteMagics": "IPython",
    "load_ipython_extension": "IPython",
    "SequenceAnalyzer": "ipywidgets",
    "AlignmentViewer": "ipywidgets",
    "PlotExplorer": "ipywidgets",
}


def __getattr__(name):
    """PEP 562: say *why* an optional export is missing, and how to get it."""
    requirement = _OPTIONAL_EXPORTS.get(name)
    if requirement is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    raise ImportError(
        f"{name} requires the optional dependency {requirement!r}, which is not "
        "installed. Install it (e.g. pip install biosuite-ultra[notebook]) to "
        f"use {name}."
    )


def _split_magic_args(line):
    """Split a magic command line into arguments, honouring quotes.

    Double or single quotes group whitespace into one argument (paths with
    spaces), and quotes may sit inside a token.  Backslashes are ordinary
    characters, so Windows paths pass through untouched.  An unterminated
    quote raises ValueError naming the quote character.
    """
    tokens = []
    current = []
    quote = None
    for char in line:
        if quote:
            if char == quote:
                quote = None
            else:
                current.append(char)
        elif char in ('"', "'"):
            quote = char
        elif char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if quote:
        raise ValueError(f"unterminated {quote!r} quote")
    if current:
        tokens.append("".join(current))
    return tokens


# ── Magic Commands ───────────────────────────────────────────────────────────

if HAS_IPYTHON:
    @magics_class
    class BioSuiteMagics(Magics):
        """IPython magic commands for BioSuite."""

        @line_magic
        def biosuite(self, line):
            """Quick bioinformatics analysis.

            Usage:
                %biosuite gc ATCGATCG
                %biosuite revcomp ATCGATCG
                %biosuite translate ATGAAATTTTAA
                %biosuite stats ATCGATCGATCG
                %biosuite search QUERY.fasta DB.fasta
                (quote arguments containing spaces: search "my query.fasta" DB.fasta)
            """
            try:
                parts = _split_magic_args(line)
            except ValueError as exc:
                print(f"Error: {exc}; quote paths that contain spaces.")
                return
            if not parts:
                print("BioSuite magic commands:")
                print("  %biosuite gc <sequence>        - GC content")
                print("  %biosuite revcomp <sequence>    - Reverse complement")
                print("  %biosuite translate <sequence>  - Translate to protein")
                print("  %biosuite stats <sequence>      - Sequence statistics")
                print("  %biosuite search <query> <db>   - BLAST search")
                print("  %biosuite help                  - Show this help")
                return

            cmd = parts[0].lower()

            if cmd == 'help':
                print("BioSuite magic commands:")
                print("  %biosuite gc <sequence>        - GC content")
                print("  %biosuite revcomp <sequence>    - Reverse complement")
                print("  %biosuite translate <sequence>  - Translate to protein")
                print("  %biosuite stats <sequence>      - Sequence statistics")
                print("  %biosuite search <query> <db>   - BLAST search")
                return

            if cmd == 'gc':
                seq = parts[1] if len(parts) > 1 else input("Sequence: ")
                from biosuite.core.sequence import gc_content
                result = gc_content(seq)
                print(f"GC content: {result:.2f}%")
                return result

            elif cmd == 'revcomp':
                seq = parts[1] if len(parts) > 1 else input("Sequence: ")
                from biosuite.core.sequence import reverse_complement
                result = reverse_complement(seq)
                print(f"Reverse complement: {result}")
                return result

            elif cmd == 'translate':
                seq = parts[1] if len(parts) > 1 else input("Sequence: ")
                from biosuite.core.sequence import translate
                result = translate(seq)
                print(f"Protein: {result}")
                return result

            elif cmd == 'stats':
                seq = parts[1] if len(parts) > 1 else input("Sequence: ")
                from biosuite.core.sequence import sequence_stats
                result = sequence_stats(seq)
                for k, v in result.items():
                    print(f"  {k}: {v}")
                return result

            elif cmd == 'search':
                if len(parts) < 3:
                    print("Usage: %biosuite search <query.fasta> <database.fasta>")
                    return
                from biosuite.core.blast import run_blast
                result = run_blast(parts[1], parts[2])
                print(f"Found {result.num_hits} hits")
                for hit in result.top_hits(5):
                    print(f"  {hit}")
                return result

            else:
                print(f"Unknown command: {cmd}. Use %biosuite help for options.")

        @line_magic
        def biostats(self, line):
            """Quick statistics on a variable.

            Usage:
                %biostats my_dataframe
                %biostats my_array
            """
            from IPython import get_ipython
            ip = get_ipython()
            var_name = line.strip()
            if not var_name:
                print("Usage: %biostats <variable_name>")
                return

            try:
                obj = ip.user_ns[var_name]
            except KeyError:
                print(f"Variable '{var_name}' not found")
                return

            import numpy as np
            import pandas as pd

            if isinstance(obj, pd.DataFrame):
                print(f"DataFrame: {obj.shape[0]} rows x {obj.shape[1]} columns")
                columns = list(obj.columns)
                if len(columns) > 10:
                    columns = columns[:10] + ["..."]
                print(f"Columns: {columns}")
                print("\nFirst 5 rows:")
                display(obj.head())
            elif isinstance(obj, np.ndarray):
                print(f"Array: shape={obj.shape}, dtype={obj.dtype}")
                if obj.size > 0:
                    print(f"Min: {obj.min():.4f}")
                    print(f"Max: {obj.max():.4f}")
                    print(f"Mean: {obj.mean():.4f}")
                    print(f"Std: {obj.std():.4f}")
            else:
                print(f"Type: {type(obj).__name__}")
                print(f"Value: {obj}")

        @line_magic
        def bioimport(self, line):
            """Import common BioSuite functions.

            Usage:
                %bioimport sequence
                %bioimport alignment
                %bioimport all
            """
            from IPython import get_ipython
            ip = get_ipython()

            # What each target promises to put in the namespace.  The module path
            # is stored next to the names instead of being rebuilt from the key,
            # because "plotting" is not under biosuite.core -- as a key-derived
            # name it imported biosuite.core.plotting, which does not exist, so
            # %bioimport plotting failed while %bioimport all loaded the same
            # functions from a different place.
            modules = {
                'sequence': ('biosuite.core.sequence',
                             ['gc_content', 'reverse_complement', 'translate',
                              'read_fasta', 'read_fastq', 'sequence_stats']),
                'alignment': ('biosuite.core.alignment',
                              ['needleman_wunsch', 'smith_waterman']),
                'phylogeny': ('biosuite.core.phylogeny',
                              ['distance_matrix', 'upgma_tree']),
                'expression': ('biosuite.core.expression',
                               ['differential_expression', 'cpm_normalization',
                                'tpm_normalization']),
                'blast': ('biosuite.core.blast', ['run_blast']),
                'plotting': ('biosuite.plotting.plot_api',
                             ['volcano', 'pca', 'manhattan', 'heatmap', 'scatter',
                              'boxplot']),
            }

            def load(name):
                """Put a module's advertised functions in the user namespace.

                Returns False when the module could not be imported or promised a
                function it does not have; either way the reason is printed
                rather than swallowed.
                """
                path, funcs = modules[name]
                try:
                    mod = __import__(path, fromlist=funcs)
                except ImportError as exc:
                    print(f"  Skipping {name}: {exc}")
                    return False
                missing = [f for f in funcs if not hasattr(mod, f)]
                for func_name in funcs:
                    if not hasattr(mod, func_name):
                        continue
                    ip.user_ns[func_name] = getattr(mod, func_name)
                print(f"  Loaded {name}: {', '.join(funcs)}")
                if missing:
                    print(f"  Not found in {path}: {', '.join(missing)}")
                return not missing

            target = line.strip().lower()
            if target == 'all':
                unready = [name for name in modules if not load(name)]
                if unready:
                    print(f"\nNot loaded: {', '.join(unready)}")
                else:
                    print("\nAll functions loaded into namespace!")
            elif target in modules:
                load(target)
            else:
                print(f"Available modules: {', '.join(modules)}, all")


    def load_ipython_extension(ipython):
        """Register BioSuite magic commands."""
        ipython.register_magics(BioSuiteMagics)
        print("BioSuite magic commands loaded!")
        print("  %biosuite help - Show available commands")
        print("  %bioimport all - Import all functions")


# ── Interactive Widgets ──────────────────────────────────────────────────────

if HAS_WIDGETS:
    class SequenceAnalyzer:
        """Interactive sequence analyzer widget for Jupyter."""

        def __init__(self):
            self.output = widgets.Output()

        def show(self):
            """Display the interactive analyzer."""
            seq_input = widgets.Textarea(
                placeholder='Enter DNA sequence (e.g., ATCGATCGATCG)',
                description='Sequence:',
                layout=widgets.Layout(width='100%', height='80px')
            )

            gc_btn = widgets.Button(description='GC Content', button_style='success')
            revcomp_btn = widgets.Button(description='Reverse Comp', button_style='info')
            translate_btn = widgets.Button(description='Translate', button_style='warning')
            stats_btn = widgets.Button(description='Full Stats', button_style='')

            output_area = widgets.Output()

            def on_gc_click(b):
                from biosuite.core.sequence import gc_content
                with output_area:
                    clear_output()
                    seq = seq_input.value.upper().strip()
                    if seq:
                        result = gc_content(seq)
                        print(f"GC content: {result:.2f}%")
                    else:
                        print("Please enter a sequence")

            def on_revcomp_click(b):
                from biosuite.core.sequence import reverse_complement
                with output_area:
                    clear_output()
                    seq = seq_input.value.upper().strip()
                    if seq:
                        result = reverse_complement(seq)
                        print(f"Reverse complement:\n{result}")
                    else:
                        print("Please enter a sequence")

            def on_translate_click(b):
                from biosuite.core.sequence import translate
                with output_area:
                    clear_output()
                    seq = seq_input.value.upper().strip()
                    if seq:
                        result = translate(seq)
                        print(f"Protein:\n{result}")
                    else:
                        print("Please enter a sequence")

            def on_stats_click(b):
                from biosuite.core.sequence import sequence_stats
                with output_area:
                    clear_output()
                    seq = seq_input.value.upper().strip()
                    if seq:
                        result = sequence_stats(seq)
                        for k, v in result.items():
                            print(f"  {k}: {v}")
                    else:
                        print("Please enter a sequence")

            gc_btn.on_click(on_gc_click)
            revcomp_btn.on_click(on_revcomp_click)
            translate_btn.on_click(on_translate_click)
            stats_btn.on_click(on_stats_click)

            button_box = widgets.HBox([gc_btn, revcomp_btn, translate_btn, stats_btn])
            display(widgets.VBox([
                widgets.HTML("<h3>BioSuite Sequence Analyzer</h3>"),
                seq_input,
                button_box,
                output_area
            ]))

        def gc_content(self, seq):
            from biosuite.core.sequence import gc_content
            return gc_content(seq)

        def reverse_complement(self, seq):
            from biosuite.core.sequence import reverse_complement
            return reverse_complement(seq)

        def translate(self, seq):
            from biosuite.core.sequence import translate
            return translate(seq)

        def sequence_stats(self, seq):
            from biosuite.core.sequence import sequence_stats
            return sequence_stats(seq)


    class AlignmentViewer:
        """Interactive pairwise alignment widget."""

        def __init__(self):
            self.output = widgets.Output()

        def show(self):
            seq1_input = widgets.Textarea(
                placeholder='Sequence 1', description='Seq 1:',
                layout=widgets.Layout(width='100%', height='60px')
            )
            seq2_input = widgets.Textarea(
                placeholder='Sequence 2', description='Seq 2:',
                layout=widgets.Layout(width='100%', height='60px')
            )

            nw_btn = widgets.Button(description='Needleman-Wunsch', button_style='success')
            sw_btn = widgets.Button(description='Smith-Waterman', button_style='info')

            output_area = widgets.Output()

            def on_nw_click(b):
                from biosuite.core.alignment import needleman_wunsch
                with output_area:
                    clear_output()
                    s1 = seq1_input.value.upper().strip()
                    s2 = seq2_input.value.upper().strip()
                    if s1 and s2:
                        a1, a2, score = needleman_wunsch(s1, s2)
                        print(f"Score: {score}")
                        print(f"Seq1: {a1}")
                        print(f"Seq2: {a2}")
                    else:
                        print("Please enter both sequences")

            def on_sw_click(b):
                from biosuite.core.alignment import smith_waterman
                with output_area:
                    clear_output()
                    s1 = seq1_input.value.upper().strip()
                    s2 = seq2_input.value.upper().strip()
                    if s1 and s2:
                        a1, a2, score = smith_waterman(s1, s2)
                        print(f"Score: {score}")
                        print(f"Seq1: {a1}")
                        print(f"Seq2: {a2}")
                    else:
                        print("Please enter both sequences")

            nw_btn.on_click(on_nw_click)
            sw_btn.on_click(on_sw_click)

            display(widgets.VBox([
                widgets.HTML("<h3>BioSuite Alignment Viewer</h3>"),
                seq1_input,
                seq2_input,
                widgets.HBox([nw_btn, sw_btn]),
                output_area
            ]))


    class PlotExplorer:
        """Interactive plot explorer widget."""

        def __init__(self):
            self.output = widgets.Output()

        def show(self):
            plot_type = widgets.Dropdown(
                options=['Volcano', 'PCA', 'Scatter', 'Heatmap', 'Boxplot'],
                description='Plot type:'
            )

            interactive_cb = widgets.Checkbox(
                value=False, description='Interactive (Plotly)',
                indent=False
            )

            run_btn = widgets.Button(description='Generate Plot', button_style='success')
            output_area = widgets.Output()

            def on_run_click(b):
                import numpy as np
                with output_area:
                    clear_output()
                    plot = plot_type.value
                    interactive = interactive_cb.value
                    rng = np.random.default_rng(42)

                    if plot == 'Volcano':
                        from biosuite.plotting.plot_api import volcano
                        fc = rng.normal(0, 1.5, 500)
                        pvals = rng.uniform(0, 1, 500)
                        pvals[:30] = rng.uniform(1e-6, 0.05, 30)
                        fig = volcano(fc, pvals, interactive=interactive)
                        display(fig)

                    elif plot == 'PCA':
                        from biosuite.plotting.plot_api import pca
                        data = rng.standard_normal((30, 50))
                        groups = ['Ctrl']*15 + ['Treat']*15
                        fig = pca(data, labels=groups, interactive=interactive)
                        display(fig)

                    elif plot == 'Scatter':
                        from biosuite.plotting.plot_api import scatter
                        x = rng.standard_normal(100)
                        y = x * 2 + rng.standard_normal(100) * 0.5
                        fig = scatter(x, y, interactive=interactive)
                        display(fig)

                    elif plot == 'Heatmap':
                        from biosuite.plotting.plot_api import heatmap
                        data = rng.standard_normal((10, 8))
                        fig = heatmap(data, interactive=interactive)
                        display(fig)

                    elif plot == 'Boxplot':
                        from biosuite.plotting.plot_api import boxplot
                        data = {
                            'Ctrl': rng.standard_normal(30).tolist(),
                            'Treat': rng.standard_normal(30) + 1
                        }
                        fig = boxplot(data, interactive=interactive)
                        display(fig)

                    else:
                        # Reached only if the option list and these branches drift
                        # apart; drawing nothing silently is how that stays hidden.
                        print(f"Cannot draw plot type {plot!r}")

            run_btn.on_click(on_run_click)

            display(widgets.VBox([
                widgets.HTML("<h3>BioSuite Plot Explorer</h3>"),
                widgets.HBox([plot_type, interactive_cb]),
                run_btn,
                output_area
            ]))


# ── Convenience functions ────────────────────────────────────────────────────

def quick_gc(sequence):
    """Quick GC content calculation for Jupyter."""
    from biosuite.core.sequence import gc_content
    return gc_content(sequence)

def quick_translate(sequence):
    """Quick translation for Jupyter."""
    from biosuite.core.sequence import translate
    return translate(sequence)

def quick_align(seq1, seq2, method='nw'):
    """Quick alignment for Jupyter. method='nw' or 'sw'."""
    from biosuite.core.alignment import needleman_wunsch, smith_waterman
    if method == 'nw':
        return needleman_wunsch(seq1, seq2)
    if method == 'sw':
        return smith_waterman(seq1, seq2)
    raise ValueError(f"Unknown alignment method {method!r}: use 'nw' or 'sw'.")

def quick_blast(query_file, database_file):
    """Quick BLAST search for Jupyter."""
    from biosuite.core.blast import run_blast
    return run_blast(query_file, database_file)
