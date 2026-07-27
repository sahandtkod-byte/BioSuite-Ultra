"""
IPython/Jupyter magics for BioSuite Ultra.

Provides %biosuite, %biostats, and %bioimport magic commands.
Load with: %load_ext biosuite.notebook.magics
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from IPython.core.magic import Magics, magics_class, line_magic
    from IPython.core.magic_arguments import argument, magic_arguments
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


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
            """
            parts = line.strip().split()
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
            from IPython.display import display
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
                print(f"Columns: {list(obj.columns[:10])}...")
                print(f"\nFirst 5 rows:")
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

            modules = {
                'sequence': ['gc_content', 'reverse_complement', 'translate',
                            'read_fasta', 'read_fastq', 'sequence_stats'],
                'alignment': ['needleman_wunsch', 'smith_waterman'],
                'phylogeny': ['distance_matrix', 'upgma_tree'],
                'expression': ['differential_expression', 'cpm_normalization', 'tpm_normalization'],
                'blast': ['run_blast'],
                'plotting': ['volcano', 'pca', 'manhattan', 'heatmap', 'scatter'],
            }

            target = line.strip().lower()
            if target == 'all':
                for mod_name, funcs in modules.items():
                    try:
                        mod = __import__(f'biosuite.core.{mod_name}', fromlist=funcs)
                        for func_name in funcs:
                            if hasattr(mod, func_name):
                                ip.user_ns[func_name] = getattr(mod, func_name)
                        print(f"  Loaded {mod_name}: {', '.join(funcs)}")
                    except ImportError:
                        print(f"  Skipping {mod_name} (dependencies missing)")
                try:
                    from biosuite.plotting.plot_api import (volcano, pca, manhattan,
                                                             heatmap, scatter, boxplot)
                    ip.user_ns['volcano'] = volcano
                    ip.user_ns['pca'] = pca
                    ip.user_ns['manhattan'] = manhattan
                    ip.user_ns['heatmap'] = heatmap
                    ip.user_ns['scatter'] = scatter
                    ip.user_ns['boxplot'] = boxplot
                    print(f"  Loaded plotting: volcano, pca, manhattan, heatmap, scatter, boxplot")
                except ImportError:
                    pass
                print("\nAll functions loaded into namespace!")
            elif target in modules:
                funcs = modules[target]
                try:
                    mod = __import__(f'biosuite.core.{target}', fromlist=funcs)
                    for func_name in funcs:
                        if hasattr(mod, func_name):
                            ip.user_ns[func_name] = getattr(mod, func_name)
                    print(f"Loaded {', '.join(funcs)} from biosuite.core.{target}")
                except ImportError as e:
                    print(f"Error importing {target}: {e}")
            else:
                print(f"Available modules: {', '.join(modules.keys())}, all")


def load_ipython_extension(ipython):
    """Register BioSuite magic commands."""
    if HAS_IPYTHON:
        ipython.register_magics(BioSuiteMagics)
        print("BioSuite magic commands loaded!")
        print("  %biosuite help - Show available commands")
        print("  %bioimport all - Import all functions")
    else:
        print("IPython not available. Magic commands disabled.")


# Re-export for ``from biosuite.notebook.magics import BioSuiteMagics``
__all__ = ["BioSuiteMagics", "load_ipython_extension"]
