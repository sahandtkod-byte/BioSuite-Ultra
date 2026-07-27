"""
Interactive Jupyter widgets for BioSuite Ultra.

Provides SequenceAnalyzer, AlignmentViewer, and PlotExplorer widgets.
Load with: from biosuite.notebook.widgets import SequenceAnalyzer
"""
from __future__ import annotations

from typing import Any

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False


class SequenceAnalyzer:
    """Interactive sequence analyzer widget for Jupyter."""

    def __init__(self):
        if not HAS_WIDGETS:
            raise ImportError("ipywidgets required. Install with: pip install ipywidgets")
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
        if not HAS_WIDGETS:
            raise ImportError("ipywidgets required. Install with: pip install ipywidgets")
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
        if not HAS_WIDGETS:
            raise ImportError("ipywidgets required. Install with: pip install ipywidgets")
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

                if plot == 'Volcano':
                    from biosuite.plotting.plot_api import volcano
                    np.random.seed(42)
                    fc = np.random.normal(0, 1.5, 500)
                    pvals = np.random.uniform(0, 1, 500)
                    pvals[:30] = np.random.uniform(1e-6, 0.05, 30)
                    fig = volcano(fc, pvals, interactive=interactive)
                    display(fig)

                elif plot == 'PCA':
                    from biosuite.plotting.plot_api import pca
                    np.random.seed(42)
                    data = np.random.randn(30, 50)
                    groups = ['Ctrl']*15 + ['Treat']*15
                    fig = pca(data, labels=groups, interactive=interactive)
                    display(fig)

                elif plot == 'Scatter':
                    from biosuite.plotting.plot_api import scatter
                    np.random.seed(42)
                    x = np.random.randn(100)
                    y = x * 2 + np.random.randn(100) * 0.5
                    fig = scatter(x, y, interactive=interactive)
                    display(fig)

                elif plot == 'Heatmap':
                    from biosuite.plotting.plot_api import heatmap
                    np.random.seed(42)
                    data = np.random.randn(10, 8)
                    fig = heatmap(data, interactive=interactive)
                    display(fig)

                elif plot == 'Boxplot':
                    from biosuite.plotting.plot_api import boxplot
                    np.random.seed(42)
                    data = {
                        'Ctrl': np.random.randn(30).tolist(),
                        'Treat': (np.random.randn(30) + 1).tolist()
                    }
                    fig = boxplot(data, interactive=interactive)
                    display(fig)

        run_btn.on_click(on_run_click)

        display(widgets.VBox([
            widgets.HTML("<h3>BioSuite Plot Explorer</h3>"),
            widgets.HBox([plot_type, interactive_cb]),
            run_btn,
            output_area
        ]))


# ── Convenience functions ──────────────────────────────────────────────

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
    else:
        return smith_waterman(seq1, seq2)

def quick_blast(query_file, database_file):
    """Quick BLAST search for Jupyter."""
    from biosuite.core.blast import run_blast
    return run_blast(query_file, database_file)


__all__ = [
    "SequenceAnalyzer", "AlignmentViewer", "PlotExplorer",
    "quick_gc", "quick_translate", "quick_align", "quick_blast",
]
