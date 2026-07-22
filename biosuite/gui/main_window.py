"""
BioSuite Ultra — Main GUI Application Window.
Slim orchestrator that composes tab mixins for each analysis domain.
"""
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import os
import builtins
import re
import threading

from ..core.utils import config, set_theme, save_config, load_dataframe_safe, center_window
from ..core.sequence import (read_fasta, read_fastq, read_genbank,
                              sequence_stats, reverse_complement, translate, gc_content)
from ..core.alignment import needleman_wunsch, smith_waterman
from ..core.phylogeny import distance_matrix, upgma_tree, plot_phylogenetic_tree
from ..core.ngs import read_vcf, manhattan_from_vcf
from ..core.expression import read_counts_matrix, cpm_normalization, differential_expression

from .themes import (THEMES, PLOT_CATEGORIES, PLOT_FUNCS, FONT_FAMILY, FONT_MONO,
                      FONT_TITLE, FONT_HEADING, FONT_SUBHEADING, FONT_BODY,
                      FONT_SMALL, FONT_SIDEBAR, FONT_CODE, FONT_BUTTON)
from .dialogs import (BioMessageDialog, BioConfirmDialog, BioInputDialog,
                       BioFilePickerDialog, BioDropdownDialog, BioSplashScreen)

# Tab mixins
from .tabs.visualization import VisualizationTabMixin
from .tabs.sequence_analysis import SequenceAnalysisTabMixin
from .tabs.transcriptomics import TranscriptomicsTabMixin
from .tabs.genomics import GenomicsTabMixin
from .tabs.advanced import AdvancedTabMixin
from .tabs.databases import DatabasesTabMixin
from .tabs.workflow import WorkflowTabMixin
from .tabs.help import HelpTabMixin
from .tabs.cloning import CloningTabMixin
from .tabs.survival import SurvivalTabMixin
from .tabs.metabolomics import MetabolomicsTabMixin

# Heavy imports deferred to _finish_startup for faster GUI launch
pd = None
np = None
plt = None


class BioSuiteApp(
    VisualizationTabMixin,
    SequenceAnalysisTabMixin,
    TranscriptomicsTabMixin,
    GenomicsTabMixin,
    AdvancedTabMixin,
    DatabasesTabMixin,
    WorkflowTabMixin,
    HelpTabMixin,
    CloningTabMixin,
    SurvivalTabMixin,
    MetabolomicsTabMixin,
    ctk.CTk,
):
    def __init__(self):
        super().__init__()

        saved_theme = config.get('theme', 'dark-green')
        if saved_theme not in THEMES:
            saved_theme = 'dark-green'
        self.current_theme_key = saved_theme
        self.T = THEMES[self.current_theme_key]

        self.title("BioSuite Ultra  ·  Bioinformatic Platform")
        ctk.set_appearance_mode(self.T['ctk_mode'])
        ctk.set_default_color_theme("blue")

        self.withdraw()
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w = min(1400, int(sw * 0.82))
        h = min(920, int(sh * 0.85))
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(1050, 700)
        self.configure(fg_color=self.T['bg'])

        self._splash = BioSplashScreen(self, self.T)
        self.after(100, self._build_with_splash)

    def _build_with_splash(self):
        steps = [
            ("Loading sequence engine...", 0.20),
            ("Loading alignment module...", 0.40),
            ("Loading expression analysis...", 0.60),
            ("Loading plot renderers...", 0.80),
            ("Building interface...", 0.95),
        ]
        def run_steps(i):
            if i < len(steps):
                text, prog = steps[i]
                self._splash.update_status(text, prog)
                self.after(60, lambda: run_steps(i + 1))
            else:
                self._splash.update_status("Ready.", 1.0)
                self.after(200, self._finish_startup)
        run_steps(0)

    def _finish_startup(self):
        global pd, np, plt
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        self._build_plot_funcs()
        self._build_sidebar()
        self._build_content()
        self._show_frame('plots')
        self._apply_plot_search()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind('<Configure>', self._on_resize)
        self.bind('<Control-q>', lambda e: self._on_close())
        self.bind('<Control-Q>', lambda e: self._on_close())
        self.bind('<Control-s>', lambda e: self._save_current())
        self.bind('<Control-S>', lambda e: self._save_current())
        self.bind('<F1>', lambda e: self._show_frame('help'))
        self.bind('<F5>', lambda e: self._refresh_current())
        self.bind('<Escape>', lambda e: self._show_frame('plots'))
        self._plot_history = []
        self._plot_history_index = -1
        self._progress_bar = None
        self._splash.animate_out()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w = min(1400, int(sw * 0.82))
        h = min(920, int(sh * 0.85))
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.after(300, self._show_main_window)

    def _show_main_window(self):
        self.deiconify()
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w = min(1400, int(sw * 0.82))
        h = min(920, int(sh * 0.85))
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.lift()
        self.focus_force()

    def _build_plot_funcs(self):
        from ..plotting.biological_plots import (volcano_plot, pca_plot, manhattan_plot, ma_plot,
            venn_diagram, barplot_custom, boxplot_custom, heatmap_custom, scatter_custom,
            timeseries_plot, qq_plot, clustered_heatmap, circos_plot, alignment_viewer,
            violin_plot, raincloud_plot, ridge_plot, dot_plot, export_all_to_folder,
            generate_markdown_story, batch_export_to_pdf)
        from ..plotting.math_plots import (sine_plot, cosine_plot, linear_plot,
            quadratic_plot, cubic_plot, exponential_plot, logistic_plot)
        from ..plotting.specialized_plots import gsea_plot, motif_logo, sankey_diagram, umap_plot
        self._export_all_to_folder = export_all_to_folder
        self._batch_export_to_pdf = batch_export_to_pdf
        self._generate_markdown_story = generate_markdown_story
        PLOT_FUNCS.update({
            'volcano': volcano_plot, 'pca': pca_plot, 'manhattan': manhattan_plot,
            'ma': ma_plot, 'venn': venn_diagram, 'barplot': barplot_custom,
            'boxplot': boxplot_custom, 'heatmap': heatmap_custom, 'scatter': scatter_custom,
            'timeseries': timeseries_plot, 'sine': sine_plot, 'cosine': cosine_plot,
            'linear': linear_plot, 'quadratic': quadratic_plot, 'cubic': cubic_plot,
            'exponential': exponential_plot, 'logistic': logistic_plot, 'gsea': gsea_plot,
            'motif': motif_logo, 'sankey': sankey_diagram, 'qq': qq_plot,
            'clustered_heatmap': clustered_heatmap, 'circos': circos_plot,
            'alignment': alignment_viewer, 'umap': umap_plot,
            'violin': violin_plot, 'raincloud': raincloud_plot, 'ridge': ridge_plot,
            'dotplot': dot_plot,
            'upset': self._gui_upset, 'genome_browser': self._gui_genome_browser,
            'seq_logo': self._gui_seq_logo, 'conservation_bar': self._gui_conservation_bar,
            'interactive_scatter': self._gui_interactive_scatter,
            'interactive_bar': self._gui_interactive_bar,
            'interactive_heatmap': self._gui_interactive_heatmap,
            'interactive_volcano': self._gui_interactive_volcano,
            'interactive_line': self._gui_interactive_line,
            'interactive_pie': self._gui_interactive_pie,
            'synteny': self._gui_synteny,
        })

    # ─── Themed Dialog Wrappers ───────────────────────────────────────────────

    def _msg_info(self, title, message):
        BioMessageDialog(self, self.T, title=title, message=message, msg_type='info')

    def _msg_warning(self, title, message):
        BioMessageDialog(self, self.T, title=title, message=message, msg_type='warning')

    def _msg_error(self, title, message):
        BioMessageDialog(self, self.T, title=title, message=message, msg_type='error')

    def _msg_success(self, title, message):
        BioMessageDialog(self, self.T, title=title, message=message, msg_type='success')

    def _confirm(self, title, message):
        d = BioConfirmDialog(self, self.T, title=title, message=message)
        self.wait_window(d)
        return d.result is True

    def _ask_input(self, title, prompt, default=""):
        d = BioInputDialog(self, self.T, title=title, prompt=prompt, default=default)
        self.wait_window(d)
        return d.result

    def _ask_dropdown(self, title, prompt, options, default=None):
        d = BioDropdownDialog(self, self.T, title=title, prompt=prompt,
                               options=options, default=default)
        self.wait_window(d)
        return d.result

    def _ask_file(self, title, prompt, filetypes=None):
        d = BioFilePickerDialog(self, self.T, title=title, prompt=prompt,
                                 filetypes=filetypes)
        self.wait_window(d)
        return d.result

    # ─── Theme Helpers ────────────────────────────────────────────────────────

    def _apply_theme(self, theme_key):
        if theme_key not in THEMES:
            return
        self.current_theme_key = theme_key
        self.T = THEMES[theme_key]
        config['theme'] = theme_key
        save_config(config)
        set_theme('dark' if self.T['ctk_mode'] == 'dark' else 'light')
        ctk.set_appearance_mode(self.T['ctk_mode'])
        self.configure(fg_color=self.T['bg'])
        self._rebuild_ui()

    def _rebuild_ui(self):
        for widget in self.winfo_children():
            widget.destroy()
        self.all_cards = []
        self._progress_bar = None
        self._build_sidebar()
        self._build_content()
        self._show_frame(self._current_frame if hasattr(self, '_current_frame') else 'plots')
        self._apply_plot_search()

    def _on_close(self):
        if plt is not None:
            plt.close('all')
        self.destroy()

    def _show_plot_from_figure(self, fig, title="Plot"):
        """Display a matplotlib figure in a popup window with Save As option."""
        import tempfile, os
        from tkinter import filedialog
        from PIL import Image, ImageTk
        
        # Save figure to temp file
        tmp = os.path.join(tempfile.gettempdir(), f"biosuite_{id(fig)}.png")
        fig.savefig(tmp, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        
        # Create window
        win = ctk.CTkToplevel(self)
        win.title(title)
        win.geometry("950x750")
        win.configure(fg_color=self.T.get('bg', '#0a0f0a'))
        photo_ref = [None]
        
        try:
            img = Image.open(tmp)
            img.thumbnail((900, 650), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            photo_ref[0] = photo
            label = ctk.CTkLabel(win, image=photo, text="")
            label.pack(fill='both', expand=True, padx=10, pady=10)
        except Exception as e:
            ctk.CTkLabel(win, text=f"Error: {e}").pack(pady=20)
        
        # Bottom buttons
        btn_frame = ctk.CTkFrame(win, fg_color='transparent')
        btn_frame.pack(fill='x', padx=10, pady=5)
        
        def save_as():
            ext = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All", "*.*")],
                title="Save Plot As"
            )
            if ext:
                import shutil
                shutil.copy2(tmp, ext)
                self._set_status(f"Plot saved to: {ext}")
        
        def on_close():
            photo_ref[0] = None
            try: os.unlink(tmp)
            except: pass
            win.destroy()
        
        ctk.CTkButton(btn_frame, text="💾 Save As...", command=save_as,
                      fg_color=self.T.get('accent', '#00ff88'),
                      text_color='#000000', width=120).pack(side='left', padx=5)
        ctk.CTkButton(btn_frame, text="Close", command=on_close,
                      fg_color='#ff4444', text_color='#ffffff', width=80).pack(side='right', padx=5)
        
        win.protocol("WM_DELETE_WINDOW", on_close)

    def _on_resize(self, event=None):
        if event and event.widget == self:
            self.update_idletasks()

    # ─── Sidebar ──────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        T = self.T
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=T['sidebar_bg'])
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        self.sidebar_scroll = ctk.CTkScrollableFrame(self.sidebar, fg_color='transparent',
                                                      corner_radius=0,
                                                      scrollbar_button_color=T['scrollbar'],
                                                      scrollbar_button_hover_color=T['border_light'])
        self.sidebar_scroll.pack(fill='both', expand=True)

        logo_area = ctk.CTkFrame(self.sidebar_scroll, fg_color='transparent')
        logo_area.pack(fill='x', padx=12, pady=(12, 4))
        ctk.CTkLabel(logo_area, text="BIOSUITE", font=(FONT_FAMILY, 16, 'bold'),
                      text_color=T['accent']).pack(anchor='w')
        ctk.CTkLabel(logo_area, text="v4.2.2", font=FONT_SMALL,
                      text_color=T['text_dim']).pack(anchor='w', pady=(2, 0))

        ctk.CTkFrame(self.sidebar_scroll, height=1, fg_color=T['border']).pack(fill='x', padx=10, pady=(8, 6))

        self.sidebar_buttons = {}

        def _sidebar_category(title):
            ctk.CTkLabel(self.sidebar_scroll, text=title, font=(FONT_FAMILY, 8, 'bold'),
                        text_color=T['text_muted'], anchor='w').pack(fill='x', padx=10, pady=(6, 1))

        def _sidebar_item(key, label):
            btn = ctk.CTkButton(self.sidebar_scroll, text=f"  {label}", anchor='w',
                                font=(FONT_FAMILY, 11), height=28, corner_radius=6,
                                fg_color='transparent', text_color=T['sidebar_text'],
                                hover_color=T['sidebar_hover'],
                                command=lambda k=key: self._show_frame(k))
            btn.pack(fill='x', padx=8, pady=1)
            self.sidebar_buttons[key] = btn

        _sidebar_category("VISUALIZATION")
        _sidebar_item('plots', '\U0001F4CA Plots Gallery')

        _sidebar_category("SEQUENCE & ALIGNMENT")
        _sidebar_item('sequence', '\U0001F9EC Sequence Analysis')
        _sidebar_item('alignment', '\U0001F504 Alignments')
        _sidebar_item('phylogeny', '\U0001F333 Phylogenetics')

        _sidebar_category("TRANSCRIPTOMICS")
        _sidebar_item('expression', '\U0001F4C8 Expression Analysis')
        _sidebar_item('trimming', '\u2702 Read Trimming')
        _sidebar_item('quant', '\U0001F4CF RNA-seq Quantification')

        _sidebar_category("GENOMICS & NGS")
        _sidebar_item('ngs', '\U0001F500 Variant Calling / VCF')
        _sidebar_item('assembly', '\U0001F9EA Genome Assembly')

        _sidebar_category("SINGLE-CELL & PROTEINS")
        _sidebar_item('singlecell', '\U0001F9EA Single-Cell RNA-seq')
        _sidebar_item('structure', '\U0001F4A0 Protein Structure')

        _sidebar_category("SPECIALIZED")
        _sidebar_item('metagenomics', '\U0001F30D Metagenomics')
        _sidebar_item('crispr', '\U0001F52E CRISPR Design')
        _sidebar_item('popgen', '\U0001F3AF Population Genetics')
        _sidebar_item('ml', '\U0001F916 Machine Learning')

        _sidebar_category("SEQUENCE TOOLS")
        _sidebar_item('orftools', '\U0001F9EC ORF / Primers / Enzymes')
        _sidebar_item('databases', '\U0001F50D Database Search')
        _sidebar_item('fileformats', '\U0001F4C4 File Formats')

        _sidebar_category("ADVANCED VISUALIZATION")
        _sidebar_item('upset', '\U0001F4CA UpSet Plots')
        _sidebar_item('genomebrowser', '\U0001F4DC Genome Browser')
        _sidebar_item('conservation', '\U0001F3B5 Sequence Logos')
        _sidebar_item('syntenytabs', '\U0001F5FA Synteny Analysis')
        _sidebar_item('interactive', '\U0001F5B1 Interactive Plots')

        _sidebar_category("WORKFLOW & DOMAIN")
        _sidebar_item('pipeline', '\U0001F504 Pipeline Builder')
        _sidebar_item('batch', '\U0001F4E6 Batch Processor')
        _sidebar_item('gobrowser', '\U0001F3DB GO Browser')
        _sidebar_item('pathway', '\U0001F9EC Pathway Visualization')
        _sidebar_item('gwas', '\U0001F3AF GWAS Analysis')
        _sidebar_item('epitope', '\U0001F9EC Epitope Prediction')

        _sidebar_category("GENOMICS TOOLS")
        _sidebar_item('16srna', '\U0001F9EA 16S rRNA Pipeline')
        _sidebar_item('svcinv', '\U0001F500 SV / CNV Detection')
        _sidebar_item('bigwig', '\U0001F4CA BigWig Reader')

        _sidebar_category("MOLECULAR CLONING")
        _sidebar_item('cloning', '\U0001F9EA Cloning Toolkit')

        _sidebar_category("ANALYSIS")
        _sidebar_item('survival', '\U0001F4C8 Survival Analysis')
        _sidebar_item('metabolomics', '\U0001F9EA Metabolomics')

        _sidebar_category("HELP & SETTINGS")
        _sidebar_item('apikey', 'API Keys Config')
        _sidebar_item('help', 'Help & Guides')

        ctk.CTkFrame(self.sidebar_scroll, height=1, fg_color=T['border']).pack(fill='x', padx=10, pady=(8, 4))

        ctk.CTkLabel(self.sidebar_scroll, text="THEME", font=(FONT_FAMILY, 8, 'bold'),
                      text_color=T['text_muted']).pack(anchor='w', padx=10, pady=(0, 4))

        self.theme_buttons = {}
        for tkey, tlabel in [('dark-green', 'Green Cyber'), ('dark-purple', 'Purple Cyber'), ('light-blue', 'Light Blue')]:
            is_active = tkey == self.current_theme_key
            btn = ctk.CTkButton(self.sidebar_scroll, text=f"  {tlabel}", anchor='w',
                                font=(FONT_FAMILY, 11), height=28, corner_radius=6,
                                fg_color=T['sidebar_active'] if is_active else 'transparent',
                                text_color=T['sidebar_active_text'] if is_active else T['sidebar_text'],
                                hover_color=T['sidebar_hover'],
                                command=lambda k=tkey: self._apply_theme(k))
            btn.pack(fill='x', padx=8, pady=1)
            self.theme_buttons[tkey] = btn

        version_frame = ctk.CTkFrame(self.sidebar, fg_color='transparent')
        version_frame.pack(side='bottom', fill='x', padx=18, pady=(0, 16))
        ctk.CTkLabel(version_frame, text="v4.2.2", font=(FONT_FAMILY, 9),
                      text_color=T['text_muted']).pack(anchor='w')
        self._current_frame = 'plots'

    # ─── Content Area ─────────────────────────────────────────────────────────

    def _build_content(self):
        T = self.T
        self.content = ctk.CTkFrame(self, fg_color=T['bg'], corner_radius=0)
        self.content.pack(side='right', fill='both', expand=True)
        self.frames = {}
        self.all_cards = []
        self._build_plot_frame()
        self._build_sequence_frame()
        self._build_alignment_frame()
        self._build_phylogeny_frame()
        self._build_expression_frame()
        self._build_ngs_frame()
        self._build_trimming_frame()
        self._build_quant_frame()
        self._build_singlecell_frame()
        self._build_structure_frame()
        self._build_assembly_frame()
        self._build_metagenomics_frame()
        self._build_crispr_frame()
        self._build_popgen_frame()
        self._build_ml_frame()
        self._build_orftools_frame()
        self._build_databases_frame()
        self._build_fileformats_frame()
        self._build_apikey_frame()
        self._build_help_frame()
        self._build_upset_frame()
        self._build_genomebrowser_frame()
        self._build_conservation_frame()
        self._build_synteny_frame()
        self._build_interactive_frame()
        self._build_pipeline_frame()
        self._build_batch_frame()
        self._build_gobrowser_frame()
        self._build_pathway_frame()
        self._build_gwas_frame()
        self._build_epitope_frame()
        self._build_16srna_frame()
        self._build_svcnv_frame()
        self._build_bigwig_frame()
        self._build_cloning_frame()
        self._build_survival_frame()
        self._build_metabolomics_frame()
        self.status_bar = ctk.CTkLabel(self, text="  Ready", anchor='w', font=FONT_SMALL,
                                        fg_color=T['card'], text_color=T['text_dim'], height=30)
        self.status_bar.pack(side='bottom', fill='x')
        self._set_status("BioSuite Ultra v4.2.2 loaded successfully")

    def _set_status(self, text):
        self.status_bar.configure(text=f"  {text}")

    # ─── Keyboard Shortcuts ─────────────────────────────────────────────────

    def _save_current(self):
        self._set_status("Save triggered (Ctrl+S)")

    def _refresh_current(self):
        key = self._current_frame
        self._show_frame(key)
        self._set_status(f"Refreshed: {key}")

    # ─── Progress Bar ───────────────────────────────────────────────────────

    def _show_progress(self, text="Working..."):
        if self._progress_bar is None:
            self._progress_bar = ctk.CTkProgressBar(self, height=3, corner_radius=1,
                                                      fg_color=self.T['border'],
                                                      progress_color=self.T['accent'])
            self._progress_bar.pack(side='bottom', fill='x', before=self.status_bar)
        self._progress_bar.set(0)
        self._progress_bar.lift()
        self._set_status(text)

    def _update_progress(self, value):
        if self._progress_bar:
            self._progress_bar.set(min(1.0, max(0.0, value)))

    def _hide_progress(self):
        if self._progress_bar:
            self._progress_bar.pack_forget()
            self._progress_bar = None
        self._set_status("Ready")

    # ─── Plot History ───────────────────────────────────────────────────────

    def _record_plot(self, fig, name="plot"):
        if not hasattr(self, '_plot_history'):
            self._plot_history = []
        self._plot_history.append({"fig": fig, "name": name})
        if len(self._plot_history) > 10:
            old = self._plot_history.pop(0)
            import matplotlib.pyplot as plt
            plt.close(old["fig"])
        self._plot_history_index = len(self._plot_history) - 1

    def _show_plot_history(self):
        if not self._plot_history:
            self._msg_info("Plot History", "No plots generated yet.")
            return
        items = [f"{i+1}. {p['name']}" for i, p in enumerate(self._plot_history)]
        self._msg_info("Plot History", "\n".join(items))

    # ─── Drag-and-Drop Support ──────────────────────────────────────────────

    def _setup_drag_drop(self, widget):
        try:
            import tkinterdnd2
            widget.drop_target_register(tkinterdnd2.DND_FILES)
            widget.dnd_bind('<<Drop>>', lambda e: self._on_drop(e, widget))
        except ImportError:
            pass

    def _on_drop(self, event, widget):
        files = self.tk.splitlist(event.data)
        if files:
            widget.delete("1.0", "end")
            widget.insert("end", files[0])
            self._set_status(f"Loaded: {os.path.basename(files[0])}")

    def _show_frame(self, key):
        for f in self.frames.values():
            f.pack_forget()
        self.frames[key].pack(in_=self.content, fill='both', expand=True, padx=16, pady=16)
        T = self.T
        for k, btn in self.sidebar_buttons.items():
            if k == key:
                btn.configure(fg_color=T['sidebar_active'], text_color=T['sidebar_active_text'])
            else:
                btn.configure(fg_color='transparent', text_color=T['sidebar_text'])
        self._current_frame = key

    # ─── UI Helpers ───────────────────────────────────────────────────────────

    def _card(self, parent, **kwargs):
        T = self.T
        d = dict(fg_color=T['card'], corner_radius=12, border_width=1, border_color=T['border'])
        d.update(kwargs)
        c = ctk.CTkFrame(parent, **d)
        self.all_cards.append(c)
        return c

    def _section_header(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=FONT_HEADING,
                      text_color=self.T['text']).pack(anchor='w', padx=4, pady=(4, 8))

    def _action_button(self, parent, text, command, color_key='accent'):
        T = self.T
        color = T.get(color_key, T['accent'])
        hover = T.get(f'{color_key}_dim', color)
        return ctk.CTkButton(parent, text=text, height=36, corner_radius=8,
                              font=FONT_BUTTON, fg_color=color, hover_color=hover,
                              text_color='#000000' if color_key == 'accent' else '#ffffff',
                              command=command)

    def _input_entry(self, parent, placeholder, **kwargs):
        T = self.T
        d = dict(height=36, font=(FONT_FAMILY, 12), corner_radius=8, fg_color=T['input_bg'],
                 border_color=T['border'], text_color=T['text'],
                 placeholder_text=placeholder, placeholder_text_color=T['text_muted'])
        d.update(kwargs)
        return ctk.CTkEntry(parent, **d)

    def _text_box(self, parent, height=200, **kwargs):
        T = self.T
        d = dict(height=height, font=('Consolas', 11), corner_radius=8, fg_color=T['input_bg'],
                 border_color=T['border'], text_color=T['text'], border_width=1)
        d.update(kwargs)
        return ctk.CTkTextbox(parent, **d)

    def _label(self, parent, text, style='body'):
        T = self.T
        fonts = {'title': FONT_HEADING, 'sub': (FONT_FAMILY, 13, 'bold'), 'body': (FONT_FAMILY, 12),
                 'small': FONT_SMALL, 'dim': FONT_SMALL}
        colors = {'title': T['text'], 'sub': T['text'], 'body': T['text'],
                   'small': T['text_dim'], 'dim': T['text_muted']}
        return ctk.CTkLabel(parent, text=text, font=fonts.get(style, (FONT_FAMILY, 12)),
                             text_color=colors.get(style, T['text']))

    # ─── Plotly Integration ──────────────────────────────────────────────────

    def _show_interactive_plot(self, fig, title="Interactive Plot"):
        """Display a Plotly figure in the GUI.

        Saves as HTML and opens in default browser, or displays in webview if available.
        """
        try:
            import plotly.io as pio
            import tempfile
            import webbrowser

            # Save to temporary HTML file
            html_path = os.path.join(tempfile.gettempdir(), f"biosuite_plot_{id(fig)}.html")
            fig.write_html(html_path, auto_open=False)

            # Try to open in browser
            webbrowser.open(f"file://{html_path}")
            self._set_status(f"Interactive plot opened in browser: {title}")

        except ImportError:
            self._msg_warning("Plotly Not Available",
                            "Install plotly for interactive plots: pip install plotly")
        except Exception as e:
            self._msg_error("Error", f"Failed to display interactive plot: {e}")

    def _save_interactive_plot(self, fig, default_name="interactive_plot"):
        """Save a Plotly figure as HTML."""
        try:
            import plotly.io as pio
            from tkinter import filedialog

            filepath = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
                initialfile=f"{default_name}.html"
            )
            if filepath:
                fig.write_html(filepath)
                self._set_status(f"Saved: {filepath}")
                self._msg_success("Saved", f"Interactive plot saved to:\n{filepath}")
        except Exception as e:
            self._msg_error("Error", f"Failed to save: {e}")

    def _gui_interactive_plot_api(self):
        """Launch the interactive plot explorer."""
        try:
            from biosuite.plotting.plot_api import (volcano, pca, manhattan, heatmap,
                                                       scatter, boxplot, violin, qqplot)
            import numpy as np

            # Demo with sample data
            np.random.seed(42)
            fc = np.random.normal(0, 1.5, 500)
            pvals = np.random.uniform(0, 1, 500)
            pvals[:30] = np.random.uniform(1e-6, 0.05, 30)

            fig = volcano(fc, pvals, interactive=True, title="Demo Volcano Plot")
            self._show_interactive_plot(fig, "Volcano Plot")
        except Exception as e:
            self._msg_error("Error", str(e))
