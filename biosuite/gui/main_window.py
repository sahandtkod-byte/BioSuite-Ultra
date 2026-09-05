"""
BioSuite Ultra — Main GUI Application Window.
Slim orchestrator that composes tab mixins for each analysis domain.
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

from .. import __version__
from ..core.utils import config, save_config, set_theme
from .dialogs import (
    BioConfirmDialog,
    BioDropdownDialog,
    BioFilePickerDialog,
    BioInputDialog,
    BioMessageDialog,
    BioSplashScreen,
)
from .tabs.advanced import AdvancedTabMixin
from .tabs.cloning import CloningTabMixin
from .tabs.databases import DatabasesTabMixin
from .tabs.genomics import GenomicsTabMixin
from .tabs.help import HelpTabMixin
from .tabs.metabolomics import MetabolomicsTabMixin
from .tabs.sequence_analysis import SequenceAnalysisTabMixin
from .tabs.survival import SurvivalTabMixin
from .tabs.transcriptomics import TranscriptomicsTabMixin

# Tab mixins
from .tabs.visualization import VisualizationTabMixin
from .tabs.workflow import WorkflowTabMixin
from .themes import FONT_BODY as FONT_BODY  # noqa: F401
from .themes import (
    FONT_BUTTON,
    FONT_FAMILY,
    FONT_HEADING,
    FONT_SMALL,
    PLOT_FUNCS,
    THEMES,
)
from .themes import FONT_MONO as FONT_MONO  # noqa: F401

# Re-exported for tests and external consumers (imported from here by tests):
from .themes import PLOT_CATEGORIES as PLOT_CATEGORIES  # noqa: F401
from .widgets import attach_tooltip

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
        self._auto_size_and_center()
        self.minsize(1050, 700)
        self.configure(fg_color=self.T['bg'])

        self._splash = BioSplashScreen(self, self.T)
        self.after(100, self._build_with_splash)

    def _auto_size_and_center(self):
        """Size the window to ~82% of the screen and center it."""
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w = min(1400, int(sw * 0.82))
        h = min(920, int(sh * 0.85))
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

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
        import matplotlib
        import numpy as np
        import pandas as pd
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
        self._auto_size_and_center()
        self.after(300, self._show_main_window)

    def _show_main_window(self):
        self.deiconify()
        self.update_idletasks()
        self._auto_size_and_center()
        self.lift()
        self.focus_force()

    def _build_plot_funcs(self):
        from ..plotting.biological_plots import (
            alignment_viewer,
            barplot_custom,
            batch_export_to_pdf,
            boxplot_custom,
            circos_plot,
            clustered_heatmap,
            dot_plot,
            export_all_to_folder,
            generate_markdown_story,
            heatmap_custom,
            ma_plot,
            manhattan_plot,
            pca_plot,
            qq_plot,
            raincloud_plot,
            ridge_plot,
            scatter_custom,
            timeseries_plot,
            venn_diagram,
            violin_plot,
            volcano_plot,
        )
        from ..plotting.math_plots import (
            cosine_plot,
            cubic_plot,
            exponential_plot,
            linear_plot,
            logistic_plot,
            quadratic_plot,
            sine_plot,
        )
        from ..plotting.specialized_plots import (
            gsea_plot,
            motif_logo,
            sankey_diagram,
            umap_plot,
        )
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
        self._refresh_status_right()

    #: Widgets whose text content should survive a theme rebuild.
    _PRESERVE_ON_REBUILD = ('seq_text',)

    def _rebuild_ui(self):
        # Stash user-typed content before destroying all widgets, so
        # switching theme does not wipe the user's input.
        stash = {}
        for attr in self._PRESERVE_ON_REBUILD:
            widget = getattr(self, attr, None)
            if widget is not None:
                try:
                    stash[attr] = widget.get("1.0", "end-1c")
                except Exception:
                    pass
        for widget in self.winfo_children():
            widget.destroy()
        self.all_cards = []
        self._progress_bar = None
        self._build_sidebar()
        self._build_content()
        self._show_frame(self._current_frame if hasattr(self, '_current_frame') else 'plots')
        self._apply_plot_search()
        for attr, content in stash.items():
            widget = getattr(self, attr, None)
            if widget is not None and content.strip():
                try:
                    widget.insert("1.0", content)
                except Exception:
                    pass

    def _on_close(self):
        self._closing = True
        try:
            if plt is not None:
                plt.close('all')
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def report_callback_exception(self, exc, val, tb):
        """Suppress harmless customtkinter teardown noise on shutdown.

        CTkScrollableFrame leaves pending after-callbacks that fire on
        already-destroyed widgets ('invalid command name'); while closing
        we swallow those instead of dumping tracebacks on the console.
        Anything else is reported normally.
        """
        import traceback
        if getattr(self, '_closing', False) and isinstance(val, tk.TclError):
            if 'invalid command name' in str(val):
                return
        print("Exception in Tkinter callback", file=sys.stderr)
        traceback.print_exception(exc, val, tb)

    def _show_plot_from_figure(self, fig, title="Plot"):
        """Display a matplotlib figure in an interactive, themed window.

        Embeds the figure with a FigureCanvasTkAgg so the user gets real
        zoom / pan / configure / save via the navigation toolbar (native
        toolbar save writes correct PNG/PDF/SVG bytes — no more PNG-in-.pdf).
        Falls back to a static image if canvas embedding fails.
        """
        self._record_plot(fig, title)
        T = self.T
        win = ctk.CTkToplevel(self)
        win.title(title)
        # Center over the main window so the plot opens where you look
        win.update_idletasks()
        pw, ph = 980, 780
        try:
            mx, my = self.winfo_rootx(), self.winfo_rooty()
            mw, mh = self.winfo_width(), self.winfo_height()
            px = mx + max(0, (mw - pw) // 2)
            py = my + max(0, (mh - ph) // 2)
            win.geometry(f"{pw}x{ph}+{px}+{py}")
        except Exception:
            win.geometry(f"{pw}x{ph}")
        win.minsize(640, 480)
        win.configure(fg_color=T.get('bg', '#0a0f0a'))
        win.transient(self)

        embedded = False
        try:
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )

            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()

            toolbar = NavigationToolbar2Tk(canvas, win, pack_toolbar=False)
            toolbar.update()
            self._style_mpl_toolbar(toolbar, T)
            toolbar.pack(side='bottom', fill='x')

            widget = canvas.get_tk_widget()
            widget.configure(bg=T.get('card', '#111c11'),
                             highlightthickness=0)
            widget.pack(fill='both', expand=True, padx=10, pady=10)
            embedded = True
        except Exception:
            embedded = False

        if not embedded:
            self._show_plot_static(win, fig)

        hint = ctk.CTkLabel(
            win, text="Toolbar: 🏠 reset  ·  ←→ history  ·  🔍 zoom  ·  ✋ pan  ·  💾 save  ·  Esc/Ctrl+W close",
            font=FONT_SMALL, text_color=T.get('text_muted', '#3d6b4a'))
        hint.pack(side='bottom', pady=(0, 6))

        def on_close():
            try:
                plt.close(fig)
            except Exception:
                pass
            win.destroy()

        win.bind('<Escape>', lambda e: on_close())
        win.bind('<Control-w>', lambda e: on_close())
        win.bind('<Control-W>', lambda e: on_close())
        win.protocol("WM_DELETE_WINDOW", on_close)
        win.lift()
        return win

    def _style_mpl_toolbar(self, toolbar, T):
        """Apply cyberpunk colors to the plain-tk matplotlib toolbar.

        Matplotlib ships dark-glyph icons only, so buttons get a LIGHT
        background for icon legibility while the strip itself stays themed.
        """
        card = T.get('card', '#111c11')
        text_c = T.get('text', '#e0ffe8')
        accent = T.get('accent', '#00ff88')
        btn_bg = '#dfe3ea'          # light chip so dark icons stay readable
        btn_active = '#c3c9d4'
        try:
            toolbar.configure(background=card)
            for child in toolbar.winfo_children():
                cls = child.winfo_class()
                if cls == 'Button':
                    child.configure(background=btn_bg,
                                    activebackground=btn_active,
                                    relief='flat', bd=0)
                elif cls == 'Label':
                    child.configure(background=card, foreground=text_c)
                elif cls in ('Frame', 'TFrame'):
                    child.configure(background=card)
            msg = getattr(toolbar, '_message_label', None)
            if msg is not None:
                msg.configure(background=card, foreground=accent)
        except Exception:
            pass

    def _show_plot_static(self, win, fig):
        """Static fallback: render figure to PNG and show it (no interaction)."""
        import tempfile

        from PIL import Image, ImageTk
        tmp = os.path.join(tempfile.gettempdir(), f"biosuite_{id(fig)}.png")
        fig.savefig(tmp, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        photo_ref = [None]
        try:
            img = Image.open(tmp)
            img.thumbnail((900, 650), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            photo_ref[0] = photo
            label = ctk.CTkLabel(win, image=photo, text="")
            label._photo_ref = photo_ref  # prevent GC
            label.pack(fill='both', expand=True, padx=10, pady=10)
        except Exception as e:
            ctk.CTkLabel(win, text=f"Error: {e}").pack(pady=20)

    def _on_resize(self, event=None):
        if event and event.widget == self:
            # Debounce: the Configure event storm during a window drag
            # collapses into one idle pass instead of forcing a full
            # update on every pixel of movement.
            if getattr(self, '_resize_scheduled', False):
                return
            self._resize_scheduled = True
            self.after_idle(lambda: setattr(self, '_resize_scheduled', False))

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
        ctk.CTkLabel(logo_area, text=f"v{__version__}", font=FONT_SMALL,
                      text_color=T['text_dim']).pack(anchor='w', pady=(2, 0))

        ctk.CTkFrame(self.sidebar_scroll, height=1, fg_color=T['border']).pack(fill='x', padx=10, pady=(8, 6))

        self.sidebar_buttons = {}

        def _sidebar_category(title):
            ctk.CTkLabel(self.sidebar_scroll, text=title, font=(FONT_FAMILY, 8, 'bold'),
                        text_color=T['text_muted'], anchor='w').pack(fill='x', padx=10, pady=(6, 1))

        FRAME_TIPS = {
            'plots': "40+ plot types: volcano, PCA, Manhattan, circos, UMAP...",
            'sequence': "GC%, reverse complement, translation, composition stats",
            'alignment': "Needleman-Wunsch & Smith-Waterman pairwise alignment",
            'phylogeny': "Distance matrices, UPGMA and Neighbor-Joining trees",
            'expression': "CPM/TPM/DESeq2 normalization & differential expression",
            'cloning': "Restriction digest, PCR, ligation and virtual gel electrophoresis",
            'crispr': "Guide RNA design with PAM finding and off-target scoring",
            'help': "Built-in guides for every module (or press F1)",
        }

        def _sidebar_item(key, label):
            btn = ctk.CTkButton(self.sidebar_scroll, text=f"  {label}", anchor='w',
                                font=(FONT_FAMILY, 11), height=28, corner_radius=6,
                                fg_color='transparent', text_color=T['sidebar_text'],
                                hover_color=T['sidebar_hover'],
                                command=lambda k=key: self._show_frame(k))
            btn.pack(fill='x', padx=8, pady=1)
            tip = FRAME_TIPS.get(key)
            if tip:
                attach_tooltip(btn, f"{label.split(' ', 1)[-1]} — {tip}", T)
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
        ctk.CTkLabel(version_frame, text=f"v{__version__}", font=(FONT_FAMILY, 9),
                      text_color=T['text_muted']).pack(anchor='w')
        # Don't reset the frame the user is on during a theme rebuild —
        # only set the initial value at first launch.
        if not hasattr(self, '_current_frame'):
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
        self.status_container = ctk.CTkFrame(self, fg_color=T['card'], height=30,
                                             corner_radius=0)
        self.status_container.pack(side='bottom', fill='x')
        self.status_container.pack_propagate(False)
        self.status_bar = ctk.CTkLabel(self.status_container, text="  Ready", anchor='w',
                                       font=FONT_SMALL, fg_color='transparent',
                                       text_color=T['text_dim'], height=30)
        self.status_bar.pack(side='left', fill='x', expand=True)
        self.status_right = ctk.CTkLabel(
            self.status_container,
            text=f"theme: {T.get('name', self.current_theme_key)}  ",
            anchor='e', font=FONT_SMALL, fg_color='transparent',
            text_color=T['text_muted'], height=30)
        self.status_right.pack(side='right')
        self._set_status(f"BioSuite Ultra v{__version__} loaded successfully")

    def _set_status(self, text, elapsed=None):
        msg = f"  {text}"
        if elapsed is not None:
            msg += f"   ·   {elapsed:,.1f}s"
        try:
            self.status_bar.configure(text=msg)
        except Exception:
            pass

    def _refresh_status_right(self):
        try:
            self.status_right.configure(
                text=f"theme: {self.T.get('name', self.current_theme_key)}  ")
        except Exception:
            pass

    # ─── Keyboard Shortcuts ─────────────────────────────────────────────────

    def _save_current(self):
        """Ctrl+S: save the current tab's main output (text or figure)."""
        key = getattr(self, '_current_frame', 'plots')

        # Sequence tab: save the results/stats box (or fall back to input).
        if key == 'sequence' and getattr(self, 'seq_stats', None) is not None:
            content = self.seq_stats.get("1.0", "end-1c")
            if content.strip():
                self._save_text_dialog(content, "sequence_results.txt")
                return

        # Plots tab: save the most recent figure at full quality.
        if key == 'plots' and getattr(self, '_plot_history', None):
            fig = self._plot_history[-1].get('fig')
            if fig is not None:
                self._save_figure_dialog(fig, "plot")
                return

        self._set_status("Nothing to save here yet (Ctrl+S)")

    def _save_text_dialog(self, content, default_name):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", initialfile=default_name,
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
            title="Save Output As")
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as fh:
                    fh.write(content)
                self._set_status(f"Saved: {path}")
            except Exception as e:
                self._msg_error("Save Error", str(e))

    def _save_figure_dialog(self, fig, default_name):
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=f"{default_name}.png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            title="Save Figure As")
        if path:
            try:
                fig.savefig(path, dpi=300, bbox_inches='tight',
                            facecolor=fig.get_facecolor())
                self._set_status(f"Figure saved: {path}")
            except Exception as e:
                self._msg_error("Save Error", str(e))

    def _refresh_current(self):
        key = self._current_frame
        self._show_frame(key)
        self._set_status(f"Refreshed: {key}")

    # ─── Background Execution ───────────────────────────────────────────────

    def _run_bg(self, work, on_result=None, on_error=None, status="Working..."):
        """Run `work()` in a daemon thread; marshal callbacks to the UI thread.

        Args:
            work: callable executed in the background; its return value is
                  passed to on_result.
            on_result: optional callable(result) — always runs on the UI thread.
            on_error: optional callable(exception) — UI thread. Defaults to a
                      themed error dialog.
            status: status-bar text while running; 'Ready' is restored after,
                    along with the operation's elapsed time.
        """
        import time
        started = time.perf_counter()
        self._set_status(status)
        self._busy_cursor(True)
        def runner():
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001 - last-resort UI guard
                if on_error is not None:
                    self.after(0, lambda e=exc: on_error(e))
                else:
                    self.after(0, lambda e=exc:
                               self._msg_error("Error", str(e)))
            else:
                if on_result is not None:
                    self.after(0, lambda r=result: on_result(r))
            finally:
                elapsed = time.perf_counter() - started
                self.after(0, lambda t=elapsed: self._set_status("Ready", elapsed=t))
                self.after(0, self._busy_cursor_off)
        threading.Thread(target=runner, daemon=True).start()

    def _busy_cursor(self, on):
        """Show a 'working' mouse cursor while a background op runs."""
        try:
            self.configure(cursor="watch" if on else "")
        except Exception:
            pass

    def _busy_cursor_off(self):
        self._busy_cursor(False)

    # ─── Progress Bar ───────────────────────────────────────────────────────

    def _show_progress(self, text="Working..."):
        if self._progress_bar is None:
            self._progress_bar = ctk.CTkProgressBar(self, height=3, corner_radius=1,
                                                      fg_color=self.T['border'],
                                                      progress_color=self.T['accent'])
            anchor = getattr(self, 'status_container', None)
            if anchor is not None:
                self._progress_bar.pack(side='bottom', fill='x', before=anchor)
            else:
                self._progress_bar.pack(side='bottom', fill='x')
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
        # Dedupe: callers may record before displaying and the plot window
        # records again — don't store the same figure twice in a row.
        if self._plot_history and self._plot_history[-1].get('fig') is fig:
            return
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
        if key not in self.frames:
            return
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
        # Keep the window title in sync with where the user is
        try:
            label = self.sidebar_buttons[key].cget('text').strip()
            self.title(f"BioSuite Ultra  ·  {label}")
        except Exception:
            pass

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

    def _action_button(self, parent, text, command, color_key='accent', tip=None):
        T = self.T
        color = T.get(color_key, T['accent'])
        hover = T.get(f'{color_key}_dim', color)
        btn = ctk.CTkButton(parent, text=text, height=36, corner_radius=8,
                             font=FONT_BUTTON, fg_color=color, hover_color=hover,
                             text_color='#000000' if color_key == 'accent' else '#ffffff',
                             command=command)
        if tip:
            attach_tooltip(btn, tip, T)
        return btn

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
            import tempfile
            import webbrowser

            import plotly.io as pio  # noqa: F401 - availability probe

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
            import numpy as np

            from biosuite.plotting.plot_api import volcano

            # Demo with sample data
            np.random.seed(42)
            fc = np.random.normal(0, 1.5, 500)
            pvals = np.random.uniform(0, 1, 500)
            pvals[:30] = np.random.uniform(1e-6, 0.05, 30)

            fig = volcano(fc, pvals, interactive=True, title="Demo Volcano Plot")
            self._show_interactive_plot(fig, "Volcano Plot")
        except Exception as e:
            self._msg_error("Error", str(e))
