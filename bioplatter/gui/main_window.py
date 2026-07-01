"""
Professional Cyberpunk GUI for BioSuite – Advanced Bioinformatic Platform.
3 themes: Dark-Green-Cyber, Dark-Purple-Cyber, Light-Blue-Cyber
Custom themed dialogs, splash screen, no native Windows widgets.
"""
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import os
import builtins
import pandas as pd
import numpy as np
import re
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from ..core.utils import config, set_theme, save_config, load_dataframe_safe, center_window
from ..core.sequence import (read_fasta, read_fastq, read_genbank,
                              sequence_stats, reverse_complement, translate, gc_content)
from ..core.alignment import needleman_wunsch, smith_waterman
from ..core.phylogeny import distance_matrix, upgma_tree, plot_phylogenetic_tree
from ..core.ngs import read_vcf, manhattan_from_vcf
from ..core.expression import read_counts_matrix, cpm_normalization, differential_expression
from ..plotting.biological_plots import (volcano_plot, pca_plot, manhattan_plot, ma_plot,
    venn_diagram, barplot_custom, boxplot_custom, heatmap_custom, scatter_custom,
    timeseries_plot, qq_plot, clustered_heatmap, circos_plot, alignment_viewer,
    violin_plot, raincloud_plot, ridge_plot, dot_plot, export_all_to_folder,
    generate_markdown_story, batch_export_to_pdf)
from ..plotting.math_plots import (sine_plot, cosine_plot, linear_plot,
    quadratic_plot, cubic_plot, exponential_plot, logistic_plot)
from ..plotting.specialized_plots import gsea_plot, motif_logo, sankey_diagram, umap_plot

# ─── Theme Definitions ───────────────────────────────────────────────────────

THEMES = {
    'dark-green': {
        'name': 'Dark-Green-Cyber', 'ctk_mode': 'dark',
        'bg': '#0a0f0a', 'bg_secondary': '#0d170d', 'card': '#111c11',
        'card_hover': '#162216', 'accent': '#00ff88', 'accent_dim': '#00cc6a',
        'accent_glow': '#00cc6a', 'text': '#e0ffe8', 'text_dim': '#6b9b7a',
        'text_muted': '#3d6b4a', 'border': '#1a3a1a', 'border_light': '#2a5a2a',
        'sidebar_bg': '#060d06', 'sidebar_text': '#a0d0a0', 'sidebar_hover': '#0f1f0f',
        'sidebar_active': '#00ff88', 'sidebar_active_text': '#000000',
        'danger': '#ff4444', 'success': '#00ff88', 'input_bg': '#0a150a',
        'scrollbar': '#1a3a1a', 'header_accent': '#00ff88',
        'overlay': '#000000', 'dialog_bg': '#0d1a0d', 'dialog_border': '#00cc6a',
    },
    'dark-purple': {
        'name': 'Dark-Purple-Cyber', 'ctk_mode': 'dark',
        'bg': '#0a0a12', 'bg_secondary': '#0f0f1a', 'card': '#13132a',
        'card_hover': '#1a1a35', 'accent': '#b44aff', 'accent_dim': '#9933e6',
        'accent_glow': '#9933e6', 'text': '#e8e0ff', 'text_dim': '#8a7aaa',
        'text_muted': '#5a4a7a', 'border': '#2a1a3a', 'border_light': '#3a2a5a',
        'sidebar_bg': '#08080f', 'sidebar_text': '#b0a0d0', 'sidebar_hover': '#150f22',
        'sidebar_active': '#b44aff', 'sidebar_active_text': '#ffffff',
        'danger': '#ff4466', 'success': '#44ffaa', 'input_bg': '#0a0a18',
        'scrollbar': '#2a1a3a', 'header_accent': '#d080ff',
        'overlay': '#000000', 'dialog_bg': '#120f1f', 'dialog_border': '#9933e6',
    },
    'light-blue': {
        'name': 'Light-Blue-Cyber', 'ctk_mode': 'light',
        'bg': '#f0f4fa', 'bg_secondary': '#e8eef8', 'card': '#ffffff',
        'card_hover': '#f5f8ff', 'accent': '#2563eb', 'accent_dim': '#1d4ed8',
        'accent_glow': '#1d4ed8', 'text': '#0f172a', 'text_dim': '#64748b',
        'text_muted': '#94a3b8', 'border': '#e2e8f0', 'border_light': '#cbd5e1',
        'sidebar_bg': '#0f172a', 'sidebar_text': '#94a3b8', 'sidebar_hover': '#1e293b',
        'sidebar_active': '#3b82f6', 'sidebar_active_text': '#ffffff',
        'danger': '#dc2626', 'success': '#16a34a', 'input_bg': '#f8fafc',
        'scrollbar': '#cbd5e1', 'header_accent': '#2563eb',
        'overlay': '#0f172a', 'dialog_bg': '#ffffff', 'dialog_border': '#2563eb',
    },
}

PLOT_CATEGORIES = {
    'Advanced Biological': [
        ('Volcano Plot', 'volcano'), ('PCA Plot', 'pca'),
        ('Manhattan Plot', 'manhattan'), ('MA Plot', 'ma'), ('Venn Diagram', 'venn'),
    ],
    'Basic Biological': [
        ('Barplot', 'barplot'), ('Boxplot', 'boxplot'), ('Heatmap', 'heatmap'),
        ('Scatter Plot', 'scatter'), ('Time Series', 'timeseries'),
    ],
    'Mathematical': [
        ('Sine', 'sine'), ('Cosine', 'cosine'), ('Linear', 'linear'),
        ('Quadratic', 'quadratic'), ('Cubic', 'cubic'),
        ('Exponential', 'exponential'), ('Logistic', 'logistic'),
    ],
    'Specialized': [
        ('GSEA Plot', 'gsea'), ('Motif Logo', 'motif'), ('Sankey Diagram', 'sankey'),
    ],
    'Additional': [
        ('QQ-plot', 'qq'), ('Clustered Heatmap', 'clustered_heatmap'),
        ('Circos Plot', 'circos'), ('Alignment Viewer', 'alignment'), ('UMAP Plot', 'umap'),
    ],
    'New Plots': [
        ('Violin Plot', 'violin'), ('Raincloud Plot', 'raincloud'),
        ('Ridge Plot', 'ridge'), ('Dot Plot', 'dotplot'),
    ],
}

PLOT_FUNCS = {
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
}

# ─── Font Constants ───────────────────────────────────────────────────────────

FONT_FAMILY = 'Segoe UI'
FONT_MONO = 'Consolas'
FONT_TITLE = (FONT_FAMILY, 22, 'bold')
FONT_HEADING = (FONT_FAMILY, 16, 'bold')
FONT_SUBHEADING = (FONT_FAMILY, 13, 'bold')
FONT_BODY = (FONT_FAMILY, 12)
FONT_SMALL = (FONT_FAMILY, 10)
FONT_SIDEBAR = (FONT_FAMILY, 12)
FONT_CODE = (FONT_MONO, 11)
FONT_BUTTON = (FONT_FAMILY, 12, 'bold')


# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM THEMED DIALOGS — no native Windows look
# ═══════════════════════════════════════════════════════════════════════════════

class _BaseDialog(ctk.CTkToplevel):
    """Base class for all custom themed dialogs."""

    def __init__(self, parent, T, title="Dialog", width=420, height=220):
        super().__init__(parent)
        self.T = T
        self.result = None
        self.title(title)
        self.overrideredirect(True)
        self.attributes('-topmost', True)

        # Center on parent
        self.update_idletasks()
        pw = parent.winfo_toplevel()
        px = pw.winfo_x() + (pw.winfo_width() - width) // 2
        py = pw.winfo_y() + (pw.winfo_height() - height) // 2
        self.geometry(f"{width}x{height}+{px}+{py}")
        self.configure(fg_color=T['bg'])

        # Outer glow border
        self._outer = ctk.CTkFrame(self, fg_color=T.get('dialog_border', T['border']),
                                    corner_radius=16)
        self._outer.pack(fill='both', expand=True, padx=2, pady=2)

        # Inner card
        self._card = ctk.CTkFrame(self._outer, fg_color=T.get('dialog_bg', T['card']),
                                   corner_radius=14)
        self._card.pack(fill='both', expand=True, padx=2, pady=2)

        # Content container (subclasses fill this)
        self._body = ctk.CTkFrame(self._card, fg_color='transparent')
        self._body.pack(fill='both', expand=True, padx=24, pady=(20, 16))

        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # Delay grab_set so the previous dialog's grab is fully released first
        self.after(50, self._do_grab)

        # Escape to close
        self.bind('<Escape>', lambda e: self._on_cancel())

    def _do_grab(self):
        try:
            self.grab_set()
        except Exception:
            pass

    def _on_cancel(self):
        self.result = None
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


class BioMessageDialog(_BaseDialog):
    """Themed info / warning / error dialog with icon and OK button."""

    def __init__(self, parent, T, title="Message", message="", msg_type="info"):
        self._msg_type = msg_type
        h = 180 if len(message) < 80 else 220
        super().__init__(parent, T, title=title, width=440, height=h)

        # Icon + message row
        top = ctk.CTkFrame(self._body, fg_color='transparent')
        top.pack(fill='x', pady=(0, 16))

        icons = {'info': '\u2139', 'warning': '\u26A0', 'error': '\u2716', 'success': '\u2714'}
        colors = {'info': T['accent'], 'warning': '#f59e0b', 'error': T['danger'], 'success': T['success']}
        icon = icons.get(msg_type, '\u2139')
        color = colors.get(msg_type, T['accent'])

        ctk.CTkLabel(top, text=icon, font=(FONT_FAMILY, 28),
                      text_color=color, width=40).pack(side='left', padx=(0, 12), anchor='n')
        ctk.CTkLabel(top, text=message, font=FONT_BODY,
                      text_color=T['text'], wraplength=340, justify='left').pack(side='left', fill='x', expand=True)

        # OK button
        btn_row = ctk.CTkFrame(self._body, fg_color='transparent')
        btn_row.pack(fill='x')
        ctk.CTkButton(btn_row, text="OK", width=100, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['accent'],
                       text_color='#000000' if msg_type != 'error' else '#ffffff',
                       hover_color=T['accent_dim'],
                       command=self._on_ok).pack(anchor='e')
        self.bind('<Return>', lambda e: self._on_ok())

    def _on_ok(self):
        self.result = 'ok'
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


class BioConfirmDialog(_BaseDialog):
    """Themed yes/no confirmation dialog."""

    def __init__(self, parent, T, title="Confirm", message=""):
        h = 180 if len(message) < 80 else 220
        super().__init__(parent, T, title=title, width=440, height=h)

        top = ctk.CTkFrame(self._body, fg_color='transparent')
        top.pack(fill='x', pady=(0, 16))

        ctk.CTkLabel(top, text='\u2753', font=(FONT_FAMILY, 28),
                      text_color=T['accent'], width=40).pack(side='left', padx=(0, 12), anchor='n')
        ctk.CTkLabel(top, text=message, font=FONT_BODY,
                      text_color=T['text'], wraplength=340, justify='left').pack(side='left', fill='x', expand=True)

        btn_row = ctk.CTkFrame(self._body, fg_color='transparent')
        btn_row.pack(fill='x')
        ctk.CTkButton(btn_row, text="Cancel", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['border_light'],
                       text_color=T['text'], hover_color=T['border'],
                       command=self._on_no).pack(side='right', padx=(8, 0))
        ctk.CTkButton(btn_row, text="Yes", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['accent'],
                       text_color='#000000', hover_color=T['accent_dim'],
                       command=self._on_yes).pack(side='right')
        self.bind('<Return>', lambda e: self._on_yes())
        self.bind('<Escape>', lambda e: self._on_no())

    def _on_yes(self):
        self.result = True
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _on_no(self):
        self.result = False
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


class BioInputDialog(_BaseDialog):
    """Themed text input dialog with placeholder and OK/Cancel."""

    def __init__(self, parent, T, title="Input", prompt="", default=""):
        super().__init__(parent, T, title=title, width=460, height=230)

        # Prompt
        ctk.CTkLabel(self._body, text=prompt, font=FONT_BODY,
                      text_color=T['text'], wraplength=400, justify='left').pack(anchor='w', pady=(0, 10))

        # Entry
        self._entry = ctk.CTkEntry(self._body, height=40, font=(FONT_MONO, 13),
                                    corner_radius=8, fg_color=T['input_bg'],
                                    border_color=T['border'], text_color=T['text'],
                                    placeholder_text_color=T['text_muted'])
        self._entry.pack(fill='x', pady=(0, 16))
        if default:
            self._entry.insert(0, default)
        self._entry.select_range(0, 'end')
        # Delay focus_set so the entry is fully rendered and can receive keystrokes
        self.after(80, lambda: self._entry.focus_force())

        # Buttons
        btn_row = ctk.CTkFrame(self._body, fg_color='transparent')
        btn_row.pack(fill='x')
        ctk.CTkButton(btn_row, text="Cancel", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['border_light'],
                       text_color=T['text'], hover_color=T['border'],
                       command=self._on_cancel).pack(side='right', padx=(8, 0))
        ctk.CTkButton(btn_row, text="OK", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['accent'],
                       text_color='#000000', hover_color=T['accent_dim'],
                       command=self._on_ok).pack(side='right')
        self.bind('<Return>', lambda e: self._on_ok())

    def _on_ok(self):
        self.result = self._entry.get()
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _on_cancel(self):
        self.result = None
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


class BioFilePickerDialog(_BaseDialog):
    """Themed file picker with browse button, path display, and file type filter."""

    def __init__(self, parent, T, title="Select File", filetypes=None, prompt="Choose a file:"):
        super().__init__(parent, T, title=title, width=520, height=240)
        self._filetypes = filetypes or [("All Files", "*.*")]

        ctk.CTkLabel(self._body, text=prompt, font=FONT_BODY,
                      text_color=T['text'], wraplength=460, justify='left').pack(anchor='w', pady=(0, 10))

        # Path display + browse
        row = ctk.CTkFrame(self._body, fg_color='transparent')
        row.pack(fill='x', pady=(0, 16))

        self._path_entry = ctk.CTkEntry(row, height=40, font=FONT_CODE,
                                         corner_radius=8, fg_color=T['input_bg'],
                                         border_color=T['border'], text_color=T['text'],
                                         placeholder_text="No file selected...",
                                         placeholder_text_color=T['text_muted'])
        self._path_entry.pack(side='left', fill='x', expand=True, padx=(0, 8))

        ctk.CTkButton(row, text="Browse", width=90, height=40, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['accent'],
                       text_color='#000000', hover_color=T['accent_dim'],
                       command=self._browse).pack(side='right')

        # Buttons
        btn_row = ctk.CTkFrame(self._body, fg_color='transparent')
        btn_row.pack(fill='x')
        ctk.CTkButton(btn_row, text="Cancel", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['border_light'],
                       text_color=T['text'], hover_color=T['border'],
                       command=self._on_cancel).pack(side='right', padx=(8, 0))
        ctk.CTkButton(btn_row, text="OK", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['accent'],
                       text_color='#000000', hover_color=T['accent_dim'],
                       command=self._on_ok).pack(side='right')
        self.bind('<Return>', lambda e: self._on_ok())
        self.after(80, lambda: self._path_entry.focus_force())

    def _browse(self):
        path = filedialog.askopenfilename(filetypes=self._filetypes)
        if path:
            self._path_entry.delete(0, 'end')
            self._path_entry.insert(0, path)

    def _on_ok(self):
        self.result = self._path_entry.get().strip()
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _on_cancel(self):
        self.result = None
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


class BioDropdownDialog(_BaseDialog):
    """Themed dropdown/combobox selection dialog."""

    def __init__(self, parent, T, title="Select", prompt="", options=None, default=None):
        super().__init__(parent, T, title=title, width=420, height=240)
        self._options = options or []

        ctk.CTkLabel(self._body, text=prompt, font=FONT_BODY,
                      text_color=T['text'], wraplength=380, justify='left').pack(anchor='w', pady=(0, 10))

        self._combo = ctk.CTkComboBox(self._body, values=self._options, height=40,
                                       font=FONT_BODY, corner_radius=8,
                                       fg_color=T['input_bg'], border_color=T['border'],
                                       button_color=T['accent'], button_hover_color=T['accent_dim'],
                                       dropdown_fg_color=T['card'], dropdown_hover_color=T['border'],
                                       dropdown_text_color=T['text'], text_color=T['text'])
        self._combo.pack(fill='x', pady=(0, 16))
        if default and default in self._options:
            self._combo.set(default)
        elif self._options:
            self._combo.set(self._options[0])
        self.after(80, lambda: self._combo.focus_force())

        btn_row = ctk.CTkFrame(self._body, fg_color='transparent')
        btn_row.pack(fill='x')
        ctk.CTkButton(btn_row, text="Cancel", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['border_light'],
                       text_color=T['text'], hover_color=T['border'],
                       command=self._on_cancel).pack(side='right', padx=(8, 0))
        ctk.CTkButton(btn_row, text="OK", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['accent'],
                       text_color='#000000', hover_color=T['accent_dim'],
                       command=self._on_ok).pack(side='right')
        self.bind('<Return>', lambda e: self._on_ok())

    def _on_ok(self):
        self.result = self._combo.get()
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _on_cancel(self):
        self.result = None
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
#  SPLASH SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

class BioSplashScreen(ctk.CTkToplevel):
    """Animated splash/loading screen shown during app initialization."""

    def __init__(self, parent, T):
        super().__init__(parent)
        self.T = T
        w, h = 480, 300
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.overrideredirect(True)
        self.configure(fg_color=T['bg'])
        self.attributes('-alpha', 0.0)

        # Outer border glow
        outer = ctk.CTkFrame(self, fg_color=T.get('dialog_border', T['border']),
                              corner_radius=18)
        outer.pack(fill='both', expand=True, padx=3, pady=3)

        card = ctk.CTkFrame(outer, fg_color=T.get('dialog_bg', T['card']),
                              corner_radius=16)
        card.pack(fill='both', expand=True, padx=3, pady=3)

        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=40, pady=36)

        # Logo
        ctk.CTkLabel(inner, text="BIOSUITE", font=(FONT_FAMILY, 32, 'bold'),
                      text_color=T['accent']).pack(anchor='w')
        ctk.CTkLabel(inner, text="Bioinformatic Platform  ·  v2.0 Pro",
                      font=(FONT_FAMILY, 12), text_color=T['text_dim']).pack(anchor='w', pady=(4, 0))

        # Separator line
        sep = ctk.CTkFrame(inner, height=2, fg_color=T['border'])
        sep.pack(fill='x', pady=(24, 20))

        # Loading text
        self._status_label = ctk.CTkLabel(inner, text="Initializing modules...",
                                           font=(FONT_FAMILY, 11), text_color=T['text_dim'])
        self._status_label.pack(anchor='w')

        # Progress bar
        self._progress = ctk.CTkProgressBar(inner, height=4, corner_radius=2,
                                             fg_color=T['border'], progress_color=T['accent'])
        self._progress.pack(fill='x', pady=(12, 0))
        self._progress.set(0)

        self._step = 0
        self._animate_in()

    def _animate_in(self):
        alpha = self.attributes('-alpha')
        if alpha < 1.0:
            alpha = min(1.0, alpha + 0.08)
            self.attributes('-alpha', alpha)
            self.after(20, self._animate_in)

    def update_status(self, text, progress):
        self._status_label.configure(text=text)
        self._progress.set(progress)
        self.update_idletasks()

    def animate_out(self):
        self._fade_out()

    def _fade_out(self):
        alpha = self.attributes('-alpha')
        if alpha > 0.0:
            alpha = max(0.0, alpha - 0.1)
            self.attributes('-alpha', alpha)
            self.after(20, self._fade_out)
        else:
            self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class BioSuiteApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        saved_theme = config.get('theme', 'dark-green')
        if saved_theme not in THEMES:
            saved_theme = 'dark-green'
        self.current_theme_key = saved_theme
        self.T = THEMES[self.current_theme_key]

        self.title("BioSuite Pro  ·  Bioinformatic Platform")
        ctk.set_appearance_mode(self.T['ctk_mode'])
        ctk.set_default_color_theme("blue")

        # Fit to screen
        self.withdraw()  # Hide main window during splash
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

        # Show splash
        self._splash = BioSplashScreen(self, self.T)
        self.after(100, self._build_with_splash)

    def _build_with_splash(self):
        steps = [
            ("Loading sequence engine...", 0.15),
            ("Loading alignment module...", 0.30),
            ("Loading expression analysis...", 0.45),
            ("Loading plot renderers...", 0.60),
            ("Building interface...", 0.80),
            ("Applying theme...", 0.95),
        ]
        def run_steps(i):
            if i < len(steps):
                text, prog = steps[i]
                self._splash.update_status(text, prog)
                self.after(120, lambda: run_steps(i + 1))
            else:
                self._splash.update_status("Ready.", 1.0)
                self.after(300, self._finish_startup)
        run_steps(0)

    def _finish_startup(self):
        self._build_sidebar()
        self._build_content()
        self._show_frame('plots')
        self._apply_plot_search()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind('<Configure>', self._on_resize)
        self._splash.animate_out()
        self.after(300, self.deiconify)  # Show main window after splash fades

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
        self._build_sidebar()
        self._build_content()
        self._show_frame(self._current_frame if hasattr(self, '_current_frame') else 'plots')
        self._apply_plot_search()

    def _on_close(self):
        plt.close('all')
        self.destroy()

    def _on_resize(self, event=None):
        if event and event.widget == self:
            self.update_idletasks()

    # ─── Sidebar ──────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        T = self.T
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=T['sidebar_bg'])
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        logo_area = ctk.CTkFrame(self.sidebar, fg_color='transparent')
        logo_area.pack(fill='x', padx=20, pady=(28, 6))
        ctk.CTkLabel(logo_area, text="BIOSUITE", font=(FONT_FAMILY, 18, 'bold'),
                      text_color=T['accent']).pack(anchor='w')
        ctk.CTkLabel(logo_area, text="Bioinformatic Platform", font=FONT_SMALL,
                      text_color=T['text_dim']).pack(anchor='w', pady=(2, 0))

        ctk.CTkFrame(self.sidebar, height=1, fg_color=T['border']).pack(fill='x', padx=18, pady=(16, 12))

        self.sidebar_buttons = {}
        nav_items = [('plots', 'Plots Gallery'), ('sequence', 'Sequence Analysis'),
                     ('alignment', 'Alignments'), ('phylogeny', 'Phylogeny'),
                     ('expression', 'Expression'), ('ngs', 'NGS / VCF')]
        for key, label in nav_items:
            btn = ctk.CTkButton(self.sidebar, text=f"  {label}", anchor='w',
                                font=FONT_SIDEBAR, height=40, corner_radius=8,
                                fg_color='transparent', text_color=T['sidebar_text'],
                                hover_color=T['sidebar_hover'],
                                command=lambda k=key: self._show_frame(k))
            btn.pack(fill='x', padx=12, pady=2)
            self.sidebar_buttons[key] = btn

        ctk.CTkFrame(self.sidebar, height=1, fg_color=T['border']).pack(fill='x', padx=18, pady=(16, 12))

        ctk.CTkLabel(self.sidebar, text="THEME", font=(FONT_FAMILY, 9, 'bold'),
                      text_color=T['text_muted']).pack(anchor='w', padx=20, pady=(0, 6))

        self.theme_buttons = {}
        for tkey, tlabel in [('dark-green', 'Green Cyber'), ('dark-purple', 'Purple Cyber'), ('light-blue', 'Light Blue')]:
            is_active = tkey == self.current_theme_key
            btn = ctk.CTkButton(self.sidebar, text=f"  {tlabel}", anchor='w',
                                font=FONT_SMALL, height=32, corner_radius=6,
                                fg_color=T['sidebar_active'] if is_active else 'transparent',
                                text_color=T['sidebar_active_text'] if is_active else T['sidebar_text'],
                                hover_color=T['sidebar_hover'],
                                command=lambda k=tkey: self._apply_theme(k))
            btn.pack(fill='x', padx=12, pady=1)
            self.theme_buttons[tkey] = btn

        version_frame = ctk.CTkFrame(self.sidebar, fg_color='transparent')
        version_frame.pack(side='bottom', fill='x', padx=18, pady=(0, 16))
        ctk.CTkLabel(version_frame, text="v2.0 Pro", font=(FONT_FAMILY, 9),
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
        self.status_bar = ctk.CTkLabel(self, text="  Ready", anchor='w', font=FONT_SMALL,
                                        fg_color=T['card'], text_color=T['text_dim'], height=30)
        self.status_bar.pack(side='bottom', fill='x')
        self._set_status("BioSuite Pro loaded successfully")

    def _set_status(self, text):
        self.status_bar.configure(text=f"  {text}")

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
        d = dict(height=36, font=FONT_BODY, corner_radius=8, fg_color=T['input_bg'],
                 border_color=T['border'], text_color=T['text'],
                 placeholder_text=placeholder, placeholder_text_color=T['text_muted'])
        d.update(kwargs)
        return ctk.CTkEntry(parent, **d)

    def _text_box(self, parent, height=200, **kwargs):
        T = self.T
        d = dict(height=height, font=FONT_CODE, corner_radius=8, fg_color=T['input_bg'],
                 border_color=T['border'], text_color=T['text'], border_width=1)
        d.update(kwargs)
        return ctk.CTkTextbox(parent, **d)

    def _label(self, parent, text, style='body'):
        T = self.T
        fonts = {'title': FONT_HEADING, 'sub': FONT_SUBHEADING, 'body': FONT_BODY,
                 'small': FONT_SMALL, 'dim': FONT_SMALL}
        colors = {'title': T['text'], 'sub': T['text'], 'body': T['text'],
                   'small': T['text_dim'], 'dim': T['text_muted']}
        return ctk.CTkLabel(parent, text=text, font=fonts.get(style, FONT_BODY),
                             text_color=colors.get(style, T['text']))

    # ─── Plots Gallery ────────────────────────────────────────────────────────

    def _build_plot_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['plots'] = f

        header = ctk.CTkFrame(f, fg_color='transparent')
        header.pack(fill='x', pady=(0, 8))
        self._label(header, 'Plots Gallery', 'title').pack(side='left')
        self.plot_search = self._input_entry(header, "Search plots...", width=280)
        self.plot_search.pack(side='right')
        self._search_after_id = None
        self.plot_search.bind('<KeyRelease>', self._on_search_key)

        mid = ctk.CTkFrame(f, fg_color='transparent')
        mid.pack(fill='both', expand=True, pady=(0, 8))

        cat_card = self._card(mid, width=220)
        cat_card.pack(side='left', fill='y', padx=(0, 12))
        cat_card.pack_propagate(False)

        self._label(cat_card, 'Categories', 'sub').pack(padx=12, pady=(14, 8), anchor='w')

        self.cat_buttons = {}
        for cat in ['All'] + list(PLOT_CATEGORIES.keys()):
            btn = ctk.CTkButton(cat_card, text=cat, anchor='w', height=34, corner_radius=8,
                                font=(FONT_FAMILY, 12), fg_color='transparent', text_color=T['text_dim'],
                                hover_color=T['border'],
                                command=lambda c=cat: self._select_category(c))
            btn.pack(fill='x', padx=8, pady=2)
            self.cat_buttons[cat] = btn
        self._selected_cat = 'All'

        list_card = self._card(mid)
        list_card.pack(side='right', fill='both', expand=True)
        self.plot_count_label = self._label(list_card, '', 'dim')
        self.plot_count_label.pack(padx=14, pady=(10, 2), anchor='w')

        scroll_frame = ctk.CTkScrollableFrame(list_card, fg_color='transparent', corner_radius=0,
                                                scrollbar_button_color=T['scrollbar'],
                                                scrollbar_button_hover_color=T['border_light'])
        scroll_frame.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        self.plot_buttons_frame = scroll_frame
        self._select_category('All')

        btn_row = ctk.CTkFrame(f, fg_color='transparent')
        btn_row.pack(fill='x')
        self._action_button(btn_row, "Generate Plot", self._generate_selected_plot).pack(side='left', padx=(0, 8))
        self._action_button(btn_row, "Export All", self._export_all_plots, 'success').pack(side='left', padx=(0, 8))
        self._action_button(btn_row, "Batch PDF", self._batch_pdf, 'accent_dim').pack(side='left')

    def _select_category(self, cat):
        T = self.T
        self._selected_cat = cat
        for k, btn in self.cat_buttons.items():
            if k == cat:
                btn.configure(fg_color=T['sidebar_active'], text_color=T['sidebar_active_text'])
            else:
                btn.configure(fg_color='transparent', text_color=T['text_dim'])
        self._apply_plot_search()

    def _on_search_key(self, event=None):
        if self._search_after_id:
            self.after_cancel(self._search_after_id)
        self._search_after_id = self.after(150, self._apply_plot_search)

    def _apply_plot_search(self):
        if not hasattr(self, 'plot_buttons_frame'):
            return
        T = self.T
        for w in self.plot_buttons_frame.winfo_children():
            w.destroy()
        term = self.plot_search.get().strip().lower() if hasattr(self, 'plot_search') else ''
        items = []
        if self._selected_cat == 'All':
            for cat, plots in PLOT_CATEGORIES.items():
                for name, pid in plots:
                    items.append((name, pid, cat))
        else:
            for name, pid in PLOT_CATEGORIES.get(self._selected_cat, []):
                items.append((name, pid, self._selected_cat))
        if term:
            items = [(n, p, c) for n, p, c in items if term in n.lower()]
        self._plot_items = items
        self.plot_count_label.configure(text=f"{len(items)} plots available")
        self._selected_plot_id = None
        for name, pid, cat in items:
            row = ctk.CTkFrame(self.plot_buttons_frame, fg_color='transparent')
            row.pack(fill='x', pady=2)
            btn = ctk.CTkButton(row, text=f"  {name}", anchor='w', height=36, corner_radius=8,
                                font=(FONT_FAMILY, 13), fg_color='transparent', text_color=T['text'],
                                hover_color=T['border'],
                                command=lambda p=pid, n=name: self._select_and_gen(p, n))
            btn.pack(side='left', fill='x', expand=True)
            ctk.CTkLabel(row, text=cat, font=(FONT_FAMILY, 10),
                          text_color=T['text_muted'], width=150, anchor='e').pack(side='right', padx=12)

    def _select_and_gen(self, plot_id, name):
        self._selected_plot_id = plot_id
        self._generate_plot_by_id(plot_id)

    def _generate_selected_plot(self):
        if not hasattr(self, '_selected_plot_id') or self._selected_plot_id is None:
            self._msg_info("Info", "Select a plot from the list, then click Generate.")
            return
        self._generate_plot_by_id(self._selected_plot_id)

    def _generate_plot_by_id(self, plot_id):
        self._set_status(f"Generating {plot_id}...")
        def run():
            original_input = builtins.input
            builtins.input = self._gui_input
            try:
                func = PLOT_FUNCS.get(plot_id)
                if func:
                    func()
                else:
                    self.after(0, lambda: self._msg_error("Error", f"Plot '{plot_id}' not found."))
            except Exception as e:
                self.after(0, lambda: self._msg_error("Plot Error", str(e)))
            finally:
                builtins.input = original_input
                self.after(0, lambda: self._set_status("Ready"))
        threading.Thread(target=run, daemon=True).start()

    def _gui_input(self, prompt):
        result = [None]
        event = threading.Event()
        def ask():
            if 'Load data from file?' in prompt or 'Load ranked list' in prompt:
                result[0] = 'y' if self._confirm("Data", prompt) else 'n'
            elif 'Show data summary' in prompt or 'Use default' in prompt or 'Use default alignment' in prompt:
                result[0] = 'y' if self._confirm("Confirm", prompt) else 'n'
            elif 'Save this plot?' in prompt or 'Save as HTML' in prompt:
                result[0] = 'y' if self._confirm("Save", prompt) else 'n'
            elif 'Correlation type' in prompt:
                result[0] = 'pearson'
            elif 'File path' in prompt or 'file path' in prompt:
                r = self._ask_file("Select File", prompt,
                                    [("CSV/Excel", "*.csv *.xlsx *.tsv *.txt"), ("All", "*.*")])
                result[0] = r if r else ''
            elif 'column' in prompt.lower():
                r = self._ask_input("Column", prompt)
                result[0] = r if r else ''
            elif 'Enter sequences' in prompt or 'Switch to' in prompt or 'Filename' in prompt:
                result[0] = ''
            else:
                match = re.search(r'\(default ([^)]+)\)', prompt)
                default = match.group(1) if match else ''
                r = self._ask_input("Input", prompt, default=default)
                result[0] = r if r is not None else default
            event.set()
        self.after(0, ask)
        event.wait(timeout=120)
        return result[0] if result[0] is not None else ''

    def _export_all_plots(self):
        folder = filedialog.askdirectory(title="Select Export Folder")
        if folder:
            self._set_status("Exporting all plots...")
            def run():
                try:
                    export_all_to_folder(folder)
                    self.after(0, lambda: self._msg_success("Done", f"All plots exported to:\n{folder}"))
                except Exception as e:
                    self.after(0, lambda: self._msg_error("Export Error", str(e)))
                finally:
                    self.after(0, lambda: self._set_status("Ready"))
            threading.Thread(target=run, daemon=True).start()

    def _batch_pdf(self):
        path = filedialog.asksaveasfilename(defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")], title="Save PDF Report")
        if path:
            self._set_status("Generating PDF report...")
            def run():
                try:
                    batch_export_to_pdf(path)
                    self.after(0, lambda: self._msg_success("Done", f"PDF saved:\n{path}"))
                except Exception as e:
                    self.after(0, lambda: self._msg_error("PDF Error", str(e)))
                finally:
                    self.after(0, lambda: self._set_status("Ready"))
            threading.Thread(target=run, daemon=True).start()

    # ─── Sequence Tab ─────────────────────────────────────────────────────────

    def _build_sequence_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['sequence'] = f
        self._section_header(f, "Sequence Analysis")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Input Sequence (FASTA/FASTQ)', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.seq_text = self._text_box(left, height=280)
        self.seq_text.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=10, pady=(0, 10))
        self._action_button(btn_row, "Load File", self._load_seq_file).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Clear", lambda: self.seq_text.delete("1.0", "end"), 'danger').pack(side='left')

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Results & Statistics', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.seq_stats = self._text_box(right, height=280)
        self.seq_stats.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        btn_row2 = ctk.CTkFrame(right, fg_color='transparent')
        btn_row2.pack(fill='x', padx=10, pady=(0, 10))
        for label, cmd in [("GC%", self._seq_gc), ("Rev Comp", self._seq_revcomp),
                           ("Translate", self._seq_translate), ("Stats", self._seq_stats_cmd)]:
            self._action_button(btn_row2, label, cmd).pack(side='left', padx=(0, 6))

    def _load_seq_file(self):
        path = filedialog.askopenfilename(filetypes=[
            ("FASTA/FASTQ/GenBank", "*.fasta *.fa *.fq *.fastq *.gb *.genbank")])
        if not path:
            return
        try:
            if path.endswith(('.fasta', '.fa')):
                data = read_fasta(path)
                content = "\n".join(f">{name}\n{seq}" for name, seq in data) if data else ""
            elif path.endswith(('.fastq', '.fq')):
                data = read_fastq(path)
                content = "\n".join(f"@{name}\n{seq}\n+\n{qual}" for name, seq, qual in data) if data else ""
            else:
                data = read_genbank(path)
                content = "\n".join(f">{name}\n{seq}" for name, seq, _ in data) if data else ""
            self.seq_text.delete("1.0", "end")
            self.seq_text.insert("end", content)
            self._set_status(f"Loaded {os.path.basename(path)}")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _get_seq(self):
        text = self.seq_text.get("1.0", "end").strip()
        return ''.join(line.strip() for line in text.split('\n')
                       if not line.startswith(('>', '@', '+'))).replace(' ', '')

    def _seq_gc(self):
        seq = self._get_seq()
        if not seq:
            self._msg_warning("No sequence", "Enter or load a sequence first.")
            return
        self.seq_stats.delete("1.0", "end")
        self.seq_stats.insert("end", f"Length: {len(seq)}\n")
        self.seq_stats.insert("end", f"GC%: {gc_content(seq):.2f}%\n")
        s = sequence_stats(seq)
        self.seq_stats.insert("end", f"A:{s['A']} T:{s['T']} G:{s['G']} C:{s['C']} N:{s['N']}\n")

    def _seq_revcomp(self):
        seq = self._get_seq()
        if not seq:
            self._msg_warning("No sequence", "Enter or load a sequence first.")
            return
        rc = reverse_complement(seq)
        self.seq_stats.delete("1.0", "end")
        self.seq_stats.insert("end", f"Original ({len(seq)} bp):\n{seq[:200]}{'...' if len(seq)>200 else ''}\n\n")
        self.seq_stats.insert("end", f"Reverse Complement ({len(rc)} bp):\n{rc[:200]}{'...' if len(rc)>200 else ''}\n")

    def _seq_translate(self):
        seq = self._get_seq()
        if not seq:
            self._msg_warning("No sequence", "Enter or load a sequence first.")
            return
        prot = translate(seq, frame=1)
        self.seq_stats.delete("1.0", "end")
        self.seq_stats.insert("end", f"Translation (frame 1, {len(prot)} aa):\n{prot[:300]}{'...' if len(prot)>300 else ''}\n")

    def _seq_stats_cmd(self):
        seq = self._get_seq()
        if not seq:
            self._msg_warning("No sequence", "Enter or load a sequence first.")
            return
        stats = sequence_stats(seq)
        self.seq_stats.delete("1.0", "end")
        for k, v in stats.items():
            self.seq_stats.insert("end", f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")

    # ─── Alignment Tab ────────────────────────────────────────────────────────

    def _build_alignment_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['alignment'] = f
        self._section_header(f, "Pairwise Sequence Alignment")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        self._label(inner, 'Sequence 1', 'sub').pack(anchor='w')
        self.align_seq1 = self._text_box(inner, height=80)
        self.align_seq1.pack(fill='x', pady=(0, 8))
        self._label(inner, 'Sequence 2', 'sub').pack(anchor='w')
        self.align_seq2 = self._text_box(inner, height=80)
        self.align_seq2.pack(fill='x', pady=(0, 8))

        btn_row = ctk.CTkFrame(inner, fg_color='transparent')
        btn_row.pack(fill='x', pady=(0, 8))
        self._action_button(btn_row, "Needleman-Wunsch (Global)", self._align_nw).pack(side='left', padx=(0, 8))
        self._action_button(btn_row, "Smith-Waterman (Local)", self._align_sw, 'accent_dim').pack(side='left')

        self.align_result = self._text_box(inner, height=120)
        self.align_result.pack(fill='both', expand=True)

    def _get_align_seq(self, widget):
        return widget.get("1.0", "end").strip().replace('\n', '').replace(' ', '')

    def _align_nw(self):
        s1 = self._get_align_seq(self.align_seq1)
        s2 = self._get_align_seq(self.align_seq2)
        if not s1 or not s2:
            self._msg_warning("Missing", "Enter both sequences.")
            return
        a1, a2, score = needleman_wunsch(s1, s2)
        self.align_result.delete("1.0", "end")
        self.align_result.insert("end", f"Score: {score}\n\nSeq1: {a1}\nSeq2: {a2}\n")

    def _align_sw(self):
        s1 = self._get_align_seq(self.align_seq1)
        s2 = self._get_align_seq(self.align_seq2)
        if not s1 or not s2:
            self._msg_warning("Missing", "Enter both sequences.")
            return
        a1, a2, score = smith_waterman(s1, s2)
        self.align_result.delete("1.0", "end")
        self.align_result.insert("end", f"Score: {score}\n\nSeq1: {a1}\nSeq2: {a2}\n")

    # ─── Phylogeny Tab ────────────────────────────────────────────────────────

    def _build_phylogeny_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['phylogeny'] = f
        self._section_header(f, "Phylogenetic Tree Builder")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        self._label(inner, 'Aligned Sequences (FASTA format)', 'sub').pack(anchor='w')
        self.phylo_input = self._text_box(inner, height=160)
        self.phylo_input.pack(fill='x', pady=(0, 8))
        self._action_button(inner, "Build UPGMA Tree", self._build_tree).pack(pady=(0, 8))
        self.phylo_result = self._text_box(inner, height=140)
        self.phylo_result.pack(fill='both', expand=True)

    def _build_tree(self):
        text = self.phylo_input.get("1.0", "end").strip()
        if not text:
            self._msg_warning("No data", "Enter aligned sequences (>name and sequence).")
            return
        lines = text.split('\n')
        names, seqs, current = [], [], []
        for line in lines:
            if line.startswith('>'):
                if current:
                    seqs.append(''.join(current))
                    current = []
                names.append(line[1:].strip())
            else:
                current.append(line.strip())
        if current:
            seqs.append(''.join(current))
        if len(seqs) < 2:
            self._msg_error("Error", "Need at least 2 sequences.")
            return
        if len(set(len(s) for s in seqs)) != 1:
            self._msg_error("Error", "Sequences must be same length (aligned).")
            return
        try:
            dist_mat = distance_matrix(seqs)
            link_mat = upgma_tree(dist_mat, names)
            fig = plot_phylogenetic_tree(link_mat, names)
            fig.show()
            self.phylo_result.delete("1.0", "end")
            self.phylo_result.insert("end", "Distance Matrix:\n")
            for i, name in enumerate(names):
                row = ", ".join(f"{dist_mat[i][j]:.3f}" for j in range(len(names)))
                self.phylo_result.insert("end", f"{name}: {row}\n")
            self.phylo_result.insert("end", "\nTree displayed in matplotlib window.")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── Expression Tab ───────────────────────────────────────────────────────

    def _build_expression_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['expression'] = f
        self._section_header(f, "Differential Expression")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 8))
        self.expr_path = self._input_entry(file_row, "Count matrix file (CSV/TSV)...")
        self.expr_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(file_row, "Browse",
                            lambda: self._browse_file(self.expr_path, [("CSV/TSV", "*.csv *.tsv *.txt")])
                            ).pack(side='right')

        self._label(inner, 'Conditions (comma-separated, e.g., control,control,treat,treat)', 'dim').pack(anchor='w', pady=(0, 2))
        self.expr_conds = self._input_entry(inner, "control,control,treat,treat")
        self.expr_conds.pack(fill='x', pady=(0, 8))
        self._action_button(inner, "Run Differential Expression", self._run_expr).pack(pady=(0, 8))
        self.expr_result = self._text_box(inner, height=200)
        self.expr_result.pack(fill='both', expand=True)

    def _browse_file(self, entry, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            entry.delete(0, 'end')
            entry.insert(0, path)

    def _run_expr(self):
        path = self.expr_path.get().strip()
        if not path:
            self._msg_warning("No file", "Select a count matrix first.")
            return
        conds_str = self.expr_conds.get().strip()
        if not conds_str:
            self._msg_warning("No conditions", "Enter conditions.")
            return
        try:
            counts = read_counts_matrix(path)
            if counts is None:
                self._msg_error("Error", "Failed to read counts matrix.")
                return
            conditions = [c.strip() for c in conds_str.split(',')]
            num_numeric = counts.select_dtypes(include=[np.number]).shape[1]
            if len(conditions) != num_numeric:
                self._msg_error("Mismatch", f"Conditions ({len(conditions)}) != numeric columns ({num_numeric}).")
                return
            de = differential_expression(counts, conditions)
            self.expr_result.delete("1.0", "end")
            self.expr_result.insert("end", "DE Results (top 20, BH-corrected):\n\n")
            self.expr_result.insert("end", de.head(20).to_string())
            fig, ax = plt.subplots(figsize=(7, 5))
            sig = de['padj'] < 0.05
            ax.scatter(de.loc[~sig, 'log2FC'], -np.log10(de.loc[~sig, 'pvalue']),
                       s=5, alpha=0.3, color='gray', label='Not sig')
            ax.scatter(de.loc[sig, 'log2FC'], -np.log10(de.loc[sig, 'pvalue']),
                       s=10, alpha=0.7, color='red', label='FDR < 0.05')
            ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
            ax.axvline(-1, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('log2 Fold Change')
            ax.set_ylabel('-log10(p-value)')
            ax.set_title(f'DE Analysis (BH-corrected) — {sig.sum()} significant genes')
            ax.legend()
            plt.show()
            self._set_status(f"DE complete: {sig.sum()} significant genes (FDR < 0.05)")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── NGS Tab ──────────────────────────────────────────────────────────────

    def _build_ngs_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['ngs'] = f
        self._section_header(f, "NGS / VCF Analysis")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 8))
        self.vcf_path = self._input_entry(file_row, "VCF file path...")
        self.vcf_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(file_row, "Browse",
                            lambda: self._browse_file(self.vcf_path, [("VCF", "*.vcf")])
                            ).pack(side='right')
        self._action_button(inner, "Load VCF & Show Manhattan Plot", self._load_vcf).pack(pady=(0, 8))
        self.vcf_result = self._text_box(inner, height=240)
        self.vcf_result.pack(fill='both', expand=True)

    def _load_vcf(self):
        path = self.vcf_path.get().strip()
        if not path:
            self._msg_warning("No file", "Select a VCF file first.")
            return
        try:
            df = read_vcf(path)
            if df is None:
                self.vcf_result.delete("1.0", "end")
                self.vcf_result.insert("end", "Failed to read VCF.")
                return
            self.vcf_result.delete("1.0", "end")
            self.vcf_result.insert("end", f"Loaded {len(df)} variants.\n\n")
            self.vcf_result.insert("end", df.head(20).to_string())
            manh = manhattan_from_vcf(df)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.scatter(manh['POS'], manh['neg_log10'], s=2, alpha=0.6)
            ax.axhline(-np.log10(5e-8), color='red', linestyle='--')
            ax.set_xlabel('Position')
            ax.set_ylabel('-log10(p)')
            ax.set_title('Manhattan Plot from VCF')
            plt.show()
            self._set_status("VCF loaded, Manhattan plot displayed")
        except Exception as e:
            self._msg_error("Error", str(e))
