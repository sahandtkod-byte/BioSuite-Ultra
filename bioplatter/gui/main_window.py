"""
Professional Cyberpunk GUI for BioSuite – Advanced Bioinformatic Platform.
3 themes: Dark-Green-Cyber, Dark-Purple-Cyber, Light-Blue-Cyber
"""
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
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
        'name': 'Dark-Green-Cyber',
        'ctk_mode': 'dark',
        'bg':           '#0a0f0a',
        'bg_secondary': '#0d170d',
        'card':         '#111c11',
        'card_hover':   '#162216',
        'accent':       '#00ff88',
        'accent_dim':   '#00cc6a',
        'accent_glow':  '#00ff8833',
        'text':         '#e0ffe8',
        'text_dim':     '#6b9b7a',
        'text_muted':   '#3d6b4a',
        'border':       '#1a3a1a',
        'border_light': '#2a5a2a',
        'sidebar_bg':   '#060d06',
        'sidebar_text': '#a0d0a0',
        'sidebar_hover':'#0f1f0f',
        'sidebar_active':'#00ff88',
        'sidebar_active_text': '#000000',
        'danger':       '#ff4444',
        'success':      '#00ff88',
        'input_bg':     '#0a150a',
        'scrollbar':    '#1a3a1a',
        'header_accent':'#00ff88',
    },
    'dark-purple': {
        'name': 'Dark-Purple-Cyber',
        'ctk_mode': 'dark',
        'bg':           '#0a0a12',
        'bg_secondary': '#0f0f1a',
        'card':         '#13132a',
        'card_hover':   '#1a1a35',
        'accent':       '#b44aff',
        'accent_dim':   '#9933e6',
        'accent_glow':  '#b44aff33',
        'text':         '#e8e0ff',
        'text_dim':     '#8a7aaa',
        'text_muted':   '#5a4a7a',
        'border':       '#2a1a3a',
        'border_light': '#3a2a5a',
        'sidebar_bg':   '#08080f',
        'sidebar_text': '#b0a0d0',
        'sidebar_hover':'#150f22',
        'sidebar_active':'#b44aff',
        'sidebar_active_text': '#ffffff',
        'danger':       '#ff4466',
        'success':      '#44ffaa',
        'input_bg':     '#0a0a18',
        'scrollbar':    '#2a1a3a',
        'header_accent':'#d080ff',
    },
    'light-blue': {
        'name': 'Light-Blue-Cyber',
        'ctk_mode': 'light',
        'bg':           '#f0f4fa',
        'bg_secondary': '#e8eef8',
        'card':         '#ffffff',
        'card_hover':   '#f5f8ff',
        'accent':       '#2563eb',
        'accent_dim':   '#1d4ed8',
        'accent_glow':  '#2563eb22',
        'text':         '#0f172a',
        'text_dim':     '#64748b',
        'text_muted':   '#94a3b8',
        'border':       '#e2e8f0',
        'border_light': '#cbd5e1',
        'sidebar_bg':   '#0f172a',
        'sidebar_text': '#94a3b8',
        'sidebar_hover':'#1e293b',
        'sidebar_active':'#3b82f6',
        'sidebar_active_text': '#ffffff',
        'danger':       '#dc2626',
        'success':      '#16a34a',
        'input_bg':     '#f8fafc',
        'scrollbar':    '#cbd5e1',
        'header_accent':'#2563eb',
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
FONT_SIDEBAR_TITLE = (FONT_FAMILY, 20, 'bold')
FONT_CODE = (FONT_MONO, 11)
FONT_BUTTON = (FONT_FAMILY, 12, 'bold')
FONT_BUTTON_SM = (FONT_FAMILY, 11)


# ─── Application ─────────────────────────────────────────────────────────────

class BioSuiteApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ── Theme init ──
        saved_theme = config.get('theme', 'dark-green')
        if saved_theme not in THEMES:
            saved_theme = 'dark-green'
        self.current_theme_key = saved_theme
        self.T = THEMES[self.current_theme_key]

        # ── Window setup ──
        self.title("BioSuite Pro  ·  Bioinformatic Platform")
        ctk.set_appearance_mode(self.T['ctk_mode'])
        ctk.set_default_color_theme("blue")

        # Fit to screen with padding
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

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Build UI ──
        self._build_sidebar()
        self._build_content()
        self._show_frame('plots')
        self._apply_plot_search()

        # Smooth startup
        self.after(50, lambda: self.configure(fg_color=self.T['bg']))

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

    # ─── Sidebar ──────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        T = self.T
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0,
                                     fg_color=T['sidebar_bg'])
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        # ── Logo area ──
        logo_area = ctk.CTkFrame(self.sidebar, fg_color='transparent')
        logo_area.pack(fill='x', padx=20, pady=(28, 6))

        ctk.CTkLabel(logo_area, text="BIOSUITE",
                      font=(FONT_FAMILY, 18, 'bold'),
                      text_color=T['accent']).pack(anchor='w')
        ctk.CTkLabel(logo_area, text="Bioinformatic Platform",
                      font=FONT_SMALL,
                      text_color=T['text_dim']).pack(anchor='w', pady=(2, 0))

        # Separator
        ctk.CTkFrame(self.sidebar, height=1, fg_color=T['border']).pack(
            fill='x', padx=18, pady=(16, 12))

        # ── Navigation ──
        self.sidebar_buttons = {}
        nav_items = [
            ('plots',      'Plots Gallery'),
            ('sequence',   'Sequence Analysis'),
            ('alignment',  'Alignments'),
            ('phylogeny',  'Phylogeny'),
            ('expression', 'Expression'),
            ('ngs',        'NGS / VCF'),
        ]
        for key, label in nav_items:
            btn = ctk.CTkButton(
                self.sidebar, text=f"  {label}", anchor='w',
                font=FONT_SIDEBAR, height=40, corner_radius=8,
                fg_color='transparent',
                text_color=T['sidebar_text'],
                hover_color=T['sidebar_hover'],
                command=lambda k=key: self._show_frame(k))
            btn.pack(fill='x', padx=12, pady=2)
            self.sidebar_buttons[key] = btn

        # Separator
        ctk.CTkFrame(self.sidebar, height=1, fg_color=T['border']).pack(
            fill='x', padx=18, pady=(16, 12))

        # ── Theme selector ──
        ctk.CTkLabel(self.sidebar, text="THEME", font=(FONT_FAMILY, 9, 'bold'),
                      text_color=T['text_muted']).pack(anchor='w', padx=20, pady=(0, 6))

        self.theme_buttons = {}
        theme_keys = [('dark-green', 'Green Cyber'), ('dark-purple', 'Purple Cyber'), ('light-blue', 'Light Blue')]
        for tkey, tlabel in theme_keys:
            is_active = tkey == self.current_theme_key
            btn = ctk.CTkButton(
                self.sidebar, text=f"  {tlabel}", anchor='w',
                font=FONT_SMALL, height=32, corner_radius=6,
                fg_color=T['sidebar_active'] if is_active else 'transparent',
                text_color=T['sidebar_active_text'] if is_active else T['sidebar_text'],
                hover_color=T['sidebar_hover'],
                command=lambda k=tkey: self._apply_theme(k))
            btn.pack(fill='x', padx=12, pady=1)
            self.theme_buttons[tkey] = btn

        # ── Version badge ──
        version_frame = ctk.CTkFrame(self.sidebar, fg_color='transparent')
        version_frame.pack(side='bottom', fill='x', padx=18, pady=(0, 16))
        ctk.CTkLabel(version_frame, text="v2.0 Pro",
                      font=(FONT_FAMILY, 9),
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

        # Status bar
        self.status_bar = ctk.CTkLabel(
            self, text="  Ready", anchor='w',
            font=FONT_SMALL,
            fg_color=T['card'],
            text_color=T['text_dim'],
            height=30)
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
                btn.configure(fg_color=T['sidebar_active'],
                              text_color=T['sidebar_active_text'])
            else:
                btn.configure(fg_color='transparent', text_color=T['sidebar_text'])
        self._current_frame = key

    # ─── Card Helper ──────────────────────────────────────────────────────────

    def _card(self, parent, **kwargs):
        T = self.T
        defaults = dict(fg_color=T['card'], corner_radius=12, border_width=1,
                        border_color=T['border'])
        defaults.update(kwargs)
        card = ctk.CTkFrame(parent, **defaults)
        self.all_cards.append(card)
        return card

    def _section_header(self, parent, text):
        T = self.T
        ctk.CTkLabel(parent, text=text, font=FONT_HEADING,
                      text_color=T['text']).pack(anchor='w', padx=4, pady=(4, 8))

    def _action_button(self, parent, text, command, color_key='accent'):
        T = self.T
        color = T.get(color_key, T['accent'])
        hover = T.get(f'{color_key}_dim', color)
        return ctk.CTkButton(
            parent, text=text, height=36, corner_radius=8,
            font=FONT_BUTTON, fg_color=color, hover_color=hover,
            text_color='#000000' if color_key == 'accent' else '#ffffff',
            command=command)

    def _input_entry(self, parent, placeholder, **kwargs):
        T = self.T
        defaults = dict(height=36, font=FONT_BODY, corner_radius=8,
                        fg_color=T['input_bg'], border_color=T['border'],
                        text_color=T['text'], placeholder_text=placeholder,
                        placeholder_text_color=T['text_muted'])
        defaults.update(kwargs)
        return ctk.CTkEntry(parent, **defaults)

    def _text_box(self, parent, height=200, **kwargs):
        T = self.T
        defaults = dict(height=height, font=FONT_CODE, corner_radius=8,
                        fg_color=T['input_bg'], border_color=T['border'],
                        text_color=T['text'], border_width=1)
        defaults.update(kwargs)
        return ctk.CTkTextbox(parent, **defaults)

    def _label(self, parent, text, style='body'):
        T = self.T
        font_map = {'title': FONT_HEADING, 'sub': FONT_SUBHEADING,
                     'body': FONT_BODY, 'small': FONT_SMALL, 'dim': FONT_SMALL}
        color_map = {'title': T['text'], 'sub': T['text'],
                      'body': T['text'], 'small': T['text_dim'], 'dim': T['text_muted']}
        return ctk.CTkLabel(parent, text=text, font=font_map.get(style, FONT_BODY),
                             text_color=color_map.get(style, T['text']))

    # ─── Plots Gallery ────────────────────────────────────────────────────────

    def _build_plot_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['plots'] = f

        # Header
        header = ctk.CTkFrame(f, fg_color='transparent')
        header.pack(fill='x', pady=(0, 8))
        self._label(header, 'Plots Gallery', 'title').pack(side='left')

        self.plot_search = self._input_entry(header, "Search plots...", width=280)
        self.plot_search.pack(side='right')
        self._search_after_id = None
        self.plot_search.bind('<KeyRelease>', self._on_search_key)

        # Main area
        mid = ctk.CTkFrame(f, fg_color='transparent')
        mid.pack(fill='both', expand=True, pady=(0, 8))

        # Categories sidebar
        cat_card = self._card(mid, width=190)
        cat_card.pack(side='left', fill='y', padx=(0, 10))
        cat_card.pack_propagate(False)

        self._label(cat_card, 'Categories', 'sub').pack(padx=14, pady=(14, 6), anchor='w')

        self.cat_buttons = {}
        for cat in ['All'] + list(PLOT_CATEGORIES.keys()):
            btn = ctk.CTkButton(
                cat_card, text=cat, anchor='w', height=30, corner_radius=6,
                font=FONT_SMALL, fg_color='transparent',
                text_color=T['text_dim'], hover_color=T['border'],
                command=lambda c=cat: self._select_category(c))
            btn.pack(fill='x', padx=8, pady=1)
            self.cat_buttons[cat] = btn
        self._selected_cat = 'All'

        # Plot list
        list_card = self._card(mid)
        list_card.pack(side='right', fill='both', expand=True)

        self.plot_count_label = self._label(list_card, '', 'dim')
        self.plot_count_label.pack(padx=14, pady=(10, 2), anchor='w')

        scroll_frame = ctk.CTkScrollableFrame(
            list_card, fg_color='transparent', corner_radius=0,
            scrollbar_button_color=T['scrollbar'],
            scrollbar_button_hover_color=T['border_light'])
        scroll_frame.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        self.plot_buttons_frame = scroll_frame

        self._select_category('All')

        # Action buttons
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
                btn.configure(fg_color=T['sidebar_active'],
                              text_color=T['sidebar_active_text'])
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
            row = ctk.CTkFrame(self.plot_buttons_frame, fg_color='transparent', height=40)
            row.pack(fill='x', pady=2)
            row.pack_propagate(False)
            btn = ctk.CTkButton(
                row, text=f"  {name}", anchor='w', height=36, corner_radius=8,
                font=FONT_BODY, fg_color='transparent',
                text_color=T['text'], hover_color=T['border'],
                command=lambda p=pid, n=name: self._select_and_gen(p, n))
            btn.pack(fill='x', padx=(0, 80))
            ctk.CTkLabel(row, text=cat, font=FONT_SMALL,
                          text_color=T['text_muted']).pack(side='right', padx=12)

    def _select_and_gen(self, plot_id, name):
        self._selected_plot_id = plot_id
        self._generate_plot_by_id(plot_id)

    def _generate_selected_plot(self):
        if not hasattr(self, '_selected_plot_id') or self._selected_plot_id is None:
            messagebox.showinfo("Info", "Select a plot from the list, then click Generate.")
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
                    messagebox.showerror("Error", f"Plot '{plot_id}' not found.")
            except Exception as e:
                messagebox.showerror("Plot Error", str(e))
            finally:
                builtins.input = original_input
                self.after(0, lambda: self._set_status("Ready"))
        threading.Thread(target=run, daemon=True).start()

    def _gui_input(self, prompt):
        result = [None]
        event = threading.Event()
        def ask():
            if 'Load data from file?' in prompt:
                result[0] = 'y' if messagebox.askyesno("Data", prompt) else 'n'
            elif 'Show data summary' in prompt or 'Use default' in prompt or 'Use default alignment' in prompt:
                result[0] = 'y' if messagebox.askyesno("Confirm", prompt) else 'n'
            elif 'Save this plot?' in prompt or 'Save as HTML' in prompt:
                result[0] = 'y' if messagebox.askyesno("Save", prompt) else 'n'
            elif 'Correlation type' in prompt:
                result[0] = 'pearson'
            elif 'File path' in prompt or 'file path' in prompt:
                path = filedialog.askopenfilename(title=prompt,
                    filetypes=[("CSV/Excel", "*.csv *.xlsx *.tsv *.txt"), ("All", "*.*")])
                result[0] = path if path else ''
            elif 'column' in prompt.lower():
                result[0] = simpledialog.askstring("Column", prompt) or ''
            elif 'Load ranked list' in prompt:
                result[0] = 'y' if messagebox.askyesno("Data", prompt) else 'n'
            elif 'Enter sequences' in prompt:
                result[0] = ''
            elif 'Switch to' in prompt:
                result[0] = ''
            elif 'Filename' in prompt:
                result[0] = ''
            else:
                match = re.search(r'\(default ([^)]+)\)', prompt)
                default = match.group(1) if match else ''
                result[0] = simpledialog.askstring("Input", prompt, initialvalue=default) or default
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
                    self.after(0, lambda: messagebox.showinfo("Done", f"All plots exported to:\n{folder}"))
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("Export Error", str(e)))
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
                    self.after(0, lambda: messagebox.showinfo("Done", f"PDF saved:\n{path}"))
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("PDF Error", str(e)))
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

        # Left: Input
        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))

        self._label(left, 'Input Sequence (FASTA/FASTQ)', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.seq_text = self._text_box(left, height=280)
        self.seq_text.pack(fill='both', expand=True, padx=10, pady=(0, 6))

        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=10, pady=(0, 10))
        self._action_button(btn_row, "Load File", self._load_seq_file).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Clear", lambda: self.seq_text.delete("1.0", "end"), 'danger').pack(side='left')

        # Right: Results
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
            messagebox.showerror("Error", str(e))

    def _get_seq(self):
        text = self.seq_text.get("1.0", "end").strip()
        return ''.join(line.strip() for line in text.split('\n')
                       if not line.startswith(('>', '@', '+'))).replace(' ', '')

    def _seq_gc(self):
        seq = self._get_seq()
        if not seq:
            messagebox.showwarning("No sequence", "Enter or load a sequence first.")
            return
        self.seq_stats.delete("1.0", "end")
        self.seq_stats.insert("end", f"Length: {len(seq)}\n")
        self.seq_stats.insert("end", f"GC%: {gc_content(seq):.2f}%\n")
        s = sequence_stats(seq)
        self.seq_stats.insert("end", f"A:{s['A']} T:{s['T']} G:{s['G']} C:{s['C']} N:{s['N']}\n")

    def _seq_revcomp(self):
        seq = self._get_seq()
        if not seq:
            messagebox.showwarning("No sequence", "Enter or load a sequence first.")
            return
        rc = reverse_complement(seq)
        self.seq_stats.delete("1.0", "end")
        self.seq_stats.insert("end", f"Original ({len(seq)} bp):\n{seq[:200]}{'...' if len(seq)>200 else ''}\n\n")
        self.seq_stats.insert("end", f"Reverse Complement ({len(rc)} bp):\n{rc[:200]}{'...' if len(rc)>200 else ''}\n")

    def _seq_translate(self):
        seq = self._get_seq()
        if not seq:
            messagebox.showwarning("No sequence", "Enter or load a sequence first.")
            return
        prot = translate(seq, frame=1)
        self.seq_stats.delete("1.0", "end")
        self.seq_stats.insert("end", f"Translation (frame 1, {len(prot)} aa):\n{prot[:300]}{'...' if len(prot)>300 else ''}\n")

    def _seq_stats_cmd(self):
        seq = self._get_seq()
        if not seq:
            messagebox.showwarning("No sequence", "Enter or load a sequence first.")
            return
        stats = sequence_stats(seq)
        self.seq_stats.delete("1.0", "end")
        for k, v in stats.items():
            if isinstance(v, float):
                self.seq_stats.insert("end", f"{k}: {v:.4f}\n")
            else:
                self.seq_stats.insert("end", f"{k}: {v}\n")

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
            messagebox.showwarning("Missing", "Enter both sequences.")
            return
        a1, a2, score = needleman_wunsch(s1, s2)
        self.align_result.delete("1.0", "end")
        self.align_result.insert("end", f"Score: {score}\n\nSeq1: {a1}\nSeq2: {a2}\n")

    def _align_sw(self):
        s1 = self._get_align_seq(self.align_seq1)
        s2 = self._get_align_seq(self.align_seq2)
        if not s1 or not s2:
            messagebox.showwarning("Missing", "Enter both sequences.")
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
            messagebox.showwarning("No data", "Enter aligned sequences (>name and sequence).")
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
            messagebox.showerror("Error", "Need at least 2 sequences.")
            return
        if len(set(len(s) for s in seqs)) != 1:
            messagebox.showerror("Error", "Sequences must be same length (aligned).")
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
            messagebox.showerror("Error", str(e))

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
            messagebox.showwarning("No file", "Select a count matrix first.")
            return
        conds_str = self.expr_conds.get().strip()
        if not conds_str:
            messagebox.showwarning("No conditions", "Enter conditions.")
            return
        try:
            counts = read_counts_matrix(path)
            if counts is None:
                messagebox.showerror("Error", "Failed to read counts matrix.")
                return
            conditions = [c.strip() for c in conds_str.split(',')]
            num_numeric = counts.select_dtypes(include=[np.number]).shape[1]
            if len(conditions) != num_numeric:
                messagebox.showerror("Mismatch",
                    f"Conditions ({len(conditions)}) != numeric columns ({num_numeric}).")
                return
            de = differential_expression(counts, conditions)
            self.expr_result.delete("1.0", "end")
            self.expr_result.insert("end", "DE Results (top 20):\n\n")
            self.expr_result.insert("end", de.head(20).to_string())
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(de['log2FC'], -np.log10(de['pvalue']), s=5, alpha=0.5)
            ax.axhline(-np.log10(0.05), color='red', linestyle='--')
            ax.axvline(-1, color='gray', linestyle='--')
            ax.axvline(1, color='gray', linestyle='--')
            ax.set_xlabel('log2FC')
            ax.set_ylabel('-log10(pvalue)')
            ax.set_title('Volcano Plot (DE Analysis)')
            plt.show()
            self._set_status("DE analysis complete")
        except Exception as e:
            messagebox.showerror("Error", str(e))

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
            messagebox.showwarning("No file", "Select a VCF file first.")
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
            messagebox.showerror("Error", str(e))
