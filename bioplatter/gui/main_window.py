"""
Modern CustomTkinter GUI for BioSuite – Professional bioinformatic platform.
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

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

COLORS = {
    'bg': '#f0f2f5', 'card': '#ffffff', 'accent': '#2563eb',
    'accent_hover': '#1d4ed8', 'text': '#1e293b', 'text_secondary': '#64748b',
    'border': '#e2e8f0', 'success': '#16a34a', 'danger': '#dc2626',
    'sidebar_bg': '#0f172a', 'sidebar_text': '#e2e8f0', 'sidebar_hover': '#1e293b',
    'sidebar_active': '#2563eb',
}
DARK_COLORS = {
    'bg': '#0f172a', 'card': '#1e293b', 'accent': '#3b82f6',
    'accent_hover': '#2563eb', 'text': '#f1f5f9', 'text_secondary': '#94a3b8',
    'border': '#334155', 'success': '#22c55e', 'danger': '#ef4444',
    'sidebar_bg': '#020617', 'sidebar_text': '#e2e8f0', 'sidebar_hover': '#1e293b',
    'sidebar_active': '#3b82f6',
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


class BioSuiteApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("BioSuite – Advanced Bioinformatic Platform")
        self.geometry("1300x900")
        self.minsize(1100, 750)
        self.is_dark = config.get('theme', 'light') == 'dark'
        self.colors = DARK_COLORS if self.is_dark else COLORS
        if self.is_dark:
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")
        self.configure(fg_color=self.colors['bg'])
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_sidebar()
        self._build_content()
        self._show_frame('plots')
        self._apply_plot_search()

    def _get_colors(self):
        return DARK_COLORS if self.is_dark else COLORS

    def _refresh_colors(self):
        self.colors = self._get_colors()
        self.configure(fg_color=self.colors['bg'])
        self.sidebar.configure(fg_color=self.colors['sidebar_bg'])
        for btn in self.sidebar_buttons.values():
            btn.configure(fg_color=self.colors['sidebar_bg'], text_color=self.colors['sidebar_text'])
        active = self._current_frame
        if active and active in self.sidebar_buttons:
            self.sidebar_buttons[active].configure(fg_color=self.colors['sidebar_active'])
        for frame in self.frames.values():
            frame.configure(fg_color=self.colors['bg'])
        for card in self.all_cards:
            card.configure(fg_color=self.colors['card'], border_color=self.colors['border'])
        self.status_bar.configure(fg_color=self.colors['card'], text_color=self.colors['text_secondary'])

    def _on_close(self):
        plt.close('all')
        self.destroy()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color=self.colors['sidebar_bg'])
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color='transparent')
        logo_frame.pack(fill='x', padx=16, pady=(20, 8))
        ctk.CTkLabel(logo_frame, text="BioSuite", font=ctk.CTkFont(size=22, weight="bold"),
                      text_color='#60a5fa').pack(anchor='w')
        ctk.CTkLabel(logo_frame, text="v2.0 Pro", font=ctk.CTkFont(size=11),
                      text_color='#94a3b8').pack(anchor='w')
        ctk.CTkFrame(self.sidebar, height=1, fg_color='#334155').pack(fill='x', padx=16, pady=8)
        self.sidebar_buttons = {}
        nav_items = [
            ('plots', 'Plots Gallery'), ('sequence', 'Sequence Analysis'),
            ('alignment', 'Alignments'), ('phylogeny', 'Phylogeny'),
            ('expression', 'Expression'), ('ngs', 'NGS / VCF'),
        ]
        for key, label in nav_items:
            btn = ctk.CTkButton(self.sidebar, text=f"  {label}", anchor='w',
                                font=ctk.CTkFont(size=13), height=38, corner_radius=8,
                                fg_color=self.colors['sidebar_bg'],
                                text_color=self.colors['sidebar_text'],
                                hover_color=self.colors['sidebar_hover'],
                                command=lambda k=key: self._show_frame(k))
            btn.pack(fill='x', padx=10, pady=2)
            self.sidebar_buttons[key] = btn
        ctk.CTkFrame(self.sidebar, height=1, fg_color='#334155').pack(fill='x', padx=16, pady=8)
        self.theme_btn = ctk.CTkButton(self.sidebar, text="Dark Mode", font=ctk.CTkFont(size=12),
                                        height=34, corner_radius=8,
                                        fg_color=self.colors['sidebar_bg'],
                                        text_color=self.colors['sidebar_text'],
                                        hover_color=self.colors['sidebar_hover'],
                                        command=self._toggle_theme)
        self.theme_btn.pack(fill='x', padx=10, pady=2)
        if self.is_dark:
            self.theme_btn.configure(text="Light Mode")
        self._current_frame = 'plots'

    def _build_content(self):
        self.content = ctk.CTkFrame(self, fg_color=self.colors['bg'], corner_radius=0)
        self.content.pack(side='right', fill='both', expand=True)
        self.frames = {}
        self.all_cards = []
        self._build_plot_frame()
        self._build_sequence_frame()
        self._build_alignment_frame()
        self._build_phylogeny_frame()
        self._build_expression_frame()
        self._build_ngs_frame()
        self.status_bar = ctk.CTkLabel(self, text="Ready", anchor='w',
                                        font=ctk.CTkFont(size=11),
                                        fg_color=self.colors['card'],
                                        text_color=self.colors['text_secondary'],
                                        height=28)
        self.status_bar.pack(side='bottom', fill='x', padx=0, pady=0)
        self._set_status("Ready – BioSuite v2.0 Pro loaded")

    def _set_status(self, text):
        self.status_bar.configure(text=f"  {text}")

    def _show_frame(self, key):
        for f in self.frames.values():
            f.pack_forget()
        self.frames[key].pack(in_=self.content, fill='both', expand=True, padx=12, pady=12)
        for k, btn in self.sidebar_buttons.items():
            if k == key:
                btn.configure(fg_color=self.colors['sidebar_active'], text_color='white')
            else:
                btn.configure(fg_color=self.colors['sidebar_bg'], text_color=self.colors['sidebar_text'])
        self._current_frame = key

    def _toggle_theme(self):
        self.is_dark = not self.is_dark
        config['theme'] = 'dark' if self.is_dark else 'light'
        save_config(config)
        set_theme(config['theme'])
        if self.is_dark:
            ctk.set_appearance_mode("dark")
            self.theme_btn.configure(text="Light Mode")
        else:
            ctk.set_appearance_mode("light")
            self.theme_btn.configure(text="Dark Mode")
        self._refresh_colors()

    # ── Plots Gallery ──────────────────────────────────────────
    def _build_plot_frame(self):
        f = ctk.CTkFrame(self.content, fg_color=self.colors['bg'])
        self.frames['plots'] = f
        header = ctk.CTkFrame(f, fg_color='transparent')
        header.pack(fill='x', padx=4, pady=(8, 4))
        ctk.CTkLabel(header, text="Plots Gallery", font=ctk.CTkFont(size=20, weight="bold"),
                      text_color=self.colors['text']).pack(side='left')
        self.plot_search = ctk.CTkEntry(header, placeholder_text="Search plots...",
                                         width=260, height=34,
                                         font=ctk.CTkFont(size=12))
        self.plot_search.pack(side='right', padx=(8, 0))
        self._search_after_id = None
        self.plot_search.bind('<KeyRelease>', self._on_search_key)
        mid = ctk.CTkFrame(f, fg_color='transparent')
        mid.pack(fill='both', expand=True, padx=4, pady=4)
        cat_frame = ctk.CTkFrame(mid, width=200, fg_color=self.colors['card'],
                                  border_width=1, border_color=self.colors['border'], corner_radius=10)
        cat_frame.pack(side='left', fill='y', padx=(0, 8))
        cat_frame.pack_propagate(False)
        ctk.CTkLabel(cat_frame, text="Categories", font=ctk.CTkFont(size=13, weight="bold"),
                      text_color=self.colors['text']).pack(padx=14, pady=(12, 6), anchor='w')
        self.cat_buttons = {}
        all_cats = ['All'] + list(PLOT_CATEGORIES.keys())
        for cat in all_cats:
            btn = ctk.CTkButton(cat_frame, text=cat, anchor='w', height=32,
                                font=ctk.CTkFont(size=12), corner_radius=6,
                                fg_color='transparent', text_color=self.colors['text_secondary'],
                                hover_color=self.colors['border'],
                                command=lambda c=cat: self._select_category(c))
            btn.pack(fill='x', padx=8, pady=1)
            self.cat_buttons[cat] = btn
        self._selected_cat = 'All'
        list_frame = ctk.CTkFrame(mid, fg_color=self.colors['card'], border_width=1,
                                   border_color=self.colors['border'], corner_radius=10)
        list_frame.pack(side='right', fill='both', expand=True)
        self.plot_count_label = ctk.CTkLabel(list_frame, text="",
                                               font=ctk.CTkFont(size=11),
                                               text_color=self.colors['text_secondary'])
        self.plot_count_label.pack(padx=14, pady=(8, 2), anchor='w')
        scroll_frame = ctk.CTkScrollableFrame(list_frame, fg_color='transparent',
                                                corner_radius=0)
        scroll_frame.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        self.plot_buttons_frame = scroll_frame
        self._select_category('All')
        btn_row = ctk.CTkFrame(f, fg_color='transparent')
        btn_row.pack(fill='x', padx=4, pady=(4, 0))
        ctk.CTkButton(btn_row, text="Generate Plot", height=36, corner_radius=8,
                       fg_color=self.colors['accent'], hover_color=self.colors['accent_hover'],
                       command=self._generate_selected_plot).pack(side='left', padx=(0, 8))
        ctk.CTkButton(btn_row, text="Export All Plots", height=36, corner_radius=8,
                       fg_color=self.colors['success'], hover_color='#15803d',
                       command=self._export_all_plots).pack(side='left', padx=(0, 8))
        ctk.CTkButton(btn_row, text="Batch PDF", height=36, corner_radius=8,
                       fg_color='#7c3aed', hover_color='#6d28d9',
                       command=self._batch_pdf).pack(side='left')

    def _select_category(self, cat):
        self._selected_cat = cat
        for k, btn in self.cat_buttons.items():
            if k == cat:
                btn.configure(fg_color=self.colors['accent'], text_color='white')
            else:
                btn.configure(fg_color='transparent', text_color=self.colors['text_secondary'])
        self._apply_plot_search()

    def _on_search_key(self, event=None):
        if self._search_after_id:
            self.after_cancel(self._search_after_id)
        self._search_after_id = self.after(150, self._apply_plot_search)

    def _apply_plot_search(self):
        if not hasattr(self, 'plot_buttons_frame'):
            return
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
            row = ctk.CTkFrame(self.plot_buttons_frame, fg_color='transparent', height=38)
            row.pack(fill='x', pady=1)
            row.pack_propagate(False)
            btn = ctk.CTkButton(row, text=f"  {name}", anchor='w', height=34,
                                font=ctk.CTkFont(size=12), corner_radius=6,
                                fg_color='transparent', text_color=self.colors['text'],
                                hover_color=self.colors['border'],
                                command=lambda p=pid, n=name: self._select_and_gen(p, n))
            btn.pack(fill='x')
            ctk.CTkLabel(row, text=cat, font=ctk.CTkFont(size=10),
                          text_color=self.colors['text_secondary']).pack(side='right', padx=8)

    def _select_and_gen(self, plot_id, name):
        self._selected_plot_id = plot_id
        self._generate_plot_by_id(plot_id)

    def _generate_selected_plot(self):
        if not hasattr(self, '_selected_plot_id') or self._selected_plot_id is None:
            messagebox.showinfo("Info", "Double-click a plot or select one and click Generate.")
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

    # ── Sequence Tab ──────────────────────────────────────────
    def _build_sequence_frame(self):
        f = ctk.CTkFrame(self.content, fg_color=self.colors['bg'])
        self.frames['sequence'] = f
        header = ctk.CTkFrame(f, fg_color='transparent')
        header.pack(fill='x', padx=4, pady=(8, 4))
        ctk.CTkLabel(header, text="Sequence Analysis", font=ctk.CTkFont(size=20, weight="bold"),
                      text_color=self.colors['text']).pack(side='left')
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True, padx=4, pady=4)
        left = ctk.CTkFrame(body, fg_color=self.colors['card'], border_width=1,
                             border_color=self.colors['border'], corner_radius=10)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self.all_cards.append(left)
        ctk.CTkLabel(left, text="Input Sequence (FASTA/FASTQ)", font=ctk.CTkFont(size=13, weight="bold"),
                      text_color=self.colors['text']).pack(padx=14, pady=(12, 4), anchor='w')
        self.seq_text = ctk.CTkTextbox(left, font=ctk.CTkFont(family="Consolas", size=11),
                                        height=280, fg_color=self.colors['bg'],
                                        text_color=self.colors['text'], corner_radius=8)
        self.seq_text.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=10, pady=(0, 10))
        ctk.CTkButton(btn_row, text="Load File", height=32, corner_radius=6,
                       fg_color=self.colors['accent'], command=self._load_seq_file).pack(side='left', padx=(0, 4))
        ctk.CTkButton(btn_row, text="Clear", height=32, corner_radius=6,
                       fg_color=self.colors['danger'],
                       command=lambda: self.seq_text.delete("1.0", "end")).pack(side='left')
        right = ctk.CTkFrame(body, fg_color=self.colors['card'], border_width=1,
                              border_color=self.colors['border'], corner_radius=10)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self.all_cards.append(right)
        ctk.CTkLabel(right, text="Results & Statistics", font=ctk.CTkFont(size=13, weight="bold"),
                      text_color=self.colors['text']).pack(padx=14, pady=(12, 4), anchor='w')
        self.seq_stats = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=11),
                                          height=280, fg_color=self.colors['bg'],
                                          text_color=self.colors['text'], corner_radius=8)
        self.seq_stats.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        btn_row2 = ctk.CTkFrame(right, fg_color='transparent')
        btn_row2.pack(fill='x', padx=10, pady=(0, 10))
        for label, cmd in [("GC%", self._seq_gc), ("Rev Comp", self._seq_revcomp),
                           ("Translate", self._seq_translate), ("Stats", self._seq_stats_cmd)]:
            ctk.CTkButton(btn_row2, text=label, height=32, corner_radius=6,
                           fg_color=self.colors['accent'], command=cmd).pack(side='left', padx=(0, 4))

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

    # ── Alignment Tab ──────────────────────────────────────────
    def _build_alignment_frame(self):
        f = ctk.CTkFrame(self.content, fg_color=self.colors['bg'])
        self.frames['alignment'] = f
        ctk.CTkLabel(f, text="Pairwise Sequence Alignment",
                      font=ctk.CTkFont(size=20, weight="bold"),
                      text_color=self.colors['text']).pack(padx=4, pady=(8, 4), anchor='w')
        card = ctk.CTkFrame(f, fg_color=self.colors['card'], border_width=1,
                             border_color=self.colors['border'], corner_radius=10)
        card.pack(fill='both', expand=True, padx=4, pady=4)
        self.all_cards.append(card)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=14, pady=10)
        ctk.CTkLabel(inner, text="Sequence 1", font=ctk.CTkFont(size=12, weight="bold"),
                      text_color=self.colors['text']).pack(anchor='w')
        self.align_seq1 = ctk.CTkTextbox(inner, font=ctk.CTkFont(family="Consolas", size=11),
                                          height=80, fg_color=self.colors['bg'],
                                          text_color=self.colors['text'], corner_radius=6)
        self.align_seq1.pack(fill='x', pady=(0, 6))
        ctk.CTkLabel(inner, text="Sequence 2", font=ctk.CTkFont(size=12, weight="bold"),
                      text_color=self.colors['text']).pack(anchor='w')
        self.align_seq2 = ctk.CTkTextbox(inner, font=ctk.CTkFont(family="Consolas", size=11),
                                          height=80, fg_color=self.colors['bg'],
                                          text_color=self.colors['text'], corner_radius=6)
        self.align_seq2.pack(fill='x', pady=(0, 6))
        btn_row = ctk.CTkFrame(inner, fg_color='transparent')
        btn_row.pack(fill='x', pady=(0, 6))
        ctk.CTkButton(btn_row, text="Needleman-Wunsch (Global)", height=34, corner_radius=6,
                       fg_color=self.colors['accent'], command=self._align_nw).pack(side='left', padx=(0, 8))
        ctk.CTkButton(btn_row, text="Smith-Waterman (Local)", height=34, corner_radius=6,
                       fg_color='#7c3aed', command=self._align_sw).pack(side='left')
        self.align_result = ctk.CTkTextbox(inner, font=ctk.CTkFont(family="Consolas", size=11),
                                            height=120, fg_color=self.colors['bg'],
                                            text_color=self.colors['text'], corner_radius=6)
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

    # ── Phylogeny Tab ──────────────────────────────────────────
    def _build_phylogeny_frame(self):
        f = ctk.CTkFrame(self.content, fg_color=self.colors['bg'])
        self.frames['phylogeny'] = f
        ctk.CTkLabel(f, text="Phylogenetic Tree Builder",
                      font=ctk.CTkFont(size=20, weight="bold"),
                      text_color=self.colors['text']).pack(padx=4, pady=(8, 4), anchor='w')
        card = ctk.CTkFrame(f, fg_color=self.colors['card'], border_width=1,
                             border_color=self.colors['border'], corner_radius=10)
        card.pack(fill='both', expand=True, padx=4, pady=4)
        self.all_cards.append(card)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=14, pady=10)
        ctk.CTkLabel(inner, text="Aligned Sequences (FASTA format)", font=ctk.CTkFont(size=12, weight="bold"),
                      text_color=self.colors['text']).pack(anchor='w')
        self.phylo_input = ctk.CTkTextbox(inner, font=ctk.CTkFont(family="Consolas", size=11),
                                           height=160, fg_color=self.colors['bg'],
                                           text_color=self.colors['text'], corner_radius=6)
        self.phylo_input.pack(fill='x', pady=(0, 6))
        ctk.CTkButton(inner, text="Build UPGMA Tree", height=34, corner_radius=6,
                       fg_color=self.colors['accent'], command=self._build_tree).pack(pady=(0, 6))
        self.phylo_result = ctk.CTkTextbox(inner, font=ctk.CTkFont(family="Consolas", size=11),
                                            height=140, fg_color=self.colors['bg'],
                                            text_color=self.colors['text'], corner_radius=6)
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

    # ── Expression Tab ──────────────────────────────────────────
    def _build_expression_frame(self):
        f = ctk.CTkFrame(self.content, fg_color=self.colors['bg'])
        self.frames['expression'] = f
        ctk.CTkLabel(f, text="Differential Expression",
                      font=ctk.CTkFont(size=20, weight="bold"),
                      text_color=self.colors['text']).pack(padx=4, pady=(8, 4), anchor='w')
        card = ctk.CTkFrame(f, fg_color=self.colors['card'], border_width=1,
                             border_color=self.colors['border'], corner_radius=10)
        card.pack(fill='both', expand=True, padx=4, pady=4)
        self.all_cards.append(card)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=14, pady=10)
        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 6))
        self.expr_path = ctk.CTkEntry(file_row, placeholder_text="Count matrix file (CSV/TSV)...",
                                       height=34, font=ctk.CTkFont(size=12))
        self.expr_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        ctk.CTkButton(file_row, text="Browse", height=34, width=80, corner_radius=6,
                       fg_color=self.colors['accent'],
                       command=lambda: self._browse_file(self.expr_path,
                           [("CSV/TSV", "*.csv *.tsv *.txt")])).pack(side='right')
        ctk.CTkLabel(inner, text="Conditions (comma-separated, e.g., control,control,treat,treat)",
                      font=ctk.CTkFont(size=11), text_color=self.colors['text_secondary']).pack(anchor='w')
        self.expr_conds = ctk.CTkEntry(inner, placeholder_text="control,control,treat,treat",
                                        height=34, font=ctk.CTkFont(size=12))
        self.expr_conds.pack(fill='x', pady=(2, 6))
        ctk.CTkButton(inner, text="Run Differential Expression", height=36, corner_radius=6,
                       fg_color=self.colors['accent'], command=self._run_expr).pack(pady=(0, 6))
        self.expr_result = ctk.CTkTextbox(inner, font=ctk.CTkFont(family="Consolas", size=11),
                                           height=200, fg_color=self.colors['bg'],
                                           text_color=self.colors['text'], corner_radius=6)
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

    # ── NGS Tab ──────────────────────────────────────────────
    def _build_ngs_frame(self):
        f = ctk.CTkFrame(self.content, fg_color=self.colors['bg'])
        self.frames['ngs'] = f
        ctk.CTkLabel(f, text="NGS / VCF Analysis",
                      font=ctk.CTkFont(size=20, weight="bold"),
                      text_color=self.colors['text']).pack(padx=4, pady=(8, 4), anchor='w')
        card = ctk.CTkFrame(f, fg_color=self.colors['card'], border_width=1,
                             border_color=self.colors['border'], corner_radius=10)
        card.pack(fill='both', expand=True, padx=4, pady=4)
        self.all_cards.append(card)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=14, pady=10)
        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 6))
        self.vcf_path = ctk.CTkEntry(file_row, placeholder_text="VCF file path...",
                                      height=34, font=ctk.CTkFont(size=12))
        self.vcf_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        ctk.CTkButton(file_row, text="Browse", height=34, width=80, corner_radius=6,
                       fg_color=self.colors['accent'],
                       command=lambda: self._browse_file(self.vcf_path,
                           [("VCF", "*.vcf")])).pack(side='right')
        ctk.CTkButton(inner, text="Load VCF & Show Manhattan Plot", height=36, corner_radius=6,
                       fg_color=self.colors['accent'], command=self._load_vcf).pack(pady=(0, 6))
        self.vcf_result = ctk.CTkTextbox(inner, font=ctk.CTkFont(family="Consolas", size=11),
                                          height=240, fg_color=self.colors['bg'],
                                          text_color=self.colors['text'], corner_radius=6)
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
