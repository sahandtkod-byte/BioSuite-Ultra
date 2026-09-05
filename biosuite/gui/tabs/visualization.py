"""
Visualization tabs: Plots Gallery, UpSet, Genome Browser, Conservation, Synteny, Interactive.
"""
import builtins
import re
import threading
from tkinter import filedialog

import customtkinter as ctk

from ..themes import FONT_BODY, FONT_FAMILY, PLOT_CATEGORIES, PLOT_FUNCS
from ..widgets import attach_tooltip

# Thread lock for matplotlib/builtins monkey-patching
_plot_lock = threading.Lock()


class VisualizationTabMixin:
    """Provides Plots Gallery, UpSet, Genome Browser, Conservation, Synteny, and Interactive tabs."""

    # ─── Plots Gallery ────────────────────────────────────────────────────────

    def _build_plot_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['plots'] = f

        header = ctk.CTkFrame(f, fg_color='transparent')
        header.pack(fill='x', pady=(0, 8))
        self._label(header, 'Plots Gallery', 'title').pack(side='left')
        search_box = ctk.CTkFrame(header, fg_color='transparent')
        search_box.pack(side='right')
        self.plot_search = self._input_entry(search_box, "Search plots...", width=260)
        self.plot_search.pack(side='left')
        clear_btn = ctk.CTkButton(search_box, text="✕", width=30, height=36, corner_radius=8,
                                  font=(FONT_FAMILY, 12), fg_color=T['card'],
                                  hover_color=T['border'], text_color=T['text_dim'],
                                  command=self._clear_plot_search)
        clear_btn.pack(side='left', padx=(6, 0))
        attach_tooltip(clear_btn, "Clear search (or press Esc in the box)", T)
        self._search_after_id = None
        self.plot_search.bind('<KeyRelease>', self._on_search_key)
        self.plot_search.bind('<Escape>', lambda e: self._clear_plot_search())

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
        self._scroll_canvas = scroll_frame._parent_canvas if hasattr(scroll_frame, '_parent_canvas') else None
        self.plot_buttons_frame = scroll_frame
        self._select_category('All')

        btn_row = ctk.CTkFrame(f, fg_color='transparent')
        btn_row.pack(fill='x')
        b1 = self._action_button(btn_row, "Generate Plot", self._generate_selected_plot)
        b1.pack(side='left', padx=(0, 8))
        attach_tooltip(b1, "Render the selected plot in an interactive viewer (zoom/pan/save)", T)
        b2 = self._action_button(btn_row, "Export All", self._export_all_plots, 'success')
        b2.pack(side='left', padx=(0, 8))
        attach_tooltip(b2, "Export every plot type to a folder as PNG files", T)
        b3 = self._action_button(btn_row, "Batch PDF", self._batch_pdf, 'accent_dim')
        b3.pack(side='left')
        attach_tooltip(b3, "Bundle all plots into a single multi-page PDF", T)

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

    def _clear_plot_search(self):
        """Clear the gallery search box and refresh the list."""
        try:
            self.plot_search.delete(0, 'end')
        except Exception:
            pass
        if self._search_after_id:
            try:
                self.after_cancel(self._search_after_id)
            except Exception:
                pass
            self._search_after_id = None
        self._apply_plot_search()

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
            row = ctk.CTkFrame(self.plot_buttons_frame, fg_color='transparent', height=32)
            row.pack(fill='x', pady=1)
            row.pack_propagate(False)
            btn = ctk.CTkButton(row, text=f"  {name}", anchor='w', height=30, corner_radius=6,
                                font=(FONT_FAMILY, 12), fg_color='transparent', text_color=T['text'],
                                hover_color=T['border'],
                                command=lambda p=pid, n=name: self._select_and_gen(p, n))
            btn.pack(side='left', fill='x', expand=True)
            ctk.CTkLabel(row, text=cat, font=(FONT_FAMILY, 9),
                          text_color=T['text_muted'], width=180, anchor='e').pack(side='right', padx=8)

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
        self._show_progress(f"Generating {plot_id}...")
        def run():
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for thread safety
            import matplotlib.pyplot as plt
            # Monkey-patch plt.show to be a no-op in background thread
            _original_show = plt.show
            plt.show = lambda *a, **k: None
            original_input = builtins.input
            builtins.input = self._gui_input
            try:
                func = PLOT_FUNCS.get(plot_id)
                if func:
                    func()
                    # Hand the live figure to the main thread so it can be
                    # embedded with real zoom/pan/save (no static PNG loss).
                    if plt.get_fignums():
                        fig = plt.gcf()
                        name = self._plot_name_for(plot_id)
                        self.after(0, lambda f=fig, n=name: self._show_plot_from_figure(f, n))
                else:
                    self.after(0, lambda: self._msg_error("Error", f"Plot '{plot_id}' not found."))
            except Exception as e:
                self.after(0, lambda e=e: self._msg_error("Plot Error", str(e)))
            finally:
                builtins.input = original_input
                plt.show = _original_show
                self.after(0, self._hide_progress)
                self.after(0, lambda: self._set_status("Ready"))
        threading.Thread(target=run, daemon=True).start()

    def _plot_name_for(self, plot_id):
        """Human-readable name for a plot id (for window title/history)."""
        for _cat, plots in PLOT_CATEGORIES.items():
            for name, pid in plots:
                if pid == plot_id:
                    return name
        return plot_id.replace('_', ' ').title()

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
                    self._export_all_to_folder(folder)
                    self.after(0, lambda: self._msg_success("Done", f"All plots exported to:\n{folder}"))
                except Exception as e:
                    self.after(0, lambda e=e: self._msg_error("Export Error", str(e)))
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
                    self._batch_export_to_pdf(path)
                    self.after(0, lambda: self._msg_success("Done", f"PDF saved:\n{path}"))
                except Exception as e:
                    self.after(0, lambda e=e: self._msg_error("PDF Error", str(e)))
                finally:
                    self.after(0, lambda: self._set_status("Ready"))
            threading.Thread(target=run, daemon=True).start()

    # ─── UpSet Plots Tab ─────────────────────────────────────────────────────

    def _build_upset_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['upset'] = f
        self._section_header(f, "UpSet Plot — Multi-Set Intersections")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Enter sets (name:elem1,elem2,...)', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.upset_text = self._text_box(left, height=280)
        self.upset_text.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        self.upset_text.insert("1.0", "Genes_A:BRCA1,TP53,MYC,EGFR\nGenes_B:TP53,MYC,PTEN,RB1\nGenes_C:BRCA1,PTEN,APC,VHL")
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=10, pady=(0, 10))
        self._action_button(btn_row, "Generate UpSet", self._run_upset).pack(side='left', padx=(0, 6))

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Statistics', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.upset_stats = self._text_box(right, height=280)
        self.upset_stats.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    def _run_upset(self):
        import matplotlib.pyplot as plt

        from ...plotting.upset_plots import compute_set_statistics, plot_upset
        text = self.upset_text.get("1.0", "end").strip()
        if not text:
            self._msg_warning("No data", "Enter set data first.")
            return
        sets_dict = {}
        for line in text.split('\n'):
            line = line.strip()
            if ':' in line:
                name, elems = line.split(':', 1)
                sets_dict[name.strip()] = set(e.strip() for e in elems.split(','))
        if not sets_dict:
            self._msg_warning("No data", "Invalid format. Use Name:elem1,elem2")
            return
        try:
            stats = compute_set_statistics(sets_dict)
            self.upset_stats.delete("1.0", "end")
            self.upset_stats.insert("end", f"Set sizes: {stats['sizes']}\n")
            self.upset_stats.insert("end", f"Union: {stats['total_union']}\n")
            self.upset_stats.insert("end", f"Intersection: {stats['total_intersection']}\n")
            self.upset_stats.insert("end", f"Unique per set: {stats['unique_per_set']}\n")
            fig = plot_upset(sets_dict, title="UpSet Plot")
            if fig is not None:
                self.after(0, lambda f=fig: self._show_plot_from_figure(f, "UpSet Plot"))
            plt.close()
        except Exception as e:
            self._msg_error("Error", str(e))

    def _gui_upset(self):
        self._show_frame('upset')
        self._run_upset()

    # ─── Genome Browser Tab ──────────────────────────────────────────────────

    def _build_genomebrowser_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['genomebrowser'] = f
        self._section_header(f, "Genome Browser — Track Viewer")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Add tracks (BED/VCF/BAM files)', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.gb_tracks_text = self._text_box(left, height=200)
        self.gb_tracks_text.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=10, pady=(0, 10))
        self._action_button(btn_row, "Add BED", self._gb_add_bed).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Add VCF", self._gb_add_vcf).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "View Tracks", self._gb_view).pack(side='left', padx=(0, 6))

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Track Info', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.gb_info = self._text_box(right, height=280)
        self.gb_info.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    def _gb_add_bed(self):
        path = filedialog.askopenfilename(filetypes=[("BED", "*.bed"), ("All", "*.*")])
        if path:
            self.gb_tracks_text.insert("end", f"bed:{path}\n")

    def _gb_add_vcf(self):
        path = filedialog.askopenfilename(filetypes=[("VCF", "*.vcf"), ("All", "*.*")])
        if path:
            self.gb_tracks_text.insert("end", f"vcf:{path}\n")

    def _gb_view(self):
        from ...plotting.genome_browser import (
            create_bed_track,
            create_variant_track,
            plot_genome_tracks,
        )
        text = self.gb_tracks_text.get("1.0", "end").strip()
        if not text:
            self._msg_warning("No tracks", "Add BED or VCF files first.")
            return
        tracks = []
        for line in text.split('\n'):
            line = line.strip()
            if ':' not in line:
                continue
            ttype, path = line.split(':', 1)
            try:
                if ttype == 'bed':
                    tracks.append(create_bed_track(path))
                elif ttype == 'vcf':
                    tracks.append(create_variant_track(path))
            except Exception as e:
                self.gb_info.insert("end", f"Error loading {path}: {e}\n")
        if tracks:
            fig = plot_genome_tracks(tracks, title="Genome Browser")
            self.after(0, lambda f=fig: self._show_plot_from_figure(f, "Genome Browser"))
            self.gb_info.delete("1.0", "end")
            self.gb_info.insert("end", f"Loaded {len(tracks)} tracks\n")
        else:
            self._msg_warning("No tracks", "No valid tracks found.")

    def _gui_genome_browser(self):
        self._show_frame('genomebrowser')

    # ─── Conservation / Sequence Logo Tab ────────────────────────────────────

    def _build_conservation_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['conservation'] = f
        self._section_header(f, "Sequence Logo & Conservation")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Enter aligned sequences (one per line)', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.cons_text = self._text_box(left, height=280)
        self.cons_text.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        self.cons_text.insert("1.0", "ACGTACGTACGT\nACGAACGTACGT\nACGTACGTACGA\nACGTACGTACGT")
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=10, pady=(0, 10))
        self._action_button(btn_row, "Logo + Conservation", self._run_cons).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Motif Enrichment", self._run_motif).pack(side='left')

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Conservation Scores', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.cons_stats = self._text_box(right, height=280)
        self.cons_stats.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    def _run_cons(self):
        from ...plotting.conservation_plots import (
            compute_conservation_scores,
            plot_logo_with_conservation,
        )
        text = self.cons_text.get("1.0", "end").strip()
        if not text:
            self._msg_warning("No sequences", "Enter aligned sequences first.")
            return
        seqs = [l.strip().upper() for l in text.split('\n') if l.strip()]
        try:
            scores = compute_conservation_scores(seqs)
            self.cons_stats.delete("1.0", "end")
            for pos, score in scores:
                bar = '#' * int(score * 30)
                self.cons_stats.insert("end", f"Pos {pos:2d}: {score:.3f} {bar}\n")
            fig = plot_logo_with_conservation(seqs)
            self.after(0, lambda f=fig: self._show_plot_from_figure(f, "Conservation Analysis"))
        except Exception as e:
            self._msg_error("Error", str(e))

    def _run_motif(self):
        from ...plotting.conservation_plots import plot_motif_enrichment
        text = self.cons_text.get("1.0", "end").strip()
        motifs = self._ask_input("Motifs", "Enter motifs (comma-sep):", "ATG,CG,GCG")
        if not motifs:
            return
        seqs = [l.strip().upper() for l in text.split('\n') if l.strip()]
        motif_list = [m.strip() for m in motifs.split(',')]
        try:
            fig = plot_motif_enrichment(seqs, motif_list)
            self.after(0, lambda f=fig: self._show_plot_from_figure(f, "Motif Enrichment"))
        except Exception as e:
            self._msg_error("Error", str(e))

    def _gui_seq_logo(self):
        self._show_frame('conservation')
        self._run_cons()

    def _gui_conservation_bar(self):
        self._show_frame('conservation')
        self._run_cons()

    # ─── Synteny Tab ─────────────────────────────────────────────────────────

    def _build_synteny_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['syntenytabs'] = f
        self._section_header(f, "Synteny Analysis — Dotplot & Gene Order")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Gene orders (comma-sep per genome)', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self._label(left, 'Genome 1:', 'body').pack(padx=14, anchor='w')
        self.syn_g1 = self._input_entry(left, "GeneA,GeneB,GeneC,GeneD")
        self.syn_g1.pack(fill='x', padx=14, pady=(0, 8))
        self.syn_g1.insert(0, "GeneA,GeneB,GeneC,GeneD,GeneE")
        self._label(left, 'Genome 2:', 'body').pack(padx=14, anchor='w')
        self.syn_g2 = self._input_entry(left, "GeneA,GeneC,GeneB,GeneE,GeneD")
        self.syn_g2.pack(fill='x', padx=14, pady=(0, 8))
        self.syn_g2.insert(0, "GeneA,GeneC,GeneB,GeneE,GeneD")
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=14, pady=(0, 10))
        self._action_button(btn_row, "Synteny Dotplot", self._run_synteny).pack(side='left', padx=(0, 6))

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Results', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.syn_stats = self._text_box(right, height=280)
        self.syn_stats.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    def _run_synteny(self):
        from ...plotting.synteny import compute_synteny_score, plot_synteny_dotplot
        g1 = [g.strip() for g in self.syn_g1.get().split(',') if g.strip()]
        g2 = [g.strip() for g in self.syn_g2.get().split(',') if g.strip()]
        if not g1 or not g2:
            self._msg_warning("No genes", "Enter gene orders for both genomes.")
            return
        try:
            score, pairs = compute_synteny_score(g1, g2)
            self.syn_stats.delete("1.0", "end")
            self.syn_stats.insert("end", f"Synteny score: {score:.3f}\n")
            self.syn_stats.insert("end", f"Genome 1: {len(g1)} genes\n")
            self.syn_stats.insert("end", f"Genome 2: {len(g2)} genes\n")
            self.syn_stats.insert("end", f"Common: {len(set(g1) & set(g2))}\n")
            fig = plot_synteny_dotplot(g1, g2, title="Synteny Dotplot")
            self.after(0, lambda f=fig: self._show_plot_from_figure(f, "Synteny Dotplot"))
        except Exception as e:
            self._msg_error("Error", str(e))

    def _gui_synteny(self):
        self._show_frame('syntenytabs')
        self._run_synteny()

    # ─── Interactive Plots Tab ───────────────────────────────────────────────

    def _build_interactive_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['interactive'] = f
        self._section_header(f, "Interactive Plots (Plotly)")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Plot Type', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.inter_type = ctk.CTkComboBox(left, values=['Scatter', 'Bar', 'Heatmap', 'Volcano', 'Line', 'Pie',
                                                          'PCA', 'Manhattan', 'Boxplot', 'Violin', 'QQ-Plot'],
                                            height=36, font=FONT_BODY, corner_radius=8,
                                            fg_color=T['input_bg'], border_color=T['border'],
                                            button_color=T['accent'], button_hover_color=T['accent_dim'],
                                            dropdown_fg_color=T['card'], dropdown_hover_color=T['border'],
                                            dropdown_text_color=T['text'], text_color=T['text'])
        self.inter_type.pack(fill='x', padx=14, pady=(0, 8))
        self.inter_type.set('Scatter')

        self._label(left, 'Data (CSV file or use demo)', 'body').pack(padx=14, anchor='w')
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=14, pady=(8, 10))
        self._action_button(btn_row, "Load CSV", self._inter_load_csv).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Generate", self._inter_generate).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Demo Data", self._inter_demo).pack(side='left')

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Info', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.inter_info = self._text_box(right, height=280)
        self.inter_info.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        self.inter_info.insert("1.0", "Interactive plots are exported as HTML files.\nLoad a CSV or click Demo Data to start.\n\nNew: PCA, Manhattan, Boxplot, Violin, QQ-Plot from plot_api!")
        self._inter_data = None

    def _inter_load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            import pandas as pd
            df = pd.read_csv(path)
            self._inter_data = df
            self.inter_info.delete("1.0", "end")
            self.inter_info.insert("end", f"Loaded: {path}\nColumns: {list(df.columns)}\nShape: {df.shape}\n")

    def _inter_demo(self):
        import numpy as np
        np.random.seed(42)
        n = 100
        self._inter_data = {'x': np.random.randn(n), 'y': np.random.randn(n),
                            'label': np.random.choice(['A', 'B', 'C'], n)}
        self.inter_info.delete("1.0", "end")
        self.inter_info.insert("end", "Demo data loaded: 100 points, 3 groups\n")

    def _inter_generate(self):
        import numpy as np
        plot_type = self.inter_type.get().lower()
        if self._inter_data is None:
            self._msg_warning("No data", "Load CSV or use Demo Data first.")
            return
        try:
            output = "interactive_output.html"

            # Try new plot_api first for new plot types
            if plot_type in ('pca', 'manhattan', 'boxplot', 'violin', 'qq-plot'):
                from ...plotting.plot_api import boxplot, manhattan, pca, qqplot, violin
                if plot_type == 'pca':
                    data = np.random.randn(30, 50) if not isinstance(self._inter_data, dict) else self._inter_data.get('x', np.random.randn(30, 50))
                    pca(data, labels=['Ctrl']*15 + ['Treat']*15, interactive=True, output_html=output)
                elif plot_type == 'manhattan':
                    chroms = np.random.choice(['chr1', 'chr2', 'chr3'], 200)
                    positions = np.random.randint(1, 1000000, 200)
                    pvals = np.random.uniform(0, 1, 200)
                    manhattan(chroms, positions, pvals, interactive=True, output_html=output)
                elif plot_type == 'boxplot':
                    data = {'Ctrl': np.random.randn(30).tolist(), 'Treat': (np.random.randn(30) + 1).tolist()}
                    boxplot(data, interactive=True, output_html=output)
                elif plot_type == 'violin':
                    data = {'Ctrl': np.random.randn(30).tolist(), 'Treat': (np.random.randn(30) + 1).tolist()}
                    violin(data, interactive=True, output_html=output)
                elif plot_type == 'qq-plot':
                    pvals = np.random.uniform(0, 1, 100)
                    qqplot(pvals, interactive=True, output_html=output)
            else:
                # Use existing interactive_plots for legacy types
                from ...plotting.interactive_plots import (
                    interactive_bar,
                    interactive_heatmap,
                    interactive_line,
                    interactive_pie,
                    interactive_scatter,
                    interactive_volcano,
                )
                if plot_type == 'scatter':
                    if isinstance(self._inter_data, dict):
                        x, y = self._inter_data['x'], self._inter_data['y']
                        labels = self._inter_data.get('label')
                        interactive_scatter(x, y, color_col=labels, output_html=output)
                    else:
                        cols = list(self._inter_data.select_dtypes(include=[np.number]).columns)
                        if len(cols) >= 2:
                            interactive_scatter(self._inter_data[cols[0]].values,
                                                      self._inter_data[cols[1]].values, output_html=output)
                elif plot_type == 'bar':
                    if isinstance(self._inter_data, dict):
                        vals = list(self._inter_data.get('y', [1, 2, 3]))
                        cats = list(self._inter_data.get('label', ['A', 'B', 'C']))
                        interactive_bar(cats[:len(vals)], vals[:len(cats)], output_html=output)
                    else:
                        cols = list(self._inter_data.columns)
                        interactive_bar(self._inter_data[cols[0]].astype(str).tolist(),
                                              self._inter_data[cols[1]].tolist(), output_html=output)
                elif plot_type == 'heatmap':
                    if isinstance(self._inter_data, dict):
                        import pandas as pd
                        df_heat = pd.DataFrame(self._inter_data)
                    else:
                        df_heat = self._inter_data
                    num = df_heat.select_dtypes(include=[np.number])
                    interactive_heatmap(num.values[:10, :10], output_html=output)
                elif plot_type == 'volcano':
                    interactive_volcano(np.random.randn(200), np.random.uniform(0, 1, 200), output_html=output)
                elif plot_type == 'line':
                    x = list(range(20))
                    ys = [np.sin(np.array(x)), np.cos(np.array(x))]
                    interactive_line(x, ys, names=['sin', 'cos'], output_html=output)
                elif plot_type == 'pie':
                    interactive_pie(['A', 'B', 'C', 'D'], [30, 25, 20, 25], output_html=output)
                else:
                    interactive_scatter(np.random.randn(50), np.random.randn(50), output_html=output)

            self.inter_info.delete("1.0", "end")
            self.inter_info.insert("end", f"Saved: {output}\nOpen in a browser to interact.\n\nPlot type: {plot_type}")
            self._msg_success("Done", f"Interactive plot saved to:\n{output}")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _gui_interactive_scatter(self):
        self._show_frame('interactive')
        self.inter_type.set('Scatter')
        self._inter_demo()
        self._inter_generate()

    def _gui_interactive_bar(self):
        self._show_frame('interactive')
        self.inter_type.set('Bar')
        self._inter_demo()
        self._inter_generate()

    def _gui_interactive_heatmap(self):
        self._show_frame('interactive')
        self.inter_type.set('Heatmap')
        self._inter_demo()
        self._inter_generate()

    def _gui_interactive_volcano(self):
        self._show_frame('interactive')
        self.inter_type.set('Volcano')
        self._inter_demo()
        self._inter_generate()

    def _gui_interactive_line(self):
        self._show_frame('interactive')
        self.inter_type.set('Line')
        self._inter_demo()
        self._inter_generate()

    def _gui_interactive_pie(self):
        self._show_frame('interactive')
        self.inter_type.set('Pie')
        self._inter_demo()
        self._inter_generate()
