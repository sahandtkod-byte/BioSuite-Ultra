"""
Visualization tabs: Plots Gallery, UpSet, Genome Browser, Conservation, Synteny, Interactive.
"""
import os
import re
import builtins
import threading
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, ttk

from ..themes import (THEMES, PLOT_CATEGORIES, PLOT_FUNCS, FONT_FAMILY, FONT_BODY,
                       FONT_SMALL, FONT_BUTTON, FONT_HEADING)

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
        self._scroll_canvas = scroll_frame._parent_canvas if hasattr(scroll_frame, '_parent_canvas') else None
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
                    # Save last figure to temp file and display on main thread
                    if plt.get_fignums():
                        fig = plt.gcf()
                        import tempfile, os
                        tmp = os.path.join(tempfile.gettempdir(), f"biosuite_plot_{plot_id}.png")
                        fig.savefig(tmp, dpi=150, bbox_inches='tight')
                        plt.close('all')
                        self.after(0, lambda p=tmp: self._show_plot_image(p))
                else:
                    self.after(0, lambda: self._msg_error("Error", f"Plot '{plot_id}' not found."))
            except Exception as e:
                self.after(0, lambda: self._msg_error("Plot Error", str(e)))
            finally:
                builtins.input = original_input
                plt.show = _original_show
                self.after(0, lambda: self._set_status("Ready"))
        threading.Thread(target=run, daemon=True).start()

    def _show_plot_image(self, path):
        """Display a saved plot image in a new window with Save As option."""
        from tkinter import filedialog
        from PIL import Image, ImageTk
        win = tk.Toplevel(self)
        win.title("Plot Result")
        win.geometry("950x750")
        win.configure(fg_color=self.T.get('bg', '#0a0f0a'))
        photo_ref = [None]  # prevent GC
        
        try:
            img = Image.open(path)
            img.thumbnail((900, 650), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            photo_ref[0] = photo
            label = ctk.CTkLabel(win, image=photo, text="")
            label.pack(fill='both', expand=True, padx=10, pady=10)
        except Exception as e:
            ctk.CTkLabel(win, text=f"Error loading plot: {e}").pack(pady=20)
        
        # Bottom buttons frame
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
                shutil.copy2(path, ext)
                self._set_status(f"Plot saved to: {ext}")
        
        def on_close():
            photo_ref[0] = None
            win.destroy()
        
        ctk.CTkButton(btn_frame, text="💾 Save As...", command=save_as,
                      fg_color=self.T.get('accent', '#00ff88'),
                      text_color='#000000', width=120).pack(side='left', padx=5)
        ctk.CTkButton(btn_frame, text="Close", command=on_close,
                      fg_color='#ff4444', text_color='#ffffff', width=80).pack(side='right', padx=5)
        
        win.protocol("WM_DELETE_WINDOW", on_close)

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
                    self._batch_export_to_pdf(path)
                    self.after(0, lambda: self._msg_success("Done", f"PDF saved:\n{path}"))
                except Exception as e:
                    self.after(0, lambda: self._msg_error("PDF Error", str(e)))
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
        from ...plotting.upset_plots import plot_upset, compute_set_statistics
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
        import matplotlib.pyplot as plt
        from ...plotting.genome_browser import plot_genome_tracks, create_bed_track, create_variant_track
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
        import matplotlib.pyplot as plt
        from ...plotting.conservation_plots import plot_logo_with_conservation, compute_conservation_scores
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
        import matplotlib.pyplot as plt
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
        import matplotlib.pyplot as plt
        from ...plotting.synteny import plot_synteny_dotplot, compute_synteny_score
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
                from ...plotting.plot_api import pca, manhattan, boxplot, violin, qqplot
                if plot_type == 'pca':
                    data = np.random.randn(30, 50) if not isinstance(self._inter_data, dict) else self._inter_data.get('x', np.random.randn(30, 50))
                    fig = pca(data, labels=['Ctrl']*15 + ['Treat']*15, interactive=True, output_html=output)
                elif plot_type == 'manhattan':
                    chroms = np.random.choice(['chr1', 'chr2', 'chr3'], 200)
                    positions = np.random.randint(1, 1000000, 200)
                    pvals = np.random.uniform(0, 1, 200)
                    fig = manhattan(chroms, positions, pvals, interactive=True, output_html=output)
                elif plot_type == 'boxplot':
                    data = {'Ctrl': np.random.randn(30).tolist(), 'Treat': (np.random.randn(30) + 1).tolist()}
                    fig = boxplot(data, interactive=True, output_html=output)
                elif plot_type == 'violin':
                    data = {'Ctrl': np.random.randn(30).tolist(), 'Treat': (np.random.randn(30) + 1).tolist()}
                    fig = violin(data, interactive=True, output_html=output)
                elif plot_type == 'qq-plot':
                    pvals = np.random.uniform(0, 1, 100)
                    fig = qqplot(pvals, interactive=True, output_html=output)
            else:
                # Use existing interactive_plots for legacy types
                from ...plotting.interactive_plots import (interactive_scatter, interactive_bar,
                    interactive_heatmap, interactive_volcano, interactive_line, interactive_pie)
                if plot_type == 'scatter':
                    if isinstance(self._inter_data, dict):
                        x, y = self._inter_data['x'], self._inter_data['y']
                        labels = self._inter_data.get('label')
                        fig = interactive_scatter(x, y, color_col=labels, output_html=output)
                    else:
                        cols = list(self._inter_data.select_dtypes(include=[np.number]).columns)
                        if len(cols) >= 2:
                            fig = interactive_scatter(self._inter_data[cols[0]].values,
                                                      self._inter_data[cols[1]].values, output_html=output)
                elif plot_type == 'bar':
                    if isinstance(self._inter_data, dict):
                        vals = list(self._inter_data.get('y', [1, 2, 3]))
                        cats = list(self._inter_data.get('label', ['A', 'B', 'C']))
                        fig = interactive_bar(cats[:len(vals)], vals[:len(cats)], output_html=output)
                    else:
                        cols = list(self._inter_data.columns)
                        fig = interactive_bar(self._inter_data[cols[0]].astype(str).tolist(),
                                              self._inter_data[cols[1]].tolist(), output_html=output)
                elif plot_type == 'heatmap':
                    if isinstance(self._inter_data, dict):
                        import pandas as pd
                        df_heat = pd.DataFrame(self._inter_data)
                    else:
                        df_heat = self._inter_data
                    num = df_heat.select_dtypes(include=[np.number])
                    fig = interactive_heatmap(num.values[:10, :10], output_html=output)
                elif plot_type == 'volcano':
                    fig = interactive_volcano(np.random.randn(200), np.random.uniform(0, 1, 200), output_html=output)
                elif plot_type == 'line':
                    x = list(range(20))
                    ys = [np.sin(np.array(x)), np.cos(np.array(x))]
                    fig = interactive_line(x, ys, names=['sin', 'cos'], output_html=output)
                elif plot_type == 'pie':
                    fig = interactive_pie(['A', 'B', 'C', 'D'], [30, 25, 20, 25], output_html=output)
                else:
                    fig = interactive_scatter(np.random.randn(50), np.random.randn(50), output_html=output)

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
