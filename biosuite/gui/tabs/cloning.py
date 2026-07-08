"""
Molecular Cloning tab: Restriction Digestion, PCR, Virtual Gel, Ligation.
"""
import os
import customtkinter as ctk
from tkinter import filedialog

from ..themes import FONT_FAMILY, FONT_SMALL, FONT_MONO


class CloningTabMixin:
    """Provides the Molecular Cloning tab with digestion, PCR, and virtual gel."""

    def _build_cloning_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['cloning'] = f
        self._section_header(f, "Molecular Cloning Toolkit")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        # ── Restriction Digestion ──
        self._label(inner, "Restriction Digestion", "sub").pack(anchor='w', pady=(0, 6))

        seq_row = ctk.CTkFrame(inner, fg_color='transparent')
        seq_row.pack(fill='x', pady=(0, 6))
        self.cloning_seq = self._input_entry(seq_row, "DNA sequence (or load from file)...")
        self.cloning_seq.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(seq_row, "Load File", self._cloning_load_file).pack(side='right')

        enzyme_row = ctk.CTkFrame(inner, fg_color='transparent')
        enzyme_row.pack(fill='x', pady=(0, 6))
        self._label(enzyme_row, "Enzyme:", "body").pack(side='left', padx=(0, 8))
        self.cloning_enzyme = ctk.CTkComboBox(
            enzyme_row, values=[
                "EcoRI", "BamHI", "HindIII", "NotI", "XhoI", "SacI",
                "KpnI", "SmaI", "XbaI", "PstI", "NcoI", "SpeI",
                "ApaI", "SalI", "SphI", "ClaI", "NheI", "MluI"
            ], width=140, height=32, font=(FONT_FAMILY, 11),
            fg_color=T['input_bg'], border_color=T['border'],
            button_color=T['accent'], button_hover_color=T['accent_dim'],
            dropdown_fg_color=T['card'], dropdown_hover_color=T['border'],
            dropdown_text_color=T['text'], text_color=T['text'])
        self.cloning_enzyme.pack(side='left', padx=(0, 8))
        self.cloning_enzyme.set("EcoRI")

        self._label(enzyme_row, "Topology:", "body").pack(side='left', padx=(0, 8))
        self.cloning_topo = ctk.CTkComboBox(
            enzyme_row, values=["circular", "linear"], width=100, height=32,
            font=(FONT_FAMILY, 11), fg_color=T['input_bg'], border_color=T['border'],
            button_color=T['accent'], button_hover_color=T['accent_dim'],
            dropdown_fg_color=T['card'], dropdown_hover_color=T['border'],
            dropdown_text_color=T['text'], text_color=T['text'])
        self.cloning_topo.pack(side='left')
        self.cloning_topo.set("circular")

        btn_row = ctk.CTkFrame(inner, fg_color='transparent')
        btn_row.pack(fill='x', pady=(0, 8))
        self._action_button(btn_row, "Run Digestion", self._run_digestion).pack(side='left', padx=(0, 8))
        self._action_button(btn_row, "Virtual Gel", self._run_virtual_gel, color_key='success').pack(side='left')

        # ── PCR Simulation ──
        ctk.CTkFrame(inner, height=1, fg_color=T['border']).pack(fill='x', pady=(8, 12))
        self._label(inner, "PCR Simulation", "sub").pack(anchor='w', pady=(0, 6))

        pcr_row = ctk.CTkFrame(inner, fg_color='transparent')
        pcr_row.pack(fill='x', pady=(0, 6))
        self.pcr_fwd = self._input_entry(pcr_row, "Forward primer...")
        self.pcr_fwd.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self.pcr_rev = self._input_entry(pcr_row, "Reverse primer...")
        self.pcr_rev.pack(side='left', fill='x', expand=True)

        pcr_btn = ctk.CTkFrame(inner, fg_color='transparent')
        pcr_btn.pack(fill='x', pady=(0, 8))
        self._action_button(pcr_btn, "Run PCR", self._run_pcr).pack(side='left')

        # ── Primer Design ──
        ctk.CTkFrame(inner, height=1, fg_color=T['border']).pack(fill='x', pady=(8, 12))
        self._label(inner, "Primer Design", "sub").pack(anchor='w', pady=(0, 6))

        primer_row = ctk.CTkFrame(inner, fg_color='transparent')
        primer_row.pack(fill='x', pady=(0, 6))
        self.primer_seq = self._input_entry(primer_row, "Target sequence for primer design...")
        self.primer_seq.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(primer_row, "Design Primers", self._design_primers).pack(side='right')

        # ── Results ──
        ctk.CTkFrame(inner, height=1, fg_color=T['border']).pack(fill='x', pady=(8, 12))
        self._label(inner, "Results", "sub").pack(anchor='w', pady=(0, 6))
        self.cloning_result = self._text_box(inner, height=180)
        self.cloning_result.pack(fill='both', expand=True)

    def _cloning_load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("FASTA", "*.fasta *.fa"), ("GenBank", "*.gb *.genbank"), ("All", "*.*")])
        if path:
            from ..core.sequence import read_fasta
            seqs = read_fasta(path)
            if seqs:
                self.cloning_seq.delete(0, 'end')
                self.cloning_seq.insert(0, seqs[0][1][:200])
                self._set_status(f"Loaded: {os.path.basename(path)}")

    def _run_digestion(self):
        seq = self.cloning_seq.get().strip()
        if not seq:
            self._msg_warning("Input Required", "Please enter or load a DNA sequence.")
            return
        enzyme = self.cloning_enzyme.get()
        topo = self.cloning_topo.get()
        try:
            from ..core.cloning import simulate_digestion, format_digest_report
            result = simulate_digestion(seq, enzyme, topology=topo)
            report = format_digest_report(result)
            self.cloning_result.delete("1.0", "end")
            self.cloning_result.insert("1.0", report)
            self._set_status(f"Digestion complete: {len(result['fragments'])} fragments")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _run_virtual_gel(self):
        seq = self.cloning_seq.get().strip()
        if not seq:
            self._msg_warning("Input Required", "Please enter or load a DNA sequence.")
            return
        enzyme = self.cloning_enzyme.get()
        try:
            from ..core.cloning import simulate_digestion, plot_virtual_gel
            result = simulate_digestion(seq, enzyme)
            fig = plot_virtual_gel(result['fragments'])
            self._record_plot(fig, f"Virtual Gel ({enzyme})")
            self._show_plot_figure(fig)
            self._set_status("Virtual gel displayed")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _run_pcr(self):
        seq = self.cloning_seq.get().strip()
        fwd = self.pcr_fwd.get().strip()
        rev = self.pcr_rev.get().strip()
        if not all([seq, fwd, rev]):
            self._msg_warning("Input Required", "Please enter template and both primers.")
            return
        try:
            from ..core.cloning import simulate_pcr
            result = simulate_pcr(seq, fwd, rev)
            self.cloning_result.delete("1.0", "end")
            self.cloning_result.insert("1.0",
                f"PCR Product Size: {result['size']} bp\n"
                f"Cycles: {result['cycles']}\n\n"
                f"Product sequence (first 100 bp):\n{result['product'][:100]}...")
            self._set_status(f"PCR complete: {result['size']} bp product")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _design_primers(self):
        seq = self.primer_seq.get().strip()
        if not seq:
            self._msg_warning("Input Required", "Please enter a target sequence.")
            return
        try:
            from ..core.cloning import design_primers, format_primer_report
            primers = design_primers(seq)
            report = format_primer_report(primers)
            self.cloning_result.delete("1.0", "end")
            self.cloning_result.insert("1.0", report)
            self._set_status("Primers designed successfully")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _show_plot_figure(self, fig):
        """Display a matplotlib figure in a popup window."""
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        win = ctk.CTkToplevel(self)
        win.title("Plot")
        win.geometry("800x600")
        win.configure(fg_color=self.T['bg'])

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        btn = ctk.CTkButton(win, text="Close", command=win.destroy,
                           fg_color=self.T['accent'], text_color='#000000')
        btn.pack(pady=8)
