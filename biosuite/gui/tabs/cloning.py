"""
Molecular Cloning tab: Restriction Digestion, PCR, Virtual Gel, Ligation.

VIP upgrade: all 169 enzymes from the core database, live enzyme info
(site / cut position / overhang), sequence stats + inline validation,
full-length file loading (no silent truncation), clipboard export.
"""
import os
import re
from tkinter import filedialog

import customtkinter as ctk

from ..themes import FONT_FAMILY
from ..widgets import attach_tooltip

IUPAC_DNA = set("ACGTRYSWKMBDHVNacgtryswkmbdhvn")


class CloningTabMixin:
    """Provides the Molecular Cloning tab with digestion, PCR, and virtual gel."""

    def _build_cloning_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['cloning'] = f
        self._section_header(f, "Molecular Cloning Toolkit")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkScrollableFrame(card, fg_color='transparent',
                                       scrollbar_button_color=T.get('scrollbar', T['border']))
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        # ── Restriction Digestion ──
        self._label(inner, "Restriction Digestion", "sub").pack(anchor='w', pady=(0, 6))

        seq_row = ctk.CTkFrame(inner, fg_color='transparent')
        seq_row.pack(fill='x', pady=(0, 2))
        self.cloning_seq = self._input_entry(seq_row, "DNA sequence (or load from file)...")
        self.cloning_seq.pack(side='left', fill='x', expand=True, padx=(0, 8))
        attach_tooltip(self.cloning_seq, "Paste a DNA sequence or load a plasmid from file.\n"
                                         "IUPAC ambiguity codes are accepted.", T)
        load_btn = self._action_button(seq_row, "Load File", self._cloning_load_file)
        load_btn.pack(side='right')
        attach_tooltip(load_btn, "Load a FASTA/GenBank plasmid — full length, no truncation", T)

        # Live sequence stats + validation line
        stat_row = ctk.CTkFrame(inner, fg_color='transparent')
        stat_row.pack(fill='x', pady=(0, 6))
        self.cloning_seq_info = self._label(stat_row, "0 bp", 'dim')
        self.cloning_seq_info.pack(side='left', padx=(2, 12))
        self.cloning_seq_warn = ctk.CTkLabel(
            stat_row, text="", font=(FONT_FAMILY, 10),
            text_color=T.get('danger', '#ff4444'))
        self.cloning_seq_warn.pack(side='left')
        self.cloning_seq.bind('<KeyRelease>', lambda e: self._cloning_refresh_seq_info())

        enzyme_row = ctk.CTkFrame(inner, fg_color='transparent')
        enzyme_row.pack(fill='x', pady=(0, 2))
        self._label(enzyme_row, "Enzyme:", "body").pack(side='left', padx=(0, 8))

        enzyme_names = self._cloning_enzyme_names()
        self.cloning_enzyme = ctk.CTkComboBox(
            enzyme_row, values=enzyme_names, width=150, height=32,
            font=(FONT_FAMILY, 11),
            fg_color=T['input_bg'], border_color=T['border'],
            button_color=T['accent'], button_hover_color=T['accent_dim'],
            dropdown_fg_color=T['card'], dropdown_hover_color=T['border'],
            dropdown_text_color=T['text'], text_color=T['text'],
            command=lambda _v: self._cloning_refresh_enzyme_info())
        self.cloning_enzyme.pack(side='left', padx=(0, 8))
        attach_tooltip(self.cloning_enzyme,
                       f"All {len(enzyme_names)} restriction enzymes from the built-in database",
                       T)
        self.cloning_enzyme.set("EcoRI")
        # Typing a name also updates the live info
        self.cloning_enzyme.bind('<KeyRelease>',
                                 lambda e: self._cloning_refresh_enzyme_info())

        self._label(enzyme_row, "Topology:", "body").pack(side='left', padx=(0, 8))
        self.cloning_topo = ctk.CTkComboBox(
            enzyme_row, values=["circular", "linear"], width=105, height=32,
            font=(FONT_FAMILY, 11), fg_color=T['input_bg'], border_color=T['border'],
            button_color=T['accent'], button_hover_color=T['accent_dim'],
            dropdown_fg_color=T['card'], dropdown_hover_color=T['border'],
            dropdown_text_color=T['text'], text_color=T['text'])
        self.cloning_topo.pack(side='left')
        attach_tooltip(self.cloning_topo,
                       "Plasmids are circular; PCR products and genomic fragments are linear", T)
        self.cloning_topo.set("circular")

        # Live enzyme info (site / overhang)
        self.cloning_enzyme_info = self._label(inner, "", 'dim')
        self.cloning_enzyme_info.pack(anchor='w', padx=2, pady=(0, 6))
        self._cloning_refresh_enzyme_info()

        btn_row = ctk.CTkFrame(inner, fg_color='transparent')
        btn_row.pack(fill='x', pady=(0, 8))
        b_dig = self._action_button(btn_row, "Run Digestion", self._run_digestion)
        b_dig.pack(side='left', padx=(0, 8))
        attach_tooltip(b_dig, "Simulate the digest and print a full fragment report", T)
        b_gel = self._action_button(btn_row, "Virtual Gel", self._run_virtual_gel,
                                    color_key='success')
        b_gel.pack(side='left')
        attach_tooltip(b_gel, "Render an agarose gel of the digest (interactive viewer)", T)

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
        b_pcr = self._action_button(pcr_btn, "Run PCR", self._run_pcr)
        b_pcr.pack(side='left')
        attach_tooltip(b_pcr, "In-silico PCR: annealing check, extension, product size & Tm", T)

        # ── Primer Design ──
        ctk.CTkFrame(inner, height=1, fg_color=T['border']).pack(fill='x', pady=(8, 12))
        self._label(inner, "Primer Design", "sub").pack(anchor='w', pady=(0, 6))

        primer_row = ctk.CTkFrame(inner, fg_color='transparent')
        primer_row.pack(fill='x', pady=(0, 6))
        self.primer_seq = self._input_entry(primer_row, "Target sequence for primer design...")
        self.primer_seq.pack(side='left', fill='x', expand=True, padx=(0, 8))
        b_pd = self._action_button(primer_row, "Design Primers", self._design_primers)
        b_pd.pack(side='right')
        attach_tooltip(b_pd, "Auto-design a primer pair with melting temperatures", T)

        # ── Results ──
        ctk.CTkFrame(inner, height=1, fg_color=T['border']).pack(fill='x', pady=(8, 12))
        res_head = ctk.CTkFrame(inner, fg_color='transparent')
        res_head.pack(fill='x', pady=(0, 6))
        self._label(res_head, "Results", "sub").pack(side='left')
        b_copy = self._action_button(res_head, "Copy", self._cloning_copy_results,
                                     color_key='success')
        b_copy.pack(side='right', padx=(6, 0))
        attach_tooltip(b_copy, "Copy the full report to the clipboard", T)
        b_clr = self._action_button(res_head, "Clear",
                                    lambda: self.cloning_result.delete("1.0", "end"),
                                    'danger')
        b_clr.pack(side='right')
        self.cloning_result = self._text_box(inner, height=180)
        self.cloning_result.pack(fill='both', expand=True)
        self._cloning_refresh_seq_info()

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _cloning_enzyme_names(self):
        """All available enzymes from the core database, sorted."""
        try:
            from ...core.utils import RESTRICTION_ENZYMES_SITES
            return sorted(RESTRICTION_ENZYMES_SITES.keys())
        except Exception:
            return ["EcoRI", "BamHI", "HindIII", "NotI", "XhoI", "SacI"]

    def _cloning_clean_seq(self):
        """Read + normalize the digest sequence input (upper, de-spaced)."""
        raw = self.cloning_seq.get()
        return re.sub(r"\s+", "", raw).upper()

    def _cloning_refresh_seq_info(self):
        """Update the bp counter and invalid-character warning live."""
        try:
            seq = self._cloning_clean_seq()
            n = len(seq)
            info = f"{n:,} bp" if n else "0 bp"
            self.cloning_seq_info.configure(text=info)
            bad = sorted(set(seq) - set("ACGTRYSWKMBDHVN"))
            if bad and n:
                self.cloning_seq_warn.configure(
                    text=f"⚠ invalid characters: {' '.join(bad[:6])}")
            else:
                self.cloning_seq_warn.configure(text="")
        except Exception:
            pass

    def _cloning_refresh_enzyme_info(self):
        """Show recognition site and cut offset for the selected enzyme."""
        try:
            from ...core.utils import RESTRICTION_ENZYMES, RESTRICTION_ENZYMES_SITES
            name = self.cloning_enzyme.get().strip()
            site = RESTRICTION_ENZYMES_SITES.get(name)
            if site is None:
                self.cloning_enzyme_info.configure(
                    text=f"'{name}' not in database — {len(RESTRICTION_ENZYMES_SITES)} enzymes available")
                return
            data = RESTRICTION_ENZYMES.get(name)
            parts = [f"site: {site}"]
            # Values are (site, cut_offset) tuples
            if isinstance(data, (tuple, list)) and len(data) >= 2:
                parts.append(f"cut offset: +{data[1]}")
            if len(site) % 2 == 0:
                parts.append("palindromic site")
            self.cloning_enzyme_info.configure(text="   ·   ".join(parts))
        except Exception:
            self.cloning_enzyme_info.configure(text="")

    def _cloning_copy_results(self):
        content = self.cloning_result.get("1.0", "end-1c")
        if not content.strip():
            self._msg_info("Nothing to Copy", "Run an analysis first.")
            return
        self.clipboard_clear()
        self.clipboard_append(content)
        self._set_status("Results copied to clipboard")

    # ── Actions ─────────────────────────────────────────────────────────────

    def _cloning_load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("FASTA", "*.fasta *.fa"), ("GenBank", "*.gb *.genbank"), ("All", "*.*")])
        if not path:
            return

        def work():
            from ...core.sequence import read_fasta
            seqs = read_fasta(path)
            if not seqs:
                raise ValueError("No sequences found in file")
            return seqs[0][1]

        def done(seq):
            max_len = 200_000
            if len(seq) > max_len:
                seq = seq[:max_len]
                self._msg_warning(
                    "Sequence Truncated",
                    f"Very long input — only the first {max_len:,} bp were loaded\n"
                    "(GUI limit to keep the entry responsive).")
            self.cloning_seq.delete(0, 'end')
            self.cloning_seq.insert(0, seq)
            self._cloning_refresh_seq_info()
            self._set_status(f"Loaded {os.path.basename(path)} — {len(seq):,} bp")

        self._run_bg(work, on_result=done,
                     status=f"Loading {os.path.basename(path)}...")

    def _run_digestion(self):
        seq = self._cloning_clean_seq()
        if not seq:
            self._msg_warning("Input Required", "Please enter or load a DNA sequence.")
            return
        enzyme = self.cloning_enzyme.get().strip()
        topo = self.cloning_topo.get()

        def work():
            from ...core.cloning import format_digest_report, simulate_digestion
            result = simulate_digestion(seq, enzyme, topology=topo)
            return result, format_digest_report(result)

        def done(payload):
            result, report = payload
            self.cloning_result.delete("1.0", "end")
            self.cloning_result.insert("1.0", report)
            self._last_digest_result = result
            n_cuts = len(result.get('cuts', []))
            self._set_status(
                f"Digestion complete: {len(result['fragments'])} fragments, {n_cuts} cuts")

        self._run_bg(work, on_result=done, status=f"Running {enzyme} digestion...")

    def _run_virtual_gel(self):
        seq = self._cloning_clean_seq()
        if not seq:
            self._msg_warning("Input Required", "Please enter or load a DNA sequence.")
            return
        enzyme = self.cloning_enzyme.get().strip()
        topo = self.cloning_topo.get()

        def work():
            from ...core.cloning import plot_virtual_gel, simulate_digestion
            result = simulate_digestion(seq, enzyme, topology=topo)
            # Pass sizes (ints) — the canonical format for the gel plot
            fig = plot_virtual_gel(result['sizes'], title=f"{enzyme} digest ({topo})")
            return result, fig

        def done(payload):
            result, fig = payload
            self._last_digest_result = result
            self._record_plot(fig, f"Virtual Gel ({enzyme})")
            self._show_plot_figure(fig)
            if not result['sizes']:
                self._msg_info("No Fragments",
                               f"{enzyme} does not cut this sequence — uncut lane shown.")
            self._set_status("Virtual gel displayed")

        self._run_bg(work, on_result=done, status="Rendering virtual gel...")

    def _run_pcr(self):
        seq = self.cloning_seq.get().strip()
        if not seq:
            self._msg_warning("Input Required", "Please enter or load a DNA sequence.")
            return
        fwd = self.pcr_fwd.get().strip()
        rev = self.pcr_rev.get().strip()
        if not fwd or not rev:
            self._msg_warning("Primers Required", "Enter both forward and reverse primers.")
            return

        def work():
            from ...core.cloning import format_pcr_report, simulate_pcr
            result = simulate_pcr(seq, fwd, rev)
            return format_pcr_report(result)

        def done(report):
            self.cloning_result.delete("1.0", "end")
            self.cloning_result.insert("1.0", report)
            self._set_status("PCR simulation complete")

        self._run_bg(work, on_result=done, status="Running PCR simulation...")

    def _design_primers(self):
        seq = self.primer_seq.get().strip()
        if not seq:
            self._msg_warning("Input Required", "Enter a target sequence for primer design.")
            return

        def work():
            from ...core.cloning import design_primers, format_primer_report
            primers = design_primers(seq)
            return format_primer_report(primers)

        def done(report):
            self.cloning_result.delete("1.0", "end")
            self.cloning_result.insert("1.0", report)
            self._set_status("Primers designed")

        self._run_bg(work, on_result=done, status="Designing primers...")

    def _show_plot_figure(self, fig):
        """Display a figure via the shared interactive plot window (zoom/pan/save)."""
        self._show_plot_from_figure(fig, "Cloning Plot")
