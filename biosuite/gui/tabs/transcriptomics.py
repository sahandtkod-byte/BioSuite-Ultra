"""
Transcriptomics tabs: Expression, Trimming, RNA-seq Quantification.
"""
import os
import customtkinter as ctk
from tkinter import filedialog

from ..themes import FONT_BODY, FONT_SMALL


class TranscriptomicsTabMixin:
    """Provides Expression, Trimming, and Quantification tabs."""

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
        import numpy as np
        import matplotlib.pyplot as plt
        from ...core.expression import read_counts_matrix, differential_expression
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
            plt.savefig('volcano_plot.png', dpi=150, bbox_inches='tight')
            plt.close('all')
            self._set_status(f"DE complete: {sig.sum()} significant genes (FDR < 0.05)")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── Trimming Tab ────────────────────────────────────────────────────────

    def _build_trimming_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['trimming'] = f
        self._section_header(f, "FASTQ Quality Trimming")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 8))
        self.trim_path = self._input_entry(file_row, "Input FASTQ file...")
        self.trim_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(file_row, "Browse",
                            lambda: self._browse_file(self.trim_path, [("FASTQ", "*.fastq *.fq")])
                            ).pack(side='right')

        params_row = ctk.CTkFrame(inner, fg_color='transparent')
        params_row.pack(fill='x', pady=(0, 8))
        self._label(params_row, 'Quality threshold:', 'small').pack(side='left')
        self.trim_qual = self._input_entry(params_row, "20", width=80)
        self.trim_qual.pack(side='left', padx=(4, 12))
        self._label(params_row, 'Min length:', 'small').pack(side='left')
        self.trim_minlen = self._input_entry(params_row, "36", width=80)
        self.trim_minlen.pack(side='left', padx=4)

        self._action_button(inner, "Run Trimming", self._run_trimming).pack(pady=(0, 8))
        self.trim_result = self._text_box(inner, height=200)
        self.trim_result.pack(fill='both', expand=True)

    def _run_trimming(self):
        path = self.trim_path.get().strip()
        if not path:
            self._msg_warning("No file", "Select a FASTQ file.")
            return
        try:
            from ...core.trimming import trim_fastq, format_trim_report
            qual = int(self.trim_qual.get().strip() or "20")
            minlen = int(self.trim_minlen.get().strip() or "36")
            out = os.path.splitext(path)[0] + '_trimmed.fastq'
            report = trim_fastq(path, out, quality_threshold=qual, min_length=minlen)
            self.trim_result.delete("1.0", "end")
            self.trim_result.insert("end", format_trim_report(report))
            self._set_status(f"Trimming complete: {report.reads_trimmed} reads trimmed")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── RNA-seq Quantification Tab ──────────────────────────────────────────

    def _build_quant_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['quant'] = f
        self._section_header(f, "RNA-seq Quantification")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        r1_row = ctk.CTkFrame(inner, fg_color='transparent')
        r1_row.pack(fill='x', pady=(0, 8))
        self.quant_reads = self._input_entry(r1_row, "Reads FASTQ file...")
        self.quant_reads.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(r1_row, "Browse",
                            lambda: self._browse_file(self.quant_reads, [("FASTQ", "*.fastq *.fq")])
                            ).pack(side='right')

        trans_row = ctk.CTkFrame(inner, fg_color='transparent')
        trans_row.pack(fill='x', pady=(0, 8))
        self.quant_trans = self._input_entry(trans_row, "Transcriptome FASTA...")
        self.quant_trans.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(trans_row, "Browse",
                            lambda: self._browse_file(self.quant_trans, [("FASTA", "*.fasta *.fa")])
                            ).pack(side='right')

        self._action_button(inner, "Run Quantification", self._run_quant).pack(pady=(0, 8))
        self.quant_result = self._text_box(inner, height=200)
        self.quant_result.pack(fill='both', expand=True)

    def _run_quant(self):
        reads = self.quant_reads.get().strip()
        trans = self.quant_trans.get().strip()
        if not reads or not trans:
            self._msg_warning("Missing input", "Select both reads and transcriptome files.")
            return
        try:
            from ...core.quantification import quantify_reads, format_quant_report
            result = quantify_reads(reads, trans)
            self.quant_result.delete("1.0", "end")
            self.quant_result.insert("end", format_quant_report(result))
            self._set_status(f"Quantification: {result.num_transcripts} transcripts")
        except Exception as e:
            self._msg_error("Error", str(e))
