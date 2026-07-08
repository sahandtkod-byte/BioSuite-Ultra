"""
Genomics tabs: NGS/VCF, Assembly, Metagenomics, 16S rRNA, SV/CNV, BigWig.
"""
import os
import customtkinter as ctk
from tkinter import filedialog

from ..themes import FONT_BODY, FONT_SMALL


class GenomicsTabMixin:
    """Provides NGS, Assembly, Metagenomics, 16S rRNA, SV/CNV, and BigWig tabs."""

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
        import numpy as np
        import matplotlib.pyplot as plt
        from ...core.ngs import read_vcf, manhattan_from_vcf
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

    # ─── Assembly Tab ────────────────────────────────────────────────────────

    def _build_assembly_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['assembly'] = f
        self._section_header(f, "Genome Assembly")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 8))
        self.asm_path = self._input_entry(file_row, "Reads FASTQ file...")
        self.asm_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(file_row, "Browse",
                            lambda: self._browse_file(self.asm_path, [("FASTQ", "*.fastq *.fq")])
                            ).pack(side='right')

        self._action_button(inner, "Run Assembly", self._run_assembly).pack(pady=(0, 8))
        self.asm_result = self._text_box(inner, height=200)
        self.asm_result.pack(fill='both', expand=True)

    def _run_assembly(self):
        path = self.asm_path.get().strip()
        if not path:
            self._msg_warning("No file", "Select a reads file.")
            return
        try:
            from ...core.assembly import assemble, format_assembly_report
            result = assemble(path)
            self.asm_result.delete("1.0", "end")
            self.asm_result.insert("end", format_assembly_report(result))
            self._set_status(f"Assembly: {result.num_contigs} contigs, N50={result.n50}")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── Metagenomics Tab ────────────────────────────────────────────────────

    def _build_metagenomics_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['metagenomics'] = f
        self._section_header(f, "Metagenomics Analysis")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 8))
        self.meta_path = self._input_entry(file_row, "Reads FASTQ file...")
        self.meta_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(file_row, "Browse",
                            lambda: self._browse_file(self.meta_path, [("FASTQ", "*.fastq *.fq")])
                            ).pack(side='right')

        self._action_button(inner, "Run Classification", self._run_meta).pack(pady=(0, 8))
        self.meta_result = self._text_box(inner, height=250)
        self.meta_result.pack(fill='both', expand=True)

    def _run_meta(self):
        path = self.meta_path.get().strip()
        if not path:
            self._msg_warning("No file", "Select a reads file.")
            return
        try:
            from ...core.metagenomics import classify_reads, format_metagenomics_report
            result = classify_reads(path)
            self.meta_result.delete("1.0", "end")
            self.meta_result.insert("end", format_metagenomics_report(result))
            self._set_status(f"Metagenomics: {len(result.classifications)} reads classified")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── 16S rRNA Pipeline Tab ──────────────────────────────────────────────

    def _build_16srna_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['16srna'] = f
        self._section_header(f, "16S rRNA Classification Pipeline")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Enter 16S sequences (name:sequence)', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.rna_text = self._text_box(left, height=200)
        self.rna_text.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        self.rna_text.insert("1.0", "Ecoli_16S:TGGAGGAAGGTGGGGACGACGTCAGTATCGAATCTTGGATCAGGATCACCTCCGGA\nStaph_16S:AGCCATGCAGCACCTGTCTCAGCTTCCCGAAGGCACTATACGTAGATCGAAAGTTGAT")
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=10, pady=(0, 10))
        self._action_button(btn_row, "Classify", self._run_16s).pack(side='left', padx=(0, 6))

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Results', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.rna_results = self._text_box(right, height=280)
        self.rna_results.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    def _run_16s(self):
        from ...core.metagenomics import classify_16s_rna, format_16s_report
        text = self.rna_text.get("1.0", "end").strip()
        if not text:
            self._msg_warning("No sequences", "Enter 16S sequences first.")
            return
        seqs = []
        for line in text.split('\n'):
            line = line.strip()
            if ':' in line:
                name, seq = line.split(':', 1)
                seqs.append((name.strip(), seq.strip().upper()))
        if not seqs:
            self._msg_warning("Invalid format", "Use name:sequence format.")
            return
        try:
            result = classify_16s_rna(seqs)
            self.rna_results.delete("1.0", "end")
            self.rna_results.insert("end", format_16s_report(result))
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── SV / CNV Detection Tab ─────────────────────────────────────────────

    def _build_svcnv_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['svcinv'] = f
        self._section_header(f, "SV / CNV Detection from Coverage")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Input', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self._label(left, 'Load CSV coverage or use demo', 'body').pack(padx=14, anchor='w')
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=14, pady=(0, 10))
        self._action_button(btn_row, "Load CSV", self._svcnv_load).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Demo Data", self._svcnv_demo).pack(side='left')

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Results', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.svcnv_results = self._text_box(right, height=280)
        self.svcnv_results.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    def _svcnv_load(self):
        import numpy as np
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Text", "*.txt")])
        if path:
            self._svcnv_data = np.loadtxt(path, delimiter=',')
            self.svcnv_results.delete("1.0", "end")
            self.svcnv_results.insert("end", f"Loaded {len(self._svcnv_data)} coverage values\n")

    def _svcnv_demo(self):
        import numpy as np
        from ...core.variant_calling import detect_structural_variants, detect_cnv, format_sv_report, format_cnv_report
        np.random.seed(42)
        cov = np.random.poisson(30, 5000).astype(float)
        cov[1500:2000] *= 0.3
        cov[3000:3500] *= 2.5
        ref = np.ones(5000) * 30
        svs = detect_structural_variants(cov, ref)
        cnv = detect_cnv(cov, ref)
        self.svcnv_results.delete("1.0", "end")
        self.svcnv_results.insert("end", format_sv_report(svs))
        self.svcnv_results.insert("end", f"\n{format_cnv_report(cnv)}")

    # ─── BigWig Reader Tab ──────────────────────────────────────────────────

    def _build_bigwig_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['bigwig'] = f
        self._section_header(f, "BigWig Reader")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'BigWig File', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=14, pady=(0, 10))
        self._action_button(btn_row, "Browse File", self._bigwig_browse).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Read File", self._bigwig_read).pack(side='left')

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Info', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.bigwig_info = self._text_box(right, height=280)
        self.bigwig_info.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        self._bigwig_path = None

    def _bigwig_browse(self):
        path = filedialog.askopenfilename(filetypes=[("BigWig", "*.bw *.bigwig"), ("All", "*.*")])
        if path:
            self._bigwig_path = path
            self.bigwig_info.delete("1.0", "end")
            self.bigwig_info.insert("end", f"Selected: {path}\n")

    def _bigwig_read(self):
        if not self._bigwig_path:
            self._msg_warning("No file", "Browse for a BigWig file first.")
            return
        from ...core.file_formats import read_bigwig, bigwig_summary, format_bigwig_summary
        try:
            data = read_bigwig(self._bigwig_path)
            self.bigwig_info.delete("1.0", "end")
            if "error" in data:
                self.bigwig_info.insert("end", f"Error: {data['error']}\n")
            elif "chroms" in data:
                self.bigwig_info.insert("end", f"Chromosomes: {len(data['chroms'])}\n\n")
                for chrom, values in list(data['chroms'].items())[:10]:
                    self.bigwig_info.insert("end", f"  {chrom}: {len(values)} values\n")
            else:
                summary = bigwig_summary(self._bigwig_path)
                self.bigwig_info.insert("end", format_bigwig_summary(summary))
        except Exception as e:
            self._msg_error("Error", str(e))
