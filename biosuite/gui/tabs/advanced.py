"""
Advanced analysis tabs: Single-Cell, Protein Structure, CRISPR, Population Genetics, ML.
"""
import os
import customtkinter as ctk
from tkinter import filedialog

from ..themes import FONT_BODY, FONT_SMALL


class AdvancedTabMixin:
    """Provides Single-Cell, Structure, CRISPR, PopGen, and ML tabs."""

    def _build_singlecell_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['singlecell'] = f
        self._section_header(f, "Single-Cell RNA-seq Analysis")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 8))
        self.sc_path = self._input_entry(file_row, "Count matrix (h5ad, csv, 10x dir)...")
        self.sc_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(file_row, "Browse",
                            lambda: self._browse_file(self.sc_path, [("All", "*.*")])
                            ).pack(side='right')

        params = ctk.CTkFrame(inner, fg_color='transparent')
        params.pack(fill='x', pady=(0, 8))
        self._label(params, 'Min genes:', 'small').pack(side='left')
        self.sc_mingenes = self._input_entry(params, "200", width=80)
        self.sc_mingenes.pack(side='left', padx=(4, 12))
        self._label(params, 'Max mito%:', 'small').pack(side='left')
        self.sc_maxmito = self._input_entry(params, "20", width=80)
        self.sc_maxmito.pack(side='left', padx=4)

        self._action_button(inner, "Run Single-Cell Pipeline", self._run_sc).pack(pady=(0, 8))
        self.sc_result = self._text_box(inner, height=200)
        self.sc_result.pack(fill='both', expand=True)

    def _run_sc(self):
        path = self.sc_path.get().strip()
        if not path:
            self._msg_warning("No file", "Select a count matrix file.")
            return
        try:
            from ...core.single_cell import load_count_matrix, run_full_pipeline, format_sc_report
            adata, err = load_count_matrix(path)
            if err:
                self._msg_error("Error", err)
                return
            min_genes = int(self.sc_mingenes.get().strip() or "200")
            max_mito = float(self.sc_maxmito.get().strip() or "20")
            adata, report = run_full_pipeline(adata, min_genes=min_genes, max_pct_mito=max_mito)
            self.sc_result.delete("1.0", "end")
            self.sc_result.insert("end", format_sc_report(report))
            self._set_status(f"scRNA-seq: {report.num_cells} cells, {report.num_clusters} clusters")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── Structure Tab ───────────────────────────────────────────────────────

    def _build_structure_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['structure'] = f
        self._section_header(f, "Protein Structure Analysis")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 8))
        self.pdb_path = self._input_entry(file_row, "PDB ID or file path...")
        self.pdb_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(file_row, "Browse",
                            lambda: self._browse_file(self.pdb_path, [("PDB", "*.pdb")])
                            ).pack(side='right')

        self._action_button(inner, "Analyze Structure", self._run_structure).pack(pady=(0, 8))
        self.struct_result = self._text_box(inner, height=250)
        self.struct_result.pack(fill='both', expand=True)

    def _run_structure(self):
        pdb_input = self.pdb_path.get().strip()
        if not pdb_input:
            self._msg_warning("No input", "Enter a PDB ID or file path.")
            return
        try:
            from ...core.structure import full_analysis, format_structure_report
            if os.path.exists(pdb_input):
                info = full_analysis(filepath=pdb_input)
            else:
                info = full_analysis(pdb_id=pdb_input)
            self.struct_result.delete("1.0", "end")
            self.struct_result.insert("end", format_structure_report(info))
            self._set_status(f"Structure: {info.num_residues} residues")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── CRISPR Tab ──────────────────────────────────────────────────────────

    def _build_crispr_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['crispr'] = f
        self._section_header(f, "CRISPR Guide RNA Design")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        self._label(inner, 'Target DNA Sequence:', 'sub').pack(anchor='w')
        self.crispr_seq = self._text_box(inner, height=100)
        self.crispr_seq.pack(fill='x', pady=(0, 8))

        params = ctk.CTkFrame(inner, fg_color='transparent')
        params.pack(fill='x', pady=(0, 8))
        self._label(params, 'PAM type:', 'small').pack(side='left')
        self.crispr_pam_combo = ctk.CTkComboBox(params, values=["SpCas9", "SaCas9", "Cas12a"],
                                                 width=150, height=32,
                                                 fg_color=T['input_bg'], border_color=T['border'],
                                                 button_color=T['accent'], button_hover_color=T['accent_dim'],
                                                 dropdown_fg_color=T['card'], dropdown_hover_color=T['border'],
                                                 dropdown_text_color=T['text'], text_color=T['text'])
        self.crispr_pam_combo.set("SpCas9")
        self.crispr_pam_combo.pack(side='left', padx=(8, 0))

        self._action_button(inner, "Design Guides", self._run_crispr).pack(pady=(0, 8))
        self.crispr_result = self._text_box(inner, height=200)
        self.crispr_result.pack(fill='both', expand=True)

    def _run_crispr(self):
        seq = self.crispr_seq.get("1.0", "end").strip()
        if not seq:
            self._msg_warning("No sequence", "Enter a target DNA sequence.")
            return
        try:
            from ...core.crispr import design_guides, format_crispr_report
            pam = self.crispr_pam_combo.get() if hasattr(self, 'crispr_pam_combo') else "SpCas9"
            result = design_guides(seq, pam_type=pam)
            self.crispr_result.delete("1.0", "end")
            self.crispr_result.insert("end", format_crispr_report(result))
            self._set_status(f"CRISPR: {result.num_guides} guides found")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── Population Genetics Tab ─────────────────────────────────────────────

    def _build_popgen_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['popgen'] = f
        self._section_header(f, "Population Genetics")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        self._label(inner, 'Genotype Matrix (rows=samples, 0/1/2):', 'sub').pack(anchor='w')
        self.popgen_input = self._text_box(inner, height=150)
        self.popgen_input.pack(fill='x', pady=(0, 8))

        self._action_button(inner, "Run Analysis", self._run_popgen).pack(pady=(0, 8))
        self.popgen_result = self._text_box(inner, height=200)
        self.popgen_result.pack(fill='both', expand=True)

    def _run_popgen(self):
        import numpy as np
        text = self.popgen_input.get("1.0", "end").strip()
        if not text:
            self._msg_warning("No data", "Enter genotype matrix rows.")
            return
        try:
            rows = []
            for line in text.split('\n'):
                line = line.strip()
                if line:
                    rows.append([int(x) for x in line.split(',')])
            if not rows:
                return
            matrix = np.array(rows)
            from ...core.popgen import full_analysis, format_popgen_report
            report = full_analysis(matrix)
            self.popgen_result.delete("1.0", "end")
            self.popgen_result.insert("end", format_popgen_report(report))
            self._set_status(f"PopGen: {report.num_sites} sites analyzed")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── Machine Learning Tab ────────────────────────────────────────────────

    def _build_ml_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['ml'] = f
        self._section_header(f, "Machine Learning for Biology")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 8))
        self.ml_path = self._input_entry(file_row, "Data CSV file (features + label column)...")
        self.ml_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(file_row, "Browse",
                            lambda: self._browse_file(self.ml_path, [("CSV", "*.csv")])
                            ).pack(side='right')

        params = ctk.CTkFrame(inner, fg_color='transparent')
        params.pack(fill='x', pady=(0, 8))
        self._label(params, 'Model:', 'small').pack(side='left')
        self.ml_model = self._input_entry(params, "RF", width=80)
        self.ml_model.pack(side='left', padx=(4, 12))
        self._label(params, 'Label column:', 'small').pack(side='left')
        self.ml_label = self._input_entry(params, "class", width=120)
        self.ml_label.pack(side='left', padx=4)

        self._action_button(inner, "Train Model", self._run_ml).pack(pady=(0, 8))
        self.ml_result = self._text_box(inner, height=250)
        self.ml_result.pack(fill='both', expand=True)

    def _run_ml(self):
        import pandas as pd
        import numpy as np
        path = self.ml_path.get().strip()
        if not path:
            self._msg_warning("No file", "Select a CSV data file.")
            return
        try:
            df = pd.read_csv(path)
            label_col = self.ml_label.get().strip() or 'class'
            model_type = self.ml_model.get().strip().upper() or 'RF'
            if label_col not in df.columns:
                self._msg_error("Error", f"Column '{label_col}' not found.")
                return
            X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
            y = df[label_col].values
            from ...core.bio_ml import train_random_forest, train_svm, format_ml_report
            if model_type == 'SVM':
                result = train_svm(X, y)
            else:
                result = train_random_forest(X, y)
            self.ml_result.delete("1.0", "end")
            self.ml_result.insert("end", format_ml_report(result))
            self._set_status(f"ML: {result.model_type}, accuracy={result.accuracy}")
        except Exception as e:
            self._msg_error("Error", str(e))
