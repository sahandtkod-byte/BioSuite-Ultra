"""
Advanced analysis tabs: Single-Cell, Protein Structure, CRISPR, Population Genetics, ML.
"""
import math
import os

import customtkinter as ctk


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
        min_genes_text = self.sc_mingenes.get().strip() or "200"
        max_mito_text = self.sc_maxmito.get().strip() or "20"
        try:
            min_genes = int(min_genes_text)
        except ValueError:
            self._msg_error("Error",
                            f"Min genes must be a whole number (got {min_genes_text!r}).")
            return
        if min_genes < 0:
            self._msg_error("Error", "Min genes must be zero or greater.")
            return
        try:
            max_mito = float(max_mito_text)
        except ValueError:
            self._msg_error("Error",
                            f"Max mito% must be a number (got {max_mito_text!r}).")
            return
        if not math.isfinite(max_mito) or not 0 <= max_mito <= 100:
            self._msg_error("Error",
                            "Max mito% must be a finite percentage between 0 and 100.")
            return
        try:
            from ...core.single_cell import (
                format_sc_report,
                load_count_matrix,
                run_full_pipeline,
            )
            adata, err = load_count_matrix(path)
            if err:
                self._msg_error("Error", err)
                return
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
        if os.path.exists(pdb_input):
            filepath = pdb_input
        else:
            suffix = os.path.splitext(pdb_input)[1]
            separators = [sep for sep in (os.sep, os.altsep) if sep]
            if suffix or any(sep in pdb_input for sep in separators):
                self._msg_error("Error", f"File not found: {pdb_input}")
                return
            filepath = None
        try:
            from ...core.structure import format_structure_report, full_analysis
            if filepath is not None:
                info = full_analysis(filepath=filepath)
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
        self.crispr_pam_combo = ctk.CTkComboBox(params, values=["SpCas9", "SaCas9", "Cas12a", "SpCas9-HF1"],
                                                 width=150, height=32, state='readonly',
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
            pam = self.crispr_pam_combo.get()
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

        btn_row = ctk.CTkFrame(inner, fg_color='transparent')
        btn_row.pack(fill='x', pady=(0, 8))
        self._action_button(btn_row, "Run Analysis", self._run_popgen).pack(side='left', padx=(0, 8))
        self._action_button(btn_row, "Run PCA", self._run_pca, 'accent_dim').pack(side='left')
        self.popgen_result = self._text_box(inner, height=200)
        self.popgen_result.pack(fill='both', expand=True)

    def _popgen_read_matrix(self):
        """Parse the genotype box into an int matrix, or None after a complaint."""
        import numpy as np
        text = self.popgen_input.get("1.0", "end").strip()
        if not text:
            self._msg_warning("No data", "Enter genotype matrix rows.")
            return None
        rows = []
        for line_no, line in enumerate(text.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            row = []
            for token in line.split(','):
                try:
                    row.append(int(token))
                except ValueError:
                    self._msg_error(
                        "Error",
                        f"Genotype values must be whole numbers "
                        f"(line {line_no}: {token.strip()!r}).")
                    return None
            rows.append((line_no, row))
        first_line, first_row = rows[0]
        for line_no, row in rows[1:]:
            if len(row) != len(first_row):
                self._msg_error(
                    "Error",
                    f"Every genotype row must have the same number of sites "
                    f"(line {line_no} has {len(row)} values, "
                    f"line {first_line} has {len(first_row)}).")
                return None
        for line_no, row in rows:
            for value in row:
                if value not in (0, 1, 2):
                    self._msg_error(
                        "Error",
                        f"Genotype values must be 0, 1 or 2 (line {line_no} has {value}).")
                    return None
        return np.array([row for _line, row in rows])

    def _run_popgen(self):
        matrix = self._popgen_read_matrix()
        if matrix is None:
            return
        try:
            from ...core.popgen import format_popgen_report, full_analysis
            report = full_analysis(matrix)
            self.popgen_result.delete("1.0", "end")
            self.popgen_result.insert("end", format_popgen_report(report))
            self._set_status(f"PopGen: {report.num_sites} sites analyzed")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _run_pca(self):
        from ...core.popgen import pca_genotypes
        matrix = self._popgen_read_matrix()
        if matrix is None:
            return
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            self._msg_error("Error", "PCA needs at least 2 samples and 2 sites.")
            return
        try:
            coords, ratios = pca_genotypes(matrix)
        except Exception as e:
            self._msg_error("Error", str(e))
            return
        self.popgen_result.delete("1.0", "end")
        self.popgen_result.insert("end", "PCA coordinates:\n")
        for i, row in enumerate(coords, 1):
            values = ", ".join(f"PC{j+1}={v:.4f}" for j, v in enumerate(row))
            self.popgen_result.insert("end", f"Sample_{i}: {values}\n")
        self.popgen_result.insert(
            "end", "\nExplained variance ratio: "
            + ", ".join(f"{r:.4f}" for r in ratios) + "\n")
        self._set_status(f"PCA: {matrix.shape[0]} samples, {len(ratios)} components")

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
        self.ml_model = self._input_entry(params, "RF, SVM or RFR", width=110)
        self.ml_model.pack(side='left', padx=(4, 12))
        self._label(params, 'Label column:', 'small').pack(side='left')
        self.ml_label = self._input_entry(params, "class", width=120)
        self.ml_label.pack(side='left', padx=4)

        btn_row = ctk.CTkFrame(inner, fg_color='transparent')
        btn_row.pack(fill='x', pady=(0, 8))
        self._action_button(btn_row, "Train Model", self._run_ml).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "ROC Curve", self._run_ml_roc).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Select Features", self._run_ml_features).pack(side='left')
        self.ml_result = self._text_box(inner, height=250)
        self.ml_result.pack(fill='both', expand=True)

    def _ml_read_xy(self):
        import numpy as np
        import pandas as pd
        path = self.ml_path.get().strip()
        if not path:
            self._msg_warning("No file", "Select a CSV data file.")
            return None
        try:
            df = pd.read_csv(path)
        except Exception as e:
            self._msg_error("Error", str(e))
            return None
        label_col = self.ml_label.get().strip() or 'class'
        if label_col not in df.columns:
            self._msg_error("Error", f"Column '{label_col}' not found.")
            return None
        features = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
        return features.values, df[label_col].values, list(features.columns)

    def _run_ml(self):
        path = self.ml_path.get().strip()
        if not path:
            self._msg_warning("No file", "Select a CSV data file.")
            return
        model_type = self.ml_model.get().strip().upper() or 'RF'
        if model_type not in ('RF', 'SVM', 'RFR'):
            self._msg_error(
                "Error",
                f"Model must be RF, SVM or RFR (got {self.ml_model.get().strip()!r}).")
            return
        data = self._ml_read_xy()
        if data is None:
            return
        X, y, _names = data
        try:
            from ...core.bio_ml import (
                format_ml_report,
                train_random_forest,
                train_random_forest_regressor,
                train_svm,
            )
            if model_type == 'SVM':
                result = train_svm(X, y)
            elif model_type == 'RFR':
                result = train_random_forest_regressor(X, y)
            else:
                result = train_random_forest(X, y)
            self.ml_result.delete("1.0", "end")
            self.ml_result.insert("end", format_ml_report(result))
            self._set_status(f"ML: {result.model_type}, accuracy={result.accuracy}")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _run_ml_roc(self):
        data = self._ml_read_xy()
        if data is None:
            return
        X, y, _names = data
        try:
            from ...core.bio_ml import compute_roc_curve, train_random_forest
            result = train_random_forest(X, y)
            if result.engine == 'none':
                self._msg_error("Error", result.message)
                return
            if result.probabilities is None:
                self._msg_error("Error", "ROC needs a binary label column (exactly two classes).")
                return
            roc = compute_roc_curve(result.eval_labels, result.probabilities)
            lines = [
                "=== ROC Curve (holdout) ===",
                f"AUC: {roc['auc']:.4f}",
                f"Curve points: {len(roc['fpr'])}",
                "FPR / TPR / threshold:",
            ]
            for fpr, tpr, thr in zip(roc['fpr'], roc['tpr'], roc['thresholds']):
                lines.append(f"  {fpr:.3f} / {tpr:.3f} / {thr:.3f}")
            self.ml_result.delete("1.0", "end")
            self.ml_result.insert("end", "\n".join(lines))
            self._set_status(f"ML ROC: AUC={roc['auc']}")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _run_ml_features(self):
        data = self._ml_read_xy()
        if data is None:
            return
        X, y, names = data
        try:
            from ...core.bio_ml import select_features
            indices, scores = select_features(X, y)
            if not indices:
                self._msg_error("Error", "Feature selection needs scikit-learn for the importance ranking.")
                return
            lines = ["=== Feature Selection (importance) ==="]
            for rank, idx in enumerate(indices, 1):
                lines.append(f"{rank}. {names[idx]}: {scores[idx]:.4f}")
            self.ml_result.delete("1.0", "end")
            self.ml_result.insert("end", "\n".join(lines))
            self._set_status(f"ML: {len(indices)} features ranked")
        except Exception as e:
            self._msg_error("Error", str(e))
