"""
Workflow and domain-specific tabs: Pipeline, Batch, GO Browser, Pathway, GWAS, Epitope.
"""
import customtkinter as ctk
from tkinter import filedialog

from ..themes import FONT_BODY, FONT_SMALL


class WorkflowTabMixin:
    """Provides Pipeline, Batch, GO Browser, Pathway, GWAS, and Epitope tabs."""

    def _build_pipeline_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['pipeline'] = f
        self._section_header(f, "Pipeline Builder")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Steps (one per line: name:func_name)', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.pipe_text = self._text_box(left, height=200)
        self.pipe_text.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        self.pipe_text.insert("1.0", "reverse:reverse_complement\ntranslate:translate")
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=10, pady=(0, 10))
        self._action_button(btn_row, "Run Pipeline", self._run_pipeline).pack(side='left', padx=(0, 6))

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Results', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.pipe_results = self._text_box(right, height=280)
        self.pipe_results.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    def _run_pipeline(self):
        from ...core.workflow.pipeline import Pipeline
        text = self.pipe_text.get("1.0", "end").strip()
        if not text:
            self._msg_warning("No steps", "Enter pipeline steps first.")
            return
        try:
            p = Pipeline("gui_pipeline")
            for line in text.split('\n'):
                line = line.strip()
                if ':' in line:
                    name, func_name = line.split(':', 1)
                    from ...core import sequence
                    func = getattr(sequence, func_name.strip(), None)
                    if func:
                        p.add_step(name.strip(), func)
            p.set_context(sequence="ATGAAATTTTAA")
            p.run()
            self.pipe_results.delete("1.0", "end")
            self.pipe_results.insert("end", p.summary())
        except Exception as e:
            self._msg_error("Pipeline Error", str(e))

    # ─── Batch Processor Tab ────────────────────────────────────────────────

    def _build_batch_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['batch'] = f
        self._section_header(f, "Batch Processor")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Sample IDs (one per line)', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.batch_text = self._text_box(left, height=200)
        self.batch_text.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        self.batch_text.insert("1.0", "sample1\nsample2\nsample3")
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=10, pady=(0, 10))
        self._action_button(btn_row, "Run Batch", self._run_batch).pack(side='left', padx=(0, 6))

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Results', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.batch_results = self._text_box(right, height=280)
        self.batch_results.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    def _run_batch(self):
        from ...core.workflow.batch import BatchProcessor
        text = self.batch_text.get("1.0", "end").strip()
        if not text:
            self._msg_warning("No samples", "Enter sample IDs first.")
            return
        samples = [l.strip() for l in text.split('\n') if l.strip()]
        try:
            bp = BatchProcessor("gui_batch")
            bp.add_samples(samples, lambda x: f"processed_{x}")
            bp.run(max_workers=2)
            self.batch_results.delete("1.0", "end")
            self.batch_results.insert("end", bp.summary())
        except Exception as e:
            self._msg_error("Batch Error", str(e))

    # ─── GO Browser Tab ─────────────────────────────────────────────────────

    def _build_gobrowser_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['gobrowser'] = f
        self._section_header(f, "Gene Ontology Browser")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Search GO terms', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.go_search = self._input_entry(left, "Search (e.g., kinase, apoptosis)")
        self.go_search.pack(fill='x', padx=14, pady=(0, 8))
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=14, pady=(0, 10))
        self._action_button(btn_row, "Search", self._go_search).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Browse BP", lambda: self._go_browse('BP')).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Browse MF", lambda: self._go_browse('MF')).pack(side='left')
        self._label(left, 'Results:', 'sub').pack(padx=14, anchor='w')
        self.go_results = self._text_box(left, height=200)
        self.go_results.pack(fill='both', expand=True, padx=10, pady=(0, 6))

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Term Details', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.go_detail = self._text_box(right, height=280)
        self.go_detail.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    def _go_search(self):
        from ...core.go_browser import GOBrowser, format_go_results
        go = GOBrowser()
        query = self.go_search.get().strip()
        if not query:
            return
        results = go.search(query)
        self.go_results.delete("1.0", "end")
        self.go_results.insert("end", format_go_results(results))
        if results:
            t = results[0]
            self.go_detail.delete("1.0", "end")
            self.go_detail.insert("end", f"ID: {t.go_id}\nName: {t.name}\nNS: {t.namespace}\n")
            self.go_detail.insert("end", f"Def: {t.definition}\n\nParents:\n")
            for p in go.get_parents(t.go_id):
                self.go_detail.insert("end", f"  {p.go_id} {p.name}\n")
            self.go_detail.insert("end", "\nChildren:\n")
            for c in go.get_children(t.go_id):
                self.go_detail.insert("end", f"  {c.go_id} {c.name}\n")

    def _go_browse(self, namespace):
        from ...core.go_browser import GOBrowser
        go = GOBrowser()
        terms = go.get_namespace_terms(namespace)
        self.go_results.delete("1.0", "end")
        for t in terms[:30]:
            self.go_results.insert("end", f"{t.go_id} {t.name}\n")

    # ─── Pathway Visualization Tab ──────────────────────────────────────────

    def _build_pathway_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['pathway'] = f
        self._section_header(f, "Pathway Visualization")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Genes (comma-sep)', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.pathway_genes = self._input_entry(left, "EGF,EGFR,RAS,RAF,MEK,ERK")
        self.pathway_genes.pack(fill='x', padx=14, pady=(0, 8))
        self.pathway_genes.insert(0, "EGF,EGFR,GRB2,RAS,RAF,MEK,ERK")
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=14, pady=(0, 10))
        self._action_button(btn_row, "Draw Pathway", self._draw_pathway).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "KEGG Demo", self._kegg_demo).pack(side='left')

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Pathway Info', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.pathway_info = self._text_box(right, height=280)
        self.pathway_info.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    def _draw_pathway(self):
        import matplotlib.pyplot as plt
        from ...core.pathway_viz import create_custom_pathway, draw_pathway, format_pathway_report
        genes = [g.strip() for g in self.pathway_genes.get().split(',') if g.strip()]
        if not genes:
            return
        pm = create_custom_pathway(genes)
        self.pathway_info.delete("1.0", "end")
        self.pathway_info.insert("end", format_pathway_report(pm))
        fig = draw_pathway(pm)
        plt.show()
        plt.close()

    def _kegg_demo(self):
        import matplotlib.pyplot as plt
        from ...core.pathway_viz import create_kegg_style_pathway, draw_pathway, format_pathway_report
        pm = create_kegg_style_pathway()
        self.pathway_info.delete("1.0", "end")
        self.pathway_info.insert("end", format_pathway_report(pm))
        fig = draw_pathway(pm)
        plt.show()
        plt.close()

    # ─── GWAS Analysis Tab ──────────────────────────────────────────────────

    def _build_gwas_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['gwas'] = f
        self._section_header(f, "GWAS Analysis")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Input', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self._label(left, 'Load CSV or use demo data', 'body').pack(padx=14, anchor='w')
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=14, pady=(0, 10))
        self._action_button(btn_row, "Load CSV", self._gwas_load).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Run Demo", self._gwas_demo).pack(side='left')

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Results', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.gwas_results = self._text_box(right, height=280)
        self.gwas_results.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    def _gwas_load(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            import pandas as pd
            self._gwas_data = pd.read_csv(path)
            self.gwas_results.delete("1.0", "end")
            self.gwas_results.insert("end", f"Loaded {len(self._gwas_data)} SNPs\n")

    def _gwas_demo(self):
        from ...core.gwas import run_gwas, detect_lead_snps, generate_gwas_data, format_gwas_report
        data = generate_gwas_data(n_snps=2000)
        results = run_gwas(data)
        leads = detect_lead_snps(results)
        self.gwas_results.delete("1.0", "end")
        self.gwas_results.insert("end", format_gwas_report(results, leads))

    # ─── Epitope Prediction Tab ─────────────────────────────────────────────

    def _build_epitope_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['epitope'] = f
        self._section_header(f, "Epitope Prediction")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Protein Sequence', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.epi_seq = self._text_box(left, height=150)
        self.epi_seq.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        self._label(left, 'HLA Type:', 'body').pack(padx=14, anchor='w')
        self.epi_hla = self._input_entry(left, "A0201")
        self.epi_hla.pack(fill='x', padx=14, pady=(0, 8))
        self.epi_hla.insert(0, "A0201")
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=14, pady=(0, 10))
        self._action_button(btn_row, "Predict", self._run_epitope).pack(side='left', padx=(0, 6))

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Epitopes', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.epi_results = self._text_box(right, height=280)
        self.epi_results.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    def _run_epitope(self):
        from ...core.epitope import predict_t_cell_epitopes, predict_b_cell_epitopes, format_epitope_report
        seq = self.epi_seq.get("1.0", "end").strip().upper()
        if not seq:
            self._msg_warning("No sequence", "Enter a protein sequence first.")
            return
        hla = self.epi_hla.get().strip() or "A0201"
        try:
            tc = predict_t_cell_epitopes(seq, mhc_type=hla)
            bc = predict_b_cell_epitopes(seq)
            self.epi_results.delete("1.0", "end")
            self.epi_results.insert("end", format_epitope_report(tc, bc, "input"))
        except Exception as e:
            self._msg_error("Epitope Error", str(e))
