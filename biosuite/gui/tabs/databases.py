"""
Database and utility tabs: Databases, File Formats, API Keys.
"""
import os
import customtkinter as ctk
from tkinter import filedialog

from ...core.utils import config, save_config
from ..themes import FONT_BODY, FONT_SMALL


class DatabasesTabMixin:
    """Provides Database Search, File Format Parsers, and API Keys tabs."""

    def _build_databases_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['databases'] = f
        self._section_header(f, "Database Search (NCBI, UniProt, PDB, KEGG, Ensembl)")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        params = ctk.CTkFrame(inner, fg_color='transparent')
        params.pack(fill='x', pady=(0, 8))
        self._label(params, 'Database:', 'small').pack(side='left')
        self.db_combo = ctk.CTkComboBox(params, values=["NCBI", "UniProt", "PDB", "KEGG", "Ensembl", "All"],
                                         width=120, height=32,
                                         fg_color=T['input_bg'], border_color=T['border'],
                                         button_color=T['accent'], button_hover_color=T['accent_dim'],
                                         dropdown_fg_color=T['card'], dropdown_text_color=T['text'])
        self.db_combo.set("NCBI")
        self.db_combo.pack(side='left', padx=(8, 12))

        self.db_query = self._input_entry(params, "Search query...", width=300)
        self.db_query.pack(side='left', fill='x', expand=True)

        self._action_button(inner, "Search", self._run_db_search).pack(pady=(0, 8))
        self.db_result = self._text_box(inner, height=250)
        self.db_result.pack(fill='both', expand=True)

    def _run_db_search(self):
        query = self.db_query.get().strip()
        if not query:
            self._msg_warning("No query", "Enter a search term.")
            return
        db = self.db_combo.get().strip().lower()
        try:
            from ...core.databases import search_ncbi, search_uniprot, search_pdb, search_kegg, search_ensembl, search_all, format_search_results
            if db == 'all':
                results = search_all(query)
            elif db == 'ncbi':
                results = search_ncbi(query)
            elif db == 'uniprot':
                results = search_uniprot(query)
            elif db == 'pdb':
                results = search_pdb(query)
            elif db == 'kegg':
                results = search_kegg(query)
            elif db == 'ensembl':
                results = search_ensembl(query)
            else:
                results = search_ncbi(query)
            self.db_result.delete("1.0", "end")
            self.db_result.insert("end", format_search_results(results))
            self._set_status(f"Database search complete")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── File Formats Tab ────────────────────────────────────────────────────

    def _build_fileformats_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['fileformats'] = f
        self._section_header(f, "File Format Parsers (BED, GFF, Newick, Stockholm)")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 8))
        self.ff_path = self._input_entry(file_row, "File path...")
        self.ff_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(file_row, "Browse",
                            lambda: self._browse_file(self.ff_path, [("All", "*.*")])
                            ).pack(side='right')

        self._action_button(inner, "Parse File", self._run_parse).pack(pady=(0, 8))
        self.ff_result = self._text_box(inner, height=250)
        self.ff_result.pack(fill='both', expand=True)

    def _run_parse(self):
        path = self.ff_path.get().strip()
        if not path:
            self._msg_warning("No file", "Select a file to parse.")
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.bed',):
                from ...core.file_formats import parse_bed, format_bed_summary
                records = parse_bed(path)
                self.ff_result.delete("1.0", "end")
                self.ff_result.insert("end", format_bed_summary(records))
            elif ext in ('.gff', '.gtf'):
                from ...core.file_formats import parse_gff, format_gff_summary
                records = parse_gff(path)
                self.ff_result.delete("1.0", "end")
                self.ff_result.insert("end", format_gff_summary(records))
            elif ext in ('.nwk', '.newick', '.tree', '.nhx'):
                from ...core.file_formats import parse_newick, tree_to_ascii
                with open(path) as fh:
                    newick = fh.read().strip()
                tree = parse_newick(newick)
                self.ff_result.delete("1.0", "end")
                for line in tree_to_ascii(tree):
                    self.ff_result.insert("end", line + "\n")
            elif ext in ('.stockholm', '.sto', '.aln'):
                from ...core.file_formats import parse_stockholm
                data = parse_stockholm(path)
                self.ff_result.delete("1.0", "end")
                self.ff_result.insert("end", f"Sequences: {len(data['alignment'])}\n")
                for name, seq in list(data['alignment'].items())[:10]:
                    self.ff_result.insert("end", f"  {name}: {seq[:60]}{'...' if len(seq)>60 else ''}\n")
            else:
                self._msg_info("Info", f"Unknown format: {ext}\nSupported: .bed, .gff, .gtf, .nwk, .sto")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── API Keys Configuration Tab ─────────────────────────────────────────

    def _build_apikey_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['apikey'] = f
        self._section_header(f, "API Keys Configuration")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        self._label(inner, 'Configure API keys for external database services.', 'dim').pack(anchor='w', pady=(0, 12))
        self._label(inner, 'All keys are optional — features work without them, just slower.', 'dim').pack(anchor='w', pady=(0, 16))

        api_keys = config.get('api_keys', {})

        self._label(inner, 'NCBI Entrez (nucleotide/protein search)', 'sub').pack(anchor='w')
        nci_frame = ctk.CTkFrame(inner, fg_color='transparent')
        nci_frame.pack(fill='x', pady=(0, 8))
        self._label(nci_frame, 'Email:', 'small').pack(side='left')
        self.api_ncbi_email = self._input_entry(nci_frame, "your@email.com", width=280)
        self.api_ncbi_email.pack(side='left', padx=(4, 12))
        self.api_ncbi_email.insert(0, api_keys.get('ncbi_email', ''))
        self._label(nci_frame, 'API Key:', 'small').pack(side='left')
        self.api_ncbi_key = self._input_entry(nci_frame, "optional - for faster access", width=280)
        self.api_ncbi_key.pack(side='left', padx=4)
        self.api_ncbi_key.insert(0, api_keys.get('ncbi_api_key', ''))

        self._label(inner, 'UniProt (protein database)', 'sub').pack(anchor='w')
        uni_frame = ctk.CTkFrame(inner, fg_color='transparent')
        uni_frame.pack(fill='x', pady=(0, 8))
        self._label(uni_frame, 'Email:', 'small').pack(side='left')
        self.api_uniprot_email = self._input_entry(uni_frame, "your@email.com", width=280)
        self.api_uniprot_email.pack(side='left', padx=(4, 12))
        self.api_uniprot_email.insert(0, api_keys.get('uniprot_email', ''))

        self._label(inner, 'KEGG (pathway database)', 'sub').pack(anchor='w')
        kegg_frame = ctk.CTkFrame(inner, fg_color='transparent')
        kegg_frame.pack(fill='x', pady=(0, 8))
        self._label(kegg_frame, 'Email:', 'small').pack(side='left')
        self.api_kegg_email = self._input_entry(kegg_frame, "your@email.com", width=280)
        self.api_kegg_email.pack(side='left', padx=(4, 12))
        self.api_kegg_email.insert(0, api_keys.get('kegg_email', ''))

        self._label(inner, 'AlphaFold (structure prediction)', 'sub').pack(anchor='w')
        af_frame = ctk.CTkFrame(inner, fg_color='transparent')
        af_frame.pack(fill='x', pady=(0, 8))
        self._label(af_frame, 'Email:', 'small').pack(side='left')
        self.api_af_email = self._input_entry(af_frame, "your@email.com", width=280)
        self.api_af_email.pack(side='left', padx=(4, 12))
        self.api_af_email.insert(0, api_keys.get('alphafold_email', ''))

        btn_row = ctk.CTkFrame(inner, fg_color='transparent')
        btn_row.pack(fill='x', pady=(12, 0))
        self._action_button(btn_row, "Save All Keys", self._save_api_keys, 'success').pack(side='left')
        self._action_button(btn_row, "Clear All Keys", self._clear_api_keys, 'danger').pack(side='left', padx=(8, 0))

        self.apikey_status = self._label(inner, '', 'small')
        self.apikey_status.pack(anchor='w', pady=(8, 0))

    def _save_api_keys(self):
        config['api_keys'] = {
            'ncbi_email': self.api_ncbi_email.get().strip(),
            'ncbi_api_key': self.api_ncbi_key.get().strip(),
            'uniprot_email': self.api_uniprot_email.get().strip(),
            'kegg_email': self.api_kegg_email.get().strip(),
            'alphafold_email': self.api_af_email.get().strip(),
        }
        save_config(config)
        self.apikey_status.configure(text="API keys saved successfully!", text_color=self.T['success'])
        self._set_status("API keys saved")

    def _clear_api_keys(self):
        self.api_ncbi_email.delete(0, 'end')
        self.api_ncbi_key.delete(0, 'end')
        self.api_uniprot_email.delete(0, 'end')
        self.api_kegg_email.delete(0, 'end')
        self.api_af_email.delete(0, 'end')
        config['api_keys'] = {}
        save_config(config)
        self.apikey_status.configure(text="All keys cleared.", text_color=self.T['text_dim'])
        self._set_status("API keys cleared")
