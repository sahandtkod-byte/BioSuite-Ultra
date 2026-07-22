"""
Sequence analysis tabs: Sequence, Alignment, Phylogeny, ORF/Primers/Enzymes.
"""
import os
import customtkinter as ctk
from tkinter import filedialog

from ..themes import FONT_BODY, FONT_BUTTON, FONT_SMALL


class SequenceAnalysisTabMixin:
    """Provides Sequence, Alignment, Phylogeny, and ORF Tools tabs."""

    def _build_sequence_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['sequence'] = f
        self._section_header(f, "Sequence Analysis")
        body = ctk.CTkFrame(f, fg_color='transparent')
        body.pack(fill='both', expand=True)

        left = self._card(body)
        left.pack(side='left', fill='both', expand=True, padx=(0, 6))
        self._label(left, 'Input Sequence (FASTA/FASTQ)', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.seq_text = self._text_box(left, height=280)
        self.seq_text.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        btn_row = ctk.CTkFrame(left, fg_color='transparent')
        btn_row.pack(fill='x', padx=10, pady=(0, 10))
        self._action_button(btn_row, "Load File", self._load_seq_file).pack(side='left', padx=(0, 6))
        self._action_button(btn_row, "Clear", lambda: self.seq_text.delete("1.0", "end"), 'danger').pack(side='left')

        right = self._card(body)
        right.pack(side='right', fill='both', expand=True, padx=(6, 0))
        self._label(right, 'Results & Statistics', 'sub').pack(padx=14, pady=(14, 4), anchor='w')
        self.seq_stats = self._text_box(right, height=280)
        self.seq_stats.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        btn_row2 = ctk.CTkFrame(right, fg_color='transparent')
        btn_row2.pack(fill='x', padx=10, pady=(0, 10))
        for label, cmd in [("GC%", self._seq_gc), ("Rev Comp", self._seq_revcomp),
                           ("Translate", self._seq_translate), ("Stats", self._seq_stats_cmd)]:
            self._action_button(btn_row2, label, cmd).pack(side='left', padx=(0, 6))

    def _load_seq_file(self):
        path = filedialog.askopenfilename(filetypes=[
            ("FASTA/FASTQ/GenBank", "*.fasta *.fa *.fq *.fastq *.gb *.genbank")])
        if not path:
            return
        try:
            from ...core.sequence import read_fasta, read_fastq, read_genbank
            if path.endswith(('.fasta', '.fa')):
                data = read_fasta(path)
                content = "\n".join(f">{name}\n{seq}" for name, seq in data) if data else ""
            elif path.endswith(('.fastq', '.fq')):
                data = read_fastq(path)
                content = "\n".join(f"@{name}\n{seq}\n+\n{qual}" for name, seq, qual in data) if data else ""
            else:
                data = read_genbank(path)
                content = "\n".join(f">{name}\n{seq}" for name, seq, _ in data) if data else ""
            self.seq_text.delete("1.0", "end")
            self.seq_text.insert("end", content)
            self._set_status(f"Loaded {os.path.basename(path)}")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _get_seq(self):
        text = self.seq_text.get("1.0", "end").strip()
        return ''.join(line.strip() for line in text.split('\n')
                       if not line.startswith(('>', '@', '+'))).replace(' ', '')

    def _seq_gc(self):
        from ...core.sequence import gc_content, sequence_stats
        seq = self._get_seq()
        if not seq:
            self._msg_warning("No sequence", "Enter or load a sequence first.")
            return
        self.seq_stats.delete("1.0", "end")
        self.seq_stats.insert("end", f"Length: {len(seq)}\n")
        self.seq_stats.insert("end", f"GC%: {gc_content(seq):.2f}%\n")
        s = sequence_stats(seq)
        self.seq_stats.insert("end", f"A:{s['A']} T:{s['T']} G:{s['G']} C:{s['C']} N:{s['N']}\n")

    def _seq_revcomp(self):
        from ...core.sequence import reverse_complement
        seq = self._get_seq()
        if not seq:
            self._msg_warning("No sequence", "Enter or load a sequence first.")
            return
        rc = reverse_complement(seq)
        self.seq_stats.delete("1.0", "end")
        self.seq_stats.insert("end", f"Original ({len(seq)} bp):\n{seq[:200]}{'...' if len(seq)>200 else ''}\n\n")
        self.seq_stats.insert("end", f"Reverse Complement ({len(rc)} bp):\n{rc[:200]}{'...' if len(rc)>200 else ''}\n")

    def _seq_translate(self):
        from ...core.sequence import translate
        seq = self._get_seq()
        if not seq:
            self._msg_warning("No sequence", "Enter or load a sequence first.")
            return
        prot = translate(seq, frame=1)
        self.seq_stats.delete("1.0", "end")
        self.seq_stats.insert("end", f"Translation (frame 1, {len(prot)} aa):\n{prot[:300]}{'...' if len(prot)>300 else ''}\n")

    def _seq_stats_cmd(self):
        from ...core.sequence import sequence_stats
        seq = self._get_seq()
        if not seq:
            self._msg_warning("No sequence", "Enter or load a sequence first.")
            return
        stats = sequence_stats(seq)
        self.seq_stats.delete("1.0", "end")
        for k, v in stats.items():
            self.seq_stats.insert("end", f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")

    # ─── Alignment Tab ────────────────────────────────────────────────────────

    def _build_alignment_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['alignment'] = f
        self._section_header(f, "Pairwise Sequence Alignment")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        self._label(inner, 'Sequence 1', 'sub').pack(anchor='w')
        self.align_seq1 = self._text_box(inner, height=80)
        self.align_seq1.pack(fill='x', pady=(0, 8))
        self._label(inner, 'Sequence 2', 'sub').pack(anchor='w')
        self.align_seq2 = self._text_box(inner, height=80)
        self.align_seq2.pack(fill='x', pady=(0, 8))

        btn_row = ctk.CTkFrame(inner, fg_color='transparent')
        btn_row.pack(fill='x', pady=(0, 8))
        self._action_button(btn_row, "Needleman-Wunsch (Global)", self._align_nw).pack(side='left', padx=(0, 8))
        self._action_button(btn_row, "Smith-Waterman (Local)", self._align_sw, 'accent_dim').pack(side='left')

        self.align_result = self._text_box(inner, height=120)
        self.align_result.pack(fill='both', expand=True)

    def _get_align_seq(self, widget):
        return widget.get("1.0", "end").strip().replace('\n', '').replace(' ', '')

    def _align_nw(self):
        from ...core.alignment import needleman_wunsch
        s1 = self._get_align_seq(self.align_seq1)
        s2 = self._get_align_seq(self.align_seq2)
        if not s1 or not s2:
            self._msg_warning("Missing", "Enter both sequences.")
            return
        a1, a2, score = needleman_wunsch(s1, s2)
        self.align_result.delete("1.0", "end")
        self.align_result.insert("end", f"Score: {score}\n\nSeq1: {a1}\nSeq2: {a2}\n")

    def _align_sw(self):
        from ...core.alignment import smith_waterman
        s1 = self._get_align_seq(self.align_seq1)
        s2 = self._get_align_seq(self.align_seq2)
        if not s1 or not s2:
            self._msg_warning("Missing", "Enter both sequences.")
            return
        a1, a2, score = smith_waterman(s1, s2)
        self.align_result.delete("1.0", "end")
        self.align_result.insert("end", f"Score: {score}\n\nSeq1: {a1}\nSeq2: {a2}\n")

    # ─── Phylogeny Tab ────────────────────────────────────────────────────────

    def _build_phylogeny_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['phylogeny'] = f
        self._section_header(f, "Phylogenetic Tree Builder")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        self._label(inner, 'Aligned Sequences (FASTA format)', 'sub').pack(anchor='w')
        self.phylo_input = self._text_box(inner, height=160)
        self.phylo_input.pack(fill='x', pady=(0, 8))
        self._action_button(inner, "Build UPGMA Tree", self._build_tree).pack(pady=(0, 8))
        self.phylo_result = self._text_box(inner, height=140)
        self.phylo_result.pack(fill='both', expand=True)

    def _build_tree(self):
        from ...core.phylogeny import distance_matrix, upgma_tree, plot_phylogenetic_tree
        text = self.phylo_input.get("1.0", "end").strip()
        if not text:
            self._msg_warning("No data", "Enter aligned sequences (>name and sequence).")
            return
        lines = text.split('\n')
        names, seqs, current = [], [], []
        for line in lines:
            if line.startswith('>'):
                if current:
                    seqs.append(''.join(current))
                    current = []
                names.append(line[1:].strip())
            else:
                current.append(line.strip())
        if current:
            seqs.append(''.join(current))
        if len(seqs) < 2:
            self._msg_error("Error", "Need at least 2 sequences.")
            return
        if len(set(len(s) for s in seqs)) != 1:
            self._msg_error("Error", "Sequences must be same length (aligned).")
            return
        try:
            dist_mat = distance_matrix(seqs)
            link_mat = upgma_tree(dist_mat, names)
            fig = plot_phylogenetic_tree(link_mat, names)
            fig.savefig(r'C:/Users/SAHAND/AppData/Local/Temp/biosuite_phylo.png', dpi=150, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close('all')
            self.phylo_result.delete("1.0", "end")
            self.phylo_result.insert("end", "Distance Matrix:\n")
            for i, name in enumerate(names):
                row = ", ".join(f"{dist_mat[i][j]:.3f}" for j in range(len(names)))
                self.phylo_result.insert("end", f"{name}: {row}\n")
            self.phylo_result.insert("end", "\nTree displayed in matplotlib window.")
        except Exception as e:
            self._msg_error("Error", str(e))

    # ─── ORF/Primers/Enzymes Tab ────────────────────────────────────────────

    def _build_orftools_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['orftools'] = f
        self._section_header(f, "ORF Finder / Primer Design / Restriction Enzymes")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        self._label(inner, 'DNA Sequence:', 'sub').pack(anchor='w')
        self.orf_seq = self._text_box(inner, height=100)
        self.orf_seq.pack(fill='x', pady=(0, 8))

        btn_row = ctk.CTkFrame(inner, fg_color='transparent')
        btn_row.pack(fill='x', pady=(0, 8))
        self._action_button(btn_row, "Find ORFs", self._run_orf).pack(side='left', padx=(0, 8))
        self._action_button(btn_row, "Find Restriction Sites", self._run_restriction).pack(side='left', padx=(0, 8))
        self._action_button(btn_row, "Design Primers", self._run_primers, 'accent_dim').pack(side='left')

        self.orf_result = self._text_box(inner, height=200)
        self.orf_result.pack(fill='both', expand=True)

    def _run_orf(self):
        from ...core.orf_finder import find_orfs, format_orf_results
        seq = self.orf_seq.get("1.0", "end").strip()
        if not seq:
            self._msg_warning("No sequence", "Enter a DNA sequence.")
            return
        try:
            orfs = find_orfs(seq.upper())
            self.orf_result.delete("1.0", "end")
            self.orf_result.insert("end", format_orf_results(orfs))
            self._set_status(f"Found {len(orfs)} ORFs")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _run_restriction(self):
        from ...core.orf_finder import find_restriction_sites, format_restriction_sites
        seq = self.orf_seq.get("1.0", "end").strip()
        if not seq:
            self._msg_warning("No sequence", "Enter a DNA sequence.")
            return
        try:
            sites = find_restriction_sites(seq.upper())
            self.orf_result.delete("1.0", "end")
            self.orf_result.insert("end", format_restriction_sites(sites))
            self._set_status(f"Found {len(sites)} restriction sites")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _run_primers(self):
        from ...core.orf_finder import design_primers, format_primers
        seq = self.orf_seq.get("1.0", "end").strip()
        if not seq:
            self._msg_warning("No sequence", "Enter a template DNA sequence.")
            return
        try:
            fwd, rev = design_primers(seq.upper())
            self.orf_result.delete("1.0", "end")
            self.orf_result.insert("end", format_primers(fwd, rev))
        except Exception as e:
            self._msg_error("Error", str(e))
