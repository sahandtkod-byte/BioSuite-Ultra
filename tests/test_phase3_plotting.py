"""
Unit tests for Phase 3 visualization modules:
  - upset_plots.py
  - genome_browser.py
  - interactive_plots.py
  - conservation_plots.py
  - synteny.py
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biosuite.plotting.upset_plots import (
    compute_upset_matrix, plot_upset, upset_from_sets,
    compute_set_statistics, plot_set_sizes
)
from biosuite.plotting.genome_browser import (
    parse_bed, parse_vcf, compute_coverage, plot_genome_tracks,
    _parse_cigar_length
)
from biosuite.plotting.interactive_plots import (
    interactive_scatter, interactive_bar, interactive_heatmap,
    interactive_volcano, interactive_line, interactive_pie,
    HAS_PLOTLY
)
from biosuite.plotting.conservation_plots import (
    compute_logo_heights, plot_sequence_logo, plot_conservation_bar,
    plot_logo_with_conservation, compute_conservation_scores,
    plot_motif_enrichment
)
from biosuite.plotting.synteny import (
    compute_dotplot, plot_dotplot, compute_synteny_score,
    plot_synteny_dotplot, plot_gene_order, plot_synteny
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─── UpSet Plots ───────────────────────────────────────────

class TestUpsetPlots:
    def test_compute_upset_matrix(self):
        sets = {'A': {1, 2, 3}, 'B': {2, 3, 4}, 'C': {3, 4, 5}}
        labels, matrix, counts = compute_upset_matrix(sets)
        assert labels == ['A', 'B', 'C']
        assert len(matrix) == len(counts)
        assert all(c > 0 for c in counts)

    def test_two_sets(self):
        sets = {'X': {1, 2}, 'Y': {2, 3}}
        labels, matrix, counts = compute_upset_matrix(sets)
        assert len(labels) == 2
        # Elements: X only=1, Y only=1, intersection=1
        total = sum(counts)
        assert total == 3

    def test_empty_intersection(self):
        sets = {'A': {1, 2}, 'B': {3, 4}}
        labels, matrix, counts = compute_upset_matrix(sets)
        # No overlap, so only 2 rows (A only, B only)
        assert sum(counts) == 4

    def test_plot_upset_returns_figure(self):
        sets = {'A': {1, 2, 3}, 'B': {2, 3, 4}, 'C': {3, 4, 5}}
        fig = plot_upset(sets)
        assert fig is not None
        plt.close(fig)

    def test_upset_from_sets(self):
        fig = upset_from_sets({1, 2, 3}, {2, 3, 4}, names=['S1', 'S2'])
        assert fig is not None
        plt.close(fig)

    def test_set_statistics(self):
        sets = {'A': {1, 2, 3}, 'B': {2, 3, 4}}
        stats = compute_set_statistics(sets)
        assert stats['sizes']['A'] == 3
        assert stats['total_union'] == 4
        assert stats['total_intersection'] == 2

    def test_plot_set_sizes(self):
        fig, ax = plt.subplots()
        plot_set_sizes({'A': {1, 2}, 'B': {3}}, ax=ax)
        plt.close(fig)


# ─── Genome Browser ────────────────────────────────────────

class TestGenomeBrowser:
    def test_parse_bed(self, tmp_path):
        bed_file = tmp_path / "test.bed"
        bed_file.write_text("chr1\t100\t200\tgeneA\t10\nchr1\t300\t500\tgeneB\t20\n")
        regions = parse_bed(str(bed_file))
        assert len(regions) == 2
        assert regions[0] == ('chr1', 100, 200, 'geneA', 10.0)

    def test_parse_vcf(self, tmp_path):
        vcf_file = tmp_path / "test.vcf"
        vcf_file.write_text(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr1\t100\t.\tA\tG\t30\tPASS\t.\n"
        )
        variants = parse_vcf(str(vcf_file))
        assert len(variants) == 1
        assert variants[0]['ref'] == 'A'
        assert variants[0]['alt'] == 'G'

    def test_parse_cigar_length(self):
        assert _parse_cigar_length("100M") == 100
        assert _parse_cigar_length("50M10I30M") == 80  # M + M = 80
        assert _parse_cigar_length("100D50M") == 150

    def test_plot_genome_tracks(self):
        tracks = [
            {'type': 'coverage', 'name': 'Cov',
             'data': {'positions': np.array([0, 100, 200]),
                      'coverage': np.array([5, 10, 3])},
             'color': '#00ff88'},
            {'type': 'bed', 'name': 'BED',
             'data': {'regions': [('chr1', 50, 150, 'gene1', 0),
                                   ('chr1', 180, 250, 'gene2', 0)]},
             'color': '#4ecdc4'},
        ]
        fig = plot_genome_tracks(tracks, title="Test Browser")
        assert fig is not None
        plt.close(fig)

    def test_empty_tracks(self):
        fig = plot_genome_tracks([])
        assert fig is None


# ─── Interactive Plots ─────────────────────────────────────

class TestInteractivePlots:
    def _close_fig(self, fig):
        try:
            import plotly.graph_objects as go
            if isinstance(fig, go.Figure):
                return
        except ImportError:
            pass
        plt.close(fig)

    def test_scatter(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 1, 5, 3])
        fig = interactive_scatter(x, y)
        assert fig is not None
        self._close_fig(fig)

    def test_scatter_with_colors(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 1, 5, 3])
        colors = ['A', 'A', 'B', 'B', 'B']
        fig = interactive_scatter(x, y, color_col=colors)
        assert fig is not None
        self._close_fig(fig)

    def test_bar(self):
        fig = interactive_bar(['a', 'b', 'c'], [1, 2, 3])
        assert fig is not None
        self._close_fig(fig)

    def test_heatmap(self):
        data = np.random.rand(4, 4)
        fig = interactive_heatmap(data, row_labels=['r1', 'r2', 'r3', 'r4'],
                                   col_labels=['c1', 'c2', 'c3', 'c4'])
        assert fig is not None
        self._close_fig(fig)

    def test_volcano(self):
        lfc = np.array([0.5, 2.0, -2.5, 0.1, 3.0])
        pvals = np.array([0.5, 0.01, 0.001, 0.8, 0.0001])
        fig = interactive_volcano(lfc, pvals)
        assert fig is not None
        self._close_fig(fig)

    def test_line(self):
        x = [1, 2, 3, 4]
        ys = [[1, 2, 3, 4], [4, 3, 2, 1]]
        fig = interactive_line(x, ys)
        assert fig is not None
        self._close_fig(fig)

    def test_pie(self):
        fig = interactive_pie(['A', 'B', 'C'], [30, 40, 30])
        assert fig is not None
        self._close_fig(fig)


# ─── Conservation Plots ────────────────────────────────────

class TestConservationPlots:
    def test_logo_heights(self):
        seqs = ['ACGT', 'ACGT', 'ACGT', 'ACGT']
        positions, heights, total_h = compute_logo_heights(seqs)
        assert len(positions) == 4
        # Fully conserved: max bits = 2, with small-sample correction ~1.46
        assert all(h > 1.4 for h in total_h)

    def test_logo_with_gaps(self):
        seqs = ['ACGT', 'AC--', 'A---', 'ACGT']
        positions, heights, total_h = compute_logo_heights(seqs)
        # Position 0 (A) should be highly conserved
        assert total_h[0] > 0

    def test_plot_sequence_logo(self):
        seqs = ['ACGTACGT', 'ACGAACGT', 'ACGTACGA', 'ACGTACGT']
        fig = plot_sequence_logo(seqs)
        assert fig is not None
        plt.close(fig)

    def test_conservation_bar(self):
        seqs = ['ACGT'] * 10
        fig = plot_conservation_bar(seqs)
        assert fig is not None
        plt.close(fig)

    def test_logo_with_conservation(self):
        seqs = ['ACGTACGT'] * 5
        fig = plot_logo_with_conservation(seqs)
        assert fig is not None
        plt.close(fig)

    def test_conservation_scores(self):
        seqs = ['AAAA', 'AAAG', 'AATT']
        scores = compute_conservation_scores(seqs)
        assert len(scores) == 4
        # Position 0 (all A) should have high conservation
        assert scores[0][1] > 0.5

    def test_motif_enrichment(self):
        seqs = ['ATGCGATG', 'GCGATGCG', 'XXXXXXX']
        fig = plot_motif_enrichment(seqs, ['ATG', 'GCG'])
        assert fig is not None
        plt.close(fig)


# ─── Synteny ───────────────────────────────────────────────

class TestSynteny:
    def test_dotplot_exact(self):
        seq1 = "ACGTACGTACGT"
        seq2 = "ACGTACGTACGT"
        hits = compute_dotplot(seq1, seq2, word_size=4)
        assert len(hits) > 0

    def test_dotplot_different(self):
        seq1 = "AAAAAAAAAAAA"
        seq2 = "TTTTTTTTTTTT"
        hits = compute_dotplot(seq1, seq2, word_size=4)
        assert len(hits) == 0

    def test_plot_dotplot(self):
        fig = plot_dotplot("ACGTACGT", "ACGTACGT", word_size=2)
        assert fig is not None
        plt.close(fig)

    def test_synteny_score(self):
        genes1 = ['A', 'B', 'C', 'D']
        genes2 = ['A', 'B', 'C', 'D']
        score, pairs = compute_synteny_score(genes1, genes2)
        assert score == 1.0  # Perfect collinearity

    def test_synteny_score_reversed(self):
        genes1 = ['A', 'B', 'C']
        genes2 = ['C', 'B', 'A']
        score, pairs = compute_synteny_score(genes1, genes2)
        assert score < 0.5

    def test_plot_synteny_dotplot(self):
        fig = plot_synteny_dotplot(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'])
        assert fig is not None
        plt.close(fig)

    def test_plot_gene_order(self):
        fig = plot_gene_order(['Gene1', 'Gene2', 'Gene3'])
        assert fig is not None
        plt.close(fig)

    def test_plot_synteny(self):
        fig = plot_synteny(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'])
        assert fig is not None
        plt.close(fig)
