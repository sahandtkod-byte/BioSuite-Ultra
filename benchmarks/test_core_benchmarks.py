"""Benchmark tests for BioSuite core operations.

Run with: pytest benchmarks/ --benchmark-only
"""
import pytest


class BenchmarkSequence:
    """Benchmarks for sequence operations."""

    def test_gc_content_1kb(self, benchmark):
        from biosuite.core.sequence import gc_content
        seq = "ATCGATCG" * 125  # 1000 bp
        benchmark(gc_content, seq)

    def test_gc_content_10kb(self, benchmark):
        from biosuite.core.sequence import gc_content
        seq = "ATCGATCG" * 1250  # 10000 bp
        benchmark(gc_content, seq)

    def test_reverse_complement_1kb(self, benchmark):
        from biosuite.core.sequence import reverse_complement
        seq = "ATCGATCG" * 125
        benchmark(reverse_complement, seq)

    def test_translate_1kb(self, benchmark):
        from biosuite.core.sequence import translate
        seq = "ATGAAATTTTAA" * 83  # ~1000 bp
        benchmark(translate, seq)

    def test_sequence_stats_1kb(self, benchmark):
        from biosuite.core.sequence import sequence_stats
        seq = "ATCGATCG" * 125
        benchmark(sequence_stats, seq)


class BenchmarkAlignment:
    """Benchmarks for alignment operations."""

    def test_nw_50bp(self, benchmark):
        from biosuite.core.alignment import needleman_wunsch
        s1 = "ACGTACGT" * 6 + "AC"   # 50bp
        s2 = "ACGACGT" * 7 + "GT"    # 50bp
        benchmark(needleman_wunsch, s1, s2)

    def test_sw_50bp(self, benchmark):
        from biosuite.core.alignment import smith_waterman
        s1 = "ACGTACGT" * 6 + "AC"
        s2 = "ACGACGT" * 7 + "GT"
        benchmark(smith_waterman, s1, s2)


class BenchmarkExpression:
    """Benchmarks for expression analysis operations."""

    def test_cpm_1000_genes(self, benchmark):
        import pandas as pd
        import numpy as np
        from biosuite.core.expression import cpm_normalization
        np.random.seed(42)
        df = pd.DataFrame(np.random.poisson(100, (1000, 6)),
                          columns=[f"s{i}" for i in range(6)],
                          index=[f"g{i}" for i in range(1000)])
        benchmark(cpm_normalization, df)

    def test_tpm_1000_genes(self, benchmark):
        import pandas as pd
        import numpy as np
        from biosuite.core.expression import tpm_normalization
        np.random.seed(42)
        df = pd.DataFrame(np.random.poisson(100, (1000, 6)),
                          columns=[f"s{i}" for i in range(6)],
                          index=[f"g{i}" for i in range(1000)])
        gene_lengths = np.random.randint(500, 5000, 1000).astype(float)
        benchmark(tpm_normalization, df, gene_lengths)
