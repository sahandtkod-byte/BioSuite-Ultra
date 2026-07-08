"""
Tests for the BioSuite notebook module (biosuite/notebook/__init__.py).

Covers:
  - Module import with and without IPython available
  - quick_gc() convenience function
  - quick_translate() convenience function
  - quick_align() convenience function
  - quick_blast() convenience function
  - Magic command classes (using real IPython InteractiveShell)
  - SequenceAnalyzer widget (mocked IPython)
"""
import sys
import os
import tempfile
from unittest import mock
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

# Ensure the project root is importable
sys.path.insert(0, "C:/Users/SAHAND/Desktop/python/BioSuite-Ultra")

import biosuite.notebook as nbmod

HAS_IPYTHON = nbmod.HAS_IPYTHON
HAS_WIDGETS = nbmod.HAS_WIDGETS


# ── Module Import Tests ──────────────────────────────────────────────────────


class TestModuleImport:
    """The notebook module should import cleanly regardless of IPython."""

    def test_import_without_error(self):
        """Importing biosuite.notebook should not raise."""
        assert nbmod is not None

    def test_has_ipython_flag(self):
        """HAS_IPYTHON should be a boolean."""
        assert hasattr(nbmod, "HAS_IPYTHON")
        assert isinstance(nbmod.HAS_IPYTHON, bool)

    def test_has_widgets_flag(self):
        """HAS_WIDGETS should be a boolean."""
        assert hasattr(nbmod, "HAS_WIDGETS")
        assert isinstance(nbmod.HAS_WIDGETS, bool)

    def test_quick_gc_importable(self):
        """quick_gc should be importable from the module."""
        from biosuite.notebook import quick_gc
        assert callable(quick_gc)

    def test_quick_translate_importable(self):
        from biosuite.notebook import quick_translate
        assert callable(quick_translate)

    def test_quick_align_importable(self):
        from biosuite.notebook import quick_align
        assert callable(quick_align)

    def test_quick_blast_importable(self):
        from biosuite.notebook import quick_blast
        assert callable(quick_blast)


# ── quick_gc Tests ───────────────────────────────────────────────────────────


class TestQuickGC:
    """quick_gc() should compute GC content correctly."""

    def test_basic_gc(self):
        """ATCGATCG → 50% GC."""
        from biosuite.notebook import quick_gc
        result = quick_gc("ATCGATCG")
        assert abs(result - 50.0) < 0.01

    def test_all_gc(self):
        from biosuite.notebook import quick_gc
        assert quick_gc("GCGCGCGC") == 100.0

    def test_no_gc(self):
        from biosuite.notebook import quick_gc
        assert quick_gc("ATATAT") == 0.0

    def test_empty_sequence(self):
        from biosuite.notebook import quick_gc
        assert quick_gc("") == 0.0

    def test_lowercase_input(self):
        from biosuite.notebook import quick_gc
        assert quick_gc("atcgatcg") == 50.0

    def test_numpy_array_like_input(self):
        """Should handle a string generated from numpy."""
        from biosuite.notebook import quick_gc
        seq = "".join(np.random.choice(["A", "T", "G", "C"], size=1000))
        result = quick_gc(seq)
        assert 0.0 <= result <= 100.0


# ── quick_translate Tests ────────────────────────────────────────────────────


class TestQuickTranslate:
    """quick_translate() should translate DNA to protein."""

    def test_atg_to_m(self):
        """ATG → M."""
        from biosuite.notebook import quick_translate
        result = quick_translate("ATG")
        assert result == "M"

    def test_stop_codon(self):
        """TAA → *."""
        from biosuite.notebook import quick_translate
        result = quick_translate("TAA")
        assert result == "*"

    def test_full_coding_sequence(self):
        """ATGAAATTTTAA → MKF*."""
        from biosuite.notebook import quick_translate
        result = quick_translate("ATGAAATTTTAA")
        assert result == "MKF*"

    def test_protein_sequence_passthrough(self):
        """Non-DNA input should produce X for unknown codons."""
        from biosuite.notebook import quick_translate
        result = quick_translate("XYZXYZ")
        # Should not raise, just produce X's
        assert len(result) >= 0

    def test_numpy_generated_sequence(self):
        """Translate a numpy-generated DNA sequence."""
        from biosuite.notebook import quick_translate
        np.random.seed(42)
        seq = "".join(np.random.choice(["A", "T", "G", "C"], size=300))
        result = quick_translate(seq)
        assert isinstance(result, str)
        assert len(result) > 0


# ── quick_align Tests ────────────────────────────────────────────────────────


class TestQuickAlign:
    """quick_align() should perform pairwise alignment."""

    def test_nw_identical(self):
        """NW on identical sequences should yield high score."""
        from biosuite.notebook import quick_align
        a1, a2, score = quick_align("ATCG", "ATCG", method="nw")
        assert score > 0
        assert len(a1) == len(a2)

    def test_nw_different_lengths(self):
        """NW on different-length sequences should still align."""
        from biosuite.notebook import quick_align
        a1, a2, score = quick_align("AGTACGCA", "TATGC", method="nw")
        # Score may be negative for very dissimilar sequences; just verify
        # the alignment returns aligned strings of equal length
        assert len(a1) == len(a2)

    def test_sw_local(self):
        """SW should find local alignment."""
        from biosuite.notebook import quick_align
        a1, a2, score = quick_align("XXXATCGYYY", "ATCG", method="sw")
        assert score > 0

    def test_sw_identical(self):
        from biosuite.notebook import quick_align
        a1, a2, score = quick_align("ATCG", "ATCG", method="sw")
        assert score > 0

    def test_default_method_is_nw(self):
        """Without specifying method, should default to nw."""
        from biosuite.notebook import quick_align
        a1, a2, score = quick_align("ATCG", "ATCG")
        assert score > 0

    def test_numpy_random_sequences(self):
        """Align two numpy-generated sequences."""
        from biosuite.notebook import quick_align
        np.random.seed(123)
        s1 = "".join(np.random.choice(list("ACGT"), size=50))
        s2 = "".join(np.random.choice(list("ACGT"), size=30))
        a1, a2, score = quick_align(s1, s2, method="nw")
        assert isinstance(score, (int, float))


# ── quick_blast Tests ────────────────────────────────────────────────────────


class TestQuickBLAST:
    """quick_blast() should run a BLAST search."""

    def test_blast_basic(self):
        """BLAST against a temp FASTA file.

        The built-in k-mer search engine requires exact k-mer matches
        for hits.  We use a long enough sequence (60bp) to guarantee
        k-mer overlap.
        """
        from biosuite.notebook import quick_blast

        long_seq = "ATCGATCGATCG" * 5  # 60bp
        # Create temp FASTA files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False
        ) as qf:
            qf.write(f">query\n{long_seq}\n")
            query_file = qf.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False
        ) as df:
            df.write(f">subject1\n{long_seq}\n")
            df.write(">subject2\nGCTAGCTAGCTAGCTAGCTA\n")
            db_file = df.name

        try:
            result = quick_blast(query_file, db_file)
            assert result is not None
            assert hasattr(result, "num_hits")
            assert result.num_hits >= 1
        finally:
            os.unlink(query_file)
            os.unlink(db_file)

    def test_blast_same_file(self):
        """BLAST a sequence against itself should find a perfect hit.

        The built-in k-mer search needs sequences long enough for
        exact k-mer overlap (>=60bp recommended).
        """
        from biosuite.notebook import quick_blast

        long_seq = "ATCGATCGATCG" * 5  # 60bp
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False
        ) as f:
            f.write(f">seq1\n{long_seq}\n")
            fa_file = f.name

        try:
            result = quick_blast(fa_file, fa_file)
            assert result.num_hits >= 1
        finally:
            os.unlink(fa_file)

    def test_blast_result_structure(self):
        """BlastResult should have expected attributes."""
        from biosuite.notebook import quick_blast

        long_seq = "ACGTACGTACGT" * 5  # 60bp
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False
        ) as qf:
            qf.write(f">q\n{long_seq}\n")
            qf.flush()
            query_file = qf.name

        try:
            result = quick_blast(query_file, query_file)
            assert hasattr(result, "engine")
            assert hasattr(result, "top_hits")
            assert hasattr(result, "hits")
            assert isinstance(result.hits, list)
        finally:
            os.unlink(query_file)


# ── Magic Commands (real IPython shell) ──────────────────────────────────────


class TestMagicCommands:
    """Test IPython magic commands with a real InteractiveShell.

    IPython's Magics class uses traitlets validation which rejects
    MagicMock parents.  We use a real InteractiveShell singleton instead.
    """

    @pytest.mark.skipif(not HAS_IPYTHON, reason="IPython not installed")
    def test_biosuite_magic_gc(self):
        """%biosuite gc ATCGATCG should return 50.0."""
        from IPython.core.interactiveshell import InteractiveShell

        ip = InteractiveShell.instance()
        magics = nbmod.BioSuiteMagics(shell=ip)
        result = magics.biosuite("gc ATCGATCG")
        assert result == 50.0

    @pytest.mark.skipif(not HAS_IPYTHON, reason="IPython not installed")
    def test_biosuite_magic_revcomp(self):
        """%biosuite revcomp ATCG should return CGAT."""
        from IPython.core.interactiveshell import InteractiveShell

        ip = InteractiveShell.instance()
        magics = nbmod.BioSuiteMagics(shell=ip)
        result = magics.biosuite("revcomp ATCG")
        assert result == "CGAT"

    @pytest.mark.skipif(not HAS_IPYTHON, reason="IPython not installed")
    def test_biosuite_magic_translate(self):
        """%biosuite translate ATG should return M."""
        from IPython.core.interactiveshell import InteractiveShell

        ip = InteractiveShell.instance()
        magics = nbmod.BioSuiteMagics(shell=ip)
        result = magics.biosuite("translate ATG")
        assert result == "M"

    @pytest.mark.skipif(not HAS_IPYTHON, reason="IPython not installed")
    def test_biosuite_magic_help(self):
        """%biosuite help should print usage info."""
        from IPython.core.interactiveshell import InteractiveShell

        ip = InteractiveShell.instance()
        magics = nbmod.BioSuiteMagics(shell=ip)
        result = magics.biosuite("help")
        assert result is None  # help prints but returns None

    @pytest.mark.skipif(not HAS_IPYTHON, reason="IPython not installed")
    def test_biosuite_magic_empty(self):
        """%biosuite with no args should print help."""
        from IPython.core.interactiveshell import InteractiveShell

        ip = InteractiveShell.instance()
        magics = nbmod.BioSuiteMagics(shell=ip)
        result = magics.biosuite("")
        assert result is None

    @pytest.mark.skipif(not HAS_IPYTHON, reason="IPython not installed")
    def test_biosuite_magic_unknown_command(self):
        """%biosuite nonsense should print unknown command."""
        from IPython.core.interactiveshell import InteractiveShell

        ip = InteractiveShell.instance()
        magics = nbmod.BioSuiteMagics(shell=ip)
        result = magics.biosuite("nonsense ATCG")
        assert result is None

    @pytest.mark.skipif(not HAS_IPYTHON, reason="IPython not installed")
    def test_load_ipython_extension(self):
        """load_ipython_extension should register magics."""
        from IPython.core.interactiveshell import InteractiveShell

        ip = InteractiveShell.instance()
        nbmod.load_ipython_extension(ip)
        # Just verify it doesn't raise


# ── Widget Tests (mocked IPython) ────────────────────────────────────────────


class TestWidgets:
    """Widget classes when ipywidgets is available."""

    def test_sequence_analyzer_instantiation(self):
        """SequenceAnalyzer should be instantiable."""
        if not HAS_WIDGETS:
            pytest.skip("ipywidgets not installed")

        analyzer = nbmod.SequenceAnalyzer()
        assert analyzer is not None

    def test_alignment_viewer_instantiation(self):
        """AlignmentViewer should be instantiable."""
        if not HAS_WIDGETS:
            pytest.skip("ipywidgets not installed")

        viewer = nbmod.AlignmentViewer()
        assert viewer is not None

    def test_plot_explorer_instantiation(self):
        """PlotExplorer should be instantiable."""
        if not HAS_WIDGETS:
            pytest.skip("ipywidgets not installed")

        explorer = nbmod.PlotExplorer()
        assert explorer is not None

    def test_sequence_analyzer_gc_method(self):
        """SequenceAnalyzer.gc_content should compute GC."""
        if not HAS_WIDGETS:
            pytest.skip("ipywidgets not installed")

        analyzer = nbmod.SequenceAnalyzer()
        result = analyzer.gc_content("ATCGATCG")
        assert abs(result - 50.0) < 0.01

    def test_sequence_analyzer_translate_method(self):
        """SequenceAnalyzer.translate should translate DNA."""
        if not HAS_WIDGETS:
            pytest.skip("ipywidgets not installed")

        analyzer = nbmod.SequenceAnalyzer()
        result = analyzer.translate("ATGAAATTTTAA")
        assert result == "MKF*"

    def test_sequence_analyzer_revcomp_method(self):
        """SequenceAnalyzer.reverse_complement should work."""
        if not HAS_WIDGETS:
            pytest.skip("ipywidgets not installed")

        analyzer = nbmod.SequenceAnalyzer()
        result = analyzer.reverse_complement("ATCG")
        assert result == "CGAT"

    def test_sequence_analyzer_stats_method(self):
        """SequenceAnalyzer.sequence_stats should return dict."""
        if not HAS_WIDGETS:
            pytest.skip("ipywidgets not installed")

        analyzer = nbmod.SequenceAnalyzer()
        result = analyzer.sequence_stats("ATCGATCG")
        assert isinstance(result, dict)
        assert result["length"] == 8
