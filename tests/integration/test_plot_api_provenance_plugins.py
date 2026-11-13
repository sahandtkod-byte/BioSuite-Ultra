"""
Tests for Phase 6 improvements:
- plot_api.py (interactive plots)
- provenance.py (workflow tracking)
- plugin.py (plugin architecture)
- file_formats.py (new formats)
- expression.py (NB GLM, normalization)
"""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


def _total_points(fig):
    """Total scatter points across all collections of a matplotlib figure."""
    return sum(len(c.get_offsets())
               for ax in fig.axes for c in ax.collections
               if hasattr(c, 'get_offsets'))


# ── Tests for plot_api.py ────────────────────────────────────────────────────

class TestPlotAPI:
    """Tests for the programmatic plot API."""

    def test_volcano_matplotlib(self):
        import matplotlib.pyplot as plt

        from biosuite.plotting.plot_api import volcano
        fc = np.random.randn(100)
        pvals = np.random.uniform(0, 1, 100)
        fig = volcano(fc, pvals, interactive=False)
        # Every point lands on the axes, split into significant/not, with a legend.
        assert _total_points(fig) == 100
        assert len(fig.axes[0].collections) == 2
        assert fig.axes[0].get_legend() is not None
        plt.close(fig)

    def test_volcano_plotly(self):
        try:
            import plotly

            from biosuite.plotting.plot_api import volcano
            fc = np.random.randn(100)
            pvals = np.random.uniform(0, 1, 100)
            fig = volcano(fc, pvals, interactive=True)
            assert len(fig.data) == 2
            assert sum(len(trace.x) for trace in fig.data) == 100
        except ImportError:
            pytest.skip("plotly not installed")

    def test_pca_matplotlib(self):
        import matplotlib.pyplot as plt

        from biosuite.plotting.plot_api import pca
        data = np.random.randn(30, 50)
        fig = pca(data, interactive=False)
        assert _total_points(fig) == 30
        plt.close(fig)

    def test_pca_with_groups(self):
        import matplotlib.pyplot as plt

        from biosuite.plotting.plot_api import pca
        data = np.random.randn(30, 50)
        groups = ['A'] * 15 + ['B'] * 15
        fig = pca(data, labels=groups, interactive=False)
        assert _total_points(fig) == 30
        assert len(fig.axes[0].collections) == 2
        assert fig.axes[0].get_legend() is not None
        plt.close(fig)

    def test_manhattan(self):
        import matplotlib.pyplot as plt

        from biosuite.plotting.plot_api import manhattan
        chroms = np.random.choice(['chr1', 'chr2', 'chr3'], 200)
        positions = np.random.randint(1, 1000000, 200)
        pvals = np.random.uniform(0, 1, 200)
        fig = manhattan(chroms, positions, pvals, interactive=False)
        assert _total_points(fig) == 200
        assert len(fig.axes[0].collections) == 3  # one per chromosome
        plt.close(fig)

    def test_heatmap(self):
        import matplotlib.pyplot as plt

        from biosuite.plotting.plot_api import heatmap
        data = np.random.randn(10, 8)
        fig = heatmap(data, interactive=False)
        assert len(fig.axes) == 2  # data axes + colour bar
        assert fig.axes[0].images[0].get_array().shape == (10, 8)
        plt.close(fig)

    def test_boxplot(self):
        import matplotlib.pyplot as plt

        from biosuite.plotting.plot_api import boxplot
        data = {'A': np.random.randn(30), 'B': np.random.randn(30)}
        fig = boxplot(data, interactive=False)
        assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ['A', 'B']
        plt.close(fig)

    def test_scatter(self):
        import matplotlib.pyplot as plt

        from biosuite.plotting.plot_api import scatter
        x = np.random.randn(100)
        y = x * 2 + np.random.randn(100)
        fig = scatter(x, y, interactive=False)
        assert _total_points(fig) == 100
        assert fig.axes[0].get_legend() is not None
        plt.close(fig)

    def test_barplot(self):
        import matplotlib.pyplot as plt

        from biosuite.plotting.plot_api import barplot
        categories = ['A', 'B', 'C', 'D']
        values = [10, 20, 15, 25]
        fig = barplot(categories, values, interactive=False)
        assert [p.get_height() for p in fig.axes[0].patches] == [10, 20, 15, 25]
        plt.close(fig)

    def test_violin(self):
        import matplotlib.pyplot as plt

        from biosuite.plotting.plot_api import violin
        data = {'A': np.random.randn(30), 'B': np.random.randn(30)}
        fig = violin(data, interactive=False)
        assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ['A', 'B']
        plt.close(fig)

    def test_timeseries(self):
        import matplotlib.pyplot as plt

        from biosuite.plotting.plot_api import timeseries
        x = np.arange(100)
        ys = [np.sin(x / 10), np.cos(x / 10)]
        fig = timeseries(x, ys, names=['sin', 'cos'], interactive=False)
        legend = fig.axes[0].get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ['sin', 'cos']
        plt.close(fig)

    def test_qqplot(self):
        import matplotlib.pyplot as plt

        from biosuite.plotting.plot_api import qqplot
        pvals = np.random.uniform(0, 1, 100)
        fig = qqplot(pvals, interactive=False)
        assert _total_points(fig) == 100
        plt.close(fig)

    def test_venn(self):
        import matplotlib.pyplot as plt

        from biosuite.plotting.plot_api import venn
        fig = venn([100, 50, 30], set_names=['A', 'B'], interactive=False)
        assert len(fig.axes[0].patches) == 2  # one circle per set
        labels = [t.get_text() for t in fig.axes[0].texts]
        assert 'A' in labels and 'B' in labels
        plt.close(fig)


# ── Tests for provenance.py ──────────────────────────────────────────────────

class TestProvenance:
    """Tests for the ProvenanceTracker."""

    def test_create_tracker(self):
        from biosuite.core.provenance import ProvenanceTracker
        tracker = ProvenanceTracker()
        assert tracker.session_id.startswith("session_")
        tracker.record("sequence", "gc_content", {"seq": "ATCG"}, "50.0%")
        assert len(tracker.get_steps()) == 1
        tracker.close()

    def test_record_step(self):
        from biosuite.core.provenance import ProvenanceTracker
        tracker = ProvenanceTracker()
        step = tracker.record("sequence", "gc_content", {"seq": "ATCG"}, "50.0%")
        assert step.step_id == 1
        assert step.module == "sequence"

    def test_get_steps(self):
        from biosuite.core.provenance import ProvenanceTracker
        tracker = ProvenanceTracker()
        tracker.record("seq", "gc", {"seq": "ATCG"}, "50%")
        tracker.record("seq", "revcomp", {"seq": "ATCG"}, "CGAT")
        steps = tracker.get_steps()
        assert len(steps) == 2

    def test_export_json(self):
        from biosuite.core.provenance import ProvenanceTracker
        tracker = ProvenanceTracker()
        tracker.record("seq", "gc", {"seq": "ATCG"}, "50%")

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            tracker.export_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data['total_steps'] == 1
        finally:
            os.unlink(path)

    def test_export_html(self):
        from biosuite.core.provenance import ProvenanceTracker
        tracker = ProvenanceTracker()
        tracker.record("seq", "gc", {"seq": "ATCG"}, "50%")

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = f.name
        try:
            tracker.export_html(path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "BioSuite Provenance Report" in content
        finally:
            os.unlink(path)

    def test_summary(self):
        from biosuite.core.provenance import ProvenanceTracker
        tracker = ProvenanceTracker()
        tracker.record("seq", "gc", {"seq": "ATCG"}, "50%")
        summary = tracker.summary()
        assert "Provenance Summary" in summary

    def test_decorator(self):
        from biosuite.core.provenance import ProvenanceTracker, tracked
        tracker = ProvenanceTracker()

        @tracked(tracker)
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10
        steps = tracker.get_steps()
        assert len(steps) == 1
        assert steps[0].function == "my_func"


# ── Tests for plugin.py ──────────────────────────────────────────────────────

class TestPlugin:
    """Tests for the plugin architecture."""

    def test_plugin_manager_create(self):
        from biosuite.core.plugin import PluginManager
        pm = PluginManager()
        assert pm.plugins == {}
        assert pm.loaded == {}

    def test_discover_plugins(self):
        from biosuite.core.plugin import PluginManager
        pm = PluginManager()
        assert pm.discover() == []  # no plugins installed in the probe environment

    def test_list_plugins_prints_a_table_or_the_install_hint(self, capsys):
        from biosuite.core.plugin import PluginManager
        pm = PluginManager()
        pm.discover()
        pm.list_plugins()
        output = capsys.readouterr().out

        if pm.plugins:
            assert "plugin(s)" in output
            assert all(info.name in output for info in pm.plugins.values())
        else:
            # The branch a fresh install sees: it has to say how to get a plugin.
            assert "No plugins found." in output
            assert "biosuite-plugin" in output

    def test_create_plugin_template(self):
        from biosuite.core.plugin import PluginManager
        pm = PluginManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            pm.create_plugin_template("test_plugin", tmpdir)
            plugin_dir = os.path.join(tmpdir, "biosuite-plugin-test_plugin")
            assert os.path.exists(plugin_dir)
            assert os.path.exists(os.path.join(plugin_dir, "test_plugin",
                                               "__init__.py"))
            assert os.path.exists(os.path.join(plugin_dir, "pyproject.toml"))


# ── Tests for file_formats.py (new formats) ─────────────────────────────────

class TestNewFormats:
    """Tests for new file format parsers."""

    def test_parse_gtf(self):
        from biosuite.core.file_formats import parse_gtf
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gtf', delete=False) as f:
            f.write("chr1\tensembl\tgene\t1000\t2000\t.\t+\t.\tgene_id \"GENE1\";\n")
            f.write("chr1\tensembl\texon\t1000\t1500\t.\t+\t.\tgene_id \"GENE1\";\n")
            path = f.name
        try:
            records = parse_gtf(path)
            assert len(records) == 2
            assert records[0].attributes.get('gene_id') == 'GENE1'
        finally:
            os.unlink(path)

    def test_parse_saf(self):
        from biosuite.core.file_formats import parse_saf
        with tempfile.NamedTemporaryFile(mode='w', suffix='.saf', delete=False) as f:
            f.write("GENE1\tchr1\t1000\t2000\t+\n")
            f.write("GENE2\tchr2\t3000\t4000\t-\n")
            path = f.name
        try:
            records = parse_saf(path)
            assert len(records) == 2
            assert records[0]['gene_id'] == 'GENE1'
        finally:
            os.unlink(path)

    def test_detect_file_format(self):
        from biosuite.core.file_formats import detect_file_format
        assert detect_file_format("test.fasta") == "fasta"
        assert detect_file_format("test.fa") == "fasta"
        assert detect_file_format("test.bam") == "bam"
        assert detect_file_format("test.cram") == "cram"
        assert detect_file_format("test.gtf") == "gtf"
        assert detect_file_format("test.vcf") == "vcf"
        assert detect_file_format("test.bed") == "bed"
        assert detect_file_format("test.unknown") == "unknown"

    def test_read_file_fasta(self):
        from biosuite.core.file_formats import read_file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">seq1\nATCGATCG\n>seq2\nGCTAGCTA\n")
            path = f.name
        try:
            result = read_file(path)
            assert result['format'] == 'fasta'
            assert len(result['records']) == 2
        finally:
            os.unlink(path)

    def test_read_file_bed(self):
        from biosuite.core.file_formats import read_file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as f:
            f.write("chr1\t1000\t2000\tgene1\t0\t+\n")
            path = f.name
        try:
            result = read_file(path)
            assert result['format'] == 'bed'
            assert len(result['records']) == 1
        finally:
            os.unlink(path)


# ── Tests for expression.py (new functions) ──────────────────────────────────

class TestExpressionAdvanced:
    """Tests for advanced expression analysis functions."""

    def test_deseq2_normalization(self):
        from biosuite.core.expression import deseq2_normalization
        np.random.seed(42)
        counts = pd.DataFrame({
            'gene': ['G1', 'G2', 'G3'],
            'S1': [100, 200, 300],
            'S2': [150, 250, 350],
            'S3': [120, 220, 320]
        })
        normalized = deseq2_normalization(counts)
        assert normalized.shape == counts.shape
        assert 'gene' in normalized.columns

    def test_vst(self):
        from biosuite.core.expression import variance_stabilizing_transformation
        np.random.seed(42)
        counts = pd.DataFrame({
            'gene': ['G1', 'G2', 'G3'],
            'S1': [100, 200, 300],
            'S2': [150, 250, 350]
        })
        vst = variance_stabilizing_transformation(counts)
        assert vst.shape == counts.shape
        # VST values should be positive
        assert (vst[['S1', 'S2']].values >= 0).all()

    def test_nb_differential_expression(self):
        from biosuite.core.expression import differential_expression
        np.random.seed(42)
        counts = pd.DataFrame({
            'gene': [f'G{i}' for i in range(100)],
            **{f'S{i}': np.random.poisson(100, 100) for i in range(6)}
        })
        # Add signal to first 10 genes
        counts.iloc[:10, 3:6] *= 3

        conditions = ['ctrl'] * 3 + ['treat'] * 3
        result = differential_expression(counts, conditions, method='nb')
        assert 'gene' in result.columns
        assert 'log2FC' in result.columns
        assert 'pvalue' in result.columns
        assert 'padj' in result.columns
        assert len(result) == 100


# ── Tests for PerformanceWarning ─────────────────────────────────────────────

class TestPerformanceWarning:
    """Tests for PerformanceWarning in dual-mode modules."""

    def test_blast_warning(self):
        import warnings

        from biosuite.core.blast import run_blast
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">query\nATCGATCGATCGATCG\n")
            query_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">db\nATCGATCGATCGATCG\n")
            db_path = f.name
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = run_blast(query_path, db_path)
                # Should emit PerformanceWarning since BLAST+ not installed
                perf_warnings = [x for x in w if issubclass(x.category, UserWarning)]
                assert len(perf_warnings) > 0
                assert result.num_hits >= 1, "the built-in engine found no self-hit"
        finally:
            os.unlink(query_path)
            os.unlink(db_path)
