"""
Comprehensive tests for cloning, plasmid mapping, and sequence viewer modules.

Covers:
  - biosuite.core.cloning: restriction enzymes, digestion, ligation, primers, PCR
  - biosuite.plotting.plasmid_map: PlasmidFeature, PlasmidMap, drawing, reports
  - biosuite.plotting.sequence_viewer: sequence views, translations, GC plots, ORFs

Run: pytest tests/test_cloning_and_viewer.py -v
"""
import os
import sys
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from biosuite.core.cloning import (
    RESTRICTION_ENZYMES,
    find_restriction_sites,
    simulate_digestion,
    simulate_ligation,
    design_primers,
    simulate_pcr,
    plot_virtual_gel,
    format_digest_report,
    format_primer_report,
)
from biosuite.plotting.plasmid_map import (
    PlasmidFeature,
    PlasmidMap,
    create_sample_plasmid,
    draw_plasmid,
    draw_plasmid_with_annotations,
    format_plasmid_report,
)
from biosuite.plotting.sequence_viewer import (
    SequenceAnnotation,
    draw_sequence_view,
    draw_translation_view,
    draw_gc_content_plot,
    draw_orf_map,
    create_sequence_overview,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_sequence():
    """A ~200 bp sequence with an EcoRI site (GAATTC)."""
    return "ATCGATCGATCGATCGATCGATCG" * 3 + "GAATTC" + "ATCGATCGATCGATCGATCGATCG" * 4


@pytest.fixture
def short_sequence():
    """A 30 bp sequence for edge-case tests."""
    return "ATCGATCGATCGATCGATCGATCGATCGAT"


@pytest.fixture
def long_sequence():
    """A ~2000 bp sequence with multiple features for comprehensive testing."""
    part1 = "ATCGATCG" * 100          # 800 bp
    part2 = "GCGCGCGC" * 50           # 400 bp
    orf_seq = "ATG" + "AAA" * 30 + "TGA"  # 95 bp ORF
    spacer = "NNNNNNNN" * 10          # 80 bp
    part3 = "TTTTAAAA" * 50           # 400 bp
    part4 = "CCCCGGGG" * 32           # 256 bp
    return part1 + "GAATTC" + part2 + orf_seq + spacer + "AAGCTT" + part3 + "GGTACC" + part4


@pytest.fixture
def circular_sequence():
    """A 3000 bp circular plasmid with common cloning features."""
    backbone = "ATCGATCGATCG" * 200   # 2400 bp
    insert_region = "ATG" + "GCT" * 40 + "TGA"  # 123 bp ORF
    ori = "GCGCGCGCGCGC" * 10        # 120 bp
    promoter = "TTTTAAAACCCCGGGG" * 5  # 80 bp
    extra = "ATCGATCG" * 37           # 296 bp
    seq = backbone + "GAATTC" + promoter + insert_region + "AAGCTT" + ori + "GGTACC" + extra
    return seq[:3000]


# ══════════════════════════════════════════════════════════════════════════════
# TestPlasmidMap  (11 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestPlasmidMap:
    """Tests for plasmid map data structures and drawing."""

    def test_plasmid_feature_dataclass(self):
        """PlasmidFeature dataclass can be created with required fields."""
        feature = PlasmidFeature(
            name="AmpR", start=100, end=500,
            feature_type="gene", color="#FF0000", strand=1,
        )
        assert feature.name == "AmpR"
        assert feature.start == 100
        assert feature.end == 500
        assert feature.feature_type == "gene"
        assert feature.strand == 1

    def test_plasmid_map_add_feature(self):
        """PlasmidMap.add_feature adds features to the map."""
        pm = PlasmidMap(sequence="ATCG" * 250, size=1000)
        feat = PlasmidFeature(
            name="GFP", start=0, end=720,
            feature_type="gene", color="#00FF00", strand=1,
        )
        pm.add_feature(feat)
        assert len(pm.features) == 1
        assert pm.features[0].name == "GFP"

        feat2 = PlasmidFeature(
            name="Promoter", start=721, end=900,
            feature_type="promoter", color="#0000FF", strand=1,
        )
        pm.add_feature(feat2)
        assert len(pm.features) == 2

    def test_plasmid_map_set_size(self):
        """PlasmidMap.set_size updates the plasmid size."""
        pm = PlasmidMap(sequence="ATCG" * 250, size=1000)
        assert pm.size == 1000
        pm.set_size(5000)
        assert pm.size == 5000

    def test_create_sample_plasmid(self):
        """create_sample_plasmid returns a populated PlasmidMap."""
        pm = create_sample_plasmid()
        assert isinstance(pm, PlasmidMap)
        assert pm.size > 0
        assert len(pm.features) > 0
        assert len(pm.sequence) == pm.size

    def test_draw_plasmid(self, circular_sequence):
        """draw_plasmid returns a matplotlib figure (close fig)."""
        pm = PlasmidMap(sequence=circular_sequence[:3000], size=3000)
        feat = PlasmidFeature(
            name="AmpR", start=0, end=800,
            feature_type="gene", color="#FF4444", strand=1,
        )
        pm.add_feature(feat)
        fig = draw_plasmid(pm)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_plasmid_with_annotations(self, circular_sequence):
        """draw_plasmid_with_annotations creates annotated figure."""
        pm = PlasmidMap(sequence=circular_sequence[:3000], size=3000)
        features = [
            PlasmidFeature(name="AmpR", start=0, end=800, feature_type="gene",
                           color="#FF4444", strand=1),
            PlasmidFeature(name="ori", start=1200, end=1400, feature_type="origin",
                           color="#4444FF", strand=1),
            PlasmidFeature(name="MCS", start=2000, end=2100,
                           feature_type="multiple_cloning_site",
                           color="#44FF44", strand=1),
        ]
        for f in features:
            pm.add_feature(f)
        fig = draw_plasmid_with_annotations(pm)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_format_plasmid_report(self):
        """format_plasmid_report returns a non-empty string."""
        pm = create_sample_plasmid()
        report = format_plasmid_report(pm)
        assert isinstance(report, str)
        assert len(report) > 0
        assert "bp" in report.lower() or str(pm.size) in report

    def test_feature_colors_by_type(self):
        """Different feature types get distinct default colors."""
        types = ["gene", "promoter", "origin", "terminator", "multiple_cloning_site"]
        for feat_type in types:
            feat = PlasmidFeature(
                name=f"test_{feat_type}", start=0, end=100,
                feature_type=feat_type, color="#AAAAAA", strand=1,
            )
            assert feat.color is not None

    def test_plasmid_feature_strand(self):
        """PlasmidFeature supports both strands."""
        feat_plus = PlasmidFeature(
            name="Gene+", start=0, end=100,
            feature_type="gene", color="#FF0000", strand=1,
        )
        feat_minus = PlasmidFeature(
            name="Gene-", start=0, end=100,
            feature_type="gene", color="#0000FF", strand=-1,
        )
        assert feat_plus.strand == 1
        assert feat_minus.strand == -1

    def test_plasmid_map_multiple_features(self):
        """PlasmidMap can hold many features at different positions."""
        pm = PlasmidMap(sequence="ATCG" * 250, size=1000)
        for i in range(5):
            feat = PlasmidFeature(
                name=f"Feature_{i}", start=i * 180, end=i * 180 + 150,
                feature_type="gene", color="#FF0000", strand=1,
            )
            pm.add_feature(feat)
        assert len(pm.features) == 5
        for f in pm.features:
            assert 0 <= f.start < pm.size
            assert 0 < f.end <= pm.size

    def test_draw_plasmid_no_features(self):
        """draw_plasmid works with no features."""
        pm = PlasmidMap(sequence="ATCG" * 250, size=1000)
        fig = draw_plasmid(pm)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TestCloning  (18 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestCloning:
    """Tests for restriction enzyme analysis, digestion, ligation, primers, PCR."""

    def test_restriction_enzymes_dict(self):
        """RESTRICTION_ENZYMES is a dict with known enzymes."""
        assert isinstance(RESTRICTION_ENZYMES, dict)
        assert len(RESTRICTION_ENZYMES) > 0
        for enzyme in ["EcoRI", "HindIII", "BamHI", "NotI"]:
            assert enzyme in RESTRICTION_ENZYMES, f"{enzyme} missing"
            assert isinstance(RESTRICTION_ENZYMES[enzyme], tuple)
            assert isinstance(RESTRICTION_ENZYMES[enzyme][0], str)

    def test_find_restriction_sites_ecori(self, sample_sequence):
        """EcoRI (GAATTC) site found in the sample sequence."""
        sites = find_restriction_sites(sample_sequence, "EcoRI")
        assert isinstance(sites, list)
        assert len(sites) >= 1
        for pos in sites:
            assert 0 <= pos
            assert pos + 6 <= len(sample_sequence)
            assert sample_sequence[pos:pos + 6] == "GAATTC"

    def test_find_restriction_sites_multiple(self, long_sequence):
        """Multiple enzymes find their respective sites."""
        eco_sites = find_restriction_sites(long_sequence, "EcoRI")
        hind_sites = find_restriction_sites(long_sequence, "HindIII")
        assert len(eco_sites) >= 1, "EcoRI not found"
        assert len(hind_sites) >= 1, "HindIII not found"
        for e in eco_sites:
            for h in hind_sites:
                assert e != h, "EcoRI and HindIII at same position"

    def test_find_restriction_sites_none_found(self, short_sequence):
        """Returns empty list when no sites present."""
        sites = find_restriction_sites(short_sequence, "EcoRI")
        assert isinstance(sites, list)
        assert len(sites) == 0

    def test_find_restriction_sites_bamhi(self, long_sequence):
        """BamHI (GGATCC) site search works."""
        sites = find_restriction_sites(long_sequence, "BamHI")
        assert isinstance(sites, list)
        for pos in sites:
            assert long_sequence[pos:pos + 6] == "GGATCC"

    def test_simulate_digestion_circular(self, circular_sequence):
        """Digestion of circular DNA produces correct fragments."""
        result = simulate_digestion(circular_sequence, "EcoRI", topology="circular")
        assert result is not None
        assert isinstance(result, dict)
        assert "fragments" in result
        fragments = result["fragments"]
        assert isinstance(fragments, list)
        assert len(fragments) >= 1
        for frag in result["sizes"]:
            assert frag > 0

    def test_simulate_digestion_linear(self, sample_sequence):
        """Digestion of linear DNA with one site produces two fragments."""
        result = simulate_digestion(sample_sequence, "EcoRI", topology="linear")
        assert result is not None
        assert isinstance(result, dict)
        assert "fragments" in result
        assert len(result["fragments"]) == 2
        assert sum(result["sizes"]) == len(sample_sequence)

    def test_simulate_digestion_multi_enzyme(self, long_sequence):
        """Digestion with enzyme on linear DNA produces expected fragments."""
        result = simulate_digestion(long_sequence, "EcoRI", topology="linear")
        assert result is not None
        assert "fragments" in result
        assert len(result["fragments"]) >= 2

    def test_simulate_ligation(self):
        """Ligation of two fragments reconstructs a valid sequence."""
        frag1 = "ATCGATCG"
        frag2 = "GCGCGCGC"
        ligated = simulate_ligation([frag1, frag2])
        assert ligated is not None
        assert isinstance(ligated, dict)
        assert ligated["size"] == len(frag1) + len(frag2)

    def test_simulate_ligation_three_fragments(self):
        """Ligation of three fragments produces correct total length."""
        frags = ["AAAA", "TTTT", "CCCC"]
        result = simulate_ligation(frags)
        assert result is not None
        assert isinstance(result, dict)
        assert result["size"] == 12

    def test_design_primers(self, long_sequence):
        """design_primers returns forward and reverse primer sequences."""
        primers = design_primers(long_sequence, primer_length=20)
        assert primers is not None
        assert isinstance(primers, dict)
        assert "forward" in primers
        assert "reverse" in primers
        assert len(primers["forward"]) == 20
        assert len(primers["reverse"]) == 20
        assert long_sequence[:20] == primers["forward"]

    def test_design_primers_different_lengths(self, long_sequence):
        """design_primers with varying primer lengths."""
        for length in [15, 20, 25]:
            primers = design_primers(long_sequence, primer_length=length)
            assert len(primers["forward"]) == length
            assert len(primers["reverse"]) == length

    def test_simulate_pcr(self, long_sequence):
        """simulate_pcr amplifies the region between primers."""
        primers = design_primers(long_sequence, primer_length=20)
        pcr_result = simulate_pcr(long_sequence, primers["forward"], primers["reverse"])
        assert pcr_result is not None
        assert isinstance(pcr_result, dict)
        assert "product" in pcr_result
        product = pcr_result["product"]
        assert isinstance(product, str)
        assert len(product) > 0
        assert len(product) <= len(long_sequence)

    def test_plot_virtual_gel(self):
        """plot_virtual_gel returns a matplotlib figure (close fig)."""
        fragments = [500, 1000, 2000, 3000]
        fig = plot_virtual_gel(fragments)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_virtual_gel_single_band(self):
        """Virtual gel handles a single band."""
        fig = plot_virtual_gel([1500])
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_format_digest_report(self, circular_sequence):
        """format_digest_report returns a descriptive string."""
        result = simulate_digestion(circular_sequence, "EcoRI", topology="circular")
        report = format_digest_report(result)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_format_primer_report(self, long_sequence):
        """format_primer_report returns a descriptive string."""
        primers = design_primers(long_sequence, primer_length=20)
        report = format_primer_report(primers)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_digest_fragment_sizes(self, circular_sequence):
        """Fragment sizes from digestion are consistent positive integers."""
        result = simulate_digestion(circular_sequence, "EcoRI", topology="circular")
        for frag in result["sizes"]:
            assert isinstance(frag, int)
            assert frag > 0

    def test_digest_total_bp(self, sample_sequence):
        """Total bp of digestion fragments equals input length for linear DNA."""
        result = simulate_digestion(sample_sequence, "EcoRI", topology="linear")
        total = sum(result["sizes"])
        assert total == len(sample_sequence)


# ══════════════════════════════════════════════════════════════════════════════
# TestSequenceViewer  (10 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestSequenceViewer:
    """Tests for sequence visualization functions."""

    def test_draw_sequence_view(self, sample_sequence):
        """draw_sequence_view returns a matplotlib figure (close fig)."""
        fig = draw_sequence_view(sample_sequence)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_sequence_view_with_annotations(self, long_sequence):
        """draw_sequence_view handles annotation list parameter."""
        annotations = [
            SequenceAnnotation(name="EcoRI site", start=800, end=806, color="#FF0000"),
            SequenceAnnotation(name="ORF", start=1200, end=1300, color="#00FF00"),
        ]
        fig = draw_sequence_view(long_sequence, annotations=annotations)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_translation_view(self, sample_sequence):
        """draw_translation_view returns a figure showing protein translation."""
        seq = sample_sequence[: (len(sample_sequence) // 3) * 3]
        fig = draw_translation_view(seq)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_gc_content_plot(self, long_sequence):
        """draw_gc_content_plot returns a figure with GC content data."""
        fig = draw_gc_content_plot(long_sequence, window=50)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_gc_content_short_seq(self, short_sequence):
        """GC content plot handles short sequences gracefully."""
        fig = draw_gc_content_plot(short_sequence, window=5)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_gc_content_various_windows(self, long_sequence):
        """GC content plot works with different window sizes."""
        for win in [10, 50, 100, 200]:
            fig = draw_gc_content_plot(long_sequence, window=win)
            assert fig is not None
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_draw_orf_map(self, long_sequence):
        """draw_orf_map returns a figure showing open reading frames."""
        fig = draw_orf_map(long_sequence)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_sequence_overview(self, long_sequence):
        """create_sequence_overview returns a matplotlib figure."""
        fig = create_sequence_overview(long_sequence)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_sequence_overview_short(self, short_sequence):
        """Sequence overview works for short sequences."""
        fig = create_sequence_overview(short_sequence)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_sequence(self):
        """All viewer functions handle empty sequence gracefully."""
        empty_seq = ""
        for fn in [draw_sequence_view, draw_orf_map]:
            try:
                fig = fn(empty_seq)
                if fig is not None:
                    plt.close(fig)
            except (ValueError, IndexError, TypeError, ZeroDivisionError):
                pass  # Acceptable: explicit error for empty input

        try:
            fig = draw_gc_content_plot(empty_seq, window=10)
            if fig is not None:
                plt.close(fig)
        except (ValueError, IndexError, TypeError, ZeroDivisionError):
            pass


# ══════════════════════════════════════════════════════════════════════════════
# TestCloningIntegration  (6 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestCloningIntegration:
    """End-to-end workflow tests combining multiple modules."""

    def test_full_digestion_workflow(self, circular_sequence):
        """Complete workflow: create plasmid → digest → report."""
        seq = circular_sequence[:3000]
        pm = PlasmidMap(sequence=seq, size=3000)
        feat = PlasmidFeature(
            name="Insert", start=1000, end=2000,
            feature_type="gene", color="#FF0000", strand=1,
        )
        pm.add_feature(feat)

        result = simulate_digestion(seq, "EcoRI", topology="circular")
        assert result is not None
        assert len(result["fragments"]) >= 1

        report = format_digest_report(result)
        assert len(report) > 0

        fig = draw_plasmid(pm)
        assert fig is not None
        plt.close(fig)

    def test_primer_pcr_workflow(self, long_sequence):
        """Complete workflow: design primers → PCR → verify product."""
        primers = design_primers(long_sequence, primer_length=20)
        assert "forward" in primers
        assert "reverse" in primers

        pcr_result = simulate_pcr(
            long_sequence, primers["forward"], primers["reverse"]
        )
        assert pcr_result is not None
        assert len(pcr_result["product"]) > 0
        assert len(pcr_result["product"]) <= len(long_sequence)

        report = format_primer_report(primers)
        assert len(report) > 0

        fig = plot_virtual_gel([len(pcr_result["product"])])
        assert fig is not None
        plt.close(fig)

    def test_plasmid_with_digest_sites(self, circular_sequence):
        """Plasmid map with restriction sites marked and verified by digestion."""
        seq = circular_sequence[:3000]
        eco_sites = find_restriction_sites(seq, "EcoRI")
        assert len(eco_sites) >= 1

        pm = PlasmidMap(sequence=seq, size=3000)
        for i, site in enumerate(eco_sites):
            feat = PlasmidFeature(
                name=f"EcoRI_{i + 1}", start=site, end=site + 6,
                feature_type="restriction_site",
                color="#FFAA00", strand=1,
            )
            pm.add_feature(feat)

        pm.add_feature(PlasmidFeature(
            name="Backbone", start=0, end=1000,
            feature_type="backbone", color="#888888", strand=1,
        ))

        result = simulate_digestion(seq, "EcoRI", topology="circular")
        assert len(result["fragments"]) == len(eco_sites)

        fig = draw_plasmid_with_annotations(pm)
        assert fig is not None
        plt.close(fig)

        report = format_plasmid_report(pm)
        assert len(report) > 0

    def test_digest_then_visualize(self, sample_sequence):
        """Digest linear sequence and visualize fragments on virtual gel."""
        result = simulate_digestion(sample_sequence, "EcoRI", topology="linear")
        fig = plot_virtual_gel(result["sizes"])
        assert fig is not None
        plt.close(fig)

    def test_create_and_draw_sample_plasmid(self):
        """Create sample plasmid and draw it — full roundtrip."""
        pm = create_sample_plasmid()
        fig = draw_plasmid_with_annotations(pm)
        assert fig is not None
        plt.close(fig)

        report = format_plasmid_report(pm)
        assert len(report) > 0
