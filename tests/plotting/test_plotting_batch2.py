"""Regression: plasmid polar arrows, genome_browser bins/CIGAR, conservation."""
import matplotlib
matplotlib.use('Agg')
import math
import os
import tempfile

import numpy as np
import pytest

from biosuite.plotting.plasmid_map import (
    PlasmidMap, PlasmidFeature, _bp_to_angle,
    create_sample_plasmid, draw_plasmid, draw_plasmid_with_annotations,
    format_plasmid_report,
)
from biosuite.plotting.genome_browser import (
    _compute_coverage_sam, _parse_cigar_length, parse_bed, parse_vcf,
    plot_genome_tracks, create_bed_track, create_variant_track,
)
from biosuite.plotting.conservation_plots import (
    compute_logo_heights, compute_conservation_scores, plot_sequence_logo,
)


# ── Plasmid ────────────────────────────────────────────────────────────────
def test_bp_to_angle_half_circle():
    assert _bp_to_angle(1343, 2686) == pytest.approx(math.pi)


def test_plasmid_renders_with_features():
    p = create_sample_plasmid()
    assert len(p.features) == 10
    f = draw_plasmid(p)
    assert f is not None
    f2 = draw_plasmid_with_annotations(p)
    assert f2 is not None


def test_plasmid_arrows_use_polar_data_coords():
    """Regression: old arrows passed Cartesian (r·cos, r·sin) as (theta, r)."""
    p = PlasmidMap(name='t', size=1000, sequence='A' * 1000)
    p.add_feature(PlasmidFeature('f1', 100, 400))
    fig = draw_plasmid(p)
    ax = fig.axes[0]
    annos = [a for a in ax.texts if getattr(a, 'arrow_patch', None) is not None]
    assert annos, 'no arrows drawn'
    for a in annos:
        tx, ty = a.get_position()  # text position = xytext in data coords
        # theta must be within [0, 2π+eps] and radius ~1.0, NOT cartesian
        assert 0 <= tx <= 2 * math.pi + 1e-6
        assert 0.9 <= ty <= 1.1


def test_plasmid_report():
    txt = format_plasmid_report(create_sample_plasmid())
    assert 'AmpR' in txt and '2686' in txt


# ── genome_browser ─────────────────────────────────────────────────────────
def _write(tmp, text):
    fd, p = tempfile.mkstemp(suffix='.sam')
    os.write(fd, text.encode()); os.close(fd)
    return p


def test_sam_coverage_midbin_read_counts_all_bins():
    sam = ('@SQ\tSN:chr1\tLN:10000\n'
           + 'r1\t0\tchr1\t96\t60\t100M\t*\t0\t0\t' + 'A' * 100 + '\t' + 'I' * 100 + '\n')
    # 1-based 96 -> 0-based 95..194: spans bin 0 and bin 1 at bin_size=100
    p = _write(None, sam)
    pos, cov = _compute_coverage_sam(p, None, 100)
    assert list(cov[:2]) == [1, 1]      # old code missed bin 1
    os.unlink(p)


def test_cigar_spliced_N_counts_reference():
    assert _parse_cigar_length('50M1000N50M') == 1100
    assert _parse_cigar_length('5S45M') == 45
    assert _parse_cigar_length('45M3D') == 48


def test_bed_vcf_parsers_roundtrip(tmp_path):
    bed = tmp_path / 'x.bed'
    bed.write_text('track name=t\nchr1\t10\t200\tgeneA\t60\nchr1\t500\t600\tgeneB\n')
    regs = parse_bed(str(bed))
    assert regs[0] == ('chr1', 10, 200, 'geneA', 60.0)
    assert regs[1][3] == 'geneB' and regs[1][4] == 0.0

    vcf = tmp_path / 'x.vcf'
    vcf.write_text('##fileformat=VCFv4.1\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\n'
                   'chr1\t100\t.\tA\tG\t99\tPASS\tDP=30\n')
    vs = parse_vcf(str(vcf))
    assert vs[0]['pos'] == 100 and vs[0]['filter'] == 'PASS' and vs[0]['qual'] == 99.0


def test_multi_track_render():
    pos = np.arange(0, 1000, 10)
    track = {'type': 'coverage', 'name': 'Cov',
             'data': {'positions': pos, 'coverage': np.sin(pos / 100) ** 2 * 30}}
    fig = plot_genome_tracks([track, {'type': 'variant', 'name': 'V',
                                      'data': {'variants': [{'pos': 500, 'ref': 'A', 'alt': 'G'}]}}])
    assert fig is not None


# ── conservation ───────────────────────────────────────────────────────────
def test_logo_conserved_column_schneider_correction():
    # 4 identical sequences, single base -> e_n = 3/(2·ln2·4) = 0.541
    _, _, tot = compute_logo_heights(['AAA', 'AAA', 'AAA', 'AAA'])
    assert tot[0] == pytest.approx(2 - 3 / (2 * math.log(2) * 4), abs=1e-9)


def test_logo_uniform_is_zero_bits():
    _, _, tot = compute_logo_heights(['AAA', 'CCC', 'GGG', 'TTT'])
    assert all(t == 0 for t in tot)


def test_conservation_scores_bounded():
    out = compute_conservation_scores(['ACGT', 'ACGA', 'ACGT'])
    assert all(0 <= c <= 1 for _, c in out)
    fig = plot_sequence_logo(['ACGT', 'ACGT'])
    assert fig is not None
