"""
Plasmid Map Viewer — circular plasmid visualization.

Implemented with matplotlib and numpy only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Default colour palette
# ---------------------------------------------------------------------------
DEFAULT_COLORS = {
    "CDS": "#2196F3",
    "gene": "#2196F3",
    "promoter": "#4CAF50",
    "terminator": "#F44336",
    "origin": "#FF9800",
    "antibiotic_resistance": "#9C27B0",
    "restriction_site": "#795548",
    "multiple_cloning_site": "#607D8B",
    "backbone": "#888888",
    "other": "#607D8B",
}


# Public surface. ``biosuite/plotting/__init__.py`` star-imports this
# module, and without this list it re-exports the module's own imports
# (numpy, pandas, os, plotly, ...) as package-level names too.
__all__ = [
    'DEFAULT_COLORS',
    'PlasmidFeature',
    'PlasmidMap',
    'draw_plasmid',
    'draw_plasmid_with_annotations',
    'create_sample_plasmid',
    'format_plasmid_report',
]

#: The two drawings differ only in radii, label text and a few font sizes; they
#: used to be two ninety-line copies that had already drifted (0.07 vs 0.09 band
#: widths, 0.28 vs 0.32 label radius).
_STYLES = {
    False: dict(band_outer=0.07, band_inner=0.07, label_r=0.28, label_size=8,
                label_position=False, alpha=0.75, backbone_lw=1.5, marker_size=5,
                marker_offset=0.08, centre_size=12, ylim=1.6, arc_step=0.005,
                arc_min=30),
    True: dict(band_outer=0.09, band_inner=0.05, label_r=0.32, label_size=7,
               label_position=True, alpha=0.8, backbone_lw=1.8, marker_size=4.5,
               marker_offset=0.09, centre_size=11, ylim=1.7, arc_step=0.004,
               arc_min=40),
}

#: Extra radii a colliding label may move out to, in axis units.
_LABEL_RINGS = (0.0, 0.10, 0.20)
#: Two labels closer than this (radians) would overlap at the base radius.
_MIN_LABEL_GAP = 0.12


def _random_seq(length: int, rng=None) -> str:
    """Generate a DNA filler sequence of *length* bases.

    *rng* is any object with ``choices`` — ``create_sample_plasmid`` passes a
    seeded ``random.Random`` so the sample is the same plasmid on every run.
    """
    generator = rng if rng is not None else random
    return "".join(generator.choices("ATCG", k=length))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class PlasmidFeature:
    """A single annotated feature on a plasmid.

    ``end`` may be smaller than ``start`` for a feature that crosses the origin;
    such a feature is drawn as one arc running past the top of the map.  A
    negative coordinate is refused: it cannot be placed on a circle.
    """
    name: str
    start: int
    end: int
    strand: int = 1          # 1 = forward, -1 = reverse
    feature_type: str = "other"
    color: Optional[str] = None
    label: Optional[str] = None

    def __post_init__(self):
        if self.start < 0 or self.end < 0:
            raise ValueError(
                f"feature {self.name!r} has a negative coordinate "
                f"({self.start}..{self.end}); positions on a plasmid start at 0")
        if self.color is None:
            self.color = DEFAULT_COLORS.get(self.feature_type, "#607D8B")
        if self.label is None:
            self.label = self.name

    def span(self, total: int = 0) -> int:
        """Length in bp, taking an origin-crossing feature past ``total``.

        ``end - start`` reported a feature that wraps the origin as a negative
        length (measured: a 600..300 feature on a 1000 bp plasmid printed -300).
        """
        if self.end >= self.start:
            return self.end - self.start
        return (total - self.start) + self.end if total else self.end - self.start


class PlasmidMap:
    """Container for a circular plasmid map."""

    def __init__(self, name: str = "Plasmid", size: int = 0,
                 description: str = "", sequence: Optional[str] = None):
        self.name = name
        self.description = description
        #: True when ``sequence`` is generated filler rather than a real record.
        self.sequence_is_synthetic = False
        if sequence is not None:
            self.sequence = sequence.upper()
            self.size = len(self.sequence)
        else:
            self.size = size
            self.sequence = _random_seq(size) if size > 0 else ""
            self.sequence_is_synthetic = size > 0
        self.features: List[PlasmidFeature] = []

    def add_feature(self, feature, **kwargs) -> PlasmidMap:
        """Add a PlasmidFeature (object or keyword args) to the map."""
        feat = feature if isinstance(feature, PlasmidFeature) else \
            PlasmidFeature(feature, **kwargs)
        if self.size > 0 and max(feat.start, feat.end) > self.size:
            raise ValueError(
                f"feature {feat.name!r} ends at {max(feat.start, feat.end)} on a "
                f"{self.size} bp plasmid; a feature that crosses the origin is "
                "written with its start after its end")
        self.features.append(feat)
        return self

    def set_size(self, bp: int) -> PlasmidMap:
        """Change the size in bp.  The sequence, if any, is left as it is."""
        self.size = bp
        return self

    def set_name(self, name: str) -> PlasmidMap:
        self.name = name
        return self

    def set_description(self, description: str) -> PlasmidMap:
        self.description = description
        return self


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _bp_to_angle(bp: int, total: float) -> float:
    """Convert bp position to angle in radians (clockwise from top)."""
    return (bp / total) * 2 * math.pi


def _polar_slot(plasmid_map: PlasmidMap, figsize, ax):
    """Return (fig, ax, owns_figure) for a drawable plasmid and a polar slot."""
    if plasmid_map.size <= 0:
        raise ValueError(
            f"plasmid {plasmid_map.name!r} has no size, so there is no circle to "
            "draw; give it one (size=... or a sequence) first")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
        return fig, ax, True
    if getattr(ax, "name", "") != "polar":
        raise ValueError(
            "a plasmid map needs a polar axes (fig.add_subplot(projection='polar')); "
            "drawing angles and radii on a Cartesian axes puts the map on a straight "
            "line from (0, 0) to (2*pi, 1)")
    return ax.get_figure(), ax, False


def _label_radii(entries, base_r):
    """A radius per label, lifting the ones that would collide by angle.

    Every label sat on one ring, and the sample plasmid's multiple cloning site
    (five features inside 52 bp, measured gaps down to 0.003 rad) drew its names
    on top of each other.
    """
    order = sorted(range(len(entries)), key=lambda i: entries[i])
    placed = []                      # (angle, ring) of the labels already set
    radii = [base_r] * len(entries)
    for i in order:
        angle = entries[i]
        ring = 0
        while ring < len(_LABEL_RINGS) and any(
                abs(angle - other) < _MIN_LABEL_GAP and other_ring == ring
                for other, other_ring in placed):
            ring += 1
        ring = min(ring, len(_LABEL_RINGS) - 1)
        placed.append((angle, ring))
        radii[i] = base_r + _LABEL_RINGS[ring]
    return radii


def _draw_map(plasmid_map: PlasmidMap, ax, annotated: bool):
    """Draw the circular map into a polar axes (the two public drawers' one body)."""
    style = _STYLES[annotated]
    total = plasmid_map.size
    r_backbone = 1.0

    ax.set_aspect("equal")
    ax.set_ylim(0, style['ylim'])
    ax.set_axis_off()

    # Backbone circle
    theta = np.linspace(0, 2 * math.pi, 500)
    ax.plot(theta, np.full_like(theta, r_backbone), color="black",
            linewidth=style['backbone_lw'], zorder=1)

    features = sorted(plasmid_map.features, key=lambda f: f.start)
    arcs = []
    for feat in features:
        start_angle = _bp_to_angle(feat.start, total)
        end_angle = _bp_to_angle(feat.end, total)
        if end_angle < start_angle:
            end_angle += 2 * math.pi
        arcs.append((start_angle, end_angle, (start_angle + end_angle) / 2.0))
    radii = _label_radii([arc[2] for arc in arcs], r_backbone + style['label_r'])

    for (start_angle, end_angle, mid_angle), feat, label_r in zip(arcs, features, radii):
        outer_r = r_backbone + style['band_outer']
        inner_r = r_backbone - style['band_inner']
        n_pts = max(int((end_angle - start_angle) / style['arc_step']),
                    style['arc_min'])
        angles_body = np.linspace(start_angle, end_angle, n_pts)

        ax.fill_between(angles_body, inner_r, outer_r, color=feat.color,
                        alpha=style['alpha'], zorder=3)

        # Direction arrow — annotated in POLAR DATA coords (theta, r);
        # the old Cartesian (r·cos, r·sin) landed at random polar spots.
        arrow_angle = end_angle if feat.strand == 1 else start_angle
        back_angle = arrow_angle - (0.08 if feat.strand == 1 else -0.08)
        ax.annotate("", xy=(arrow_angle, r_backbone),
                    xytext=(back_angle, r_backbone),
                    arrowprops=dict(arrowstyle="-|>", color=feat.color,
                                    lw=2, mutation_scale=16),
                    zorder=5)

        label_text = (f"{feat.label} ({feat.start}-{feat.end})"
                      if style['label_position'] else feat.label)
        ax.text(mid_angle, label_r, label_text, fontsize=style['label_size'],
                fontweight="bold", ha="center", va="center", color=feat.color,
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=feat.color if style['label_position'] else "none",
                          alpha=0.9 if style['label_position'] else 0.8,
                          linewidth=0.5))

    # Size markers every 500 bp
    for bp in range(0, total, 500):
        angle = _bp_to_angle(bp, total)
        ax.plot([angle, angle], [r_backbone - 0.02, r_backbone + 0.02],
                color="gray", linewidth=0.6, zorder=2)
        ax.text(angle, r_backbone - style['marker_offset'], f"{bp}",
                fontsize=style['marker_size'], ha="center", va="center",
                color="gray", zorder=2)

    # Centre label
    ax.text(0, 0, f"{plasmid_map.name}\n{total} bp", fontsize=style['centre_size'],
            fontweight="bold", ha="center", va="center", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # Legend, one entry per feature type
    legend_handles = []
    seen = set()
    for feat in sorted(plasmid_map.features, key=lambda f: f.feature_type):
        if feat.feature_type not in seen:
            seen.add(feat.feature_type)
            legend_handles.append(
                mpatches.Patch(color=feat.color,
                              label=feat.feature_type.replace("_", " ").title()))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.05),
                  ncol=min(4, len(legend_handles)), fontsize=7, framealpha=0.9)


def draw_plasmid(plasmid_map: PlasmidMap, figsize: Tuple[int, int] = (10, 10),
                 ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Draw a circular plasmid map.

    Returns matplotlib Figure.

    Raises:
        ValueError: if the map has no size, or *ax* is not a polar axes (the
            angles and radii would be drawn as x and y on a straight line).
    """
    fig, ax, owns = _polar_slot(plasmid_map, figsize, ax)
    _draw_map(plasmid_map, ax, annotated=False)
    if owns:
        fig.tight_layout()
    return fig


def draw_plasmid_with_annotations(plasmid_map: PlasmidMap,
                                  figsize: Tuple[int, int] = (10, 10),
                                  ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Draw a plasmid map with richer annotations (feature names, positions,
    colour-coded arcs, and a detailed sidebar legend).

    Raises the same two errors as :func:`draw_plasmid`.
    """
    fig, ax, owns = _polar_slot(plasmid_map, figsize, ax)
    _draw_map(plasmid_map, ax, annotated=True)
    if owns:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sample plasmid
# ---------------------------------------------------------------------------
def create_sample_plasmid() -> PlasmidMap:
    """
    Create a pUC19-like sample plasmid (2686 bp).

    Includes AmpR, ori, and a multiple cloning site with EcoRI, BamHI,
    HindIII, and PstI.

    The backbone is synthetic: a seeded filler with those four recognition sites
    written in at the offsets the features name, so the map does not advertise a
    site its own sequence lacks.  It is not the pUC19 reference sequence, and two
    runs build the same plasmid.
    """
    total_size = 2686
    bases = list(_random_seq(total_size, random.Random(42)))
    for offset, motif in ((396, "GAATTC"), (412, "GGATCC"),
                          (433, "AAGCTT"), (440, "CTGCAG")):
        bases[offset:offset + len(motif)] = list(motif)
    seq = "".join(bases)

    p = PlasmidMap(name="pUC19", size=total_size, sequence=seq,
                   description="Synthetic pUC19-like cloning vector (2686 bp)")
    p.sequence_is_synthetic = True

    p.add_feature(PlasmidFeature("AmpR", 1864, 2524, strand=1,
                                 feature_type="antibiotic_resistance"))
    p.add_feature(PlasmidFeature("ori", 397, 685, strand=1,
                                 feature_type="origin"))
    p.add_feature(PlasmidFeature("lac promoter", 110, 145, strand=1,
                                 feature_type="promoter"))
    p.add_feature(PlasmidFeature("rrnB T1", 70, 108, strand=1,
                                 feature_type="terminator"))
    p.add_feature(PlasmidFeature("EcoRI", 396, 402, strand=1,
                                 feature_type="restriction_site"))
    p.add_feature(PlasmidFeature("BamHI", 412, 418, strand=1,
                                 feature_type="restriction_site"))
    p.add_feature(PlasmidFeature("HindIII", 433, 439, strand=1,
                                 feature_type="restriction_site"))
    p.add_feature(PlasmidFeature("PstI", 440, 446, strand=1,
                                 feature_type="restriction_site"))
    p.add_feature(PlasmidFeature("MCS", 395, 447, strand=1,
                                 feature_type="multiple_cloning_site"))
    p.add_feature(PlasmidFeature("lacZα", 154, 395, strand=1,
                                 feature_type="gene"))

    return p


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def format_plasmid_report(plasmid_map: PlasmidMap) -> str:
    """Return a formatted text report for a plasmid map."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  PLASMID MAP REPORT: {plasmid_map.name}")
    lines.append("=" * 60)
    lines.append(f"  Size: {plasmid_map.size} bp")
    if plasmid_map.description:
        lines.append(f"  Description: {plasmid_map.description}")
    if plasmid_map.sequence_is_synthetic:
        lines.append("  Sequence: synthetic filler written for this map, not a "
                     "reference record")
    lines.append(f"  Features: {len(plasmid_map.features)}")
    lines.append("-" * 60)
    lines.append(f"  {'Name':<20} {'Start':>7} {'End':>7} {'Size':>7} {'Strand':>6} {'Type'}")
    lines.append("-" * 60)

    for feat in sorted(plasmid_map.features, key=lambda f: f.start):
        fsize = feat.span(plasmid_map.size)
        kind = feat.feature_type
        if feat.end < feat.start:
            kind += " (wraps)"
        lines.append(
            f"  {feat.name:<20} {feat.start:>7,} {feat.end:>7,} {fsize:>7,} "
            f"{feat.strand:>6} {kind}"
        )

    lines.append("-" * 60)
    type_counts = {}
    for feat in plasmid_map.features:
        type_counts[feat.feature_type] = type_counts.get(feat.feature_type, 0) + 1
    lines.append("  Feature summary:")
    for ftype, count in sorted(type_counts.items()):
        lines.append(f"    {ftype.replace('_', ' ').title()}: {count}")
    lines.append("=" * 60)
    return "\n".join(lines)
