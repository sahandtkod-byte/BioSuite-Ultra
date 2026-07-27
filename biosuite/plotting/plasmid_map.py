"""
Plasmid Map Viewer — SnapGene-killer circular plasmid visualization.

Pure matplotlib/numpy. No paid dependencies.
"""

from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import matplotlib
# matplotlib.use("Agg")  # Removed: caller manages backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def _random_seq(length: int) -> str:
    """Generate a random DNA sequence of given length."""
    return "".join(random.choices("ATCG", k=length))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class PlasmidFeature:
    """A single annotated feature on a plasmid."""
    name: str
    start: int
    end: int
    strand: int = 1          # 1 = forward, -1 = reverse
    feature_type: str = "other"
    color: Optional[str] = None
    label: Optional[str] = None

    def __post_init__(self):
        if self.color is None:
            self.color = DEFAULT_COLORS.get(self.feature_type, "#607D8B")
        if self.label is None:
            self.label = self.name


class PlasmidMap:
    """Container for a circular plasmid map."""

    def __init__(self, name: str = "Plasmid", size: int = 0,
                 description: str = "", sequence: Optional[str] = None):
        self.name = name
        self.description = description
        if sequence is not None:
            self.sequence = sequence.upper()
            self.size = len(self.sequence)
        else:
            self.size = size
            self.sequence = _random_seq(size) if size > 0 else ""
        self.features: List[PlasmidFeature] = []

    def add_feature(self, feature: PlasmidFeature, **kwargs) -> PlasmidMap:
        """Add a PlasmidFeature (object or keyword args) to the map."""
        if isinstance(feature, PlasmidFeature):
            self.features.append(feature)
        else:
            self.features.append(PlasmidFeature(feature, **kwargs))
        return self

    def set_size(self, bp: int) -> PlasmidMap:
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


def draw_plasmid(plasmid_map: PlasmidMap, figsize: Tuple[int, int] = (10, 10),
                 ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Draw a circular plasmid map.

    Returns matplotlib Figure.
    """
    total = plasmid_map.size if plasmid_map.size > 0 else 1000
    r_backbone = 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    else:
        fig = ax.get_figure()

    ax.set_aspect("equal")
    ax.set_ylim(0, 1.6)
    ax.set_axis_off()

    # Backbone circle
    theta = np.linspace(0, 2 * math.pi, 500)
    ax.plot(theta, np.full_like(theta, r_backbone), color="black",
            linewidth=1.5, zorder=1)

    # Features
    for feat in sorted(plasmid_map.features, key=lambda f: f.start):
        start_angle = _bp_to_angle(feat.start, total)
        end_angle = _bp_to_angle(feat.end, total)
        if end_angle < start_angle:
            end_angle += 2 * math.pi

        mid_angle = (start_angle + end_angle) / 2.0
        outer_r = r_backbone + 0.07
        inner_r = r_backbone - 0.07
        n_pts = max(int((end_angle - start_angle) / 0.005), 30)
        angles_body = np.linspace(start_angle, end_angle, n_pts)

        ax.fill_between(angles_body, inner_r, outer_r, color=feat.color,
                        alpha=0.75, zorder=3)

        # Direction arrow
        arrow_angle = end_angle if feat.strand == 1 else start_angle
        tip_x = r_backbone * math.cos(arrow_angle)
        tip_y = r_backbone * math.sin(arrow_angle)
        perp = arrow_angle + math.pi / 2
        offset = 0.08 if feat.strand == 1 else -0.08
        base_x = r_backbone * math.cos(arrow_angle - offset)
        base_y = r_backbone * math.sin(arrow_angle - offset)
        aw = 0.06
        for sign in (+1, -1):
            ax.annotate("", xy=(tip_x, tip_y),
                        xytext=(base_x + sign * aw * math.cos(perp),
                                base_y + sign * aw * math.sin(perp)),
                        arrowprops=dict(arrowstyle="->", color=feat.color, lw=2),
                        zorder=5)

        # Label outside
        label_r = r_backbone + 0.28
        ax.text(mid_angle, label_r, feat.label, fontsize=8, fontweight="bold",
                ha="center", va="center", color=feat.color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # Size markers every 500 bp
    for bp in range(0, total, 500):
        angle = _bp_to_angle(bp, total)
        ax.plot([angle, angle], [r_backbone - 0.02, r_backbone + 0.02],
                color="gray", linewidth=0.6, zorder=2)
        ax.text(angle, r_backbone - 0.08, f"{bp}", fontsize=5,
                ha="center", va="center", color="gray", zorder=2)

    # Centre label
    ax.text(0, 0, f"{plasmid_map.name}\n{total} bp", fontsize=12,
            fontweight="bold", ha="center", va="center", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # Legend
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

    fig.tight_layout()
    return fig


def draw_plasmid_with_annotations(plasmid_map: PlasmidMap,
                                  figsize: Tuple[int, int] = (10, 10),
                                  ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Draw a plasmid map with richer annotations (feature names, positions,
    colour-coded arcs, and a detailed sidebar legend).
    """
    total = plasmid_map.size if plasmid_map.size > 0 else 1000

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    else:
        fig = ax.get_figure()

    ax.set_aspect("equal")
    ax.set_ylim(0, 1.7)
    ax.set_axis_off()

    # Backbone
    theta = np.linspace(0, 2 * math.pi, 500)
    r_backbone = 1.0
    ax.plot(theta, np.full_like(theta, r_backbone), color="black",
            linewidth=1.8, zorder=1)

    # Features — two rings for visual depth
    for feat in sorted(plasmid_map.features, key=lambda f: f.start):
        start_angle = _bp_to_angle(feat.start, total)
        end_angle = _bp_to_angle(feat.end, total)
        if end_angle < start_angle:
            end_angle += 2 * math.pi
        mid_angle = (start_angle + end_angle) / 2.0

        outer_r = r_backbone + 0.09
        inner_r = r_backbone - 0.05
        n_pts = max(int((end_angle - start_angle) / 0.004), 40)
        angles_body = np.linspace(start_angle, end_angle, n_pts)

        ax.fill_between(angles_body, inner_r, outer_r, color=feat.color,
                        alpha=0.8, zorder=3)

        # Arrow
        arrow_angle = end_angle if feat.strand == 1 else start_angle
        tip_x = r_backbone * math.cos(arrow_angle)
        tip_y = r_backbone * math.sin(arrow_angle)
        perp = arrow_angle + math.pi / 2
        offset = 0.08 if feat.strand == 1 else -0.08
        base_x = r_backbone * math.cos(arrow_angle - offset)
        base_y = r_backbone * math.sin(arrow_angle - offset)
        aw = 0.06
        for sign in (+1, -1):
            ax.annotate("", xy=(tip_x, tip_y),
                        xytext=(base_x + sign * aw * math.cos(perp),
                                base_y + sign * aw * math.sin(perp)),
                        arrowprops=dict(arrowstyle="->", color=feat.color, lw=2),
                        zorder=5)

        # Label with position
        label_r = r_backbone + 0.32
        label_text = f"{feat.label} ({feat.start}-{feat.end})"
        ax.text(mid_angle, label_r, label_text, fontsize=7, fontweight="bold",
                ha="center", va="center", color=feat.color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=feat.color,
                          alpha=0.9, linewidth=0.5))

    # Size markers
    for bp in range(0, total, 500):
        angle = _bp_to_angle(bp, total)
        ax.plot([angle, angle], [r_backbone - 0.02, r_backbone + 0.02],
                color="gray", linewidth=0.6, zorder=2)
        ax.text(angle, r_backbone - 0.09, f"{bp}", fontsize=4.5,
                ha="center", va="center", color="gray", zorder=2)

    # Centre
    ax.text(0, 0, f"{plasmid_map.name}\n{total} bp", fontsize=11,
            fontweight="bold", ha="center", va="center", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # Legend
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
    """
    total_size = 2686
    seq = _random_seq(total_size)

    p = PlasmidMap(name="pUC19", size=total_size, sequence=seq,
                   description="Classic E. coli cloning vector (2686 bp)")

    p.add_feature(PlasmidFeature("AmpR", 1864, 2524, strand=1,
                                 feature_type="antibiotic_resistance"))
    p.add_feature(PlasmidFeature("ori", 397, 685, strand=1,
                                 feature_type="origin"))
    p.add_feature(PlasmidFeature("lac promoter", 110, 145, strand=1,
                                 feature_type="promoter"))
    p.add_feature(PlasmidFeature("rrnB T1", 70, 108, strand=1,
                                 feature_type="terminator"))
    p.add_feature(PlasmidFeature("EcoRI", 396, 401, strand=1,
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
    lines.append(f"  Features: {len(plasmid_map.features)}")
    lines.append("-" * 60)
    lines.append(f"  {'Name':<20} {'Start':>7} {'End':>7} {'Size':>7} {'Strand':>6} {'Type'}")
    lines.append("-" * 60)

    for feat in sorted(plasmid_map.features, key=lambda f: f.start):
        fsize = feat.end - feat.start
        lines.append(
            f"  {feat.name:<20} {feat.start:>7,} {feat.end:>7,} {fsize:>7,} "
            f"{feat.strand:>6} {feat.feature_type}"
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
