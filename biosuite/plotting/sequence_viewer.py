"""
Interactive Sequence Viewer — SnapGene-style sequence visualization.

All pure matplotlib; no paid dependencies.

Provides:
  • draw_sequence_view       — colour-coded nucleotides + 6-frame translation
  • draw_translation_view    — DNA ↔ amino acid alignment coloured by property
  • draw_gc_content_plot     — sliding-window GC% with mean line
  • draw_orf_map             — 6-frame ORF landscape
  • create_sequence_overview — 4-panel dashboard
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Sequence as Seq, Tuple

import matplotlib
matplotlib.use("Agg")                       # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Genetic code ──────────────────────────────────────────────────────────────
GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C",
              "a": "t", "t": "a", "c": "g", "g": "c"}

# ── Nucleotide colours ────────────────────────────────────────────────────────
NT_COLORS = {
    "A": "#4CAF50",   # green
    "T": "#F44336",   # red
    "G": "#FFC107",   # amber
    "C": "#2196F3",   # blue
}

# ── Amino-acid property colours ──────────────────────────────────────────────
AA_PROPERTIES = {
    # Hydrophobic
    "A": "hydrophobic", "V": "hydrophobic", "I": "hydrophobic",
    "L": "hydrophobic", "M": "hydrophobic", "F": "hydrophobic",
    "W": "hydrophobic", "P": "hydrophobic",
    # Polar uncharged
    "G": "polar", "S": "polar", "T": "polar", "C": "polar",
    "Y": "polar", "N": "polar", "Q": "polar",
    # Positively charged
    "K": "charged+", "R": "charged+", "H": "charged+",
    # Negatively charged
    "D": "charged-", "E": "charged-",
    # Special
    "*": "special",
}

AA_COLORS = {
    "hydrophobic": "#2196F3",
    "polar":       "#4CAF50",
    "charged+":    "#F44336",
    "charged-":    "#FF9800",
    "special":     "#9C27B0",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SequenceAnnotation:
    """A feature annotation to display above a sequence view."""
    name: str
    start: int
    end: int
    strand: str = "+"       # "+" or "-"
    color: str = "#7B1FA2"
    feature_type: str = "feature"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _revcomp(seq: str) -> str:
    """Return the reverse complement of *seq*."""
    return "".join(COMPLEMENT.get(c, c) for c in reversed(seq))


def _translate_frame(seq: str, frame: int) -> List[str]:
    """Translate *seq* in the given reading frame (0-indexed offset).

    Returns a list of single-letter amino-acid strings, one per codon.
    """
    s = seq[frame:]
    aa = []
    for i in range(0, len(s) - 2, 3):
        codon = s[i:i + 3].upper()
        aa.append(GENETIC_CODE.get(codon, "X"))
    return aa


def _find_orfs_forward(seq: str, min_aa: int = 10) -> list:
    """Scan all 3 forward reading frames for ORFs (ATG … stop)."""
    seq_upper = seq.upper().replace("U", "T")
    orfs = []
    for frame in range(3):
        i = frame
        while i <= len(seq_upper) - 3:
            codon = seq_upper[i:i + 3]
            if codon == "ATG":
                start = i
                j = i
                while j <= len(seq_upper) - 3:
                    c = seq_upper[j:j + 3]
                    if GENETIC_CODE.get(c, "X") == "*":
                        break
                    j += 3
                protein_len = (j - start) // 3
                if protein_len >= min_aa:
                    orfs.append({
                        "frame": frame + 1,
                        "start": start,
                        "end": j,
                        "length": protein_len,
                        "strand": "+",
                    })
                i = j + 3 if GENETIC_CODE.get(seq_upper[j:j + 3], "X") == "*" else len(seq_upper)
            else:
                i += 3
    return orfs


def _find_orfs_reverse(seq: str, min_aa: int = 10) -> list:
    """Scan all 3 reverse reading frames for ORFs on the reverse-complement."""
    rc = _revcomp(seq.upper())
    orfs = []
    seq_len = len(seq)
    for frame in range(3):
        i = frame
        while i <= len(rc) - 3:
            codon = rc[i:i + 3]
            if codon == "ATG":
                start = i
                j = i
                while j <= len(rc) - 3:
                    c = rc[j:j + 3]
                    if GENETIC_CODE.get(c, "X") == "*":
                        break
                    j += 3
                protein_len = (j - start) // 3
                if protein_len >= min_aa:
                    # Map back to original coordinate space
                    orig_end = seq_len - (frame + i)       # approximate mapping
                    orig_start = seq_len - (frame + j)
                    orfs.append({
                        "frame": -(frame + 1),
                        "start": max(0, orig_start),
                        "end": min(seq_len, orig_end),
                        "length": protein_len,
                        "strand": "-",
                    })
                i = j + 3 if GENETIC_CODE.get(rc[j:j + 3], "X") == "*" else len(rc)
            else:
                i += 3
    return orfs


# ══════════════════════════════════════════════════════════════════════════════
# 1. draw_sequence_view
# ══════════════════════════════════════════════════════════════════════════════

def draw_sequence_view(
    sequence: str,
    annotations: Optional[List[SequenceAnnotation]] = None,
    start: int = 0,
    end: Optional[int] = None,
    figsize: Tuple[int, int] = (16, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """SnapGene-style colour-coded nucleotide view with 6-frame translation.

    Parameters
    ----------
    sequence : str
        DNA sequence (bases A/C/G/T).
    annotations : list of SequenceAnnotation, optional
        Features to draw as coloured bars above the sequence.
    start, end : int
        Sub-range to display (0-based, end-exclusive). *end=None* → end of seq.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes, optional
        If provided, draw into *ax*; otherwise a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    seq = sequence.upper().replace("U", "T")
    if end is None:
        end = len(seq)
    seq = seq[start:end]
    seq_len = len(seq)

    # --- figure / axes layout -------------------------------------------------
    # We stack: ruler, annotation track, nucleotide track, 6 translation tracks
    n_frames = 6
    track_heights = [0.3, 1.0]                     # ruler, nts
    if annotations:
        track_heights.append(0.6)
    track_heights += [0.4] * n_frames              # 6 translation rows

    if ax is not None:
        fig = ax.figure
        ax.remove()
        gridspec_kw = {"height_ratios": track_heights}
        fig_tmp, axes = plt.subplots(
            len(track_heights), 1, figsize=figsize, gridspec_kw=gridspec_kw,
        )
        # Remove existing axes safely (compatible with matplotlib 3.8+)
        for ax in fig.axes[:]:
            fig.delaxes(ax)
        for i, a in enumerate(axes):
            fig.add_subplot(len(track_heights), 1, i + 1, axes[i] if hasattr(axes[i], 'figure') else axes[i])
    else:
        fig, axes = plt.subplots(
            len(track_heights), 1, figsize=figsize,
            gridspec_kw={"height_ratios": track_heights},
        )

    # Flatten in case of 1-element
    if not hasattr(axes, '__len__'):
        axes = [axes]
    axes = list(axes)

    idx = 0  # current axes index

    # ── Ruler ──────────────────────────────────────────────────────────────
    ax_ruler = axes[idx]; idx += 1
    ax_ruler.set_xlim(0, seq_len)
    ax_ruler.set_ylim(0, 1)
    ax_ruler.axis("off")
    # Tick marks every 10 bp
    tick_step = max(1, seq_len // max(1, seq_len // 10))
    tick_step = max(10, tick_step - tick_step % 10) if tick_step >= 10 else 10
    tick_positions = list(range(0, seq_len, tick_step))
    for tp in tick_positions:
        ax_ruler.axvline(tp, color="#888888", linewidth=0.5)
        ax_ruler.text(tp, 0.5, f"{tp + start}", ha="center", va="center",
                      fontsize=7, color="#555555")

    # ── Nucleotide track ──────────────────────────────────────────────────
    ax_nt = axes[idx]; idx += 1
    ax_nt.set_xlim(0, seq_len)
    ax_nt.set_ylim(0, 1)
    ax_nt.axis("off")
    ax_nt.set_title("Sequence", fontsize=8, loc="left", pad=2)

    # Adaptive font size
    font_size = min(7, max(3, 800 / max(1, seq_len)))
    char_width = 1  # one character = 1 bp in our coordinate system

    # Render each nucleotide
    for i, nt in enumerate(seq):
        color = NT_COLORS.get(nt, "#999999")
        ax_nt.text(i + 0.5, 0.5, nt, ha="center", va="center",
                   fontsize=font_size, fontfamily="monospace",
                   color=color, fontweight="bold")

    # ── Annotation bars ──────────────────────────────────────────────────
    if annotations:
        ax_ann = axes[idx]; idx += 1
        ax_ann.set_xlim(0, seq_len)
        ax_ann.set_ylim(0, 1)
        ax_ann.axis("off")
        ax_ann.set_title("Annotations", fontsize=8, loc="left", pad=2)

        # Stack non-overlapping annotations vertically
        occupied = []  # list of (start, end) for placed annotations
        for ann in annotations:
            a_start = max(0, ann.start - start)
            a_end = min(seq_len, ann.end - start)
            if a_start >= seq_len or a_end <= 0:
                continue
            # find lowest available row
            row = 0
            for os, oe in occupied:
                if a_start < oe and a_end > os:
                    row += 1
            occupied.append((a_start, a_end))
            y = 0.15 + 0.25 * row
            rect = mpatches.FancyBboxPatch(
                (a_start, y - 0.08), a_end - a_start, 0.16,
                boxstyle="round,pad=0.02", facecolor=ann.color, edgecolor="black",
                linewidth=0.5, alpha=0.85,
            )
            ax_ann.add_patch(rect)
            midpoint = (a_start + a_end) / 2
            ax_ann.text(midpoint, y, ann.name, ha="center", va="center",
                        fontsize=max(5, font_size - 1), color="white",
                        fontweight="bold",
                        clip_on=True)

    # ── 6-Frame translations ─────────────────────────────────────────────
    frame_labels = ["+1", "+2", "+3", "-1", "-2", "-3"]
    for fi in range(n_frames):
        ax_tr = axes[idx]; idx += 1
        ax_tr.set_xlim(0, seq_len)
        ax_tr.set_ylim(0, 1)
        ax_tr.axis("off")

        label = frame_labels[fi]
        ax_tr.set_title(f"Frame {label}", fontsize=7, loc="left", pad=1,
                        fontstyle="italic")

        if fi < 3:
            # Forward frame
            amino = _translate_frame(seq, fi)
            for ci, aa in enumerate(amino):
                pos = fi + ci * 3
                if pos >= seq_len:
                    break
                is_atg = (ci == 0 and seq[fi:fi+3].upper() == "ATG")
                is_stop = (aa == "*")
                if is_stop:
                    color, fw = "#F44336", "bold"
                elif is_atg or aa == "M":
                    color, fw = "#388E3C", "bold"
                else:
                    color, fw = "#333333", "normal"
                ax_tr.text(pos + 1.5, 0.5, aa, ha="center", va="center",
                           fontsize=max(4, font_size - 1),
                           fontfamily="monospace", color=color, fontweight=fw)
        else:
            # Reverse frame — work on reverse complement
            rfi = fi - 3  # 0, 1, 2
            rc_seq = _revcomp(seq)
            amino = _translate_frame(rc_seq, rfi)
            for ci, aa in enumerate(amino):
                # Map position back to forward coordinates
                pos = seq_len - (rfi + ci * 3 + 3)
                if pos < 0 or pos >= seq_len:
                    continue
                is_stop = (aa == "*")
                is_atg = (ci == 0 and rc_seq[rfi:rfi+3].upper() == "ATG")
                if is_stop:
                    color, fw = "#F44336", "bold"
                elif is_atg or aa == "M":
                    color, fw = "#388E3C", "bold"
                else:
                    color, fw = "#666666", "normal"
                ax_tr.text(pos + 1.5, 0.5, aa, ha="center", va="center",
                           fontsize=max(4, font_size - 1),
                           fontfamily="monospace", color=color, fontweight=fw,
                           fontstyle="italic")

    fig.tight_layout(h_pad=0.3)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. draw_translation_view
# ══════════════════════════════════════════════════════════════════════════════

def draw_translation_view(
    dna_sequence: str,
    frame: int = 1,
    figsize: Tuple[int, int] = (14, 4),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """DNA → amino-acid alignment coloured by amino-acid property.

    Parameters
    ----------
    dna_sequence : str
        DNA string.
    frame : int
        1-based reading frame (1, 2, or 3).
    figsize, ax : standard matplotlib kwargs.

    Returns
    -------
    matplotlib.figure.Figure
    """
    seq = dna_sequence.upper().replace("U", "T")
    offset = frame - 1
    codons = [seq[i:i+3] for i in range(offset, len(seq) - 2, 3)]
    amino = [GENETIC_CODE.get(c, "X") for c in codons]

    n_codons = len(codons)
    if n_codons == 0:
        fig, ax_f = _ensure_axes(ax, figsize)
        ax_f.text(0.5, 0.5, "Sequence too short for translation", ha="center",
                  va="center", transform=ax_f.transAxes, fontsize=12)
        return fig

    # Layout: 2 rows — DNA, AA
    if ax is not None:
        fig = ax.figure
        ax.remove()
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
        ax_dna = fig.add_subplot(gs[0])
        ax_aa = fig.add_subplot(gs[1])
    else:
        fig, (ax_dna, ax_aa) = plt.subplots(2, 1, figsize=figsize,
                                             gridspec_kw={"height_ratios": [1, 1]})

    ax_dna.set_xlim(0, n_codons)
    ax_dna.set_ylim(0, 1)
    ax_dna.axis("off")
    ax_dna.set_title(f"DNA  (frame {frame})", fontsize=9, loc="left", pad=2)

    ax_aa.set_xlim(0, n_codons)
    ax_aa.set_ylim(0, 1)
    ax_aa.axis("off")
    ax_aa.set_title("Protein", fontsize=9, loc="left", pad=2)

    font_dna = min(6, max(3, 600 / max(1, n_codons)))
    font_aa = min(9, max(5, 800 / max(1, n_codons)))

    for ci, (codon, aa) in enumerate(zip(codons, amino)):
        # DNA — colour per nucleotide
        for ni, nt in enumerate(codon):
            x = ci + ni * 0.33 + 0.17
            ax_dna.text(x, 0.5, nt, ha="center", va="center",
                        fontsize=font_dna, fontfamily="monospace",
                        color=NT_COLORS.get(nt, "#999"))

        # Amino acid — colour per property
        prop = AA_PROPERTIES.get(aa, "polar")
        bg = AA_COLORS.get(prop, "#EEEEEE")
        rect = mpatches.FancyBboxPatch(
            (ci + 0.05, 0.1), 0.9, 0.8,
            boxstyle="round,pad=0.05", facecolor=bg, edgecolor="white",
            linewidth=0.5, alpha=0.85,
        )
        ax_aa.add_patch(rect)

        # Highlight start/stop
        fw = "normal"
        txt_color = "white"
        if aa == "*":
            fw = "bold"
            txt_color = "#FFD54F"
        elif codon == "ATG":
            fw = "bold"

        ax_aa.text(ci + 0.5, 0.5, aa, ha="center", va="center",
                   fontsize=font_aa, fontfamily="monospace",
                   fontweight=fw, color=txt_color)

    # Codon-position ticks on DNA axis
    tick_step = max(1, n_codons // 20)
    for ci in range(0, n_codons, tick_step):
        pos = offset + ci * 3 + 1
        ax_dna.axvline(ci, color="#cccccc", linewidth=0.3)
        ax_dna.text(ci, 0.9, str(pos), ha="left", va="top",
                    fontsize=5, color="#888")
        ax_aa.axvline(ci, color="#cccccc", linewidth=0.3)

    try:
        fig.tight_layout(h_pad=0.5)
    except Exception:
        pass
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. draw_gc_content_plot
# ══════════════════════════════════════════════════════════════════════════════

def draw_gc_content_plot(
    sequence: str,
    window: int = 100,
    figsize: Tuple[int, int] = (12, 4),
    ax: Optional[plt.Axes] = None,
    window_size: Optional[int] = None,
) -> plt.Figure:
    """Sliding-window GC% plot with mean reference line.

    Parameters
    ----------
    sequence : str
        DNA sequence.
    window : int
        Window size in base pairs.
    window_size : int, optional
        Alias for *window* (for API compatibility).
    figsize, ax : standard matplotlib kwargs.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if window_size is not None:
        window = window_size
    seq = sequence.upper()
    seq_len = len(seq)
    if seq_len < window:
        window = max(1, seq_len)

    arr = np.array([1.0 if c in ("G", "C") else 0.0 for c in seq])
    cumsum = np.concatenate(([0.0], np.cumsum(arr)))

    positions = np.arange(window - 1, seq_len)
    gc_vals = (cumsum[positions + 1] - cumsum[positions - (window - 1) + 1]) / window * 100
    gc_avg = np.mean(gc_vals)

    fig, ax_f = _ensure_axes(ax, figsize)
    ax_f.fill_between(positions, gc_avg, gc_vals,
                      where=gc_vals >= gc_avg,
                      color="#81C784", alpha=0.5, interpolate=True,
                      label="Above avg")
    ax_f.fill_between(positions, gc_avg, gc_vals,
                      where=gc_vals < gc_avg,
                      color="#E57373", alpha=0.4, interpolate=True,
                      label="Below avg")
    ax_f.plot(positions, gc_vals, color="#1565C0", linewidth=0.8, label="GC%")
    ax_f.axhline(gc_avg, color="#D32F2F", linestyle="--", linewidth=1.2,
                 label=f"Avg {gc_avg:.1f}%")
    ax_f.set_xlabel("Position (bp)", fontsize=9)
    ax_f.set_ylabel("GC%", fontsize=9)
    ax_f.set_title(f"GC Content (window={window})", fontsize=10)
    ax_f.legend(fontsize=7, loc="upper right")
    ax_f.set_ylim(0, 100)
    ax_f.margins(x=0.01)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. draw_orf_map
# ══════════════════════════════════════════════════════════════════════════════

def draw_orf_map(
    sequence: str,
    orfs: Optional[list] = None,
    figsize: Tuple[int, int] = (14, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """6-frame ORF landscape (forward on top, reverse on bottom).

    Parameters
    ----------
    sequence : str
        DNA sequence.
    orfs : list of dict, optional
        Pre-computed ORFs (keys: frame, start, end, length, strand).
        If *None*, auto-detected.
    figsize, ax : standard matplotlib kwargs.

    Returns
    -------
    matplotlib.figure.Figure
    """
    seq = sequence.upper().replace("U", "T")
    seq_len = len(seq)

    if orfs is None:
        orfs = _find_orfs_forward(seq) + _find_orfs_reverse(seq)

    if not orfs:
        orfs = []

    fig, ax_f = _ensure_axes(ax, figsize)

    # Organise by frame
    frames = [1, 2, 3, -1, -2, -3]
    frame_y = {f: i for i, f in enumerate(frames)}
    row_height = 0.8

    # Colour by normalised length
    max_len = max((o["length"] for o in orfs), default=1)
    cmap = plt.cm.viridis

    for orf in orfs:
        f = orf["frame"]
        if f not in frame_y:
            continue
        y_center = frame_y[f]
        norm_len = orf["length"] / max_len if max_len else 0
        color = cmap(0.2 + 0.75 * norm_len)
        width = max(orf["end"] - orf["start"], 1)
        rect = mpatches.FancyBboxPatch(
            (orf["start"], y_center - row_height / 2),
            width, row_height,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="#333333",
            linewidth=0.4, alpha=0.85,
        )
        ax_f.add_patch(rect)
        # Length label inside ORF bar
        if width > seq_len * 0.02:
            ax_f.text(orf["start"] + width / 2, y_center,
                      f"{orf['length']}aa", ha="center", va="center",
                      fontsize=max(5, min(8, int(width / seq_len * 100))),
                      color="white", fontweight="bold")

    # Frame labels
    label_names = ["+1", "+2", "+3", "-1", "-2", "-3"]
    for i, lbl in enumerate(label_names):
        ax_f.text(-seq_len * 0.01, i, lbl, ha="right", va="center",
                  fontsize=9, fontweight="bold",
                  color="#1565C0" if "+" in lbl else "#C62828")

    # Separator between forward/reverse
    ax_f.axhline(2.5, color="#999999", linewidth=0.8, linestyle="--")

    ax_f.set_xlim(-seq_len * 0.04, seq_len)
    ax_f.set_ylim(-0.6, 5.6)
    ax_f.set_xlabel("Position (bp)", fontsize=9)
    ax_f.set_title("ORF Map (6-frame)", fontsize=11)
    ax_f.set_yticks([])

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_len))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_f, fraction=0.02, pad=0.02)
    cbar.set_label("ORF length (aa)", fontsize=7)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. create_sequence_overview
# ══════════════════════════════════════════════════════════════════════════════

def create_sequence_overview(sequence: str) -> plt.Figure:
    """4-panel dashboard: stats, GC%, ORF map, translation.

    Parameters
    ----------
    sequence : str
        DNA sequence.

    Returns
    -------
    matplotlib.figure.Figure
    """
    seq = sequence.upper().replace("U", "T")
    seq_len = len(seq)

    # ── Basic stats ──────────────────────────────────────────────────────
    gc = sum(1 for c in seq if c in "GC") / max(1, seq_len) * 100
    counts = {nt: seq.count(nt) for nt in "ACGT"}
    orfs = _find_orfs_forward(seq) + _find_orfs_reverse(seq)

    stats_text = (
        f"Length:  {seq_len:,} bp\n"
        f"A:  {counts.get('A', 0):,}  ({counts.get('A', 0)/max(1,seq_len)*100:.1f}%)\n"
        f"T:  {counts.get('T', 0):,}  ({counts.get('T', 0)/max(1,seq_len)*100:.1f}%)\n"
        f"G:  {counts.get('G', 0):,}  ({counts.get('G', 0)/max(1,seq_len)*100:.1f}%)\n"
        f"C:  {counts.get('C', 0):,}  ({counts.get('C', 0)/max(1,seq_len)*100:.1f}%)\n"
        f"GC: {gc:.1f}%\n"
        f"ORFs: {len(orfs)}"
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1 — Stats text
    ax_stats = fig.add_subplot(gs[0, 0])
    ax_stats.axis("off")
    ax_stats.text(0.1, 0.9, "Sequence Statistics", fontsize=13,
                  fontweight="bold", va="top", transform=ax_stats.transAxes)
    ax_stats.text(0.1, 0.75, stats_text, fontsize=10, va="top",
                  fontfamily="monospace", transform=ax_stats.transAxes,
                  linespacing=1.5)

    # Panel 2 — GC content
    ax_gc = fig.add_subplot(gs[0, 1])
    if seq_len >= 10:
        window = min(100, seq_len // 5)
        draw_gc_content_plot(seq, window=window, ax=ax_gc)
    else:
        ax_gc.text(0.5, 0.5, "Sequence too short", ha="center", va="center",
                   transform=ax_gc.transAxes)

    # Panel 3 — ORF map
    ax_orf = fig.add_subplot(gs[1, 0])
    draw_orf_map(seq, orfs=orfs, ax=ax_orf)

    # Panel 4 — Translation (frame 1)
    ax_tr = fig.add_subplot(gs[1, 1])
    if seq_len >= 3:
        draw_translation_view(seq, frame=1, ax=ax_tr)
    else:
        ax_tr.text(0.5, 0.5, "Sequence too short", ha="center", va="center",
                   transform=ax_tr.transAxes)

    fig.suptitle("Sequence Overview", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── Convenience wrapper ───────────────────────────────────────────────────────

def draw_sequence_view_with_annotations(
    sequence: str,
    annotations: List[dict],
    figsize: Tuple[int, int] = (16, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Draw a sequence view with annotations specified as plain dicts.

    Each annotation dict should have keys: name, start, end, color.

    Parameters
    ----------
    sequence : str
        DNA sequence.
    annotations : list of dict
        Each dict: {"name": str, "start": int, "end": int, "color": str}.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes, optional
        If provided, draw into *ax*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Convert dict annotations to SequenceAnnotation objects
    seq_annots = []
    for ann in annotations:
        seq_annots.append(SequenceAnnotation(
            name=ann.get("name", ""),
            start=ann.get("start", 0),
            end=ann.get("end", 0),
            color=ann.get("color", "#FF0000"),
        ))
    return draw_sequence_view(sequence, annotations=seq_annots, figsize=figsize, ax=ax)


# ── Internal utility ──────────────────────────────────────────────────────────

def _ensure_axes(ax, figsize):
    """Return (fig, ax), creating a new figure if *ax* is None."""
    if ax is not None:
        return ax.figure, ax
    fig, ax_new = plt.subplots(1, 1, figsize=figsize)
    return fig, ax_new
