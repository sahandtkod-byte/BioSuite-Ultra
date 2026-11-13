"""
Interactive Sequence Viewer — annotated sequence visualization.

Implemented with matplotlib only.

Provides:
  • draw_sequence_view       — colour-coded nucleotides + 6-frame translation
  • draw_sequence_view_with_annotations — the same view from plain dicts
  • draw_translation_view    — DNA ↔ amino acid alignment coloured by property
  • draw_gc_content_plot     — sliding-window GC% with mean line
  • draw_orf_map             — 6-frame ORF landscape
  • create_sequence_overview — 4-panel dashboard
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
import numpy as np

from ..core.sequence import reverse_complement, validate_nucleotide_sequence
from ..core.utils import GENETIC_CODE

# ── Genetic code ──────────────────────────────────────────────────────────────
# Imported from core.utils, not repeated here.  The two tables held the same 64
# codons and nothing checked that they stayed that way; a corrected copy in one
# module would have left every figure in the other one reading the old code.

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
    # An ambiguous codon (NNN and friends) is not an amino acid and must not
    # borrow the colour of a real one.
    "X": "unknown",
    # Special
    "*": "special",
}

AA_COLORS = {
    "hydrophobic": "#2196F3",
    "polar":       "#4CAF50",
    "charged+":    "#F44336",
    "charged-":    "#FF9800",
    "special":     "#9C27B0",
    "unknown":     "#9E9E9E",
}

# ── How much of a sequence one figure can hold ───────────────────────────────
# Measured on this machine: a view costs about 3.2 text artists and 1.1 ms of
# build time per base, so a whole 50 kb FASTA record — which is what the CLI
# hands it — builds 160 000 artists.  Long before that the picture stops being
# a picture: the adaptive font floors at 3 pt around 270 bases, and past roughly
# a thousand the letters are one grey band.  A wider window is cut to these
# budgets and the figure says what was left out.
_MAX_VIEW_BASES = 1000
_MAX_VIEW_CODONS = 500

#: Shortest ORF the automatic scan reports.  It changes the count a reader sees
#: on the dashboard, so the dashboard quotes it instead of hiding it.
_MIN_ORF_AA = 10


# ── Data classes ──────────────────────────────────────────────────────────────

# Public surface. ``biosuite/plotting/__init__.py`` star-imports this
# module, and without this list it re-exports the module's own imports
# (numpy, pandas, os, plotly, ...) as package-level names too.
__all__ = [
    'GENETIC_CODE',
    'NT_COLORS',
    'AA_PROPERTIES',
    'AA_COLORS',
    'SequenceAnnotation',
    'draw_sequence_view',
    'draw_translation_view',
    'draw_gc_content_plot',
    'draw_orf_map',
    'create_sequence_overview',
    'draw_sequence_view_with_annotations',
]

@dataclass
class SequenceAnnotation:
    """A feature annotation to display above a sequence view."""
    name: str
    start: int
    end: int
    strand: str = "+"       # "+" or "-"
    color: str = "#7B1FA2"
    feature_type: str = "feature"

    def __post_init__(self):
        """Turn the coordinates into base positions and refuse an impossible one.

        A backwards span used to reach the drawing code and come back as a box
        with negative width, a feature missing its end came back as nothing at
        all, and a coordinate that arrived as text from a parsed feature table
        crashed inside the drawing loop.
        """
        label = f"annotation {self.name!r}" if self.name else "unnamed annotation"
        self.start = _coordinate(self.start, f"{label} start")
        self.end = _coordinate(self.end, f"{label} end")
        if self.end <= self.start:
            raise ValueError(
                f"{label} spans {self.start}..{self.end}: end must be after start")
        if self.strand not in ("+", "-"):
            raise ValueError(
                f"{label} has strand {self.strand!r}: use '+' or '-'")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _as_dna(sequence: str, label: str = "sequence") -> str:
    """Validate *sequence* and fold it into the uppercase DNA the views read.

    The check itself belongs to ``core.sequence``; these views used to do
    ``sequence.upper().replace("U", "T")`` and draw whatever came back, so a
    pasted FASTA header became a plausible-looking row of coloured letters.
    """
    return validate_nucleotide_sequence(
        sequence, name=label, allow_empty=False).upper().replace("U", "T")


def _whole_number(value, label: str) -> int:
    """Return *value* as an int, or name the thing that is not a whole number."""
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(
            f"{label} must be a whole number, got {value!r}") from None


def _coordinate(value, label: str) -> int:
    """Return *value* as a base position, accepting what a parser hands over.

    A feature table reads coordinates as text, so ``"123"`` is a position;
    ``"123.5"`` and ``"abc"`` are not, and are named rather than truncated.
    """
    try:
        return operator.index(value)
    except TypeError:
        pass
    try:
        return int(str(value).strip())
    except ValueError:
        raise ValueError(f"{label} must be a base position, got {value!r}") from None


def _view_window(sequence: str, start: int, end: Optional[int],
                 label: str) -> Tuple[str, int, int]:
    """Return ``(bases, offset, total)`` for the window *start*:*end* asks for.

    Every other coordinate the figure draws is relative to the window, so the
    absolute offset has to travel with it or the ruler lies about which bases
    these are.
    """
    seq = _as_dna(sequence, label)
    total = len(seq)
    start = _coordinate(start, "start")
    if start < 0:
        raise ValueError(
            f"start must be 0 or more, got {start} (a negative start selects "
            f"the end of the sequence and labels it with negative positions)")
    if start >= total:
        raise ValueError(f"start ({start}) is past the end of "
                         f"a {total} bp sequence")
    if end is not None:
        end = _coordinate(end, "end")
        if end <= start:
            raise ValueError(f"end ({end}) must be after start ({start})")
    else:
        end = total
    return seq[start:min(end, total)], start, total


def _check_orf_records(orfs: list) -> list:
    """Name the ORF record that is missing a coordinate the map draws from.

    A hand-built or converted ORF list is a boundary: without this the figure
    died on ``KeyError: 'length'`` with no clue which of forty records it was.
    """
    for i, orf in enumerate(orfs):
        if not isinstance(orf, dict):
            raise ValueError(f"ORF #{i + 1} is a {type(orf).__name__}, "
                             f"not a dict of coordinates")
        missing = [key for key in ("frame", "start", "end", "length")
                   if key not in orf]
        if missing:
            raise ValueError(f"ORF #{i + 1} has no {', '.join(missing)}: {orf!r}")
    return orfs


def _dna(seq: str) -> str:
    """Fold *seq* into uppercase DNA (RNA is read with T in place of U).

    The length is kept: reverse ORFs are reported in the caller's coordinates,
    so a helper that dropped characters here would shift every one of them.
    """
    return seq.upper().replace("U", "T")


def _revcomp(seq: str) -> str:
    """Return the reverse complement of *seq*.

    This is the package's canonical implementation.  The local table it
    replaced knew only ACGT: ``R, Y, S, W, K, M, B, D, H, V`` came back
    uncomplemented and ``U`` came back as ``U``, so a sequence read from a
    real FASTA got -1/-2/-3 frames translated from something that was only
    half reversed.
    """
    return reverse_complement(seq.upper())


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


def _find_orfs_forward(seq: str, min_aa: int = _MIN_ORF_AA) -> list:
    """Scan all 3 forward reading frames for ORFs (ATG … stop).

    ``end`` is the position of the stop codon, so the span it describes holds
    the coding sequence without the stop, and ``terminated`` says whether a stop
    codon was found at all — an ORF that simply ran into the end of the sequence
    is not the same claim.
    """
    seq_upper = _dna(seq)
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
                stopped = GENETIC_CODE.get(seq_upper[j:j + 3], "X") == "*"
                protein_len = (j - start) // 3
                if protein_len >= min_aa:
                    orfs.append({
                        "frame": frame + 1,
                        "start": start,
                        "end": j,
                        "length": protein_len,
                        "strand": "+",
                        "terminated": stopped,
                    })
                i = j + 3 if stopped else len(seq_upper)
            else:
                i += 3
    return orfs


def _find_orfs_reverse(seq: str, min_aa: int = _MIN_ORF_AA) -> list:
    """Scan all 3 reverse reading frames for ORFs on the reverse-complement."""
    seq = _dna(seq)
    rc = _revcomp(seq)
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
                stopped = GENETIC_CODE.get(rc[j:j + 3], "X") == "*"
                protein_len = (j - start) // 3
                if protein_len >= min_aa:
                    # Map back to original coordinate space EXACTLY:
                    # rc span [start, j) == original span [L-j, L-start).
                    # (old code subtracted `frame` too — silently off by
                    # up to 2 bp on every reverse ORF.)
                    orig_end = seq_len - start
                    orig_start = seq_len - j
                    orfs.append({
                        "frame": -(frame + 1),
                        "start": max(0, orig_start),
                        "end": min(seq_len, orig_end),
                        "length": protein_len,
                        "strand": "-",
                        "terminated": stopped,
                    })
                i = j + 3 if stopped else len(rc)
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
    """Colour-coded nucleotide view with 6-frame translation.

    Parameters
    ----------
    sequence : str
        DNA (or RNA) sequence; IUPAC ambiguity codes are drawn in grey.
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

    Raises
    ------
    ValueError
        If the sequence is empty or not nucleotides, or if *start*/*end* do not
        describe a window inside it.  A view wider than ``_MAX_VIEW_BASES`` is
        not an error: it is drawn as far as it stays legible and the nucleotide
        track says which bases that is.
    """
    seq, offset, total = _view_window(sequence, start, end, "sequence")
    window_len = len(seq)
    shown = min(window_len, _MAX_VIEW_BASES)
    seq = seq[:shown]
    seq_len = shown

    # --- figure / axes layout -------------------------------------------------
    # We stack: ruler, annotation track, nucleotide track, 6 translation tracks
    n_frames = 6
    track_heights = [0.3, 1.0]                     # ruler, nts
    if annotations:
        track_heights.append(0.6)
    track_heights += [0.4] * n_frames              # 6 translation rows

    if ax is not None:
        # Reuse the caller's gridspec slot — the old code DELETED EVERY
        # axes in the shared figure, leaked a throwaway subplots figure,
        # then crashed on an illegal 4-positional-arg add_subplot call.
        fig = ax.figure
        sub = ax.get_subplotspec()
        ax.remove()
        gs_tracks = sub.subgridspec(len(track_heights), 1,
                                    height_ratios=track_heights)
        axes = [fig.add_subplot(gs_tracks[i])
                for i in range(len(track_heights))]
    else:
        fig, axes = plt.subplots(
            len(track_heights), 1, figsize=figsize,
            gridspec_kw={"height_ratios": track_heights},
        )

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
        ax_ruler.text(tp, 0.5, f"{tp + offset}", ha="center", va="center",
                      fontsize=7, color="#555555")

    # ── Nucleotide track ──────────────────────────────────────────────────
    ax_nt = axes[idx]; idx += 1
    ax_nt.set_xlim(0, seq_len)
    ax_nt.set_ylim(0, 1)
    ax_nt.axis("off")
    title = "Sequence"
    if shown < window_len:
        title = (f"Sequence  (bases {offset:,}–{offset + shown:,} of {total:,}; "
                 f"pass start={offset + shown:,} for the rest)")
    ax_nt.set_title(title, fontsize=8, loc="left", pad=2)

    # Adaptive font size
    font_size = min(7, max(3, 800 / max(1, seq_len)))

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

        # One row per stack of features that do not touch each other.  The old
        # code counted how many already-placed features overlapped this one and
        # called that number the row, so two features that overlapped could be
        # handed the same row the lower one had vacated and were drawn through
        # each other.
        rows = []
        drawn = 0
        for ann in annotations:
            a_start = max(0, ann.start - offset)
            a_end = min(seq_len, ann.end - offset)
            if a_start >= seq_len or a_end <= 0:
                continue
            for row, placed in enumerate(rows):
                if all(a_end <= ps or a_start >= pe for ps, pe in placed):
                    break
            else:
                rows.append([])
                row = len(rows) - 1
            rows[row].append((a_start, a_end))
            drawn += 1
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

        if drawn != len(annotations):
            ax_ann.set_title(f"Annotations  ({drawn} of {len(annotations)} "
                             f"features reach this window)",
                             fontsize=8, loc="left", pad=2)
        else:
            ax_ann.set_title("Annotations", fontsize=8, loc="left", pad=2)

    # ── 6-Frame translations ─────────────────────────────────────────────
    frame_labels = ["+1", "+2", "+3", "-1", "-2", "-3"]
    rc_seq = _revcomp(seq)                 # three frames read the same string
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
                is_atg = (ci == 0 and seq[fi:fi+3].upper() == "ATG")
                is_stop = (aa == "*")
                if is_stop:
                    color, fw = "#F44336", "bold"
                elif is_atg or aa == "M":
                    color, fw = "#388E3C", "bold"
                elif aa == "X":
                    color, fw = AA_COLORS["unknown"], "normal"
                else:
                    color, fw = "#333333", "normal"
                ax_tr.text(pos + 1.5, 0.5, aa, ha="center", va="center",
                           fontsize=max(4, font_size - 1),
                           fontfamily="monospace", color=color, fontweight=fw)
        else:
            # Reverse frame — work on reverse complement
            rfi = fi - 3  # 0, 1, 2
            amino = _translate_frame(rc_seq, rfi)
            for ci, aa in enumerate(amino):
                # Map position back to forward coordinates
                pos = seq_len - (rfi + ci * 3 + 3)
                is_stop = (aa == "*")
                is_atg = (ci == 0 and rc_seq[rfi:rfi+3].upper() == "ATG")
                if is_stop:
                    color, fw = "#F44336", "bold"
                elif is_atg or aa == "M":
                    color, fw = "#388E3C", "bold"
                elif aa == "X":
                    color, fw = AA_COLORS["unknown"], "normal"
                else:
                    color, fw = "#666666", "normal"
                ax_tr.text(pos + 1.5, 0.5, aa, ha="center", va="center",
                           fontsize=max(4, font_size - 1),
                           fontfamily="monospace", color=color, fontweight=fw,
                           fontstyle="italic")

    if ax is None:
        # Only the code that made the figure gets to lay it out: called from a
        # dashboard slot, ``fig.tight_layout`` re-flowed every sibling panel.
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
        DNA (or RNA) string.
    frame : int
        1-based reading frame (1, 2, or 3).
    figsize, ax : standard matplotlib kwargs.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the string is empty or not nucleotides, or *frame* is not 1, 2 or 3.
    """
    seq = _as_dna(dna_sequence, "sequence")
    frame = _whole_number(frame, "frame")
    if frame not in (1, 2, 3):
        raise ValueError(
            f"frame must be 1, 2 or 3, got {frame}: any other value reads the "
            "sequence from a different offset and the plot does not say so")

    offset = frame - 1
    codons = [seq[i:i+3] for i in range(offset, len(seq) - 2, 3)]
    n_codons = len(codons)
    if n_codons == 0:
        fig, ax_f = _ensure_axes(ax, figsize)
        ax_f.text(0.5, 0.5, f"Sequence too short for translation: {len(seq)} "
                             f"bases leave no complete codon in frame {frame}",
                  ha="center", va="center", transform=ax_f.transAxes, fontsize=12)
        return fig

    shown = min(n_codons, _MAX_VIEW_CODONS)
    codons = codons[:shown]
    amino = [GENETIC_CODE.get(c, "X") for c in codons]
    leftover = len(seq) - (offset + 3 * n_codons)

    # Layout: 2 rows — DNA, AA
    if ax is not None:
        # Stay INSIDE the caller's slot (old code spanned the whole
        # figure, covering every other dashboard panel).
        fig = ax.figure
        sub = ax.get_subplotspec()
        ax.remove()
        gs = sub.subgridspec(2, 1, height_ratios=[1, 1])
        ax_dna = fig.add_subplot(gs[0])
        ax_aa = fig.add_subplot(gs[1])
    else:
        fig, (ax_dna, ax_aa) = plt.subplots(2, 1, figsize=figsize,
                                             gridspec_kw={"height_ratios": [1, 1]})

    ax_dna.set_xlim(0, shown)
    ax_dna.set_ylim(0, 1)
    ax_dna.axis("off")
    dna_title = f"DNA  (frame {frame})"
    if shown < n_codons:
        dna_title += f" — first {shown:,} of {n_codons:,} codons"
    if leftover:
        plural = "" if leftover == 1 else "s"
        dna_title += f" ({leftover} trailing base{plural} in no codon)"
    ax_dna.set_title(dna_title, fontsize=9, loc="left", pad=2)

    ax_aa.set_xlim(0, shown)
    ax_aa.set_ylim(0, 1)
    ax_aa.axis("off")
    ax_aa.set_title("Protein" if shown == n_codons else
                    f"Protein  (first {shown:,} of {n_codons:,} codons)",
                    fontsize=9, loc="left", pad=2)

    font_dna = min(6, max(3, 600 / max(1, shown)))
    font_aa = min(9, max(5, 800 / max(1, shown)))

    for ci, (codon, aa) in enumerate(zip(codons, amino)):
        # DNA — colour per nucleotide
        for ni, nt in enumerate(codon):
            x = ci + ni * 0.33 + 0.17
            ax_dna.text(x, 0.5, nt, ha="center", va="center",
                        fontsize=font_dna, fontfamily="monospace",
                        color=NT_COLORS.get(nt, "#999"))

        # Amino acid — colour per property
        prop = AA_PROPERTIES.get(aa, "unknown")
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
    tick_step = max(1, shown // 20)
    for ci in range(0, shown, tick_step):
        pos = offset + ci * 3 + 1
        ax_dna.axvline(ci, color="#cccccc", linewidth=0.3)
        ax_dna.text(ci, 0.9, str(pos), ha="left", va="top",
                    fontsize=5, color="#888")
        ax_aa.axvline(ci, color="#cccccc", linewidth=0.3)

    if ax is None:
        fig.tight_layout(h_pad=0.5)
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
        DNA (or RNA) sequence.
    window : int
        Window size in base pairs.
    window_size : int, optional
        Alias for *window* (for API compatibility).
    figsize, ax : standard matplotlib kwargs.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the sequence is empty or not nucleotides, or *window* is not a whole
        number of at least one base.
    """
    if window_size is not None:
        window = window_size
    window = _whole_number(window, "window")
    if window < 1:
        raise ValueError(f"window must cover at least one base, got {window}")
    seq = _as_dna(sequence, "sequence")
    seq_len = len(seq)
    requested = window
    if seq_len < window:
        window = seq_len

    arr = np.array([1.0 if c in ("G", "C") else 0.0 for c in seq])
    cumsum = np.concatenate(([0.0], np.cumsum(arr)))

    positions = np.arange(window - 1, seq_len)
    # Window [p-w+1, p]: cumsum[p+1] - cumsum[p-w+1].  The old index
    # (cumsum[p-w+2]) dropped the FIRST base of every window while still
    # dividing by w — a systematic GC underestimate (e.g. 2% -> 0%).
    gc_vals = (cumsum[positions + 1] - cumsum[positions - window + 1]) / window * 100
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
    title = f"GC Content (window={window:,})"
    if window < requested:
        title = (f"GC Content (window={window:,} — the sequence is {seq_len:,} "
                 f"bp, shorter than the {requested:,} bp window asked for)")
    ax_f.set_title(title, fontsize=10)
    ax_f.legend(fontsize=7, loc="upper right")
    ax_f.set_ylim(0, 100)
    ax_f.margins(x=0.01)
    if ax is None:
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
        DNA (or RNA) sequence.
    orfs : list of dict, optional
        Pre-computed ORFs (keys: frame, start, end, length, and optionally
        strand and terminated).  If *None*, auto-detected.
    figsize, ax : standard matplotlib kwargs.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the sequence is empty or not nucleotides, or an ORF record is missing
        a coordinate the map needs.
    """
    seq = _as_dna(sequence, "sequence")
    seq_len = len(seq)

    if orfs is None:
        orfs = _find_orfs_forward(seq) + _find_orfs_reverse(seq)
    else:
        # list(), not iteration: the length scan below walks this once and the
        # drawing walks it again, so a generator would draw an empty map.
        orfs = list(orfs)
        _check_orf_records(orfs)

    fig, ax_f = _ensure_axes(ax, figsize)

    # Organise by frame
    frames = [1, 2, 3, -1, -2, -3]
    frame_y = {f: i for i, f in enumerate(frames)}
    row_height = 0.8

    # Colour by normalised length.  The bars are coloured over the middle of
    # viridis (the darkest fifth is invisible under the white length label), so
    # the colour bar has to be built from that same squeezed ramp — a bar
    # coloured at viridis(0.2 + 0.75*f) does not match the colour the raw
    # colormap gives at f, and the legend used to promise the match.
    max_len = max((o["length"] for o in orfs), default=1)
    max_len = max(1, max_len)
    cmap = LinearSegmentedColormap.from_list(
        "orf_length",
        [plt.cm.viridis(0.2 + 0.75 * t) for t in np.linspace(0, 1, 256)])

    drawn = 0
    open_ended = 0
    for orf in orfs:
        f = orf["frame"]
        if f not in frame_y:
            continue
        y_center = frame_y[f]
        norm_len = orf["length"] / max_len
        color = cmap(norm_len)
        width = max(orf["end"] - orf["start"], 1)
        terminated = orf.get("terminated", True)
        if not terminated:
            open_ended += 1
        rect = mpatches.FancyBboxPatch(
            (orf["start"], y_center - row_height / 2),
            width, row_height,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="#333333",
            linewidth=0.4, alpha=0.85,
            linestyle="-" if terminated else "--",
        )
        ax_f.add_patch(rect)
        drawn += 1
        # Length label inside ORF bar
        if width > seq_len * 0.02:
            ax_f.text(orf["start"] + width / 2, y_center,
                      f"{orf['length']}aa" if terminated else f"{orf['length']}aa?",
                      ha="center", va="center",
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
    notes = []
    if drawn != len(orfs):
        notes.append(f"{len(orfs) - drawn} of {len(orfs)} records have no row "
                     f"in ±1..±3")
    if open_ended:
        notes.append(f"{open_ended} run to the end without a stop codon "
                     "(dashed edge, '?' label)")
    if not drawn:
        notes.append(f"no ORF of {_MIN_ORF_AA} aa or more in {seq_len:,} bp")
    title = "ORF Map (6-frame)"
    if notes:
        title += " — " + "; ".join(notes)
    ax_f.set_title(title, fontsize=11)
    ax_f.set_yticks([])

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_len))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_f, fraction=0.02, pad=0.02)
    cbar.set_label("ORF length (aa)", fontsize=7)

    if ax is None:
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
        DNA (or RNA) sequence.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the sequence is empty or not nucleotides.
    """
    seq = _as_dna(sequence, "sequence")
    seq_len = len(seq)

    # ── Basic stats ──────────────────────────────────────────────────────
    counts = {nt: seq.count(nt) for nt in "ACGT"}
    other = seq_len - sum(counts.values())
    gc = sum(1 for c in seq if c in "GC") / seq_len * 100
    orfs = _find_orfs_forward(seq) + _find_orfs_reverse(seq)
    unfinished = sum(1 for o in orfs if not o.get("terminated", True))

    lines = [f"Length:  {seq_len:,} bp"]
    for nt in "ATGC":
        n = counts[nt]
        lines.append(f"{nt}:  {n:,}  ({n / seq_len * 100:.1f}%)")
    if other:
        letters = " ".join(sorted(set(seq) - set("ACGT")))
        lines.append(f"other:  {other:,}  ({other / seq_len * 100:.1f}%)"
                     f"  [{letters}]")
    lines.append(f"GC: {gc:.1f}%")
    orf_line = f"ORFs: {len(orfs)} ({_MIN_ORF_AA} aa or more)"
    if unfinished:
        orf_line += f", {unfinished} not stop-terminated"
    lines.append(orf_line)
    stats_text = "\n".join(lines)

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
    # No tight_layout here: the gridspec was built with the spacing this figure
    # wants, and re-flowing it around the ORF map's colour bar only earned a
    # warning (measured: the six axes land in the same places either way).
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

    Each annotation dict needs ``name``, ``start`` and ``end``; ``color``,
    ``strand`` and ``feature_type`` are optional.

    Parameters
    ----------
    sequence : str
        DNA (or RNA) sequence.
    annotations : list of dict
        Each dict: {"name": str, "start": int, "end": int, "color": str,
        "strand": "+"/"-", "feature_type": str}.  Coordinates may arrive as
        text, the way a parsed feature table hands them over.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes, optional
        If provided, draw into *ax*.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        Naming the entry that is not a dict, has no coordinates, or spans
        backwards.  A feature the wrapper cannot make sense of used to be
        dropped from the track without a word.
    """
    seq_annots = []
    for i, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            raise ValueError(f"annotation #{i + 1} is a "
                             f"{type(ann).__name__}, not a dict with "
                             "name/start/end")
        seq_annots.append(SequenceAnnotation(
            name=str(ann.get("name", "")),
            start=ann.get("start"),
            end=ann.get("end"),
            strand=ann.get("strand", "+"),
            color=ann.get("color", "#FF0000"),
            feature_type=ann.get("feature_type", "feature"),
        ))
    return draw_sequence_view(sequence, annotations=seq_annots, figsize=figsize, ax=ax)


# ── Internal utility ──────────────────────────────────────────────────────────

def _ensure_axes(ax, figsize):
    """Return (fig, ax), creating a new figure if *ax* is None."""
    if ax is not None:
        return ax.figure, ax
    fig, ax_new = plt.subplots(1, 1, figsize=figsize)
    return fig, ax_new
