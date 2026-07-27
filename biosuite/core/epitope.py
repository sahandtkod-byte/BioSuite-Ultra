"""
Epitope prediction — MHC binding prediction for vaccine design.
B-cell and T-cell epitope prediction using amino acid properties.
Pure Python implementation.
"""
import numpy as np
from collections import Counter


# ─── Amino Acid Properties ──────────────────────────────────────────────────

# Hydrophobicity (Kyte-Doolittle)
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Surface accessibility (Emini)
SURFACE_ACCESSIBILITY = {
    'A': 1.252, 'R': 1.326, 'N': 1.534, 'D': 1.589, 'C': 1.145,
    'Q': 1.508, 'E': 1.534, 'G': 0.990, 'H': 1.534, 'I': 1.100,
    'L': 1.100, 'K': 1.448, 'M': 1.100, 'F': 1.100, 'P': 0.990,
    'S': 1.495, 'T': 1.495, 'W': 1.100, 'Y': 1.379, 'V': 1.100,
}

# Flexibility (Karplus-Schulz)
FLEXIBILITY = {
    'A': 0.984, 'R': 1.012, 'N': 1.047, 'D': 1.029, 'C': 0.905,
    'Q': 1.012, 'E': 1.012, 'G': 1.047, 'H': 0.950, 'I': 0.905,
    'L': 0.905, 'K': 1.012, 'M': 0.905, 'F': 0.905, 'P': 1.047,
    'S': 1.047, 'T': 1.047, 'W': 0.950, 'Y': 0.950, 'V': 0.905,
}

# MHC Class I binding anchors (common HLA-A*02:01)
MHC_I_ANCHORS = {
    'A': 0.5, 'V': 1.0, 'I': 1.0, 'L': 1.0, 'M': 0.8,
    'F': 0.7, 'W': 0.6, 'T': 0.3, 'P': 0.2, 'G': 0.1,
    'S': 0.2, 'C': 0.3, 'Y': 0.5, 'H': 0.1, 'Q': 0.2,
    'N': 0.1, 'D': 0.0, 'E': 0.0, 'K': 0.0, 'R': 0.0,
}

# B-cell propensity (Parker)
B_CELL_PROPENSITY = {
    'A': 0.69, 'R': 0.73, 'N': 0.89, 'D': 1.10, 'C': 1.15,
    'Q': 1.10, 'E': 1.33, 'G': 0.55, 'H': 0.95, 'I': 0.51,
    'L': 0.53, 'K': 1.24, 'M': 0.64, 'F': 0.40, 'P': 0.69,
    'S': 0.80, 'T': 0.72, 'W': 0.32, 'Y': 0.43, 'V': 0.57,
}

# HLA supertype binding
HLA_SUPERTYPES = {
    'A0201': {'P2': 'AVLIM', 'P9': 'VILM'},
    'A0101': {'P2': 'TSNQD', 'P9': 'VILMF'},
    'A0301': {'P2': 'KRNQST', 'P9': 'KRY'},
    'A2402': {'P2': 'FYW', 'P9': 'FLIVMW'},
    'B0702': {'P2': 'P', 'P9': 'LMFW'},
    'B2705': {'P2': 'RHKDEN', 'P9': 'LFVSM'},
    'B4001': {'P2': 'EKQR', 'P9': 'LIVMF'},
    'B5801': {'P2': 'SA', 'P9': 'LIVMF'},
}


class EpitopeResult:
    """Result of epitope prediction."""

    def __init__(self, peptide, start, end, score, epitope_type="T-cell",
                 prediction_method="matrix"):
        self.peptide = peptide
        self.start = start
        self.end = end
        self.score = score
        self.epitope_type = epitope_type
        self.prediction_method = prediction_method

    def to_dict(self):
        return {
            "peptide": self.peptide,
            "start": self.start,
            "end": self.end,
            "score": round(self.score, 3),
            "type": self.epitope_type,
            "method": self.prediction_method,
        }


def predict_t_cell_epitopes(sequence, mhc_type="A0201", window_sizes=(8, 9, 10, 11),
                              threshold=0.5, max_results=50):
    """Predict T-cell epitopes based on MHC binding affinity.

    Args:
        sequence: protein sequence.
        mhc_type: HLA type (e.g., 'A0201').
        window_sizes: peptide lengths to scan.
        threshold: minimum binding score.
        max_results: max epitopes to return.

    Returns:
        list of EpitopeResult sorted by score descending.
    """
    results = []
    anchors = HLA_SUPERTYPES.get(mhc_type, HLA_SUPERTYPES['A0201'])

    for w in window_sizes:
        for i in range(len(sequence) - w + 1):
            peptide = sequence[i:i + w]
            if len(peptide) != w:
                continue

            # MHC binding score
            p2 = peptide[1] if len(peptide) > 1 else peptide[0]
            p9 = peptide[-1]
            p2_score = MHC_I_ANCHORS.get(p2, 0.1)
            p9_score = MHC_I_ANCHORS.get(p9, 0.1)

            # Anchor match bonus
            anchor_match = 0
            if p2 in anchors.get('P2', ''):
                anchor_match += 0.3
            if p9 in anchors.get('P9', ''):
                anchor_match += 0.3

            # Hydrophobicity contribution
            hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in peptide) / len(peptide)
            hydro_score = max(0, min(1, (hydro + 4.5) / 9))

            # Combine scores
            score = 0.3 * p2_score + 0.3 * p9_score + 0.2 * hydro_score + 0.2 * anchor_match

            if score >= threshold:
                results.append(EpitopeResult(
                    peptide=peptide, start=i, end=i + w,
                    score=score, epitope_type="T-cell",
                    prediction_method=f"MHC-{mhc_type}-binding"
                ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:max_results]


def predict_b_cell_epitopes(sequence: str, window_size: int = 7, threshold: float = 1.0, max_results: int = 50) -> list:
    """Predict B-cell epitopes based on surface accessibility and propensity.

    Args:
        sequence: protein sequence.
        window_size: epitope window size.
        threshold: minimum propensity score.
        max_results: max epitopes to return.

    Returns:
        list of EpitopeResult.
    """
    results = []
    n = len(sequence)

    for i in range(n - window_size + 1):
        peptide = sequence[i:i + window_size]

        # Average surface accessibility
        sa = sum(SURFACE_ACCESSIBILITY.get(aa, 1.0) for aa in peptide) / window_size
        # Average flexibility
        flex = sum(FLEXIBILITY.get(aa, 1.0) for aa in peptide) / window_size
        # Average B-cell propensity
        bp = sum(B_CELL_PROPENSITY.get(aa, 0.7) for aa in peptide) / window_size

        # Hydrophilicity (inverse of hydrophobicity)
        hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in peptide) / window_size
        hydrophilic = max(0, -hydro / 4.5)

        score = 0.3 * sa + 0.2 * flex + 0.3 * bp + 0.2 * hydrophilic

        if score >= threshold:
            results.append(EpitopeResult(
                peptide=peptide, start=i, end=i + window_size,
                score=score, epitope_type="B-cell",
                prediction_method="surface-accessibility"
            ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:max_results]


def predict_linear_epitopes(sequence, window_sizes=(10, 12, 15), threshold=0.6, max_results=50):
    """Predict linear B-cell epitopes using multiple properties.

    Combines hydrophilicity, accessibility, flexibility, and propensity.
    """
    results = []
    for w in window_sizes:
        for i in range(len(sequence) - w + 1):
            peptide = sequence[i:i + w]
            scores = []
            for aa in peptide:
                hydro = HYDROPHOBICITY.get(aa, 0)
                sa = SURFACE_ACCESSIBILITY.get(aa, 1.0)
                bp = B_CELL_PROPENSITY.get(aa, 0.7)
                scores.append((-hydro / 4.5 + sa + bp) / 3)
            score = sum(scores) / len(scores)
            if score >= threshold:
                results.append(EpitopeResult(
                    peptide=peptide, start=i, end=i + w,
                    score=score, epitope_type="linear-B-cell",
                    prediction_method="multi-property"
                ))
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:max_results]


def cleavage_site_prediction(sequence):
    """Predict proteasomal cleavage sites (immunoproteasome).

    Based on C-terminal residue preferences.
    """
    good_cleavage = set('AGSVLMT')
    bad_cleavage = set('PDERKNQHFYW')
    results = []
    for i in range(len(sequence) - 1):
        aa = sequence[i]
        next_aa = sequence[i + 1] if i + 1 < len(sequence) else ''
        if aa in good_cleavage and next_aa not in bad_cleavage:
            results.append({"position": i, "residue": aa, "preference": "favorable"})
        elif aa in bad_cleavage:
            results.append({"position": i, "residue": aa, "preference": "unfavorable"})
    return results


def iedb_to_table(epitopes):
    """Format epitope results as a table."""
    lines = [f"{'Peptide':<20} {'Type':<12} {'Start':<6} {'End':<6} {'Score':<8} {'Method'}"]
    lines.append("-" * 80)
    for e in epitopes:
        lines.append(f"{e.peptide:<20} {e.epitope_type:<12} {e.start:<6} {e.end:<6} "
                     f"{e.score:<8.3f} {e.prediction_method}")
    lines.append(f"\nTotal epitopes: {len(epitopes)}")
    return "\n".join(lines)


def format_epitope_report(t_cell, b_cell, sequence_name="protein"):
    """Format combined T-cell and B-cell epitope results."""
    lines = [
        f"Epitope Prediction Report — {sequence_name}",
        "=" * 50,
        f"T-cell epitopes predicted: {len(t_cell)}",
        f"B-cell epitopes predicted: {len(b_cell)}",
        "",
        "Top 5 T-cell epitopes:",
    ]
    for e in t_cell[:5]:
        lines.append(f"  {e.peptide} (score={e.score:.3f}, pos={e.start}-{e.end})")
    lines.append("\nTop 5 B-cell epitopes:")
    for e in b_cell[:5]:
        lines.append(f"  {e.peptide} (score={e.score:.3f}, pos={e.start}-{e.end})")
    return "\n".join(lines)
