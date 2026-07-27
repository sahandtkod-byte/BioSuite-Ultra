"""
CRISPR guide RNA designer with dual-mode execution.

Pure Python PAM-finding and scoring as default, CRISPOR API as optional.
"""
import os
import re
import tempfile
import numpy as np
import warnings
from dataclasses import dataclass, field

try:
    import urllib.request
    import json
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

from .utils import PerformanceWarning, reverse_complement_dna


@dataclass
class GuideRNA:
    sequence: str
    pam: str
    position: int
    strand: str
    score: float
    gc_content: float
    off_target_count: int
    on_target_score: float


@dataclass
class CRISPRResult:
    engine: str
    guides: list = field(default_factory=list)
    target_sequence: str = ""
    num_guides: int = 0
    message: str = ""


def check_crispr_tools():
    return {'crispor_api': HAS_URLLIB}


# ── Pure Python Guide RNA Designer ──────────────────────────────────────────

PAM_PATTERNS = {
    'SpCas9': ('NGG', 'forward'),
    'SaCas9': ('NNGRRT', 'forward'),
    'Cas12a': ('TTTV', 'reverse'),
    'SpCas9-HF1': ('NGG', 'forward'),
}


def _find_pam_sites(sequence, pam_pattern='NGG', guide_length=20):
    sites = []
    seq_upper = sequence.upper()
    regex = pam_pattern.replace('N', '[ACGT]').replace('R', '[AG]').replace('Y', '[CT]').replace('V', '[ACG]')

    for match in re.finditer(f'(?=({regex}))', seq_upper):
        pam_start = match.start()
        pam_seq = match.group(1)

        if pam_start >= guide_length:
            guide = seq_upper[pam_start - guide_length:pam_start]
            sites.append({
                'guide': guide,
                'pam': pam_seq,
                'position': pam_start - guide_length,
                'strand': '+'
            })

    rc_seq = _reverse_complement(seq_upper)
    for match in re.finditer(f'(?=({regex}))', rc_seq):
        pam_start = match.start()
        pam_seq = match.group(1)
        if pam_start >= guide_length:
            guide_rc = rc_seq[pam_start - guide_length:pam_start]
            guide = _reverse_complement(guide_rc)
            orig_pos = len(seq_upper) - pam_start - len(pam_seq)
            sites.append({
                'guide': guide,
                'pam': pam_seq,
                'position': orig_pos,
                'strand': '-'
            })

    return sites


# Use shared reverse_complement_dna from utils
_reverse_complement = reverse_complement_dna


def _score_guide(guide_seq):
    gc = (guide_seq.count('G') + guide_seq.count('C')) / len(guide_seq) * 100
    gc_score = 1.0 - abs(gc - 50) / 50

    poly_n = max(len(m.group()) for m in re.finditer(r'(.)\1{3,}', guide_seq)) if re.search(r'(.)\1{3,}', guide_seq) else 0
    poly_penalty = -0.1 * poly_n

    gc_runs = len(re.findall(r'[GC]{5,}', guide_seq))
    gc_penalty = -0.05 * gc_runs

    score = gc_score + poly_penalty + gc_penalty
    return max(0, min(1, score))


def _builtin_design_guides(sequence, pam_type='SpCas9', guide_length=20, max_guides=20):
    pam_pattern, _ = PAM_PATTERNS.get(pam_type, ('NGG', 'forward'))
    sites = _find_pam_sites(sequence, pam_pattern, guide_length)

    guides = []
    for site in sites:
        guide_seq = site['guide']
        score = _score_guide(guide_seq)
        guides.append(GuideRNA(
            sequence=guide_seq,
            pam=site['pam'],
            position=site['position'],
            strand=site['strand'],
            score=round(score, 3),
            gc_content=round((guide_seq.count('G') + guide_seq.count('C')) / len(guide_seq) * 100, 1),
            off_target_count=0,
            on_target_score=round(score, 3)
        ))

    guides.sort(key=lambda g: g.score, reverse=True)
    return guides[:max_guides]


# ── CRISPOR API ─────────────────────────────────────────────────────────────

def _crispor_api_search(sequence):
    if not HAS_URLLIB:
        return None
    try:
        url = "http://crispor.tefor.net/api/"
        data = f"sequence={sequence}&genome=mm10".encode()
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
        return result
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


# ── Public API ──────────────────────────────────────────────────────────────

def design_guides(sequence, pam_type='SpCas9', guide_length=20, max_guides=20, tool='auto'):
    if not sequence:
        return CRISPRResult(engine='none', message="No sequence provided")

    tools = check_crispr_tools()
    if tool in ('crispor', 'auto') and tools.get('crispor_api'):
        api_result = _crispor_api_search(sequence)
        if api_result:
            return CRISPRResult(engine='crispor', guides=[], message="CRISPOR API results")

    warnings.warn(
        "CRISPOR API not available. Using built-in PAM finder. "
        "For off-target analysis, visit https://crispor.tefor.net/ manually.",
        PerformanceWarning, stacklevel=2
    )
    guides = _builtin_design_guides(sequence, pam_type, guide_length, max_guides)
    return CRISPRResult(
        engine='builtin',
        guides=guides,
        target_sequence=sequence,
        num_guides=len(guides),
        message=f"Found {len(guides)} guide RNAs for {pam_type}"
    )


def format_crispr_report(result):
    lines = [
        "=== CRISPR Guide RNA Design Report ===",
        f"Engine: {result.engine}",
        f"Target length: {len(result.target_sequence)} bp",
        f"Guides found: {result.num_guides}",
    ]
    if result.guides:
        lines.append(f"\n{'#':<4} {'Guide Sequence':<24} {'PAM':<6} {'Strand':<7} {'Score':>6} {'GC%':>6}")
        lines.append("-" * 65)
        for i, g in enumerate(result.guides[:10]):
            lines.append(f"{i+1:<4} {g.sequence:<24} {g.pam:<6} {g.strand:<7} {g.score:>6.3f} {g.gc_content:>5.1f}%")
    if result.message:
        lines.append(f"\nNote: {result.message}")
    return '\n'.join(lines)
