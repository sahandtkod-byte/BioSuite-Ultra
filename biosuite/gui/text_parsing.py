"""Headless text helpers shared by the GUI.

These functions are pure string logic with **no GUI dependency**.  They live
outside ``biosuite.gui.tabs`` so they can be imported (and unit-tested) on a
machine without ``tkinter``/``customtkinter`` installed, which is the normal
situation on CI runners and in containers.
"""
from __future__ import annotations

import unicodedata

_SEARCH_SYNONYMS = {
    'توالی': 'sequence',
    'سکانس': 'sequence',
    'تراز': 'alignment',
    'همردیف': 'alignment',
    'الاینمنت': 'alignment',
    'ژن': 'gene',
    'ژنوم': 'genome',
    'پروتئین': 'protein',
    'بیان': 'expression',
    'درخت': 'tree',
    'فیلوژنی': 'phylogeny',
    'تک سلول': 'single cell',
    'ساختار': 'structure',
    'واریانت': 'variant',
    'جهش': 'variant',
    'میکروبیوم': 'microbiome',
    'متابولیت': 'metabolite',
    'متابولومیک': 'metabolomics',
    'بقا': 'survival',
    'جمعیت': 'population',
}

_PERSIAN_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')


def normalise_search_text(text):
    """Fold case, accents, Persian digits and Persian Y/K variants."""
    text = text.translate(_PERSIAN_DIGITS)
    text = text.replace('ي', 'ی').replace('ك', 'ک')
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def _one_letter_off(a, b):
    if len(a) != len(b):
        return False
    return sum(1 for x, y in zip(a, b) if x != y) <= 1


def search_score(query, haystack):
    """How well *query* matches *haystack*; 0.0 means "does not match".

    Every query token must match something in *haystack* (in any order) for a
    non-zero score.  A token scores 1.0 as an exact substring, 0.6 for a
    one-letter typo (tokens of at least 4 characters) or a synonym mapping.
    The result is the mean token score: completeness counts, not length.
    """
    query = normalise_search_text(query)
    hay = normalise_search_text(haystack)
    tokens = query.split()
    if not tokens:
        return 0.0
    total = 0.0
    for token in tokens:
        terms = {token}
        if token in _SEARCH_SYNONYMS:
            terms.add(_SEARCH_SYNONYMS[token])
        if query in _SEARCH_SYNONYMS:
            terms.add(_SEARCH_SYNONYMS[query])
        best = 0.0
        for term in terms:
            if term in hay:
                best = 1.0
                break
            if len(term) >= 4:
                for i in range(len(hay) - len(term) + 1):
                    if _one_letter_off(term, hay[i:i + len(term)]):
                        best = 0.6
                        break
            if best:
                break
        if not best:
            return 0.0
        total += best
    return total / len(tokens)


def _parse_sequence_text(text):
    """Extract a DNA sequence from pasted text.

    Handles FASTA (header lines start with '>'), FASTQ (4-line records:
    @header / seq / + / quality — the quality line may itself start with
    A/C/G/T and must NOT be treated as sequence), and plain raw sequence.
    Whitespace inside a sequence line (a tab pasted from a spreadsheet, say)
    is removed before the line is joined.

    Pure string logic kept module-level so it is unit-testable headlessly.
    """
    text = text.strip()
    if not text:
        return ""
    lines = text.split('\n')
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith('>'):
            i += 1  # FASTA header
        elif line.startswith('@') and i + 2 < n and lines[i + 2].lstrip().startswith('+'):
            # FASTQ record: header, sequence, +, quality
            out.append(''.join(lines[i + 1].split()))
            i += 4
        elif line.startswith('+'):
            i += 1  # stray FASTQ separator
        elif line.startswith('@'):
            i += 1  # a header that is not part of a FASTQ record — never sequence
        else:
            out.append(''.join(line.split()))
            i += 1
    return ''.join(out)
