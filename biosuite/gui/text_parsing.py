"""Headless text helpers shared by the GUI.

These functions are pure string logic with **no GUI dependency**.  They live
outside ``biosuite.gui.tabs`` so they can be imported (and unit-tested) on a
machine without ``tkinter``/``customtkinter`` installed, which is the normal
situation on CI runners and in containers.
"""
from __future__ import annotations


def _parse_sequence_text(text):
    """Extract a DNA sequence from pasted text.

    Handles FASTA (header lines start with '>'), FASTQ (4-line records:
    @header / seq / + / quality — the quality line may itself start with
    A/C/G/T and must NOT be treated as sequence), and plain raw sequence.

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
            out.append(lines[i + 1].strip().replace(' ', ''))
            i += 4
        elif line.startswith('+'):
            i += 1  # stray FASTQ separator
        else:
            out.append(line.replace(' ', ''))
            i += 1
    return ''.join(out)
