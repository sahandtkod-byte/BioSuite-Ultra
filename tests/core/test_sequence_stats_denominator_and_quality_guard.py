"""Composition and quality statistics of ``core/sequence.py``.

The three sibling functions that take a sequence — `gc_content`,
`reverse_complement`, `translate` — already refuse text that is not nucleotides, and
`sequence_stats` and `quality_stats` did not: measured on HEAD, `sequence_stats(None)`
returned a full dict of zeros, `sequence_stats('ACGT!@#')` returned
`AT=28.57, GC=28.57, other=42.86` where `gc_content` raised, and `quality_stats('A B')`
reported the space as **Q-1**.

Before/after numbers were measured on the HEAD copy of the module (executed in memory
from `git show HEAD:biosuite/core/sequence.py`) or on the live module where the value
does not depend on the edits.
"""
import re

import pytest

from biosuite.core import sequence as seq_mod
from biosuite.core.sequence import gc_content, quality_stats, sequence_stats

HEAD_KEYS = {'length', 'A', 'T', 'G', 'C', 'N', 'AT', 'GC', 'other'}


# ── sequence_stats: what counts as a base ────────────────────────────────────

def test_gap_characters_stay_out_of_the_denominator_as_gc_content_already_does():
    """HEAD: `sequence_stats('AC-GT.*N')` answered GC 25.0 (two of eight lines)
    while `gc_content` of the same string answered 40.0 (two of five bases)."""
    s = "AC-GT.*N"
    stats = sequence_stats(s)
    assert stats == {'length': 8, 'A': 1, 'T': 1, 'G': 1, 'C': 1, 'N': 1,
                     'AT': 40.0, 'GC': 40.0, 'other': 0.0}
    assert stats['GC'] == gc_content(s)


def test_a_plain_sequence_reports_the_expected_percentages():
    assert sequence_stats("ATGC") == {
        'length': 4, 'A': 1, 'T': 1, 'G': 1, 'C': 1, 'N': 0,
        'AT': 50.0, 'GC': 50.0, 'other': 0.0}
    assert sequence_stats("ACGTN")['GC'] == 2 / 5 * 100


def test_ambiguity_codes_are_the_other_bucket_not_garbage():
    stats = sequence_stats("ACGTRYSWKMBDHVN")
    assert stats['length'] == 15
    # 15 bases, 5 of them A/T/G/C/N, so the ten IUPAC codes are two thirds of the
    # counted positions — 'other' is a real composition statement, not an error bin.
    assert stats['other'] == pytest.approx(10 / 15 * 100)
    assert stats['AT'] == pytest.approx(2 / 15 * 100)


def test_an_all_gap_column_reports_zeros_instead_of_dividing_by_zero():
    assert sequence_stats("---") == {
        'length': 3, 'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0,
        'AT': 0.0, 'GC': 0.0, 'other': 0.0}


def test_an_empty_sequence_keeps_returning_the_zero_dict():
    assert sequence_stats("") == {'length': 0, 'A': 0, 'T': 0, 'G': 0, 'C': 0,
                                  'N': 0, 'AT': 0.0, 'GC': 0.0, 'other': 0.0}


@pytest.mark.parametrize("bad,type_name", [(None, 'NoneType'), (123, 'int'),
                                           ([1, 2], 'list'), (3.5, 'float')])
def test_sequence_stats_refuses_a_non_string_instead_of_measuring_nothing(bad, type_name):
    """HEAD took ``None`` down the "empty" branch and printed a complete dict of
    zeros — a measurement of a sequence that was never supplied — while an int died
    inside numpy with `AttributeError: 'int' object has no attribute 'upper'`."""
    with pytest.raises(TypeError, match=(
            rf"^seq must be a string of nucleotides, got {type_name}$")):
        sequence_stats(bad)


def test_sequence_stats_refuses_the_text_that_gc_content_already_refuses():
    """One module, two answers for the same input was the inconsistency: HEAD gave
    `AT=28.57, GC=28.57, other=42.86` for 'ACGT!@#' while gc_content raised."""
    expect = ("seq contains characters that are not IUPAC nucleotides: "
              "'!#@'. Pass a DNA/RNA sequence")
    with pytest.raises(ValueError, match=rf"^{re.escape(expect)}"):
        sequence_stats("ACGT!@#")
    with pytest.raises(ValueError, match=rf"^{re.escape(expect)}"):
        gc_content("ACGT!@#")


def test_the_key_set_of_sequence_stats_did_not_change():
    assert set(sequence_stats("ACGT")) == HEAD_KEYS


# ── quality_stats: the Phred+33 range ────────────────────────────────────────

def test_quality_stats_reads_the_printable_phred_range():
    got = quality_stats("!~4")
    assert got['scores'] == [0, 93, 19]
    assert got['positions'] == [1, 2, 3]
    assert got['min'] == 0 and got['max'] == 93
    assert got['mean'] == pytest.approx((0 + 93 + 19) / 3)


def test_quality_stats_refuses_a_character_that_is_not_a_quality():
    """HEAD turned a space into Q-1 and a tab into Q-24 and reported them in the same
    dict as real scores, next to a docstring that explains Q10 as 90 % accuracy."""
    with pytest.raises(ValueError, match=re.escape(
            "qual_string contains characters outside the Phred+33 range "
            "('!' to '~'): ' ' (Q-1). Decode")):
        quality_stats("A B")
    with pytest.raises(ValueError, match=re.escape("'\\t' (Q-24)")):
        quality_stats("A\tB")


@pytest.mark.parametrize("bad,type_name", [(None, 'NoneType'), ([40, 40], 'list'),
                                           (40, 'int')])
def test_quality_stats_refuses_a_non_string(bad, type_name):
    """``[40, 40]`` used to raise `TypeError: ord() expected string of length 1, but
    int found` from inside the comprehension, and ``None`` answered a dict of zeros."""
    with pytest.raises(TypeError, match=(
            rf"^qual_string must be a FASTQ quality string, got {type_name}$")):
        quality_stats(bad)


def test_quality_stats_of_an_empty_string_is_still_the_zero_dict():
    assert quality_stats("") == {'mean': 0.0, 'min': 0, 'max': 0,
                                 'positions': [], 'scores': []}


# ── the consumers that must not be broken by the guards ──────────────────────

def test_a_fastq_record_round_trips_through_both_helpers(tmp_path):
    """The point of the guards is that a real FASTQ still measures cleanly."""
    name, sequence, quality = seq_mod.read_fastq(
        _write(tmp_path, "@r1\nACGTACGT\n+\nIIIIIIII\n"))[0]
    assert sequence_stats(sequence)['GC'] == 50.0
    assert quality_stats(quality)['min'] == ord('I') - 33 == 40


def _write(tmp_path, text):
    path = tmp_path / "reads.fq"
    path.write_text(text, encoding="utf-8")
    return str(path)
