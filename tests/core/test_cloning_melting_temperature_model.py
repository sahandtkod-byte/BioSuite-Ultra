"""What `design_primers` actually computes for a melting temperature.

The helper described itself as a nearest-neighbour calculation using "SantaLucia
1998 unified parameters".  Nearest-neighbour thermodynamics depend on the order
of the bases; this formula reads only the counts, and three primers with the same
composition in three different orders get one identical value.  The tests below
pin the two regimes and the order-independence, so the number can never quietly
become something else while still being sold as a thermodynamic model.
"""
import pytest

from biosuite.core import cloning as cl


def _wallace(seq):
    n_a, n_t, n_g, n_c = (seq.count(b) for b in "ATGC")
    return 2.0 * (n_a + n_t) + 4.0 * (n_g + n_c)


def _composition_formula(seq):
    n_gc = seq.count("G") + seq.count("C")
    return 64.9 + 41.0 * (n_gc - 16.4) / len(seq)


@pytest.mark.parametrize("seq", [
    "AAAACCCCGGGGTTTT",
    "ACGTACGTACGTACGT",
    "TTTTGGGGCCCCAAAA",
])
def test_base_order_does_not_enter_the_model(seq):
    assert cl._melting_temp(seq) == pytest.approx(43.375000000000014)
    assert cl._melting_temp(seq) == pytest.approx(
        _composition_formula("AAAACCCCGGGGTTTT"))


def test_thirteen_bases_are_still_read_by_the_wallace_rule():
    seq = "AAAACCCCGGGGT"                      # 13 nt: 5 A/T, 8 G/C
    assert len(seq) == 13
    assert cl._melting_temp(seq) == _wallace(seq) == 42.0
    # the long-form branch would have answered 38.4 °C for the same bases
    assert cl._melting_temp(seq) != pytest.approx(_composition_formula(seq))


def test_fourteen_bases_switch_to_the_composition_formula():
    seq = "AAAACCCCGGGGTT"                     # 14 nt: 8 A/T, 8 G/C
    assert len(seq) == 14
    assert cl._melting_temp(seq) == pytest.approx(40.30000000000001)
    assert cl._melting_temp(seq) == pytest.approx(_composition_formula(seq))
    assert cl._melting_temp(seq) != pytest.approx(_wallace(seq))


def test_a_twenty_mer_of_alternating_bases_is_read_low_by_design():
    # Reported so that the absolute value is not mistaken for a measured Tm:
    # 20 nt of 50 % GC anneals near 60 °C, not at this number.
    assert cl._melting_temp("ATGC" * 5) == pytest.approx(51.78000000000001)


@pytest.mark.parametrize("seq,expected", [
    ("", 0.0),
    ("NNNN", 0.0),
    ("RYWKMBDHV", 0.0),
    ("ACGT", 0.5),
    ("GGCANT", 0.6),          # the N is outside both the numerator and the length
    ("GC", 1.0),
    ("AT", 0.0),
])
def test_gc_content_is_taken_over_the_four_concrete_bases_only(seq, expected):
    assert cl._gc_content(seq) == pytest.approx(expected)


def test_an_ambiguous_sequence_has_no_melting_temperature_rather_than_a_false_one():
    assert cl._melting_temp("NNNNNNNNNNNNNN") == 0.0
    assert cl._melting_temp("atgc") == cl._melting_temp("ATGC")
