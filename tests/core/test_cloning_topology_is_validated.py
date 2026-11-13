"""The molecule's shape decides the fragment count, so it must not be guessed.

`simulate_digestion` compared *topology* against the exact word "circular", which
made every other spelling -- including a capital C -- a linear digest: a plasmid
with one origin-straddling site reported two fragments instead of one, and said
nothing.
"""
import pytest

from biosuite.core import cloning as cl

# One EcoRI site whose cut (offset 1) sits at position 5 of a 30 bp molecule.
SEQUENCE = "TTTT" + "GAATTC" + "A" * 20


@pytest.mark.parametrize("spelling", ["circular", " Circular ", "CIRCULAR", "cIrcUlAr"])
def test_a_circular_spelling_that_only_differs_in_case_or_spaces_is_circular(spelling):
    res = cl.simulate_digestion(SEQUENCE, "EcoRI", topology=spelling)
    assert res["sizes"] == [30]
    assert len(res["fragments"]) == 1


@pytest.mark.parametrize("spelling", ["linear", " Linear", "LINEAR "])
def test_the_same_molecule_read_as_linear_gives_the_two_pieces(spelling):
    res = cl.simulate_digestion(SEQUENCE, "EcoRI", topology=spelling)
    assert res["sizes"] == [5, 25]


@pytest.mark.parametrize("bad", ["circle", "plasmid", "linear or circular",
                                 "", " ", None, 0, True, ["circular"]])
def test_a_topology_that_is_not_one_of_the_two_words_is_refused(bad):
    with pytest.raises(ValueError) as exc:
        cl.simulate_digestion(SEQUENCE, "EcoRI", topology=bad)
    assert str(exc.value) == (
        f"Unknown topology: {bad!r}. Use 'circular' (a plasmid) or 'linear' "
        "(a PCR product or genomic fragment).")


def test_the_canonical_spellings_are_the_documented_pair():
    assert cl.TOPOLOGIES == ("circular", "linear")


def test_the_default_topology_is_the_one_a_plasmid_needs():
    # Without the default, every GUI/CLI caller that omits the argument would
    # have to restate it to get a circular digest at all.
    assert cl.simulate_digestion(SEQUENCE, "EcoRI")["sizes"] == [30]
    assert cl.simulate_digestion(SEQUENCE, "EcoRI", topology="circular")["sizes"] == [30]
