"""Fragment coordinates produced by a digestion, measured against the molecule.

These were the "assert res['cuts']"-style contract checks of the old end-to-end
file.  A truthy list proves nothing about a coordinate, so every size, start and
end below is the number the digest actually returns.
"""
from biosuite.core import cloning as cl

ONE_SITE = "ATCGGAATTC" + "A" * 100                  # 110 nt, EcoRI at 4, cut at 5
TWO_SITES = "GAATTC" + "T" * 30 + "GAATTC"           # 42 nt
LONG_MOLECULE = ("ACGT" * 50) + "GAATTC" + ("TTAA" * 50) + "GAATTC" + "GGCC" * 50


def test_a_single_cut_on_linear_dna_gives_two_pieces_that_rebuild_the_molecule():
    res = cl.simulate_digestion(ONE_SITE, "EcoRI", topology="linear")
    assert res["sizes"] == [5, 105]
    assert res["fragments"][0] == {"size": 5, "sequence": "ATCGG", "start": 0, "end": 5}
    assert res["fragments"][1]["start"] == 5 and res["fragments"][1]["end"] == 110
    assert "".join(f["sequence"] for f in res["fragments"]) == ONE_SITE
    assert sum(res["sizes"]) == res["total_bp"] == 110
    assert [(c["site_start"], c["cut_pos"]) for c in res["cuts"]] == [(4, 5)]


def test_the_same_cut_on_a_circle_linearises_it_into_one_full_length_piece():
    res = cl.simulate_digestion(ONE_SITE, "EcoRI")
    assert res["sizes"] == [110]
    assert [(f["start"], f["end"], f["size"]) for f in res["fragments"]] == [(5, 5, 110)]
    assert res["fragments"][0]["sequence"] == ONE_SITE[5:] + ONE_SITE[:5]


def test_two_sites_read_differently_on_a_circle_than_on_a_line():
    circular = cl.simulate_digestion(TWO_SITES, "EcoRI")
    assert circular["sizes"] == [36, 6]
    assert sum(circular["sizes"]) == 42
    linear = cl.simulate_digestion(TWO_SITES, "EcoRI", topology="linear")
    assert linear["sizes"] == [1, 36, 5]
    assert sum(linear["sizes"]) == 42


def test_a_molecule_with_three_sites_and_three_different_flanks():
    circular = cl.simulate_digestion(LONG_MOLECULE, "EcoRI")
    assert [(c["site_start"], c["cut_pos"]) for c in circular["cuts"]] == \
        [(200, 201), (406, 407)]
    assert circular["sizes"] == [206, 406]
    linear = cl.simulate_digestion(LONG_MOLECULE, "EcoRI", topology="linear")
    assert linear["sizes"] == [201, 206, 205]
    assert sum(linear["sizes"]) == len(LONG_MOLECULE) == 612


def test_every_fragment_is_internally_consistent_whichever_topology_is_chosen():
    for topology in ("circular", "linear"):
        res = cl.simulate_digestion(LONG_MOLECULE, ["EcoRI", "BamHI"], topology=topology)
        for frag in res["fragments"]:
            assert len(frag["sequence"]) == frag["size"]
            if frag["end"] > frag["start"]:
                assert frag["end"] - frag["start"] == frag["size"]
            else:
                assert (len(LONG_MOLECULE) - frag["start"]) + frag["end"] == frag["size"]
        assert sum(res["sizes"]) == len(LONG_MOLECULE)


def test_an_enzyme_with_no_site_returns_the_whole_molecule_as_one_fragment():
    res = cl.simulate_digestion("A" * 50, "EcoRI", topology="circular")
    assert res["cuts"] == []
    assert res["sizes"] == [50]
    assert res["fragments"] == [{"size": 50, "sequence": "A" * 50, "start": 0, "end": 50}]
