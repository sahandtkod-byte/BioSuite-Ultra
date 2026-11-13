"""Proteasomal cleavage sites: every residue is judged, the last one included.

The loop stopped one short of the end while the expression inside it carried a
guard — `i + 1 < len(sequence)` — written for exactly the position the loop never
reached.  So the guard was dead code, and a favourable C-terminal residue, which
is a real cleavage preference, was missing from the list.
"""
import pytest

from biosuite.core import epitope as ep

GOOD = "AGSVLMT"
BAD = "PDERKNQHFYW"


def test_a_two_residue_sequence_reports_both_positions():
    """'LA' used to answer one entry: the terminal 'A' was never looked at."""
    assert ep.cleavage_site_prediction("LA") == [
        {"position": 0, "residue": "L", "preference": "favorable"},
        {"position": 1, "residue": "A", "preference": "favorable"},
    ]


def test_the_guard_for_a_missing_next_residue_is_reachable():
    """The last position has no residue after it, and is still favourable."""
    assert ep.cleavage_site_prediction("LV")[-1] == {
        "position": 1, "residue": "V", "preference": "favorable"}


def test_a_good_residue_followed_by_a_bad_one_is_listed_neither_way():
    """P at the next position cancels the preference for the A before it."""
    results = ep.cleavage_site_prediction("LAP")

    assert results == [{"position": 0, "residue": "L", "preference": "favorable"},
                       {"position": 2, "residue": "P", "preference": "unfavorable"}], results


@pytest.mark.parametrize("residue", list(GOOD))
def test_each_favourable_residue_ends_a_cleavage_site(residue):
    results = ep.cleavage_site_prediction("G" + residue)

    assert results[-1] == {"position": 1, "residue": residue,
                           "preference": "favorable"}, residue


@pytest.mark.parametrize("residue", list(BAD))
def test_each_unfavourable_residue_is_flagged_wherever_it_sits(residue):
    results = ep.cleavage_site_prediction("A" + residue + "A")

    assert {"position": 1, "residue": residue, "preference": "unfavorable"} in results, residue


def test_a_favourable_residue_is_not_also_listed_as_unfavourable():
    for residue in GOOD:
        entries = [e for e in ep.cleavage_site_prediction(residue + residue)
                   if e["residue"] == residue]
        assert len(entries) == 2, (residue, entries)
        assert {e["preference"] for e in entries} == {"favorable"}, residue


def test_the_two_preference_sets_do_not_overlap():
    assert not set(GOOD) & set(BAD)


def test_a_letter_no_scale_covers_is_skipped_and_said_aloud(caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.epitope"):
        results = ep.cleavage_site_prediction("LAX")

    assert [e["residue"] for e in results] == ["L", "A"], results
    assert "no property scale covers" in caplog.text, caplog.text


def test_case_does_not_change_the_answer():
    assert ep.cleavage_site_prediction("laP") == ep.cleavage_site_prediction("LAP")


def test_an_empty_sequence_has_no_cleavage_sites():
    assert ep.cleavage_site_prediction("") == []


def test_positions_index_the_normalised_sequence_exactly_once_each():
    sequence = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLV"

    results = ep.cleavage_site_prediction(sequence)

    assert [e["position"] for e in results] == sorted(e["position"] for e in results)
    assert len({e["position"] for e in results}) == len(results)
    assert max(e["position"] for e in results) == len(sequence) - 1, \
        "the terminal residue must be considered"
    for entry in results:
        assert sequence[entry["position"]].upper() == entry["residue"], entry
