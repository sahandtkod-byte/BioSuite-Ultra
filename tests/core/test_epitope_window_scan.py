"""The sliding window itself: the peptides a scan returns and where it says they are.

Every caller reads these results by their coordinates — the notebook exports
`start`/`end`, the API serialises them — so a window that is one residue off its
own slice is a wrong answer even when the score is right.
"""
import pytest

from biosuite.core import epitope as ep

UPPER = "MAVPKRRTSRKEFVGIPSRLMNHQFSDLLSLFMESSGSTSPSINSTQIPSRKNHTWDEQQRGQIR"


def test_a_reported_window_always_indexes_the_sequence_it_came_from():
    for predictor, kwargs in [(ep.predict_t_cell_epitopes, {}),
                              (ep.predict_b_cell_epitopes, {}),
                              (ep.predict_linear_epitopes, {})]:
        for entry in predictor(UPPER, max_results=10 ** 6, **kwargs):
            assert entry.peptide == UPPER[entry.start:entry.end], (
                predictor.__name__, entry.to_dict())
            assert entry.end - entry.start == len(entry.peptide)
            assert 0 <= entry.start < entry.end <= len(UPPER)


def test_each_predictor_scans_the_window_sizes_it_advertises():
    t_cell = ep.predict_t_cell_epitopes(UPPER, max_results=10 ** 6)
    b_cell = ep.predict_b_cell_epitopes(UPPER, max_results=10 ** 6)
    linear = ep.predict_linear_epitopes(UPPER, max_results=10 ** 6)

    assert {len(e.peptide) for e in t_cell} <= {8, 9, 10, 11}, t_cell[0].peptide
    assert {len(e.peptide) for e in b_cell} == {7}
    assert {len(e.peptide) for e in linear} <= {10, 12, 15}


def test_every_predictor_returns_its_hits_strongest_first():
    for predictor in (ep.predict_t_cell_epitopes, ep.predict_b_cell_epitopes,
                      ep.predict_linear_epitopes):
        scores = [e.score for e in predictor(UPPER, max_results=10 ** 6)]

        assert scores == sorted(scores, reverse=True), predictor.__name__


def test_the_t_cell_score_stays_inside_the_unit_interval_it_is_thresholded_on():
    scores = [e.score for e in ep.predict_t_cell_epitopes(UPPER, threshold=0.0,
                                                          max_results=10 ** 6)]

    assert all(0.0 <= s <= 1.0 for s in scores), (min(scores), max(scores))
    # The weights are 0.3 + 0.3 + 0.2 + 0.2 * anchor_match, and anchor_match caps
    # at 0.6, so 1.0 is not reachable — the threshold a caller sets is a fraction
    # of 0.92, not of 1.
    assert max(scores) <= 0.3 + 0.3 + 0.2 + 0.2 * 0.6


def test_the_threshold_selects_how_many_windows_survive():
    strict = ep.predict_t_cell_epitopes(UPPER, threshold=0.8, max_results=10 ** 6)
    loose = ep.predict_t_cell_epitopes(UPPER, threshold=0.0, max_results=10 ** 6)

    assert all(e.score >= 0.8 for e in strict), strict
    assert len(loose) == sum(len(UPPER) - w + 1 for w in (8, 9, 10, 11))
    assert len(strict) < len(loose)


def test_a_top_hla_a0201_hit_ends_on_a_class_i_anchor_residue():
    top = ep.predict_t_cell_epitopes(UPPER)[0]

    assert top.peptide[-1] in set('VILM'), top.peptide


def test_the_hydrophobicity_term_is_the_only_one_that_uses_the_whole_window():
    """Same P2 and same P9, different middle: only the hydrophobicity can tell them apart."""
    polar = "MV" + "D" * 7 + "V"
    hydrophobic = "MV" + "I" * 7 + "V"

    low = ep.predict_t_cell_epitopes(polar, window_sizes=(10,), threshold=0.0)[0]
    high = ep.predict_t_cell_epitopes(hydrophobic, window_sizes=(10,), threshold=0.0)[0]

    assert (low.peptide[1], low.peptide[-1]) == (high.peptide[1], high.peptide[-1])
    assert high.score > low.score, (low.score, high.score)


def test_the_b_cell_score_is_a_weighted_average_of_four_scales():
    entry = ep.predict_b_cell_epitopes("EEEEEK", window_size=6, threshold=0.0)[0]

    assert entry.prediction_method == "surface-accessibility"
    assert entry.epitope_type == "B-cell"
    assert entry.score > 1.0, "glutamate-rich and lysine-rich is surface-friendly"


def test_the_linear_scan_reports_the_multi_property_method():
    results = ep.predict_linear_epitopes(UPPER, threshold=0.0, max_results=10 ** 6)

    assert {e.epitope_type for e in results} == {"linear-B-cell"}
    assert {e.prediction_method for e in results} == {"multi-property"}
    assert all(0.0 <= e.score <= 1.4 for e in results), max(e.score for e in results)


def test_a_sequence_shorter_than_any_window_answers_no_epitopes():
    for predictor in (ep.predict_t_cell_epitopes, ep.predict_b_cell_epitopes,
                      ep.predict_linear_epitopes):
        assert predictor("MA", max_results=10 ** 6) == [], predictor.__name__


def test_results_carry_their_type_into_the_json_view():
    entry = ep.predict_t_cell_epitopes(UPPER)[0]

    assert entry.to_dict()["type"] == "T-cell"
    assert entry.to_dict()["method"] == "MHC-A0201-binding"
