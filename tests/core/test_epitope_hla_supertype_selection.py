"""Choosing the HLA supertype: a name the table does not know is a failure.

The CLI reads the HLA type from `input()`, the GUI workflow tab from a free-text
entry and the API from a plain `str` field, so whatever a user types reaches
`predict_t_cell_epitopes` as `mhc_type`.  An unknown name used to be answered with
HLA-A*02:01's anchors while the result was labelled with the name that was typed.
"""
import pytest

from biosuite.core import epitope as ep

UPPER = "MAVPKRRTSRKEFVGIPSRLMNHQFSDLLSLFMESSGSTSPSINSTQIPSRKNHTWDEQQRGQIR"
KNOWN = ("A0101", "A0201", "A0301", "A2402", "B0702", "B2705", "B4001", "B5801")


def test_every_supertype_in_the_table_is_answered_under_its_own_name():
    for name in KNOWN:
        results = ep.predict_t_cell_epitopes(UPPER, mhc_type=name, threshold=0.0,
                                             max_results=10 ** 6)
        assert results, name
        assert {e.prediction_method for e in results} == {f"MHC-{name}-binding"}, name


def test_a_name_the_table_does_not_know_is_refused_rather_than_substituted():
    """Measured on the old code: a typo answered A0201's peptides, labelled
    `MHC-B0702_typo-binding`."""
    with pytest.raises(ValueError, match="Unknown HLA supertype 'B0702_typo'"):
        ep.predict_t_cell_epitopes(UPPER, mhc_type="B0702_typo")


def test_the_refusal_lists_the_names_that_would_have_worked():
    with pytest.raises(ValueError) as exc:
        ep.predict_t_cell_epitopes(UPPER, mhc_type="DRB1")

    assert ("Known: A0101, A0201, A0301, A2402, B0702, B2705, B4001, B5801."
            in str(exc.value)), str(exc.value)


@pytest.mark.parametrize("typed", ["a0201", " A0201 ", "A0201"])
def test_the_name_is_matched_loosely_but_answered_canonically(typed):
    results = ep.predict_t_cell_epitopes(UPPER, mhc_type=typed)

    assert results, results
    assert {e.prediction_method for e in results} == {"MHC-A0201-binding"}, results[0]


@pytest.mark.parametrize("typed", [None, 2, ["A0201"], ""])
def test_a_missing_or_non_textual_type_is_refused(typed):
    with pytest.raises(ValueError, match="HLA supertype"):
        ep.predict_t_cell_epitopes(UPPER, mhc_type=typed)


def test_the_lookup_returns_the_supertype_it_was_asked_for():
    key, anchors = ep._mhc_anchors(" b0702 ")

    assert key == "B0702"
    assert anchors is ep.HLA_SUPERTYPES["B0702"]
    assert anchors["P2"] == "P", "B*07:02 is the proline-at-P2 supertype"


def test_the_anchor_set_changes_the_scores_of_the_same_peptide():
    """If the supertype never mattered, asking for one would be a decoration."""
    sequence = "MAKWRPTVLMLIKQNAGA"

    def scores(name):
        return [round(e.score, 12) for e in ep.predict_t_cell_epitopes(
            sequence, mhc_type=name, window_sizes=(9,), threshold=0.0,
            max_results=10 ** 6)]

    assert scores("A0201") != scores("A0301"), (scores("A0201"), scores("A0301"))
    assert len(scores("A0201")) == len(sequence) - 9 + 1


def test_the_default_supertype_is_the_one_the_rest_of_the_module_documents():
    def key(results):
        return [(e.peptide, round(e.score, 12), e.start, e.end) for e in results]

    assert key(ep.predict_t_cell_epitopes(UPPER)) == \
        key(ep.predict_t_cell_epitopes(UPPER, mhc_type="A0201"))
