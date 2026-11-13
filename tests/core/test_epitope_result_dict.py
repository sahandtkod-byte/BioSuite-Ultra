"""The epitope result dictionary — the one serialisable view of a prediction.

`EpitopeResult.to_dict` is what the API and the notebook export hand to JSON; it
rounds the score and renames three fields, so its exact key set and rounding are
the contract.
"""
import pytest

from biosuite.core.epitope import EpitopeResult


def test_to_dict_exposes_the_prediction_as_plain_json_values():
    result = EpitopeResult(peptide="GLPLSLCTF", start=4, end=13, score=0.8123456,
                           epitope_type="T-cell",
                           prediction_method="MHC-A0201-binding")
    assert result.to_dict() == {
        "peptide": "GLPLSLCTF",
        "start": 4,
        "end": 13,
        "score": 0.812,
        "type": "T-cell",
        "method": "MHC-A0201-binding",
    }


def test_to_dict_uses_the_constructor_defaults():
    assert EpitopeResult("PEPTIDE", 0, 7, 0.5).to_dict() == {
        "peptide": "PEPTIDE", "start": 0, "end": 7, "score": 0.5,
        "type": "T-cell", "method": "matrix"}


def test_the_score_is_rounded_not_truncated_and_stays_a_float():
    entry = EpitopeResult("X", 0, 1, 0.9996).to_dict()
    assert entry["score"] == pytest.approx(1.0)
    assert isinstance(entry["score"], float)
    assert EpitopeResult("X", 0, 1, 1e-9).to_dict()["score"] == 0.0


def test_the_json_view_can_be_serialised_directly():
    import json
    text = json.dumps(EpitopeResult("AAAA", 1, 5, 0.25).to_dict(), sort_keys=True)
    assert '"method": "matrix"' in text and '"score": 0.25' in text
