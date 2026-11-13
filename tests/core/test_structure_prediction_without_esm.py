"""biosuite.core.structure_prediction without the ESM model weights.

The result record, pLDDT extraction, the built-in fallback predictor and the report
text, all on the path that needs no torch install.
"""
import pytest


def _has_esm():
    try:
        import esm
        import torch
        return True
    except ImportError:
        return False


class TestStructurePredictionDataclass:
    def test_prediction_result_defaults(self):
        from biosuite.core.structure_prediction import PredictionResult
        r = PredictionResult(engine='esmfold')
        assert r.pdb_string == ""
        assert r.plddt_scores == []
        assert r.confidence == 0.0
        assert r.num_residues == 0


class TestExtractPlddt:
    def test_extract_plddt_basic(self):
        from biosuite.core.structure_prediction import _extract_plddt
        pdb = (
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 95.23\n"
            "ATOM      2  CA  ALA A   2       4.000   5.000   6.000  1.00 80.50\n"
            "ATOM      3  CA  ALA A   3       7.000   8.000   9.000  1.00 65.00\n"
            "END\n"
        )
        scores = _extract_plddt(pdb)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(95.23, abs=0.01)
        assert scores[1] == pytest.approx(80.50, abs=0.01)
        assert scores[2] == pytest.approx(65.00, abs=0.01)

    def test_extract_plddt_empty(self):
        from biosuite.core.structure_prediction import _extract_plddt
        scores = _extract_plddt("END\n")
        assert scores == []

    def test_extract_plddt_short_line(self):
        from biosuite.core.structure_prediction import _extract_plddt
        # Line too short to have B-factor field
        scores = _extract_plddt("ATOM  short line\nEND\n")
        assert scores == []


class TestStructurePredictionPredict:
    def test_predict_structure_no_input(self):
        from biosuite.core.structure_prediction import predict_structure
        result = predict_structure()
        assert result.engine == 'none'
        assert "No sequence" in result.message

    @pytest.mark.skipif(
        not _has_esm(), reason="ESM/torch not installed"
    )
    def test_predict_structure_with_sequence(self):
        from biosuite.core.structure_prediction import predict_structure
        # Short sequence to keep test fast
        result = predict_structure(sequence="ACDEF")
        assert result.engine == 'esmfold'
        assert result.pdb_string != ""
        assert result.num_residues == 5

    @pytest.mark.skipif(
        not _has_esm(), reason="ESM/torch not installed"
    )
    def test_predict_structure_with_output_file(self, tmp_path):
        from biosuite.core.structure_prediction import predict_structure
        out = tmp_path / "output.pdb"
        result = predict_structure(sequence="ACDEFGHIK", output_file=str(out))
        assert out.exists()
        assert result.output_file == str(out)


class TestStructurePredictionFormatReport:
    def test_format_report_basic(self):
        from biosuite.core.structure_prediction import (
            PredictionResult,
            format_prediction_report,
        )
        r = PredictionResult(engine='esmfold', num_residues=100, confidence=85.0,
                             output_file='/tmp/test.pdb', message="test")
        text = format_prediction_report(r)
        assert "Structure Prediction Report" in text
        assert "100" in text
        assert "85.0%" in text

    def test_format_report_with_plddt(self):
        from biosuite.core.structure_prediction import (
            PredictionResult,
            format_prediction_report,
        )
        r = PredictionResult(engine='esmfold', num_residues=3,
                             confidence=80.0, plddt_scores=[95.0, 80.0, 60.0],
                             output_file='out.pdb')
        text = format_prediction_report(r)
        assert "Confident (>90): 1" in text
        assert "Good (>70-90): 1" in text
        assert "Low (<=70): 1" in text
