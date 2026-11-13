"""pLDDT residue identity and the PDB path that was reserved before anything ran.

`_extract_plddt` keyed each residue by (chain, sequence number) and ignored the
PDB insertion-code column, so residues 1, 1A and 1B — three distinct residues in
an insertional alignment — collapsed into one score.  `predict_structure` reserved
a temporary `.pdb` before it knew whether any engine had produced a model,
leaving a 0-byte `biosuite_*.pdb` behind on every failed prediction.
"""
import inspect

import pytest

from biosuite.core import structure_prediction as sp

BASE = "ATOM      1  CA  ALA A   1      12.460  14.100   9.800  1.00 78.20           C"


def _icode(line, code):
    return line[:26] + code + line[27:]


def _bfactor(line, value):
    return line[:60] + f"{value:>6.2f}" + line[66:]


def test_insertion_codes_are_separate_residues():
    lines = [_bfactor(BASE, 90.0), _bfactor(_icode(BASE, 'A'), 80.0),
             _bfactor(_icode(BASE, 'B'), 70.0)]
    assert sp._extract_plddt("\n".join(lines)) == [90.0, 80.0, 70.0]


def test_atoms_of_one_residue_still_collapse_to_one_score():
    lines = [_bfactor(BASE, 90.0), _bfactor(_icode(BASE, ' '), 88.0)]
    assert sp._extract_plddt("\n".join(lines)) == [90.0]


def test_different_chains_with_the_same_number_are_separate():
    second = BASE[:21] + 'B' + BASE[22:]
    assert sp._extract_plddt("\n".join([_bfactor(BASE, 90.0), _bfactor(second, 60.0)])) \
        == [90.0, 60.0]


def test_a_failed_prediction_reserves_no_temporary_file(monkeypatch):
    """The reservation ran before the engines, so a failed model still cost a
    0-byte file whose path nobody was told about."""
    def forbid(*args, **kwargs):
        raise AssertionError("no path may be reserved when nothing was predicted")

    monkeypatch.setattr(sp, "secure_temp_path", forbid)
    monkeypatch.setattr(sp, "_esmfold_predict",
                        lambda seq, out: sp.PredictionResult(engine='esmfold',
                                                             message="ESMFold error: boom"))
    result = sp.predict_structure(sequence="MGSH")
    assert result.pdb_string == ""
    assert result.message == "ESMFold error: boom"


def test_a_successful_prediction_without_a_path_writes_one_and_reports_it(
        tmp_path, monkeypatch):
    target = str(tmp_path / "reserved.pdb")
    calls = []
    monkeypatch.setattr(sp, "secure_temp_path",
                        lambda suffix="": calls.append(suffix) or target)
    monkeypatch.setattr(sp, "_esmfold_predict",
                        lambda seq, out: sp.PredictionResult(
                            engine='esmfold', pdb_string="ATOM      1  CA  ALA A   1"
                                                         "      12.460  14.100   9.800"
                                                         "  1.00 78.20           C\n",
                            sequence="M", num_residues=1))

    result = sp.predict_structure(sequence="M")
    assert calls == ['.pdb'] and result.output_file == target
    assert open(target, encoding="utf-8").read().startswith("ATOM")


def test_a_caller_supplied_path_is_never_replaced_by_a_temporary_one(
        tmp_path, monkeypatch):
    chosen = str(tmp_path / "caller.pdb")
    monkeypatch.setattr(sp, "secure_temp_path",
                        lambda suffix="": pytest.fail("the caller's path must be used"))
    monkeypatch.setattr(sp, "_esmfold_predict",
                        lambda seq, out: sp.PredictionResult(
                            engine='esmfold', pdb_string="ATOM line\n",
                            output_file=out))
    result = sp.predict_structure(sequence="M", output_file=chosen)
    assert result.output_file == chosen


def test_the_module_imports_json_at_the_top_not_inside_a_function(monkeypatch):
    """The AlphaFold reader used `__import__('json')` inside the function body."""
    source = inspect.getsource(sp)
    assert "__import__(" not in source
    assert "\nimport json\n" in source
