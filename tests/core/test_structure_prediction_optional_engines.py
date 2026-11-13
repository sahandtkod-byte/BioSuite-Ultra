"""Structure-prediction contracts for the optional engines around ESMFold and the
AlphaFold DB API: dependency guards re-executed from scratch, a faked ``esm`` model,
a faked ``urllib`` transport, per-residue pLDDT extraction, and the messages that
must reach the caller when an engine cannot produce a model.

No network request leaves the process and no torch model is ever downloaded.
"""
import contextlib
import importlib.util
import json
import pathlib
import sys

import pytest

from biosuite.core import structure_prediction as sp

PDB_TEXT = (
    "ATOM      1  N   ALA A   1      11.104  13.207   9.447  1.00 95.40           N\n"
    "ATOM      2  CA  ALA A   1      12.460  14.100   9.800  1.00 95.40           C\n"
    "ATOM      3  N   GLY A   2      13.500  14.900  10.100  1.00 62.00           N\n"
    "HETATM    4  ZN  ZN A 501      15.000  15.000  10.500  1.00 33.50          ZN\n"
)


def _load_instance(monkeypatch, tag, **fake_modules):
    """Re-execute the module top level with the given optional imports faked.

    ``importlib.reload`` cannot toggle these: the attributes bound by the first
    import survive a reload.  The dotted name stays inside ``biosuite.core``
    because the module does ``from .utils import secure_temp_path``.
    """
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    full_name = f"biosuite.core.structure_prediction_{tag}"
    spec = importlib.util.spec_from_file_location(full_name, pathlib.Path(sp.__file__))
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, full_name, fresh)
    spec.loader.exec_module(fresh)
    return fresh


class _FakeTorch:
    """Only ``no_grad`` is used by the module under test."""

    def __init__(self):
        self.no_grad_calls = 0

    @contextlib.contextmanager
    def no_grad(self):
        self.no_grad_calls += 1
        yield


def _fake_esm(pdb_text=PDB_TEXT, error=None):
    calls = []

    class _Model:
        def eval(self):
            calls.append("eval")
            return self

        def infer_pdb(self, sequence):
            calls.append(sequence)
            if error is not None:
                raise error
            return pdb_text

    module = __import__("types").ModuleType("esm")
    module.pretrained = __import__("types").SimpleNamespace(
        esmfold_v1=lambda: calls.append("load") or _Model())
    return module, calls


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


def _fake_urllib(payload_by_url, error=None):
    """A stand-in for ``urllib.request`` keyed by URL; records (url, timeout)."""
    seen = []

    class _Request:
        def __init__(self, url):
            self.url = url

    def urlopen(request, timeout=None):
        seen.append((request.url, timeout))
        if error is not None:
            raise error
        return _Response(payload_by_url(request.url))

    return __import__("types").SimpleNamespace(
        request=__import__("types").SimpleNamespace(Request=_Request, urlopen=urlopen)), seen


@pytest.fixture
def esm_available(monkeypatch):
    torch = _FakeTorch()
    esm, calls = _fake_esm()
    fresh = _load_instance(monkeypatch, "with_esm", esm=esm, torch=torch)
    assert fresh.HAS_ESM is True, "the faked dependency must be the one under test"
    fresh.torch_stub, fresh.esm_calls = torch, calls
    return fresh


@pytest.fixture
def esm_absent(monkeypatch):
    fresh = _load_instance(monkeypatch, "without_esm", esm=None, torch=None)
    assert fresh.HAS_ESM is False
    return fresh


@pytest.fixture
def urllib_absent(monkeypatch):
    fresh = _load_instance(monkeypatch, "without_urllib", urllib=None)
    assert fresh.HAS_URLLIB is False
    return fresh


def test_esm_import_failure_is_reported_by_the_engine_probe_and_the_predictor(esm_absent):
    assert esm_absent.check_prediction_tools()['esmfold'] is False
    result = esm_absent._esmfold_predict("MGSHLVDAL")
    assert result.engine == "esmfold"
    assert result.pdb_string == "" and result.plddt_scores == []
    assert result.message == "esm library not installed. Run: pip install esm"


def test_urllib_import_failure_short_circuits_the_alphafold_api(urllib_absent):
    result = urllib_absent._alphafold_fetch("P04637")
    assert (result.engine, result.pdb_string, result.message) == (
        "alphafold", "", "urllib not available")


def test_check_prediction_tools_reports_an_installed_torch(esm_available):
    assert esm_available.check_prediction_tools() == {'esmfold': True, 'torch': True}


def test_esmfold_prediction_runs_the_model_and_scores_each_residue(esm_available, tmp_path):
    out = str(tmp_path / "model.pdb")
    result = esm_available._esmfold_predict("MGSH", output_file=out)
    assert esm_available.esm_calls == ["load", "eval", "MGSH"]
    assert esm_available.torch_stub.no_grad_calls == 1, "inference must run under no_grad"
    assert result.engine == "esmfold"
    assert result.pdb_string == PDB_TEXT
    assert result.plddt_scores == pytest.approx([95.40, 62.00, 33.50])
    assert result.confidence == pytest.approx(63.63333333333333)
    assert (result.sequence, result.num_residues) == ("MGSH", 4)
    assert result.output_file == out
    assert pathlib.Path(out).read_text(encoding="utf-8") == PDB_TEXT
    assert result.message == "ESMFold prediction: 4 residues, confidence 63.6%"


def test_esmfold_without_an_output_file_writes_nothing(esm_available):
    result = esm_available._esmfold_predict("MGSH")
    assert result.output_file == ""
    assert result.plddt_scores == pytest.approx([95.40, 62.00, 33.50])


def test_esmfold_model_failure_becomes_the_engine_message(esm_available):
    boom = _fake_esm(error=RuntimeError("CUDA out of memory"))[0]
    esm_available.esm = boom
    result = esm_available._esmfold_predict("MGSH")
    assert result.engine == "esmfold"
    assert result.message == "ESMFold error: CUDA out of memory"
    assert result.pdb_string == "" and result.confidence == 0.0


def test_plddt_extraction_skips_lines_with_an_unreadable_b_factor():
    broken = ("ATOM      5  CA  ALA A   3      14.000  15.000  10.900  1.00  ??.??           C\n")
    scores = sp._extract_plddt(broken + PDB_TEXT)
    assert scores == pytest.approx([95.40, 62.00, 33.50]), "the unparseable residue is dropped"
    assert sp._extract_plddt("ATOM 1 N ALA A 1") == [], "a truncated record is ignored"
    assert sp._extract_plddt("TER\nEND\n") == []


def test_plddt_extraction_is_one_value_per_residue_across_chains():
    multi = (
        "ATOM      1  CA  ALA A   1      11.104  13.207   9.447  1.00 90.00           C\n"
        "ATOM      2  CB  ALA A   1      11.204  13.207   9.447  1.00 90.00           C\n"
        "ATOM      3  CA  ALA B   1      12.104  13.207   9.447  1.00 80.00           C\n"
    )
    assert sp._extract_plddt(multi) == pytest.approx([90.0, 80.0])


def test_alphafold_fetch_downloads_the_model_and_scores_it(esm_available, tmp_path):
    transport, seen = _fake_urllib(lambda url: (
        json.dumps([{"pdbUrl": "https://models.example.org/1234-model.pdb"}]).encode()
        if url.startswith("https://alphafold.ebi.ac.uk") else PDB_TEXT.encode()))
    esm_available.urllib = transport
    out = str(tmp_path / "af.pdb")
    result = esm_available._alphafold_fetch("P04637", out)
    assert seen == [("https://alphafold.ebi.ac.uk/api/prediction/P04637", 30),
                    ("https://models.example.org/1234-model.pdb", 60)]
    assert result.engine == "alphafold"
    assert result.pdb_string == PDB_TEXT
    assert result.plddt_scores == pytest.approx([95.40, 62.00, 33.50])
    assert result.num_residues == 3, "AlphaFold counts parsed residues, not a query string"
    assert result.confidence == pytest.approx(63.63333333333333)
    assert result.message == "AlphaFold DB: P04637, confidence 63.6%"
    assert pathlib.Path(out).read_text(encoding="utf-8") == PDB_TEXT


def test_alphafold_answer_without_a_pdb_url_reports_no_prediction(esm_available):
    transport, seen = _fake_urllib(lambda url: json.dumps([{"entry": "x"}]).encode())
    esm_available.urllib = transport
    result = esm_available._alphafold_fetch("Q99999")
    assert (result.engine, result.pdb_string, result.output_file) == ("alphafold", "", "")
    assert result.message == "No prediction found"
    assert seen == [("https://alphafold.ebi.ac.uk/api/prediction/Q99999", 30)]


def test_alphafold_keeps_the_model_in_memory_when_no_output_file_was_given(esm_available):
    transport, _ = _fake_urllib(lambda url: (
        json.dumps([{"pdbUrl": "https://models.example.org/1234-model.pdb"}]).encode()
        if url.startswith("https://alphafold.ebi.ac.uk") else PDB_TEXT.encode()))
    esm_available.urllib = transport
    result = esm_available._alphafold_fetch("P04637")
    assert result.output_file == ""
    assert result.pdb_string == PDB_TEXT
    assert result.plddt_scores == pytest.approx([95.40, 62.00, 33.50])


def test_alphafold_network_failure_is_surfaced_as_an_engine_error(esm_available):
    transport, _ = _fake_urllib(lambda url: b"", error=OSError("offline"))
    esm_available.urllib = transport
    result = esm_available._alphafold_fetch("P04637")
    assert result.engine == "alphafold" and result.pdb_string == ""
    assert result.message == "AlphaFold API error: offline"


def test_predict_structure_prefers_a_hit_from_the_requested_source(esm_available, tmp_path):
    transport, seen = _fake_urllib(lambda url: (
        json.dumps([{"pdbUrl": "https://models.example.org/1234-model.pdb"}]).encode()
        if url.startswith("https://alphafold.ebi.ac.uk") else PDB_TEXT.encode()))
    esm_available.urllib = transport
    esm_available.esm = _fake_esm(pdb_text="ATOM      1  CA  ALA A   1      1.000   2.000   3.000  1.00 50.00           C\n")[0]

    by_id = esm_available.predict_structure(uniprot_id="P04637",
                                            output_file=str(tmp_path / "by_id.pdb"))
    assert by_id.engine == "alphafold" and seen, "the API result wins when it has a model"

    by_sequence = esm_available.predict_structure(sequence="MGSH",
                                                  output_file=str(tmp_path / "by_seq.pdb"))
    assert by_sequence.engine == "esmfold"
    assert by_sequence.num_residues == 4
    assert by_sequence.output_file == str(tmp_path / "by_seq.pdb")


def test_predict_structure_surfaces_the_failed_engine_instead_of_a_generic_note(esm_available):
    transport, _ = _fake_urllib(lambda url: json.dumps([{"no": "pdb"}]).encode())
    esm_available.urllib = transport
    result = esm_available.predict_structure(uniprot_id="Q99999")
    assert (result.engine, result.message) == ("alphafold", "No prediction found")

    esm_available.esm = _fake_esm(error=RuntimeError("no weights locally"))[0]
    result = esm_available.predict_structure(sequence="MGSH")
    assert result.engine == "esmfold"
    assert result.message == "ESMFold error: no weights locally"


def test_predict_structure_reserves_a_temporary_path_when_none_is_given(esm_available):
    result = esm_available.predict_structure()
    assert result.engine == "none"
    assert result.message == "No sequence or UniProt ID provided"


def test_prediction_report_counts_the_plddt_bands_and_the_engine_note():
    result = sp.PredictionResult(engine="esmfold", plddt_scores=[95.4, 62.0, 33.5, 88.0],
                                 sequence="MGSH", num_residues=4, confidence=69.725,
                                 output_file="/tmp/model.pdb", message="done")
    assert sp.format_prediction_report(result).splitlines() == [
        "=== Structure Prediction Report ===",
        "Engine: esmfold",
        "Residues: 4",
        "Confidence (pLDDT): 69.7%",
        "Output: /tmp/model.pdb",
        "Confident (>90): 1 | Good (>70-90): 1 | Low (<=70): 2",
        "Note: done",
    ]


def test_the_plddt_band_labels_match_the_arithmetic_at_the_boundaries():
    """The labels used to read 'Good (70-90)' and 'Low (<70)' while the code
    counted 70.0 as Low and 90.0 as Good, so a reader reconciling the three
    numbers with the labels got the boundary wrong."""
    result = sp.PredictionResult(engine="esmfold", plddt_scores=[90.0, 70.0, 90.1, 69.9])
    line = sp.format_prediction_report(result).splitlines()[5]
    assert line == "Confident (>90): 1 | Good (>70-90): 1 | Low (<=70): 2"


def test_prediction_report_omits_the_band_line_without_scores():
    text = sp.format_prediction_report(sp.PredictionResult(engine="none", message="no input"))
    assert text.splitlines() == [
        "=== Structure Prediction Report ===",
        "Engine: none",
        "Residues: 0",
        "Confidence (pLDDT): 0.0%",
        "Output: ",
        "Note: no input",
    ]
    assert "Confident" not in text


def test_prediction_report_has_no_note_line_when_the_engine_said_nothing():
    silent = sp.PredictionResult(engine="alphafold", plddt_scores=[91.0], num_residues=1,
                                 confidence=91.0, output_file="/tmp/af.pdb")
    assert sp.format_prediction_report(silent).splitlines() == [
        "=== Structure Prediction Report ===",
        "Engine: alphafold",
        "Residues: 1",
        "Confidence (pLDDT): 91.0%",
        "Output: /tmp/af.pdb",
        "Confident (>90): 1 | Good (>70-90): 0 | Low (<=70): 0",
    ]
