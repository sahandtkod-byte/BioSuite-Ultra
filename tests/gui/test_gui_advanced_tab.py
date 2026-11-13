"""The Single-Cell, Structure, CRISPR, PopGen and ML pages of one real app.

Each handler reads its entries, wraps one ``biosuite.core`` function
(``single_cell``, ``structure``, ``crispr``, ``popgen``, ``bio_ml``) and writes that
module's own report into the tab's box, so every expectation here is recomputed by
calling the same core function on the same typed input.  Two of them cannot run for
real on this machine: scanpy is not installed, and a PDB ID would be downloaded
from the PDB — those two are reached through recorders, which is also what proves
the entry is what actually gets sent.
"""
import textwrap

import numpy as np
import pandas as pd
import pytest

from biosuite.core import single_cell as core_sc
from biosuite.core import structure as core_structure
from biosuite.core.bio_ml import (
    compute_roc_curve,
    format_ml_report,
    select_features,
    train_random_forest,
    train_random_forest_regressor,
    train_svm,
)
from biosuite.core.crispr import PAM_PATTERNS, design_guides, format_crispr_report
from biosuite.core.popgen import format_popgen_report, pca_genotypes
from biosuite.core.popgen import full_analysis as popgen_analysis
from biosuite.core.single_cell import format_sc_report, load_count_matrix
from biosuite.core.single_cell import run_full_pipeline as sc_pipeline
from biosuite.core.structure import StructureInfo, format_structure_report
from biosuite.core.structure import full_analysis as structure_analysis

CRISPR_TARGET = ("ATGCGATCGATCGATCGAATTGGCCGGCATCGATCGATCGATCGATTGGCCCC"
                 "ATCGATCGATCGATCGAATTGGCC")
GENOTYPES = [[0, 1, 2, 1, 0], [1, 1, 2, 0, 2], [0, 0, 1, 1, 2],
             [2, 1, 0, 1, 0], [1, 2, 2, 0, 1]]
MATRIX_TEXT = "\n".join(",".join(str(v) for v in row) for row in GENOTYPES)
FEATURE_NAMES = ["expression_a", "expression_b", "ratio"]

PDB_TEXT = textwrap.dedent("""\
    HEADER    TEST STRUCTURE                            01-JAN-00   0000
    TITLE     MINIMAL TEST
    ATOM      1  N   ALA A   1     -10.000   0.000   0.000  1.00 20.00           N
    ATOM      2  CA  ALA A   1      -9.000   0.500   0.000  1.00 20.00           C
    ATOM      3  C   ALA A   1      -8.200  -0.500   0.800  1.00 20.00           C
    ATOM      4  O   ALA A   1      -8.200  -1.700   0.700  1.00 20.00           O
    ATOM      5  N   GLY A   2      -7.400   0.100   1.700  1.00 20.00           N
    ATOM      6  CA  GLY A   2      -6.400  -0.400   2.500  1.00 20.00           C
    ATOM      7  C   GLY A   2      -5.200   0.200   2.400  1.00 20.00           C
    ATOM      8  O   GLY A   2      -5.000   1.200   3.000  1.00 20.00           O
    ATOM      9  N   ALA A   3      -4.400  -0.300   1.500  1.00 20.00           N
    ATOM     10  CA  ALA A   3      -3.200   0.300   1.400  1.00 20.00           C
    ATOM     11  C   ALA A   3      -2.100  -0.300   2.200  1.00 20.00           C
    ATOM     12  O   ALA A   3      -1.000   0.200   2.200  1.00 20.00           O
    END
    """)


@pytest.fixture
def counts_csv(tmp_path):
    """A tiny count matrix — the file the single-cell tab is pointed at."""
    rng = np.random.default_rng(5)
    counts = rng.poisson(4, size=(6, 8))
    path = tmp_path / "counts.csv"
    header = ",".join(f"gene{i}" for i in range(counts.shape[1]))
    body = "\n".join(",".join(str(int(v)) for v in row) for row in counts)
    path.write_text(f"{header}\n{body}\n", encoding="utf-8")
    return path


@pytest.fixture
def pdb_file(tmp_path):
    path = tmp_path / "mini.pdb"
    path.write_text(PDB_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def expression_csv(tmp_path):
    """Numeric features plus a text column and a class label, as the ML tab reads it."""
    rng = np.random.default_rng(11)
    n = 24
    separated = np.where(np.arange(n) < 12, 0.0, 2.0) + rng.normal(0.0, 0.6, n)
    noise = rng.normal(1.0, 1.0, n)
    frame = pd.DataFrame({
        "sample": [f"s{i}" for i in range(n)],
        "expression_a": np.round(separated, 6),
        "expression_b": np.round(noise, 6),
        "ratio": np.round(separated * noise - 0.5, 6),
        "note": ["batch1" if i % 2 else "batch2" for i in range(n)],
        "class": ["case"] * (n // 2) + ["control"] * (n // 2),
    })
    path = tmp_path / "expression.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def sc_loaded(ui, monkeypatch, counts_csv):
    """The matrix loader fed from memory, with the real pipeline still running.

    ``load_count_matrix`` cannot return a matrix on this machine at all — scanpy is
    missing — so only the loader is canned; ``run_full_pipeline`` and
    ``format_sc_report`` stay the module's own code, and their kwargs are recorded.
    """
    real_pipeline = core_sc.run_full_pipeline
    seen = {}
    matrix = np.zeros((4, 3))

    def fake_loader(path):
        seen["path"] = path
        return matrix, None

    def spy(adata, **kwargs):
        seen["kwargs"] = kwargs
        return real_pipeline(adata, **kwargs)

    monkeypatch.setattr(core_sc, "load_count_matrix", fake_loader)
    monkeypatch.setattr(core_sc, "run_full_pipeline", spy)
    ui.set("sc_path", str(counts_csv))
    ui.clear("sc_result")
    return seen


def _designated_matrix(path, label="class"):
    """The tab's own view of the file: its numeric features and the label column."""
    frame = pd.read_csv(str(path))
    features = frame.drop(columns=[label]).select_dtypes(include=[np.number]).values
    return features, frame[label].values


@pytest.fixture(autouse=True)
def _default_entries(ui):
    """The app is session-scoped, so a value typed by one test would leak into the next."""
    ui.set("sc_mingenes", "200")
    ui.set("sc_maxmito", "20")
    ui.set("ml_model", "RF")
    ui.set("ml_label", "class")
    ui.app.crispr_pam_combo.set("SpCas9")


# ─── Single cell ────────────────────────────────────────────────────────────

def test_single_cell_without_a_file_asks_for_one(ui):
    ui.clear("sc_path")
    ui.clear("sc_result")

    ui.app._run_sc()

    assert ui.messages == [("warning", "No file", "Select a count matrix file.")]
    assert ui.text("sc_result").strip() == ""


def test_single_cell_shows_the_loaders_own_refusal(ui, counts_csv):
    """scanpy is not installed here, so the core's loader answers with its reason;
    the box must stay empty and the dialog must repeat that reason verbatim."""
    matrix, err = load_count_matrix(str(counts_csv))
    assert matrix is None and err, "the loader is expected to refuse without scanpy"
    assert "scanpy" in err, f"the refusal must name the missing package: {err}"
    ui.set("sc_path", str(counts_csv))
    ui.clear("sc_result")
    before = ui.status()

    ui.app._run_sc()

    assert ui.messages_of("error") == [err]
    assert ui.text("sc_result").strip() == ""
    assert ui.status() == before, "a refused matrix must not rewrite the status line"


def test_the_qc_entries_reach_the_pipeline(ui, sc_loaded, counts_csv):
    assert not core_sc.HAS_SCANPY, "scanpy is absent here, which is why the pipeline answers with a stub report"
    ui.set("sc_mingenes", "150")
    ui.set("sc_maxmito", "5.5")

    ui.app._run_sc()

    assert sc_loaded["path"] == str(counts_csv)
    assert sc_loaded["kwargs"] == {"min_genes": 150, "max_pct_mito": 5.5}
    _adata, expected = sc_pipeline(np.zeros((4, 3)), min_genes=150, max_pct_mito=5.5)
    assert ui.text("sc_result").strip() == format_sc_report(expected).strip()
    assert f"Cells: {expected.num_cells}" in ui.text("sc_result")
    assert f"Clusters: {expected.num_clusters}" in ui.text("sc_result")
    assert ui.status() == (f"scRNA-seq: {expected.num_cells} cells, "
                           f"{expected.num_clusters} clusters")
    assert ui.messages == []


def test_cleared_qc_entries_fall_back_to_the_defaults(ui, sc_loaded):
    ui.set("sc_mingenes", "150")
    ui.set("sc_maxmito", "5.5")
    ui.clear("sc_mingenes")
    ui.clear("sc_maxmito")

    ui.app._run_sc()

    assert sc_loaded["kwargs"] == {"min_genes": 200, "max_pct_mito": 20.0}


def test_a_non_numeric_gene_threshold_is_surfaced(ui, monkeypatch, counts_csv):
    calls = []

    def loader(path):
        calls.append(path)
        return np.zeros((4, 3)), None

    monkeypatch.setattr(core_sc, "load_count_matrix", loader)
    ui.set("sc_path", str(counts_csv))
    ui.set("sc_mingenes", "twenty")
    ui.set("sc_maxmito", "20")
    ui.set("sc_result", "KEEP THIS REPORT")

    ui.app._run_sc()

    assert ui.messages_of("error") == [
        "Min genes must be a whole number (got 'twenty')."]
    assert calls == [], "a typo must be refused before the matrix is loaded"
    assert ui.text("sc_result").strip() == "KEEP THIS REPORT"


def test_a_gene_threshold_below_zero_is_refused(ui, monkeypatch, counts_csv):
    calls = []

    def loader(path):
        calls.append(path)
        return np.zeros((4, 3)), None

    monkeypatch.setattr(core_sc, "load_count_matrix", loader)
    ui.set("sc_path", str(counts_csv))
    ui.set("sc_mingenes", "-5")
    ui.set("sc_result", "KEEP THIS REPORT")

    ui.app._run_sc()

    assert ui.messages_of("error") == ["Min genes must be zero or greater."]
    assert calls == []
    assert ui.text("sc_result").strip() == "KEEP THIS REPORT"


def test_a_non_numeric_mitochondrial_percentage_is_named(ui, monkeypatch, counts_csv):
    calls = []
    monkeypatch.setattr(core_sc, "load_count_matrix",
                        lambda path: calls.append(path) or (np.zeros((4, 3)), None))
    ui.set("sc_path", str(counts_csv))
    ui.set("sc_maxmito", "lots")
    ui.set("sc_result", "KEEP THIS REPORT")

    ui.app._run_sc()

    assert ui.messages_of("error") == [
        "Max mito% must be a number (got 'lots')."]
    assert calls == []
    assert ui.text("sc_result").strip() == "KEEP THIS REPORT"


def test_a_pipeline_failure_is_reported_without_touching_the_old_report(ui, monkeypatch, counts_csv):
    def broken(*args, **kwargs):
        raise RuntimeError("the scanpy pipeline exploded")

    monkeypatch.setattr(core_sc, "load_count_matrix", lambda path: (np.zeros((4, 3)), None))
    monkeypatch.setattr(core_sc, "run_full_pipeline", broken)
    ui.set("sc_path", str(counts_csv))
    ui.set("sc_result", "KEEP THIS REPORT")

    ui.app._run_sc()

    assert ui.messages_of("error") == ["the scanpy pipeline exploded"]
    assert ui.text("sc_result").strip() == "KEEP THIS REPORT"


@pytest.mark.parametrize("value", ["nan", "101", "-1"])
def test_a_mitochondrial_percentage_outside_0_100_is_refused(ui, monkeypatch, counts_csv, value):
    calls = []
    monkeypatch.setattr(core_sc, "load_count_matrix",
                        lambda path: calls.append(path) or (np.zeros((4, 3)), None))
    ui.set("sc_path", str(counts_csv))
    ui.set("sc_maxmito", value)
    ui.set("sc_result", "KEEP THIS REPORT")

    ui.app._run_sc()

    assert ui.messages_of("error") == [
        "Max mito% must be a finite percentage between 0 and 100."]
    assert calls == []
    assert ui.text("sc_result").strip() == "KEEP THIS REPORT"


# ─── Protein structure ──────────────────────────────────────────────────────

def test_structure_without_an_input_asks_for_one(ui):
    ui.clear("pdb_path")
    ui.clear("struct_result")

    ui.app._run_structure()

    assert ui.messages == [("warning", "No input", "Enter a PDB ID or file path.")]
    assert ui.text("struct_result").strip() == ""


def test_a_local_pdb_file_is_measured_by_the_core(ui, pdb_file):
    expected = structure_analysis(filepath=str(pdb_file))
    assert expected.num_residues == 3 and expected.chains == ["A"], "three residues on one chain"
    ui.set("pdb_path", str(pdb_file))
    ui.clear("struct_result")

    ui.app._run_structure()

    assert ui.text("struct_result").strip() == format_structure_report(expected).strip()
    assert f"Residues: {expected.num_residues}" in ui.text("struct_result")
    assert f"Atoms: {expected.num_atoms:,}" in ui.text("struct_result")
    assert ui.status() == f"Structure: {expected.num_residues} residues"
    assert ui.messages == []


def test_a_pdb_id_is_sent_as_an_id_and_rendered_as_given(ui, monkeypatch):
    """``full_analysis(pdb_id=…)`` downloads from the PDB, so it is replaced by a
    recorder: nothing leaves the process, and the test still sees what was asked."""
    calls = []
    canned = StructureInfo(name="1UBQ", num_atoms=118, num_residues=76, num_chains=1,
                           chains=["A"], resolution=1.8, method="X-RAY",
                           molecule_type="PROTEIN", surface_area=4200.5,
                           secondary_structure={"H": 40, "E": 10, "C": 26},
                           ramachandran={"phi": [-60.0], "psi": [-40.0]})

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return canned

    monkeypatch.setattr(core_structure, "full_analysis", recorder)
    ui.set("pdb_path", "  1UBQ  ")
    ui.clear("struct_result")

    ui.app._run_structure()

    assert calls == [((), {"pdb_id": "1UBQ"})], "the entry goes out as an ID, and as nothing else"
    assert ui.text("struct_result").strip() == format_structure_report(canned).strip()
    assert ui.status() == f"Structure: {canned.num_residues} residues"
    assert ui.messages == []


@pytest.mark.parametrize("name", ["absent.pdb", "absent"])
def test_a_structure_path_that_does_not_exist_is_not_sent_to_the_pdb(
        ui, monkeypatch, tmp_path, name):
    """A typo in a local path used to travel to the PDB as if it were an ID, which
    is a network request for a structure that was already on disk."""
    calls = []
    monkeypatch.setattr(core_structure, "full_analysis",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    missing = tmp_path / name
    ui.set("pdb_path", str(missing))
    ui.set("struct_result", "KEEP THIS REPORT")

    ui.app._run_structure()

    assert calls == [], "a path must never be treated as a PDB ID"
    assert ui.messages_of("error") == [f"File not found: {missing}"]
    assert ui.text("struct_result").strip() == "KEEP THIS REPORT"


def test_a_structure_failure_is_reported_instead_of_a_stale_report(ui, monkeypatch, pdb_file):
    ui.set("pdb_path", str(pdb_file))
    ui.app._run_structure()
    assert ui.messages == []

    def boom(*args, **kwargs):
        raise RuntimeError("PDB is unreachable")

    monkeypatch.setattr(core_structure, "full_analysis", boom)
    ui.set("pdb_path", "1UBQ")
    ui.set("struct_result", "KEEP THIS REPORT")

    ui.app._run_structure()

    assert ui.messages_of("error") == ["PDB is unreachable"]
    assert ui.text("struct_result").strip() == "KEEP THIS REPORT"


# ─── CRISPR ─────────────────────────────────────────────────────────────────

def test_crispr_without_a_sequence_asks_for_one(ui):
    ui.clear("crispr_seq")
    ui.clear("crispr_result")

    ui.app._run_crispr()

    assert ui.messages == [("warning", "No sequence", "Enter a target DNA sequence.")]
    assert ui.text("crispr_result").strip() == ""


def test_the_guide_report_matches_the_core_designer(ui):
    expected = design_guides(CRISPR_TARGET, pam_type="SpCas9")
    assert expected.num_guides == len(expected.guides) > 0
    ui.set("crispr_seq", CRISPR_TARGET)
    ui.app.crispr_pam_combo.set("SpCas9")
    ui.clear("crispr_result")

    ui.app._run_crispr()

    assert ui.text("crispr_result").strip() == format_crispr_report(expected).strip()
    assert f"Guides found: {expected.num_guides}" in ui.text("crispr_result")
    assert ui.status() == f"CRISPR: {expected.num_guides} guides found"
    assert ui.messages == []


@pytest.mark.parametrize("pam", ["SpCas9", "SaCas9", "Cas12a"])
def test_the_pam_choice_reaches_the_designer(ui, pam):
    chosen = design_guides(CRISPR_TARGET, pam_type=pam)
    spcas9 = design_guides(CRISPR_TARGET, pam_type="SpCas9")
    assert spcas9.num_guides > 0, "the target must carry SpCas9 sites for the comparison to mean anything"
    ui.set("crispr_seq", CRISPR_TARGET)
    ui.app.crispr_pam_combo.set(pam)
    ui.clear("crispr_result")

    ui.app._run_crispr()

    assert ui.text("crispr_result").strip() == format_crispr_report(chosen).strip()
    assert f"Engine: {chosen.engine}" in ui.text("crispr_result")
    if pam != "SpCas9":
        assert chosen.num_guides != spcas9.num_guides, f"{pam} found the same guides as SpCas9"
    if chosen.num_guides == 0:
        assert "Guides found: 0" in ui.text("crispr_result")


def test_the_pam_choice_cannot_be_typed_over(ui):
    assert ui.app.crispr_pam_combo.cget("state") == "readonly"


def test_the_picker_offers_every_core_pam_type(ui):
    offered = ui.app.crispr_pam_combo.cget("values")
    assert set(offered) == set(PAM_PATTERNS), \
        "the picker must offer every enzyme the core designs for"


def test_the_high_fidelity_variant_designs_the_same_ngg_guides(ui):
    expected = design_guides(CRISPR_TARGET, pam_type="SpCas9-HF1")
    spcas9 = design_guides(CRISPR_TARGET, pam_type="SpCas9")
    assert expected.num_guides == spcas9.num_guides > 0, "HF1 shares SpCas9's NGG PAM"
    ui.set("crispr_seq", CRISPR_TARGET)
    ui.app.crispr_pam_combo.set("SpCas9-HF1")
    ui.clear("crispr_result")

    ui.app._run_crispr()

    assert ui.text("crispr_result").strip() == format_crispr_report(expected).strip()
    assert ui.status() == f"CRISPR: {expected.num_guides} guides found"
    assert ui.messages == []


def test_an_unknown_pam_choice_is_reported_not_redirected(ui):
    """``set()`` can hand the widget a value outside its list; substituting SpCas9
    would design guides the user did not ask for, so the core's refusal is shown."""
    try:
        design_guides(CRISPR_TARGET, pam_type="SpCas9x")
    except ValueError as exc:
        expected = str(exc)
    else:
        raise AssertionError("design_guides accepted an unknown PAM type")
    ui.set("crispr_seq", CRISPR_TARGET)
    ui.app.crispr_pam_combo.set("SpCas9x")
    ui.set("crispr_result", "KEEP THIS REPORT")

    ui.app._run_crispr()

    assert ui.messages_of("error") == [expected]
    assert ui.text("crispr_result").strip() == "KEEP THIS REPORT"


def test_a_crispr_failure_is_surfaced(ui, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no PAM pattern is configured")

    monkeypatch.setattr("biosuite.core.crispr.design_guides", broken)
    ui.set("crispr_seq", CRISPR_TARGET)
    ui.set("crispr_result", "KEEP THIS REPORT")

    ui.app._run_crispr()

    assert ui.messages_of("error") == ["no PAM pattern is configured"]
    assert ui.text("crispr_result").strip() == "KEEP THIS REPORT"


# ─── Population genetics ────────────────────────────────────────────────────

def test_popgen_without_data_asks_for_some(ui):
    ui.clear("popgen_input")
    ui.clear("popgen_result")

    ui.app._run_popgen()

    assert ui.messages == [("warning", "No data", "Enter genotype matrix rows.")]
    assert ui.text("popgen_result").strip() == ""


def test_the_popgen_report_matches_the_core_analysis(ui):
    expected = popgen_analysis(np.array(GENOTYPES))
    assert expected.num_sites == 5 and expected.ld, "the matrix must give the core something to measure"
    ui.set("popgen_input", MATRIX_TEXT)
    ui.clear("popgen_result")

    ui.app._run_popgen()

    assert ui.text("popgen_result").strip() == format_popgen_report(expected).strip()
    assert f"Sites: {expected.num_sites}" in ui.text("popgen_result")
    assert ui.status() == f"PopGen: {expected.num_sites} sites analyzed"
    assert ui.messages == []


def test_blank_lines_and_padding_are_not_genotypes(ui):
    """The matrix is typed by hand, so stray blank lines must not become samples."""
    expected = popgen_analysis(np.array(GENOTYPES))
    typed = "\n\n" + "\n \n".join("  " + ",".join(str(v) for v in row) + "  "
                                  for row in GENOTYPES) + "\n \n"
    assert len([line for line in typed.split("\n") if line.strip()]) == len(GENOTYPES)
    ui.set("popgen_input", typed)
    ui.clear("popgen_result")

    ui.app._run_popgen()

    assert ui.text("popgen_result").strip() == format_popgen_report(expected).strip()
    assert f"Sites: {expected.num_sites}" in ui.text("popgen_result")
    assert ui.messages == []


def test_a_non_numeric_genotype_is_reported(ui, monkeypatch):
    calls = []
    monkeypatch.setattr("biosuite.core.popgen.full_analysis",
                        lambda matrix: calls.append(matrix))
    ui.set("popgen_input", "0,1,2\n1,two,0")
    ui.set("popgen_result", "KEEP THIS REPORT")

    ui.app._run_popgen()

    assert ui.messages_of("error") == [
        "Genotype values must be whole numbers (line 2: 'two')."]
    assert calls == []
    assert ui.text("popgen_result").strip() == "KEEP THIS REPORT"


def test_ragged_genotype_rows_are_refused(ui, monkeypatch):
    """numpy's own message for ragged rows is an implementation detail, and the
    core would otherwise index a matrix the user did not type."""
    calls = []
    monkeypatch.setattr("biosuite.core.popgen.full_analysis",
                        lambda matrix: calls.append(matrix))
    ui.set("popgen_input", "0,1,2\n1,2\n")
    ui.set("popgen_result", "KEEP THIS REPORT")

    ui.app._run_popgen()

    assert ui.messages_of("error") == [
        "Every genotype row must have the same number of sites "
        "(line 2 has 2 values, line 1 has 3)."]
    assert calls == []
    assert ui.text("popgen_result").strip() == "KEEP THIS REPORT"


def test_a_popgen_failure_is_reported_without_touching_the_old_report(ui, monkeypatch):
    def broken(matrix):
        raise RuntimeError("no site could be analysed")

    monkeypatch.setattr("biosuite.core.popgen.full_analysis", broken)
    ui.set("popgen_input", MATRIX_TEXT)
    ui.set("popgen_result", "KEEP THIS REPORT")

    ui.app._run_popgen()

    assert ui.messages_of("error") == ["no site could be analysed"]
    assert ui.text("popgen_result").strip() == "KEEP THIS REPORT"


def test_a_genotype_outside_0_1_2_is_refused(ui, monkeypatch):
    """A 3 is neither ``AA`` nor ``Aa`` nor ``aa``, so the HWE table silently
    counts it as none of them and tests different allele frequencies."""
    calls = []
    monkeypatch.setattr("biosuite.core.popgen.full_analysis",
                        lambda matrix: calls.append(matrix))
    ui.set("popgen_input", "0,1,2\n1,3,0\n")
    ui.set("popgen_result", "KEEP THIS REPORT")

    ui.app._run_popgen()

    assert ui.messages_of("error") == [
        "Genotype values must be 0, 1 or 2 (line 2 has 3)."]
    assert calls == []
    assert ui.text("popgen_result").strip() == "KEEP THIS REPORT"


# ─── Population genetics PCA ────────────────────────────────────────────────

def _pca_text(coords, ratios):
    lines = ["PCA coordinates:\n"]
    for i, row in enumerate(coords, 1):
        values = ", ".join(f"PC{j+1}={v:.4f}" for j, v in enumerate(row))
        lines.append(f"Sample_{i}: {values}\n")
    lines.append("\nExplained variance ratio: "
                 + ", ".join(f"{r:.4f}" for r in ratios) + "\n")
    return "".join(lines).rstrip("\n")


def test_pca_without_data_asks_for_some(ui):
    ui.clear("popgen_input")
    ui.clear("popgen_result")

    ui.app._run_pca()

    assert ui.messages_of("warning") == ["Enter genotype matrix rows."]
    assert ui.text("popgen_result").strip() == ""


def test_the_pca_report_matches_the_core_projection(ui):
    coords, ratios = pca_genotypes(np.array(GENOTYPES))
    # Positive control: the core really projected five samples onto two axes.
    assert coords.shape == (5, 2)
    assert 0.0 < float(ratios.sum()) <= 1.0 + 1e-9
    ui.set("popgen_input", MATRIX_TEXT)
    ui.clear("popgen_result")

    ui.app._run_pca()

    assert ui.text("popgen_result").rstrip("\n") == _pca_text(coords, ratios)
    assert ui.status() == "PCA: 5 samples, 2 components"
    assert ui.messages == []


def test_pca_needs_two_samples(ui):
    ui.set("popgen_input", "0,1\n")
    ui.set("popgen_result", "KEEP THIS REPORT")

    ui.app._run_pca()

    assert ui.messages_of("error") == ["PCA needs at least 2 samples and 2 sites."]
    assert ui.text("popgen_result").strip() == "KEEP THIS REPORT"


def test_pca_needs_two_sites(ui):
    ui.set("popgen_input", "0\n1\n")
    ui.set("popgen_result", "KEEP THIS REPORT")

    ui.app._run_pca()

    assert ui.messages_of("error") == ["PCA needs at least 2 samples and 2 sites."]
    assert ui.text("popgen_result").strip() == "KEEP THIS REPORT"


def test_pca_refuses_bad_genotypes_before_projecting(ui, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("PCA must not run on invalid genotypes")
    monkeypatch.setattr("biosuite.core.popgen.pca_genotypes", never)
    ui.set("popgen_input", "0,two\n1,0")
    ui.set("popgen_result", "KEEP THIS REPORT")

    ui.app._run_pca()

    assert any("whole numbers" in m for m in ui.messages_of("error"))
    assert ui.text("popgen_result").strip() == "KEEP THIS REPORT"


def test_a_pca_failure_is_reported_without_touching_the_old_report(ui, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("projection engine down")
    monkeypatch.setattr("biosuite.core.popgen.pca_genotypes", broken)
    ui.set("popgen_input", MATRIX_TEXT)
    ui.set("popgen_result", "KEEP THIS REPORT")

    ui.app._run_pca()

    assert ui.messages_of("error") == ["projection engine down"]
    assert ui.text("popgen_result").strip() == "KEEP THIS REPORT"


# ─── Machine learning ───────────────────────────────────────────────────────

def test_ml_without_a_file_asks_for_one(ui):
    ui.clear("ml_path")
    ui.clear("ml_result")

    ui.app._run_ml()

    assert ui.messages == [("warning", "No file", "Select a CSV data file.")]
    assert ui.text("ml_result").strip() == ""


def test_the_random_forest_report_matches_the_core(ui, expression_csv):
    features, labels = _designated_matrix(expression_csv)
    assert features.shape == (24, 3), "the text columns are dropped, three features remain"
    expected = train_random_forest(features, labels)
    assert expected.model_type == "random_forest"
    ui.set("ml_path", str(expression_csv))
    ui.set("ml_model", "RF")
    ui.set("ml_label", "class")
    ui.clear("ml_result")

    ui.app._run_ml()

    assert ui.text("ml_result").strip() == format_ml_report(expected).strip()
    assert f"Model: {expected.model_type}" in ui.text("ml_result")
    assert f"Accuracy: {expected.accuracy:.4f}" in ui.text("ml_result")
    assert ui.status() == f"ML: {expected.model_type}, accuracy={expected.accuracy}"
    assert ui.messages == []


def test_the_model_entry_switches_to_the_svm(ui, expression_csv):
    features, labels = _designated_matrix(expression_csv)
    expected = train_svm(features, labels)
    assert expected.model_type == "svm"
    assert expected.model_type != train_random_forest(features, labels).model_type
    ui.set("ml_path", str(expression_csv))
    ui.set("ml_model", " svm ")
    ui.set("ml_label", "class")
    ui.clear("ml_result")

    ui.app._run_ml()

    assert ui.text("ml_result").strip() == format_ml_report(expected).strip()
    assert ui.status() == f"ML: {expected.model_type}, accuracy={expected.accuracy}"
    assert ui.messages == []


def test_cleared_model_and_label_entries_fall_back_to_rf_on_class(ui, expression_csv):
    features, labels = _designated_matrix(expression_csv)
    expected = train_random_forest(features, labels)
    ui.set("ml_path", str(expression_csv))
    ui.set("ml_model", "svm")
    ui.set("ml_label", "class")
    ui.clear("ml_model")
    ui.clear("ml_label")

    ui.app._run_ml()

    assert ui.text("ml_result").strip() == format_ml_report(expected).strip()
    assert f"Model: {expected.model_type}" in ui.text("ml_result")
    assert "Model: svm" not in ui.text("ml_result")


def test_the_label_entry_selects_another_column(ui, tmp_path):
    outcomes = ["case", "control", "control", "case", "case", "control", "case", "control"]
    path = tmp_path / "two_labels.csv"
    path.write_text("x,y,quality,outcome\n"
                    + "\n".join(f"{i},{i * 2.5},{i % 3},{o}"
                                for i, o in enumerate(outcomes)) + "\n",
                    encoding="utf-8")
    features, labels = _designated_matrix(path, label="outcome")
    assert labels.tolist() == outcomes, "the typed column is the one that gets modelled"
    expected = train_random_forest(features, labels)
    ui.set("ml_path", str(path))
    ui.set("ml_model", "RF")
    ui.set("ml_label", "outcome")
    ui.clear("ml_result")

    ui.app._run_ml()

    assert ui.text("ml_result").strip() == format_ml_report(expected).strip()
    assert f"Model: {expected.model_type}" in ui.text("ml_result")
    assert ui.messages == []


def test_a_missing_label_column_is_named(ui, expression_csv):
    ui.set("ml_path", str(expression_csv))
    ui.set("ml_label", "absent_column")
    ui.set("ml_model", "RF")
    ui.set("ml_result", "KEEP THIS REPORT")

    ui.app._run_ml()

    assert ui.messages_of("error") == ["Column 'absent_column' not found."]
    assert ui.text("ml_result").strip() == "KEEP THIS REPORT"


def test_an_unknown_model_is_refused_before_the_file_is_read(ui, tmp_path):
    """``else: train_random_forest`` used to make every unrecognised name an RF
    run; the missing file proves the check comes before any reading."""
    missing = tmp_path / "absent.csv"
    ui.set("ml_path", str(missing))
    ui.set("ml_model", "XGB")
    ui.set("ml_result", "KEEP THIS REPORT")

    ui.app._run_ml()

    assert ui.messages_of("error") == ["Model must be RF, SVM or RFR (got 'XGB')."]
    assert ui.text("ml_result").strip() == "KEEP THIS REPORT"


def test_a_data_file_that_is_not_there_is_reported(ui, tmp_path):
    missing = tmp_path / "absent.csv"
    assert not missing.exists()
    ui.set("ml_path", str(missing))
    ui.set("ml_label", "class")
    ui.set("ml_model", "RF")
    ui.set("ml_result", "KEEP THIS REPORT")

    ui.app._run_ml()

    errors = ui.messages_of("error")
    assert len(errors) == 1 and "absent.csv" in errors[0], errors
    assert ui.text("ml_result").strip() == "KEEP THIS REPORT"


def test_a_csv_without_numeric_features_fails_the_way_the_core_does(ui, tmp_path):
    """Nothing survives ``select_dtypes(include=[np.number])``, so the tab must
    surface the model's own complaint instead of scoring an empty matrix."""
    path = tmp_path / "notes.csv"
    path.write_text("text,other,class\n"
                    + "\n".join(f"alpha{i},beta{i},{'yes' if i % 2 else 'no'}"
                                for i in range(8)) + "\n", encoding="utf-8")
    features, labels = _designated_matrix(path)
    assert features.shape[1] == 0
    with pytest.raises(ValueError) as raised:
        train_random_forest(features, labels)

    ui.set("ml_path", str(path))
    ui.set("ml_label", "class")
    ui.set("ml_model", "RF")
    ui.set("ml_result", "KEEP THIS REPORT")

    ui.app._run_ml()

    assert ui.messages_of("error") == [str(raised.value)]
    assert ui.text("ml_result").strip() == "KEEP THIS REPORT"


@pytest.fixture
def regression_csv(tmp_path):
    """Numeric features with a continuous target, as the regressor reads it."""
    rng = np.random.default_rng(7)
    n = 30
    x = rng.normal(0.0, 1.0, n)
    frame = pd.DataFrame({
        "feature_a": np.round(x, 6),
        "feature_b": np.round(rng.normal(0.0, 1.0, n), 6),
        "target": np.round(2.0 * x + rng.normal(0.0, 0.1, n), 6),
    })
    path = tmp_path / "regression.csv"
    frame.to_csv(path, index=False)
    return path


def test_the_regressor_report_matches_the_core(ui, regression_csv):
    features, labels = _designated_matrix(regression_csv, label="target")
    expected = train_random_forest_regressor(features, labels)
    assert expected.model_type == "rf_regressor"
    ui.set("ml_path", str(regression_csv))
    ui.set("ml_model", "rfr")
    ui.set("ml_label", "target")
    ui.clear("ml_result")

    ui.app._run_ml()

    assert ui.text("ml_result").strip() == format_ml_report(expected).strip()
    assert ui.status() == f"ML: {expected.model_type}, accuracy={expected.accuracy}"
    assert ui.messages == []


def test_the_roc_report_matches_the_core(ui, expression_csv):
    features, labels = _designated_matrix(expression_csv)
    trained = train_random_forest(features, labels)
    assert trained.probabilities is not None, "the fixture must be binary"
    expected = compute_roc_curve(trained.eval_labels, trained.probabilities)
    ui.set("ml_path", str(expression_csv))
    ui.set("ml_label", "class")
    ui.clear("ml_result")

    ui.app._run_ml_roc()

    report = ui.text("ml_result")
    assert f"AUC: {expected['auc']:.4f}" in report
    assert f"Curve points: {len(expected['fpr'])}" in report
    assert ui.status() == f"ML ROC: AUC={expected['auc']}"
    assert ui.messages == []


def test_the_roc_button_refuses_a_multiclass_label(ui, tmp_path):
    path = tmp_path / "three_classes.csv"
    rows = ["x,y,class"] + [f"{i},{i * 2},{['a', 'b', 'c'][i % 3]}" for i in range(12)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ui.set("ml_path", str(path))
    ui.set("ml_label", "class")
    ui.clear("ml_result")

    ui.app._run_ml_roc()

    assert ui.messages_of("error") == ["ROC needs a binary label column (exactly two classes)."]
    assert ui.text("ml_result").strip() == ""


def test_the_roc_button_names_a_missing_engine(ui, expression_csv, monkeypatch):
    monkeypatch.setattr("biosuite.core.bio_ml.HAS_SKLEARN", False)
    ui.set("ml_path", str(expression_csv))
    ui.set("ml_label", "class")
    ui.clear("ml_result")

    ui.app._run_ml_roc()

    assert ui.messages_of("error") == ["scikit-learn not installed"]
    assert ui.text("ml_result").strip() == ""


def test_the_feature_ranking_matches_the_core(ui, expression_csv):
    features, labels = _designated_matrix(expression_csv)
    indices, scores = select_features(features, labels)
    assert indices, "the fixture must produce a ranking"
    ui.set("ml_path", str(expression_csv))
    ui.set("ml_label", "class")
    ui.clear("ml_result")

    ui.app._run_ml_features()

    report = ui.text("ml_result")
    for rank, idx in enumerate(indices, 1):
        assert f"{rank}. {FEATURE_NAMES[idx]}: {scores[idx]:.4f}" in report
    assert ui.status() == f"ML: {len(indices)} features ranked"
    assert ui.messages == []


def test_the_feature_button_names_a_missing_engine(ui, expression_csv, monkeypatch):
    monkeypatch.setattr("biosuite.core.bio_ml.HAS_SKLEARN", False)
    ui.set("ml_path", str(expression_csv))
    ui.set("ml_label", "class")
    ui.clear("ml_result")

    ui.app._run_ml_features()

    assert ui.messages_of("error") == [
        "Feature selection needs scikit-learn for the importance ranking."]
    assert ui.text("ml_result").strip() == ""


def test_the_roc_button_without_a_file_asks_for_one(ui):
    ui.clear("ml_path")
    ui.clear("ml_result")

    ui.app._run_ml_roc()

    assert ui.messages == [("warning", "No file", "Select a CSV data file.")]
    assert ui.text("ml_result").strip() == ""


def test_the_feature_button_without_a_file_asks_for_one(ui):
    ui.clear("ml_path")
    ui.clear("ml_result")

    ui.app._run_ml_features()

    assert ui.messages == [("warning", "No file", "Select a CSV data file.")]
    assert ui.text("ml_result").strip() == ""


def test_the_feature_button_reports_a_file_it_could_not_read(ui, tmp_path):
    ui.set("ml_path", str(tmp_path / "absent.csv"))
    ui.set("ml_label", "class")

    ui.app._run_ml_features()

    errors = ui.messages_of("error")
    assert len(errors) == 1 and "absent.csv" in errors[0], errors


def test_the_roc_button_names_a_missing_label_column(ui, expression_csv):
    ui.set("ml_path", str(expression_csv))
    ui.set("ml_label", "absent_column")

    ui.app._run_ml_roc()

    assert ui.messages_of("error") == ["Column 'absent_column' not found."]


def test_the_roc_button_surfaces_a_core_failure(ui, tmp_path):
    path = tmp_path / "notes.csv"
    path.write_text("text,class\n" + "\n".join(
        f"alpha{i},{'yes' if i % 2 else 'no'}" for i in range(8)) + "\n", encoding="utf-8")
    ui.set("ml_path", str(path))
    ui.set("ml_label", "class")
    ui.clear("ml_result")

    ui.app._run_ml_roc()

    errors = ui.messages_of("error")
    assert len(errors) == 1, "the failure must be named, not swallowed"
    assert ui.text("ml_result").strip() == ""


def test_the_feature_button_surfaces_a_core_failure(ui, tmp_path):
    path = tmp_path / "notes.csv"
    path.write_text("text,class\n" + "\n".join(
        f"alpha{i},{'yes' if i % 2 else 'no'}" for i in range(8)) + "\n", encoding="utf-8")
    ui.set("ml_path", str(path))
    ui.set("ml_label", "class")
    ui.clear("ml_result")

    ui.app._run_ml_features()

    errors = ui.messages_of("error")
    assert len(errors) == 1, "the failure must be named, not swallowed"
    assert ui.text("ml_result").strip() == ""
