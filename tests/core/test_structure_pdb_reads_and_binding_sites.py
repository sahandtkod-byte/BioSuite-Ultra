"""Structure reads that must never be silently wrong: Biopython-missing guards,
PDB header extraction, the DSSP-versus-phi/psi secondary-structure split, ligand
neighbourhood searches, PDB-ID download, and the report's optional sections.

PDB geometry here is generated with ideal backbone internals (NeRF construction), and
every expected angle, count, SASA or report line below was read off the running
package rather than derived on paper.  No DSSP binary and no network is ever used:
``DSSP``, ``_has_tool`` and ``Bio.PDB.PDBList`` are replaced in the module that
looks them up.
"""
import importlib.util
import math
import pathlib
import sys

import numpy as np
import pytest

from biosuite.core import structure as S

HELIX = (-57.0, -47.0)
SHEET = (-120.0, 130.0)
COIL = (60.0, 60.0)


def _nerf(a, b, c, bond, angle_deg, dihedral_deg):
    """Place atom *d* bonded to *c*: |c-d|, angle(b,c,d), dihedral(a,b,c,d)."""
    angle, dihedral = math.radians(angle_deg), math.radians(dihedral_deg)
    v = (c - b) / np.linalg.norm(c - b)
    n = np.cross(b - a, v)
    n = n / np.linalg.norm(n)
    m = np.cross(n, v)
    return (c + v * (-bond * math.cos(angle))
            + m * (bond * math.sin(angle) * math.cos(dihedral))
            + n * (bond * math.sin(angle) * math.sin(dihedral)))


def _backbone(n_residues, phi_deg, psi_deg):
    """Ideal N/CA/C triplets whose measured phi/psi equal the requested angles."""
    n_atom = np.array([0.0, 0.0, 0.0])
    ca_atom = np.array([1.458, 0.0, 0.0])
    c_atom = ca_atom + 1.525 * np.array(
        [-math.cos(math.radians(111.0)), math.sin(math.radians(111.0)), 0.0])
    placed = [[n_atom, ca_atom, c_atom]]
    for index in range(1, n_residues):
        prev_n, prev_ca, prev_c = placed[index - 1]
        next_n = _nerf(prev_n, prev_ca, prev_c, 1.329, 116.2, psi_deg)
        next_ca = _nerf(prev_ca, prev_c, next_n, 1.458, 121.7, 180.0)
        next_c = _nerf(prev_c, next_n, next_ca, 1.525, 111.0, phi_deg)
        placed.append([next_n, next_ca, next_c])
    return placed


def _atom_line(record, serial, name, resname, chain, resseq, coord, bfactor=0.0, element="C"):
    """Fixed-width PDB ATOM/HETATM record: name 13-16, chain 22, resSeq 23-26, element 77-78."""
    return ("%-6s%5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f  1.00%6.2f%10s%2s"
            % (record, serial, name, resname, chain, resseq,
               coord[0], coord[1], coord[2], bfactor, "", element))


def _pdb_text(chains):
    """chains: list of (resname, chain_id, n_residues, (phi, psi)) tuples."""
    lines, serial = [], 1
    for resname, chain_id, length, (phi, psi) in chains:
        for index, (n_atom, ca_atom, c_atom) in enumerate(_backbone(length, phi, psi), start=1):
            for name, coord, element in (("N", n_atom, "N"), ("CA", ca_atom, "C"),
                                         ("C", c_atom, "C"),
                                         ("CB", c_atom + np.array([0.5, 0.5, 0.5]), "C")):
                lines.append(_atom_line("ATOM", serial, name, resname, chain_id, index,
                                        coord, element=element))
                serial += 1
    lines.append("TER")
    return "\n".join(lines) + "\nEND\n"


def _hetero_text():
    """Three protein residues around a three-atom HEM ligand that spans 12 A."""
    lines, serial = [], 1
    for x, resname, residue_number in ((0.0, "ALA", 1), (12.0, "GLY", 2), (6.0, "VAL", 3)):
        for offset, name, element in ((0.0, "N", "N"), (0.5, "CA", "C"),
                                      (1.0, "C", "C"), (1.5, "O", "O")):
            lines.append(_atom_line("ATOM", serial, name, resname, "A", residue_number,
                                    np.array([x + offset, 0.0, 0.0]), element=element))
            serial += 1
    for x, name, element in ((0.0, "FE", "FE"), (12.0, "N1", "N"), (6.0, "N2", "N")):
        lines.append(_atom_line("HETATM", serial, name, "HEM", "B", 400,
                                np.array([x, 0.0, 3.0]), element=element))
        serial += 1
    lines.append("TER")
    return "\n".join(lines) + "\nEND\n"


MINI_PDB = (
    "HEADER    HYDROLASE (TEST)                          01-JAN-00   0000\n"
    "EXPDTA    X-RAY DIFFRACTION\n"
    "REMARK   2 RESOLUTION.    1.85 ANGSTROMS.\n"
    "ATOM      1  N   ALA A   1     -10.000   0.000   0.000  1.00 20.00           N\n"
    "ATOM      2  CA  ALA A   1      -9.000   0.500   0.000  1.00 20.00           C\n"
    "ATOM      3  C   ALA A   1      -8.200  -0.500   0.800  1.00 20.00           C\n"
    "ATOM      4  O   ALA A   1      -8.200  -1.700   0.700  1.00 20.00           O\n"
    "END\n"
)


@pytest.fixture
def mini_pdb(tmp_path):
    path = tmp_path / "mini.pdb"
    path.write_text(MINI_PDB, encoding="utf-8")
    return str(path)


@pytest.fixture
def mixed_pdb(tmp_path):
    """Helix chain A, sheet chain B, coil chain C — all ideal geometry."""
    path = tmp_path / "mixed.pdb"
    path.write_text(_pdb_text([("ALA", "A", 6, HELIX), ("GLY", "B", 6, SHEET),
                               ("SER", "C", 4, COIL)]), encoding="utf-8")
    return str(path)


@pytest.fixture
def ligand_pdb(tmp_path):
    path = tmp_path / "ligand.pdb"
    path.write_text(_hetero_text(), encoding="utf-8")
    return str(path)


@pytest.fixture
def structure_of(mini_pdb):
    def _load(path):
        structure, error = S.parse_pdb(path)
        assert error is None
        return structure
    return _load


@pytest.fixture
def biopython_free(monkeypatch):
    """Re-execute the module top level with ``Bio`` unimportable.

    ``importlib.reload`` keeps the attributes bound by the first import, so HAS_BIO
    would still be True; the dotted name stays inside ``biosuite.core`` for the
    module's relative ``from .utils import`` line.
    """
    monkeypatch.setitem(sys.modules, "Bio", None)
    monkeypatch.setitem(sys.modules, "Bio.PDB", None)
    name = "biosuite.core.structure_without_biopython"
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(S.__file__))
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, fresh)
    spec.loader.exec_module(fresh)
    assert fresh.HAS_BIO is False, "the optional-dependency guard is what is under test"
    return fresh


def test_module_without_biopython_degrades_on_every_entry_point(biopython_free):
    fresh = biopython_free
    assert fresh.parse_pdb("anything.pdb") == (None, "Biopython not installed")
    info = fresh.get_structure_info(object(), "1UBQ")
    assert (info.name, info.num_atoms, info.num_residues) == ("1UBQ", 0, 0)
    assert info.message == "No structure loaded"
    assert fresh.compute_secondary_structure(object()) == {}
    assert fresh.compute_ramachandran(object()) == {}
    assert fresh.compute_sasa(object()) == 0.0
    assert fresh.find_binding_sites(object()) == []
    assert fresh.load_pdb(filepath="x.pdb") == (None, "Biopython not installed")


def test_dssp_binary_names_are_both_accepted(monkeypatch):
    monkeypatch.setattr(S, "_has_tool", lambda name: name == "mkdssp")
    assert S.check_structure_tools() == {'dssp': True}
    monkeypatch.setattr(S, "_has_tool", lambda name: name == "dssp")
    assert S.check_structure_tools() == {'dssp': True}
    monkeypatch.setattr(S, "_has_tool", lambda name: False)
    assert S.check_structure_tools() == {'dssp': False}


def test_parse_pdb_reports_the_file_error_it_hit(tmp_path):
    missing = str(tmp_path / "absent.pdb")
    structure, error = S.parse_pdb(missing)
    assert structure is None
    assert "No such file or directory" in error
    assert "absent.pdb" in error


def test_header_metadata_comes_from_the_pdb_remarks(mini_pdb):
    structure, _ = S.parse_pdb(mini_pdb)
    info = S.get_structure_info(structure, "0000")
    assert (info.num_atoms, info.num_residues, info.num_chains) == (4, 1, 1)
    assert info.chains == ["A"]
    assert info.resolution == pytest.approx(1.85)
    assert info.method == "x-ray diffraction"
    assert info.molecule_type == "unknown", "PDB headers carry no molecule_type key"
    assert info.message == ""


def test_header_extraction_without_a_header_dict_is_reported_not_swallowed(structure_of,
                                                                          ligand_pdb,
                                                                          monkeypatch):
    """A Model/Chain has no ``.header``; the caller must be told, not left guessing."""
    class Headless:
        def __init__(self, model):
            self._model = model

        def __getitem__(self, index):
            return self._model

    model_only = Headless(structure_of(ligand_pdb)[0])
    with pytest.warns(RuntimeWarning, match="PDB header metadata unavailable"):
        info = S.get_structure_info(model_only, "headless")
    assert (info.num_atoms, info.num_residues, info.num_chains) == (15, 4, 2)
    assert info.message.startswith("No header metadata (AttributeError")
    assert info.resolution == 0.0 and info.method == ""


def test_get_structure_info_without_a_structure_is_explicit():
    info = S.get_structure_info(None, "1UBQ")
    assert info.message == "No structure loaded"
    assert info.chains == [] and info.surface_area == 0.0


def test_secondary_structure_prefers_dssp_rows_and_bins_the_three_states(mixed_pdb,
                                                                        monkeypatch):
    """H/G/I -> helix, E/B -> sheet, everything else -> coil; DSSP gets the file path."""
    structure, _ = S.parse_pdb(mixed_pdb)
    seen = []
    codes = ['H', 'G', 'I', 'E', 'B', '-', '3', 'T']

    class FakeDSSP:
        """Biopython's DSSP layout: flat 14-element rows looked up by (chain, res_id).

        ``__iter__`` yields the rows themselves, not ``(key, row)`` pairs, and the
        secondary structure code sits at index 2 of a row, so a stand-in has to
        reproduce that shape or the test proves nothing about the real class.
        """

        def __init__(self, model, filepath):
            seen.append((model, filepath))

        def keys(self):
            return [("A", index) for index in range(len(codes))]

        def __getitem__(self, key):
            index = key[1]
            return (index + 1, 'X', codes[index], 0.5, -60.0, -45.0,
                    0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0)

    monkeypatch.setattr(S, "DSSP", FakeDSSP)
    monkeypatch.setattr(S, "check_structure_tools", lambda: {'dssp': True})
    assert S.compute_secondary_structure(structure, filepath=mixed_pdb) == {
        'H': 3, 'E': 2, 'C': 3}
    assert seen and seen[0][1] == mixed_pdb
    assert seen[0][0] is structure[0]


@pytest.mark.parametrize("dssp_installed, filepath_state", [
    pytest.param(True, "missing-path", id="dssp-works-no-path"),
    pytest.param(True, "absent-file", id="dssp-works-stale-path"),
    pytest.param(False, "real-file", id="no-dssp"),
])
def test_dssp_is_only_attempted_when_a_readable_file_is_at_hand(mixed_pdb, monkeypatch,
                                                                dssp_installed, filepath_state):
    structure, _ = S.parse_pdb(mixed_pdb)
    called = []
    monkeypatch.setattr(S, "DSSP",
                        lambda model, filepath: called.append(filepath) or iter([]))
    monkeypatch.setattr(S, "check_structure_tools", lambda: {'dssp': dssp_installed})
    filepath = {"missing-path": None,
                "absent-file": str(pathlib.Path(mixed_pdb).parent / "gone.pdb"),
                "real-file": mixed_pdb}[filepath_state]
    assert S.compute_secondary_structure(structure, filepath=filepath) == {
        'H': 4, 'E': 4, 'C': 8}
    assert called == [], "the heuristic is the only path when DSSP cannot be used"


def test_dssp_failure_warns_and_degrades_to_the_phi_psi_heuristic(mixed_pdb, monkeypatch):
    structure, _ = S.parse_pdb(mixed_pdb)

    def refuse(model, filepath):
        raise FileNotFoundError(2, "mkdssp: No such file or directory")

    monkeypatch.setattr(S, "DSSP", refuse)
    monkeypatch.setattr(S, "check_structure_tools", lambda: {'dssp': True})
    with pytest.warns(RuntimeWarning, match="DSSP could not be run"):
        counts = S.compute_secondary_structure(structure, filepath=mixed_pdb)
    assert counts == {'H': 4, 'E': 4, 'C': 8}


def test_phi_psi_heuristic_separates_helix_sheet_and_coil(mixed_pdb):
    """Measured: 6-residue helix and sheet chains each lose their two terminal residues
    (undefined phi/psi) to the coil bin, and the 60/60 chain is coil throughout."""
    structure, _ = S.parse_pdb(mixed_pdb)
    assert S.compute_secondary_structure(structure) == {'H': 4, 'E': 4, 'C': 8}
    assert S.compute_secondary_structure(structure, filepath=None) == {'H': 4, 'E': 4, 'C': 8}


def test_ramachandran_pairs_phi_with_the_psi_of_the_same_residue(mixed_pdb):
    """Measured on 16 ideal residues: 6 of them (the first and last of each of the
    three peptides) lack one angle, so 10 complete pairs remain.  The lists used to
    hold 13 entries each, sorted per angle, which paired A6's phi with chain B's
    first psi when zipped — a point that exists in no residue."""
    structure, _ = S.parse_pdb(mixed_pdb)
    angles = S.compute_ramachandran(structure)
    assert sorted(angles) == ['phi', 'psi']
    assert len(angles['phi']) == 10 and len(angles['psi']) == 10
    assert angles['phi'] == pytest.approx(
        [-57.01, -57.02, -57.04, -57.08, -119.98, -119.96, -120.00, -120.03,
         60.01, 60.08], abs=0.01)
    assert angles['psi'] == pytest.approx(
        [-46.97, -47.00, -46.92, -46.94, 129.95, 129.99, 130.03, 129.98,
         59.96, 60.01], abs=0.01)
    pairs = list(zip(angles['phi'], angles['psi']))
    assert all(phi == pytest.approx(HELIX[0], abs=0.1)
               and psi == pytest.approx(HELIX[1], abs=0.1) for phi, psi in pairs[:4])
    assert all(phi == pytest.approx(SHEET[0], abs=0.1)
               and psi == pytest.approx(SHEET[1], abs=0.1) for phi, psi in pairs[4:8])
    assert all(phi == pytest.approx(COIL[0], abs=0.1)
               and psi == pytest.approx(COIL[1], abs=0.1) for phi, psi in pairs[8:])
    assert {type(value).__name__ for value in angles['phi']} == {'float'}


def test_compute_sasa_sums_per_residue_areas(mixed_pdb, structure_of):
    assert S.compute_sasa(structure_of(mixed_pdb)) == pytest.approx(903.6096976074595)


def test_compute_sasa_returns_zero_when_the_level_is_not_a_structure(ligand_pdb,
                                                                    structure_of):
    """A Model has no ``[0]`` sub-model: the Shrake-Rupley call cannot run."""
    model_only = structure_of(ligand_pdb)[0]
    assert S.compute_sasa(model_only) == 0.0


def test_binding_sites_search_from_every_ligand_atom(ligand_pdb, structure_of):
    """HEM spans 12 A; seeding only its first atom hid GLY 2 and crashed on ``.vector``."""
    structure = structure_of(ligand_pdb)
    assert S.find_binding_sites(structure, radius=8.0) == [
        {'residue': 'ALA', 'chain': 'A', 'number': 1},
        {'residue': 'VAL', 'chain': 'A', 'number': 3},
        {'residue': 'GLY', 'chain': 'A', 'number': 2},
    ]


@pytest.mark.parametrize("radius, expected", [
    pytest.param(2.0, [], id="too-far"),
    pytest.param(8.0, 3, id="all-three"),
])
def test_binding_site_radius_is_honoured(ligand_pdb, structure_of, radius, expected):
    found = S.find_binding_sites(structure_of(ligand_pdb), radius=radius)
    if expected == 3:
        assert {entry['residue'] for entry in found} == {'ALA', 'GLY', 'VAL'}
        assert len(found) == 3, "one entry per residue, not per contacting atom"
    else:
        assert found == expected


def test_binding_sites_without_the_named_ligand_is_empty(ligand_pdb, structure_of):
    assert S.find_binding_sites(structure_of(ligand_pdb), ligand_residue="ZZZ") == []


def test_binding_sites_treats_the_requested_residue_as_the_ligand(ligand_pdb, structure_of):
    found = S.find_binding_sites(structure_of(ligand_pdb), ligand_residue="ALA", radius=4.0)
    assert found == [{'residue': 'HEM', 'chain': 'B', 'number': 400}]


def test_load_pdb_reads_a_local_file_before_considering_an_id(mini_pdb):
    structure, error = S.load_pdb(pdb_id="9999", filepath=mini_pdb)
    assert error is None
    assert structure.id == "protein"
    assert [c.id for c in structure[0]] == ["A"]


def test_load_pdb_fetches_a_pdb_id_through_biopython(mini_pdb, monkeypatch):
    calls = []

    class FakePDBList:
        def retrieve_pdb_file(self, pdb_id, *args, **kwargs):
            calls.append((pdb_id, kwargs))
            return mini_pdb

    import Bio.PDB
    monkeypatch.setattr(Bio.PDB, "PDBList", FakePDBList)
    structure, error = S.load_pdb(pdb_id="0000")
    assert error is None
    assert calls == [("0000", {"file_format": "pdb"})]
    assert structure.id == "0000"
    assert S.get_structure_info(structure, "0000").num_residues == 1


def test_load_pdb_reports_a_failed_download_without_raising(monkeypatch):
    class BoomPDBList:
        def retrieve_pdb_file(self, *args, **kwargs):
            raise OSError("no network")

    import Bio.PDB
    monkeypatch.setattr(Bio.PDB, "PDBList", BoomPDBList)
    structure, error = S.load_pdb(pdb_id="9ABC")
    assert structure is None
    assert error == "Could not download PDB 9ABC: no network"


def test_load_pdb_without_input_says_so(tmp_path):
    assert S.load_pdb() == (None, "No PDB ID or file path provided")
    # A stale path with no id is reported as "no input" — pinned, see the audit note.
    assert S.load_pdb(filepath=str(tmp_path / "gone.pdb")) == (
        None, "No PDB ID or file path provided")


def test_full_analysis_composes_counts_geometry_and_surface(mixed_pdb):
    info = S.full_analysis(filepath=mixed_pdb)
    assert info.name == "loaded"
    assert (info.num_atoms, info.num_residues, info.num_chains) == (64, 16, 3)
    assert info.chains == ["A", "B", "C"]
    assert info.secondary_structure == {'H': 4, 'E': 4, 'C': 8}
    assert len(info.ramachandran['phi']) == 10
    assert info.surface_area == pytest.approx(903.6096976074595)
    assert info.message == ""


def test_full_analysis_returns_the_load_error_as_a_report_note(tmp_path):
    info = S.full_analysis(pdb_id=None, filepath=str(tmp_path / "gone.pdb"))
    assert info.name == "unknown"
    assert info.message == "No PDB ID or file path provided"
    assert info.num_residues == 0


def test_format_structure_report_lists_every_populated_section():
    info = S.StructureInfo(name="1UBQ", num_atoms=118, num_residues=76, num_chains=2,
                           chains=["A", "B"], resolution=1.8, method="X-RAY DIFFRACTION",
                           surface_area=4200.5, secondary_structure={"H": 40, "E": 10, "C": 26},
                           ramachandran={"phi": [-60.0, -57.0, 120.0],
                                         "psi": [-45.0, -47.0, 130.0]},
                           message="boom")
    assert S.format_structure_report(info).splitlines() == [
        "=== Protein Structure Report ===",
        "Name: 1UBQ",
        "Atoms: 118",
        "Residues: 76",
        "Chains: 2 (A, B)",
        "Resolution: 1.80 Å",
        "Method: X-RAY DIFFRACTION",
        "Surface area: 4200 Å²",
        "Secondary structure: 53% helix, 13% sheet, 34% coil",
        "Ramachandran: 3 angles (phi: 1.0±84.2°)",
        "Note: boom",
    ]


def test_format_structure_report_omits_sections_that_carry_no_values():
    text = S.format_structure_report(S.StructureInfo(name="x"))
    assert text.splitlines() == [
        "=== Protein Structure Report ===", "Name: x", "Atoms: 0", "Residues: 0",
        "Chains: 0 ()"]
    assert "Resolution" not in text and "Method" not in text and "Surface area" not in text
    assert "Secondary structure" not in text and "Ramachandran" not in text
    assert "Note" not in text


@pytest.mark.parametrize("secondary, shows_line", [
    pytest.param({"H": 0, "E": 0, "C": 0}, False, id="all-zero-total"),
    pytest.param({"H": 1, "E": 0, "C": 0}, True, id="single-helix"),
    pytest.param(["H"], False, id="not-a-dict"),
    pytest.param({"Q": 3}, False, id="no-H-key"),
])
def test_format_structure_report_secondary_structure_line_needs_usable_counts(secondary,
                                                                             shows_line):
    info = S.StructureInfo(name="x", secondary_structure=secondary)
    assert ("Secondary structure" in S.format_structure_report(info)) is shows_line


def test_format_structure_report_skips_an_empty_phi_list():
    assert "Ramachandran" not in S.format_structure_report(
        S.StructureInfo(name="x", ramachandran={"phi": [], "psi": []}))
    assert "Ramachandran" not in S.format_structure_report(
        S.StructureInfo(name="x", ramachandran={"psi": [1.0]}))
