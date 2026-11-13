"""What a Vina output file says about each pose, and what it does not say.

Vina writes one `REMARK VINA RESULT` line and the pose's own atom records per
MODEL.  Reading only the REMARK line — which is all the parser used to do — threw
the coordinates away, so every pose in the report carried the centre of the search
box as its position and three different placements printed as the same point.
"""
import math
import warnings

import pytest

from biosuite.core import docking as dk


def atom_line(serial, x, y, z):
    return (f"ATOM  {serial:>5} C1{'':<2} LIG A{1:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n")


def pose_file(tmp_path, name, blocks):
    path = tmp_path / name
    path.write_text("".join(blocks), encoding="utf-8")
    return str(path)


def single_model(*atom_lines):
    return ["MODEL 1\n", "REMARK VINA RESULT:    -8.4      0.000  0.000\n",
            *atom_lines, "ENDMDL\n"]


@pytest.fixture
def receptor(tmp_path):
    path = tmp_path / "rec.pdb"
    path.write_text("".join(
        f"ATOM  {i + 1:>5}  CA  ALA A{i + 1:>4}    "
        f"{i * 3.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00           C\n"
        for i in range(10)), encoding="utf-8")
    return str(path)


@pytest.fixture
def ligand(tmp_path):
    path = tmp_path / "lig.pdb"
    path.write_text("HETATM    1  C1  LIG B   1       0.000   0.000   0.000  1.00  0.00\n"
                    "HETATM    2  C2  LIG B   1       1.500   0.000   0.000  1.00  0.00\n",
                    encoding="utf-8")
    return str(path)


def test_a_model_reports_the_centroid_of_its_own_atoms(tmp_path):
    path = pose_file(tmp_path, "out.pdbqt", single_model(
        atom_line(1, 100.0, 200.0, 300.0),
        atom_line(2, 102.0, 200.0, 300.0)))

    assert dk._parse_vina_output_pdbqt(path) == [(1, -8.4, [101.0, 200.0, 300.0])]


def test_two_models_are_two_poses_at_two_different_places(tmp_path):
    path = pose_file(tmp_path, "out.pdbqt",
                     single_model(atom_line(1, 0.0, 0.0, 0.0))
                     + ["MODEL 2\n",
                        "REMARK VINA RESULT:    -7.15     0.000  0.000\n",
                        atom_line(1, 40.0, 10.0, -6.0), "ENDMDL\n"])

    poses = dk._parse_vina_output_pdbqt(path)

    assert [p[0] for p in poses] == [1, 2], poses
    assert [p[1] for p in poses] == [-8.4, -7.15], poses
    assert poses[0][2] == [0.0, 0.0, 0.0] and poses[1][2] == [40.0, 10.0, -6.0]


def test_a_pose_without_atoms_keeps_its_energy_and_admits_no_position(tmp_path):
    path = pose_file(tmp_path, "out.pdbqt", single_model())

    rank, energy, coordinate = dk._parse_vina_output_pdbqt(path)[0]

    assert (rank, energy) == (1, -8.4)
    assert len(coordinate) == 3 and all(math.isnan(v) for v in coordinate), coordinate


def test_a_remark_line_is_a_pose_even_when_the_file_has_no_model_wrapper(tmp_path):
    """The contract is one pose per `REMARK VINA RESULT`, not one per MODEL."""
    path = pose_file(tmp_path, "out.pdbqt",
                    ["REMARK VINA RESULT:    -9.0      0.000  0.000\n",
                     atom_line(1, 5.0, 5.0, 5.0)])

    assert dk._parse_vina_output_pdbqt(path) == [(1, -9.0, [5.0, 5.0, 5.0])]


def test_atom_records_before_the_first_remark_belong_to_no_pose(tmp_path):
    """The autofile ligand is not a docked pose, and must not be counted as one."""
    path = pose_file(tmp_path, "out.pdbqt",
                    [atom_line(1, 999.0, 999.0, 999.0),
                     "REMARK VINA RESULT:    -8.4      0.000  0.000\n",
                     atom_line(2, 1.0, 2.0, 3.0)])

    assert dk._parse_vina_output_pdbqt(path) == [(1, -8.4, [1.0, 2.0, 3.0])]


def test_a_pose_record_whose_coordinates_are_not_numbers_is_counted(tmp_path, caplog):
    path = pose_file(tmp_path, "out.pdbqt", single_model(
        atom_line(1, 1.0, 2.0, 3.0),
        "ATOM      2  C1  LIG A   1     XXXXXXX   2.000   3.000  1.00  0.00\n"))

    with caplog.at_level("WARNING", logger="biosuite.core.docking"):
        poses = dk._parse_vina_output_pdbqt(path)

    assert poses[0][2] == [1.0, 2.0, 3.0], poses
    assert "skipped 1 pose ATOM/HETATM record" in caplog.text, caplog.text


def test_an_energy_that_is_not_a_number_is_not_a_pose(tmp_path):
    path = pose_file(tmp_path, "out.pdbqt",
                    ["REMARK VINA RESULT:   abc      0.000  0.000\n"])

    assert dk._parse_vina_output_pdbqt(path) == []


def test_an_integer_energy_is_a_pose(tmp_path):
    path = pose_file(tmp_path, "out.pdbqt",
                    ["REMARK VINA RESULT:      -9      0.000  0.000\n"])

    assert dk._parse_vina_output_pdbqt(path)[0][1] == -9.0


def test_a_pose_file_that_is_not_there_reads_as_no_poses(tmp_path):
    assert dk._parse_vina_output_pdbqt(str(tmp_path / "absent.pdbqt")) == []


def test_the_positions_in_a_docking_result_are_the_pose_positions(tmp_path, monkeypatch,
                                                                 receptor, ligand):
    monkeypatch.setattr(dk, "check_docking_tools", lambda: {'vina': True})

    body = "".join(
        ["MODEL 1\n", "REMARK VINA RESULT:    -8.4      0.000  0.000\n",
         atom_line(1, 100.0, 200.0, 300.0), "ENDMDL\n",
         "MODEL 2\n", "REMARK VINA RESULT:    -7.1      0.000  0.000\n",
         atom_line(1, -20.0, 5.0, 7.5), "ENDMDL\n"])

    def run(cmd, **kwargs):
        with open(cmd[cmd.index('--out') + 1], 'w', encoding='utf-8') as fh:
            fh.write(body)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()

    monkeypatch.setattr(dk.subprocess, "run", run)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dk.dock(receptor, ligand, center=[0, 0, 0], tool='vina')

    assert [(p.x, p.y, p.z) for p in result.poses] == [(100.0, 200.0, 300.0),
                                                      (-20.0, 5.0, 7.5)]
    assert len({(p.x, p.y, p.z) for p in result.poses}) == 2, \
        "two poses at one point is the bug this replaces"
