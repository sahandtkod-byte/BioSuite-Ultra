"""Where the search box goes, how many poses may be asked for, and which paths dock.

The centre of the box is the one number a caller most often leaves out, and the
two engines used to answer that omission differently: the built-in search centred
itself on the receptor while Vina was handed the origin (0, 0, 0).  A real receptor
whose atoms sit at x=100..160, y=200, z=300 therefore got a box of empty space, and
the run still reported an energy.
"""
import warnings

import numpy as np
import pytest

from biosuite.core import docking as dk


def atom_line(serial, name, x, y, z, record="ATOM"):
    return (f"{record:<6}{serial:>5} {name:<4} ALA A{1:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n")


@pytest.fixture
def receptor(tmp_path):
    """20 atoms along x at y=200, z=300 — centroid (128.5, 200, 300)."""
    path = tmp_path / "rec.pdb"
    path.write_text("".join(atom_line(i + 1, "CA", 100 + i * 3.0, 200.0, 300.0)
                            for i in range(20)), encoding="utf-8")
    return str(path)


@pytest.fixture
def ligand(tmp_path):
    path = tmp_path / "lig.pdb"
    path.write_text("".join(atom_line(i + 1, "C1", i * 1.5, 0.0, 0.0)
                            for i in range(4)), encoding="utf-8")
    return str(path)


@pytest.fixture
def headers_only(tmp_path):
    path = tmp_path / "empty.pdb"
    path.write_text("HEADER    no coordinates\nEND\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def vina_installed(monkeypatch):
    """Pretend AutoDock Vina is on PATH and record the command lines it is given."""
    monkeypatch.setattr(dk, "check_docking_tools", lambda: {'vina': True})
    calls = []

    def run(cmd, **kwargs):
        calls.append(list(cmd))
        with open(cmd[cmd.index('--out') + 1], 'w', encoding='utf-8') as fh:
            fh.write("MODEL 1\nREMARK VINA RESULT: -8.4      0.000  0.000\nENDMDL\n")

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()

    monkeypatch.setattr(dk.subprocess, "run", run)
    return calls


def _option(cmd, name):
    return cmd[cmd.index(name) + 1]


# ── the centre nobody supplied ──────────────────────────────────────────────

def test_vina_gets_the_receptor_centre_when_the_caller_named_none(receptor, ligand,
                                                                 vina_installed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dk.dock(receptor, ligand, tool='vina')

    cmd = vina_installed[0]
    assert [float(_option(cmd, name)) for name in
            ('--center_x', '--center_y', '--center_z')] == [128.5, 200.0, 300.0]


def test_an_explicit_centre_still_wins_over_the_receptor(receptor, ligand,
                                                        vina_installed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dk.dock(receptor, ligand, center=[7.0, 8.0, 9.0], tool='vina')

    assert [float(_option(vina_installed[0], name))
            for name in ('--center_x', '--center_y', '--center_z')] == [7.0, 8.0, 9.0]


def test_the_built_in_search_also_centres_on_the_receptor(receptor, ligand):
    """Both engines must read an omitted centre the same way.

    This one always did; the test exists because the asymmetry between the two
    engines is the bug, and only a shared expectation keeps it from coming back.
    """
    result = dk.dock(receptor, ligand, center=None, box_size=(20, 20, 20),
                     num_poses=4, tool='builtin')

    centroid = np.array([128.5, 200.0, 300.0])
    assert len(result.poses) == 4, result.poses
    for pose in result.poses:
        offset = np.array([pose.x, pose.y, pose.z]) - centroid
        assert max(abs(offset)) <= 10.0 + 1e-3, pose


def test_an_explicit_centre_moves_the_built_in_boxes_with_it(receptor, ligand):
    result = dk._builtin_dock(receptor, ligand, center=[0.0, 0.0, 0.0],
                              box_size=(8.0, 8.0, 8.0), num_poses=3)

    for pose in result.poses:
        assert max(abs(pose.x), abs(pose.y), abs(pose.z)) <= 4.0 + 1e-3, pose


def test_a_receptor_without_atoms_is_never_handed_to_vina(headers_only, ligand,
                                                          vina_installed):
    """No atoms means no centre, and running vina on a box nobody chose is worse.

    The receptor's own file is what says where the site is; without it the search
    would be pointed at (0, 0, 0) and still return an energy.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = dk.dock(headers_only, ligand, tool='vina')

    assert vina_installed == [], "vina was invoked although the box has no centre"
    assert result.engine == 'builtin', result
    assert result.message == "Could not parse atoms from input files", result.message
    assert "no ATOM or HETATM records" in str(caught[0].message), str(caught[0].message)


# ── the arguments, validated before anything is read ────────────────────────

@pytest.mark.parametrize("center", [
    [0, 0],
    (1, 2, 3, 4),
    "abc",
    20,
    [float('nan'), 0, 0],
    [0, 0, float('inf')],
])
def test_center_must_be_three_finite_numbers(center, receptor, ligand):
    """``[0, 0]`` used to reach numpy as ``operands could not be broadcast`` and as
    ``list index out of range`` on the Vina path — two crashes, neither of which
    names the argument that was wrong."""
    with pytest.raises(ValueError, match="center"):
        dk.dock(receptor, ligand, center=center)


@pytest.mark.parametrize("box_size", [20, (10, 10), (0, 5, 5), (-1, 5, 5), "wide"])
def test_box_size_must_be_three_positive_numbers(box_size, receptor, ligand):
    with pytest.raises(ValueError, match="box_size"):
        dk.dock(receptor, ligand, box_size=box_size)


@pytest.mark.parametrize("num_poses", [0, -3, 2.5, True, "3", None])
def test_num_poses_must_be_an_integer_of_at_least_one(num_poses, receptor, ligand):
    with pytest.raises(ValueError, match="num_poses"):
        dk.dock(receptor, ligand, num_poses=num_poses)


def test_num_poses_is_rejected_before_the_input_files_are_read(headers_only):
    """The old order parsed both PDB files, then complained about the count.

    A run that cannot produce poses should not pay for reading a receptor first.
    """
    with pytest.raises(ValueError, match="num_poses"):
        dk._builtin_dock(headers_only, headers_only, num_poses=0)


def test_the_built_in_engine_validates_the_same_arguments_as_the_public_one(
        receptor, ligand):
    """The engines are callable on their own, so each has to check its own domain."""
    with pytest.raises(ValueError, match="num_poses"):
        dk._builtin_dock(receptor, ligand, num_poses=-1)
    with pytest.raises(ValueError, match="box_size"):
        dk._builtin_dock(receptor, ligand, box_size=(10, 10))


def test_vina_is_told_the_box_and_the_pose_count_it_was_given(receptor, ligand,
                                                             vina_installed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dk.dock(receptor, ligand, center=[1.0, 2.0, 3.0], box_size=(10, 12, 14),
                num_poses=7, tool='vina')

    cmd = vina_installed[0]
    assert [float(_option(cmd, name)) for name in
            ('--size_x', '--size_y', '--size_z')] == [10.0, 12.0, 14.0]
    assert _option(cmd, '--num_modes') == "7"
    assert cmd[0] == 'vina'


# ── the paths a run may be given ────────────────────────────────────────────

def test_a_missing_receptor_names_itself_in_the_result(tmp_path, ligand):
    absent = tmp_path / "nope.pdb"

    result = dk.dock(str(absent), ligand, tool='builtin')

    assert result.engine == 'none', result
    assert result.message == f"Receptor not found: {absent}", result.message


def test_a_missing_ligand_names_the_ligand(receptor, tmp_path):
    absent = tmp_path / "none.pdb"

    result = dk.dock(receptor, str(absent), tool='builtin')

    assert result.engine == 'none', result
    assert result.message == f"Ligand not found: {absent}", result.message


def test_a_directory_is_not_a_receptor_even_though_it_exists(receptor, tmp_path,
                                                             ligand):
    """``os.path.exists`` is true for a folder, and opening one raised.

    A friendly "not found" for one bad path and a PermissionError traceback for
    another taught callers that no path could be trusted.
    """
    directory = tmp_path / "a_folder"
    directory.mkdir()

    result = dk.dock(str(directory), ligand, tool='builtin')

    assert result.engine == 'none', result
    assert result.message == f"Receptor is not a file: {directory}", result.message


def test_a_file_without_coordinate_records_docks_nothing(receptor, headers_only):
    result = dk.dock(receptor, headers_only, tool='builtin')

    assert result.engine == 'builtin'
    assert result.message == "Could not parse atoms from input files", result.message
    assert result.poses == [] and result.num_poses == 0
