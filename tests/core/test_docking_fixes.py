"""Regression tests for docking vina parsing + builtin ranking."""
import os
import tempfile
import textwrap

import numpy as np
import pytest

from biosuite.core import docking


def _pdb(tmp_path, name, n=12, offset=(0, 0, 0)):
    lines = []
    for i in range(n):
        x, y, z = i * 3.8 + offset[0], offset[1], offset[2]
        lines.append(f"ATOM  {i+1:5d}  CA  ALA A{1:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.0011.11           C")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\nEND\n")
    return str(p)


def test_parse_vina_output(tmp_path):
    out = tmp_path / "out.pdbqt"
    out.write_text(textwrap.dedent("""\
        MODEL 1
        REMARK VINA RESULT:    -7.5      0.000      0.000
        ATOM      1  C1  LIG A   1       1.0   2.0   3.0
        ENDMDL
        MODEL 2
        REMARK VINA RESULT:    -6.1      0.000      0.000
        ENDMDL
        """))
    assert docking._parse_vina_output_pdbqt(str(out)) == [(1, -7.5), (2, -6.1)]


def test_vina_path_populates_poses(tmp_path, monkeypatch):
    rec = _pdb(tmp_path, "rec.pdb")
    lig = _pdb(tmp_path, "lig.pdb")
    monkeypatch.setattr(docking, "check_docking_tools", lambda: {'vina': True})

    def fake_run(cmd, **kw):
        grep_path = cmd[-1]  # --out value
        with open(grep_path, 'w') as fh:
            fh.write("REMARK VINA RESULT: -8.4 0.0 0.0\n"
                     "REMARK VINA RESULT: -7.2 0.0 0.0\n")

        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(docking.subprocess, "run", fake_run)
    res = docking.dock(rec, lig, tool='vina')
    assert res.engine == 'vina'
    assert res.num_poses == 2
    assert res.binding_energy == pytest.approx(-8.4)
    # before the fix: num_poses=0, binding_energy=0.0 — output discarded


def test_builtin_ranking_is_monotonic(tmp_path):
    rec = _pdb(tmp_path, "r.pdb")
    lig = _pdb(tmp_path, "l.pdb")
    res = docking._builtin_dock(rec, lig, num_poses=8)
    energies = [p.energy for p in res.poses]
    assert energies == sorted(energies)
    assert [p.rank for p in res.poses] == list(range(1, 9))


def test_dock_missing_files(tmp_path):
    res = docking.dock(str(tmp_path / "nope.pdb"), str(tmp_path / "none.pdb"))
    assert res.engine == 'none'
    assert "not found" in res.message.lower()


def test_temp_file_is_cleaned_up(tmp_path, monkeypatch):
    rec = _pdb(tmp_path, "r2.pdb")
    lig = _pdb(tmp_path, "l2.pdb")
    monkeypatch.setattr(docking, "check_docking_tools", lambda: {'vina': True})
    captured = []

    def fake_run(cmd, **kw):
        captured.append(cmd[-1])

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(docking.subprocess, "run", fake_run)
    docking.dock(rec, lig, tool='vina')
    assert captured and not os.path.exists(captured[0])


# ── BSU-010: the built-in engine must actually search, not invent ───────────

def _shell_receptor(tmp_path, name="shell.pdb", radius=12.0, n=300):
    """A hollow spherical shell of atoms: a receptor with a central cavity."""
    import numpy as np
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    coords = v * radius
    path = tmp_path / name
    path.write_text("".join(
        f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        for i, (x, y, z) in enumerate(coords)))
    return str(path)


def _small_ligand(tmp_path, name="lig.pdb", offset=(40.0, 40.0, 40.0)):
    import numpy as np
    coords = np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]],
                      dtype=float) + np.asarray(offset, dtype=float)
    path = tmp_path / name
    path.write_text("".join(
        f"HETATM{i + 1:5d}  C   LIG B{i + 1:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        for i, (x, y, z) in enumerate(coords)))
    return str(path)


def test_builtin_poses_are_scored_placements_not_noise(tmp_path):
    """Every reported energy must be the score of the reported placement.

    The old implementation scored the input geometry once and then produced
    `num_poses` results by adding `np.random.normal` noise to the receptor
    centroid and `np.random.uniform(-0.5, 0.5)` to that single score, so the
    poses carried no information about the receptor at all.
    """
    import numpy as np
    rec = _shell_receptor(tmp_path)
    lig = _small_ligand(tmp_path)
    res = docking._builtin_dock(rec, lig, num_poses=6, seed=0)

    rec_coords = np.array([[a['x'], a['y'], a['z']]
                           for a in docking._parse_pdb_atoms(rec)])
    # Distinct placements must give distinct scores, and the ordering must be
    # a real ranking of the score function.
    energies = [p.energy for p in res.poses]
    assert energies == sorted(energies)
    assert len({(p.x, p.y, p.z) for p in res.poses}) == len(res.poses)
    assert len(set(energies)) > 1

    # The best pose must beat a placement far outside the receptor, which
    # cannot happen if the energies are noise around a constant.
    far = np.array([[500.0, 500.0, 500.0]])
    assert res.poses[0].energy < docking._score_contacts(rec_coords, far)


def test_builtin_dock_is_reproducible_for_a_fixed_seed(tmp_path):
    rec = _shell_receptor(tmp_path)
    lig = _small_ligand(tmp_path)
    a = docking._builtin_dock(rec, lig, num_poses=5, seed=7)
    b = docking._builtin_dock(rec, lig, num_poses=5, seed=7)
    c = docking._builtin_dock(rec, lig, num_poses=5, seed=8)
    assert [(p.energy, p.x, p.y, p.z) for p in a.poses] == \
           [(p.energy, p.x, p.y, p.z) for p in b.poses]
    # A different seed explores different placements (it is a random search).
    assert [(p.x, p.y, p.z) for p in a.poses] != [(p.x, p.y, p.z) for p in c.poses]


def test_builtin_dock_finds_the_cavity(tmp_path):
    """The search must place the ligand inside the cavity, not at random."""
    import numpy as np
    rec = _shell_receptor(tmp_path, radius=12.0)
    lig = _small_ligand(tmp_path, offset=(60.0, 60.0, 60.0))
    res = docking._builtin_dock(rec, lig, center=(0, 0, 0),
                                box_size=(30, 30, 30), num_poses=3, seed=1)
    best = res.poses[0]
    # Inside the shell, in contact with it, and clearly not the input position.
    assert np.linalg.norm([best.x, best.y, best.z]) < 14.0
    assert best.energy < 0


def test_report_does_not_claim_kcal_per_mol_for_the_heuristic_engine():
    """Regression guard: heuristic scores were printed as 'kcal/mol'."""
    result = docking.DockingResult(
        engine='builtin', binding_energy=-6.5, num_poses=1,
        poses=[docking.Pose(rank=1, energy=-6.5, x=0, y=0, z=0)])
    text = docking.format_docking_report(result)
    assert 'kcal/mol' not in text
    assert 'arbitrary units' in text

    vina = docking.DockingResult(
        engine='vina', binding_energy=-8.4, num_poses=1,
        poses=[docking.Pose(rank=1, energy=-8.4, x=0, y=0, z=0)])
    assert 'kcal/mol' in docking.format_docking_report(vina)


def test_builtin_dock_rejects_invalid_pose_count(tmp_path):
    rec = _shell_receptor(tmp_path)
    lig = _small_ligand(tmp_path)
    with pytest.raises(ValueError):
        docking._builtin_dock(rec, lig, num_poses=0)
