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
