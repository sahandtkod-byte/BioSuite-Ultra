"""biosuite.core.md_simulation: which engine a request gets.

Three promises are tested here: an unknown engine name is refused instead of
quietly becoming 'builtin', an engine the user named explicitly is never
substituted, and a fallback that does happen is announced.  The failure channel
between the wrapper and this dispatcher is a returned reason, not a string the
caller pattern-matches.
"""
import logging

import numpy as np
import pytest

from biosuite.core import md_simulation as md


def atom_line(serial=1, name="CA", x=0.0):
    return (f"{'ATOM  ':<6s}{serial:>5d} {name:<4s} {'ALA':>3s} A{1:>4d}    "
            f"{x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00          {'C':>2s}")


@pytest.fixture
def pdb_file(tmp_path):
    path = tmp_path / "in.pdb"
    path.write_text("\n".join([atom_line(1, x=0.0), atom_line(2, x=1.5)]) + "\nEND\n",
                    encoding="utf-8")
    return str(path)


@pytest.fixture
def stub_engine(monkeypatch):
    """Replace the OpenMM call itself, keeping the (result, reason) contract."""
    calls = []

    def install(result=None, failure=None):
        def fake(pdb_file_arg, output_pdb=None, steps=1000, temperature=300):
            calls.append((pdb_file_arg, output_pdb, steps, temperature))
            return result, failure

        monkeypatch.setattr(md, "HAS_OPENMM", True)
        monkeypatch.setattr(md, "_openmm_simulate", fake)
        return calls
    return install


def test_only_the_three_documented_engines_are_accepted(pdb_file):
    assert md.TOOLS == ("auto", "openmm", "builtin")
    for tool in md.TOOLS:
        assert md.run_simulation(pdb_file, tool=tool, steps=1, minimize_steps=1)


@pytest.mark.parametrize("tool", ["opnevm", "BuiltIn", "AUTO", "", None, "openmm "])
def test_a_misspelt_engine_is_refused_instead_of_running_another_one(pdb_file, tool):
    """Measured before the fix: tool='opnevm' silently produced a built-in run."""
    with pytest.raises(ValueError, match=r"tool must be one of "
                                         r"\('auto', 'openmm', 'builtin'\)"):
        md.run_simulation(pdb_file, tool=tool)


def test_a_missing_file_names_the_file_it_was_given(tmp_path):
    absent = str(tmp_path / "absent.pdb")
    result = md.run_simulation(absent)
    assert result.engine == "none"
    assert result.message == f"File not found: {absent}"


def test_a_directory_is_reported_as_one_rather_than_crashing_the_menu(tmp_path):
    """os.path.exists() was true for a directory and open() raised PermissionError."""
    result = md.run_simulation(str(tmp_path))
    assert result.engine == "none"
    assert result.message == f"File is a directory: {tmp_path}"


def test_a_refused_run_reports_no_measurement(pdb_file, tmp_path):
    result = md.run_simulation(str(tmp_path / "absent.pdb"))
    assert result.steps == 0
    assert result.energy is None
    assert result.kinetic_energy is None
    assert result.temperature is None


def test_without_openmm_auto_uses_the_builtin_engine(pdb_file, monkeypatch):
    monkeypatch.setattr(md, "HAS_OPENMM", False)
    result = md.run_simulation(pdb_file, steps=2, minimize_steps=2, tool="auto")
    assert result.engine == "builtin"
    assert result.steps == 4


def test_without_openmm_an_explicit_request_is_answered_with_the_reason(
        pdb_file, monkeypatch):
    """tool='openmm' used to fall through and run the built-in engine anyway."""
    monkeypatch.setattr(md, "HAS_OPENMM", False)
    result = md.run_simulation(pdb_file, steps=2, tool="openmm")
    assert result.engine == "openmm"
    assert result.message == f"OpenMM could not run {pdb_file}: openmm is not installed"
    assert result.energy is None, "no simulation happened, so nothing was measured"


def test_an_explicit_openmm_failure_is_not_replaced_by_a_silent_fallback(
        pdb_file, stub_engine):
    calls = stub_engine(failure="RuntimeError: no CUDA device")
    result = md.run_simulation(pdb_file, steps=7, tool="openmm")
    assert result.engine == "openmm"
    assert result.message == f"OpenMM could not run {pdb_file}: " \
                             f"RuntimeError: no CUDA device"
    assert len(calls) == 1
    assert result.steps == 0, "the built-in engine never ran"


def test_auto_falls_back_to_the_builtin_engine_and_announces_it(
        pdb_file, tmp_path, stub_engine, caplog):
    calls = stub_engine(failure="ValueError: missing parameters")
    out = str(tmp_path / "o.pdb")
    with caplog.at_level(logging.WARNING, logger="biosuite.core.md_simulation"):
        result = md.run_simulation(pdb_file, out, steps=2, minimize_steps=1,
                                   tool="auto")
    assert result.engine == "builtin"
    assert calls == [(pdb_file, out, 2, 300)]
    assert "OpenMM failed (ValueError: missing parameters)" in caplog.text
    assert "built-in engine" in caplog.text
    assert "coarse-grained" in caplog.text


def test_auto_returns_the_openmm_result_when_it_succeeds(pdb_file, tmp_path,
                                                         stub_engine):
    winner = md.MDSimulationResult(engine="openmm", steps=9, energy=-3.5,
                                   kinetic_energy=1.5, temperature=300.0,
                                   output_pdb=str(tmp_path / "o.pdb"))
    calls = stub_engine(result=winner)
    result = md.run_simulation(pdb_file, str(tmp_path / "o.pdb"), steps=9,
                               tool="auto")
    assert result is winner
    assert len(calls) == 1


def test_the_builtin_arguments_reach_the_builtin_engine(pdb_file, tmp_path):
    out = str(tmp_path / "o.pdb")
    result = md.run_simulation(pdb_file, out, steps=4, minimize_steps=2,
                               tool="builtin", dt=0.25, cutoff=7.5, tau=20.0,
                               temperature=320.0, seed=5)
    assert "dt=0.25 fs" in result.message
    assert "cutoff=7.5" in result.message
    assert "tau=20.0 fs" in result.message
    assert "T_target=320.0 K" in result.message
    assert result.temperature_history[0] > 0.0, "the draw is reported, not invented"


def test_the_dispatcher_passes_the_file_and_request_through(pdb_file, tmp_path,
                                                           stub_engine):
    calls = stub_engine(failure="ValueError: no")
    out = str(tmp_path / "o.pdb")
    with pytest.raises(ValueError, match="steps cannot be negative"):
        md.run_simulation(pdb_file, out, steps=-1, temperature=280, tool="builtin")
    assert calls == [], "a refused request never reaches the optional engine"


def test_the_check_reports_exactly_the_one_optional_engine(monkeypatch):
    monkeypatch.setattr(md, "HAS_OPENMM", True)
    assert md.check_md_tools() == {"openmm": True}
    monkeypatch.setattr(md, "HAS_OPENMM", False)
    assert md.check_md_tools() == {"openmm": False}
    assert set(md.check_md_tools()) == {"openmm"}
