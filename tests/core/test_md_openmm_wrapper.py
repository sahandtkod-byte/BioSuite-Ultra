"""biosuite.core.md_simulation: the OpenMM wrapper, against a stand-in engine.

OpenMM is not installed here, so these tests drive the wrapper with a fake built
out of the shapes the real API has — including `Simulation.step`, which the
wrapper never called and therefore never had a stand-in.  What is asserted is
this project's arithmetic and control flow: the unit conversion, the integration
step count, the error channel and the file handle.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

from biosuite.core import md_simulation as md

MODULE_PATH = os.path.abspath(md.__file__)
POTENTIAL_KJ = -1234.5678
KINETIC_KJ = 500.0


def atom_line(serial=1, name="CA", x=0.0):
    return (f"{'ATOM  ':<6s}{serial:>5d} {name:<4s} {'ALA':>3s} A{1:>4d}    "
            f"{x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00          {'C':>2s}")


class Quantity:
    """What `310 * unit.kelvin` produces, and what getState returns."""

    def __init__(self, magnitude):
        self.magnitude = magnitude

    def value_in_unit(self, unit):
        return self.magnitude


class UnitTerm:
    def __init__(self, name):
        self.name = name

    def __mul__(self, magnitude):
        return Quantity(magnitude)

    __rmul__ = __mul__

    def __rtruediv__(self, magnitude):
        return Quantity(magnitude)


class Units:
    nanometers = UnitTerm("nm")
    kelvin = UnitTerm("K")
    picosecond = UnitTerm("ps")
    femtoseconds = UnitTerm("fs")
    kilojoule_per_mole = UnitTerm("kJ/mol")


def install_fake_openmm(monkeypatch, explode=None):
    """Stand in for every openmm symbol `_openmm_simulate` touches."""
    recorder = {}

    class PDBFile:
        def __init__(self, path):
            recorder["read"] = path
            self.topology = "topology"
            self.positions = Quantity(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))

        @staticmethod
        def writeFile(topology, positions, handle):
            recorder["written"] = (topology, positions, handle.closed)
            handle.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
            handle.write("END\n")

    class ForceField:
        def __init__(self, name):
            recorder["forcefield"] = name

        def createSystem(self, topology, nonbondedMethod=None, constraints=None):
            recorder["system"] = (topology, nonbondedMethod, constraints)
            return "system"

    class Modeller:
        def __init__(self, topology, positions):
            recorder["modeller"] = (topology, positions)
            self.positions = positions
            self.topology = topology

        def addSolvent(self, forcefield, model=None, padding=None):
            recorder["solvent"] = (model, padding.magnitude)

    class LangevinMiddleIntegrator:
        def __init__(self, temperature, friction, step):
            recorder["integrator"] = (temperature.magnitude, friction.magnitude,
                                       step.magnitude)

    class State:
        def getPotentialEnergy(self):
            return Quantity(POTENTIAL_KJ)

        def getKineticEnergy(self):
            return Quantity(KINETIC_KJ)

        def getPositions(self, asNumpy=False):
            return Quantity(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))

    class Context:
        def setPositions(self, positions):
            if explode is not None:
                raise RuntimeError(explode)
            recorder["positions_set"] = positions

        def getState(self, getEnergy=False, getPositions=False):
            recorder["state_requested"] = (getEnergy, getPositions)
            return State()

    class Simulation:
        def __init__(self, topology, system, integrator):
            recorder["simulation"] = (topology, system)
            self.context = Context()
            self.topology = topology

        def minimizeEnergy(self, maxIterations=0):
            recorder["minimized"] = maxIterations

        def step(self, n):
            recorder["stepped"] = n

    openmm = types.ModuleType("openmm")
    openmm.unit = Units
    openmm.LangevinMiddleIntegrator = LangevinMiddleIntegrator
    app = types.ModuleType("openmm.app")
    app.PDBFile, app.ForceField, app.Modeller = PDBFile, ForceField, Modeller
    app.Simulation, app.PME, app.HBonds = Simulation, "PME", "HBonds"
    monkeypatch.setitem(sys.modules, "openmm", openmm)
    monkeypatch.setitem(sys.modules, "openmm.app", app)
    monkeypatch.setattr(md, "HAS_OPENMM", True)
    for name, value in (("PDBFile", PDBFile), ("ForceField", ForceField),
                        ("Modeller", Modeller), ("Simulation", Simulation),
                        ("LangevinMiddleIntegrator", LangevinMiddleIntegrator),
                        ("PME", "PME"), ("HBonds", "HBonds"), ("unit", Units)):
        monkeypatch.setattr(md, name, value, raising=False)
    return recorder


@pytest.fixture
def pdb_file(tmp_path):
    path = tmp_path / "in.pdb"
    path.write_text("\n".join([atom_line(1, x=0.0), atom_line(2, x=1.5)]) + "\nEND\n",
                    encoding="utf-8")
    return str(path)


def test_openmm_is_absent_here_and_the_wrapper_says_why():
    assert md.HAS_OPENMM is False
    assert md.check_md_tools() == {"openmm": False}
    assert md._openmm_simulate("whatever.pdb") == (None, "openmm is not installed")


def test_the_module_admits_openmm_when_the_import_works(monkeypatch):
    fake = types.ModuleType("openmm")
    fake.LangevinMiddleIntegrator = object
    fake.unit = types.SimpleNamespace()
    app = types.ModuleType("openmm.app")
    for symbol in ("PME", "ForceField", "HBonds", "Modeller", "PDBFile", "Simulation"):
        setattr(app, symbol, type(symbol, (), {}))
    monkeypatch.setitem(sys.modules, "openmm", fake)
    monkeypatch.setitem(sys.modules, "openmm.app", app)
    spec = importlib.util.spec_from_file_location("biosuite.core.md_with_openmm",
                                                  MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.HAS_OPENMM is True
    assert mod.check_md_tools() == {"openmm": True}


def test_the_wrapper_runs_the_documented_pipeline(tmp_path, monkeypatch, pdb_file):
    out = str(tmp_path / "out.pdb")
    recorder = install_fake_openmm(monkeypatch)
    result, failure = md._openmm_simulate(pdb_file, output_pdb=out, steps=250,
                                          temperature=310)
    assert failure is None
    assert result.engine == "openmm"
    assert result.steps == 250
    assert result.output_pdb == out
    assert recorder["read"] == pdb_file
    assert recorder["forcefield"] == "amber14-all.xml"
    assert recorder["solvent"] == ("tip3p", 1.0)
    assert recorder["system"] == ("topology", "PME", "HBonds")
    assert recorder["integrator"] == (310, 1, 2)
    assert recorder["positions_set"] is not None
    assert recorder["state_requested"] == (True, True)
    assert recorder["minimized"] == md.OPENMM_MINIMISE_ITERATIONS


def test_the_wrapper_actually_integrates(tmp_path, monkeypatch, pdb_file):
    """minimizeEnergy alone is not a simulation; step() had never been called."""
    recorder = install_fake_openmm(monkeypatch)
    result, _f = md._openmm_simulate(pdb_file, output_pdb=str(tmp_path / "o.pdb"),
                                     steps=250)
    assert recorder["stepped"] == 250
    assert "integrated 250 steps" in result.message
    assert recorder["minimized"] == md.OPENMM_MINIMISE_ITERATIONS, \
        "the minimiser budget is not the MD step count"


def test_the_energy_is_converted_out_of_kilojoules(monkeypatch, pdb_file, tmp_path):
    install_fake_openmm(monkeypatch)
    result, _f = md._openmm_simulate(pdb_file, output_pdb=str(tmp_path / "o.pdb"),
                                     steps=10)
    assert result.energy == pytest.approx(POTENTIAL_KJ / 4.184, rel=1e-12)
    assert result.kinetic_energy == pytest.approx(KINETIC_KJ / 4.184, rel=1e-12)
    assert result.energy == pytest.approx(-295.0688, abs=1e-3)
    assert "potential energy -1234.57 kJ/mol" in result.message, \
        "the message keeps the unit OpenMM reported in, the field does not"


def test_the_reported_temperature_is_the_one_asked_for(monkeypatch, pdb_file, tmp_path):
    install_fake_openmm(monkeypatch)
    result, _f = md._openmm_simulate(pdb_file, output_pdb=str(tmp_path / "o.pdb"),
                                     steps=5, temperature=277)
    assert result.temperature == 277.0


def test_positions_are_handed_to_the_writer_as_a_quantity(monkeypatch, pdb_file,
                                                           tmp_path):
    """value_in_unit(nanometers) produced bare floats; writeFile wants a Quantity."""
    recorder = install_fake_openmm(monkeypatch)
    out = str(tmp_path / "o.pdb")
    result, _f = md._openmm_simulate(pdb_file, output_pdb=out, steps=5)
    assert isinstance(recorder["written"][1], Quantity)
    assert result.output_pdb == out
    assert os.path.exists(out)


def test_the_output_file_is_opened_and_closed_around_the_write(monkeypatch, pdb_file,
                                                               tmp_path):
    recorder = install_fake_openmm(monkeypatch)
    out = str(tmp_path / "written.pdb")
    md._openmm_simulate(pdb_file, output_pdb=out, steps=10)
    _topology, _positions, was_closed = recorder["written"]
    assert was_closed is False, "the handle is still open inside the writer"
    assert os.path.exists(out)


def test_no_output_path_derives_one_from_the_input(monkeypatch, pdb_file):
    recorder = install_fake_openmm(monkeypatch)
    result, _f = md._openmm_simulate(pdb_file, output_pdb=None, steps=5)
    assert result.output_pdb == os.path.splitext(pdb_file)[0] + "_md_final.pdb"
    assert "written" in recorder, "the built-in engine always writes; so does this one"


def test_a_zero_step_run_is_named_as_a_minimisation(monkeypatch, pdb_file):
    install_fake_openmm(monkeypatch)
    result, _f = md._openmm_simulate(pdb_file, output_pdb=None, steps=0)
    assert result.output_pdb.endswith("_minimized.pdb")
    assert result.steps == 0


@pytest.mark.parametrize("steps, temperature, message", [
    (-1, 300, "steps cannot be negative, got -1"),
    (5, -10, "temperature cannot be negative, got -10"),
])
def test_an_impossible_request_fails_through_the_error_channel(
        monkeypatch, pdb_file, steps, temperature, message):
    install_fake_openmm(monkeypatch)
    result, failure = md._openmm_simulate(pdb_file, steps=steps,
                                          temperature=temperature)
    assert result is None
    assert failure.startswith("ValueError: ")
    assert message in failure


def test_an_openmm_failure_returns_a_reason_rather_than_a_message(monkeypatch):
    """Failure detection used to be `message.startswith('OpenMM error')` in the caller."""
    recorder = install_fake_openmm(monkeypatch, explode="no CUDA device")
    result, failure = md._openmm_simulate("in.pdb", steps=5)
    assert result is None
    assert failure == "RuntimeError: no CUDA device"
    assert "minimized" not in recorder, "the run stopped where it failed"
