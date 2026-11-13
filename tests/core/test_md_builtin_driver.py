"""biosuite.core.md_simulation: the built-in minimise-then-integrate driver.

What a caller reads out of a run is decided here: the step count, the sampling of
temperature and radius of gyration, the default output name, and the message that
says what was actually asked for.  Three separate defects lived in these forty
lines — the last MD steps were never sampled, so the "final" temperature was the
one from step 10 of a 15-step run; the thermostat coupling time was a literal
inside the loop and reached no report; and a run of 1,000 integration steps wrote
its result to a file called '_minimized.pdb'.
"""
import os

import numpy as np
import pytest

from biosuite.core import md_simulation as md


def atom_line(serial, x):
    """One strict 78-column CA record; the 1.5 A spacing makes its neighbours bonded."""
    return (f"{'ATOM  ':<6s}{serial:>5d} {'CA':<4s} {'ALA':>3s} A{1:>4d}    "
            f"{x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00          {'C':>2s}")


@pytest.fixture
def chain(tmp_path):
    path = tmp_path / "chain.pdb"
    path.write_text("\n".join(atom_line(i + 1, i * 1.5) for i in range(6)) + "\nEND\n",
                    encoding="utf-8")
    return str(path)


@pytest.fixture
def out(tmp_path):
    return str(tmp_path / "final.pdb")


def run(chain, out, **kwargs):
    kwargs.setdefault("steps", 5)
    kwargs.setdefault("minimize_steps", 3)
    kwargs.setdefault("tool", "builtin")
    return md.run_simulation(chain, out, **kwargs)


def test_the_step_count_is_minimisation_plus_integration(chain, out):
    result = run(chain, out, steps=15, minimize_steps=3)
    assert result.engine == "builtin"
    assert result.steps == 18
    assert len(result.energy_history) == 19, "one entry per iteration plus the start"


def test_the_final_md_step_is_always_sampled(chain, out):
    """The old rule sampled step % 10, so a 15-step run reported step 10's heat."""
    result = run(chain, out, steps=3, minimize_steps=2)
    assert len(result.temperature_history) == 2, "the initial draw and the last step"
    assert len(result.radius_gyration) == 2
    assert result.temperature == pytest.approx(result.temperature_history[-1])


def test_sampling_still_happens_every_ten_steps(chain, out):
    result = run(chain, out, steps=15, minimize_steps=3)
    assert len(result.temperature_history) == 3, "initial, step 10, step 15"
    assert len(result.radius_gyration) == 3


def test_the_message_carries_the_parameters_the_run_used(chain, out):
    result = run(chain, out, steps=4, minimize_steps=2, dt=0.5, cutoff=8.0,
                 temperature=310.0, tau=50.0, seed=99)
    assert result.message == (
        "Minimised 2 steps, MD 4 steps (dt=0.5 fs, cutoff=8.0 \u00c5, "
        "tau=50.0 fs), T_target=310.0 K, seed=99"
    )


def test_tau_is_a_parameter_of_its_own_and_not_a_literal_in_the_loop(chain, out):
    """The coupling time used to be hard-coded at 100 fs inside the MD loop."""
    assert "tau=100.0 fs" in run(chain, out, steps=1, minimize_steps=1).message
    assert "tau=1.0 fs" in run(chain, out, steps=1, minimize_steps=1, tau=1.0).message


def test_an_unseeded_run_does_not_claim_a_seed(chain, out):
    assert "seed" not in run(chain, out, steps=1, minimize_steps=1).message


def test_the_same_seed_reproduces_the_run(chain, out):
    a = run(chain, out, steps=12, minimize_steps=4, seed=7)
    b = run(chain, out, steps=12, minimize_steps=4, seed=7)
    assert a.temperature == b.temperature
    assert a.rmsd == b.rmsd
    assert a.energy_history == b.energy_history
    assert a.radius_gyration == b.radius_gyration


def test_a_different_seed_gives_a_different_run(chain, out):
    a = run(chain, out, steps=12, minimize_steps=4, seed=7)
    b = run(chain, out, steps=12, minimize_steps=4, seed=8)
    assert a.temperature != b.temperature


def test_the_run_measures_something(chain, out):
    result = run(chain, out, steps=10, minimize_steps=5, temperature=300.0, seed=1)
    assert result.energy is not None and np.isfinite(result.energy)
    assert result.kinetic_energy is not None and np.isfinite(result.kinetic_energy)
    assert result.temperature > 0
    assert len(result.rmsd) == 1 and result.rmsd[0] >= 0.0
    assert result.radius_gyration[0] > 0


def test_the_output_file_round_trips(chain, out):
    result = run(chain, out, steps=6, minimize_steps=2, seed=3)
    assert os.path.exists(out)
    back = md.parse_pdb(out)
    assert back.n_atoms == 6
    assert np.isfinite(back.coordinates).all()
    assert np.max(np.abs(back.coordinates)) < 1000.0, "still inside the PDB columns"
    assert result.output_pdb == out


def test_a_trajectory_is_not_named_as_a_minimisation(chain, tmp_path):
    result = md.run_simulation(chain, steps=20, minimize_steps=5, tool="builtin")
    assert os.path.basename(result.output_pdb) == "chain_md_final.pdb"
    assert (tmp_path / "chain_md_final.pdb").exists()


def test_a_pure_minimisation_is_named_as_one(chain, tmp_path):
    result = md.run_simulation(chain, steps=0, minimize_steps=5, tool="builtin")
    assert os.path.basename(result.output_pdb) == "chain_minimized.pdb"


def test_the_input_file_is_never_the_default_output(chain, tmp_path):
    before = open(chain, encoding="utf-8").read()
    md.run_simulation(chain, steps=1, minimize_steps=1, tool="builtin")
    assert open(chain, encoding="utf-8").read() == before


def test_a_file_with_no_atoms_reports_the_path_it_tried(tmp_path, out):
    empty = tmp_path / "empty.pdb"
    empty.write_text("END\n", encoding="utf-8")
    result = md.run_simulation(str(empty), out, tool="builtin")
    assert result.engine == "builtin"
    assert result.message == f"No atoms found in {empty}."
    assert result.steps == 0
    assert result.energy is None
    assert result.temperature is None
    assert not os.path.exists(out), "nothing was simulated, nothing was written"


@pytest.mark.parametrize("steps", [-1, -1000])
def test_a_negative_md_step_count_is_refused(chain, out, steps):
    with pytest.raises(ValueError, match="steps cannot be negative"):
        run(chain, out, steps=steps)


@pytest.mark.parametrize("value", [1.0, "5", None, True])
def test_a_step_count_that_is_not_an_integer_count_is_refused(chain, out, value):
    with pytest.raises(ValueError, match="must be an integer"):
        md._builtin_simulate(chain, out, steps=value)
    with pytest.raises(ValueError, match="must be an integer"):
        md._builtin_simulate(chain, out, minimize_steps=value)


def test_a_negative_minimisation_budget_is_refused(chain, out):
    with pytest.raises(ValueError, match="minimize_steps cannot be negative"):
        run(chain, out, minimize_steps=-5)


@pytest.mark.parametrize("dt", [0, 0.0, -0.5])
def test_a_timestep_that_cannot_integrate_is_refused(chain, out, dt):
    with pytest.raises(ValueError, match="dt must be a positive timestep"):
        run(chain, out, dt=dt)


@pytest.mark.parametrize("cutoff", [0, 0.0, -10.0])
def test_a_cutoff_that_cannot_select_pairs_is_refused(chain, out, cutoff):
    with pytest.raises(ValueError, match="cutoff must be a positive distance"):
        run(chain, out, cutoff=cutoff)


@pytest.mark.parametrize("tau", [0, 0.0, -1.0])
def test_a_coupling_time_that_cannot_coupling_is_refused(chain, out, tau):
    with pytest.raises(ValueError, match="tau must be a positive coupling time"):
        run(chain, out, tau=tau)


@pytest.mark.parametrize("temperature", [-1.0, -300.0])
def test_a_negative_target_temperature_is_refused_at_the_door(chain, out, temperature):
    """sqrt(kB * -T / m) used to be nan, and nan reached the printed report."""
    with pytest.raises(ValueError, match="temperature cannot be negative"):
        run(chain, out, temperature=temperature)


def test_zero_kelvin_is_a_still_system_not_an_error(chain, out):
    """T = 0 gives sqrt(kB*0/m) = 0: a legitimate still start, not a nan."""
    result = run(chain, out, steps=0, minimize_steps=2, temperature=0.0)
    assert result.temperature == 0.0
    assert result.kinetic_energy == 0.0
    assert result.temperature_history == [0.0]
    assert result.rmsd == [0.0], "nothing moves without velocities"
