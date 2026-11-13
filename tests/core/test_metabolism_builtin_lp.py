"""The built-in linear program: what it solves, and what it says when it cannot.

`_builtin_fba` is the engine every FBA call takes on this machine (COBRApy is not
installed), so each of its answers is the answer a user sees.
"""
import warnings

import numpy as np
import pytest

from biosuite.core import metabolism as mb
from biosuite.core.metabolism import _builtin_fba


def test_the_first_reaction_is_the_objective_and_it_is_maximised():
    """S = [[1, -1], [0, 1]] forces x = (0, 0); a free reaction shows the sign.

    The bug this replaced minimised the objective instead, which drove a free
    reaction to its lower bound and reported that as the best solution.
    """
    result = _builtin_fba(stoich_matrix=[[0.0]])
    assert result.engine == "builtin"
    assert result.objective_value == pytest.approx(1000.0)
    assert result.reaction_count == 1 and result.metabolite_count == 1
    assert result.fluxes == {"R0": 1000.0}


def test_a_forced_zero_flux_is_reported_as_zero_not_negative_zero():
    """Measured before the fix: objective was -0.0 and printed 'objective=-0.0000'.

    The negation the maximisation needs leaked its sign onto a value that is
    exactly zero, so the report showed a flux nobody can have.
    """
    result = _builtin_fba(stoich_matrix=[[1.0], [1.0]])
    assert result.objective_value == 0.0
    assert not str(result.message).startswith("FBA solved: objective=-0")
    assert str(result.message) == "FBA solved: objective=0.0000 (reaction 0 maximised)"


def test_a_plain_python_list_is_accepted_as_a_matrix():
    result = _builtin_fba(stoich_matrix=[[1.0, 0.0], [0.0, 1.0]])
    assert result.reaction_count == 2 and result.metabolite_count == 2


def test_fluxes_are_rounded_to_four_places_and_stay_floats():
    result = _builtin_fba(stoich_matrix=[[1.0, -1.0 / 3]])
    assert all(isinstance(v, float) for v in result.fluxes.values())
    assert result.fluxes["R1"] == pytest.approx(1000.0)
    assert result.fluxes["R0"] == pytest.approx(1000.0 / 3, abs=1e-3)


def test_the_bounds_that_were_given_are_the_bounds_that_are_used():
    # A zero row leaves the reaction unconstrained by the steady state, so the
    # bounds are the only thing deciding how far the objective can go.
    assert _builtin_fba(stoich_matrix=[[0.0]], flux_bounds=(-5, 7)).objective_value == \
        pytest.approx(7.0)
    assert _builtin_fba(stoich_matrix=[[0.0]], flux_bounds=(0.0, 2.5)).objective_value == \
        pytest.approx(2.5)
    assert _builtin_fba(stoich_matrix=[[1.0]]).objective_value == pytest.approx(0.0)


@pytest.mark.parametrize("bad,expected", [
    ([[1.0, 2.0], [3.0]], "rectangular grid of numbers"),
    ([], "2-dimensional"),
    (["x"], "rectangular grid of numbers"),
])
def test_a_matrix_that_is_not_a_grid_is_described_by_its_own_shape(bad, expected):
    """numpy's complaint was about array construction, not about the caller's input.

    "inhomogeneous shape after 1 dimensions" is true but unactionable for someone
    who built a list of reaction rows.
    """
    with pytest.raises(ValueError) as exc:
        _builtin_fba(stoich_matrix=bad)
    assert expected in str(exc.value)


def test_an_infeasible_system_reports_the_solver_verdict_rather_than_nothing():
    """Measured before the fix: the caller's matrix was discarded and the answer
    was the string "No model provided"."""
    result = _builtin_fba(stoich_matrix=[[1.0, 1.0]], flux_bounds=(2.0, 3.0))
    assert result.objective_value is None
    assert result.reaction_count == 2 and result.metabolite_count == 1
    assert "FBA found no optimum" in result.message
    assert "Infeasible" in result.message


def test_a_matrix_the_solver_refuses_is_named_as_the_cause(monkeypatch):
    import scipy.optimize

    def reject(*args, **kwargs):
        raise ValueError("zero-size array to reduction")

    monkeypatch.setattr(scipy.optimize, "linprog", reject)
    result = _builtin_fba(stoich_matrix=[[1.0, 0.0]])
    assert result.objective_value is None
    assert "the solver rejected this matrix" in result.message
    assert "zero-size array" in result.message


def test_a_solver_that_raises_something_unexpected_is_not_swallowed(monkeypatch):
    """The old `except Exception: pass` turned any crash into "No model provided".

    A bug of our own — a wrong argument shape, a missing scipy — has to reach the
    caller, not arrive as a message about the caller's input.
    """
    import scipy.optimize

    def explode(*args, **kwargs):
        raise RuntimeError("no HiGHS on this machine")

    monkeypatch.setattr(scipy.optimize, "linprog", explode)
    with pytest.raises(RuntimeError):
        _builtin_fba(stoich_matrix=[[1.0, 0.0]])


def test_a_matrix_with_no_reaction_column_cannot_be_optimised():
    result = _builtin_fba(stoich_matrix=np.zeros((2, 0)))
    assert result.objective_value is None
    assert "no reaction columns" in result.message


@pytest.mark.parametrize("bounds", [(5, 1), "x", None, (float("nan"), 1), (1,), (1, 2, 3),
                                     (True, 2)])
def test_bounds_that_are_not_a_usable_pair_are_refused_before_the_solver(bounds):
    with pytest.raises(ValueError) as exc:
        _builtin_fba(stoich_matrix=[[1.0]], flux_bounds=bounds)
    assert "flux_bounds" in str(exc.value)


def test_no_arguments_at_all_says_what_to_pass():
    result = _builtin_fba()
    assert result.objective_value is None and result.reaction_count == 0
    assert "pass a stoichiometric matrix or an SBML file" in result.message


def test_run_fba_warns_about_the_missing_engine_when_a_matrix_is_solved():
    assert mb.HAS_COBRA is False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mb.run_fba(stoich_matrix=[[1.0]])
    messages = [str(w.message) for w in caught]
    assert any("COBRApy not found" in m for m in messages), messages
    assert not any("did not solve" in m for m in messages), messages


def test_run_fba_warns_nothing_when_nothing_was_asked_of_an_engine():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mb.run_fba()
    assert [w for w in caught] == []
    assert "No model was given" in result.message


def test_the_objective_of_an_unsolved_run_stays_unset_rather_than_zero():
    """`objective_value=0.0` is a flux result; None means no LP ever ran.

    The default used to be 0.0, so a parse-only call printed "Objective value:
    0.0000" — a viable network that produces nothing, instead of no answer.
    """
    assert mb.FluxResult(engine="builtin").objective_value is None
    assert _builtin_fba().objective_value is None
