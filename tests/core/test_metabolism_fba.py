"""FBA engine tests incl. list-input robustness + biomass-flux maximisation."""
import numpy as np
import pytest

from biosuite.core import metabolism as mb


def test_create_stoich_matrix_shape_and_signs():
    S = mb.create_stoichiometric_matrix({
        'R1': {'substrates': [('A', 1)], 'products': [('B', 1)]},
        'R2': {'substrates': [('B', 1)], 'products': [('C', 2)]},
    }, ['A', 'B', 'C'])
    assert S.shape == (3, 2)
    assert S[0, 0] == -1      # substrate consumed
    assert S[1, 0] == 1       # product made
    assert S[2, 1] == 2       # stoichiometric coefficient respected


def test_fba_maximises_objective_flux():
    # v0 -> metA -> v1 -> metB: steady state lets R0 hit the upper bound
    S = np.array([[1., -1., 0.], [0., 1., -1.]])
    res = mb.run_fba(stoich_matrix=S)
    assert res.engine == 'builtin'
    assert res.objective_value > 0     # pushed to the upper bound
    assert abs(res.fluxes['R0']) == pytest.approx(res.objective_value)


def test_fba_accepts_plain_python_lists():
    S = [[-1., 0.], [1., 1.]]          # plain nested list, not ndarray
    res = mb.run_fba(stoich_matrix=S)
    assert res.engine == 'builtin'
    assert res.objective_value >= 0    # used to crash with AttributeError


def test_fba_bad_matrix_shape_raises():
    with pytest.raises(ValueError):
        mb.run_fba(stoich_matrix=[1, 2, 3])


def test_flux_report_format():
    S = np.array([[1., -1., 0.], [0., 1., -1.]])
    res = mb.run_fba(stoich_matrix=S)
    rep = mb.format_flux_report(res)
    assert isinstance(rep, str) and 'Engine' in rep


def test_knockout_dispatch_no_model():
    result = mb.knockout_analysis(None, ['g1'])
    assert isinstance(result, list)
