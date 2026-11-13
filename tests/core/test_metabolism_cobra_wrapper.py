"""The COBRApy wrapper, exercised against a stand-in for the package.

COBRApy is not installed here, so this file is the only place its code paths run —
which makes the shape of the stand-in load-bearing: a fake friendlier than cobra
launders a dead branch through a green suite.
"""
import sys
import types

import pandas as pd
import pytest

from biosuite.core import metabolism as mb


class FakeGene:
    def __init__(self, gene_id, model):
        self.id = gene_id
        self.model = model

    def knock_out(self):
        self.model.knocked_out.append(self.id)


class FakeGeneCollection(list):
    """cobra's ElementList: get_by_id raises KeyError for an id it does not hold."""

    def get_by_id(self, gene_id):
        for gene in self:
            if gene.id == gene_id:
                return gene
        raise KeyError(gene_id)


class FakeSolution:
    def __init__(self, objective_value, fluxes, status="optimal"):
        self.objective_value = objective_value
        self.fluxes = pd.Series(fluxes)
        self.status = status


class FakeModel:
    def __init__(self, reaction_ids, gene_ids, solution):
        self.reactions = [types.SimpleNamespace(id=rid) for rid in reaction_ids]
        self.metabolites = [types.SimpleNamespace(id=f"m{i}") for i in range(3)]
        self.genes = FakeGeneCollection(FakeGene(g, self) for g in gene_ids)
        self._solution = solution
        self.knocked_out = []

    def optimize(self):
        return self._solution

    def __enter__(self):
        # cobra's context manager works on a copy, so a knock-out inside the block
        # is gone again on exit.  Without this the second gene would inherit the
        # first one's knock-out and every later number would be wrong.
        self._saved = list(self.knocked_out)
        return self

    def __exit__(self, *exc):
        self.knocked_out = self._saved
        return False


class FakeIO:
    def __init__(self, model=None, error=None):
        self.model, self.error = model, error
        self.reads = []

    def read_sbml_model(self, path):
        self.reads.append(path)
        if self.error:
            raise self.error
        return self.model


@pytest.fixture()
def cobra(monkeypatch):
    """Install a cobra-shaped module and report it as importable."""
    module = types.ModuleType("cobra")
    module.io = FakeIO()
    monkeypatch.setitem(sys.modules, "cobra", module)
    # The name does not exist in the module while cobra is uninstalled, which is
    # exactly the state this machine is in, so it has to be created rather than set.
    monkeypatch.setattr(mb, "cobra", module, raising=False)
    monkeypatch.setattr(mb, "HAS_COBRA", True)
    return module


def _model(solution=None, reaction_ids=("R1", "R2"), gene_ids=("g1", "g2")):
    return FakeModel(reaction_ids, gene_ids,
                     solution or FakeSolution(0.8765432, {"R1": 1.23456789, "R2": -0.5}))


def test_a_solved_model_comes_back_with_its_own_engine_and_numbers(cobra, tmp_path):
    cobra.io = FakeIO(_model())
    result, reason = mb._cobra_fba(str(tmp_path / "m.xml"))
    assert reason is None
    assert result.engine == "cobra"
    assert result.objective_value == pytest.approx(0.8765)
    assert result.fluxes == {"R1": 1.2346, "R2": -0.5}
    assert (result.reaction_count, result.metabolite_count, result.gene_count) == (2, 3, 2)
    assert result.message == "COBRApy FBA: objective=0.8765"
    assert cobra.io.reads == [str(tmp_path / "m.xml")]


def test_a_model_that_cannot_be_read_returns_the_reason_it_could_not(cobra, tmp_path,
                                                                    caplog):
    """Measured before the fix: `_cobra_fba` answered None and the reason vanished.

    run_fba then warned "COBRApy not found" on a machine that had COBRApy, because
    the only fact it had left was that the call returned nothing.
    """
    cobra.io = FakeIO(error=OSError("no sbml level here"))
    path = str(tmp_path / "broken.xml")
    with caplog.at_level("ERROR", logger="biosuite.core.metabolism"):
        result, reason = mb._cobra_fba(path)
    assert result is None
    assert "no sbml level here" in reason
    assert f"COBRApy could not read or solve {path}" in caplog.text, caplog.text


@pytest.mark.parametrize("objective", [None, float("nan"), float("inf")])
def test_a_solution_without_a_number_is_not_reported_as_zero_growth(cobra, objective):
    cobra.io = FakeIO(_model(FakeSolution(objective, {"R1": 0.0, "R2": 0.0})))
    result, reason = mb._cobra_fba("m.xml")
    assert result is None
    assert "returned no objective value" in reason


def test_the_engine_being_absent_is_itself_a_reason(monkeypatch):
    monkeypatch.setattr(mb, "HAS_COBRA", False)
    assert mb._cobra_fba("m.xml") == (None, "COBRApy is not installed")


def test_run_fba_hands_a_solved_model_back_without_warning(cobra, tmp_path):
    import warnings
    cobra.io = FakeIO(_model())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mb.run_fba(str(tmp_path / "m.xml"))
    assert result.engine == "cobra"
    assert [w for w in caught] == []


def test_run_fba_says_the_engine_refused_rather_than_that_it_is_missing(cobra, tmp_path):
    import warnings
    cobra.io = FakeIO(error=ValueError("model is inconsistent"))
    path = str(tmp_path / "m.xml")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mb.run_fba(path)
    messages = [str(w.message) for w in caught]
    assert any(f"COBRApy did not solve {path}" in m for m in messages), messages
    assert any("model is inconsistent" in m for m in messages), messages
    assert not any("not found" in m for m in messages), messages
    assert result.engine == "builtin"


def test_knockouts_are_scored_against_the_wild_type_flux(cobra, tmp_path):
    class TwoStageSolution:
        """Wild type grows, g1's knock-out does not, g2's barely changes."""

        def __init__(self):
            self.calls = 0

        def solution_for(self, model):
            if "g1" in model.knocked_out:
                return FakeSolution(0.0001, {"R1": 0.0, "R2": 0.0})
            if "g2" in model.knocked_out:
                return FakeSolution(0.9, {"R1": 0.9, "R2": 0.0})
            return FakeSolution(1.0, {"R1": 1.0, "R2": 0.0})

    model = _model(None)
    model.optimize = lambda: TwoStageSolution().solution_for(model)
    cobra.io = FakeIO(model)
    results, missing = mb._cobra_knockout("m.xml", ["g1", "g2"])
    assert missing == []
    assert [(r.gene, r.growth_reduction, r.essential) for r in results] == [
        ("g1", 99.99, True), ("g2", 10.0, False)]
    assert results[0].wild_type_flux == 1.0 and results[0].knockout_flux == 0.0001


def test_a_gene_the_model_does_not_hold_is_counted_and_the_rest_still_scores(
        cobra, caplog):
    """Measured before the fix: one bad id cost every result the loop had produced.

    The whole batch sat inside one broad handler, so a mistyped gene name came back
    as an empty list — indistinguishable from a model with no essential genes.
    """
    model = _model(FakeSolution(1.0, {"R1": 1.0, "R2": 0.0}))
    cobra.io = FakeIO(model)
    with caplog.at_level("WARNING", logger="biosuite.core.metabolism"):
        results, missing = mb._cobra_knockout("m.xml", ["g1", "no-such-gene", "g2"])
    assert missing == ["no-such-gene"]
    assert [r.gene for r in results] == ["g1", "g2"]
    assert "1 gene(s) were not in the model and were not tested: no-such-gene" \
        in caplog.text, caplog.text


@pytest.mark.parametrize("wild_type", [0.0, -1.0, None, float("nan")])
def test_a_wild_type_that_does_not_grow_leaves_no_ratio_to_reduce_by(cobra, wild_type,
                                                                    caplog):
    """Measured before the fix: reduction 0.0 and essential False.

    Zero growth has no ratio, and reporting 0 % read as "this gene is not
    essential" in a model that cannot grow at all.
    """
    model = _model(FakeSolution(wild_type, {"R1": 0.0, "R2": 0.0}))
    cobra.io = FakeIO(model)
    with caplog.at_level("WARNING", logger="biosuite.core.metabolism"):
        results, missing = mb._cobra_knockout("m.xml", ["g1"])
    assert missing == []
    assert results[0].growth_reduction is None
    assert results[0].essential is False
    assert "no gene can be shown essential" in caplog.text, caplog.text


def test_a_model_that_cannot_be_read_ends_the_knockout_run_with_no_results(
        cobra, caplog):
    cobra.io = FakeIO(error=OSError("unreadable sbml"))
    with caplog.at_level("ERROR", logger="biosuite.core.metabolism"):
        assert mb._cobra_knockout("m.xml", ["g1"]) == ([], [])
    assert "COBRApy could not prepare the knockout run" in caplog.text, caplog.text


def test_knockout_analysis_delegates_to_the_engine_and_returns_its_results(cobra):
    cobra.io = FakeIO(_model(FakeSolution(1.0, {"R1": 1.0, "R2": 0.0})))
    results = mb.knockout_analysis("m.xml", ["g1"])
    assert [r.gene for r in results] == ["g1"]


def test_knockout_analysis_without_the_engine_says_nothing_was_tested(monkeypatch,
                                                                     caplog):
    """The empty list is the same shape a run with no essential genes returns.

    The log line is what keeps "COBRApy is not installed" from being read as a
    biological result.
    """
    monkeypatch.setattr(mb, "HAS_COBRA", False)
    with caplog.at_level("WARNING", logger="biosuite.core.metabolism"):
        assert mb.knockout_analysis("m.xml", ["g1", "g2"]) == []
    assert "COBRApy is not installed, so no gene was tested" in caplog.text, caplog.text


def test_the_module_picks_cobra_up_when_the_package_is_installed():
    """The import-time branch this machine cannot take: cobra is not installed here.

    Without it, a change that stopped binding the name would look identical to this
    environment — and would send every model to the built-in solver in the field.
    """
    import inspect
    import types

    fake = types.ModuleType("cobra")
    fake.io = types.SimpleNamespace(read_sbml_model=lambda path: None)
    saved = sys.modules.get("cobra")
    sys.modules["cobra"] = fake
    try:
        fresh = types.ModuleType("metabolism_with_cobra")
        fresh.__package__ = "biosuite.core"
        source = inspect.getsource(mb)
        exec(compile(source, inspect.getfile(mb), "exec"), fresh.__dict__)
    finally:
        if saved is None:
            del sys.modules["cobra"]
        else:
            sys.modules["cobra"] = saved
    assert fresh.HAS_COBRA is True
    assert fresh.cobra is fake
    assert fresh.check_metabolism_tools() == {"cobra": True}


def test_the_import_probe_reports_absence_honestly_in_this_environment():
    assert mb.HAS_COBRA is False
    assert mb.check_metabolism_tools() == {"cobra": False}


def test_knockout_without_the_engine_returns_no_results_and_no_missing_ids(monkeypatch):
    """The early return that keeps a COBRApy-free run from touching the model."""
    monkeypatch.setattr(mb, "HAS_COBRA", False)
    assert mb._cobra_knockout("m.xml", ["g1"]) == ([], [])


def test_a_gene_collection_that_answers_none_instead_of_raising_is_still_counted(
        cobra, caplog):
    """Some cobra versions hand back a falsy object for an id they do not hold."""
    class QuietCollection(list):
        def get_by_id(self, gene_id):
            for gene in self:
                if gene.id == gene_id:
                    return gene

    model = _model(FakeSolution(1.0, {"R1": 1.0, "R2": 0.0}))
    model.genes = QuietCollection([FakeGene("g1", model)])
    cobra.io = FakeIO(model)
    with caplog.at_level("WARNING", logger="biosuite.core.metabolism"):
        results, missing = mb._cobra_knockout("m.xml", ["g1", "gone"])
    assert [r.gene for r in results] == ["g1"]
    assert missing == ["gone"]
    assert "1 gene(s) were not in the model" in caplog.text, caplog.text


def test_an_empty_gene_list_is_not_reported_as_a_missing_engine(monkeypatch, caplog):
    monkeypatch.setattr(mb, "HAS_COBRA", False)
    with caplog.at_level("WARNING", logger="biosuite.core.metabolism"):
        assert mb.knockout_analysis("m.xml", []) == []
    assert caplog.text == ""
