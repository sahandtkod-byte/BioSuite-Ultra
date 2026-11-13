"""`format_flux_report`: the text a user reads as the result of an FBA run.

Every line here is a claim about a model, so a number that was never computed, a
list cut short without saying so, and a count of genes that is never printed are
all things this function has to get right.
"""
import pytest

from biosuite.core import metabolism as mb
from biosuite.core.metabolism import FluxResult, format_flux_report


def _report(**kwargs):
    return format_flux_report(FluxResult(**kwargs)).splitlines()


def test_a_solved_run_is_reported_with_engine_counts_and_objective():
    assert _report(engine="cobra", reaction_count=1_149, metabolite_count=727,
                   gene_count=529, objective_value=0.873928,
                   fluxes={"BIOMASS_Ec_iJO1366_core_53p95M": 0.8739, "G6PDH2r": -3.1,
                           "ATPM": 3.15}) == [
        "=== Flux Balance Analysis Report ===",
        "Engine: cobra",
        "Reactions: 1149",
        "Metabolites: 727",
        "Objective value: 0.8739",
        "Genes: 529",
        "Non-zero fluxes: 3",
        "  ATPM: 3.1500",
        "  G6PDH2r: -3.1000",
        "  BIOMASS_Ec_iJO1366_core_53p95M: 0.8739",
    ]


def test_the_counts_are_grouped_with_thousands_separators():
    lines = _report(engine="builtin", reaction_count=12345, metabolite_count=67890)
    assert lines[2] == "Reactions: 12345"
    assert lines[3] == "Metabolites: 67890"


def test_an_objective_that_was_never_computed_is_not_printed_as_zero():
    """Measured before the fix: "Objective value: 0.0000" under a parse-only run.

    A zero objective is a biological statement — the network cannot grow — while
    "not computed" is a statement about what this call actually did.
    """
    assert _report(engine="builtin", message="No model was given") == [
        "=== Flux Balance Analysis Report ===", "Engine: builtin", "Reactions: 0",
        "Metabolites: 0", "Objective value: not computed", "",
        "Note: No model was given"]


def test_a_flux_of_exactly_zero_still_prints_as_zero_not_as_nothing():
    assert _report(engine="builtin", objective_value=0.0)[4] == "Objective value: 0.0000"


def test_a_negative_zero_objective_is_printed_without_its_sign():
    assert _report(engine="builtin", objective_value=-0.0)[4] == "Objective value: 0.0000"


def test_the_gene_line_appears_only_when_genes_were_counted():
    assert any(line.startswith("Genes:") for line in _report(engine="cobra", gene_count=7))
    assert not any(line.startswith("Genes:")
                   for line in _report(engine="builtin", gene_count=0))


def test_fluxes_smaller_than_the_threshold_are_not_reported_as_non_zero():
    lines = _report(engine="builtin", objective_value=1.0,
                    fluxes={"BIG": 0.5, "tiny": 0.0005, "zero": 0.0, "minus": -0.0009})
    assert "Non-zero fluxes: 1" in lines
    assert [line for line in lines if line.startswith("  ")] == ["  BIG: 0.5000"]


def test_only_the_ten_largest_fluxes_are_listed_and_the_rest_are_counted():
    """Measured before the fix: "Non-zero fluxes: 42" above a list of ten.

    The number the report gave and the number the reader could see were two
    different things, with nothing to say which one was the truth.
    """
    fluxes = {f"R{i}": 1.0 + i for i in range(42)}
    lines = _report(engine="cobra", objective_value=1.0, reaction_count=42,
                    metabolite_count=10, fluxes=fluxes)
    listed = [line for line in lines if line.startswith("  R")]
    assert len(listed) == 10
    assert listed[0] == "  R41: 42.0000"
    assert listed[-1] == "  R32: 33.0000"
    assert lines[-1] == "... and 32 more flux(es)."


def test_exactly_ten_fluxes_are_listed_without_a_truncation_line():
    lines = _report(engine="builtin", objective_value=1.0,
                    fluxes={f"R{i}": float(i + 1) for i in range(10)})
    assert not any("more flux" in line for line in lines)
    assert len([line for line in lines if line.startswith("  R")]) == 10


def test_ranking_is_by_magnitude_then_name_so_a_tie_has_one_order():
    lines = _report(engine="builtin", objective_value=1.0,
                    fluxes={"b": -2.0, "a": 2.0, "c": 1.0})
    listed = [line for line in lines if line.startswith("  ")]
    assert listed == ["  a: 2.0000", "  b: -2.0000", "  c: 1.0000"]


def test_no_fluxes_at_all_adds_no_block():
    lines = _report(engine="builtin", reaction_count=3, metabolite_count=2,
                    objective_value=1.0)
    assert not any("Non-zero fluxes" in line for line in lines)


def test_an_empty_message_adds_no_note():
    assert _report(engine="builtin", objective_value=1.0, message="")[-1] == \
        "Objective value: 1.0000"


def test_the_note_is_the_last_line_so_it_reads_as_an_afterword():
    lines = _report(engine="builtin", objective_value=1.0,
                    message="FBA found no optimum: infeasible")
    assert lines[-2] == "" and lines[-1] == "Note: FBA found no optimum: infeasible"


def test_a_real_builtin_solve_is_reported_end_to_end():
    result = mb._builtin_fba(stoich_matrix=[[1.0], [1.0]])
    lines = format_flux_report(result).splitlines()
    assert lines[4] == "Objective value: 0.0000"
    assert lines[-1] == "Note: FBA solved: objective=0.0000 (reaction 0 maximised)"


def test_the_engine_name_is_printed_as_the_producer_that_answered():
    assert _report(engine="cobra")[1] == "Engine: cobra"
    assert _report(engine="builtin")[1] == "Engine: builtin"
