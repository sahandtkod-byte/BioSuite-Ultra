"""biosuite.core.md_simulation: the report a user reads.

The report has one duty: to print what was measured and nothing else.  A run that
never happened used to print 'Potential E : 0.00 kcal/mol' and
'Temperature : 300.0 K' — two numbers that are the dataclass's defaults, not
observations — and a reader had no way to tell them from a real result.
"""
import pytest

from biosuite.core import md_simulation as md


def test_a_run_that_did_not_happen_prints_no_invented_numbers():
    result = md.run_simulation("/nonexistent/file.pdb")
    text = md.format_md_report(result)
    assert "Potential E   : (not run)" in text
    assert "Kinetic E     : (not run)" in text
    assert "Temperature   : (not run)" in text
    assert "0.00 kcal/mol" not in text
    assert "300.0 K" not in text


def test_a_header_only_structure_is_reported_as_not_run():
    result = md.MDSimulationResult(engine="builtin",
                                   message="No atoms found in x.pdb.")
    text = md.format_md_report(result)
    assert "Total steps   : 0" in text
    assert "Temperature   : (not run)" in text
    assert "RMSD" not in text
    assert "Note          : No atoms found in x.pdb." in text


def test_a_complete_result_prints_every_measurement():
    result = md.MDSimulationResult(
        engine="builtin", steps=1200, energy=-512.3456, kinetic_energy=100.789,
        temperature=311.44, rmsd=[0.1, 0.25], radius_gyration=[7.5, 8.25],
        output_pdb="/tmp/x.pdb", message="done")
    text = md.format_md_report(result)
    assert "Engine        : builtin" in text
    assert "Total steps   : 1200" in text
    assert "Potential E   : -512.35 kcal/mol" in text
    assert "Kinetic E     : 100.79 kcal/mol" in text
    assert "Temperature   : 311.4 K" in text
    assert "RMSD          : 0.250" in text
    assert "Rg (final)    : 8.250" in text
    assert "Rg (initial)  : 7.500" in text
    assert "Output PDB    : /tmp/x.pdb" in text
    assert "Note          : done" in text


def test_the_initial_radius_is_shown_only_when_there_is_a_second_sample():
    one = md.format_md_report(md.MDSimulationResult(engine="builtin",
                                                    radius_gyration=[7.5]))
    assert "Rg (final)    : 7.500" in one
    assert "Rg (initial)" not in one


def test_the_border_closes_the_report():
    lines = md.format_md_report(md.MDSimulationResult(engine="none")).splitlines()
    assert "Molecular Dynamics Report" in lines[0]
    assert lines[-1] == "═" * 40
    assert lines[-2] == "  Temperature   : (not run)"


def test_a_report_without_a_note_or_an_output_path_omits_both_lines():
    text = md.format_md_report(md.MDSimulationResult(engine="builtin", energy=-1.0,
                                                     kinetic_energy=2.0,
                                                     temperature=300.0))
    assert "Note" not in text
    assert "Output PDB" not in text
    assert "Potential E   : -1.00 kcal/mol" in text


def test_a_single_float_alone_is_still_formatted_to_two_places():
    """Half-measurements print the ones that exist and mark the ones that do not."""
    text = md.format_md_report(md.MDSimulationResult(engine="openmm", energy=-0.004))
    assert "Potential E   : -0.00 kcal/mol" in text
    assert "Kinetic E     : (not run)" in text
    assert "Temperature   : (not run)" in text


def test_the_engine_name_is_printed_as_the_run_reported_it():
    for engine in ("builtin", "openmm", "none", "coconut"):
        text = md.format_md_report(md.MDSimulationResult(engine=engine))
        assert f"Engine        : {engine}" in text
