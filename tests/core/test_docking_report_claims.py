"""Every claim `format_docking_report` is allowed to make about a result.

The report is what the CLI prints, so its wording is the only thing between a
score and a misread number: an engine that never ran used to be printed with
Vina's unit, and a pose whose position nobody measured used to print the origin.
"""
import pytest

from biosuite.core import docking as dk


def result(engine, energy=0.0, poses=(), num_poses=0, message=""):
    return dk.DockingResult(engine=engine, binding_energy=energy, poses=list(poses),
                            num_poses=num_poses, message=message)


def pose(rank=1, energy=-5.0, x=1.0, y=2.0, z=3.0):
    return dk.Pose(rank=rank, energy=energy, x=x, y=y, z=z)


def test_the_built_in_engine_is_quoted_in_arbitrary_units_and_warned_about():
    text = dk.format_docking_report(
        result('builtin', energy=-6.5, poses=[pose()], num_poses=1))

    assert "arbitrary units" in text, text
    assert "kcal/mol" not in text, text
    assert "WARNING: heuristic contact score" in text, text


def test_vina_is_quoted_in_kcal_per_mol_without_the_heuristic_warning():
    text = dk.format_docking_report(
        result('vina', energy=-8.4, poses=[pose(energy=-8.4)], num_poses=1))

    assert "kcal/mol" in text, text
    assert "arbitrary units" not in text, text
    assert "WARNING" not in text, text


@pytest.mark.parametrize("engine", ['none', '', 'autodock', 'unknown'])
def test_an_engine_that_never_ran_is_given_no_unit(engine):
    """"0.00 kcal/mol" from a docking that did not happen reads like a measurement."""
    text = dk.format_docking_report(result(engine, message="Receptor not found: x"))

    assert "kcal/mol" not in text, text
    assert "arbitrary units" not in text, text
    assert "Best score: 0.00 (no scoring engine ran)" in text, text


def test_the_warning_follows_the_engine_not_the_number_of_poses():
    """A built-in run that found nothing still ran, and its zero is still heuristic."""
    text = dk.format_docking_report(result('builtin', energy=0.0, num_poses=0))

    assert "arbitrary units" in text and "Poses generated: 0" in text, text
    assert "Top poses" not in text, text


def test_the_pose_table_lists_at_most_three_of_the_poses():
    poses = [pose(rank=i, energy=-float(i)) for i in range(1, 6)]

    text = dk.format_docking_report(result('vina', poses=poses, num_poses=5))

    assert text.count("  Pose ") == 3, text
    assert "Pose 1:" in text and "Pose 3:" in text and "Pose 4:" not in text, text


def test_a_measured_pose_position_is_printed_to_one_decimal():
    text = dk.format_docking_report(
        result('vina', poses=[pose(x=100.04, y=2.56, z=12.345)], num_poses=1))

    assert "at (100.0, 2.6, 12.3)" in text, text


def test_a_pose_without_a_recorded_position_says_so_instead_of_printing_the_origin():
    nan = float('nan')
    text = dk.format_docking_report(
        result('vina', poses=[pose(x=nan, y=nan, z=nan)], num_poses=1))

    assert "at a position the pose file does not record" in text, text
    assert "(0.0, 0.0, 0.0)" not in text, text


def test_a_partially_measured_pose_is_not_claimed_as_measured():
    nan = float('nan')
    text = dk.format_docking_report(result('vina', poses=[pose(x=1.0, y=nan, z=3.0)],
                                           num_poses=1))

    assert "does not record" in text, text


def test_the_note_repeats_the_engine_message_and_only_when_there_is_one():
    with_message = dk.format_docking_report(result('vina', message="2 poses kept"))
    without = dk.format_docking_report(result('vina'))

    assert "Note: 2 poses kept" in with_message, with_message
    assert "Note:" not in without, without


def test_the_header_names_the_engine_and_the_pose_count():
    text = dk.format_docking_report(
        result('vina', energy=-8.4, poses=[pose(energy=-8.4)] * 2, num_poses=2))

    lines = text.splitlines()
    assert lines[0] == "=== Molecular Docking Report ===", lines
    assert lines[1] == "Engine: vina", lines
    assert lines[2] == "Poses generated: 2", lines
    assert lines[3] == "Best score: -8.40 kcal/mol", lines
