"""Building a stoichiometric matrix from parsed reactions.

S is the object the linear program actually solves, so a species that lands in the
wrong column, or is quietly dropped, changes the fluxes without changing anything
the caller can see.
"""
import pytest

from biosuite.core.metabolism import create_stoichiometric_matrix as matrix


def test_substrates_are_negative_products_positive_and_both_scaled():
    S = matrix({"R1": {"substrates": [("A", 1)], "products": [("B", 2)]}}, ["A", "B"])
    assert S.tolist() == [[-1.0], [2.0]]


def test_the_columns_follow_the_reaction_order_of_the_dict():
    S = matrix({"R2": {"substrates": [("B", 1)]}, "R1": {"substrates": [("A", 1)]}},
               ["A", "B"])
    # Rows are metabolites, columns are reactions in dict order: column 0 is R2
    # (which consumes B) and column 1 is R1 (which consumes A).
    assert S.tolist() == [[0.0, -1.0], [-1.0, 0.0]]
    assert S.shape == (2, 2)


def test_a_species_on_both_sides_nets_out_instead_of_being_overwritten():
    """A catalyst cancels: measured before the fix the column held +1.

    The product pass assigned over the substrate pass instead of adding to it, so
    A → A was a reaction that manufactured A out of nothing.
    """
    S = matrix({"R": {"substrates": [("A", 1)], "products": [("A", 1)]}}, ["A"])
    assert S.tolist() == [[0.0]]
    S = matrix({"R": {"substrates": [("A", 3)], "products": [("A", 1)]}}, ["A"])
    assert S.tolist() == [[-2.0]]


def test_a_multi_substrate_reaction_fills_every_row_it_names():
    S = matrix({"R": {"substrates": [("A", 1), ("B", 2)], "products": [("C", 1)]}},
               ["A", "B", "C"])
    assert S.tolist() == [[-1.0], [-2.0], [1.0]]


def test_a_metabolite_the_index_does_not_hold_is_reported_not_dropped_in_silence(
        caplog):
    """Measured before the fix: the species vanished and nothing was said.

    A reaction whose every metabolite is missing becomes an all-zero column, which
    the solver then reads as a free flux — so the count of skipped references is
    the only trace of what happened.
    """
    S = matrix({"R1": {"substrates": [("zzz", 1)], "products": [("B", 1)]}}, ["A", "B"])
    assert S.tolist() == [[0.0], [1.0]]
    assert "1 species reference(s) named a metabolite outside the given list" \
        in caplog.text, caplog.text


def test_a_reaction_whose_species_are_all_unknown_leaves_an_empty_column(caplog):
    S = matrix({"R1": {"substrates": [("x", 1)]}, "R2": {"substrates": [("A", 1)]}},
               ["A"])
    assert S.tolist() == [[0.0, -1.0]]
    assert "outside the given list" in caplog.text, caplog.text


@pytest.mark.parametrize("coefficient", [-3, 0, -0.5, None, float("nan"), float("inf"),
                                          "2"])
def test_a_stoichiometry_that_is_not_a_positive_number_is_refused(coefficient):
    """Measured before the fix: `abs()` quietly repaired the sign on one side only.

    A substrate coefficient of -3 became -3 again (abs then negate) while a product
    of -3 became +3, so the same input produced two different readings depending on
    which list it was written in.
    """
    with pytest.raises(ValueError) as exc:
        matrix({"R": {"substrates": [("A", coefficient)]}}, ["A"])
    assert "stoichiometry must be a positive finite number" in str(exc.value)
    assert "coefficient of" in str(exc.value)


def test_the_offending_reaction_and_species_are_named_in_the_refusal():
    with pytest.raises(ValueError) as exc:
        matrix({"R9": {"products": [("B", -1)]}}, ["A", "B"])
    assert "reaction 'R9'" in str(exc.value)
    assert "for 'B'" in str(exc.value)


def test_a_reaction_with_neither_side_contributes_an_empty_column():
    S = matrix({"R": {}}, ["A", "B"])
    assert S.tolist() == [[0.0], [0.0]]


def test_no_reactions_is_an_empty_matrix_of_the_right_shape():
    S = matrix({}, ["A", "B"])
    assert S.shape == (2, 0)


def test_no_metabolites_is_an_empty_matrix_too():
    S = matrix({"R": {"substrates": [("A", 1)]}}, [])
    assert S.shape == (0, 1)


def test_the_parsed_sbml_shape_and_the_index_work_together():
    """The tuple shape `_parse_sbml_simple` produces is this function's input."""
    reactions = {"R1": {"substrates": [("A", 1.0)], "products": [("B", 2.0)]}}
    S = matrix(reactions, ["A", "B"])
    assert S.tolist() == [[-1.0], [2.0]]
