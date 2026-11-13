"""What a sequence has to look like before an epitope predictor may read it.

Every property scale in this module is keyed by an upper-case one-letter code, so
a sequence that is not upper-case and whitespace-free does not fail — it silently
scores each residue against the *default* of the corresponding lookup.  The API
takes the sequence straight from a JSON body, and that is where this shows up.
"""
import pytest

from biosuite.core import epitope as ep

UPPER = "MAVPKRRTSRKEFVGIPSRLMNHQFSDLLSLFMESSGSTSPSINSTQIPSRKNHTWDEQQRGQIR"


def test_a_lower_case_sequence_answers_the_same_peptides_as_its_upper_case_form():
    """Measured on the old code: 0 T-cell epitopes against 50 for the same protein."""
    lowered = ep.predict_t_cell_epitopes(UPPER.lower(), max_results=10 ** 6)
    uppered = ep.predict_t_cell_epitopes(UPPER, max_results=10 ** 6)

    assert [e.peptide for e in lowered] == [e.peptide for e in uppered]
    assert [round(e.score, 12) for e in lowered] == [round(e.score, 12) for e in uppered]
    assert len(uppered) > 0, "the reference run must find something"


def test_a_pasted_fasta_does_not_report_a_line_break_as_part_of_an_epitope():
    pasted = UPPER[:20] + "\n" + UPPER[20:45] + "\r\n" + UPPER[45:]

    results = ep.predict_t_cell_epitopes(pasted, max_results=10 ** 6)

    assert results, pasted
    assert not any(char in e.peptide for e in results
                   for char in (" ", "\n", "\r", "\t")), results[:3]
    assert [e.peptide for e in results] == \
        [e.peptide for e in ep.predict_t_cell_epitopes(UPPER, max_results=10 ** 6)]


def test_the_coordinates_a_window_is_reported_at_still_point_at_its_peptide():
    pasted = UPPER[:20] + "\n" + UPPER[20:]

    for entry in ep.predict_t_cell_epitopes(pasted, max_results=10 ** 6):
        assert entry.peptide == UPPER[entry.start:entry.end], entry.to_dict()


def test_letters_no_scale_covers_are_scored_neutrally_and_said_aloud(caplog):
    """X and * are real characters in UniProt sequences, so they are not an error."""
    with caplog.at_level("WARNING", logger="biosuite.core.epitope"):
        results = ep.predict_t_cell_epitopes("VILMWWQQVILMX*", max_results=10 ** 6)

    assert results, results
    assert "2 residue position(s)" in caplog.text, caplog.text
    assert "(*, X)" in caplog.text, caplog.text


def test_the_warning_names_the_parameter_that_was_normalised(caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.epitope"):
        ep.predict_b_cell_epitopes("AAAAXBBB", window_size=4)

    assert "sequence:" in caplog.text, caplog.text


def test_a_clean_sequence_is_not_apologised_about(caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.epitope"):
        ep.predict_t_cell_epitopes(UPPER)

    assert "no property scale covers" not in caplog.text, caplog.text


def _key(results):
    """The part of a prediction that normalisation could plausibly change."""
    return [(e.peptide, round(e.score, 12), e.start, e.end) for e in results]


@pytest.mark.parametrize("predictor", [
    ep.predict_t_cell_epitopes, ep.predict_b_cell_epitopes, ep.predict_linear_epitopes,
])
def test_every_predictor_normalises_before_it_scores(predictor):
    assert _key(predictor(UPPER.lower(), max_results=5)) == \
        _key(predictor(UPPER, max_results=5))


def test_the_property_tables_all_cover_the_twenty_standard_residues():
    assert len(ep.STANDARD_AMINO_ACIDS) == 20
    assert set(ep.HYDROPHOBICITY) == ep.STANDARD_AMINO_ACIDS
    assert set(ep.B_CELL_PROPENSITY) == ep.STANDARD_AMINO_ACIDS
    assert set(ep.SURFACE_ACCESSIBILITY) == ep.STANDARD_AMINO_ACIDS
    assert set(ep.FLEXIBILITY) == ep.STANDARD_AMINO_ACIDS
    assert ep.MHC_I_ANCHORS.keys() >= ep.STANDARD_AMINO_ACIDS


@pytest.mark.parametrize("value", [1234, None, ["M", "A"], 1.5, b"MA"])
def test_a_sequence_must_be_text(value):
    with pytest.raises(TypeError, match="sequence"):
        ep.predict_t_cell_epitopes(value)


def test_normalising_a_sequence_returns_the_cleaned_letters():
    assert ep._normalise_sequence("  mav\npkrr ") == "MAVPKRR"


def test_cleavage_sites_are_read_from_the_same_normalised_sequence():
    assert ep.cleavage_site_prediction("mavpkrr") == ep.cleavage_site_prediction("MAVPKRR")
