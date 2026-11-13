"""The gene-list and score arguments the three enrichment analyses share.

`len(gene_list)` sat outside every guard, so a generator died with
``TypeError: object of type 'list_iterator' has no len()`` while a list worked — and
a bare string passed for a list of one-character genes, matched none of them against
the Entrez-keyed background, and came back as "No input genes found in association
database", which reads as a biological result.
"""
import pytest

from biosuite.core import enrichment as en

ENTREZ = ["672", "7157", "2016"]


def test_a_list_of_identifiers_is_taken_as_it_stands():
    assert en._gene_arguments(ENTREZ) == ENTREZ


def test_a_set_or_tuple_is_materialised_so_len_is_defined():
    assert sorted(en._gene_arguments({"672", "7157"})) == ["672", "7157"]
    assert en._gene_arguments(("672",)) == ["672"]


def test_numbers_are_read_as_the_text_the_background_uses():
    """gene2go's Entrez column is text, and an int column is what a parser gives."""
    assert en._gene_arguments([672, 7157]) == ["672", "7157"]


@pytest.mark.parametrize("predictor", [en.run_ora, en.run_gsea, en.run_kegg_enrichment])
def test_one_bare_string_is_refused_instead_of_being_read_as_single_letter_genes(predictor):
    """The old answer for 'BRCA1' was a five-gene input list of characters."""
    with pytest.raises(ValueError, match="not one bare str"):
        predictor("BRCA1")


@pytest.mark.parametrize("predictor", [en.run_ora, en.run_gsea, en.run_kegg_enrichment])
def test_bytes_are_refused_for_the_same_reason(predictor):
    with pytest.raises(ValueError, match="not one bare bytes"):
        predictor(b"BRCA1")


@pytest.mark.parametrize("predictor", [en.run_ora, en.run_gsea, en.run_kegg_enrichment])
def test_something_that_is_not_a_series_of_genes_is_refused(predictor):
    with pytest.raises(ValueError, match="gene_list must be an iterable"):
        predictor(None)


@pytest.mark.parametrize("predictor", [en.run_ora, en.run_gsea, en.run_kegg_enrichment])
def test_a_blank_identifier_is_named_instead_of_entering_the_analysis(predictor):
    with pytest.raises(ValueError, match="blank gene identifier"):
        predictor(["BRCA1", "  ", "TP53"])


@pytest.mark.parametrize("predictor", [en.run_ora, en.run_gsea, en.run_kegg_enrichment])
def test_an_empty_list_is_answered_before_the_engine_is_touched(predictor, monkeypatch):
    monkeypatch.setattr(en, "HAS_GOATOOLS", True)
    monkeypatch.setattr(en, "HAS_GSEAPY", True)
    calls = []
    monkeypatch.setattr(en, "GODag", lambda *a, **k: calls.append("GODag"), raising=False)
    monkeypatch.setattr(en, "GOEnrichmentStudy",
                        lambda *a, **k: calls.append("study"), raising=False)
    monkeypatch.setattr(en, "gp", type("GP", (), {
        "prerank": lambda **k: calls.append("prerank"),
        "enrich": lambda **k: calls.append("enrich")})(), raising=False)

    report = predictor([])

    assert report.message == "Empty gene list provided.", report.message
    assert (report.num_input_genes, report.num_significant, report.results) == (0, 0, [])
    assert calls == [], "the engine was asked to analyse nothing"


def test_kegg_grew_the_empty_guard_the_other_two_already_had():
    """Before the fix an empty KEGG list reached gseapy, the only one of the three to."""
    report = en.run_kegg_enrichment([])

    assert report.analysis_type == "KEGG"
    assert report.message == "Empty gene list provided."


def test_scores_are_matched_to_genes_one_to_one():
    """`GSEA error: Length of values (1) does not match length of index (3)`.

    is a pandas complaint about two arguments the caller could have been told about.
    """
    with pytest.raises(ValueError, match=r"gene_scores has 1 value\(s\) for 3 gene"):
        en.run_gsea(["672", "7157", "2016"], gene_scores=[1.0])

    with pytest.raises(ValueError, match="gene_scores"):
        en.run_gsea(["672"], gene_scores="1.0")


def test_scores_of_the_right_length_are_accepted_without_an_engine():
    report = en.run_gsea(["672", "7157"], gene_scores=[1.2, -0.4])

    assert report.message.startswith("gseapy is not usable") or \
        report.num_input_genes == 2, report.message


@pytest.mark.parametrize("cutoff", [0, -1, 2, "0.05x", None, float('nan')])
def test_a_cutoff_that_filters_everything_or_nothing_is_refused(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        en.run_ora(ENTREZ, cutoff=cutoff)


def test_a_cutoff_written_as_text_still_works_because_cli_arguments_are_text():
    report = en.run_ora(ENTREZ, cutoff="0.01")

    assert report.threshold == 0.01
    assert report.engine in ("goatools", "none")


@pytest.mark.parametrize("method", [None, "", "   ", 7, ["fdr_bh"]])
def test_a_correction_that_names_nothing_is_refused(method):
    with pytest.raises(ValueError, match="method must name a multiple-testing correction"):
        en.run_ora(ENTREZ, method=method)


@pytest.mark.parametrize("kwargs", [
    {"min_size": 0}, {"min_size": -5}, {"max_size": 0}, {"max_size": "500"},
    {"min_size": 100, "max_size": 50},
])
def test_gene_set_size_limits_are_checked_before_the_engine_runs(kwargs):
    with pytest.raises(ValueError, match="min_size|max_size"):
        en.run_gsea(ENTREZ, **kwargs)

    with pytest.raises(ValueError, match="min_size|max_size"):
        en.run_kegg_enrichment(ENTREZ, **kwargs)


@pytest.mark.parametrize("permutation_num", [0, -1, 10.5, "1000", True])
def test_the_permutation_count_must_be_a_positive_whole_number(permutation_num):
    """`permutation_num=0` used to be forwarded to the engine as asked."""
    with pytest.raises(ValueError, match="permutation_num"):
        en.run_gsea(ENTREZ, permutation_num=permutation_num)


@pytest.mark.parametrize("seed", [-1, 1.5, "42"])
def test_the_seed_must_be_a_non_negative_whole_number(seed):
    with pytest.raises(ValueError, match="seed"):
        en.run_gsea(ENTREZ, seed=seed)
