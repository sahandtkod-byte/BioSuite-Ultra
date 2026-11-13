"""go_enrichment: Fisher's exact test over a declared background.

The p-values this function returns are what a user reads as the result of a GO
analysis, so every number in them has to come from the data they passed — not from a
default the function invented for them.
"""
import math
import os
import subprocess
import sys

import pytest

from biosuite.core.go_browser import go_enrichment


def _hyper_geq(k_observed, total, annotated, query):
    """Reference P(X >= k) for the hypergeometric of a 2x2 contingency table.

    Written out in plain math, independently of scipy, so the pinned values below
    are checked against arithmetic rather than against the code under test.
    """
    def pmass(k):
        return (math.comb(annotated, k) * math.comb(total - annotated, query - k)
                / math.comb(total, query))

    return sum(pmass(k) for k in range(k_observed, min(annotated, query) + 1))


def test_a_two_of_four_overlap_is_scored_by_hand():
    genes = ["G1", "G2", "G3", "G4"]
    result = go_enrichment(genes, {"GO:x": ["G1", "G2", "G9", "G10"]}, 20)
    row = result[0]
    assert (row["count"], row["genes"]) == (2, ["G1", "G2"])
    assert row["p_value"] == pytest.approx(785 / 4845)
    assert row["p_value"] == pytest.approx(_hyper_geq(2, 20, 4, 4))
    assert row["enrichment"] == pytest.approx(2.5)


def test_a_term_covering_the_whole_query_is_the_strongest_hit():
    genes = ["G1", "G2", "G3", "G4"]
    result = go_enrichment(genes, {"GO:all": genes,
                                  "GO:half": ["G1", "G2", "G9", "G10"]}, 20)
    assert [r["go_term"] for r in result] == ["GO:all", "GO:half"]
    assert result[0]["p_value"] == pytest.approx(1 / 4845)
    assert result[0]["enrichment"] == pytest.approx(5.0)


def test_genes_shared_with_no_term_of_the_map_cannot_be_enriched():
    result = go_enrichment(["G1", "G2"], {"GO:other": ["X1", "X2", "X3"]}, 50)
    assert [(r["count"], r["genes"], r["enrichment"]) for r in result] == [(0, [], 0.0)]
    assert result[0]["p_value"] == pytest.approx(1.0)


def test_a_term_that_annotates_no_gene_is_kept_and_says_so_rather_than_dividing_by_zero(
        caplog):
    """Measured before the fix: an empty annotation list raised ZeroDivisionError.

    One such term killed the whole analysis, so a gene-set file with a term nobody
    was annotated to returned no result at all — the opposite of the truth.
    """
    with caplog.at_level("WARNING", logger="biosuite.core.go_browser"):
        result = go_enrichment(["G1"], {"GO:empty": [], "GO:hit": ["G1", "G2"]}, 10)
    empty = [r for r in result if r["go_term"] == "GO:empty"][0]
    assert (empty["count"], empty["p_value"], empty["enrichment"], empty["genes"]) == (
        0, 1.0, 0.0, [])
    assert "annotates no genes" in caplog.text, caplog.text


def test_the_background_has_to_be_declared_by_the_caller():
    """The old default invented len(gene_list) * 10 and the statistics followed it.

    Measured: with 4 query genes and a term annotating exactly those 4, the
    invented background of 40 reported enrichment=10.0 — the magic factor itself,
    presented as a result.
    """
    with pytest.raises(TypeError) as exc:
        go_enrichment(["G1"], {"GO:a": ["G1"]})
    assert "background_size" in str(exc.value)


@pytest.mark.parametrize("bad", [0, -5, 1.5, None, True, "100"])
def test_a_background_that_is_not_a_count_of_genes_is_refused(bad):
    with pytest.raises(ValueError) as exc:
        go_enrichment(["G1"], {"GO:a": ["G1"]}, bad)
    assert f"background_size must be a whole number of genes of at least 1, got {bad!r}" \
        in str(exc.value)


def test_a_background_too_small_for_the_two_sets_is_named_not_left_to_scipy():
    """A 2x2 table with a negative cell has no p-value.

    Measured before the fix, the caller saw scipy's own complaint —
    "All values in `table` must be nonnegative" — which named neither the argument
    that was wrong nor the term that exposed it.
    """
    genes = [f"G{i}" for i in range(4)]
    with pytest.raises(ValueError) as exc:
        go_enrichment(genes, {"GO:big": [f"G{i}" for i in range(60)]}, 20)
    assert "background_size=20 cannot hold the 60 genes" in str(exc.value)
    assert "GO:big" in str(exc.value)


def test_an_exactly_large_enough_background_is_accepted():
    genes = ["G1", "G2"]
    result = go_enrichment(genes, {"GO:x": ["G1", "G2", "G3", "G4"]}, 4)
    assert result[0]["count"] == 2 and math.isfinite(result[0]["p_value"])


def test_an_empty_query_is_refused_rather_than_ranked(caplog):
    with pytest.raises(ValueError) as exc:
        go_enrichment([], {"GO:a": ["G1"]}, 100)
    assert "at least one gene in gene_list" in str(exc.value)
    assert caplog.text == ""


@pytest.mark.parametrize("bad", [None, 7, 3.5])
def test_a_query_that_cannot_be_iterated_is_named_as_such(bad):
    with pytest.raises(ValueError) as exc:
        go_enrichment(bad, {"GO:a": ["G1"]}, 100)
    assert f"as an iterable of gene identifiers, got {bad!r}" in str(exc.value)


def test_a_string_query_is_refused_because_iterating_it_yields_letters():
    with pytest.raises(ValueError) as exc:
        go_enrichment("G1G2", {"GO:a": ["G1", "G2"]}, 100)
    assert "not one bare str" in str(exc.value)


def test_a_bytes_query_is_refused_the_same_way():
    with pytest.raises(ValueError) as exc:
        go_enrichment(b"G1", {"GO:a": ["G1"]}, 100)
    assert "not one bare bytes" in str(exc.value)


@pytest.mark.parametrize("bad", [["G1", ""], ["  "]])
def test_a_blank_gene_name_is_refused_rather_than_never_matching_anything(bad):
    with pytest.raises(ValueError) as exc:
        go_enrichment(bad, {"GO:a": ["G1"]}, 100)
    assert "found 1 blank gene identifier(s)" in str(exc.value)


@pytest.mark.parametrize("value", [None, True, ["G2"], {"G2": 1}])
def test_an_entry_that_is_not_a_gene_identifier_is_named_not_stringified(value):
    """str(None) is 'None', which is a name that matches nothing.

    Turning it into text and carrying on would put a gene that no annotation table
    can hold into the counts, so the query size and every ratio over it would be
    about an entry the caller never meant as a gene.
    """
    with pytest.raises(ValueError) as exc:
        go_enrichment(["G1", value], {"GO:a": ["G1"]}, 100)
    assert f"that is not a name, e.g. {value!r}" in str(exc.value)


def test_a_blank_inside_a_terms_annotation_list_names_the_term():
    with pytest.raises(ValueError) as exc:
        go_enrichment(["G1"], {"GO:x": ["G1", ""]}, 100)
    assert "blank gene identifier(s) for 'GO:x'" in str(exc.value)


def test_a_number_as_a_gene_identifier_is_readable_as_a_name():
    """Entrez ids arrive as integers often enough that the module must not lose them."""
    result = go_enrichment([1001, 1002], {"GO:a": [1001, 1002]}, 20)
    assert result[0]["count"] == 2


def test_p_values_are_plain_floats_ready_for_a_report():
    result = go_enrichment(["G1", "G2"], {"GO:a": ["G1", "G2"]}, 20)
    assert type(result[0]["p_value"]) is float


def test_equal_p_values_are_ordered_by_term_id_so_the_report_is_stable():
    genes = ["G1", "G2"]
    result = go_enrichment(genes, {"GO:z": ["X9"], "GO:a": ["X8"]}, 100)
    assert [r["go_term"] for r in result] == ["GO:a", "GO:z"]


def test_the_hit_list_is_in_a_fixed_order_across_interpreters(tmp_path):
    """Set iteration order depends on PYTHONHASHSEED, so the report used to drift.

    The README promises pipelines that are re-runnable; a gene list whose order
    changed between two identical runs is the kind of difference that makes a user
    doubt every other number on the page.
    """
    code = ("from biosuite.core.go_browser import go_enrichment;"
            "print([r['genes'] for r in go_enrichment("
            "['G1','G2','G3','G4'],{'GO:x':['G4','G3','G2','G1','Z9','Z8']}, 40)])")
    outputs = []
    for seed in ("0", "1", "12345"):
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, env=dict(os.environ, PYTHONHASHSEED=seed))
        outputs.append(proc.stdout.strip())
    assert len(set(outputs)) == 1, outputs
    assert "['G1', 'G2', 'G3', 'G4']" in outputs[0], outputs[0]


def test_duplicated_genes_in_the_query_are_counted_once():
    result = go_enrichment(["G1", "G1", "G2"], {"GO:a": ["G1", "G2"]}, 20)
    assert result[0]["count"] == 2
    assert result[0]["genes"] == ["G1", "G2"]


def test_the_report_lists_every_term_the_map_held():
    result = go_enrichment(["G1"], {"GO:a": ["G1"], "GO:b": ["G2"], "GO:c": ["G1", "G3"]},
                           20)
    assert sorted(r["go_term"] for r in result) == ["GO:a", "GO:b", "GO:c"]
