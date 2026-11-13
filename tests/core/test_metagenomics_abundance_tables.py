"""What metagenomics' diversity wrappers demand of an abundance table.

`compute_alpha_diversity` and `compute_beta_diversity` take DataFrames, and until
now took any DataFrame: a table whose count column was named 'abundance' answered
with a bare ``KeyError: 'count'``, an empty sample list answered with a Python list
instead of a matrix, and one sample with no rows crashed the whole
`analyze_diversity` call.
"""
import numpy as np
import pandas as pd
import pytest

from biosuite.core import metagenomics as mg


def _table(taxa, counts):
    return pd.DataFrame({"taxon": taxa, "count": counts})


# ── alpha ───────────────────────────────────────────────────────────────────

def test_alpha_diversity_reports_the_four_metrics():
    out = mg.compute_alpha_diversity(_table(["A", "B"], [50, 50]))
    assert out == {
        'observed_taxa': 2,
        'shannon': round(float(np.log(2)), 4),
        'simpson': pytest.approx(0.5051, abs=1e-4),
        'chao1': pytest.approx(2.0, abs=1e-6),
    }


def test_alpha_diversity_rounds_to_four_places():
    out = mg.compute_alpha_diversity(_table(["A", "B", "C"], [1, 2, 3]))
    for key in ("shannon", "simpson", "chao1"):
        assert out[key] == round(out[key], 4)


def test_alpha_diversity_of_a_missing_table_is_empty():
    assert mg.compute_alpha_diversity(None) == {}


def test_alpha_diversity_of_a_table_with_no_rows_is_empty():
    assert mg.compute_alpha_diversity(pd.DataFrame()) == {}
    assert mg.compute_alpha_diversity(_table([], [])) == {}


def test_a_wrongly_named_count_column_is_named_back():
    table = pd.DataFrame({"taxon": ["a"], "abundance": [5]})
    with pytest.raises(ValueError, match=r"needs the column\(s\) count"):
        mg.compute_alpha_diversity(table)
    with pytest.raises(ValueError, match=r"has \['taxon', 'abundance'\]"):
        mg.compute_alpha_diversity(table)


def test_a_negative_count_in_a_table_is_refused_by_the_metric():
    with pytest.raises(ValueError, match="cannot be negative"):
        mg.compute_alpha_diversity(_table(["a", "b"], [-1, 5]))


# ── beta ────────────────────────────────────────────────────────────────────

def test_beta_diversity_aligns_taxa_by_name_not_by_position():
    """The zip version compared 'a' against 'b' simply because they were both
    first in their own table, and reported 0 for two disjoint communities."""
    t1 = _table(["a", "b"], [10, 0])
    t2 = _table(["b", "a"], [10, 0])
    assert mg.compute_beta_diversity([t1, t2])[0, 1] == pytest.approx(1.0)
    t3 = _table(["a", "b"], [10, 0])
    assert mg.compute_beta_diversity([t1, t3])[0, 1] == pytest.approx(0.0)


def test_beta_diversity_is_a_square_symmetric_matrix_with_a_zero_diagonal():
    tables = [_table(["a", "b"], [10, 0]), _table(["b", "c"], [5, 5]),
              _table(["a", "c"], [1, 1])]
    d = mg.compute_beta_diversity(tables)
    assert d.shape == (3, 3)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0)
    assert (d >= 0).all() and (d <= 1).all()


def test_a_taxon_present_in_one_sample_only_counts_as_zero_elsewhere():
    d = mg.compute_beta_diversity([_table(["a"], [10]), _table(["b"], [10])])
    assert d[0, 1] == pytest.approx(1.0)


def test_a_duplicate_taxon_in_one_table_is_accumulated():
    d = mg.compute_beta_diversity([_table(["a", "a"], [5, 5]), _table(["a"], [10])])
    assert d[0, 1] == pytest.approx(0.0)


def test_no_samples_at_all_is_a_zero_by_zero_matrix():
    """A bare [] used to come back, which the caller could not index like the
    matrix every other answer is."""
    d = mg.compute_beta_diversity([])
    assert isinstance(d, np.ndarray)
    assert d.shape == (0, 0)


def test_a_single_sample_is_a_one_by_one_zero_matrix():
    d = mg.compute_beta_diversity([_table(["a"], [3])])
    assert d.shape == (1, 1)
    assert d[0, 0] == 0.0


def test_a_malformed_sample_table_is_described_not_keyerrored():
    with pytest.raises(ValueError, match=r"a sample table needs the column\(s\)"):
        mg.compute_beta_diversity([pd.DataFrame({"taxon": ["a"], "n": [1]})])


def test_an_empty_sample_does_not_break_the_pairwise_matrix():
    d = mg.compute_beta_diversity([_table(["a"], [3]), _table([], [])])
    assert d.shape == (2, 2)
    # A sample with no reads at all is maximally unlike one with reads, and
    # identical to another sample with no reads.
    assert d[0, 1] == pytest.approx(1.0)


# ── the analyze wrapper ─────────────────────────────────────────────────────

def test_one_sample_has_no_beta_matrix():
    res = mg.analyze_diversity([_table(["a", "b"], [3, 1])])
    assert res.beta_diversity is None
    assert res.alpha_diversity[0]["sample"] == "Sample_1"
    assert res.message == "Analyzed 1 samples"


def test_samples_are_numbered_in_input_order():
    res = mg.analyze_diversity([_table(["a"], [1]), _table(["a"], [1]),
                                _table(["a"], [1])])
    assert [d["sample"] for d in res.alpha_diversity] == \
        ["Sample_1", "Sample_2", "Sample_3"]
    assert res.message == "Analyzed 3 samples"
    assert res.beta_diversity is not None
    assert res.beta_diversity.shape == (3, 3)


def test_a_sample_with_no_data_keeps_the_same_keys_and_says_so_in_the_message():
    """The old answer was a dict holding only the sample label, which a caller
    could not tell apart from a community with genuinely zero metrics."""
    res = mg.analyze_diversity([_table(["a"], [3]), pd.DataFrame()])
    assert res.message == "Analyzed 2 samples (1 with no abundance data)"
    empty = res.alpha_diversity[1]
    assert empty == {"observed_taxa": 0, "shannon": 0.0, "simpson": 0.0,
                     "chao1": 0.0, "sample": "Sample_2"}


def test_no_samples_at_all_analyzes_nothing():
    res = mg.analyze_diversity([])
    assert res.alpha_diversity == []
    assert res.beta_diversity is None
    assert res.message == "Analyzed 0 samples"


def test_pcoa_coordinates_are_not_pretended_at():
    """The result type used to carry a pcoa_coords field that nothing ever
    computed; ordination is not part of this module."""
    assert not hasattr(mg.DiversityResult(), "pcoa_coords")
    res = mg.analyze_diversity([_table(["a"], [1]), _table(["b"], [1])])
    assert sorted(vars(res)) == ["alpha_diversity", "beta_diversity", "message"]
