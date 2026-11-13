"""The simulated GWAS input: the demo data every GUI and CLI demo path runs on.

Two promises live here.  The frame has exactly the number of rows that was asked
for, and generating it does not disturb the process-wide random stream — the
"Add some signals" branch makes the counts differ per chromosome, so a caller that
seeded numpy before a demo run expected the demo not to move it.
"""
import numpy as np
import pytest

from biosuite.core.gwas import generate_gwas_data, run_gwas

COLUMNS = ["chrom", "pos", "snp_id", "case_alt", "case_ref", "ctrl_alt", "ctrl_ref"]


@pytest.mark.parametrize("n_snps", [0, 1, 21, 22, 23, 250, 2000])
def test_the_row_count_is_the_number_the_caller_passed(n_snps):
    """Per-chromosome truncation used to answer 242 rows for a request of 250."""
    data = generate_gwas_data(n_snps=n_snps)

    assert len(data) == n_snps, (n_snps, len(data))
    if n_snps:
        assert {f"rs{i}" for i in range(1, n_snps + 1)} == set(data["snp_id"])
        assert list(data.columns) == COLUMNS, list(data.columns)


@pytest.mark.parametrize("n_snps, n_chromosomes", [(250, 22), (7, 3), (100, 1), (5, 22)])
def test_the_rows_are_spread_across_the_chromosomes_without_a_gap(n_snps, n_chromosomes):
    data = generate_gwas_data(n_snps=n_snps, n_chromosomes=n_chromosomes)

    counts = data["chrom"].value_counts()
    assert counts.max() - counts.min() <= 1, counts.to_dict()
    assert len(counts) == min(n_snps, n_chromosomes), counts


def test_a_request_smaller_than_the_chromosome_count_fills_the_first_ones():
    data = generate_gwas_data(n_snps=3, n_chromosomes=22)

    assert list(data["chrom"]) == ["chr1", "chr2", "chr3"]


def test_positions_stay_inside_the_declared_range_and_counts_stay_non_negative():
    data = generate_gwas_data(n_snps=400, seed=3)

    assert data["pos"].between(1, 249_999_999).all(), data["pos"].describe()
    for column in ("case_alt", "case_ref", "ctrl_alt", "ctrl_ref"):
        assert (data[column] >= 0).all(), column
    assert (data["case_alt"] + data["case_ref"] > 0).all()
    assert (data["ctrl_alt"] + data["ctrl_ref"] > 0).all()


def test_the_generated_input_needs_no_rows_dropped_by_the_association_test():
    """Every simulated row is testable, so `run_gwas` on it is a full run."""
    data = generate_gwas_data(n_snps=200, seed=1)

    assert len(run_gwas(data)) == len(data)


def test_the_same_seed_answers_the_same_table_and_another_seed_does_not():
    first = generate_gwas_data(n_snps=60, seed=9)
    second = generate_gwas_data(n_snps=60, seed=9)
    third = generate_gwas_data(n_snps=60, seed=10)

    assert first.equals(second)
    assert not first["pos"].equals(third["pos"])


def test_generating_demo_data_leaves_the_global_random_stream_untouched():
    np.random.seed(1234)
    before = np.random.random()

    np.random.seed(1234)
    generate_gwas_data(n_snps=100, n_chromosomes=2, seed=7)
    after = np.random.random()

    assert before == pytest.approx(after)


@pytest.mark.parametrize("n_chromosomes", [0, -1, 1.0, "22", None])
def test_a_chromosome_count_that_cannot_spread_rows_is_refused(n_chromosomes):
    """`divmod(n_snps, 0)` was a ZeroDivisionError with no mention of the argument."""
    with pytest.raises(ValueError, match="n_chromosomes"):
        generate_gwas_data(n_snps=10, n_chromosomes=n_chromosomes)


@pytest.mark.parametrize("n_snps", [-5, -1, 10.5, "10", None, True])
def test_a_snp_count_that_cannot_be_answered_is_refused(n_snps):
    """A negative count used to answer an empty frame instead of the rows asked for."""
    with pytest.raises(ValueError, match=r"n_snps must be"):
        generate_gwas_data(n_snps=n_snps)


def test_zero_snps_is_a_legitimate_request_not_an_error():
    assert generate_gwas_data(n_snps=0).empty


def test_the_named_signal_region_is_still_where_the_chr6_effect_lives():
    data = generate_gwas_data(n_snps=2000, seed=42)
    region = data[(data["chrom"] == "chr6")
                  & (data["pos"] > 25_000_000)
                  & (data["pos"] < 35_000_000)]

    if len(region):
        assert (region["case_alt"] >= region["ctrl_alt"]).all(), region.head()
