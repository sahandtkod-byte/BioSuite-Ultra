"""Lead-SNP detection: which row claims a locus, and what is said when none does.

The scan de-duplicates loci by the value of `snp_id`.  A frame whose identifiers
are blank — which is what `run_gwas` produced for input without a `snp_id` column —
therefore retired every locus after the first one and reported a single lead SNP per
chromosome.
"""
import numpy as np
import pandas as pd
import pytest

from biosuite.core.gwas import detect_lead_snps

SIGNIFICANT = pd.DataFrame([
    {"chrom": "chr1", "pos": 1000, "snp_id": "rs1", "p_value": 1e-40},
    {"chrom": "chr1", "pos": 2000, "snp_id": "rs2", "p_value": 1e-30},
    {"chrom": "chr1", "pos": 900_000, "snp_id": "rs3", "p_value": 1e-20},
    {"chrom": "chr2", "pos": 5000, "snp_id": "rs4", "p_value": 1e-25},
])


def test_a_locus_is_claimed_by_its_strongest_snp_not_the_leftmost():
    """Sorting by position before clumping made the first row the lead by accident."""
    one_locus = SIGNIFICANT.iloc[[0, 1]].copy()
    one_locus.loc[1, "p_value"] = 1e-300

    leads = detect_lead_snps(one_locus, window_kb=1)

    assert list(leads["snp_id"]) == ["rs2"], list(leads["snp_id"])


def test_two_loci_a_megabase_apart_are_both_reported():
    leads = detect_lead_snps(SIGNIFICANT, window_kb=500)

    assert list(leads["snp_id"]) == ["rs1", "rs3", "rs4"], list(leads["snp_id"])
    assert list(leads.index) == [0, 1, 2], leads.index


def test_the_window_is_measured_in_kilobases_of_the_pos_column():
    """±1 kb of rs1 at position 1000 reaches rs2 at 2000; ±0 kb reaches nothing else."""
    no_clumping = detect_lead_snps(SIGNIFICANT, window_kb=0)
    one_kb = detect_lead_snps(SIGNIFICANT, window_kb=1)

    assert list(no_clumping["snp_id"]) == ["rs1", "rs2", "rs3", "rs4"], \
        list(no_clumping["snp_id"])
    assert list(one_kb["snp_id"]) == ["rs1", "rs3", "rs4"], list(one_kb["snp_id"])


def test_a_window_wider_than_the_chromosome_clumps_it_into_one_locus():
    assert list(detect_lead_snps(SIGNIFICANT, window_kb=1_000_000)["snp_id"]) == \
        ["rs1", "rs4"], list(detect_lead_snps(SIGNIFICANT, window_kb=1_000_000)["snp_id"])


def test_a_window_of_zero_clumps_only_snps_at_the_same_position():
    same_place = pd.DataFrame([
        {"chrom": "chr1", "pos": 100, "snp_id": "a", "p_value": 1e-9},
        {"chrom": "chr1", "pos": 100, "snp_id": "b", "p_value": 1e-12},
        {"chrom": "chr1", "pos": 101, "snp_id": "c", "p_value": 1e-11},
    ])

    assert list(detect_lead_snps(same_place, window_kb=0)["snp_id"]) == ["b", "c"]


def test_identifiers_that_repeat_or_are_blank_do_not_retire_a_locus():
    """`used` was keyed on the snp_id string, so three empty ids were one locus."""
    blank_ids = pd.DataFrame([
        {"chrom": "chr1", "pos": 1000, "snp_id": "", "p_value": 1e-40},
        {"chrom": "chr1", "pos": 900_000, "snp_id": "", "p_value": 1e-30},
        {"chrom": "chr1", "pos": 500, "snp_id": "rs-x", "p_value": 1e-20},
        {"chrom": "chr1", "pos": 600, "snp_id": "rs-x", "p_value": 1e-19},
    ])

    leads = detect_lead_snps(blank_ids, window_kb=0)

    assert len(leads) == 4, list(leads["snp_id"])


def test_the_lead_frame_carries_the_input_columns_rather_than_a_rebuilt_subset():
    with_extra = SIGNIFICANT.assign(odds_ratio=2.0, neg_log10=40.0)

    leads = detect_lead_snps(with_extra, window_kb=0)

    assert list(leads.columns) == list(with_extra.columns), list(leads.columns)


def test_nothing_significant_answers_an_empty_frame_that_its_callers_can_read():
    """A bare ``pd.DataFrame()`` has no columns, so ``leads["snp_id"]`` died on it."""
    leads = detect_lead_snps(SIGNIFICANT, p_threshold=1e-300)

    assert leads.empty
    assert list(leads.columns) == list(SIGNIFICANT.columns), list(leads.columns)
    assert list(leads["snp_id"]) == []


def test_the_result_of_a_run_with_no_testable_snp_survives_the_lead_scan():
    empty_run = pd.DataFrame(columns=["chrom", "pos", "snp_id", "p_value"])

    leads = detect_lead_snps(empty_run)

    assert leads.empty and list(leads.columns) == list(empty_run.columns)


def test_a_p_value_of_nan_is_not_significant_at_any_threshold():
    frame = SIGNIFICANT.assign(p_value=[float('nan'), 1e-9, 1e-8, 1e-7])

    leads = detect_lead_snps(frame, p_threshold=1.0, window_kb=0)

    assert list(leads["snp_id"]) == ["rs2", "rs3", "rs4"], list(leads["snp_id"])


@pytest.mark.parametrize("window_kb", [-5, -0.5, float('nan'), float('inf')])
def test_a_window_that_cannot_define_a_locus_is_refused(window_kb):
    """A negative window used to disable clumping silently: every SNP became a lead."""
    with pytest.raises(ValueError, match="window_kb"):
        detect_lead_snps(SIGNIFICANT, window_kb=window_kb)


@pytest.mark.parametrize("window_kb", [None, [500], "wide"])
def test_a_window_that_is_not_a_number_at_all_is_refused(window_kb):
    with pytest.raises(ValueError, match="window_kb"):
        detect_lead_snps(SIGNIFICANT, window_kb=window_kb)


def test_a_window_written_as_text_is_accepted_because_cli_arguments_are_text():
    assert list(detect_lead_snps(SIGNIFICANT, window_kb="500")["snp_id"]) == \
        ["rs1", "rs3", "rs4"]


def test_a_window_that_is_a_plain_number_in_any_unit_is_accepted():
    for window in (0, 500, 500.0, np.float64(500)):
        assert not detect_lead_snps(SIGNIFICANT, window_kb=window).empty


def test_a_snp_at_the_edge_of_two_loci_clumps_whichever_way_it_lands():
    chain = pd.DataFrame([
        {"chrom": "chr1", "pos": 0, "snp_id": "a", "p_value": 1e-30},
        {"chrom": "chr1", "pos": 400_000, "snp_id": "b", "p_value": 1e-29},
        {"chrom": "chr1", "pos": 800_000, "snp_id": "c", "p_value": 1e-28},
    ])

    # a claims 0±500 kb, which covers b but not c; c is then free to lead its own locus.
    assert list(detect_lead_snps(chain, window_kb=500)["snp_id"]) == ["a", "c"]
