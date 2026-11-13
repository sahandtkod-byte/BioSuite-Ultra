"""What a GWAS run puts in its result frame, and which rows it leaves out.

One monomorphic SNP used to abort a whole run with scipy's
``expected frequencies has a zero element`` error — a table with no information in
one of its columns is undefined, not broken, and a study of a thousand SNPs should
not stop at the first one that is fixed in both arms.
"""
import numpy as np
import pandas as pd
import pytest

from biosuite.core import gwas
from biosuite.core.gwas import GWAS_RESULT_COLUMNS, run_gwas

RS1 = {"chrom": "chr1", "pos": 100, "snp_id": "rs1",
       "case_alt": 90, "case_ref": 10, "ctrl_alt": 10, "ctrl_ref": 90}
MONOMORPHIC = {"chrom": "chr1", "pos": 200, "snp_id": "mono",
               "case_alt": 0, "case_ref": 100, "ctrl_alt": 0, "ctrl_ref": 100}
FIXED = {"chrom": "chr1", "pos": 300, "snp_id": "fixed",
         "case_alt": 100, "case_ref": 0, "ctrl_alt": 100, "ctrl_ref": 0}
UNTESTED = {"chrom": "chr1", "pos": 400, "snp_id": "empty-arm",
            "case_alt": 0, "case_ref": 0, "ctrl_alt": 10, "ctrl_ref": 10}


def frame(*rows):
    return pd.DataFrame(list(rows))


def test_an_undefined_row_no_longer_stops_the_snp_it_sits_next_to():
    """Measured on the old code: ValueError from scipy for the whole frame."""
    results = run_gwas(frame(RS1, MONOMORPHIC, FIXED))

    assert list(results["snp_id"]) == ["rs1"], list(results["snp_id"])


def test_monomorphic_and_untyped_rows_are_counted_in_one_visible_warning(caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.gwas"):
        results = run_gwas(frame(RS1, MONOMORPHIC, FIXED, UNTESTED))

    assert len(results) == 1
    text = caplog.text
    assert "tested 1 of 4 SNP row(s)" in text, text
    assert "2 monomorphic row(s)" in text, text
    assert "1 row(s) with no alleles typed in one arm" in text, text


def test_a_run_with_nothing_to_warn_about_logs_nothing(caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.gwas"):
        run_gwas(frame(RS1))

    assert "skipped" not in caplog.text, caplog.text


def test_a_count_that_is_not_a_number_is_skipped_and_named_in_the_warning(caplog):
    """A JSON body with an omitted field becomes NaN, which used to be reported
    as a p-value of NaN in the frame the plots and the report then read."""
    missing = dict(RS1, snp_id="rs-nan", pos=999, case_alt=float('nan'))
    negative = dict(RS1, snp_id="rs-negative", pos=998, case_alt=-5)

    with caplog.at_level("WARNING", logger="biosuite.core.gwas"):
        results = run_gwas(frame(RS1, missing, negative))

    assert list(results["snp_id"]) == ["rs1"], list(results["snp_id"])
    assert "counts that are not usable numbers" in caplog.text, caplog.text
    assert "rs-nan: case_alt" in caplog.text, caplog.text
    assert "rs-negative: case_alt" in caplog.text, caplog.text
    assert not results["p_value"].isna().any(), results


def test_no_row_of_a_result_frame_carries_a_nan_p_value():
    rows = [RS1, MONOMORPHIC, UNTESTED, dict(RS1, snp_id="nan-ctrl",
                                             ctrl_ref=float('inf'))]

    results = run_gwas(frame(*rows))

    assert list(results["snp_id"]) == ["rs1"], list(results["snp_id"])
    assert results[["p_value", "odds_ratio", "chi2_stat", "neg_log10",
                    "p_adjusted"]].notna().all().all(), results


def test_the_rows_are_kept_in_p_value_order_with_a_monotonic_adjusted_column():
    results = run_gwas(pd.DataFrame([
        dict(RS1, snp_id="weak", pos=1, case_alt=55, case_ref=45, ctrl_alt=45, ctrl_ref=55),
        dict(RS1, snp_id="strong", pos=2, case_alt=99, case_ref=1, ctrl_alt=1, ctrl_ref=99),
        dict(RS1, snp_id="middle", pos=3, case_alt=80, case_ref=20, ctrl_alt=20, ctrl_ref=80),
    ]))

    assert list(results["snp_id"]) == ["strong", "middle", "weak"]
    p = results["p_value"].tolist()
    assert p == sorted(p)
    adjusted = results["p_adjusted"].tolist()
    assert adjusted == sorted(adjusted), adjusted
    assert all(a >= b or np.isclose(a, b) for a, b in zip(adjusted, p))


def test_the_adjusted_p_values_come_from_the_one_shared_bh_implementation():
    p = np.array([1e-30, 1e-9, 0.5])

    assert gwas._benjamini_hochberg(p).tolist() == gwas._bh_shared(p).tolist()


def test_a_run_that_tested_nothing_still_answers_the_full_result_schema():
    results = run_gwas(frame(MONOMORPHIC, UNTESTED))

    assert results.empty
    assert list(results.columns) == list(GWAS_RESULT_COLUMNS)
    assert "p_adjusted" not in results.columns, "nothing was tested, nothing to adjust"


@pytest.mark.parametrize("row", [MONOMORPHIC, FIXED, UNTESTED])
def test_each_undefined_shape_is_filtered_on_its_own(row):
    assert run_gwas(frame(row)).empty


def test_an_identifier_of_zero_is_not_mistaken_for_a_missing_one():
    results = run_gwas(pd.DataFrame([dict(RS1, snp_id=0)]))

    assert list(results["snp_id"]) == ["0"], list(results["snp_id"])


def test_identifiers_that_are_nan_none_or_blank_are_all_replaced_by_a_row_number():
    """A JSON body whose rows are not all the same shape gives NaN, not ''."""
    rows = pd.DataFrame([
        dict(RS1, pos=1, snp_id=float('nan')),
        dict(RS1, pos=2, snp_id=None),
        dict(RS1, pos=3, snp_id=""),
        dict(RS1, pos=4, snp_id="rs-real"),
    ])

    assert list(run_gwas(rows)["snp_id"]) == ["row 0", "row 1", "row 2", "rs-real"]


def test_a_frame_without_identifiers_numbers_its_rows_instead_of_blanking_them():
    """All-empty `snp_id` values used to be indistinguishable downstream."""
    without_ids = pd.DataFrame([
        {"chrom": "chr1", "pos": 1000, "case_alt": 90, "case_ref": 10,
         "ctrl_alt": 10, "ctrl_ref": 90},
        {"chrom": "chr1", "pos": 9000, "case_alt": 85, "case_ref": 15,
         "ctrl_alt": 15, "ctrl_ref": 85},
    ])

    results = run_gwas(without_ids)

    assert list(results["snp_id"]) == ["row 0", "row 1"], list(results["snp_id"])


def test_an_identifier_of_zero_is_not_mistaken_for_a_missing_one():
    results = run_gwas(pd.DataFrame([dict(RS1, snp_id=0)]))

    assert list(results["snp_id"]) == ["0"], list(results["snp_id"])


def test_the_end_to_end_path_from_simulated_input_still_produces_hits():
    from biosuite.core.gwas import generate_gwas_data

    data = generate_gwas_data(n_snps=804, n_chromosomes=2, seed=5)
    results = run_gwas(data)

    assert {"p_value", "p_adjusted", "odds_ratio"}.issubset(results.columns)
    assert len(results) == len(data), len(results)
    # An odds ratio of exactly 0 is a real answer here: the alt allele is absent
    # from the cases of a simulated row, which the correction must not hide.
    assert (results["odds_ratio"] >= 0).all(), results["odds_ratio"].min()
    assert np.isfinite(results["odds_ratio"]).all(), results["odds_ratio"].max()
    assert set(results["chrom"]) == {"chr1", "chr2"}


def test_neg_log10_is_the_p_value_reported_and_not_a_second_guess_at_it():
    results = run_gwas(frame(RS1))

    assert results["neg_log10"].iloc[0] == pytest.approx(
        -np.log10(results["p_value"].iloc[0] + 1e-300))
