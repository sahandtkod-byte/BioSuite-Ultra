"""The Survival Analysis tab: load, run, demo, log-rank, Cox — what each path shows.

Every handler here is a thin slice of the same contract: read what the user typed,
call the core, show the core's own report, and say something when any step fails.
The expected report is recomputed from ``biosuite.core.survival`` in the test rather
than quoted, so the tab is measured against the module it wraps instead of against a
string somebody typed once.
"""
import numpy as np
import pandas as pd
import pytest

from biosuite.core.survival import (
    cox_ph_summary,
    format_km_result,
    kaplan_meier,
    log_rank_test,
)

TIMES = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
EVENTS = [1, 0, 1, 1, 0, 1]


@pytest.fixture
def survival_csv(tmp_path):
    path = tmp_path / "times.csv"
    path.write_text("time,event,covariate\n" + "\n".join(
        f"{t},{e},x" for t, e in zip(TIMES, EVENTS)) + "\n", encoding="utf-8")
    return path


def test_load_puts_the_chosen_file_in_the_entry_and_says_so(ui, survival_csv):
    ui.open_path = str(survival_csv)

    ui.app._surv_load()

    assert ui.app.surv_path.get() == str(survival_csv)
    assert ui.status() == f"Loaded: {survival_csv.name}"


def test_a_cancelled_load_leaves_the_entry_alone(ui):
    ui.open_path = ""
    ui.clear("surv_path")
    ui.set("surv_path", "keep/me.csv")

    ui.app._surv_load()

    assert ui.app.surv_path.get() == "keep/me.csv"
    assert ui.messages == []


def test_run_without_a_path_asks_for_one(ui):
    ui.clear("surv_path")
    ui.clear("surv_result")

    ui.app._surv_run()

    assert ui.messages_of("warning") == [
        "Please load a CSV file with time and event columns."]
    assert ui.text("surv_result").strip() == ""


def test_run_reports_the_same_numbers_the_core_computed(ui, survival_csv):
    expected = format_km_result(kaplan_meier(
        pd.Series(TIMES).values, pd.Series(EVENTS).values))

    ui.set("surv_path", str(survival_csv))
    ui.app._surv_run()

    assert ui.text("surv_result").strip() == expected.strip()
    assert ui.status() == "Survival analysis complete"
    assert ui.messages == []


def test_run_refuses_a_file_without_the_two_columns(ui, tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("hour,state\n1,0\n2,1\n", encoding="utf-8")
    ui.clear("surv_result")

    ui.set("surv_path", str(path))
    ui.app._surv_run()

    assert ui.messages_of("error") == ["CSV must have 'time' and 'event' columns."]
    assert ui.text("surv_result").strip() == ""


def test_run_reports_a_file_it_could_not_read(ui, tmp_path):
    ui.set("surv_path", str(tmp_path / "absent.csv"))

    ui.app._surv_run()

    errors = ui.messages_of("error")
    assert len(errors) == 1, "the failure must be named, not swallowed"
    assert "absent.csv" in errors[0], errors[0]


def test_the_demo_analysis_fills_the_results_box(ui):
    ui.clear("surv_result")

    ui.app._surv_demo()

    report = ui.text("surv_result")
    assert report.strip(), "the demo produced nothing"
    assert ui.status() == "Demo survival analysis complete"
    assert ui.messages == []


def test_the_demo_does_not_touch_the_global_random_state(ui):
    before = np.random.get_state()
    ui.app._surv_demo()
    after = np.random.get_state()

    assert after[0] == before[0]
    assert np.array_equal(after[1], before[1])
    assert after[2:] == before[2:]


def test_the_demo_reports_a_core_failure(ui, monkeypatch):
    def broken(times, events, *args, **kwargs):
        raise RuntimeError("no events to count")

    monkeypatch.setattr("biosuite.core.survival.kaplan_meier", broken)
    ui.clear("surv_result")

    ui.app._surv_demo()

    assert ui.messages_of("error") == ["no events to count"]
    assert ui.text("surv_result").strip() == ""


@pytest.fixture
def grouped_csv(tmp_path):
    path = tmp_path / "groups.csv"
    rows = ["time,event,group"]
    rows += [f"{t},{e},A" for t, e in zip(TIMES, EVENTS)]
    rows += [f"{t * 2},{e},B" for t, e in zip(TIMES, EVENTS)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def covariate_csv(tmp_path):
    path = tmp_path / "covariates.csv"
    covariates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    path.write_text("time,event,covariate\n" + "\n".join(
        f"{t},{e},{c}" for t, e, c in zip(TIMES, EVENTS, covariates)) + "\n",
        encoding="utf-8")
    return path


def _expected_logrank_report(df, group_col="group"):
    lines = ["=== Log-Rank Group Comparison ==="]
    pairs = []
    for name in df[group_col].dropna().unique():
        sub = df[df[group_col] == name]
        km = kaplan_meier(sub["time"].values, sub["event"].values)
        lines.append(f"Group {name!r}: n={len(sub)}, events={km.number_events}, "
                     f"median survival={km.median_survival}")
        pairs.append((sub["time"].values, sub["event"].values))
    result = log_rank_test(pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])
    lines.append(f"Chi-square statistic: {result['statistic']}")
    lines.append(f"p-value: {result['p_value']}")
    return "\n".join(lines)


def _expected_cox_report(result):
    ratio = result["hazard_ratio"]
    lines = [
        "=== Cox Proportional Hazards (median split) ===",
        f"Hazard ratio: {ratio if ratio is not None else 'not estimable'}",
        f"p-value: {result['p_value']}",
        f"Median survival, high group: {result['median_high_group']}",
        f"Median survival, low group: {result['median_low_group']}",
        f"Split: {result['split']}",
        result["note"],
    ]
    if result["message"]:
        lines.append(result["message"])
    return "\n".join(lines)


def test_logrank_without_a_path_asks_for_one(ui):
    ui.clear("surv_path")
    ui.clear("surv_result")

    ui.app._surv_logrank()

    assert ui.messages_of("warning") == [
        "Please load a CSV file with time and event columns."]
    assert ui.text("surv_result").strip() == ""


def test_cox_without_a_path_asks_for_one(ui):
    ui.clear("surv_path")
    ui.clear("surv_result")

    ui.app._surv_cox()

    assert ui.messages_of("warning") == [
        "Please load a CSV file with time and event columns."]
    assert ui.text("surv_result").strip() == ""


def test_logrank_reports_a_file_it_could_not_read(ui, tmp_path):
    ui.set("surv_path", str(tmp_path / "absent.csv"))

    ui.app._surv_logrank()

    errors = ui.messages_of("error")
    assert len(errors) == 1, "the failure must be named, not swallowed"
    assert "absent.csv" in errors[0], errors[0]


def test_cox_reports_a_file_it_could_not_read(ui, tmp_path):
    ui.set("surv_path", str(tmp_path / "absent.csv"))

    ui.app._surv_cox()

    errors = ui.messages_of("error")
    assert len(errors) == 1, "the failure must be named, not swallowed"
    assert "absent.csv" in errors[0], errors[0]


def test_logrank_refuses_a_file_without_the_group_column(ui, survival_csv):
    ui.clear("surv_result")

    ui.set("surv_path", str(survival_csv))
    ui.app._surv_logrank()

    assert ui.messages_of("error") == ["CSV is missing column(s): group."]
    assert ui.text("surv_result").strip() == ""


def test_cox_refuses_a_file_without_the_covariate_column(ui, tmp_path):
    path = tmp_path / "bare.csv"
    path.write_text("time,event\n1,1\n2,0\n", encoding="utf-8")
    ui.clear("surv_result")

    ui.set("surv_path", str(path))
    ui.app._surv_cox()

    assert ui.messages_of("error") == ["CSV is missing column(s): covariate."]
    assert ui.text("surv_result").strip() == ""


def test_logrank_needs_exactly_two_groups(ui, tmp_path):
    path = tmp_path / "three.csv"
    rows = ["time,event,group"] + [f"{t},{e},{g}" for t, e in zip(TIMES, EVENTS)
                                   for g in ("A", "B", "C")]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ui.clear("surv_result")

    ui.set("surv_path", str(path))
    ui.app._surv_logrank()

    assert ui.messages_of("error") == [
        "Log-rank needs exactly 2 groups in 'group', found 3."]
    assert ui.text("surv_result").strip() == ""


def test_logrank_reports_the_same_numbers_the_core_computed(ui, grouped_csv):
    expected = _expected_logrank_report(pd.read_csv(grouped_csv))

    ui.set("surv_path", str(grouped_csv))
    ui.app._surv_logrank()

    assert ui.text("surv_result").strip() == expected
    assert ui.status() == "Log-rank comparison complete"
    assert ui.messages == []


def test_logrank_honours_a_custom_group_column(ui, tmp_path):
    path = tmp_path / "arms.csv"
    rows = ["time,event,arm"]
    rows += [f"{t},{e},left" for t, e in zip(TIMES, EVENTS)]
    rows += [f"{t * 2},{e},right" for t, e in zip(TIMES, EVENTS)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ui.set("surv_group_col", "arm")

    ui.set("surv_path", str(path))
    ui.app._surv_logrank()

    assert ui.messages == []
    assert "Group 'left'" in ui.text("surv_result")
    assert ui.status() == "Log-rank comparison complete"


def test_logrank_falls_back_to_the_default_column_when_the_entry_is_empty(ui, grouped_csv):
    ui.clear("surv_group_col")

    ui.set("surv_path", str(grouped_csv))
    ui.app._surv_logrank()

    assert ui.messages == []
    assert ui.status() == "Log-rank comparison complete"


def test_logrank_reports_a_core_failure(ui, tmp_path):
    path = tmp_path / "negative.csv"
    path.write_text("time,event,group\n-1,1,A\n2,0,B\n", encoding="utf-8")
    ui.clear("surv_result")

    ui.set("surv_path", str(path))
    ui.app._surv_logrank()

    errors = ui.messages_of("error")
    assert len(errors) == 1, "the failure must be named, not swallowed"
    assert "negative" in errors[0], errors[0]
    assert ui.text("surv_result").strip() == ""


def test_cox_reports_the_same_numbers_the_core_computed(ui, covariate_csv):
    df = pd.read_csv(covariate_csv)
    result = cox_ph_summary(df["time"].values, df["event"].values, df["covariate"].values)
    assert result["hazard_ratio"] is not None, "the fixture must exercise the estimable path"

    ui.set("surv_path", str(covariate_csv))
    ui.app._surv_cox()

    assert ui.text("surv_result").strip() == _expected_cox_report(result)
    assert ui.status() == "Cox summary complete"
    assert ui.messages == []


def test_cox_says_not_estimable_when_nothing_can_be_split(ui, tmp_path):
    path = tmp_path / "constant.csv"
    path.write_text("time,event,covariate\n" + "\n".join(
        f"{t},{e},2.0" for t, e in zip(TIMES, EVENTS)) + "\n", encoding="utf-8")
    df = pd.read_csv(path)
    result = cox_ph_summary(df["time"].values, df["event"].values, df["covariate"].values)
    assert result["hazard_ratio"] is None, "a constant covariate must be unestimable"

    ui.set("surv_path", str(path))
    ui.app._surv_cox()

    report = ui.text("surv_result")
    assert "Hazard ratio: not estimable" in report
    assert result["message"] in report
    assert ui.status() == "Cox summary complete"
    assert ui.messages == []


def test_cox_reports_a_non_numeric_covariate(ui, survival_csv):
    ui.clear("surv_result")

    ui.set("surv_path", str(survival_csv))
    ui.app._surv_cox()

    errors = ui.messages_of("error")
    assert len(errors) == 1, "the failure must be named, not swallowed"
    assert "numeric covariate" in errors[0], errors[0]
    assert ui.text("surv_result").strip() == ""


def test_cox_honours_a_custom_covariate_column(ui, tmp_path):
    path = tmp_path / "dose.csv"
    covariates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    path.write_text("time,event,dose\n" + "\n".join(
        f"{t},{e},{c}" for t, e, c in zip(TIMES, EVENTS, covariates)) + "\n",
        encoding="utf-8")
    ui.set("surv_cov_col", "dose")

    ui.set("surv_path", str(path))
    ui.app._surv_cox()

    assert ui.messages == []
    assert ui.status() == "Cox summary complete"


def test_cox_falls_back_to_the_default_column_when_the_entry_is_empty(ui, covariate_csv):
    ui.clear("surv_cov_col")

    ui.set("surv_path", str(covariate_csv))
    ui.app._surv_cox()

    assert ui.messages == []
    assert ui.status() == "Cox summary complete"
