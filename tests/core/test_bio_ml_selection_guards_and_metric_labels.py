"""`select_features` refusals, the sklearn-free branch and honest metric labels.

Every value below was read off the running code; the docstrings record what the
same input produced before the fix.
"""
import logging
import re

import numpy as np
import pytest

from biosuite.core import bio_ml as ml
from biosuite.core.bio_ml import _metric_label, select_features


@pytest.fixture()
def three_column_matrix():
    # column 2 varies most, column 0 least (population variance, ddof=0)
    X = np.array([[1., 2., 3.], [2., 4., 6.], [3., 6., 9.], [4., 8., 12.]])
    y = [0, 1, 0, 1]
    return X, y


def test_variance_ranking_returns_exact_variances(three_column_matrix):
    X, y = three_column_matrix
    assert select_features(X, y, n_features=3, method='variance') == (
        [2, 1, 0], {2: 11.25, 1: 5.0, 0: 1.25})


def test_importance_ranking_sees_the_labels(three_column_matrix):
    X, y = three_column_matrix
    indices, scores = select_features(X, y, n_features=3, method='importance')
    assert indices[0] in (0, 1), "the two columns that separate the classes"
    assert sum(scores.values()) == pytest.approx(1.0), "importances are a distribution"


@pytest.mark.parametrize("bad, shown", [("varience", "varience"), ("importnce", "importnce"),
                                        ("Variance", "Variance"), ("", ""), ("rf", "rf"),
                                        (None, "None")])
def test_an_unknown_selection_method_is_refused(bad, shown, three_column_matrix):
    """Anything that was not exactly 'importance' used to fall into the variance
    branch, so a typo reported a different ranking than the caller asked for."""
    X, y = three_column_matrix
    with pytest.raises(
            ValueError,
            match=f"^Unknown method: {re.escape(shown)}"
                  r"\. Use 'importance' or 'variance'\.$"):
        select_features(X, y, method=bad)


@pytest.mark.parametrize("bad, rendered", [(0, "0"), (-2, "-2"), (2.5, "2.5")])
def test_a_useless_feature_count_is_refused(bad, rendered, three_column_matrix):
    """A negative n_features was a slice: -2 on a wide matrix dropped the last two
    ranks and returned nearly every column; 0 returned nothing at all."""
    X, y = three_column_matrix
    with pytest.raises(ValueError,
                       match=rf"n_features must be a positive integer, got "
                             rf"{re.escape(rendered)}\."):
        select_features(X, y, n_features=bad)


def test_a_numpy_integer_count_is_accepted(three_column_matrix):
    X, y = three_column_matrix
    assert select_features(X, y, n_features=np.int64(2), method='variance') == (
        [2, 1], {2: 11.25, 1: 5.0})


def test_asking_for_more_columns_than_exist_returns_every_one(three_column_matrix):
    X, y = three_column_matrix
    indices, scores = select_features(X, y, n_features=99, method='variance')
    assert indices == [2, 1, 0] and len(scores) == X.shape[1]


def test_the_variance_branch_needs_no_scikit_learn(monkeypatch, three_column_matrix):
    """The whole function answered ([], {}) without scikit-learn, even though
    ranking by column variance is pure numpy."""
    X, y = three_column_matrix
    monkeypatch.setattr(ml, "HAS_SKLEARN", False)
    assert select_features(X, y, n_features=2, method='variance') == ([2, 1],
                                                                     {2: 11.25, 1: 5.0})


def test_the_importance_branch_still_reports_no_engine(monkeypatch, three_column_matrix):
    X, y = three_column_matrix
    monkeypatch.setattr(ml, "HAS_SKLEARN", False)
    assert select_features(X, y, n_features=2) == ([], {})


# ── honest metric labels ─────────────────────────────────────────────────────

def test_metric_label_names_a_holdout_a_holdout():
    assert _metric_label("accuracy", 0.91666, True) == "holdout accuracy=0.917"


def test_metric_label_names_resubstitution_what_it_is():
    assert _metric_label("R²", 0.9874, False) == "in-sample R²=0.987"


def test_a_failed_split_is_reported_as_in_sample(monkeypatch):
    """`train_test_split` raises on data too small to split.  The fallback scored
    the rows the forest had just memorised and still called the figure a holdout
    accuracy."""
    def refuse(*args, **kwargs):
        raise ValueError("With n_samples=1 ...")

    monkeypatch.setattr(ml, "train_test_split", refuse)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array(['a', 'a', 'b'])
    result = ml.train_random_forest(X, y)
    assert "in-sample accuracy=" in result.message
    assert "holdout" not in result.message
    svm = ml.train_svm(X, y)
    assert "in-sample accuracy=" in svm.message
    reg = ml.train_random_forest_regressor(X, np.array([1.0, 2.0, 3.0]))
    assert "in-sample R²=" in reg.message


def test_a_real_split_is_still_reported_as_a_holdout():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    assert "holdout accuracy=" in ml.train_random_forest(X, y).message


# ── a swallowed failure must leave a trace ───────────────────────────────────

def test_a_shap_failure_is_logged_and_changes_nothing_else(monkeypatch, caplog):
    """The SHAP step used to end in `except Exception: pass`, so an explanation
    that never ran was indistinguishable from one that found nothing."""
    broken = type("mod", (), {"TreeExplainer": _boom})
    # shap is an optional extra: on a machine without it the module never binds
    # the name, so the patch has to be allowed to create it.
    monkeypatch.setattr(ml, "shap", broken, raising=False)
    monkeypatch.setattr(ml, "HAS_SHAP", True)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] > 0).astype(int)
    with caplog.at_level(logging.WARNING, logger="biosuite.core.bio_ml"):
        result = ml.train_random_forest(X, y, n_estimators=5)
    assert result.engine == "sklearn"
    assert result.shap_values is None
    assert result.accuracy > 0.5, "the metrics are still the honest holdout ones"
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "SHAP explanation failed for random_forest" in caplog.text
    assert "shap exploded" in caplog.text


def _boom(*args, **kwargs):
    raise RuntimeError("shap exploded")


def test_no_warning_is_logged_when_shap_is_not_installed(monkeypatch, caplog):
    """Positive control: the logging path above must not fire when SHAP is absent."""
    monkeypatch.setattr(ml, "HAS_SHAP", False)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] > 0).astype(int)
    with caplog.at_level(logging.WARNING, logger="biosuite.core.bio_ml"):
        result = ml.train_random_forest(X, y, n_estimators=5)
    assert result.shap_values is None
    assert caplog.records == [], "the missing engine is not a failure worth logging"
