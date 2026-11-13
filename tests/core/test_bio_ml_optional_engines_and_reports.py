"""`bio_ml`: the missing-engine answers, the split fallback, SHAP, the report.

scikit-learn is installed here and shap is not, so half of this module's
branches — the engine='none' refusals, the in-sample fallback when a split
cannot be made, the SHAP block, and every optional line of the report — are
invisible to a plain test run.  They are reached by re-executing the module with
the optional import redirected (the same technique the sequence readers use) and
by standing in a fake `shap`, so the *wrapper's* decisions get asserted rather
than the third-party library's.

The cross-validation guard is pinned too: a dataset with a single-member class
used to raise ``n_splits=2 cannot be greater than the number of members in each
class`` out of the GUI's Train button, while the docstring promised a fallback.
"""
import importlib.util
import os
import sys
import types
import warnings

import numpy as np
import pytest

from biosuite.core import bio_ml as bm
from biosuite.core.bio_ml import (
    MLResult,
    check_ml_tools,
    compute_roc_curve,
    format_ml_report,
    select_features,
    train_random_forest,
    train_random_forest_regressor,
    train_svm,
)

MODULE_PATH = os.path.abspath(bm.__file__)


def reexec(monkeypatch, block=(), provide=()):
    """Re-import biosuite.core.bio_ml with optional dependencies redirected."""
    for name in block:
        monkeypatch.setitem(sys.modules, name, None)
    for name, module in provide:
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "biosuite.core.bio_ml_probe", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def separable():
    """Twelve samples in two well separated classes."""
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(-3, 0.4, size=(6, 3)), rng.normal(3, 0.4, size=(6, 3))])
    y = np.array(["a"] * 6 + ["b"] * 6)
    return X, y


@pytest.fixture()
def quiet_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ── the engine flags ────────────────────────────────────────────────────────

def test_check_ml_tools_reports_both_optional_dependencies():
    tools = check_ml_tools()
    assert set(tools) == {"sklearn", "shap"}
    assert tools["sklearn"] is True, "scikit-learn is installed in this environment"
    assert tools["shap"] is False, "shap is not installed here; the fake is only for tests"


def test_the_module_survives_without_scikit_learn(monkeypatch):
    """The ImportError branch, which the installed environment never takes."""
    mod = reexec(monkeypatch, block=("sklearn.ensemble", "sklearn.metrics",
                                     "sklearn.model_selection", "sklearn.pipeline",
                                     "sklearn.preprocessing", "sklearn.svm"))
    assert mod.HAS_SKLEARN is False
    assert mod.check_ml_tools()["sklearn"] is False


def test_the_module_reports_shap_when_it_is_importable(monkeypatch):
    fake = types.ModuleType("shap")
    fake.TreeExplainer = object
    mod = reexec(monkeypatch, provide=[("shap", fake)])
    assert mod.HAS_SHAP is True
    assert mod.check_ml_tools() == {"sklearn": True, "shap": True}


def test_the_classifiers_refuse_without_scikit_learn(monkeypatch):
    """The `engine='none'` guards: the honest answer on a machine that has no
    scikit-learn is a result saying so, not a NameError from a missing import."""
    X = [[1.0, 2.0], [3.0, 4.0]]
    y = ["a", "b"]
    monkeypatch.setattr(bm, "HAS_SKLEARN", False)
    for name, call in (("random_forest", lambda: train_random_forest(X, y)),
                       ("svm", lambda: train_svm(X, y)),
                       ("rf_regressor",
                        lambda: train_random_forest_regressor(X, [1.0, 2.0]))):
        res = call()
        assert res.engine == "none", name
        assert res.message == "scikit-learn not installed", name
        assert res.model_type == name, name
        assert res.accuracy == 0.0 and res.cv_scores == [], name


def test_roc_and_selection_answer_without_scikit_learn(monkeypatch):
    monkeypatch.setattr(bm, "HAS_SKLEARN", False)
    assert compute_roc_curve([0, 1], [0.2, 0.9]) == {}
    assert select_features([[1.0, 2.0], [3.0, 4.0]], ["a", "b"]) == ([], {})


# ── the split fallback ──────────────────────────────────────────────────────

def test_a_split_that_cannot_be_made_fits_and_scores_in_sample(monkeypatch, separable,
                                                                quiet_warnings):
    """`train_test_split` raises ValueError on a single-row matrix; the fallback
    must produce a result, not lose the run."""
    X, y = separable

    def refuse(*args, **kwargs):
        raise ValueError("With n_samples=1 ...")

    monkeypatch.setattr(bm, "train_test_split", refuse)
    res = train_random_forest(X, y, n_estimators=5)
    assert res.engine == "sklearn"
    assert res.accuracy == 1.0, "in-sample scoring of a separable set"
    assert len(res.cv_scores) == 5
    assert "in-sample accuracy=1.000" in res.message
    assert "holdout" not in res.message, \
        "the split failed, so calling the figure a holdout score would be false"


def test_the_svm_falls_back_the_same_way(monkeypatch, separable, quiet_warnings):
    X, y = separable

    def refuse(*args, **kwargs):
        raise ValueError("no split for you")

    monkeypatch.setattr(bm, "train_test_split", refuse)
    res = train_svm(X, y)
    assert res.engine == "sklearn"
    assert res.accuracy == 1.0
    assert res.auc == 1.0


def test_the_regressor_falls_back_the_same_way(monkeypatch, quiet_warnings):
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = X[:, 0] * 2.0 + X[:, 1]

    def refuse(*args, **kwargs):
        raise ValueError("cannot split")

    monkeypatch.setattr(bm, "train_test_split", refuse)
    res = train_random_forest_regressor(X, y)
    assert res.engine == "sklearn"
    assert res.accuracy > 0.9, "R² reported in the accuracy field"
    assert "in-sample R²" in res.message and "holdout" not in res.message


# ── small datasets ───────────────────────────────────────────────────────────

def test_a_class_with_one_member_still_returns_a_result(separable, quiet_warnings):
    """Regression: this raised `n_splits=2 cannot be greater than the number of
    members in each class` through the GUI's exception handler."""
    X, y = separable
    y = np.array(list(y) + ["c"])
    X = np.vstack([X, [[0.0, 0.0, 0.0]]])
    res = train_random_forest(X, y, n_estimators=5)
    assert res.engine == "sklearn"
    assert res.cv_scores == [], "a 1-member class cannot be cross-validated"
    assert "CV=not possible" in res.message
    assert "holdout accuracy=" in res.message
    for name in ("a", "b", "c"):
        assert name in res.classification_rep, "the report must keep every class"


def test_two_rows_of_two_classes_is_answered_not_crashed(quiet_warnings):
    res = train_random_forest([[1.0, 2.0], [3.0, 4.0]], ["a", "b"], n_estimators=5)
    assert res.engine == "sklearn"
    assert res.cv_scores == []
    assert res.confusion_mat.shape == (2, 2)


def test_a_single_class_dataset_is_not_called_discriminating(separable, quiet_warnings):
    X, y = separable
    res = train_random_forest(X, ["healthy"] * 12, n_estimators=5)
    assert res.accuracy == 1.0
    assert res.auc == 0.0, "no AUC exists for one class"
    # `predictions` are the LabelEncoder indices over the holdout rows, not the
    # class names and not the whole dataset — the model is fitted on the encoded
    # vector and scores only the split.
    assert 0 < len(res.predictions) < 12
    assert res.predictions.tolist() == [0] * len(res.predictions)
    assert res.feature_importances and set(res.feature_importances) == {0, 1, 2}


def test_the_svm_also_gives_up_on_cross_validation_for_a_thin_class(quiet_warnings):
    """The same guard as the forest: 11 + 1 samples cannot be stratified into
    folds, and an SVM has to answer with a result instead of a traceback."""
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(-3, 0.3, size=(11, 2)), rng.normal(3, 0.3, size=(1, 2))])
    y = np.array(["a"] * 11 + ["b"])
    res = train_svm(X, y)
    assert res.engine == "sklearn"
    assert res.cv_scores == []
    assert "CV=not possible" in res.message


# ── SHAP ────────────────────────────────────────────────────────────────────

def test_shap_values_are_attached_when_the_library_is_present(monkeypatch, separable,
                                                               quiet_warnings):
    X, y = separable
    seen = {}

    class FakeExplainer:
        def __init__(self, forest):
            seen["forest"] = forest

        def shap_values(self, matrix):
            seen["rows"] = matrix.shape[0]
            return np.zeros((matrix.shape[0], matrix.shape[1]))

    fake = types.ModuleType("shap")
    fake.TreeExplainer = FakeExplainer
    monkeypatch.setattr(bm, "HAS_SHAP", True)
    monkeypatch.setattr(bm, "shap", fake, raising=False)

    res = train_random_forest(X, y, n_estimators=5)
    assert res.shap_values is not None
    assert res.shap_values.shape == (12, 3)
    assert seen["rows"] == 12
    assert seen["forest"] is not None


def test_shap_is_skipped_for_a_large_dataset(monkeypatch, quiet_warnings):
    """The cap is a deliberate cost guard: TreeExplainer on thousands of rows is
    slow, and the classifier result must still come back."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(501, 2))
    y = np.array(["a", "b"] * 250 + ["a"])
    called = []

    class FakeExplainer:
        def __init__(self, forest):
            called.append(1)

        def shap_values(self, matrix):
            return np.zeros((1, 1))

    fake = types.ModuleType("shap")
    fake.TreeExplainer = FakeExplainer
    monkeypatch.setattr(bm, "HAS_SHAP", True)
    monkeypatch.setattr(bm, "shap", fake, raising=False)
    res = train_random_forest(X, y, n_estimators=3)
    assert res.shap_values is None
    assert called == []


def test_a_shap_failure_degrades_to_no_explanations(monkeypatch, separable,
                                                    quiet_warnings):
    X, y = separable

    class Exploding:
        def __init__(self, forest):
            raise RuntimeError("shap cannot handle this forest")

    fake = types.ModuleType("shap")
    fake.TreeExplainer = Exploding
    monkeypatch.setattr(bm, "HAS_SHAP", True)
    monkeypatch.setattr(bm, "shap", fake, raising=False)
    res = train_random_forest(X, y, n_estimators=5)
    assert res.shap_values is None
    assert res.engine == "sklearn", "interpretability is optional, the fit is not"


def test_no_shap_here_means_the_field_stays_none(separable, quiet_warnings):
    X, y = separable
    assert train_random_forest(X, y, n_estimators=5).shap_values is None


# ── roc and feature selection ───────────────────────────────────────────────

def test_roc_curve_reports_points_and_auc():
    roc = compute_roc_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert sorted(roc) == ["auc", "fpr", "thresholds", "tpr"]
    assert roc["auc"] == 1.0
    assert roc["fpr"][0] == 0.0 and roc["tpr"][-1] == 1.0
    assert len(roc["fpr"]) == len(roc["tpr"]) == len(roc["thresholds"])


def test_roc_curve_is_a_perfect_inverse_for_bad_probabilities():
    roc = compute_roc_curve([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2])
    assert roc["auc"] == 0.0


def test_select_features_prefers_the_informative_column(separable):
    X, y = separable
    X = np.hstack([X, rng_noise(X)])
    indices, scores = select_features(X, y, n_features=3)
    assert len(indices) == 3
    assert set(scores) == set(indices)
    assert scores[indices[0]] >= scores[indices[1]] >= scores[indices[2]]
    assert 0 in indices or 1 in indices, "the separating columns must rank first"


def rng_noise(base):
    rng = np.random.default_rng(5)
    return rng.normal(size=(base.shape[0], 2))


def test_select_features_by_variance_does_not_look_at_the_labels(separable):
    """The variance method is unsupervised, so relabelling must not change which
    columns it returns — the one property that separates it from 'importance'."""
    X, y = separable
    idx_a, scores_a = select_features(X, y, n_features=2, method="variance")
    idx_b, scores_b = select_features(X, list(reversed(list(y))), n_features=2,
                                      method="variance")
    assert idx_a == idx_b
    assert scores_a == scores_b
    assert len(idx_a) == 2
    assert all(scores_a[i] >= 0 for i in scores_a)


def test_select_features_returns_fewer_indices_than_asked_for(separable):
    X, y = separable
    indices, scores = select_features(X, y, n_features=99)
    assert len(indices) == X.shape[1] == len(scores)


# ── the report ──────────────────────────────────────────────────────────────

def test_report_of_a_bare_result_prints_only_the_mandatory_lines():
    text = format_ml_report(MLResult(model_type="random_forest", engine="none",
                                     accuracy=0.0))
    assert text == ("=== Machine Learning Report ===\nModel: random_forest\n"
                    "Engine: none\nAccuracy: 0.0000")
    assert "AUC" not in text and "CV Accuracy" not in text
    assert "Top feature importances" not in text and "Note:" not in text


def test_report_prints_every_optional_block_when_it_is_filled():
    result = MLResult(model_type="random_forest", engine="sklearn", accuracy=0.9167,
                      auc=0.95, cv_scores=[0.9, 0.9, 0.9], feature_importances={2: 0.5,
                                                                               0: 0.3},
                      message="Random Forest: holdout accuracy=0.917")
    text = format_ml_report(result)
    assert "AUC: 0.9500" in text
    assert "CV Accuracy: 0.9000 ± 0.0000" in text
    assert "Feature 2: 0.5000" in text
    assert "Feature 0: 0.3000" in text
    assert text.index("Feature 2") < text.index("Feature 0"), "insertion order is kept"
    assert text.endswith("Note: Random Forest: holdout accuracy=0.917")


def test_report_caps_the_importance_list_at_five():
    result = MLResult(model_type="rf", engine="sklearn",
                      feature_importances={i: 1.0 / (i + 1) for i in range(9)})
    text = format_ml_report(result)
    assert text.count("Feature ") == 5


def test_a_zero_auc_is_not_printed_as_a_metric():
    assert "AUC" not in format_ml_report(MLResult(model_type="rf", engine="sklearn",
                                                  auc=0.0))
