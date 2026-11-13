"""
Machine Learning for Biology.

Random Forest, SVM, cross-validation, ROC curves, and SHAP interpretability.
All pip-installable via scikit-learn and shap.
"""
from dataclasses import dataclass, field

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        mean_squared_error,
        r2_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_score,
        train_test_split,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import SVC
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from .log import get_logger

logger = get_logger(__name__)


@dataclass
class MLResult:
    model_type: str
    engine: str
    accuracy: float = 0.0
    auc: float = 0.0
    cv_scores: list = field(default_factory=list)
    feature_importances: dict = field(default_factory=dict)
    shap_values: np.ndarray = None
    confusion_mat: np.ndarray = None
    classification_rep: str = ""
    predictions: np.ndarray = None
    message: str = ""
    # Positive-class holdout scores, paired row-for-row with eval_labels.
    probabilities: np.ndarray = None
    eval_labels: np.ndarray = None


def check_ml_tools():
    return {'sklearn': HAS_SKLEARN, 'shap': HAS_SHAP}


def _metric_label(kind, value, holdout):
    """Name a score by how it was really obtained.

    ``train_test_split`` falls back to scoring on the rows the model was fitted
    on when the dataset cannot be split, and labelling that "holdout accuracy"
    presented a resubstitution figure as an honest estimate.
    """
    return f"{'holdout' if holdout else 'in-sample'} {kind}={value:.3f}"


def train_random_forest(X, y, n_estimators=100, test_size=0.2, random_state=42):
    """Train Random Forest classifier with cross-validation.

    Args:
        X: feature matrix (samples x features).
        y: labels.
        n_estimators: number of trees.
        test_size: fraction for test set.
        random_state: random seed.

    Returns:
        MLResult with metrics and feature importances.
    """
    if not HAS_SKLEARN:
        return MLResult(model_type='random_forest', engine='none',
                       message="scikit-learn not installed")

    X = np.array(X)
    y = np.array(y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Honest holdout split (test_size was accepted but silently ignored,
    # so the reported accuracy/AUC were resubstitution metrics on the data
    # the forest had memorised).  Falls back to in-sample fit when the
    # dataset is too small or a class has a single member for stratification.
    try:
        counts = np.bincount(y_encoded)
        stratify = y_encoded if counts.min() >= 2 else None
        X_fit, X_eval, y_fit, y_eval = train_test_split(
            X, y_encoded, test_size=test_size,
            random_state=random_state, stratify=stratify)
    except ValueError:
        X_fit, X_eval, y_fit, y_eval = X, X, y_encoded, y_encoded

    # The scaler is a step inside the estimator, so it is fitted on the training
    # part of each split only: a fit over the whole matrix lets held-out rows shape
    # the transform they are scored under (F2-1).
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("forest", RandomForestClassifier(n_estimators=n_estimators,
                                          random_state=random_state)),
    ])
    model.fit(X_fit, y_fit)

    # Cross-validation refits the whole pipeline per fold, from raw X.  Every
    # class needs at least n_splits members for StratifiedKFold, and the split
    # above already gave up on stratifying a single-member class — so when that
    # happens the honest answer is "no CV", not a ValueError escaping the GUI's
    # Train button.
    min_class = int(np.bincount(y_encoded).min())
    if min_class >= 2:
        n_splits = int(min(5, max(2, min_class)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')
    else:
        cv_scores = np.array([])

    # Held-out predictions
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval) if len(le.classes_) > 1 else None
    binary_prob = y_prob[:, 1] if y_prob is not None and y_prob.shape[1] == 2 else None

    acc = accuracy_score(y_eval, y_pred)
    auc = roc_auc_score(y_eval, binary_prob) if binary_prob is not None else 0.0

    # Feature importances
    forest = model.named_steps["forest"]
    importances = dict(enumerate(forest.feature_importances_))
    top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20])

    # SHAP values
    shap_vals = None
    if HAS_SHAP and X.shape[0] <= 500:
        try:
            explainer = shap.TreeExplainer(forest)
            shap_vals = explainer.shap_values(
                model.named_steps["scaler"].transform(X)[:100])
        except Exception as exc:
            # SHAP supports a moving set of estimators; a failure here must not
            # change the reported metrics, but it has to be visible in the log
            # instead of disappearing into a bare ``pass``.
            logger.warning("SHAP explanation failed for random_forest: %s", exc)

    return MLResult(
        model_type='random_forest',
        engine='sklearn',
        accuracy=round(float(acc), 4),
        auc=round(float(auc), 4),
        cv_scores=[round(float(s), 4) for s in cv_scores],
        feature_importances=top_features,
        shap_values=shap_vals,
        confusion_mat=confusion_matrix(y_eval, y_pred),
        # The eval split of a small dataset may hold only one class, and
        # classification_report then refuses the full target_names list; naming
        # the labels explicitly keeps the report describing all classes.
        classification_rep=classification_report(
            y_eval, y_pred, labels=list(range(len(le.classes_))),
            target_names=[str(c) for c in le.classes_], zero_division=0),
        predictions=y_pred,
        probabilities=binary_prob,
        eval_labels=y_eval,
        message=(f"Random Forest: {_metric_label('accuracy', acc, X_fit is not X)}"
                 + (f", CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
                    if cv_scores.size else ", CV=not possible (a class has <2 samples)"))
    )


def train_svm(X, y, kernel='rbf', test_size=0.2, random_state=42):
    """Train SVM classifier.

    Args:
        X: feature matrix.
        y: labels.
        kernel: kernel type ('rbf', 'linear', 'poly').
        test_size: fraction for test set.
        random_state: random seed.

    Returns:
        MLResult with metrics.
    """
    if not HAS_SKLEARN:
        return MLResult(model_type='svm', engine='none', message="scikit-learn not installed")

    X = np.array(X)
    y = np.array(y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    try:
        counts = np.bincount(y_encoded)
        stratify = y_encoded if counts.min() >= 2 else None
        X_fit, X_eval, y_fit, y_eval = train_test_split(
            X, y_encoded, test_size=test_size,
            random_state=random_state, stratify=stratify)
    except ValueError:
        X_fit, X_eval, y_fit, y_eval = X, X, y_encoded, y_encoded

    # An SVM is not scale-free, so the standardisation matters here — learned from
    # the training part of each split only (F2-1).
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, probability=True, random_state=random_state)),
    ])
    model.fit(X_fit, y_fit)

    min_class = int(np.bincount(y_encoded).min())
    if min_class >= 2:
        n_splits = int(min(5, max(2, min_class)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')
    else:
        cv_scores = np.array([])

    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval) if len(le.classes_) > 1 else None
    binary_prob = y_prob[:, 1] if y_prob is not None and y_prob.shape[1] == 2 else None

    acc = accuracy_score(y_eval, y_pred)
    auc = roc_auc_score(y_eval, binary_prob) if binary_prob is not None else 0.0

    return MLResult(
        model_type='svm',
        engine='sklearn',
        accuracy=round(float(acc), 4),
        auc=round(float(auc), 4),
        cv_scores=[round(float(s), 4) for s in cv_scores],
        confusion_mat=confusion_matrix(y_eval, y_pred),
        predictions=y_pred,
        probabilities=binary_prob,
        eval_labels=y_eval,
        message=(f"SVM ({kernel}): {_metric_label('accuracy', acc, X_fit is not X)}"
                 + (f", CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
                    if cv_scores.size else ", CV=not possible (a class has <2 samples)"))
    )


def train_random_forest_regressor(X, y, n_estimators=100, random_state=42):
    """Train Random Forest regressor for continuous outcomes.

    Args:
        X: feature matrix.
        y: continuous target values.
        n_estimators: number of trees.
        random_state: random seed.

    Returns:
        MLResult with R² and RMSE.
    """
    if not HAS_SKLEARN:
        return MLResult(model_type='rf_regressor', engine='none', message="scikit-learn not installed")

    X = np.array(X)
    y = np.array(y)

    try:
        X_fit, X_eval, y_fit, y_eval = train_test_split(
            X, y, test_size=0.2, random_state=random_state)
    except ValueError:
        X_fit, X_eval, y_fit, y_eval = X, X, y, y

    # Same reason as the classifiers: the scaling must not see the scored rows.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=n_estimators,
                                            random_state=random_state)),
    ])
    model.fit(X_fit, y_fit)

    y_pred = model.predict(X_eval)
    r2 = r2_score(y_eval, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))

    importances = dict(enumerate(model.named_steps["regressor"].feature_importances_))
    top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20])

    return MLResult(
        model_type='rf_regressor',
        engine='sklearn',
        accuracy=round(float(r2), 4),
        feature_importances=top_features,
        predictions=y_pred,
        message=(f"RF Regressor: {_metric_label('R²', r2, X_fit is not X)}, "
                 f"RMSE={rmse:.3f}")
    )


def compute_roc_curve(y_true, y_prob):
    """Compute ROC curve data points.

    Args:
        y_true: true binary labels.
        y_prob: predicted probabilities.

    Returns:
        Dict with fpr, tpr, thresholds, and auc.
    """
    if not HAS_SKLEARN:
        return {}
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'auc': round(float(auc), 4)
    }


def select_features(X, y, n_features=20, method='importance'):
    """Rank feature columns and return the top ``n_features`` of them.

    Args:
        X: feature matrix (samples x features).
        y: labels — only ``method='importance'`` looks at them.
        n_features: how many columns to return, at least 1.  Asking for more
            columns than exist returns every column.
        method: 'importance' ranks by Random Forest impurity decrease (needs
            scikit-learn); 'variance' ranks by column variance (numpy only).

    Returns:
        Indices of the selected features in rank order, and a dict of their
        scores.  ``([], {})`` when 'importance' is asked for on a machine without
        scikit-learn.

    Raises:
        ValueError: On an unknown ``method`` or an ``n_features`` below 1.

    Note:
        The importance ranking sees ``y``, so it has to run inside each
        cross-validation fold.  Selecting on the full labelled matrix and then
        scoring those same rows leaks the labels into the test set.
    """
    if method not in ('importance', 'variance'):
        # Anything else used to fall through to the variance branch, so
        # method='varience' silently returned a different ranking than requested.
        raise ValueError(f"Unknown method: {method}. Use 'importance' or 'variance'.")
    if not isinstance(n_features, (int, np.integer)) or n_features < 1:
        # A negative value was a slice: n_features=-2 dropped the last two ranks
        # and returned nearly every column of a wide matrix.
        raise ValueError(f"n_features must be a positive integer, got {n_features!r}.")

    if method == 'variance':
        # Pure numpy: the whole function used to answer ([], {}) whenever
        # scikit-learn was missing, even on this branch.
        scores = np.var(np.asarray(X, dtype=float), axis=0)
    else:
        if not HAS_SKLEARN:
            return [], {}
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        scores = model.feature_importances_

    indices = np.argsort(scores)[::-1][:n_features]
    return indices.tolist(), {int(i): round(float(scores[i]), 4) for i in indices}


def format_ml_report(result):
    lines = [
        "=== Machine Learning Report ===",
        f"Model: {result.model_type}",
        f"Engine: {result.engine}",
        f"Accuracy: {result.accuracy:.4f}",
    ]
    if result.auc > 0:
        lines.append(f"AUC: {result.auc:.4f}")
    if result.cv_scores:
        mean_cv = np.mean(result.cv_scores)
        std_cv = np.std(result.cv_scores)
        lines.append(f"CV Accuracy: {mean_cv:.4f} ± {std_cv:.4f}")
    if result.feature_importances:
        lines.append("\nTop feature importances:")
        for idx, imp in list(result.feature_importances.items())[:5]:
            lines.append(f"  Feature {idx}: {imp:.4f}")
    if result.message:
        lines.append(f"\nNote: {result.message}")
    return '\n'.join(lines)
