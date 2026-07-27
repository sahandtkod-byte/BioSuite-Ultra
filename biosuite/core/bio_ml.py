"""
Machine Learning for Biology.

Random Forest, SVM, cross-validation, ROC curves, and SHAP interpretability.
All pip-installable via scikit-learn and shap.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                                 confusion_matrix, classification_report,
                                 mean_squared_error, r2_score)
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


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


def check_ml_tools():
    return {'sklearn': HAS_SKLEARN, 'shap': HAS_SHAP}


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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_scaled, y_encoded)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')

    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled) if len(le.classes_) > 1 else None

    acc = accuracy_score(y_encoded, y_pred)
    auc = roc_auc_score(y_encoded, y_prob[:, 1]) if y_prob is not None and y_prob.shape[1] == 2 else 0.0

    # Feature importances
    importances = dict(enumerate(model.feature_importances_))
    top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20])

    # SHAP values
    shap_vals = None
    if HAS_SHAP and X.shape[0] <= 500:
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_scaled[:100])
        except Exception:  # SHAP can fail for many model types
            pass

    return MLResult(
        model_type='random_forest',
        engine='sklearn',
        accuracy=round(float(acc), 4),
        auc=round(float(auc), 4),
        cv_scores=[round(float(s), 4) for s in cv_scores],
        feature_importances=top_features,
        shap_values=shap_vals,
        confusion_mat=confusion_matrix(y_encoded, y_pred),
        classification_rep=classification_report(y_encoded, y_pred, target_names=le.classes_),
        predictions=y_pred,
        message=f"Random Forest: accuracy={acc:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel=kernel, probability=True, random_state=random_state)
    model.fit(X_scaled, y_encoded)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled) if len(le.classes_) > 1 else None

    acc = accuracy_score(y_encoded, y_pred)
    auc = roc_auc_score(y_encoded, y_prob[:, 1]) if y_prob is not None and y_prob.shape[1] == 2 else 0.0

    return MLResult(
        model_type='svm',
        engine='sklearn',
        accuracy=round(float(acc), 4),
        auc=round(float(auc), 4),
        cv_scores=[round(float(s), 4) for s in cv_scores],
        confusion_mat=confusion_matrix(y_encoded, y_pred),
        predictions=y_pred,
        message=f"SVM ({kernel}): accuracy={acc:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    importances = dict(enumerate(model.feature_importances_))
    top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20])

    return MLResult(
        model_type='rf_regressor',
        engine='sklearn',
        accuracy=round(float(r2), 4),
        feature_importances=top_features,
        predictions=y_pred,
        message=f"RF Regressor: R²={r2:.3f}, RMSE={rmse:.3f}"
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
    """Feature selection using Random Forest importance.

    Args:
        X: feature matrix.
        y: labels.
        n_features: number of top features to select.
        method: selection method ('importance' or 'variance').

    Returns:
        Indices of selected features and their importance scores.
    """
    if not HAS_SKLEARN:
        return [], {}

    if method == 'importance':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
        return indices.tolist(), {int(i): round(float(importances[i]), 4) for i in indices}
    else:
        variances = np.var(X, axis=0)
        indices = np.argsort(variances)[::-1][:n_features]
        return indices.tolist(), {int(i): round(float(variances[i]), 4) for i in indices}


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
