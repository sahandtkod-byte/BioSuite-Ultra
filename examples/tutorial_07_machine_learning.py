"""
BioSuite Tutorial 7: Machine Learning for Biology

This notebook demonstrates machine learning applications in bioinformatics:
- Random Forest classification
- SVM classification
- Feature importance
- Cross-validation
"""
# %%
# ## Setup

import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
from biosuite.core.bio_ml import (
    train_random_forest, train_svm, train_random_forest_regressor,
    select_features, compute_roc_curve, format_ml_report
)

print("BioSuite Machine Learning Tutorial")
print("=" * 50)

# %%
# ## 1. Create Sample Data

np.random.seed(42)
n_samples = 200
n_features = 50

# Generate features
X = np.random.randn(n_samples, n_features)

# Create labels based on first 5 features
y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)

# Add some noise
y[:20] = 1 - y[:20]  # Flip 20 labels

print(f"Dataset: {n_samples} samples, {n_features} features")
print(f"Classes: {np.unique(y)} (counts: {np.bincount(y)})")

# %%
# ## 2. Random Forest Classification

rf_result = train_random_forest(X, y, n_estimators=100)
print("\n" + format_ml_report(rf_result))

# %%
# ## 3. SVM Classification

svm_result = train_svm(X, y, kernel='rbf')
print("\n" + format_ml_report(svm_result))

# %%
# ## 4. Compare Models

print("\nModel Comparison:")
print(f"{'Model':<20} {'Accuracy':>10} {'CV Mean':>10} {'CV Std':>10}")
print("-" * 55)
print(f"{'Random Forest':<20} {rf_result.accuracy:>10.4f} "
      f"{np.mean(rf_result.cv_scores):>10.4f} {np.std(rf_result.cv_scores):>10.4f}")
print(f"{'SVM':<20} {svm_result.accuracy:>10.4f} "
      f"{np.mean(svm_result.cv_scores):>10.4f} {np.std(svm_result.cv_scores):>10.4f}")

# %%
# ## 5. Feature Importance

if rf_result.feature_importances:
    print("\nTop 10 Feature Importances (Random Forest):")
    for idx, imp in list(rf_result.feature_importances.items())[:10]:
        print(f"  Feature {idx}: {imp:.4f}")

# %%
# ## 6. Feature Selection

selected_idx, importances = select_features(X, y, n_features=10, method='importance')
print(f"\nSelected {len(selected_idx)} features: {selected_idx}")

# %%
# ## 7. Regression Example

# Generate continuous target
y_reg = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(n_samples) * 0.5

rf_reg = train_random_forest_regressor(X, y_reg)
print("\n" + format_ml_report(rf_reg))

# %%
# ## Summary
#
# This tutorial covered:
# 1. Random Forest classification
# 2. SVM classification
# 3. Model comparison
# 4. Feature importance
# 5. Feature selection
# 6. Random Forest regression
#
# Next: Tutorial 8 - Advanced Visualization
