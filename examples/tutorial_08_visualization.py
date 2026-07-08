"""
BioSuite Tutorial 8: Advanced Visualization

This notebook demonstrates advanced visualization options:
- Interactive Plotly plots
- Custom themes
- Export options
- Publication-quality figures
"""
# %%
# ## Setup

import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
from biosuite.plotting.plot_api import (
    volcano, pca, manhattan, heatmap, scatter, boxplot,
    violin, qqplot, timeseries, barplot, venn,
    export_all_interactive
)

print("BioSuite Advanced Visualization Tutorial")
print("=" * 50)

# %%
# ## 1. Interactive Volcano Plot

np.random.seed(42)
fc = np.random.normal(0, 1.5, 500)
pvals = np.random.uniform(0, 1, 500)
pvals[:30] = np.random.uniform(1e-6, 0.05, 30)
gene_names = [f"Gene_{i}" for i in range(500)]

fig = volcano(fc, pvals, gene_names=gene_names, interactive=True,
              title="Interactive Volcano Plot")
print("Created interactive volcano plot")
# fig.show()  # Uncomment in Jupyter
# fig.write_html("volcano_interactive.html")

# %%
# ## 2. Interactive PCA

data = np.random.randn(30, 50)
groups = ['Ctrl'] * 15 + ['Treat'] * 15

fig = pca(data, labels=groups, interactive=True, title="Interactive PCA")
print("Created interactive PCA plot")

# %%
# ## 3. Interactive Manhattan Plot

chroms = np.random.choice(['chr1', 'chr2', 'chr3', 'chr4', 'chr5'], 500)
positions = np.random.randint(1, 100000000, 500)
pvals = np.random.uniform(0, 1, 500)

fig = manhattan(chroms, positions, pvals, interactive=True,
                title="Interactive Manhattan Plot")
print("Created interactive Manhattan plot")

# %%
# ## 4. Interactive Heatmap

data = np.random.randn(20, 15)
row_labels = [f"Gene_{i}" for i in range(20)]
col_labels = [f"Sample_{i}" for i in range(15)]

fig = heatmap(data, row_labels=row_labels, col_labels=col_labels,
              interactive=True, title="Interactive Heatmap")
print("Created interactive heatmap")

# %%
# ## 5. Interactive Boxplot

data_dict = {
    'Control': np.random.randn(30).tolist(),
    'Treatment_A': (np.random.randn(30) + 1).tolist(),
    'Treatment_B': (np.random.randn(30) + 2).tolist(),
}

fig = boxplot(data_dict, interactive=True, title="Interactive Boxplot")
print("Created interactive boxplot")

# %%
# ## 6. Interactive Violin Plot

fig = violin(data_dict, interactive=True, title="Interactive Violin Plot")
print("Created interactive violin plot")

# %%
# ## 7. Interactive Q-Q Plot

pvals = np.random.uniform(0, 1, 200)
fig = qqplot(pvals, interactive=True, title="Interactive Q-Q Plot")
print("Created interactive Q-Q plot")

# %%
# ## 8. Interactive Scatter with Regression

x = np.random.randn(100)
y = x * 2 + np.random.randn(100) * 0.5

fig = scatter(x, y, interactive=True, show_regression=True,
              title="Interactive Scatter with Regression")
print("Created interactive scatter plot")

# %%
# ## 9. Interactive Barplot

categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 33]
errors = [3, 5, 4, 6, 3]

fig = barplot(categories, values, errors=errors, interactive=True,
              title="Interactive Barplot")
print("Created interactive barplot")

# %%
# ## 10. Interactive Time Series

x = np.arange(100)
ys = [np.sin(x / 10), np.cos(x / 10), np.sin(x / 10 + 1)]
names = ['sin', 'cos', 'sin shifted']

fig = timeseries(x, ys, names=names, interactive=True,
                 title="Interactive Time Series")
print("Created interactive time series")

# %%
# ## 11. Combine Multiple Plots into Report

plots = {
    'Volcano Plot': volcano(fc, pvals, interactive=True),
    'PCA Plot': pca(data, labels=groups, interactive=True),
    'Heatmap': heatmap(data[:10, :10], interactive=True),
}

# export_all_interactive(plots, "combined_report.html")
print(f"\nCombined {len(plots)} plots into a single report")

# %%
# ## 12. Static Plots (Matplotlib)

# Generate static plots
fig = volcano(fc, pvals, interactive=False, title="Static Volcano Plot")
fig = pca(data, labels=groups, interactive=False, title="Static PCA Plot")
fig = scatter(x, y, interactive=False, show_regression=True, title="Static Scatter")
print("\nGenerated static matplotlib plots")

# %%
# ## Export Options Summary

print("\nExport Options:")
print("  1. Interactive HTML: fig.write_html('plot.html')")
print("  2. Static PNG: fig.savefig('plot.png', dpi=150)")
print("  3. Static PDF: fig.savefig('plot.pdf')")
print("  4. Static SVG: fig.savefig('plot.svg')")
print("  5. Combined report: export_all_interactive(plots_dict, 'report.html')")

# %%
# ## Summary
#
# This tutorial covered:
# 1. Interactive Plotly plots (volcano, PCA, Manhattan, heatmap)
# 2. Boxplot, violin, Q-Q plot
# 3. Scatter with regression
# 4. Barplot and time series
# 5. Combining plots into reports
# 6. Static matplotlib plots
# 7. Export options
#
# Congratulations! You've completed all BioSuite tutorials.
