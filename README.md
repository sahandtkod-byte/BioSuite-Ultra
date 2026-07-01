# BioSuite Pro v2.0

**An integrated bioinformatics platform for sequence analysis, visualization, and computational biology.**

![Theme: Dark-Green-Cyber](https://img.shields.io/badge/Theme-Dark--Green--Cyber-00ff88?style=flat-square)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

BioSuite is a desktop bioinformatics platform built with Python that integrates sequence analysis, pairwise alignment, phylogenetics, differential expression analysis, NGS/VCF processing, and 30+ publication-quality visualizations into a single unified application with a professional cyberpunk-themed GUI.

Designed for molecular cell biology researchers and bioinformatics students who need快速 (fast), accessible tools for common computational biology workflows without the overhead of learning R or command-line pipelines.

### Key Features

- **30+ Plot Types** — Volcano, PCA, Manhattan, MA, Venn, Heatmap, Scatter, GSEA, Motif Logo, Sankey, Circos, UMAP, Violin, Raincloud, Ridge, Dot Plot, and more
- **Sequence Analysis** — FASTA/FASTQ/GenBank parsing, GC content, reverse complement, translation (all 6 frames)
- **Pairwise Alignment** — Needleman-Wunsch (global) and Smith-Waterman (local) with vectorized numpy acceleration
- **Phylogenetics** — p-distance matrix, UPGMA tree construction, dendrogram visualization
- **Differential Expression** — CPM/TPM normalization, vectorized t-test DE analysis
- **NGS Support** — BAM/SAM parsing (via pysam), VCF variant analysis, Manhattan plots from VCF
- **Dual Interface** — Professional GUI (CustomTkinter) + interactive CLI menu
- **3 Cyberpunk Themes** — Dark-Green, Dark-Purple, Light-Blue with instant switching
- **Batch Export** — PDF reports, folder export, Markdown story generation

---

## Screenshots

### Plots Gallery
The main interface features a searchable plot gallery with 30+ visualizations organized by category.

### Sequence Analysis
Input FASTA/FASTQ sequences and compute GC content, reverse complement, and protein translation in real-time.

### Pairwise Alignment
Run Needleman-Wunsch or Smith-Waterman alignment directly from the GUI with results displayed instantly.

### Custom Themed Dialogs
All input dialogs use custom-themed cyberpunk-styled popups — no native Windows widgets.

---

## Architecture

```
BioSuite-Better/
├── run.py                          # Entry point (CLI, GUI, batch modes)
├── bioplatter/
│   ├── core/
│   │   ├── sequence.py             # FASTA/FASTQ/GenBank I/O, GC%, translation
│   │   ├── alignment.py            # NW/SW pairwise alignment (vectorized)
│   │   ├── phylogeny.py            # Distance matrix, UPGMA, dendrogram
│   │   ├── expression.py           # CPM/TPM normalization, DE analysis
│   │   ├── ngs.py                  # BAM/SAM/VCF parsing, Manhattan from VCF
│   │   └── utils.py                # Config, session, input helpers, theming
│   ├── plotting/
│   │   ├── biological_plots.py     # 18 biological/statistical plots
│   │   ├── math_plots.py           # 7 mathematical function plots
│   │   └── specialized_plots.py    # GSEA, Motif Logo, Sankey, UMAP
│   ├── gui/
│   │   └── main_window.py          # CustomTkinter GUI (3 themes, splash, dialogs)
│   └── cli/
│       └── menu.py                 # Interactive CLI menu (39 options)
├── tests/
│   ├── test_sequence.py            # Unit tests for sequence module
│   ├── test_alignment.py           # Unit tests for alignment algorithms
│   ├── test_expression.py          # Unit tests for DE analysis
│   └── test_physics.py             # Unit tests for phylogeny
└── requirements.txt
```

---

## Installation

### Requirements

- Python 3.10 or higher
- Operating System: Windows, macOS, or Linux

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# For GenBank file support
pip install biopython

# For BAM/SAM file support
pip install pysam

# For UMAP dimensionality reduction
pip install umap-learn
```

---

## Usage

### Launch GUI

```bash
python run.py --gui
```

### Launch CLI

```bash
python run.py
```

### Batch Export All Plots to PDF

```bash
python run.py --batch --pdf report.pdf
```

### Export All Plots to Folder

```bash
python run.py --export-folder myplots --story
```

### Edit Configuration

```bash
python run.py --config
```

---

## Technical Highlights

### Vectorized Alignment Algorithms

The Needleman-Wunsch and Smith-Waterman implementations use precomputed match/mismatch score matrices and row-wise numpy operations instead of per-cell Python loops:

```python
# Precompute all match/mismatch scores as a numpy matrix
match_scores = _match_array(seq1, seq2, match, mismatch)

# Fill each DP row with vectorized numpy operations
for i in range(1, n+1):
    diag = dp[i-1, :m] + match_scores[i-1]
    up = dp[i-1, 1:] + gap
    left = dp[i, :m] + gap
    dp[i, 1:] = np.maximum(diag, np.maximum(up, left))
```

**Result**: ~5-10x speedup for sequences >200bp compared to pure Python nested loops.

### Vectorized Differential Expression

Instead of iterating row-by-row with `pandas.iterrows()`, the DE analysis operates on entire numpy arrays at once:

```python
vals1 = counts_df.iloc[:, group1].values.astype(float)  # All group1 samples
vals2 = counts_df.iloc[:, group2].values.astype(float)  # All group2 samples
mean1 = np.mean(vals1 + 1, axis=1)                       # Vectorized mean
log2fc = np.log2(mean2 / mean1)                          # Vectorized log2FC
_, p = ttest_ind(vals1, vals2, axis=1)                   # Vectorized t-test
```

**Result**: Processes 20,000 genes in milliseconds instead of seconds.

### Custom Cyberpunk Dialog System

All GUI dialogs (input, confirm, message, file picker, dropdown) are custom-built with theme-matching colors, glow borders, and proper tkinter grab management:

```python
class _BaseDialog(ctk.CTkToplevel):
    def __init__(self, parent, T, ...):
        # Delayed grab_set prevents focus conflicts between consecutive popups
        self.after(50, self._do_grab)

    def _do_grab(self):
        try:
            self.grab_set()
        except Exception:
            pass
```

---

## Configuration

BioSuite stores settings in `bioplatter_config.json`:

```json
{
  "theme": "dark-green",
  "default_dpi": 180,
  "save_format": "png",
  "interactive": false,
  "downsample_threshold": 5000,
  "quiet": false
}
```

| Key | Options | Description |
|-----|---------|-------------|
| `theme` | `dark-green`, `dark-purple`, `light-blue` | GUI color theme |
| `default_dpi` | Any integer | Resolution for saved plots |
| `save_format` | `png`, `svg`, `pdf` | Default image format |
| `downsample_threshold` | Integer | Max points before downsampling |
| `quiet` | `true`/`false` | Suppress interactive prompts |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.24 | Numerical computation, vectorized alignment |
| pandas | >= 2.0 | Data manipulation, DataFrame operations |
| matplotlib | >= 3.7 | Plotting engine |
| seaborn | >= 0.12 | Statistical visualization |
| scipy | >= 1.10 | Statistics, clustering, dendrograms |
| scikit-learn | >= 1.3 | PCA decomposition |
| customtkinter | >= 5.2 | Modern GUI framework |
| tqdm | >= 4.65 | Progress bars for batch operations |

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=bioplatter

# Run specific test file
python -m pytest tests/test_sequence.py -v
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

**Sahand** — Molecular Cell Biology Student

Built as an integrated bioinformatics platform for academic research and computational biology coursework.
