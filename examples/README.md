# BioSuite Ultra — Examples

Runnable tutorials and notebooks that exercise the public API on small,
self-contained inputs. No private data and no large downloads are required.

## Running the tutorials

Each `tutorial_*.py` inserts the repository root into `sys.path`, so run it from
this directory:

```bash
cd examples
python tutorial_01_sequence_analysis.py
```

Alternatively install the package first (`pip install -e ".."`) and run the
script from anywhere. On a headless machine set `MPLBACKEND=Agg` so the plotting
tutorials do not try to open a window.

| Script | Topic | Dependencies |
| --- | --- | --- |
| `tutorial_01_sequence_analysis.py` | FASTA/FASTQ, GC content, reverse complement, translation | core |
| `tutorial_02_alignment.py` | Pairwise alignment (Needleman–Wunsch, Smith–Waterman) | core |
| `tutorial_03_differential_expression.py` | Count matrices and differential expression | core |
| `tutorial_04_single_cell.py` | Single-cell workflow | core + `scanpy`/`anndata` (optional; skipped gracefully if absent) |
| `tutorial_05_crispr.py` | CRISPR guide design | core |
| `tutorial_06_metagenomics.py` | Metagenomics profiling | core |
| `tutorial_07_machine_learning.py` | Biological machine-learning helpers | core (scikit-learn) |
| `tutorial_08_visualization.py` | Plot catalogue | core (matplotlib) |

"core" means the package's install-time dependencies; no extra install is needed.

## Notebooks

`01_sequence_analysis.ipynb` through `05_gwas_and_epitope.ipynb` cover the same
topics in Jupyter form. Open them with `jupyter notebook` or `jupyter lab`; they
use the same optional extras as the matching tutorials.

## A note on scientific scope

The built-in engines are approximations intended for teaching and exploration.
Where an optional external tool is installed it is used in preference to the
built-in implementation. See the top-level `README.md` for the full
scientific-scope and validation statement.