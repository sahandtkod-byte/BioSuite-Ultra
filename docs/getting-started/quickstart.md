# Quick Start

## Sequence Analysis

```python
from biosuite.core.sequence import gc_content, reverse_complement, translate, sequence_stats

# GC content
gc = gc_content("ATCGATCGATCG")
print(f"GC: {gc}%")  # 50.0%

# Reverse complement
rc = reverse_complement("ATCG")
print(f"RC: {rc}")  # CGAT

# Translate DNA to protein
protein = translate("ATGAAATTTTAA")
print(f"Protein: {protein}")  # MKF*

# Full statistics
stats = sequence_stats("ATCGATCG")
for k, v in stats.items():
    print(f"  {k}: {v}")
```

## Streaming Large Files

```python
from biosuite.core.sequence import iter_fasta, iter_fastq

# Memory-efficient FASTA reading
for header, seq in iter_fasta("huge_genome.fasta"):
    if len(seq) > 1000:
        print(f"{header}: {len(seq)} bp")

# Memory-efficient FASTQ reading
for name, seq, qual in iter_fastq("reads.fastq"):
    print(f"{name}: {len(seq)} bp, avg qual: {sum(ord(c)-33 for c in qual)/len(qual):.1f}")
```

## Pairwise Alignment

```python
from biosuite.core.alignment import needleman_wunsch, smith_waterman

# Global alignment (Needleman-Wunsch)
aligned1, aligned2, score = needleman_wunsch("ACGT", "ACGACGT")
print(f"Score: {score}")

# Local alignment (Smith-Waterman)
aligned1, aligned2, score = smith_waterman("TTTTACGTTTTT", "ACGT")
print(f"Score: {score}")
```

## Differential Expression

```python
import pandas as pd
import numpy as np
from biosuite.core.expression import cpm_normalization, differential_expression

# Create count matrix
np.random.seed(42)
counts = pd.DataFrame(
    np.random.poisson(100, (1000, 6)),
    columns=["ctrl1", "ctrl2", "ctrl3", "treat1", "treat2", "treat3"],
    index=[f"gene_{i}" for i in range(1000)]
)

# Normalize
cpm = cpm_normalization(counts)
print(cpm.head())
```

## REST API

```bash
# Start the API server
export BIOSUITE_API_KEY=your-secret-key
export BIOSUITE_ADMIN_USER=admin
export BIOSUITE_ADMIN_PASSWORD=your-password
uvicorn biosuite.api:app --host 0.0.0.0 --port 8000

# Access documentation
open http://localhost:8000/docs
```

## Jupyter Notebook

```python
# Load magic commands
%load_ext biosuite.notebook.magics

# Quick analysis
%biosuite gc ATCGATCGATCG
%biosuite translate ATGAAATTTTAA

# Interactive widgets
from biosuite.notebook.widgets import SequenceAnalyzer
analyzer = SequenceAnalyzer()
analyzer.show()
```
