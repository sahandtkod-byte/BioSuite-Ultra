# BioSuite Ultra

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-150%20passing-brightgreen?logo=pytest&logoColor=white)
![Modules](https://img.shields.io/badge/Modules-96-orange?logo=python&logoColor=white)
![Version](https://img.shields.io/badge/Version-5.0.2-blueviolet?logo=pypi&logoColor=white)
![PyPI](https://img.shields.io/pypi/v/biosuite-ultra?logo=pypi&logoColor=white)
![Downloads](https://img.shields.io/pypi/dm/biosuite-ultra?logo=pypi&logoColor=white)
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21256296.svg)
![JOSS](https://img.shields.io/badge/JOSS-Submitted-blue.svg)

<p align="center">
  <strong>🧬 The most comprehensive open-source bioinformatics platform</strong>
</p>

<p align="center">
  96 analysis modules • 26 visualization types • Cyberpunk GUI • REST API • Jupyter Integration
</p>

<p align="center">
  <a href="https://pypi.org/project/biosuite-ultra/">
    <img src="https://img.shields.io/badge/Install-pip%20install%20biosuite--ultra-brightgreen" alt="Install">
  </a>
  <a href="https://github.com/sahandtkod-byte/BioSuite-Ultra/releases">
    <img src="https://img.shields.io/github/v/release/sahandtkod-byte/BioSuite-Ultra?logo=github" alt="GitHub Release">
  </a>
</p>

---

## ✨ What is BioSuite Ultra?

BioSuite Ultra is a **full-stack bioinformatics platform** built in pure Python. It provides everything you need for computational biology research:

- 🧬 **96 Analysis Modules** - From sequence analysis to structural biology
- 📊 **26 Visualization Types** - Publication-ready plots with 123+ functions
- 🖥️ **Cyberpunk GUI** - Beautiful, modern interface with 29 tabs
- ⌨️ **CLI with 99+ Options** - Command-line power for automation
- 🔌 **REST API** - 42+ endpoints for web integration
- 📓 **Jupyter Integration** - Magic commands and widgets
- 🐳 **Docker Ready** - One-click deployment

**No external bioinformatics tools required. 100% free and open-source.**

---

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install biosuite-ultra

# Or install with all optional dependencies
pip install biosuite-ultra[all]
```

### GUI Mode

```bash
# Launch the GUI
python -m biosuite

# Or use the entry point
biosuite
```

### CLI Mode

```bash
# Interactive menu
python run.py

# Direct commands
biosuite sequence --input data.fasta
biosuite expression --counts counts.csv --conditions tumor,normal
```

### REST API

```bash
# Start the API server
python -m biosuite.api

# Access at http://localhost:8000
```

### Docker

```bash
# Pull and run
docker pull sahandtkod/biosuite-ultra:latest
docker run -p 8000:8000 sahandtkod/biosuite-ultra

# Or use docker-compose
docker-compose up
```

---

## 🧬 Features

### 96 Analysis Modules

| Category | Modules | Description |
|----------|---------|-------------|
| **Sequence Analysis** | 12 | FASTA/FASTQ I/O, GC content, translation, ORF finding |
| **Alignment** | 8 | Needleman-Wunsch, Smith-Waterman, MSA, BLAST |
| **Phylogeny** | 6 | Neighbor-Joining, Bayesian, ML, consensus trees |
| **Genomics** | 15 | Variant calling, NGS, assembly, gene prediction |
| **Transcriptomics** | 8 | Differential expression, normalization, enrichment |
| **Proteomics** | 6 | Structure prediction, docking, motif analysis |
| **Metagenomics** | 4 | Taxonomic classification, diversity analysis |
| **Structural Biology** | 5 | MD simulation, protein structure, molecular docking |
| **Molecular Cloning** | 8 | Primer design, restriction enzymes, plasmid maps |
| **CRISPR** | 3 | Guide RNA design, off-target analysis |
| **Statistics** | 10 | Survival analysis, GWAS, population genetics |
| **Machine Learning** | 5 | Classification, clustering, feature importance |
| **Visualization** | 6 | Volcano, PCA, heatmap, network, Circos plots |

### 26 Visualization Types

- **Statistical**: Volcano plot, MA plot, QQ plot, Manhattan plot
- **Dimensionality**: PCA, t-SNE, UMAP, MDS
- **Clustering**: Heatmap, dendrogram, silhouette plot
- **Network**: Network graph, Circos plot, pathway visualization
- **Genomic**: Gene browser, syntenic plot, plasmid map
- **Publication**: Boxplot, violin, raincloud, ridge, dot plot

### Cyberpunk GUI

```bash
python -m biosuite
```

Features:
- 🎨 Modern cyberpunk design with neon accents
- 📁 Drag-and-drop file support
- 📊 Interactive visualizations
- 💾 Auto-save sessions
- 🔍 Real-time search
- ⚙️ Customizable themes

### REST API

```python
import requests

# Analyze a sequence
response = requests.post(
    "http://localhost:8000/api/v1/sequence/analyze",
    headers={"X-API-Key": "your-api-key"},
    json={"sequence": "ATCGATCGATCG"}
)
print(response.json())
```

### Jupyter Integration

```python
%load_ext biosuite.notebook.magics

# Analyze FASTA file
%%biosuite_fasta input.fasta
# Automatically parses and displays sequences

# Create interactive plot
from biosuite.notebook.widgets import quick_gc
quick_gc("ATCGATCGATCG")
```

---

## 📦 Installation Options

### Core (required)

```bash
numpy>=1.24, pandas>=2.0, matplotlib>=3.7, seaborn>=0.12
scipy>=1.10, scikit-learn>=1.3, tqdm>=4.65
```

### Optional Dependencies

```bash
# For advanced analysis
pip install biosuite-ultra[genomics]    # NGS, variant calling
pip install biosuite-ultra[proteomics]  # Structure prediction
pip install biosuite-ultra[ml]          # Machine learning
pip install biosuite-ultra[api]         # REST API server
```

### External Tools (optional, for speed)

```bash
# Bioinformatics tools
BLAST+, Clustal Omega, MUSCLE, MAFFT
BWA, Bowtie2, FreeBayes, MACS2
RAxML, IQ-TREE, MrBayes
SPAdES, MEGAHIT, Kraken2
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=biosuite --cov-report=html

# Run specific module tests
pytest tests/test_sequence.py -v
```

**Test Results:**
- ✅ 150 tests passing
- ✅ 96 modules tested
- ✅ ~68 seconds runtime

---

## 🐳 Docker

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["python", "-m", "biosuite.api"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  biosuite:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - API_KEY=your-api-key
```

---

## 📚 Documentation

- [Quick Start Guide](docs/getting-started/quickstart.md)
- [API Reference](docs/api/)
- [Module Documentation](docs/modules/)
- [Tutorials](examples/)
- [Changelog](CHANGELOG.md)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork the repository
git clone https://github.com/your-username/BioSuite-Ultra.git

# Create a branch
git checkout -b feature/amazing-feature

# Make changes and commit
git commit -m "Add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use BioSuite Ultra in your research, please cite:

```bibtex
@software{biosuite_ultra,
  author = {Sahand Touri},
  title = {BioSuite Ultra: A Comprehensive Bioinformatics Platform},
  year = {2024},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.21256296},
  url = {https://github.com/sahandtkod-byte/BioSuite-Ultra}
}
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21256296.svg)](https://doi.org/10.5281/zenodo.21256296)

---

## 🔗 Links

- **PyPI**: [pypi.org/project/biosuite-ultra](https://pypi.org/project/biosuite-ultra/)
- **GitHub**: [github.com/sahandtkod-byte/BioSuite-Ultra](https://github.com/sahandtkod-byte/BioSuite-Ultra)
- **Documentation**: [biosuite.readthedocs.io](https://biosuite.readthedocs.io)
- **Docker Hub**: [hub.docker.com/r/sahandtkod/biosuite-ultra](https://hub.docker.com/r/sahandtkod/biosuite-ultra)

---

## 👥 Contributors

Thanks to all contributors who have helped make BioSuite Ultra better!

<a href="https://github.com/sahandtkod-byte/BioSuite-Ultra/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=sahandtkod-byte/BioSuite-Ultra" />
</a>

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=sahandtkod-byte/BioSuite-Ultra&type=Date)](https://star-history.com/#sahandtkod-byte/BioSuite-Ultra&Date)
