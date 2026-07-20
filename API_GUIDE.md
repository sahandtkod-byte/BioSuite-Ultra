# BioSuite Ultra — API Key Guide
## REST API Server Authentication

> This section covers securing the BioSuite **REST API server** itself (`biosuite.api`). It is separate from the external database keys (NCBI, UniProt, etc.) described below.

The REST API requires two things:

| Requirement | Header | Purpose |
|---|---|---|
| API key | `X-API-Key: <key>` | Required on every endpoint |
| JWT token | `Authorization: Bearer <token>` | Required only on `/api/v1/admin/*` routes |

Set these environment variables before running the server (defaults are for local dev only — **do not use them in production**):

```bash
export BIOSUITE_API_KEY="your-secret-key"
export BIOSUITE_JWT_SECRET="your-jwt-signing-secret"
export BIOSUITE_ADMIN_USER="admin"
export BIOSUITE_ADMIN_PASSWORD="your-admin-password"
```

Get an admin token:
```bash
curl -X POST "http://localhost:8000/api/v1/admin/login?username=admin&password=your-admin-password"
```

Call a protected endpoint:
```bash
curl "http://localhost:8000/api/v1/sequence/gc-content" \
     -H "X-API-Key: your-secret-key" \
     -H "Content-Type: application/json" \
     -d '{"sequence": "ATCGATCG"}'
```

The API is also rate-limited to **100 requests/minute per client**; exceeding it returns `429 Too Many Requests`.

Both the API key and JWT bearer schemes are registered in the OpenAPI schema, so you can authenticate directly from the Swagger UI ("Authorize" button at `/docs`).

---
## Which Sections Work Without Any Setup

Everything in BioSuite works out of the box with just `pip install`. The database search features (NCBI, UniProt, PDB, KEGG, Ensembl) function without keys but have slower rate limits. Adding keys makes them faster.

---

## How to Add Your Keys

Go to **GUI**: System tab → API Keys section
Or use **CLI**: Option 85 → API Keys submenu
Keys are saved in `biosuite_config.json` on your machine.

---

## Database-by-Database Reference

### NCBI Entrez (Nucleotide/Protein Search)
- Free to use immediately
- Without key: 3 requests/second
- With key: 10 requests/second
- Register: ncbi.nlm.nih.gov/account → Settings → API Key
- Enter key in: `config['api_keys']['ncbi_api_key']`
- Also set email: `config['api_keys']['ncbi_email']`

### UniProt (Protein Database)
- Works without any registration
- Rate limit: 3 requests/second
- No key needed, just search

### RCSB PDB (Protein Structures)
- Works without registration
- Generous rate limits
- No key required

### KEGG (Pathway Database)
- Free for academic users
- Register at genome.jp/kegg → get email
- Set email: `config['api_keys']['kegg_email']`
- Rate limit: 1 request/second

### Ensembl (Genome Annotations)
- No registration needed
- Rate limit: 15 requests per 30 seconds
- Just use it directly

### AlphaFold (Structure Prediction)
- Download predicted structures free
- Register at alphafold.ebi.ac.uk for API access
- Set email: `config['api_keys']['alphafold_email']`

---

## Which External Tools Are Optional Speed Boosts

These are NOT required. The pure Python versions work always.

| Tool | Module | What It Speeds Up | Install |
|------|--------|-------------------|---------|
| BLAST+ | blast.py | Sequence database search | ncbi.nlm.nih.gov/blast |
| Clustal Omega | msa.py | Multiple alignment | clustal.org/omega |
| MUSCLE | msa.py | Multiple alignment | github.com/rcedgar/muscle |
| MAFFT | msa.py | Large-scale alignment | mafft.cbrc.jp |
| Cutadapt | trimming.py | Read quality trimming | pip install cutadapt |
| Salmon | quantification.py | RNA-seq quantification | github.com/COMBINE-lab/salmon |
| Kallisto | quantification.py | RNA-seq quantification | github.com/pachterlab/kallisto |
| BWA | read_aligner.py | Short read alignment | github.com/lh3/bwa |
| Bowtie2 | read_aligner.py | Short read alignment | bowtie-bio.sourceforge.net |
| FreeBayes | variant_calling.py | Variant calling | github.com/freebayes/freebayes |
| MACS2 | peak_calling.py | ChIP-seq peak calling | github.com/macs3-project/MACS |
| RAxML | ml_phylogeny.py | ML phylogenetics | github.com/stamatak/standard-RAxML |
| IQ-TREE | ml_phylogeny.py | ML phylogenetics | iqtree.org |
| MrBayes | bayesian_phylogeny.py | Bayesian trees | mrbayes.sourceforge.net |
| SPAdes | assembly.py | Genome assembly | cab.spbu.ru/software/spades |
| MEGAHIT | assembly.py | Metagenome assembly | github.com/voutcn/megahit |
| Kraken2 | metagenomics.py | Taxonomic classification | github.com/DerrickWood/kraken2 |
| AutoDock Vina | docking.py | Molecular docking | vina.scripps.edu |
| OpenMM | md_simulation.py | Molecular dynamics | openmm.org |
| COBRApy | metabolism.py | Flux balance analysis | pip install cobra |

---

## Quick Start

1. Install: `pip install -r requirements.txt`
2. Run: `python run.py --gui`
3. Everything works immediately — no keys needed
4. Add NCBI key later if you search databases frequently
5. Install external tools only if you need faster processing on large datasets

---

## New in v4.1.0

### Parallel Processing
Process large datasets faster with built-in parallel execution:

```python
from biosuite.core.parallel import parallel_map, parallel_gc_content

# Process 10,000 sequences in parallel
sequences = ["ATCG...", "GCTA...", ...]  # 10,000 sequences
gc_values = parallel_gc_content(sequences, workers=8)

# Or use the batch processor for large datasets
from biosuite.core.parallel import ParallelBatchProcessor
processor = ParallelBatchProcessor(workers=4)
results = processor.process(gc_content, sequences, batch_size=1000)
```

### 100+ Restriction Enzymes
Full database of Type II restriction enzymes:

```python
from biosuite.core.utils import RESTRICTION_ENZYMES, RESTRICTION_ENZYMES_SITES

# List all available enzymes
print(f"Available enzymes: {len(RESTRICTION_ENZYMES)}")

# Get enzyme recognition site
site = RESTRICTION_ENZYMES_SITES['EcoRI']  # 'GAATTC'
```

---

## Support

If you have questions:
- Check the [README](README.md)
- Read the [CHANGELOG](CHANGELOG.md)
- Open a GitHub issue
