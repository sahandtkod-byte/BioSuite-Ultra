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

### Scoped API keys (opt-in)

By default the single `BIOSUITE_API_KEY` grants full access to every endpoint. Optionally, you can issue keys limited to a scope with `BIOSUITE_SCOPED_KEYS`:

```bash
export BIOSUITE_SCOPED_KEYS="readonly-key:read;runner-key:execute;owner-key:admin"
```

- `read` — call read-only endpoints.
- `execute` — everything `read` allows, plus analysis endpoints that run computations.
- `admin` — everything `execute` allows, plus classified admin operations; JWT admin routes still require the bearer token.

Authorization uses an explicit `(HTTP method, resolved route template)` map in `biosuite/api/auth.py`. Every unclassified operation denies scoped keys with 403. The legacy master key retains unrestricted API-key access, including unclassified routes, but never replaces endpoint-specific JWT authentication. On `/api/v1/admin/status`, no API key is needed with a valid JWT; if a key is supplied, it must be the master key or an `admin` scoped key. A `read` or `execute` key is denied even with a valid JWT.

Missing/invalid keys return 401; insufficient scopes return 403 with the unchanged legacy error shape. The opt-in envelope uses code `forbidden` and includes the request ID. Health/readiness and credential-based admin login remain public. Discovery accepts scoped read access, or anonymous access only when `BIOSUITE_PUBLIC_MANIFEST=1`. No API versions, persistence, transaction boundaries, retries, timeouts, or background-job behavior change.

Separate entries with `;` and the scope list with `,` (for example `key:read,execute`). When `BIOSUITE_SCOPED_KEYS` is unset or empty, the legacy single-key behaviour is unchanged. A malformed configuration stops the server at startup instead of serving with a partial key table, and key values never appear in error messages or logs.

Both the API key and JWT bearer schemes are registered in the OpenAPI schema, so you can authenticate directly from the Swagger UI ("Authorize" button at `/docs`).

---
## Deterministic capability manifest

The manifest retains only the fixed `schema_version` format identifier (`1.0.0`).
It excludes package/Python/platform/host versions, availability, timestamps and
host paths. JSON keys are sorted and NaN/Infinity are rejected. Runtime dependency
availability is reported separately by `discover_availability()`.

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

## Parallel Processing and Restriction Enzymes

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

### Restriction Enzymes
Full table of Type II restriction enzymes with recognition sites and cut
positions. The exact count is available at runtime rather than quoted here:

```python
from biosuite.core.utils import RESTRICTION_ENZYMES, RESTRICTION_ENZYMES_SITES

# List all available enzymes
print(f"Available enzymes: {len(RESTRICTION_ENZYMES)}")

# Get enzyme recognition site
site = RESTRICTION_ENZYMES_SITES['EcoRI']  # 'GAATTC'
```

---

## Asynchronous jobs (WS4)

Run exactly one owning API process (`uvicorn biosuite.api.server:app --workers 1`).
The app lifespan owns a lazily started thread and a SQLite-backed queue; a second
process using the same store is rejected by a separate SQLite exclusive-lock file.
No Redis, external workers, network requests, or new dependencies are involved.
Metadata and results persist under `BIOSUITE_DATA_DIR/.biosuite-jobs/jobs.sqlite3`
(default: the current working directory). The schema uses `PRAGMA user_version=1`.
Keep this directory private and on a local filesystem; do not share it over NFS.

| Method and path | Required scope | Success |
|---|---|---|
| `POST /api/v1/jobs` | execute | 202 with job metadata |
| `GET /api/v1/jobs/{id}` | read | 200 with job metadata |
| `GET /api/v1/jobs/{id}/result` | read | 200 with `id` and `result` |
| `POST /api/v1/jobs/{id}/cancel` | execute | 200 with current job metadata |

`execute` inherits `read`; `admin` and the legacy master key can access/cancel all
jobs. Other keys can access only jobs owned by that exact key, even if two keys
have the same scope. Owners are SHA-256 key identifiers, never raw keys. A key
whose scope is changed retains ownership; a rotated key does not. Missing/invalid
credentials produce 401, insufficient scope 403, and missing/other-owner jobs 404.
Existing JWT admin routes are unchanged and still require JWT authentication.

The initial explicit allowlist contains only `sequence.gc_content`; other catalog
capabilities are not enabled for jobs. Submission is strictly:

```json
{"capability_id":"sequence.gc_content","params":{"sequence":"ATGC"}}
```

No extra fields, file paths, module/function names, or URLs are accepted. Sequence
input is 1–100,000 nucleotide/IUPAC characters, without whitespace. The real
`biosuite.core.sequence.gc_content` returns, for this example:

```json
{"gc_percent":50.0,"sequence_length":4}
```

Use `curl -X POST http://127.0.0.1:8000/api/v1/jobs -H "X-API-Key: $BIOSUITE_API_KEY" -H 'Content-Type: application/json' -d '{"capability_id":"sequence.gc_content","params":{"sequence":"ATGC"}}'`.
Poll `GET /api/v1/jobs/{id}`, then retrieve `GET /api/v1/jobs/{id}/result`; use
`POST /api/v1/jobs/{id}/cancel` with the same key to request cancellation.

States are `queued`, `running`, `cancellation_requested`, `succeeded`, `failed`,
and `cancelled`. Cancelling queued work completes immediately. Running work is
not forcibly interrupted: it becomes `cancellation_requested`, finishes its bounded
calculation, discards the result, then becomes `cancelled`. Cancelling a terminal
job is idempotent and leaves its state unchanged. On restart, queued work resumes;
orphaned running work fails with `worker_restarted`, while pending cancellations
become `cancelled`. A result request before success returns 409.

Limits: 101,024 request bytes (413 above this), 32 active jobs including running
work, 1,024 retained jobs, 16,384 UTF-8 result bytes, and 24-hour terminal retention.
Queue/storage saturation returns 429 with `Retry-After: 1`; invalid submissions
return a sanitized 422. Cleanup runs during requests and worker polling (one
second). Results exceeding the bound fail with `result_too_large`; non-finite or
non-JSON results fail with `invalid_result`. Exceptions become `execution_failed`,
without exception text or submitted parameters in errors/logs. The five-second
execution budget is checked after calculation; it is not a preemptive thread kill.
Execution is bounded by the fixed linear-time GC operation and input length.
Over-budget completion fails with `execution_timeout`. Graceful shutdown waits
for that bounded operation; pending queued work stays persisted for restart.

POST submission is **not idempotent**: a retry creates a separate job. There are
no automatic execution retries. Admission and state transitions use SQLite
transactions; active admission is serialized with `BEGIN IMMEDIATE`. Indexes
cover state/creation, owner, and retention timestamps. Error envelopes remain
opt-in via `X-BioSuite-Error-Envelope: 1`; responses carry `X-Request-ID`.

## Support

If you have questions:
- Check the [README](README.md)
- Read the [CHANGELOG](CHANGELOG.md)
- Open a GitHub issue
