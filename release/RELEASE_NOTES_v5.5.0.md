# BioSuite Ultra v5.5.0

An integrated, pure-Python bioinformatics platform: 47 analysis modules, 105 public plotting
functions, molecular cloning with a 169-enzyme restriction table, and reproducible workflows —
usable as a Python library, a command-line interface, a desktop GUI or a REST API.

This is a correctness, security and reproducibility release. Several analysis routines produced
results that disagreed with reference implementations; those are corrected and now verified
against independent oracles. The REST API moves to a secure-by-default posture.

---

## Highlights

### Scientific correctness

- **Pairwise alignment.** The vectorised Needleman–Wunsch / Smith–Waterman implementation read
  the horizontal predecessor from the wrong row, producing scores that disagreed with a textbook
  scalar dynamic-programming reference — sometimes exceeding the true optimum. Replaced with an
  exact max-plus prefix scan and verified against a brute-force reference.
- **ChIP-seq peak calling.** Peaks were reported with a fixed p-value and a threshold derived
  from the sample's own signal, so uniform background produced spurious peaks. Peaks are now
  called against a Poisson background model with computed p-values, Benjamini–Hochberg q-values
  and enrichment over background.
- **ChIP-seq coverage.** Reads spanning a chunk boundary were dropped, corrupting the pileup at
  every boundary. Chunked coverage is now exact against a dense reference at any chunk size.
- **Multiple sequence alignment.** Alignments are returned in input order, external aligner
  output (Clustal Omega, MUSCLE, MAFFT) is no longer silently discarded, and conservation is
  computed correctly.
- **Differential expression.** Mismatched condition lists no longer drop samples silently, zero
  within-group variance no longer yields a fabricated p-value, and gene identifiers held in the
  DataFrame index are preserved.
- **Population genetics.** Hardy–Weinberg p-values keep full floating-point precision instead of
  being rounded, and are cross-checked against an independent chi-square implementation.
- **Docking.** The built-in engine performs a seeded rigid-body search that scores every
  placement and penalises steric clashes. Its scores are reported in arbitrary units and are
  documented as a heuristic, not as binding free energies.

### Security

- **No default credentials.** The API server refuses to bind a socket unless `BIOSUITE_API_KEY`,
  `BIOSUITE_JWT_SECRET` and `BIOSUITE_ADMIN_PASSWORD` are set to sufficiently strong values.
  There is no built-in admin password and no fallback signing secret.
- **Confined file access.** File endpoints resolve paths only inside `BIOSUITE_DATA_DIR`.
  Absolute paths, `..` traversal, percent-encoded and double-encoded traversal, UNC and drive
  prefixes, NUL bytes, dotfiles and escaping symlinks are rejected before any filesystem call.
- **Restricted CORS.** Allowed origins come from `BIOSUITE_CORS_ORIGINS`; arbitrary origins are
  not reflected and credentials are disabled when a wildcard is configured.
- **Closed documentation endpoints.** `/docs`, `/redoc` and `/openapi.json` require
  authentication unless `BIOSUITE_DEV_MODE=1`.
- **Safer command-line handling.** Untrusted input is no longer passed to dynamic evaluation.
- **Safer temporary files.** Temporary paths are created atomically with restrictive
  permissions, removing a time-of-check/time-of-use window.

### Reproducibility

- Workflow state is isolated per run, explicit step arguments are no longer overridden by shared
  context, and duplicate sample identifiers no longer collapse results.
- The result cache keys entries on content rather than on a string rendering of the arguments,
  so distinct inputs can no longer collide.
- Provenance records capture the parameters and inputs of an analysis and are exposed through
  the `/api/v1/provenance/*` endpoints.

### Documentation and project layout

- Every published figure is measured from the source tree and checked automatically: 47 analysis
  modules, 105 public plotting functions of 123 total, 169 restriction enzymes, 40 plot types,
  11 GUI tabs, 99 CLI menu options and 38 REST endpoints.
- The README has been rewritten as a project landing page covering the four interfaces,
  installation via the published extras, verified examples, architecture, provenance, security
  and an explicit statement of scientific limitations.
- The security policy documents credential handling, the required environment variables and the
  credential-rotation requirement.
- The repository layout now separates user documentation from developer, maintainer and internal
  material: deployment notes live under `docs/deployment/`, manuscript sources under
  `docs/paper/`, engineering records under `docs/internal/`, and maintainer scripts under
  `scripts/`. This release archive ships the user-facing subset only.

---

## Compatibility

| | |
|---|---|
| Python | 3.10, 3.11, 3.12 (each exercised in continuous integration) |
| Operating systems | Linux, macOS, Windows |
| License | MIT |

`requires-python` is now `>=3.10`, matching the versions that are actually tested. Python 3.9 is
no longer declared as supported.

The public Python API is unchanged; code written against v5.0.x continues to work. Results from
the corrected routines listed above will differ from previous versions, because the previous
values were wrong.

---

## Installation

```bash
pip install biosuite-ultra
```

Optional extras:

```bash
pip install "biosuite-ultra[api]"        # REST API: FastAPI, Uvicorn, JWT auth, rate limiting
pip install "biosuite-ultra[gui]"        # desktop application (also needs system Tk)
pip install "biosuite-ultra[notebook]"   # IPython magics and widgets
pip install "biosuite-ultra[bio]"        # goatools, gseapy, scanpy, scikit-bio and others
pip install "biosuite-ultra[full]"       # everything above
```

Extras combine, for example `pip install "biosuite-ultra[api,gui,notebook]"`.

The desktop GUI additionally requires the system Tk bindings, which cannot be installed from
PyPI. On Debian or Ubuntu: `sudo apt-get install python3-tk`.

### From the release archive

Verify the download, then install:

```bash
sha256sum -c BioSuite-Ultra-v5.5.0.zip.sha256
unzip BioSuite-Ultra-v5.5.0.zip
pip install ./BioSuite-Ultra-v5.5.0
```

### Getting started

```bash
biosuite                       # interactive command-line menu
biosuite gc ATCGATCG           # single command
biosuite-gui                   # desktop application
python -m biosuite.api.server  # REST API (credentials required)
```

---

## Upgrade notes

Deployments that ran an earlier version with credentials taken from the repository must treat
the API key, the JWT signing secret and the admin password as compromised and rotate them.
Tokens signed with a previous secret remain valid until that secret is changed.

The API server now validates its configuration at startup and exits with a non-zero status
rather than serving traffic with weak or missing credentials. Set the three required environment
variables before upgrading a running deployment; `.env.example` lists them.

---

## Known limitations

- Built-in engines for BLAST-style search, read alignment and docking are pure-Python
  approximations that keep pipelines runnable without external binaries. Install the
  corresponding external tool for production-scale or publication-grade work.
- Enrichment analysis requires ontology data from the `bio` extra; without it the relevant
  functions report the missing dependency rather than returning a result.
- Structure prediction and molecular dynamics are simplified implementations intended for
  teaching and exploration.
- A number of plotting tests still assert only that a figure object was returned rather than
  checking its contents.

No part of this project is validated for clinical or diagnostic use.
