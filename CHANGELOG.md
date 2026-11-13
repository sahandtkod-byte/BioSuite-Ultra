# Changelog

## [6.0.0] - 2026-09-30

Audit-remediation release. Every item below was either reproduced in code before being
changed and re-measured after, or found by comparing an implementation against an
independent reference (Biopython, statsmodels, REBASE, or a scanner written for the
purpose). `audit_output/PROGRESS.md` records the commands and raw output per step; the
21 scripts in `audit_output/evidence/` re-run the measurements and all exit 0.

### Fixed - scientific correctness

- **Restriction enzyme database**: 14 of the 164 recognition sequences that BioSuite
  shares with REBASE did not match it — BanII `GRAGCY`→`GRGCYC`, RsrII `CGGACCG`→`CGGWCCG`,
  MspA1I `CMGCAG`→`CMGCKG`, HpyCH4III `ACGT`→`ACNGT`, DraIII `CACGTG`→`CACNNNGTG`,
  BaeI, BtgI, BtgZI, CviJI, DrdI, EarI, EciI, HpyAV, XcmI. Those enzymes reported cuts at
  sites that do not exist and missed sites that do; the whole table now agrees with REBASE
  and a test fails if it drifts. Cut *offsets* were deliberately not rewritten from
  Biopython's printed `elucidate()` form, which mis-reads blunt ends: only the 10 enzymes
  verified against REBASE claim a geometry, the rest report `unverified`.
- **Cox PH summary** (`survival.cox_ph_summary`): a covariate whose median equals its mode
  (`[1,2,2,2]`) emptied the high group and produced `hazard_ratio = 0.0` with
  `median_high_group = inf`; with no observed events it produced `nan`; plain lists raised
  `TypeError`. The split falls back to ranks when the median split degenerates, an
  unobserved median yields `hazard_ratio = None` plus an explicit `message`, and lists are
  accepted like `kaplan_meier` always did.
- **Log-rank p-values**: `1 - chi2.cdf(...)` underflowed to exactly `0.0` from about
  chi2 = 100 upward, so a strong difference was reported as p = 0.0. Now `chi2.sf`, and the
  result keeps significant digits instead of rounding a genuine 1e-15 to 0.0. Measured on
  survival data: p 0.0 → 3.7494e-08 with the hazard ratio unchanged.
- **ML preprocessing leakage** (`bio_ml`): `StandardScaler.fit_transform` ran over the whole
  matrix *before* the split and cross-validation then scored that pre-scaled matrix, so
  held-out rows shaped the transform they were evaluated under. The scaler is now a step
  inside a `Pipeline` in all three trainers. Reported honestly: the metric effect is real
  for an SVM (ΔAUC ≈ 0.003 on data built to show it) and provably zero for a RandomForest,
  because per-feature scaling is monotone and axis-parallel splits are invariant to it — the
  audit's "0.55+ AUC on noise" claim did not reproduce, in either version of the code.
- **Differential expression**: Welch t-tests now run on `log1p` counts, which is the scale
  where the test's assumptions hold for count data, and the negative-binomial helper no
  longer describes itself as DESeq2-equivalent.
- **Variant calling**: unmapped/secondary/duplicate/QC-fail flags are excluded, base
  comparisons are strand-agnostic, REF comes from the reference file when given, and
  DEL/DUP windows no longer overrun the sequence end.
- **I/O**: gzip/bz2 inputs are actually decompressed (the sniffers detected them and every
  parser then read binary noise), BOMs are consumed, FASTQ records whose quality length
  disagrees with the sequence are rejected rather than silently truncating, and GFF
  attributes no longer collide with reserved column names.

### Fixed - resources, lifetime and thread-safety

- **matplotlib figure lifetime**: `utils.ask_save_plot` and the batch exporter's own saver
  now close the figure they saved, in every branch including a declined prompt or a failed
  write. Measured end to end: `export_all_to_folder` writes 29 PNGs and leaves **0** live
  figures (before: one per plot, tripping matplotlib's own `max_open_warning`); the PDF
  batch behaves the same and no page is lost.
- **Import side effects**: `biosuite.core.cloning` no longer calls `matplotlib.use("Agg")`.
  Importing an analysis module used to move the whole process to a non-interactive
  backend, silently killing Jupyter and GUI plots; the headless choice now lives in
  `biosuite.api`, where it is actually true, and is covered by a scan that rejects any
  future library module doing it.
- **External tool work directories**: `ml_phylogeny.build_tree` removed its temporary
  directory only when the engine failed; five successful runs left five directories behind.
  Every path is now guarded by `try/finally` — measured 5 → 0 for both success and failure.
- **`CachedResult`** is thread-safe: one `RLock` (reentrant, so a cached function may call
  itself) guards the cache and is held while the function runs, so eight concurrent callers
  of a key compute once instead of eight times.
- **Process pool selection**: `parallel_submit` had no picklability guard, unlike
  `parallel_map`. The rule is now one shared helper used by both.

### Fixed - security and the API

- **`/health`** is reachable without a credential and is exempt from the shared
  100/minute budget (FIX-23; previously an unauthenticated monitor could exhaust the budget
  and lock a legitimate client out — measured: 120 keyed pings all answered 429).
  Anonymous callers get `{"status", "timestamp"}` only; version and module inventory still
  require the key, so opening a probe did not open a fingerprint.
- **`HTMLReport`** escapes titles, subtitles, stats, free text, error and success messages,
  and generates url-safe anchor ids; `content` remains markup by documented contract, so
  plots and tables still render. Three injection markers survived into the generated HTML
  before this change.
- Startup still refuses to bind a socket without `BIOSUITE_API_KEY`,
  `BIOSUITE_JWT_SECRET` and `BIOSUITE_ADMIN_PASSWORD`; the README security section now
  describes the `/health` exception correctly instead of claiming every endpoint is keyed.

### Fixed - input and output encoding

67 text-mode I/O sites in 22 files depended on the machine's ANSI code page. Measured on
Windows (cp1252): writing a non-ASCII sequence name raised `UnicodeEncodeError` outright,
and reading a UTF-8 BED or FASTA silently renamed the user's data. Text files now state
`encoding="utf-8"`; the same class of bug in the test harness is fixed too.

### Fixed - dependencies and tooling

- `pydantic` is declared in the `api` extra: it is imported directly, not only through
  FastAPI. An AST check now asserts that every import-time dependency of `biosuite.api` and
  `biosuite.gui` is installed by the core requirements or the matching extra.
- The audit's suggestion to uncomment `customtkinter` in `requirements.txt` was rejected
  with evidence: it is already declared in the `gui` extra and `requirements.txt` documents
  its optional blocks as reference-only, so the change would have given a GUI dependency to
  every headless install.
- Optional-dependency tests skip instead of failing (`tests/core/test_enrichment_fixtures.py`),
  and `EnrichmentReport.engine` now records which backend actually ran, so "goatools is
  missing" can never again be read as "nothing was enriched".
- External engines: RAxML-NG gets `--prefix` (not the IQ-TREE `-o`, which means outgroup),
  its exit status is checked, and a requested engine that is absent or fails warns before
  falling back to neighbour-joining instead of returning a silently non-ML tree.

### Changed - breaking

- `cox_ph_summary` can return `hazard_ratio=None` and adds `split` / `message` keys.
- `EnrichmentReport` has a new `engine` field; `PhyloResult.tree_file` is no longer set to a
  path inside a temporary directory that the call deletes — the tree is in `newick_tree`.
- `utils.ask_save_plot` closes the figure it was given. Code that routed a figure through it
  and then displayed that figure must keep the `Figure` returned by the plot function.
- `/health` answers unauthenticated with a reduced payload.

### Added - tests and verification

425 regression tests in nine files, including an accuracy suite that compares reverse
complement over every IUPAC string up to length 3, translation over all 64 codons, GC
content, Benjamini-Hochberg against `statsmodels` to 1e-12, Needleman-Wunsch and
Smith-Waterman scores against `Bio.Align.PairwiseAligner`, and every enzyme's recognition
positions against an independent scanner. `tests/api` is now collectable and passing
locally (126 tests) once the API extras are installed; six `tests/cli` tests that had never
run to completion on Windows pass now. A line-by-line review of all 90 source modules
strengthened 107 weak tests, removed unused imports, and wired the sam/bcf/newick/bigwig
readers. Final measured gate: **9,068 passed / 17 skipped / 0 failed**, with **100% branch
coverage** of `biosuite/` (21,711 statements / 7,022 branches, 0 missing or partial),
doc-accuracy tests 35/35 and the security smoke suite 4/4. The 17 skips are optional
scientific extras and platform-specific tests, each reported by `pytest -rs`.

### Added - features (Tier 1 & Tier 2)

Deferred core capabilities are now exposed on the existing CLI and GUI surfaces (Tier 1
and the 10-item Tier 2), with no new top-level interfaces: the interactive menu keeps
117 options and gains a search affordance, the GUI gains a robust search, and sectioning
uses the real module names. The same core functions back the library, CLI, GUI and REST
API, so results are identical across interfaces.

### Known and still open

Branch coverage of `biosuite/` is measured on every CI run, reported to Codecov, and stands
at 100% in the verified local run; CI does not yet enforce a hard `--cov-fail-under` gate.
`ruff check .` could not be installed in
this environment (network), so only `pyflakes` ran — 153 findings, none an undefined name;
`cli/menu.py` and `api/__init__.py` remain large; `peak_calling.py` keeps a third copy of
Benjamini-Hochberg; `bio_ml.select_features` selects on the full matrix; multi-allelic sites
with duplicate POS and depth-independent QUAL in variant calling remain; the GUI freeze
reported in Pass 6 is unreproduced; third-party data licensing (REBASE, KEGG, SILVA) is
documented nowhere.

## [5.5.0] - 2026-09-04

Correctness, security and reproducibility release.  Every item below was found
by an independent audit of the 5.5.0 testing snapshot and is covered by a
regression test that fails on the previous behaviour.

### Fixed - scientific correctness
- **Pairwise alignment**: the vectorised Needleman-Wunsch/Smith-Waterman DP
  read the horizontal predecessor from the previous row, producing scores that
  disagreed with a textbook scalar DP on 297 of 400 randomised inputs (and were
  sometimes *higher* than the true optimum, i.e. not achievable by any
  alignment).  Replaced with an exact max-plus prefix scan.
- **ChIP-seq peak calling**: peaks were emitted with a hard-coded
  `p_value=1e-5`, and the detection threshold was the 95th percentile of the
  sample's own signal, so uniform background noise yielded 115 "peaks".  Peaks
  are now called against a Poisson background model with computed p-values,
  Benjamini-Hochberg q-values and enrichment over background.
- **ChIP-seq coverage**: reads that started before a chunk boundary but
  extended past it were dropped, corrupting the pileup at every 5 Mbp boundary
  (up to 6.9x depth error).  Chunked coverage is now bit-exact against a dense
  reference at any chunk size.
- **Docking**: the built-in engine scored the input geometry once and then
  invented poses by adding Gaussian noise to the receptor centroid.  It now
  performs a seeded rigid-body search that scores every placement and
  penalises steric clashes; its scores are labelled as arbitrary units, not
  kcal/mol.
- **Multiple sequence alignment**: `auto_align` returned an empty conservation
  vector, and every external-tool alignment (Clustal Omega / MUSCLE / MAFFT)
  was silently discarded because `compute_conservation` was handed a Biopython
  object inside a blanket `except Exception`.
- **Differential expression**: mismatched `conditions` lists silently dropped
  samples; zero within-group variance produced a fabricated `p = 1.0`; gene
  names held in the DataFrame index were replaced by row numbers.
- **Workflows**: shared-context values overrode explicit step arguments,
  context and errors leaked between runs, and duplicate sample ids silently
  collapsed results.
- **Caching**: `CachedResult` keyed entries on `str(args)`, so two different
  large arrays shared a cache entry and one call returned the other's result.
  Keys are now content-addressed and eviction is a true LRU.
- `biosuite hwe` could never succeed: the registry passed three positional
  arguments to a function taking a single dict.

### Fixed - security
- **No working default credentials.**  `changeme-dev-secret` signed valid admin
  tokens and `changeme-dev-password` logged in.  Unconfigured secrets now fall
  back to random per-process values, admin login is disabled unless a password
  is configured, and the server refuses to start in production mode with
  missing or well-known credentials (`BIOSUITE_DEV_MODE=1` to override
  locally).
- **Remote code execution in the CLI** (menu options 92/93): user input was
  passed to `eval()`.  Function targets are now resolved from a whitelist or a
  `biosuite.module:function` path.
- **Arbitrary file read / path traversal** in the file endpoints: paths are
  confined to `BIOSUITE_DATA_DIR`, including through symlinks.
- **CORS**: any origin was reflected with `allow_credentials=true`.  The
  default is now a loopback allow-list; `BIOSUITE_CORS_ORIGINS` configures
  real deployments and a wildcard is never combined with credentials.
- Admin passwords are stored as salted PBKDF2-SHA256 hashes and compared in
  constant time; login is rate-limited and locked out after repeated failures;
  credentials in query strings are deprecated.
- `/docs`, `/redoc` and `/openapi.json` require the API key unless
  `BIOSUITE_DEV_MODE` or `BIOSUITE_PUBLIC_DOCS` is set.
- The bundled server binds loopback by default (`BIOSUITE_API_HOST` to change).
- User configuration - which holds third-party API keys - moved out of the
  source tree into the user config directory, written atomically with mode
  0600.  `biosuite_config.json` is no longer tracked in git.

### Fixed - robustness
- An exception in any interactive CLI action no longer terminates the session;
  Ctrl-D/Ctrl-C exit cleanly; usage errors report exit code 2 with a usage
  line instead of a traceback.
- `biosuite.api.server:app` - the ASGI target documented in the README and used
  by the container - now exists.
- Provenance recording is thread-safe, serialises numpy/pandas parameters and
  escapes HTML in exported reports; the API uses one application-scoped
  tracker instead of a per-request one.
- Malformed API request bodies produce 4xx responses instead of 500s.
- Matplotlib figures and temporary plot files are managed and pruned.

### Changed
- `customtkinter` moved from a hard dependency to the `gui` extra; importing
  `biosuite.gui` no longer requires tkinter until a GUI object is used.
- Dependency constraints gained upper bounds; the unsatisfiable `cobra>=3.0`
  in the `bio` extra was corrected to `cobra>=0.26`.
- CI runs on `testing/**` and `release/**` branches, installs the Tk bindings
  so the GUI tests actually execute, gates on `ruff --select E9,F,B`, runs the
  security regression suite as a hard gate, and checks version consistency.
- Container image runs as an unprivileged user, ships no build toolchain and
  has a health check; `docker-compose.yml` requires explicit secrets and
  publishes to loopback.
- `requires-python` narrowed from `>=3.9` to `>=3.10`, and the 3.9 and 3.13
  trove classifiers were removed, so the declared support matches the versions
  CI actually verifies (3.10, 3.11, 3.12).  The published 5.0.3 wheel already
  declared `>=3.10`; this aligns the source tree with it.

### Fixed - CI and static analysis
- The test workflow installed `python3-tk` but not the `gui` extra, which is
  where `customtkinter` lives.  Every module under `biosuite/gui` imports it,
  so the 141 GUI tests had never actually executed in CI.  The extra is now
  installed and a dedicated step fails loudly if the GUI dependencies are not
  importable.
- `goatools` was not installed in CI, so three known-answer enrichment tests
  failed.  The dependency is installed rather than the tests being skipped.
- The API path resolver relied on `ntpath.isabs()`, whose result for a bare UNC
  root differs between CPython 3.10 and 3.11+, producing a different HTTP
  status for the same input depending on the interpreter.  Absolute and UNC
  prefixes are now matched explicitly.
- CodeQL reported four High "uncontrolled data used in a path expression"
  findings against the file endpoints.  The resolver now validates the raw
  string against an allowlist before any filesystem call and then matches each
  component against the real directory listing, so untrusted text is never
  concatenated into a path.  This also closed real gaps: dotfiles such as
  `.env` inside the data directory were previously reachable, NUL bytes were
  not rejected by the resolver, and double-encoded traversal survived the
  single decode performed by the web framework.
- CodeQL reported three Medium findings for workflow jobs without a
  `permissions:` block; least-privilege `contents: read` is now set at workflow
  scope and on every job.
- The test suite republishes its summary as workflow annotations, so failures
  are visible without downloading raw logs.

### Documentation
- README rewritten against the current tree; every count is now measured
  (47 analysis modules, 105 public plotting functions of 123 total, 169
  restriction enzymes, 11 GUI tabs, 99 CLI menu options, 38 REST endpoints)
  and asserted by `tests/test_documentation_accuracy.py`.
- Removed stale and unverifiable claims, including "1,444 tests", "117 CLI
  options", "40 endpoints", "26"/"36+" visualization types, "43K+ lines",
  "53 analysis modules", superlatives and marketing language.
- Hard-coded `v4.0` version strings in the CLI `--help` banner, the GUI About
  dialog and the GUI help tab now derive from `biosuite.__version__`; the help
  tab's "52+ analysis modules" claim was corrected to the measured 47.
- `SECURITY.md` updated: supported versions now reflect the 5.x series, and it
  documents credential handling, the required environment variables and the
  credential-rotation requirement.
- `CITATION.cff`, `AGENTS.md`, `docs/index.rst` and the package description
  synchronised; the Sphinx toctree no longer references five documents that do
  not exist.

### Known limitations
- **BSU-022 is only partially fixed.**  Always-true assertions,
  defect-documenting tests and `(200, 500)` status tolerances were removed, but
  roughly 297 shallow `assert x is not None` assertions remain, mostly under
  `tests/plotting/`.
- Coverage is measured in CI but no coverage threshold is enforced, and the
  project does not claim full coverage.

### Security notice
Any deployment that ran a previous version with the shipped defaults must
treat its API key, JWT secret and admin password as compromised and **rotate
them**.  Tokens signed with the old secret remain valid until it is changed.


## [5.0.0] - 2026-08-28

### ✨ Type Safety (BREAKING)
- **100% type hint coverage** across all 49 `biosuite/core/` modules
- Added `from __future__ import annotations` to all core modules
- Full return type annotations on all public and private functions
- Comprehensive parameter typing with `Optional`, `Dict`, `List`, `Tuple`, `Any`
- Zero runtime behavior changes — purely static analysis improvements

### 📊 Coverage Summary
| Metric | Before (v4.2.5) | After (v5.0.0) |
|--------|-----------------|----------------|
| Files ≥80% typed | 23/49 (47%) | **49/49 (100%)** |
| Tests passing | 1,421 | **1,421** ✅ |
| Untyped functions | ~200+ | **0** |

### Modules Updated
- `assembly.py`, `bayesian_phylogeny.py`, `blast.py`, `databases.py`
- `msa.py`, `read_aligner.py`, `trimming.py`, `utils.py`
- Plus 41 additional core modules from prior releases

### 🔒 Compatibility
- Python 3.9+ (unchanged)
- All existing APIs remain backward compatible
- No breaking changes to function signatures or behavior


All notable changes to BioSuite Ultra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.5] - 2026-08-09

### Fixed
- Reverted from the unstable v5.0.1 line back to the stable 4.2.x series
- All 41 registered plotting entries verified working end-to-end
- Version strings unified across `pyproject.toml`, `biosuite/__init__.py`, the REST API,
  the CLI banner, the GUI labels, `CITATION.cff`, `docs/conf.py` and the README
- `tests/test_api.py` no longer depends on a hard-coded absolute path, and reads the
  API key from the `BIOSUITE_API_KEY` environment variable
- Release archives (`*.zip`) are now distributed as GitHub Release assets instead of
  being committed into the repository

## [4.2.4] - 2026-07-22

### Fixed
- Missing `tkinter` import in `biosuite/gui/tabs/visualization.py`

## [4.2.3] - 2026-07-22

### Changed
- Plots now open in a dedicated window with a "Save As..." action instead of blocking
  the tkinter event loop via `plt.show()`

## [4.2.2] - 2026-07-22

### Fixed
- Deep bug-fix pass across the plotting and GUI layers

## [4.2.1] - 2026-07-21

### Fixed
- Heatmap and other plotting crashes

## [4.2.0] - 2026-07-21

### Changed
- General bug fixes and stability work

## [4.1.1] - 2026-07-09

### Added
- JOSS paper and bibliography for academic submission
- Zenodo DOI for citation
- Docker and Binder quick start instructions
- Comprehensive test suite
- CI/CD pipeline with GitHub Actions
- Extended validation against BioPython and R
- Supplementary File S1 for journal submission

### Changed
- Updated `CITATION.cff` with correct module counts
- Updated benchmark results on real hardware (i5-4590)
- Improved README with badges and quick start

### Fixed
- Corrected module counts in documentation
- Fixed dependency claims (external bioinformatics tools)
- Updated machine specifications for benchmarks

## [4.1.0] - 2026-07-07

### Added
- Dual-Mode architecture (external tools + pure Python fallback)
- 47 analysis modules
- 123 visualization functions
- GUI with 11 tabs
- CLI with 117 menu options
- REST API with 40 endpoints
- BioSuite Ultra publication paper

### Changed
- Renamed packages for PyPI compatibility
- Updated all documentation

## [4.0.0] - 2026-06-15

### Added
- Initial release of BioSuite Ultra
- Complete bioinformatics platform
- Multi-omics analysis support
