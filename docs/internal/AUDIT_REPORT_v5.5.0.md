# BioSuite-Ultra — Independent Adversarial Engineering Audit

**Target:** `sahandtkod-byte/BioSuite-Ultra`, branch `testing/v5.5.0`, commit `fcd72335bc551713e708d248afe6ceb87a50d5cc`
**Audit performed:** 2026-09-04
**Auditor posture:** adversarial. Every prior claim (README, `AGENTS.md`, `FIXES.md`, `docs/QUALITY_REPORT.md`, `docs/REVIEW_TRACKER.md`, "all tests pass", "lint clean", "line-by-line reviewed") was treated as an unverified hypothesis and independently re-tested.
**Production code modified:** none. One deliberate, reverted experiment on a config file is documented in §0.4; `git status` is clean apart from this report file.

---

## 0. Method, environment, and honesty statement

### 0.1 Environment actually used

| Item | Value |
|---|---|
| Python | 3.11.2 (venv at `/home/user/.venv`; system pip is PEP-668 managed) |
| Key deps | numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, matplotlib 3.11.1, scikit-learn 1.9.0, statsmodels 0.15.0, biopython 1.88, fastapi 0.141.1, pydantic 2.13.5, pytest 9.1.1, ruff, coverage |
| Deliberate deviations from CI | `pytest-asyncio 1.4.0` and `goatools 1.6.5` were installed **after** the baseline run, solely to prove that 9 baseline failures are dependency-declaration defects rather than product bugs. Both are called out where used. |
| Not installed (optional heavy deps) | gseapy, cutadapt, scanpy/anndata, scikit-bio, biotite, ete3, cobra, shap, umap-learn, pysam, torch/esm, openmm, pymzml. Every failure was checked against this list before being attributed to the product. |
| `tkinter` | **absent**; `sudo apt-get install -y python3-tk` failed in this sandbox |

### 0.2 What was verified dynamically vs. statically

* **Dynamically executed:** full pytest suite; 7 custom adversarial probe batteries (~190 probes) against `core/`, `api/` (live `TestClient`), `cli/` (17 real subprocess invocations incl. piped interactive sessions), `plotting/`, `workflow/`, `provenance`, `cloning`; a 400-trial differential test of the aligners against an independent reference DP implementation; `ruff`; `coverage`.
* **`biosuite/gui/` — NOT VERIFIED — ENVIRONMENT LIMITATION.** `tkinter` is not installable in this sandbox, so the GUI was **never launched and never interactively exercised**. All GUI statements below come from source reading and AST analysis only. Injecting fake `tkinter`/`customtkinter` stubs was considered and rejected: it would have produced a synthetic harness whose results could be mistaken for real GUI verification.
* No tool output in this report is invented. Every number is reproducible with the commands quoted.

### 0.3 Repository inventory (measured, not quoted from docs)

```
258 tracked files · 213 Python files · 52,339 LOC
biosuite/core/       45 analysis modules + core/workflow/ (batch, pipeline, report)
biosuite/api/        __init__.py (697 lines, all routes + app), auth.py, security.py, server.py
biosuite/cli/        menu.py (1,483 lines) — console scripts: biosuite, biosuite-gui
biosuite/gui/        17 files, 3,117 statements — main_window.py (973) + 11 tab mixins
biosuite/plotting/   12 modules (biological_plots.py 1,030 lines, plot_api.py, …)
biosuite/notebook/   single 498-line __init__.py
tests/               115 tracked files, 2,110 tests collected, 2,086 test functions
docs/ examples/ benchmarks/ + run.py, Dockerfile, docker-compose.yml, CI workflow
```

Import sweep of all 70 `biosuite.*` packages: **69 import cleanly; only `biosuite.gui` fails** (`No module named 'tkinter'`). Importing `biosuite.api` prints two default-credential warnings to stderr.

**Architecture map.** Layering is a clean star: `core/*` (leaf domain modules, mostly independent) ← `plotting/*`, `cli/menu.py`, `api/__init__.py`, `gui/tabs/*`, `notebook/`. Four independent front-ends re-implement dispatch over the same core functions, with **no shared service layer** — so every core defect is reproduced four times and every input-validation rule has to be written four times (it isn't; see BSU-005, BSU-014). `core/utils.py` is a god-module (config, theming, caching, tool detection, downsampling) imported by nearly everything, which is how a *test-only* config write ends up mutating a tracked repository file (BSU-004).

**Dead / orphan / suspicious artifacts:** `biosuite/api/server.py` is 32 lines of which only a `__main__` block is live and its documented ASGI target does not exist (BSU-006); `min_distance` in `peak_calling._find_peaks_from_coverage` is accepted and never used; `core/__init__.py` uses a star-import (`F403`) so its export surface is undefined; `docs/QUALITY_REPORT.md` and `docs/REVIEW_TRACKER.md` are status reports for a *different* branch (`arena/improvements`, "85 commits ahead of main") committed into this snapshot and describing code that is not here (BSU-026).

### 0.4 The one experiment that touched a tracked file

To test the hypothesis "the test suite writes into the repository", three keys were removed from `biosuite_config.json`, two test modules were run, and the file was inspected. The keys **reappeared**. The file was then restored with `git checkout -- biosuite_config.json`; the tree is clean. This is documented as evidence for BSU-004 and is clearly separated from the branch state.

---

## 1. Findings

Severity is assigned on impact × reachability, without inflation. Security findings are labelled **Confirmed** (attack path executed here), **Probable** (mechanism proven, exploitation context-dependent), or **Hardening**.

---

### BSU-001 — `needleman_wunsch` returns non-optimal scores and structurally invalid alignments

* **Severity:** CRITICAL
* **Category:** Scientific correctness / core algorithm
* **File:** `biosuite/core/alignment.py`
* **Location:** the vectorised DP fill in `needleman_wunsch` (horizontal predecessor read from the pre-update row); `smith_waterman` shares the pattern
* **Problem:** the row-vectorised recurrence computes the "gap in sequence 2" (left) predecessor from `dp[i, :m]` *before* that row is written. The left-gap dependency is intra-row and cannot be vectorised this way, so entire classes of optimal paths are never explored. Both the returned score and the returned alignment are wrong, and — worse — the reported score frequently does not even match the alignment that is returned.
* **Why it matters:** Needleman–Wunsch is the most-used primitive in the package. It is reached from `POST /api/v1/alignment/needleman-wunsch`, CLI options `nw`/`sw`, the GUI sequence-analysis tab, `biosuite.notebook.quick_align`, `core/parallel.py`, `core/msa.py` (progressive MSA and therefore every downstream phylogeny), and `run.py --benchmark`. Any published result derived from it is unsound.
* **Reproduction:**

```python
from biosuite.core.alignment import needleman_wunsch as nw

def score(a1, a2, match=1, mismatch=-1, gap=-2):
    return sum(gap if '-' in (x, y) else (match if x == y else mismatch)
               for x, y in zip(a1, a2))

for s1, s2 in [("ACGTACGTAA", "ACGT"), ("AAAAAAAA", "AA"),
               ("ACGTTTTTTTTACGT", "ACGTACGT")]:
    a1, a2, s = nw(s1, s2)
    print(s1, s2, "reported:", s, "actual:", score(a1, a2), a1, "/", a2)
```

* **Observed:**

```
ACGTACGTAA / ACGT        reported=-2   actual score of returned alignment=-23
   ACGTACGTAA---  /  ---------ACGT      (the two sequences are placed disjointly)
AAAAAAAA / AA            reported=-2   actual=-15
ACGTTTTTTTTACGT / ACGTACGT reported=3  actual=-16
nw("A","AAA") -> ('-A-','AAA',-2)   # that alignment actually scores -3
```

  A 400-trial differential test against an independent scalar reference DP (`/tmp/audit/t_align.py`, seeded, parameters m=1/mm=−1/gap=−2) found **355/400 = 88.75 % of random inputs produce a non-optimal score**; `smith_waterman` was wrong in 1/400.
* **Expected:** score equals the optimum of the scoring model, and equals the score of the alignment returned.
* **Root cause:** invalid vectorisation of a recurrence with an intra-row dependency; the traceback matrix is filled from the same stale values.
* **Recommended fix:** revert to a scalar (or Numba/Cython) DP with a correct three-way recurrence, or keep the vectorised diagonal-wavefront formulation which *is* valid. Then add (a) known-value tests, (b) a property test `score(returned_alignment) == returned_score`, and (c) a randomised differential test against Biopython's `PairwiseAligner`.
* **Confidence:** Certain (dynamically proven, two independent oracles).

---

### BSU-002 — Admin authentication bypass: forgeable JWTs and default admin credentials, shipped that way by `docker-compose`

* **Severity:** CRITICAL
* **Category:** Security — authentication bypass · **Confirmed**
* **File:** `biosuite/api/security.py` (lines 10, 15, 17-20, 32), `biosuite/api/server.py` (line 28), `docker-compose.yml` (`biosuite-api` service)
* **Problem:** `JWT_SECRET` falls back to the hard-coded literal `"changeme-dev-secret"` and `ADMIN_PASSWORD` to `"changeme-dev-password"`. Anyone with the source (it is a public repo) can mint a valid admin token offline. The only guard is a `logging.warning` on import — nothing refuses to start, and nothing marks the deployment as unsafe. `server.py` binds `0.0.0.0`, and `docker-compose.yml` publishes port 8000 to the host **without setting a single one of `BIOSUITE_JWT_SECRET` / `BIOSUITE_ADMIN_PASSWORD` / `BIOSUITE_API_KEY`**, so the documented container deployment is insecure by default rather than by misconfiguration.
* **Reproduction (executed against the live app via `fastapi.testclient`):**

```python
from jose import jwt
forged = jwt.encode({"sub": "attacker", "exp": time.time() + 3600},
                    "changeme-dev-secret", algorithm="HS256")
client.get("/api/v1/admin/status",
           headers={"Authorization": f"Bearer {forged}", "X-API-Key": "changeme-dev-key"})
```

* **Observed:** `200 {"admin": "attacker", "status": "ok"}`. Logging in with `admin` / `changeme-dev-password` also returns `200` plus a genuine token. (Expired tokens are correctly rejected with 401 — the JWT verification itself is fine; the *secret* is the flaw.)
* **Expected:** absent an explicitly configured secret, the application must refuse to start (or bind only to loopback and disable admin routes).
* **Root cause:** insecure-by-default constants with a non-blocking warning.
* **Recommended fix:** raise at import time when `BIOSUITE_JWT_SECRET`/`BIOSUITE_ADMIN_PASSWORD`/`BIOSUITE_API_KEY` are unset **unless** an explicit `BIOSUITE_DEV_MODE=1` is present; store the admin password as a hash (argon2/bcrypt), not a comparable literal; add the three variables to `docker-compose.yml` as required `${...}` references so the stack fails fast.
* **Confidence:** Certain.

---

### BSU-003 — Arbitrary code execution via `eval()` on CLI input

* **Severity:** HIGH
* **Category:** Security — code injection · **Confirmed**
* **File:** `biosuite/cli/menu.py` — lines ~1025 (unguarded path) and ~1036 (`eval()` on `input()`)
* **Problem:** two CLI menu paths pass user-entered text straight to `eval()` with no sandbox, no AST whitelist, and no confirmation.
* **Why it matters:** the CLI is scriptable (`biosuite < commands.txt`, CI jobs, GUI "run command" wrappers). Any pipeline, tutorial file, or copy-pasted snippet from an untrusted source becomes code execution with the user's privileges. It is rated HIGH rather than CRITICAL because a *human at an interactive prompt* could already run Python — the elevation only occurs when input is machine-fed or attacker-authored.
* **Reproduction:**

```bash
printf '93\n__import__("os").system("touch /tmp/PWNED.txt")\nq\n' | biosuite
ls -l /tmp/PWNED.txt
```

* **Observed:** `/tmp/PWNED.txt` created. **Expected:** the expression evaluator must not execute arbitrary Python.
* **Root cause:** `eval()` used as an expression calculator.
* **Recommended fix:** replace with `ast.literal_eval` plus a whitelisted math evaluator (`ast.parse` + node allow-list), or remove the feature.
* **Confidence:** Certain.

---

### BSU-004 — Runtime/test config writes land in a **git-tracked** file, and test residue is already committed

* **Severity:** HIGH
* **Category:** Security (secret handling) + data integrity
* **File:** `biosuite/core/utils.py` (~line 200, `CONFIG_FILE = APP_DIR / "biosuite_config.json"`, `save_config`), `biosuite_config.json` (tracked, not git-ignored)
* **Problem:** `CONFIG_FILE` resolves to the **package/repository directory**, not the user profile — while the surrounding docstring says `~/.biosuite/config.json`. `CONFIG_FILE` is bound at import time, so the tests' `monkeypatch`/`tmp_path` isolation never takes effect and real writes hit the repository. `save_config` swallows `OSError`, so failures are silent.
* **Why it matters:** the file holds `api_keys` (NCBI/UniProt/etc.). A user who enters a real key has it written into a tracked file that `git add -A` will commit and push. This has **already happened in this repository**: `git show HEAD:biosuite_config.json` contains unit-test residue (a fake service key, a `unittest_*` entry, a stray marker value) — i.e. the committed file is the output of somebody's test run. *(Per the audit rules the values are not reproduced here; the file and the key names are enough to locate them.)*
* **Reproduction:** remove the test-residue keys from `biosuite_config.json`, run `pytest tests/core/test_utils_more.py tests/core/test_utils_helpers.py` (24 passed), then `git diff` → the keys are back.
* **Expected:** config in `platformdirs.user_config_dir()`; nothing under the source tree; secrets never in VCS.
* **Root cause:** app-directory-relative config path + import-time binding + swallowed I/O errors.
* **Recommended fix:** move config to the user config dir with a `BIOSUITE_CONFIG` override; read `CONFIG_FILE` through a function so tests can patch it; add `biosuite_config.json` to `.gitignore`, `git rm --cached` it, ship `biosuite_config.example.json`; **rotate anything real that was ever stored there**; stop swallowing `OSError`.
* **Confidence:** Certain (mutation experiment executed and reverted).

---

### BSU-005 — Differential expression silently discards data (or raises `IndexError`) when `conditions` and columns disagree

* **Severity:** HIGH
* **Category:** Scientific correctness / input validation
* **File:** `biosuite/core/expression.py` (~lines 551-560, `differential_expression`, `calculate_fold_change`)
* **Problem:** the condition labels are zipped against the numeric columns with no length check.
* **Why it matters:** `conditions` is **client-supplied at the API layer**. Too few labels → surplus samples are silently dropped and the statistics are computed on a subset with no warning; too many → a raw `IndexError` reaches the caller. Silent sample loss in a DE analysis is the worst possible failure mode: the result looks plausible and is wrong.
* **Reproduction:** a 4-count-column matrix with `conditions=['A','B']` versus the same matrix with `conditions=['A','A','B','B']`.
* **Observed:** log2FC collapses from **5.916 → 0.585 with p = 1.0**, no warning. With 6 labels: `IndexError`.
* **Expected:** `ValueError("len(conditions) (2) must equal the number of numeric columns (4)")`.
* **Root cause:** missing precondition check on a public, network-reachable entry point.
* **Recommended fix:** validate lengths, require ≥2 replicates per group, and raise a typed error; add the same check at the API schema level (`pydantic` validator).
* **Confidence:** Certain.

---

### BSU-006 — The documented and containerised API start command does not work

* **Severity:** HIGH
* **Category:** Reliability / deployment
* **File:** `biosuite/api/server.py:7`, `biosuite/api/__init__.py:10`, `docker-compose.yml:27`
* **Problem:** all three tell the operator to run `uvicorn biosuite.api.server:app`, but `server.py` defines **no module-level `app`** (it only calls `uvicorn.run("biosuite.api:app", …)` inside `if __name__ == "__main__"`).
* **Reproduction:** `python -m uvicorn biosuite.api.server:app --host 0.0.0.0 --port 8123`
* **Observed:** `ERROR: Error loading ASGI app. Attribute "app" not found in module "biosuite.api.server".` `hasattr(biosuite.api.server, "app")` → `False`. The `biosuite-api` docker-compose service therefore cannot start at all.
* **Expected:** the documented command serves the API.
* **Root cause:** the ASGI object lives in `biosuite/api/__init__.py`; the entry-point module was never given a re-export, and no test or CI job ever starts the server.
* **Recommended fix:** add `from biosuite.api import app` at module level in `server.py`; add a smoke test that resolves the ASGI target string and a compose healthcheck.
* **Confidence:** Certain.

---

### BSU-007 — Provenance tracking is non-functional in every real execution context

* **Severity:** HIGH
* **Category:** Reliability / reproducibility
* **File:** `biosuite/core/provenance.py`, `biosuite/api/__init__.py` (`/api/v1/provenance/summary`)
* **Problem:** three independent defects:
  1. **Not thread-safe.** The SQLite connection is created on the constructing thread and reused; any record from a worker thread raises `sqlite3.ProgrammingError`. FastAPI runs sync endpoints in a threadpool, and `core/parallel.py` / `workflow/batch.py` use `ThreadPoolExecutor` — i.e. exactly the paths that generate provenance.
  2. **Per-request tracker.** The API constructs a fresh in-memory tracker per request, so `/provenance/summary` always answers "Total steps: 0".
  3. **`record()` crashes on numpy parameters** — `TypeError: Object of type ndarray is not JSON serializable`, unhandled, so recording an ordinary array-valued parameter kills the calling analysis.
* **Reproduction:** 8 threads each calling `ProvenanceTracker().record(...)`.
* **Observed:** `provenance concurrent record errors: 8` — *"SQLite objects created in a thread can only be used in that same thread"* — and `steps recorded: 0`. All eight records lost.
* **Expected:** thread-safe append-only recording; a process-wide (or persisted) tracker behind the API; serialisable coercion of numpy/pandas values.
* **Root cause:** connection created eagerly without `check_same_thread=False` + a lock; no shared instance; `json.dumps` without a `default=` encoder.
* **Recommended fix:** open the connection per-operation (or thread-local) with a `threading.Lock`, hold one tracker in app state, and add a numpy-aware JSON encoder. Provenance that silently records nothing is worse than no provenance, because reports claim it is on.
* **Confidence:** Certain.

---

### BSU-008 — `CachedResult` returns another call's result for distinct inputs

* **Severity:** HIGH
* **Category:** Correctness (caching)
* **File:** `biosuite/core/utils.py` (`CachedResult`)
* **Problem:** the cache key is `str(args) + str(kwargs)`. Numpy stringifies with ellipsis (`[0. 0. 0. ... 0. 0. 0.]`), so two completely different 10,000-element arrays produce the **same key**. Eviction is also FIFO despite LRU naming, and `access_order.pop(0)` can raise `IndexError` on a key that was already removed.
* **Reproduction:** call a cached function with `np.zeros(10_000)` then with an array that is zeros except a large final element.
* **Observed:** second call returns `0.0` (the first result) instead of `999999.0`; the wrapped function was invoked only once.
* **Expected:** distinct inputs → distinct keys, or refusal to cache unhashable inputs.
* **Root cause:** repr-based cache keys.
* **Recommended fix:** hash content (`hashlib.blake2b(np.ascontiguousarray(a).tobytes())` plus dtype/shape), fall back to *not caching* when a key cannot be derived, and use `collections.OrderedDict.move_to_end` for real LRU. *Mitigating context:* no shipped module currently calls `CachedResult`, so this is a latent public-API defect rather than an active miscalculation — which is also why the tests never caught it.
* **Confidence:** Certain.

---

### BSU-009 — Pipeline/batch state contamination: context overrides explicit arguments, and state leaks between runs

* **Severity:** HIGH
* **Category:** Correctness / workflow engine
* **File:** `biosuite/core/workflow/pipeline.py`, `biosuite/core/workflow/batch.py`
* **Problem:** four defects proven in one battery:
  1. Values from the shared context **override** kwargs explicitly declared on a step — the opposite of the expected precedence.
  2. `Pipeline.run()` never resets `self.context`, so a second run inherits the first run's values.
  3. After a failed run, `step.error` is retained; a later successful run reports `status="done"` **and** a stale `"boom"` error.
  4. `BatchProcessor` keys results by `sample_id`; duplicate IDs mean 3 jobs → 2 results, silently.
* **Reproduction:** step declared with `kwargs={'val': 'EXPLICIT'}` while the context holds `val='FROM_CONTEXT'`.
* **Observed:** the step receives `'FROM_CONTEXT'`.
* **Expected:** explicit kwargs win; context is per-run; error cleared on success; duplicate sample IDs rejected or suffixed.
* **Root cause:** `{**context, **kwargs}` written in the wrong order, plus mutable instance state reused across runs.
* **Recommended fix:** `{**context, **step.kwargs}` with step kwargs last; deep-copy the initial context at the start of `run()`; reset `error`/`status` when a step starts; validate sample-ID uniqueness.
* **Confidence:** Certain.

---

### BSU-010 — Fabricated statistics presented as analysis results (peak calling and docking)

* **Severity:** HIGH
* **Category:** Scientific validity
* **File:** `biosuite/core/peak_calling.py` (`_find_peaks_from_coverage`), `biosuite/core/docking.py` (`_builtin_dock`)
* **Problem:**
  * **Peak calling:** every emitted `Peak` gets `p_value=1e-5` — a hard-coded constant, not a computed statistic. The detection threshold is the *median of the 95th percentile of the sample's own positive coverage*, i.e. a self-referential quantile with no background/Poisson model, so uniform background noise still yields "peaks", and `fold_enrichment` is enrichment over the 95th percentile rather than over background.
  * **Docking:** `_builtin_dock` computes one crude distance-based energy and then manufactures N "poses" by adding `np.random.normal(0, 1.0, 3)` to the receptor centroid and `np.random.uniform(-0.5, 0.5)` to the energy. There is no conformational search, no scoring of the perturbed pose, and no seed — results are non-reproducible run to run and are reported in `kcal/mol` alongside a rank ordering.
* **Why it matters:** both produce output whose *form* (p-values, kcal/mol, ranked poses) asserts a statistical/physical meaning the computation does not have. Users cannot distinguish these from real results.
* **Reproduction:** read the two functions; `p_value=1e-5` is a literal, and the pose loop contains the two `np.random` calls.
* **Expected:** either a genuine model (Poisson/negative-binomial vs. local background for peaks; a real sampling+scoring loop for docking) or an explicit `p_value=None` / `engine="heuristic (not a statistical/physical estimate)"` in the report, with the docstring and CLI/GUI labels saying so.
* **Root cause:** placeholder values left in a shipped code path.
* **Recommended fix:** implement the real statistic, or downgrade the output types and label them unmistakably; at minimum accept and thread a `random_state` through docking.
* **Confidence:** Certain (static, unambiguous).

---

### BSU-011 — `msa.auto_align` reports empty conservation and can silently drop input

* **Severity:** HIGH
* **Category:** Scientific correctness
* **File:** `biosuite/core/msa.py` (~line 559)
* **Problem:** `auto_align` constructs its result with `conservation=[]` hard-coded and never calls `compute_conservation`, even though the function exists and is tested in isolation. A single-sequence input is silently dropped rather than rejected.
* **Why it matters:** every consumer (GUI MSA viewer, plots, reports) reads an always-empty conservation track and renders "no conservation" as if it were a computed result.
* **Reproduction:** `auto_align([...]).conservation` → `[]` for any input.
* **Expected:** populated conservation vector, or the field removed.
* **Root cause:** unfinished wiring; no test asserts the field is non-empty.
* **Recommended fix:** call `compute_conservation(aligned)`; raise on <2 sequences; add an assertion that `len(conservation) == alignment_length`.
* **Confidence:** Certain.

---

### BSU-012 — CORS reflects any origin with credentials enabled

* **Severity:** HIGH
* **Category:** Security — CORS misconfiguration · **Confirmed**
* **File:** `biosuite/api/__init__.py` (`CORSMiddleware` configuration)
* **Problem:** the middleware echoes the request `Origin` back in `access-control-allow-origin` **and** sets `access-control-allow-credentials: true`.
* **Reproduction:** `client.get("/health", headers={"Origin": "https://evil.example", "X-API-Key": ...})`
* **Observed:** `access-control-allow-origin: https://evil.example`, `access-control-allow-credentials: true`.
* **Why it matters:** combined with the publicly known default API key (BSU-002), a page the researcher merely visits can script their locally running BioSuite API — read files through `/file/read` (BSU-013), run analyses, and exfiltrate the responses.
* **Expected:** an explicit origin allow-list; never `*`/reflection together with credentials.
* **Recommended fix:** `allow_origins=os.environ["BIOSUITE_CORS_ORIGINS"].split(",")` with a loopback-only default, and `allow_credentials=False` unless an allow-list is configured.
* **Confidence:** Certain.

---

### BSU-013 — Arbitrary file read / path traversal and an existence oracle in the file endpoints

* **Severity:** HIGH
* **Category:** Security — path traversal · **Confirmed** (bounded by an extension filter)
* **File:** `biosuite/api/__init__.py` — `POST /api/v1/file/read`, `POST /api/v1/file/detect-format`
* **Problem:** `file_path` is taken from the caller and passed to the loader with no base-directory confinement, no `realpath` check and no symlink handling. Absolute paths and `../../../../` both work.
* **Reproduction:** `POST /api/v1/file/read?file_path=/tmp/secret_data.fasta` and the same with a traversal prefix.
* **Observed:** the file outside any workspace is read and its metadata/summary returned. `/etc/passwd` is refused, but only because of an extension check — any `.fasta/.fastq/.bed/.gff/.vcf/.csv` file anywhere on the host is readable, and `detect-format` answers differently for existing vs. non-existing paths, giving an unauthenticated-shape filesystem oracle.
* **Expected:** resolve under a configured data root and reject anything outside it.
* **Root cause:** no path confinement.
* **Recommended fix:** `root = Path(os.environ["BIOSUITE_DATA_DIR"]).resolve()`; `p = (root / user_path).resolve()`; `if not p.is_relative_to(root): 400`; never accept absolute paths; return a uniform error for "not found" and "not permitted".
* **Confidence:** Certain.

---

### BSU-014 — Unhandled exceptions terminate the interactive CLI, and non-interactive errors surface as raw tracebacks

* **Severity:** HIGH
* **Category:** Reliability / UX
* **File:** `biosuite/cli/menu.py` (`main_cli` loop ~line 432; option handlers 74, 75, 90; argparse dispatch)
* **Problem:** the menu loop has no per-iteration exception boundary and no `EOFError` handling, so a single bad input ends the session and discards state.
* **Reproduction / Observed (17 real subprocess invocations):**

| Input | Result |
|---|---|
| Ctrl-D / EOF at the menu | `EOFError` traceback, session dies (`menu.py:432`) |
| option 75 with a malformed Newick string | `IndexError: string index out of range`, session dies |
| option 90 with a non-numeric port | `ValueError` at `menu.py:1469`, session dies |
| option 74 with a missing GFF path | `FileNotFoundError`, session dies — while **option 28** with a missing FASTA is handled gracefully (inconsistent) |
| `biosuite gc` (missing arg) | raw `TypeError: gc_content() missing 1 required positional argument`, rc=1 |
| `biosuite nw ATCG` | raw traceback, rc=1 |
| `biosuite hwe 10 20 30` | raw `TypeError: … takes 1 positional argument but 3 were given`, rc=1 |
| `biosuite translate --frame 2` | argparse rc=2 — the documented `--frame` flag is not implemented |

  Working baseline: `biosuite gc ATCGATCG` → `50.0`; unknown command → friendly message, rc=0.
* **Expected:** `try/except` around each menu iteration printing a one-line error and returning to the menu; `EOFError`/`KeyboardInterrupt` → clean exit 0; argument errors → usage message and rc=2, never a traceback.
* **Root cause:** no error boundary; core functions called with unvalidated arity.
* **Recommended fix:** wrap the dispatch loop; centralise argument parsing per subcommand; add subprocess-level tests asserting exit codes and the absence of `Traceback` in stderr.
* **Confidence:** Certain.

---

### BSU-015 — Multiple API endpoints return 500 on ordinary malformed input

* **Severity:** MEDIUM
* **Category:** Reliability / error handling · security-relevant only as availability
* **File:** `biosuite/api/__init__.py` (no global exception handler; endpoints lack input validation)
* **Reproduction / Observed** (live `TestClient`, all authenticated): UPGMA with 1 sequence → 500; ragged distance matrix → 500; monomorphic GWAS SNP → 500 (`chi2_contingency` `ValueError`); volcano plot with mismatched vector lengths → 500; CRISPR with a negative guide length → 500; BLAST with an arbitrary `database` path → 500. Response bodies are generic (no traceback leak — good), but the wrong status code is returned for what are client errors.
* **Also observed (wrong-but-200):** `translate` with `frame=99` → 200 with an empty protein; `diversity` on `[-5, 3]` → shannon `−2.29` (a negative Shannon index is impossible).
* **Expected:** 422/400 with a field-level message; 500 reserved for genuine server faults.
* **Recommended fix:** pydantic validators for every numeric/length constraint, a `@app.exception_handler(ValueError)` mapping to 422, and a catch-all handler that logs with a correlation ID.
* **Confidence:** Certain.

---

### BSU-016 — Resource leaks: matplotlib figures are never closed and API plot endpoints leave temp files behind

* **Severity:** MEDIUM
* **Category:** Performance / resource management
* **File:** `biosuite/plotting/plot_api.py` (all plot functions), `biosuite/api/__init__.py` (plot endpoints)
* **Observed:** 5 successive `volcano()` calls leave 5 open figures (`len(plt.get_fignums()) == 5`) — in a long GUI or CLI session this grows without bound and eventually triggers matplotlib's `RuntimeWarning` and real memory growth. 3 plot API calls left 3 PNGs in `/tmp` with no cleanup: unbounded disk growth on a long-lived server, plus server-side path disclosure in the response.
* **Expected:** figures closed (or returned and closed by the caller under a context manager); temp files streamed and deleted, or served from a size-capped, garbage-collected cache.
* **Recommended fix:** `try/finally: plt.close(fig)`; use `tempfile.NamedTemporaryFile` + FastAPI `BackgroundTask` to unlink after the response; add a regression test asserting `plt.get_fignums() == []` after each plot helper.
* **Confidence:** Certain.

---

### BSU-017 — Generated HTML reports interpolate untrusted data without escaping

* **Severity:** MEDIUM
* **Category:** Security — stored XSS in generated artifacts · **Probable**
* **File:** `biosuite/core/workflow/report.py` — `HTMLReport.add_section`, `add_text`, `add_error`, `add_success`, `_build_stats_grid`, `_build_toc`, `to_html`
* **Problem:** `generate_pipeline_report`/`generate_batch_report` correctly call `html.escape`, but the general-purpose `HTMLReport` API does not: titles, text, stat keys/values and the report title are inserted raw.
* **Why it matters:** report content routinely comes from data files — FASTA headers, sample IDs, filenames, error strings echoing user input. A crafted header (`<img src=x onerror=…>`) becomes script in an HTML file the researcher opens locally (`file://`), able to read other local files the browser allows and exfiltrate them. Classified Probable because it requires the operator to build a report from attacker-influenced data.
* **Expected:** escape by default; require an explicit `raw_html=True` for the plot/table paths that legitimately need markup.
* **Recommended fix:** escape in `ReportSection.to_html` and at every string interpolation; keep an explicit `add_raw_html()` for internal use.
* **Confidence:** High.

---

### BSU-018 — No authentication throttling, and admin credentials are accepted as query parameters

* **Severity:** MEDIUM
* **Category:** Security — brute force / credential exposure · **Confirmed**
* **File:** `biosuite/api/__init__.py` (login route), rate-limiter configuration
* **Observed:** 50 consecutive failed logins all returned 401 with no lockout, delay or alert. The login route takes `username`/`password` as **query parameters**, so credentials land in access logs, proxy logs, browser history and `Referer` headers. Global rate limiting does work (100 requests → 200, the next 60 → 429), but it is keyed on `get_remote_address`, which behind any reverse proxy sees the proxy IP and therefore throttles all users as one.
* **Expected:** credentials in a POST body (`OAuth2PasswordRequestForm`), a dedicated stricter limit on the login route, exponential backoff/lockout, and a trusted-proxy-aware client-IP resolver.
* **Recommended fix:** as above; add `ProxyHeadersMiddleware` with an explicit trusted-hosts list.
* **Confidence:** Certain.

---

### BSU-019 — Interactive docs and the OpenAPI schema are served unauthenticated

* **Severity:** MEDIUM
* **Category:** Security — information disclosure · **Confirmed** (Hardening-class impact)
* **File:** `biosuite/api/__init__.py` (FastAPI app construction)
* **Observed:** every other route requires `X-API-Key` (even `/health` returns 401 without it — good), but `/docs`, `/redoc` and `/openapi.json` return 200 to anonymous callers, publishing all ~40 endpoints, their parameters and schemas.
* **Expected:** in production, disable them (`docs_url=None`) or place them behind the same dependency.
* **Recommended fix:** `FastAPI(docs_url=None, redoc_url=None, openapi_url=None)` unless `BIOSUITE_DEV_MODE`, or mount them behind `Depends(verify_api_key)`.
* **Confidence:** Certain.

---

### BSU-020 — Systemic input-validation and silent-wrong-answer defects across `core/`

* **Severity:** MEDIUM (individually LOW–MEDIUM; systemic as a group)
* **Category:** Scientific correctness / robustness
* **Findings (each observed directly):**

| # | Module / function | Observed | Expected |
|---|---|---|---|
| a | `sequence.reverse_complement('RYSWKM')` | `'MKWSYR'` — merely reversed; IUPAC ambiguity codes are not complemented; RNA `U` is left uncomplemented | complement ambiguity codes, or reject non-ACGTN |
| b | `sequence.translate(table=2)` | the `table` argument is ignored; the standard code is always used | honour NCBI table IDs or reject unsupported ones |
| c | `sequence.translate(frame=0)` | `'XEI'` — frame indexing is inconsistent with the 1-based convention used elsewhere/API `frame=99` accepted | one documented convention, validated |
| d | `orf_finder` | scans 3 frames while docs/menu advertise "6-frame" | scan both strands or fix the docs |
| e | `popgen.tajimas_d(np.ones((10,5)))` | `0.0` for monomorphic data | `nan` with a warning (D is undefined) |
| f | `popgen.hardy_weinberg_test` with negative counts | `p = 0` | `ValueError` |
| g | `gwas.run_gwas` monomorphic SNP | raw `ValueError` from `chi2_contingency` | skip the SNP with `nan`, or a typed error |
| h | `survival.cox_ph_summary` with plain lists | `TypeError` | accept sequences or raise a clear message |
| i | `survival.kaplan_meier([], [])` | median `inf` | `nan`/typed error |
| j | `file_formats.read_file` on a missing path | `{'format': 'fasta', 'records': None}` | `FileNotFoundError` |
| k | `file_formats.parse_gff` vs `parse_bed` | GFF raises on one malformed line; BED tolerates | one documented policy (`strict=` flag) |
| l | `phylogeny.parse_newick("")` / unbalanced | `IndexError: string index out of range` | `ValueError("malformed Newick")` |
| m | `blast.run_blast` with a missing query file | 0 hits, no error | `FileNotFoundError` |
| n | `parallel.parallel_map` | per-item exceptions swallowed; `None` inserted into results | propagate or return a typed `Result` |
| o | `trimming._pure_python_trim` | truncates the remainder of a FASTQ after an empty read | keep processing; report the count |
| p | `trimming._cutadapt_trim` | annotated `-> TrimReport`, returns `None` on the tool-missing path | return a report with `engine='none'` |
| q | `metabolism._builtin_fba` | objective and bounds hard-coded; `except Exception: pass` hides solver failures | parameterise; surface failures |
| r | `enrichment.run_ora` when the GO download fails | message `"ORA analysis error: stat: path should be string, bytes, os.P…"` (a leaked `TypeError`) | `"GO ontology unavailable (offline); install goatools data or pass obo_path"` |

* **Root cause (common):** validation is done ad hoc per function, and error paths are untested — the tests supply only well-formed inputs.
* **Recommended fix:** a shared `core/validators.py` contract applied at every public entry point (the module exists but is not used consistently), plus a table-driven "garbage input" test matrix run against every public function.
* **Confidence:** Certain (each row reproduced).

---

### BSU-021 — Built-in peak caller is O(genome length) in pure Python

* **Severity:** MEDIUM
* **Category:** Performance
* **File:** `biosuite/core/peak_calling.py` (`_find_peaks_from_coverage`)
* **Problem:** after the (good) chunked-coverage fix for memory, the state machine still iterates **every base pair in a Python `for` loop**, twice over the data (threshold pass + peak pass). For a human chromosome that is ~2.5 × 10⁸ Python iterations per pass.
* **Expected:** vectorised threshold crossing (`np.flatnonzero(np.diff((smoothed > t).astype(np.int8)))`) reduces this to array operations.
* **Recommended fix:** as above; add a benchmark guard on a synthetic 10 Mbp chromosome. Also remove or implement the accepted-but-unused `min_distance` parameter.
* **Confidence:** High (static; not benchmarked on a real chromosome here — that would take hours, which is the point).

---

### BSU-022 — Test-suite quality: broad but shallow; the critical bug lives inside a 94 %-covered file

* **Severity:** MEDIUM
* **Category:** Test quality
* **Measured:** 2,086 test functions / 3,421 assertions. **102 test functions contain no assertion at all**; 171 assertions are of the trivial `assert True` / `is not None` / `isinstance(...)` form. 24 files use mocking; 6 test files touch real network endpoints.
* **The decisive example:** `biosuite/core/alignment.py` has **93.6 % line coverage** and its own "deep" test file — yet BSU-001 (an 88 % wrong-answer rate) is invisible to it, because every alignment test is tautological:

```python
def test_needleman_wunsch_known_alignments():        # tests/core/test_alignment_deep.py
    a1, a2, score = al.needleman_wunsch("GATTACA", "GCATGCU")
    assert len(a1) == len(a2)                        # true for any output
    _, _, score2 = al.needleman_wunsch("ACGT", "ACGT")
    assert score2 > score                            # true for any output
```

  The name promises known-value verification; the body asserts only that the function returned two equal-length strings. Similar patterns: `assert score <= 0`, `assert "A" in a1`, `assert score >= 3`.
* **Effectively untested:** optimality of any DP algorithm; the default-secret authentication weakness (the *only* tests of `verify_api_key`/`verify_admin_token` are the 6 async tests that never run — see BSU-023); the whole interactive CLI loop (35 % coverage of `cli/menu.py`); the entire GUI (0 %); malformed/adversarial inputs; concurrency; resource cleanup.
* **Recommended fix:** add differential/property tests for every numeric kernel (compare to Biopython/scipy on random inputs, assert internal consistency such as `score(alignment) == reported_score`); add a "garbage input" matrix; assert on exit codes and stderr in CLI tests; fail CI on assert-less tests via a lint rule.
* **Confidence:** Certain.

---

### BSU-023 — CI does not test this branch, and silently skips the security tests

* **Severity:** MEDIUM
* **Category:** Process / CI
* **File:** `.github/workflows/ci.yml`
* **Problems:**
  1. `on: push/pull_request: branches: [main]` — **`testing/v5.5.0` is never built or tested by CI at all.**
  2. CI installs `.[api,notebook,dev]`, which declares **neither `pytest-asyncio` nor `goatools`**. Consequently the 6 async API auth/JWT tests never execute (pytest reports them as failures/errors rather than skips, and nobody notices because CI does not run on this branch) and 3 ORA tests fail with no `skipif` guard. Proof: re-running `tests/api/ + tests/core/test_enrichment_fixtures.py` with those two packages installed and `--asyncio-mode=auto` gives **49 passed** — the failures are dependency-declaration defects, not product bugs.
  3. The hard lint gate is only `ruff --select E9,F821`; the full ruff run is `continue-on-error: true`, so 236 issues never block.
  4. `--cov=biosuite` is passed with **no `--cov-fail-under`**, so coverage can regress freely.
* **Recommended fix:** add the working branches (or `branches-ignore: []`) to the triggers; declare `pytest-asyncio` and `goatools` in the `dev`/`api` extras and set `asyncio_mode = "auto"` in `pyproject.toml`; make optional-dependency tests `pytest.importorskip`; make ruff blocking on `F,B,E9`; add `--cov-fail-under=75`.
* **Confidence:** Certain.

---

### BSU-024 — Version, changelog and documented counts do not match the code

* **Severity:** LOW
* **Category:** Documentation / release hygiene
* **Observed:** the snapshot is labelled **v5.5.0**, but `biosuite/__init__.py` and `pyproject.toml` declare `__version__ = "5.0.0"`, `/health` reports `5.0.0`, and `biosuite --version` prints `BioSuite Ultra 5.0.0`. `CHANGELOG.md` has **no 5.5.0 entry** (newest: `[5.0.0] – 2026-08-28`), so there is no record of what changed in the release under review — which also makes an evidence-based regression analysis of "v5.5.0 changes" impossible: the branch is a single squashed snapshot commit with no in-branch history. Counts in `README.md` and `AGENTS.md` are stale or self-contradictory: "47 analysis modules" vs "45" vs "49 core modules"; "**1,444 tests in 30 test files**" vs the measured **2,110 tests in 115 files**; "100 % type-hint coverage" is asserted with no checker configured (no mypy config exists in the repo).
* **Recommended fix:** single-source the version, add the 5.5.0 changelog entry, generate the counts in CI rather than hand-writing them, and either add `mypy --strict` to CI or drop the typing claim.
* **Confidence:** Certain.

---

### BSU-025 — Prior quality reports in the repository make claims that are false for this tree

* **Severity:** LOW (INFO-level for the code, MEDIUM for process trust)
* **Category:** Process / documentation integrity
* **File:** `docs/QUALITY_REPORT.md`, `docs/REVIEW_TRACKER.md`, `FIXES.md`
* **Observed:** `docs/QUALITY_REPORT.md` claims **"2103 passed / 0 failed"**, **"Lint (ruff) clean (rules F, B, E9)"** and **"line-by-line review: 87/87 files ✅"**. Independently measured on this commit: `ruff check biosuite/ --select F,B,E9` → **25 errors** (F841 ×9, F541 ×7, B007 ×5, B027 ×2, F403, F401); the test suite cannot reach 0 failures in a correctly-provisioned CI environment because of BSU-023; and a line-by-line review that missed an 88 %-wrong Needleman–Wunsch cannot have been what the phrase implies. Both documents describe a *different branch* (`arena/improvements`, "85 commits ahead of main") and were committed into this snapshot. `FIXES.md` lists 10 CRITICAL + 13 HIGH + MEDIUM items, **all still marked "⬜ Open"** — it is an accurate to-do list, not a record of fixes, and several of its entries are re-confirmed here.
* **Recommended fix:** delete or clearly date/scope status documents that refer to other branches; never assert lint/test/coverage status in prose that CI does not enforce.
* **Confidence:** Certain.

---

### BSU-026 — Dependency constraints admit versions that break the suite

* **Severity:** LOW
* **Category:** Reproducibility / packaging
* **Observed:** `pandas>=2.0` admits pandas 3.x. Installing the latest pandas (3.0.5) first produced widespread incompatibilities; the environment was deliberately pinned back to 2.3.3 to obtain the baseline. This is reported as a *loose-constraint* finding, **not** as a baseline test failure.
* **Recommended fix:** upper-bound the majors that the suite is actually tested against (`pandas>=2.0,<3`, likewise numpy/scipy/matplotlib), and add a scheduled CI job on unpinned latest to detect breakage early.
* **Confidence:** Certain.

---

### BSU-027 — Deployment hardening gaps

* **Severity:** LOW · **Hardening**
* **Observed:** `server.py` runs uvicorn with `reload=True` (a development-only watcher) while binding `0.0.0.0`; the Dockerfile runs as **root** with no `USER` directive; there is no `.dockerignore`, so `biosuite_config.json` (BSU-004) and the `.git` directory are copied into the image; `docker-compose.yml` still carries the obsolete `version: '3.8'` key and defines no healthchecks or resource limits.
* **Recommended fix:** `reload=False` (or gate on `BIOSUITE_DEV_MODE`); add a non-root `USER`; add `.dockerignore`; add healthchecks and `mem_limit`.
* **Confidence:** Certain.

---

### BSU-028 — Non-reproducible randomness in library code

* **Severity:** LOW
* **Category:** Reproducibility
* **Observed:** `core/docking.py` uses `np.random.normal`/`uniform` with no seed parameter (see BSU-010); `core/utils.py:469` downsamples plot data with `np.random.choice` unseeded, so the *same* dataset produces different figures across runs; `cli/menu.py` and `gui/tabs/genomics.py` call the global `np.random.seed(42)`, mutating process-wide RNG state for any library the user is also running in that process. `core/bio_ml.py` correctly threads `random_state` — that pattern should be applied everywhere.
* **Recommended fix:** accept `random_state`/`rng: np.random.Generator` parameters; never touch the global seed inside library code.
* **Confidence:** Certain.

---

### BSU-029 — `biosuite.gui` cannot be imported without a Tk display stack, even for pure-data submodules

* **Severity:** LOW
* **Category:** Architecture / packaging
* **File:** `biosuite/gui/__init__.py`, `biosuite/gui/themes.py`
* **Observed:** `themes.py` contains only dictionaries and font constants and is imported by tests as a data source, but `import biosuite.gui.themes` fails with `No module named 'tkinter'` because the package `__init__` eagerly imports `main_window`. `tkinter` is also not declared as a dependency or extra, so `pip install biosuite-ultra[full]` on a slim Linux image yields a `biosuite-gui` console script that cannot start.
* **Recommended fix:** make `biosuite/gui/__init__.py` lazy (`__getattr__`), and have the `biosuite-gui` entry point print an actionable message when Tk is missing.
* **Confidence:** High (import failure observed; the GUI itself was not run — see §0.2).

---

### BSU-030 — GUI static review (NOT VERIFIED — ENVIRONMENT LIMITATION)

* **Severity:** INFO
* **Category:** Coverage gap, not a defect claim
* **Method:** AST analysis of all 17 GUI files: every `self.<attr>` read was cross-checked against all assignments and all 233 method definitions across `BioSuiteApp` and its 11 tab mixins.
* **Result:** the files parse; **no unresolved callback or attribute reference was found** — the 9 flagged names are all inherited Tk/CTk API (`grab_release`, `after_idle`, `after_cancel`, `winfo_rootx/rooty`) or class-level constants (`DELAY_MS`, `PAD_X`, `OFFSET_Y`, `_PRESERVE_ON_REBUILD`). `PLOT_FUNCS` is initialised empty and populated at runtime by `_build_plot_funcs`, so the GUI plot registry **cannot be validated statically** and the GUI tests that check plot categories only compare static strings.
* **Explicit limitation:** **3,117 statements (0 % coverage) of GUI code were never executed during this audit.** Widget layout, threading behaviour, the `_finish_startup` deferred-import path, dialog modality and every user interaction are **unverified**. No statement in this report should be read as "the GUI works".
* **Recommended fix:** run the GUI suite in CI under `xvfb-run` so this 3,117-statement blind spot is closed.
* **Confidence:** N/A — explicitly unverified.

---

### BSU-031 — Minor items (INFO)

* `DEPLOY.md` lines 7, 34, 47 disclose a developer's local filesystem layout and the location of a stored PyPI API token (`…\.pypirc`). No credential value is present in the repository; the path reference should still be removed.
* `ruff check biosuite/ --statistics`: **236 advisory issues** — W293 ×47, C408 ×43, E701 ×38, E402 ×26, E741 ×13 (ambiguous `l`), E702 ×12, UP015 ×12, F841 ×9, F541 ×7, C401 ×7, W292 ×5, B007 ×5, others; 54 auto-fixable. `core/__init__.py` uses a star-import (`F403`), leaving the package's public surface undefined.
* `benchmarks/test_core_benchmarks.py` compiles and imports cleanly (a prior report suggested it was corrupt — **not reproduced**).
* Positive findings worth preserving: `core/cloning.py` validates thoroughly (unknown enzyme → `ValueError` listing valid names; empty template → `ValueError`; short target → clear message) and its digestion fragment arithmetic is **correct** for both topologies (linear, 2 sites → 3 fragments summing to the template length; circular → 2). `core/databases.py` degrades gracefully offline (bounded retries, then empty results). JWT *expiry* is enforced correctly. Global rate limiting works. API error bodies do not leak tracebacks. `bio_ml` threads `random_state` properly. `report.py`'s pipeline/batch generators do escape their input.

---

## 2. Section 29 — Final verdict

### Overall Status

> **NOT PRODUCTION READY.**

### Executive Summary

BioSuite-Ultra v5.5.0 is a large, ambitious and *architecturally tidy* codebase — 52 k lines, 45 core modules, four front-ends, 2,110 tests that mostly pass — whose **verification does not reach the things that matter**. The audit found two CRITICAL defects that invalidate the project's two headline promises. Scientifically, the central pairwise aligner returns non-optimal scores for **88.75 % of random inputs** and, worse, returns alignments whose true score differs from the score it reports (−23 vs. a reported −2 in one measured case) — inside a file with 93.6 % line coverage, because every alignment test asserts only tautologies. Operationally, the REST API can be fully compromised by anyone who has read the public repository: the JWT signing secret and admin password fall back to hard-coded literals, and `docker-compose.yml` publishes the API on `0.0.0.0:8000` without setting a single credential variable — forging an admin token was demonstrated here in three lines. That same deployment path is broken in a second way: the documented uvicorn target `biosuite.api.server:app` does not exist, so the container cannot start at all — which is itself proof that no test or CI job has ever launched the server.

Beyond those, the audit confirmed: silent sample loss in differential expression driven by client-supplied input; a workflow engine whose shared context overrides explicit arguments and leaks state between runs; provenance tracking that records nothing at all under FastAPI's threadpool and always reports "0 steps"; a cache that returns another call's result for different numpy arrays; peak p-values and docking poses that are hard-coded/randomly generated rather than computed; CORS reflecting any origin with credentials enabled; and arbitrary file read via path traversal. The interactive CLI dies on Ctrl-D. Existing quality documents in the repository assert "0 failures", "ruff clean" and "87/87 files reviewed line-by-line"; measured on this commit, ruff reports 25 F/B/E9 errors and CI does not even run on this branch. `FIXES.md`'s 23 CRITICAL/HIGH items are all still marked open, and several were independently re-confirmed here.

The good news is that the failure mode is consistent and therefore fixable: the *structure* is sound, the *domain coverage* is genuinely broad, several modules (cloning, databases, the JWT verification logic itself, rate limiting) are well built, and almost every defect above is a bounded, well-localised change. What the project lacks is not engineering effort but **adversarial verification**: differential/property tests for numeric kernels, negative tests for error paths, and CI that actually runs on the branch being shipped.

### Test Results (exact)

```
Command (baseline, deps as declared by CI):
  MPLBACKEND=Agg BIOSUITE_API_KEY=ci-test-key \
  python -m pytest tests/ -q -rf --ignore=tests/gui/test_gui_sequence_parse.py

  2,110 collected → 150 failed · 1,939 passed · 22 skipped · 80.38 s
```

*A plain `pytest tests/` aborts during collection* — `tests/gui/test_gui_sequence_parse.py` raises at import (`tkinter` missing) rather than skipping, so the suite cannot be run at all on a machine without Tk.

Failure attribution (all 150 inspected individually):

| Count | Cause | Product bug? |
|---:|---|---|
| 138 | `tkinter` unavailable in this sandbox | No — ENVIRONMENT LIMITATION (but see BSU-029: the tests hard-fail instead of skipping) |
| 6 | async API auth/JWT tests: `pytest-asyncio` not declared | No — CI dependency defect (BSU-023) |
| 3 | ORA/enrichment: `goatools` not declared, no `skipif` | No — CI dependency defect (BSU-023) |
| 3 | `tests/integration/test_phase{3,4,5}.py`: optional heavy deps | No — missing skip guards |

```
Verification run (pytest-asyncio + goatools installed, --asyncio-mode=auto):
  tests/api/ + tests/core/test_enrichment_fixtures.py  →  49 passed
```

**Interpretation: the 150 failures are environment/CI-configuration defects, not product bugs — and that is precisely the problem.** The suite is green on the maintainer's machine and therefore provides no signal about any of the 2 CRITICAL / 12 HIGH defects above, all of which were found by probes written specifically to break the code.

### Coverage

```
Command: pytest tests/ --cov=biosuite --cov-branch (GUI + 3 optional-dep integration files excluded)
Result:  1,873 passed · 22 skipped · 113.34 s
         16,075 statements · 6,320 missed · 4,816 branches · 755 partial  →  60 %
```

| Package | Line coverage |
|---|---|
| `biosuite/api` (auth/security only; routes are exercised via TestClient) | 95.1 % |
| `biosuite/core` | 79.1 % |
| `biosuite/plotting` | 77.9 % |
| `biosuite/cli` | **35.4 %** |
| `biosuite/gui` | **0 %** (3,117 statements — NOT VERIFIED — ENVIRONMENT LIMITATION) |
| **Total excluding GUI** | **75.3 %** |

Lowest-covered substantial modules: `cli/menu.py` 35.6 % (1,063 stmts), `core/single_cell.py` 38.0 %, `core/enrichment.py` 55.0 %, `plotting/plot_api.py` 64.0 %, `core/workflow/pipeline.py` 66.9 %, `api/server.py` 0 %. **Coverage percentage is actively misleading here** — BSU-001 sits in a 93.6 %-covered file.

### Static Analysis (tools actually executed)

| Tool / gate | Result |
|---|---|
| `ruff check biosuite/ --select E9,F821` (CI's blocking gate) | **All checks passed** |
| `ruff check biosuite/ --select F,B,E9` (claimed clean by `docs/QUALITY_REPORT.md`) | **25 errors** — F841 ×9, F541 ×7, B007 ×5, B027 ×2, F403 ×1, F401 ×1 |
| `ruff check biosuite/ --statistics` (full rule set) | **236 issues**, 54 auto-fixable |
| Import sweep, 70 `biosuite.*` modules | 69 clean; `biosuite.gui` fails (tkinter) |
| AST reference analysis of `biosuite/gui` (17 files) | no unresolved callbacks/attributes |
| `mypy` | not configured in the repository; the "100 % type hint coverage" claim is unenforced |

### Findings by severity

**CRITICAL (2)** — BSU-001 Needleman–Wunsch returns non-optimal scores and invalid alignments · BSU-002 admin auth bypass via default JWT secret/password, shipped by `docker-compose`

**HIGH (12)** — BSU-003 CLI `eval()` RCE · BSU-004 secrets written into a tracked config file (residue already committed) · BSU-005 silent sample loss in differential expression · BSU-006 documented/containerised API start command is broken · BSU-007 provenance non-functional (thread-unsafe, per-request, crashes on numpy) · BSU-008 cache-key collision returns wrong results · BSU-009 pipeline context overrides explicit kwargs + cross-run state leakage · BSU-010 fabricated peak p-values and docking poses · BSU-011 `auto_align` conservation hard-coded empty · BSU-012 credentialed wildcard CORS · BSU-013 path traversal / arbitrary file read · BSU-014 unhandled exceptions kill the interactive CLI

**MEDIUM (9)** — BSU-015 500s on malformed API input · BSU-016 figure and temp-file leaks · BSU-017 unescaped HTML report generation · BSU-018 no login throttling, credentials in query strings · BSU-019 unauthenticated `/docs` + `/openapi.json` · BSU-020 systemic core input-validation gaps (18 sub-items) · BSU-021 O(genome) Python loop in the peak caller · BSU-022 shallow/tautological tests · BSU-023 CI does not run on this branch and skips the security tests

**LOW (5)** — BSU-024 version/changelog/count inconsistencies · BSU-025 false claims in in-repo quality reports · BSU-026 loose dependency constraints · BSU-027 deployment hardening (root container, `reload=True`, no `.dockerignore`) · BSU-028 unseeded randomness · BSU-029 `biosuite.gui` import coupling

**INFO (2)** — BSU-030 GUI static-only review (explicitly unverified) · BSU-031 minor items, ruff advisories, and positive findings

### Security findings (consolidated)

| ID | Issue | Class |
|---|---|---|
| BSU-002 | Default JWT secret → admin token forgery; default admin password; `docker-compose` sets no credentials; binds `0.0.0.0` | **Confirmed — CRITICAL** |
| BSU-003 | `eval()` on CLI input → arbitrary code execution | **Confirmed — HIGH** |
| BSU-012 | CORS reflects any origin with `allow_credentials: true` | **Confirmed — HIGH** |
| BSU-013 | Path traversal / arbitrary file read; format-detection existence oracle | **Confirmed — HIGH** |
| BSU-004 | Secrets persisted into a git-tracked file; test residue already committed | **Confirmed — HIGH** |
| BSU-018 | No login throttling; credentials in query parameters; proxy-blind rate-limit key | **Confirmed — MEDIUM** |
| BSU-019 | Unauthenticated `/docs`, `/redoc`, `/openapi.json` | **Confirmed — MEDIUM** |
| BSU-017 | Unescaped user data in generated HTML reports (stored XSS) | **Probable — MEDIUM** |
| BSU-015 | Unhandled 500s (availability only; no traceback leakage) | **Confirmed — MEDIUM** |
| BSU-027 | Root container, `reload=True` in the production entry point, no `.dockerignore` | **Hardening — LOW** |
| BSU-031 | Local paths / token location disclosed in `DEPLOY.md` | **Hardening — INFO** |

No hard-coded live credential value was found in source; the exposure in BSU-004 is *user-entered* keys persisted into version control. Values are deliberately not reproduced in this report.

### Scientific-correctness findings (consolidated)

1. **BSU-001** — Needleman–Wunsch non-optimal in 88.75 % of random trials; reported score ≠ score of the returned alignment; Smith–Waterman wrong in 1/400. Contaminates MSA → phylogeny, the API, the CLI, the GUI and the notebook helpers.
2. **BSU-005** — differential expression silently drops samples on a conditions/columns mismatch (log2FC 5.916 → 0.585, p = 1.0).
3. **BSU-010** — peak p-values hard-coded to `1e-5` with a self-referential 95th-percentile threshold and no background model; docking "poses" are random jitter around the receptor centroid, unseeded, reported in kcal/mol.
4. **BSU-011** — `auto_align` always returns an empty conservation track.
5. **BSU-020** — Tajima's D returns 0.0 for monomorphic data (should be undefined); HWE accepts negative counts (p = 0); Shannon diversity of negative counts returns −2.29; `reverse_complement` does not complement IUPAC/RNA; `translate(table=…)` silently ignored; `orf_finder` scans 3 frames while documented as 6; `metabolism._builtin_fba` hard-codes objective and bounds and hides solver failures.
6. **BSU-028** — unseeded randomness makes docking and downsampled plots non-reproducible.

### Architecture findings (consolidated)

1. **No shared service layer.** Four front-ends (API, CLI, GUI, notebook) each re-implement dispatch and validation over `core/`. Validation is therefore inconsistent (option 28 handles a missing file; option 74 crashes) and every core defect is exposed four times.
2. **`core/utils.py` is a god-module** holding config, theming, caching, tool detection and downsampling. Its app-relative `CONFIG_FILE`, bound at import time, is the direct cause of BSU-004 and defeats test isolation.
3. **Entry-point drift.** `api/server.py` is effectively dead code whose documented ASGI target does not exist (BSU-006); `biosuite/gui/__init__.py` forces a Tk dependency on pure-data submodules (BSU-029); `core/__init__.py` star-imports, leaving the public surface undefined.
4. **Cross-cutting concerns are per-call rather than per-application:** a new provenance tracker per API request, matplotlib figures never closed, temp files never reaped.
5. **Optional dependencies are handled inconsistently** — some modules degrade gracefully (`databases`, `single_cell`, `assembly`), others raise or emit leaked exception text (`enrichment`), and the tests lack `importorskip` guards.
6. **Positive:** `core/` modules are genuinely decoupled from one another, the dual-mode (built-in ⟷ external tool) pattern is consistent and sensible, and the plotting layer is cleanly separated from analysis.

### Test-suite weaknesses (consolidated)

1. Tautological assertions in the highest-risk module: 93.6 % coverage of `alignment.py` with zero optimality checking (BSU-022).
2. 102 test functions with no assertion; 171 trivial assertions.
3. No differential testing against reference implementations, and no property-based tests, for any numeric kernel.
4. The only tests of `verify_api_key`/`verify_admin_token` are the 6 async tests that never execute; **none** tests the default-secret weakness.
5. `cli/menu.py` 35 % covered; the interactive loop, EOF handling and error recovery are untested.
6. GUI 0 % — and its tests *fail at import* instead of skipping, aborting collection of the entire suite.
7. Error paths, malformed inputs, concurrency and resource cleanup are essentially untested across the board.
8. Six test files perform real network I/O, making the suite slow and non-hermetic.
9. No coverage gate, no lint gate beyond `E9,F821`, and CI never runs on this branch.

### Ordered fix-priority plan

**P0 — before any further use of results or any deployment**
1. Rewrite `needleman_wunsch`/`smith_waterman` with a correct recurrence; add known-value + differential (Biopython) + self-consistency (`score(alignment) == reported_score`) tests. Re-validate everything downstream (MSA, phylogeny, benchmarks). *(BSU-001)*
2. Make missing `BIOSUITE_JWT_SECRET`/`BIOSUITE_ADMIN_PASSWORD`/`BIOSUITE_API_KEY` a startup failure; hash the admin password; require the variables in `docker-compose.yml`. *(BSU-002)*
3. Fix the ASGI entry point (`from biosuite.api import app` in `server.py`) and add a smoke test that starts the server. *(BSU-006)*

**P1 — this sprint**
4. Move config out of the source tree, `.gitignore` + `git rm --cached biosuite_config.json`, rotate any key ever stored in it, stop swallowing `OSError`. *(BSU-004)*
5. Validate `conditions` length in `differential_expression`/`calculate_fold_change` at both the core and API layers. *(BSU-005)*
6. Lock down CORS to an allow-list; confine `/file/read` to a data root; remove `eval()` from the CLI. *(BSU-012, BSU-013, BSU-003)*
7. Fix the pipeline kwargs precedence, reset per-run state, clear stale errors, reject duplicate batch sample IDs. *(BSU-009)*
8. Make provenance thread-safe, application-scoped and numpy-serialisable — or disable the feature and remove the endpoint. *(BSU-007)*
9. Fix or remove `CachedResult`. *(BSU-008)*

**P2 — next sprint**
10. Populate `auto_align` conservation; replace the fabricated peak p-values and docking poses with real computations or unambiguous "heuristic" labelling. *(BSU-011, BSU-010)*
11. Add an error boundary to the CLI loop and clean argument handling to every subcommand. *(BSU-014)*
12. Add pydantic validators + a 422 exception handler across the API; close figures; reap temp files; escape HTML reports; throttle login and move credentials into the POST body. *(BSU-015 – BSU-018)*
13. Fix CI: run on all branches, declare `pytest-asyncio`/`goatools`, `importorskip` for optional deps, blocking `ruff --select F,B,E9`, `--cov-fail-under=75`, and a GUI job under `xvfb-run`. *(BSU-023, BSU-030)*

**P3 — hardening and hygiene**
14. Apply `core/validators.py` uniformly and add a garbage-input test matrix. *(BSU-020)*
15. Vectorise the peak-caller state machine; remove the unused `min_distance`. *(BSU-021)*
16. Replace tautological tests with property/differential tests; ban assert-less tests in CI. *(BSU-022)*
17. Single-source the version, add the 5.5.0 changelog entry, generate documented counts in CI, remove or scope the stale in-repo quality reports, upper-bound dependencies, seed all RNG use, make `biosuite.gui` import lazily, and harden the container. *(BSU-024 – BSU-029, BSU-031)*

---

*Prepared from commit `fcd7233` on branch `testing/v5.5.0`. No production code was modified. Probe scripts used for the dynamic evidence live under `/tmp/audit/` (`t_align.py`, `nw_impact.py`, `adv1.py`, `adv2.py`, `adv3.py`, `adv_api.py`, `adv_api2.py`, `adv_cli.py`, `adv_plot.py`, `adv_wf.py`, `adv_clone.py`, `gui_static.py`); every command needed to reproduce a finding is quoted inline above.*
