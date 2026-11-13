"""Audit-remediation regression tests — Phase 5 (resource lifetime and import
side effects).

Two shapes of defect are pinned here:

* a library module mutating process-global state at import time (F2-4), and
* resources that are created and never released (FIX-16), plus mutable shared
  state that is read and written from several threads (F2-5).

The backend tests spawn a real interpreter because a backend switch only shows
up in a process that has not imported the module yet; the rest run in-process.
"""
import ast
import os
import pathlib
import subprocess
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_CORE = _REPO / "biosuite" / "core"


def _module_level_calls(path, *, attr, owner):
    """Return the calls written ``owner.attr(...)`` that run at import time.

    Anything inside a ``def`` or ``class`` body is ignored: a function that
    picks a backend when the *caller* invokes it is not an import side effect.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents = {child: parent
               for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def runs_at_import_time(node):
        while node is not None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return False
            node = parents.get(node)
        return True

    found = []
    for node in ast.walk(tree):
        func = node.func if isinstance(node, ast.Call) else None
        if (func is not None
                and isinstance(func, ast.Attribute) and func.attr == attr
                and isinstance(func.value, ast.Name) and func.value.id == owner
                and runs_at_import_time(node)):
            found.append(node)
    return found


# ── F2-4 / FIX-16: importing an analysis module must not change the backend ──

def test_importing_cloning_leaves_the_process_backend_untouched():
    """Regression guard for F2-4: importing an analysis module used to move the
    whole process onto Agg, silently killing interactive plots.  MPLBACKEND is
    cleared in the child so the comparison cannot pass vacuously.
    """
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("MPLBACKEND", None)
    probe = (
        "import matplotlib;"
        "before = matplotlib.get_backend();"
        "import biosuite.core.cloning;"
        "import matplotlib.pyplot as plt;"
        "print(before, matplotlib.get_backend(), plt.get_backend())"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=str(_REPO), env=env,
                            capture_output=True, text=True, encoding="utf-8",
                            timeout=180)
    assert result.returncode == 0, result.stderr
    before, after_module, after_pylab = result.stdout.strip().split()[-3:]
    assert before == after_module == after_pylab, (
        f"import changed the backend: {before} -> {after_module}/{after_pylab}")


def test_no_core_module_selects_a_matplotlib_backend_at_import():
    """Regression guard for F2-4: no ``matplotlib.use()`` in any library module."""
    offenders = sorted(
        path.name
        for path in _CORE.rglob("*.py")
        if _module_level_calls(path, attr="use", owner="matplotlib")
    )
    assert offenders == [], f"modules that force a backend at import: {offenders}"


def test_api_entry_point_declares_the_headless_backend():
    """The headless promise moved to the place that is actually headless (F2-4).

    ``biosuite.api`` is the object every documented ASGI target builds, so it
    also covers ``uvicorn --reload`` workers, which never import server.py.
    """
    calls = _module_level_calls(_REPO / "biosuite" / "api" / "__init__.py",
                                attr="use", owner="matplotlib")
    assert len(calls) == 1, f"expected exactly one backend call, got {len(calls)}"
    args = [ast.literal_eval(arg) for arg in calls[0].args if isinstance(arg, ast.Constant)]
    assert args[:1] == ["Agg"], f"API pins backend {args!r}, expected 'Agg'"


def test_backend_scan_sees_an_import_time_call_and_ignores_a_deferred_one(tmp_path):
    """Positive control for the guard above: the scan has teeth.

    Without this, a scan that always returned an empty list would make
    test_no_core_module_selects_a_matplotlib_backend_at_import pass forever.
    """
    forcing = tmp_path / "forcing.py"
    forcing.write_text("import matplotlib\nmatplotlib.use('Agg')\n", encoding="utf-8")
    assert len(_module_level_calls(forcing, attr="use", owner="matplotlib")) == 1

    deferred = tmp_path / "deferred.py"
    deferred.write_text(
        "import matplotlib\n\n\ndef run_headless():\n"
        "    matplotlib.use('Agg')\n    return True\n", encoding="utf-8")
    assert _module_level_calls(deferred, attr="use", owner="matplotlib") == []


# ── F2-3 / FIX-16: a figure that exists only to be saved must not outlive it ─

def _open_figures():
    import matplotlib.pyplot as plt
    return len(plt.get_fignums())


def _feed(values):
    """A ``builtins.input`` stand-in that replays ``values``, then answers ''."""
    stream = iter(values)
    return lambda prompt="": next(stream, "")


def _declines_everything(prompt=""):
    p = prompt.lower()
    if "number of sets" in p:
        return "2"
    if "use default" in p:
        return "y"
    return "n"


@pytest.fixture
def _clean_pyplot_state():
    """Pyplot state is process-global: hand the rest of the session what we found."""
    import matplotlib.pyplot as plt
    plt.close("all")
    yield
    plt.close("all")


def test_ask_save_plot_closes_the_figure_it_saved(tmp_path, monkeypatch,
                                                  _clean_pyplot_state):
    """Regression guard for F2-3 on the save branch."""
    import builtins

    import matplotlib.pyplot as plt

    from biosuite.core.utils import ask_save_plot

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "input", _feed(["y", "volcano"]))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    ask_save_plot("unused_default", "png", 100)

    assert _open_figures() == 0
    assert (tmp_path / "volcano.png").exists()


def test_ask_save_plot_closes_the_figure_when_the_user_declines(tmp_path, monkeypatch,
                                                                _clean_pyplot_state):
    """The declined branch is the one a later refactor would "optimise" away."""
    import builtins

    import matplotlib.pyplot as plt

    from biosuite.core.utils import ask_save_plot

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "input", _feed(["n"]))
    plt.subplots()

    ask_save_plot("plot", "png", 100)

    assert _open_figures() == 0
    assert list(tmp_path.glob("*.png")) == []


def test_ask_save_plot_releases_the_figure_even_when_saving_fails(tmp_path, monkeypatch,
                                                                  _clean_pyplot_state):
    """The release sits in a finally: an OSError must not strand the figure."""
    import builtins

    import matplotlib.pyplot as plt

    from biosuite.core import utils

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "input", _feed(["y", "plot"]))

    def refusing_savefig(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(plt, "savefig", refusing_savefig)
    plt.subplots()

    utils.ask_save_plot("plot", "png", 100)

    assert _open_figures() == 0


def test_ask_save_plot_releases_every_figure_it_adds_to_a_pdf(tmp_path,
                                                              _clean_pyplot_state):
    """F2-3 on the PDF branch.  Closing after pdf.savefig() must not eat the page,
    so a one-page document written the same way is the size control.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from biosuite.core.utils import ask_save_plot

    def write_pages(path, count):
        with PdfPages(str(path)) as pdf:
            for index in range(count):
                fig, ax = plt.subplots()
                ax.plot([0, 1], [index, index + 1])
                ask_save_plot("p%d" % index, "png", 100, pdf=pdf)
        return path.stat().st_size

    one_page = tmp_path / "one.pdf"
    three_pages = tmp_path / "three.pdf"
    size_one = write_pages(one_page, 1)
    size_three = write_pages(three_pages, 3)

    assert _open_figures() == 0
    assert size_three > size_one, (
        "three pages produced no more PDF than one page — a page was dropped")


@pytest.mark.parametrize("plot_name", ["volcano_plot", "boxplot_custom", "dot_plot"])
def test_demo_plot_releases_the_figure_it_drew(plot_name, tmp_path, monkeypatch,
                                               _clean_pyplot_state):
    """End to end for the interactive CLI path: one demo, zero surviving figures.

    Before FIX-16 each of these left its figure in matplotlib's global registry
    for the life of the process.
    """
    import builtins

    from biosuite.core import utils
    from biosuite.plotting import biological_plots

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "input", _declines_everything)
    monkeypatch.setitem(utils.config, "quiet", True)

    getattr(biological_plots, plot_name)(pdf=None)

    assert _open_figures() == 0


def test_plot_functions_that_own_a_figure_return_it_or_release_it():
    """Structural guard against a new demo plot stranding a figure: a function that
    takes no ax owns the figure it creates.  This pins routing only; the behaviour
    tests above are what show ask_save_plot really releases.
    """
    plotting_dir = _REPO / "biosuite" / "plotting"
    offenders = []
    examined = 0
    for path in sorted(plotting_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            creates = [n for n in ast.walk(fn)
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr in ("subplots", "figure")]
            args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
            if not creates or "ax" in args:
                continue
            examined += 1
            returns_fig = any(isinstance(n, ast.Return)
                              and getattr(n.value, "id", None) == "fig"
                              for n in ast.walk(fn))
            releases = any(isinstance(n, ast.Call)
                           and ((isinstance(n.func, ast.Name) and n.func.id == "ask_save_plot")
                                or (isinstance(n.func, ast.Attribute) and n.func.attr == "close"))
                           for n in ast.walk(fn))
            if not (returns_fig or releases):
                offenders.append("%s::%s" % (path.name, fn.name))
    assert examined > 0, "the scan found nothing to examine; it cannot be vacuous"
    assert offenders == [], f"figures that outlive their plot function: {offenders}"


# ── F2-5 / FIX-17: the cache is shared mutable state ────────────────────────

def test_cached_result_computes_a_key_once_while_eight_threads_ask_for_it():
    """Regression guard for F2-5: the unguarded body let every thread miss the
    cache and recompute.  No CachedResult is instantiated inside biosuite/, so
    this is a public-API defect, not a live race (Pass 9, correction 2).
    """
    import threading
    import time

    from biosuite.core.utils import CachedResult

    calls = []

    def slow(value):
        calls.append(value)
        time.sleep(0.05)
        return value * 2

    cache = CachedResult(slow)
    threads = [threading.Thread(target=cache, args=(7,)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(calls) == 1, f"the same key was computed {len(calls)} times"
    assert cache(7) == 14
    assert list(cache.cache) == [cache.make_key((7,), {})]


def test_cached_result_keeps_cache_and_lru_order_the_same_size_under_threads():
    """The inconsistent-state half of F2-5: eviction touched two structures."""
    import threading

    from biosuite.core.utils import CachedResult

    cache = CachedResult(lambda value: value, maxsize=8)
    keys = list(range(40))

    def worker(chunk):
        for value in chunk:
            cache(value)

    threads = [threading.Thread(target=worker, args=(keys[i::8],)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(cache.cache) == len(cache.access_order)
    assert len(cache.cache) <= 8
    assert set(cache.access_order) == set(cache.cache)


def test_cached_result_lets_the_wrapped_function_recurse_through_the_same_cache():
    """The lock has to be reentrant, or a recursive cached function deadlocks."""
    import threading

    from biosuite.core.utils import CachedResult

    def fib(value):
        if value < 2:
            return value
        return cache(value - 1) + cache(value - 2)

    cache = CachedResult(fib)
    done = threading.Event()
    outcome = []

    def run():
        outcome.append(cache(12))
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert done.wait(15), "the cache deadlocked on a re-entrant call"
    assert outcome == [144]


# ── F2-5 / FIX-17: executor choice must respect what a process pool can carry ─

def test_threads_preferred_for_an_unpicklable_callable_even_on_posix(monkeypatch):
    from biosuite.core import parallel

    monkeypatch.setattr(parallel.sys, "platform", "linux")
    assert parallel._threads_preferred([lambda value: value], io_bound=False)
    assert not parallel._threads_preferred([parallel._apply_chunk], io_bound=False)


def test_threads_preferred_on_windows_for_a_perfectly_picklable_function(monkeypatch):
    from biosuite.core import parallel

    monkeypatch.setattr(parallel.sys, "platform", "win32")
    assert parallel._threads_preferred([parallel._apply_chunk], io_bound=False)
    monkeypatch.setattr(parallel.sys, "platform", "linux")
    assert parallel._threads_preferred([parallel._apply_chunk], io_bound=True)


def test_parallel_submit_never_sends_a_closure_to_a_process_pool(monkeypatch):
    """The other half of FIX-17: parallel_submit had no picklability guard.  Spying
    on the executor class is what gives this teeth, because the function falls back
    to sequential execution and correct results alone would hide that.
    """
    from biosuite.core import parallel

    def refused(*args, **kwargs):
        raise AssertionError("a process pool was chosen for an unpicklable task")

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", refused)
    double = lambda value: value * 2  # noqa: E731 - deliberately not picklable
    assert parallel.parallel_submit([(double, (3,), {}), (double, (4,), {})],
                                    workers=2) == [6, 8]


# ── F2-1 / FIX-14: no preprocessing may see the rows it is scored on ────────

def _recording_scaler():
    """A StandardScaler subclass that records the row count of every fit.  sklearn
    routes fit_transform through fit, so per-fold CV fits are caught too.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    seen = []

    class RecordingScaler(StandardScaler):
        def fit(self, X, y=None, **kwargs):
            seen.append(int(np.asarray(X).shape[0]))
            return super().fit(X, y, **kwargs)

    return seen, RecordingScaler


def _noise_matrix(rows=200, features=20, seed=0):
    """Labels that carry no information at all, so any score is pure noise."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, features)), rng.integers(0, 2, rows)


@pytest.mark.parametrize("trainer", ["train_random_forest", "train_svm"])
def test_the_scaler_is_never_fitted_on_the_full_matrix(trainer, monkeypatch):
    """Regression guard for F2-1: fitting over every row let held-out rows shape the
    transform they were scored under.  The invariant is structural — no fit may see
    n rows — so it does not depend on how large the effect happens to be.
    """
    from biosuite.core import bio_ml

    pytest.importorskip("sklearn")
    seen, RecordingScaler = _recording_scaler()
    monkeypatch.setattr(bio_ml, "StandardScaler", RecordingScaler)

    X, y = _noise_matrix()
    getattr(bio_ml, trainer)(X, y, random_state=7)
    assert seen, "the scaler was never fitted; the spy is not wired up"
    assert max(seen) < len(X), f"a scaler fit saw all {len(X)} rows: {seen}"


def test_the_regressor_scaler_is_fitted_only_on_the_training_part(monkeypatch):
    import numpy as np

    from biosuite.core import bio_ml

    pytest.importorskip("sklearn")
    seen, RecordingScaler = _recording_scaler()
    monkeypatch.setattr(bio_ml, "StandardScaler", RecordingScaler)

    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 8))
    y = rng.normal(size=120)
    bio_ml.train_random_forest_regressor(X, y, n_estimators=20, random_state=7)

    assert len(seen) == 1 and seen[0] < 120, seen


def test_cross_validation_refits_the_scaler_for_every_fold(monkeypatch):
    """The other half of F2-1: CV used to reuse one globally scaled matrix.

    Fitted once for the holdout and once per CV fold, so more than one fit is
    proof the scaler lives inside the estimable pipeline.
    """
    from biosuite.core import bio_ml

    pytest.importorskip("sklearn")
    seen, RecordingScaler = _recording_scaler()
    monkeypatch.setattr(bio_ml, "StandardScaler", RecordingScaler)

    X, y = _noise_matrix()
    bio_ml.train_random_forest(X, y, n_estimators=20, random_state=7)

    assert len(seen) > 1, f"only one scaler fit -> CV scored a pre-scaled matrix: {seen}"


def test_random_labels_are_not_scored_as_a_discovery():
    """The regression test the audit proposed for F2-1, with the band justified.

    Under a null of 20 positives and 20 negatives the AUC standard error is
    about 0.065, so averaging 4 seeds puts the null band well inside +-0.15 of
    0.5.  Measured here: mean AUC 0.48.  This test does NOT discriminate the old
    code from the new one on a RandomForest — scaling is a monotone per-feature
    transform and cannot change a tree split — the scaler spy above is what does.
    """
    import numpy as np

    from biosuite.core import bio_ml

    pytest.importorskip("sklearn")
    aucs = []
    for seed in range(4):
        X, y = _noise_matrix(seed=seed)
        aucs.append(bio_ml.train_random_forest(X, y, n_estimators=50,
                                               random_state=7).auc)
    mean_auc = float(np.mean(aucs))
    assert abs(mean_auc - 0.5) < 0.15, aucs


def test_a_real_signal_is_still_learned():
    """The fix must not flatten genuine learning: a separable dataset scores high.

    Measured at 0.825 holdout accuracy / 0.924 AUC for this seed; the bound is
    deliberately loose so it pins "learns far better than the 0.5 chance level",
    not a snapshot of one run.
    """
    import numpy as np

    from biosuite.core import bio_ml

    pytest.importorskip("sklearn")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 20))
    beta = rng.normal(size=20)
    y = (X @ beta > 0).astype(int)

    result = bio_ml.train_random_forest(X, y, n_estimators=100, random_state=7)

    assert result.accuracy > 0.7, result.message
    assert result.auc > 0.75, result.message


# ── N15 / FIX-15 tail: text I/O must not depend on the machine's code page ───

# Built from escapes so this source file stays pure ASCII: typing Persian
# literals into a file has silently dropped characters in earlier sessions.
PROBE_TEXT = "ژن-سلام-β"


def _text_io_calls_without_encoding(path):
    """The text-mode I/O calls in *path* that never name an encoding.  Binary modes
    and any call forwarding **kwargs are skipped: nothing is provable about those
    statically, so smart_open is tested behaviourally instead.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tempfile_calls = {"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name, package = func.id, None
        elif isinstance(func, ast.Attribute):
            name = func.attr
            package = func.value.id if isinstance(func.value, ast.Name) else None
        else:
            continue
        is_text_call = (name in {"read_text", "write_text"}
                        or (name == "open" and package in {None, "codecs"})
                        or (name in tempfile_calls and package in {None, "tempfile"}))
        if not is_text_call:
            continue
        if any(k.arg == "encoding" for k in node.keywords if isinstance(k, ast.keyword)):
            continue
        if any(isinstance(a, ast.Starred) for a in node.args):
            continue
        if any(k.arg is None for k in node.keywords):          # **kwargs forwarding
            continue
        mode = ""
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = str(kw.value.value)
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            mode = str(node.args[1].value)
        if "b" in mode:
            continue
        if name in tempfile_calls and not mode:                 # default mode is 'w+b'
            continue
        found.append((str(path.relative_to(_REPO)), node.lineno, name, mode or "default"))
    return found


def test_no_text_io_call_in_the_package_is_left_without_an_explicit_encoding():
    """Regression guard for the encoding sweep: text written under cp1252 on Windows
    is not the same bytes as text written in UTF-8.
    """
    offenders = []
    for path in sorted((_REPO / "biosuite").rglob("*.py")):
        offenders.extend(_text_io_calls_without_encoding(path))
    assert offenders == [], "text I/O at the mercy of the locale: %s" % offenders


def test_the_locale_default_on_this_box_could_not_have_written_that_text():
    """Non-vacuity for the guard above, asserted where the hazard is real.

    On a box that prefers UTF-8 the old code was already correct, so there is
    nothing to demonstrate and the test says so instead of passing silently.
    """
    import locale

    encoding = locale.getpreferredencoding(False)
    if encoding.lower().replace("-", "").replace("_", "") == "utf8":
        pytest.skip("this box prefers UTF-8: the sweep changed nothing here")

    with pytest.raises((UnicodeEncodeError, LookupError)):
        PROBE_TEXT.encode(encoding)


def test_a_fasta_header_survives_the_reader_verbatim(tmp_path):
    """A reader that used to mangle UTF-8 headers on a Windows default locale.

    The header is the only place a user's sample name lives, so a code-page read
    silently renames the data instead of failing.
    """
    from biosuite.core.assembly import _load_reads_fasta

    fasta = tmp_path / "reads.fasta"
    fasta.write_bytes((">%s\n" % PROBE_TEXT + "ACGTACGTAA\n").encode("utf-8"))

    assert _load_reads_fasta(str(fasta)) == [(PROBE_TEXT, "ACGTACGTAA")]


def test_parse_bed_and_parse_vcf_return_non_ascii_fields_verbatim(tmp_path):
    """Two readers that used to mangle UTF-8 labels on a Windows default locale."""
    from biosuite.plotting import genome_browser as gb

    bed = tmp_path / "regions.bed"
    bed.write_text("chr1\t100\t200\t%s\t3\t+\n" % PROBE_TEXT, encoding="utf-8")
    regions = gb.parse_bed(str(bed))
    assert regions == [("chr1", 100, 200, PROBE_TEXT, 3.0)], regions

    vcf = tmp_path / "calls.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                   "chr1\t150\t.\tA\tT\t.\tPASS\tGENE=%s\n" % PROBE_TEXT, encoding="utf-8")
    variants = gb.parse_vcf(str(vcf))
    assert [v["info"] for v in variants] == ["GENE=" + PROBE_TEXT], variants


def test_smart_open_pins_utf_8_and_swaps_the_bom_for_text_modes(tmp_path):
    """The one **kwargs site the static guard cannot see.  smart_open uses utf-8-sig
    deliberately, so a BOM is consumed on read and written back; binary mode stays
    byte-exact, which is why the default applies to text modes only.
    """
    from biosuite.core.utils import smart_open

    path = tmp_path / "note.txt"
    with smart_open(str(path), "w") as handle:
        handle.write(PROBE_TEXT)

    raw = path.read_bytes()
    assert raw.decode("utf-8-sig") == PROBE_TEXT
    assert raw == PROBE_TEXT.encode("utf-8-sig")

    with smart_open(str(path), "r") as handle:
        assert handle.read() == PROBE_TEXT

    with smart_open(str(path), "rb") as handle:
        assert handle.read() == raw

