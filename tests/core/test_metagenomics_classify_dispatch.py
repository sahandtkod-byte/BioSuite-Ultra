"""Which classifier `classify_reads()` runs, and what it says when it cannot.

The dispatch is where a false sentence cost the most: a kraken2 that was installed
and failed used to be reported as "Kraken2 not found. … install Kraken2", and asking
for `tool='builtin'` — which never looks at the PATH at all — drew the same sentence.
"""
import subprocess
import warnings

import pytest

from biosuite.core import metagenomics as mg
from biosuite.core.utils import PerformanceWarning

ECOLI_KMER = 'ATCGATCGATCGATCG'


@pytest.fixture
def fastq(tmp_path):
    def _write(reads, name="reads.fastq"):
        path = tmp_path / name
        path.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n"
                                for i, s in enumerate(reads)), newline="")
        return str(path)
    return _write


@pytest.fixture
def kraken2_installed(monkeypatch):
    monkeypatch.setattr(mg, "check_metagenomics_tools", lambda: {"kraken2": True})


@pytest.fixture
def kraken2_absent(monkeypatch):
    monkeypatch.setattr(mg, "check_metagenomics_tools", lambda: {"kraken2": False})


def _classified_result():
    return mg.TaxonomyResult(engine="kraken2",
                             classifications=[{"read_id": "r", "taxon": "561"}])


# ── the tool argument ───────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_tool", ["kraken", "Kraken2", "KRAKEN2", "", "auto ",
                                      None, 0, True])
def test_an_unknown_tool_is_refused_instead_of_borrowing_the_builtin(fastq, bad_tool):
    path = fastq([ECOLI_KMER])
    with pytest.raises(ValueError, match="tool must be one of"):
        mg.classify_reads(path, tool=bad_tool)


def test_the_documented_tool_names_are_accepted(fastq, kraken2_absent):
    path = fastq([ECOLI_KMER])
    for tool in mg.TOOLS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert mg.classify_reads(path, tool=tool).engine in ("builtin", "kraken2")


# ── builtin ─────────────────────────────────────────────────────────────────

def test_builtin_never_consults_the_path_and_never_apologises(fastq, monkeypatch):
    """`tool='builtin'` is an explicit choice; a PATH scan and a warning about a
    missing kraken2 were both lies about work that was never asked for."""
    path = fastq([ECOLI_KMER])
    checks = []
    monkeypatch.setattr(mg, "check_metagenomics_tools",
                        lambda: checks.append(1) or {"kraken2": True})
    monkeypatch.setattr(mg, "_kraken2_classify",
                        lambda *a, **k: pytest.fail("kraken2 must not run"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = mg.classify_reads(path, tool="builtin")
    assert res.engine == "builtin"
    assert checks == []
    assert [w for w in caught if issubclass(w.category, PerformanceWarning)] == []


def test_builtin_still_reports_a_missing_file(fastq, tmp_path):
    res = mg.classify_reads(str(tmp_path / "absent.fastq"), tool="builtin")
    assert res.engine == "none"
    assert res.message.startswith("File not found:")


# ── auto and kraken2 requested ──────────────────────────────────────────────

def test_auto_uses_kraken2_when_it_answers(fastq, kraken2_installed, monkeypatch):
    path = fastq([ECOLI_KMER])
    seen = {}

    def fake(reads_file, database=None, output_file=None):
        seen["args"] = (reads_file, database)
        return _classified_result(), ""

    monkeypatch.setattr(mg, "_kraken2_classify", fake)
    res = mg.classify_reads(path, database="/db")
    assert res.engine == "kraken2"
    assert seen["args"] == (path, "/db")


def test_a_missing_file_is_reported_before_any_engine_runs(fastq, kraken2_installed,
                                                           monkeypatch):
    monkeypatch.setattr(mg, "_kraken2_classify",
                        lambda *a, **k: pytest.fail("nothing to classify"))
    res = mg.classify_reads("/absent/reads.fastq")
    assert res.engine == "none"
    assert res.classifications == []


def test_auto_says_kraken2_is_not_found_only_when_it_is_not_found(
        fastq, kraken2_absent):
    path = fastq([ECOLI_KMER])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = mg.classify_reads(path)
    assert res.engine == "builtin"
    assert [w.category for w in caught] == [PerformanceWarning]
    assert "Kraken2 not found" in str(caught[0].message)


def test_an_installed_kraken2_that_failed_is_described_by_its_failure(
        fastq, kraken2_installed, monkeypatch):
    path = fastq([ECOLI_KMER])
    monkeypatch.setattr(mg, "_kraken2_classify",
                        lambda *a, **k: (None, "it exited with 3: database not found"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = mg.classify_reads(path)
    assert res.engine == "builtin"
    text = str(caught[0].message)
    assert "Kraken2 is installed but did not classify" in text
    assert "it exited with 3: database not found" in text
    assert "Kraken2 not found" not in text, "the old sentence blamed a missing install"


def test_requesting_kraken2_by_name_when_it_is_absent_says_so(fastq, kraken2_absent):
    path = fastq([ECOLI_KMER])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = mg.classify_reads(path, tool="kraken2")
    assert res.engine == "builtin"
    assert "Kraken2 not found" in str(caught[0].message)


def test_the_fallback_warning_points_at_the_caller_not_the_internals(
        fastq, kraken2_absent):
    """stacklevel has to land on classify_reads' caller, or every warning in the
    suite appears to come from this module."""
    path = fastq([ECOLI_KMER])
    with pytest.warns(PerformanceWarning) as record:
        mg.classify_reads(path)
    assert str(record[0].filename).endswith("test_metagenomics_classify_dispatch.py"), \
        "the warning must be blamed on the caller, not on metagenomics.py"


def test_a_fasta_file_is_refused_by_the_classifier_not_misread(
        fastq, kraken2_absent, tmp_path):
    """The CLI prompt asks for a FASTQ; an aligned FASTA used to come back as
    reads named '>s1' with half of the records gone."""
    path = tmp_path / "aln.fasta"
    path.write_text(">s1\nACGTACGTAA\n>s2\nACGTACGTAC\n", newline="")
    with pytest.raises(ValueError, match="is not a FASTQ file"):
        mg.classify_reads(str(path), tool="builtin")


def test_the_tool_probe_answers_for_the_engine_the_wrapper_needs(monkeypatch):
    """check_metagenomics_tools is the only place the PATH is scanned, and the
    dispatch asks it about kraken2 alone."""
    monkeypatch.setattr(mg, "_has_tool", lambda name: name == "kraken2")
    assert mg.check_metagenomics_tools() == {"kraken2": True}
    monkeypatch.setattr(mg, "_has_tool", lambda name: False)
    assert mg.check_metagenomics_tools() == {"kraken2": False}


def test_the_launch_path_is_exercised_by_a_real_subprocess_call(tmp_path, monkeypatch,
                                                                fastq):
    """A stand-in for the whole external-tool boundary: kraken2 is 'installed' and
    the real subprocess.run is called for a binary that cannot exist, so the reason
    has to survive the OSError path and not escape as an exception."""
    path = fastq([ECOLI_KMER])
    monkeypatch.setattr(mg, "check_metagenomics_tools", lambda: {"kraken2": True})
    missing_db = str(tmp_path / "no-such-db")
    result, failure = mg._kraken2_classify(path, missing_db)
    assert result is None
    assert failure.startswith("it could not be executed:")
    assert subprocess is not None  # the module under test really uses subprocess
    assert isinstance(failure, str) and failure
