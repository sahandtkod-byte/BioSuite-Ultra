"""The command lines ml_phylogeny sends to RAxML-NG and IQ-TREE, and what it
accepts as an installed engine.

No test here runs a real binary: ``subprocess.run`` is intercepted, so a flag
regression fails in this file instead of in a run that takes hours.  The engine
choice matters twice over — FIX-2 was a wrong flag name and IQ-TREE's ``-o``
(outgroup) being used as an output prefix, and N16 was legacy RAxML being accepted
and then driven with RAxML-NG flags.
"""
import os
import subprocess

import pytest

from biosuite.core import ml_phylogeny as mp
from biosuite.core.utils import PerformanceWarning

SEQS = [("s1", "ACGTACGTAA"), ("s2", "ACGTACGTAC"), ("s3", "ACGAACGTAA"),
        ("s4", "ACGTGCGTAA")]


@pytest.fixture
def alignment_file(tmp_path):
    path = tmp_path / "aln.fasta"
    path.write_text("".join(f">{n}\n{s}\n" for n, s in SEQS), encoding="utf-8")
    return str(path)


def _proc(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["unused"], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def _tool_for(*names):
    return lambda name: name in names


# ── which engines the machine has ───────────────────────────────────────────

def test_legacy_raxml_is_not_reported_as_a_usable_ml_engine(monkeypatch):
    """`_raxml_run` emits RAxML-NG flags a legacy binary cannot parse."""
    monkeypatch.setattr(mp, "_has_tool", lambda name: name == "RAxML")
    tools = mp.check_phylo_tools()
    assert tools["raxml_legacy"] is True
    assert tools["raxml"] is False


def test_raxmlng_and_iqtree_are_reported_separately(monkeypatch):
    monkeypatch.setattr(mp, "_has_tool", lambda name: name == "raxml-ng")
    assert mp.check_phylo_tools() == {"raxml": True, "raxml_legacy": False,
                                      "iqtree": False}
    monkeypatch.setattr(mp, "_has_tool", lambda name: name == "iqtree")
    assert mp.check_phylo_tools()["iqtree"] is True


def test_every_tool_answer_is_a_boolean(monkeypatch):
    monkeypatch.setattr(mp, "_has_tool", lambda name: True)
    assert all(isinstance(v, bool) for v in mp.check_phylo_tools().values())
    monkeypatch.setattr(mp, "_has_tool", lambda name: False)
    assert mp.check_phylo_tools() == {"raxml": False, "raxml_legacy": False,
                                      "iqtree": False}


def test_nothing_is_probed_when_builtin_is_asked_for(alignment_file, monkeypatch):
    calls = []
    monkeypatch.setattr(mp, "_has_tool",
                        lambda name: calls.append(name) or False)
    assert mp.build_tree(alignment_file, method="builtin", bootstrap=0).engine == "builtin"
    assert calls == [], "the PATH scan is not needed for a built-in run"


# ── RAxML-NG ────────────────────────────────────────────────────────────────

def test_raxml_argv_is_prefix_and_msa_not_out_prefix(alignment_file, monkeypatch):
    seen = {}

    def fake_run(cmd, *args, **kwargs):
        seen["cmd"] = list(cmd)
        prefix = cmd[cmd.index("--prefix") + 1]
        with open(prefix + ".raxml.bestTree", "w", encoding="utf-8") as handle:
            handle.write("((s1:0.1,s2:0.2)s3);\n")
        return _proc(0)

    monkeypatch.setattr(mp.subprocess, "run", fake_run)
    monkeypatch.setattr(mp, "_has_tool", _tool_for("raxml-ng"))
    result = mp.build_tree(alignment_file, method="raxml", bootstrap=20)
    assert seen["cmd"][:2] == ["raxml-ng", "--all"]
    assert "--prefix" in seen["cmd"] and "--out-prefix" not in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--msa") + 1] == alignment_file
    assert seen["cmd"][seen["cmd"].index("--bs-trees") + 1] == "20"
    assert seen["cmd"][seen["cmd"].index("--model") + 1] == "GTRGAMMA"
    assert result.engine == "raxml"


def test_a_named_model_reaches_the_engine(alignment_file, monkeypatch):
    seen = {}

    def fake_run(cmd, *args, **kwargs):
        seen["cmd"] = list(cmd)
        with open(cmd[cmd.index("--prefix") + 1] + ".raxml.bestTree", "w",
                  encoding="utf-8") as handle:
            handle.write("(s1,s2);\n")
        return _proc(0)

    monkeypatch.setattr(mp.subprocess, "run", fake_run)
    monkeypatch.setattr(mp, "_has_tool", _tool_for("raxml-ng"))
    mp.build_tree(alignment_file, method="raxml", model="GTR+G", bootstrap=0)
    assert seen["cmd"][seen["cmd"].index("--model") + 1] == "GTR+G"


def test_raxml_nonzero_exit_is_reported_not_swallowed(alignment_file, monkeypatch,
                                                      tmp_path):
    monkeypatch.setattr(mp.subprocess, "run",
                        lambda cmd, *a, **k: _proc(3, stderr="FATAL: unrecognized option"))
    with pytest.warns(RuntimeWarning, match="exited with 3: FATAL: unrecognized option"):
        assert mp._raxml_run(alignment_file, str(tmp_path), bootstrap=10) is None


def test_raxml_exit_zero_without_a_tree_is_a_warning(alignment_file, monkeypatch,
                                                    tmp_path):
    monkeypatch.setattr(mp.subprocess, "run", lambda cmd, *a, **k: _proc(0))
    with pytest.warns(RuntimeWarning, match="succeeded but.*was not produced"):
        assert mp._raxml_run(alignment_file, str(tmp_path), bootstrap=10) is None


def test_a_missing_binary_warns_instead_of_returning_none_silently(alignment_file,
                                                                   monkeypatch,
                                                                   tmp_path):
    def raise_fne(cmd, *args, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(mp.subprocess, "run", raise_fne)
    with pytest.warns(RuntimeWarning, match="raxml-ng could not be executed"):
        assert mp._raxml_run(alignment_file, str(tmp_path), bootstrap=10) is None


# ── IQ-TREE ─────────────────────────────────────────────────────────────────

def test_iqtree_never_passes_the_outgroup_flag_as_a_prefix(alignment_file, monkeypatch,
                                                           tmp_path):
    seen = {}

    def fake_run(cmd, *args, **kwargs):
        seen["cmd"] = list(cmd)
        staged = cmd[cmd.index("-s") + 1]
        with open(staged + ".treefile", "w", encoding="utf-8") as handle:
            handle.write("(s1,s2,(s3,s4));\n")
        return _proc(0)

    monkeypatch.setattr(mp.subprocess, "run", fake_run)
    monkeypatch.setattr(mp, "_has_tool", _tool_for("iqtree"))
    res = mp._iqtree_run(alignment_file, str(tmp_path), bootstrap=10)
    assert "-o" not in seen["cmd"], "-o is IQ-TREE's outgroup flag, not an output prefix"
    assert "-nt" in seen["cmd"] and "-bb" in seen["cmd"]
    assert res is not None and res.engine == "iqtree"
    assert res.newick_tree.startswith("(")


def test_iqtree_writes_its_artefacts_into_a_stage_directory_of_our_own(
        alignment_file, monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, *args, **kwargs):
        seen["cmd"] = list(cmd)
        return _proc(0)

    monkeypatch.setattr(mp.subprocess, "run", fake_run)
    with pytest.warns(RuntimeWarning, match="was not produced"):
        mp._iqtree_run(alignment_file, str(tmp_path), bootstrap=10)
    staged = seen["cmd"][seen["cmd"].index("-s") + 1]
    assert os.path.dirname(staged) == os.path.join(str(tmp_path), "iqtree-stage")
    assert staged != alignment_file, "the caller's own file is never the prefix"


def test_staging_survives_the_output_dir_being_the_alignment_folder(
        alignment_file, monkeypatch, tmp_path):
    """Copying a file onto itself raised SameFileError and the wrapper died."""
    monkeypatch.setattr(mp.subprocess, "run", lambda cmd, *a, **k: _proc(0))
    with pytest.warns(RuntimeWarning, match="was not produced"):
        assert mp._iqtree_run(alignment_file, str(tmp_path), bootstrap=10) is None


def test_iqtree_nonzero_exit_is_reported_not_swallowed(alignment_file, monkeypatch,
                                                       tmp_path):
    monkeypatch.setattr(mp.subprocess, "run",
                        lambda cmd, *a, **k: _proc(2, stderr="error: bad model"))
    with pytest.warns(RuntimeWarning, match="exited with 2: error: bad model"):
        assert mp._iqtree_run(alignment_file, str(tmp_path), bootstrap=10) is None


# ── the fall-back ───────────────────────────────────────────────────────────

def test_explicit_engine_request_that_cannot_run_warns(alignment_file, monkeypatch):
    monkeypatch.setattr(mp, "_has_tool", _tool_for())
    with pytest.warns(PerformanceWarning, match="NOT maximum likelihood"):
        res = mp.build_tree(alignment_file, method="iqtree", bootstrap=0)
    assert res.engine == "builtin"


def test_explicit_raxml_request_does_not_run_a_legacy_binary(alignment_file, monkeypatch):
    monkeypatch.setattr(mp, "_has_tool", lambda name: name == "RAxML")
    called = []
    monkeypatch.setattr(mp.subprocess, "run",
                        lambda cmd, *a, **k: called.append(cmd) or _proc(0))
    with pytest.warns(PerformanceWarning):
        res = mp.build_tree(alignment_file, method="raxml", bootstrap=5)
    assert called == [], "no subprocess may be launched against a binary we do not support"
    assert res.engine == "builtin"


def test_a_failed_attempt_does_not_leak_its_directory(alignment_file, monkeypatch):
    made = []
    real_mkdtemp = mp.tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        made.append(path)
        return path

    monkeypatch.setattr(mp.tempfile, "mkdtemp", tracking_mkdtemp)
    monkeypatch.setattr(mp, "_has_tool", _tool_for("iqtree"))
    monkeypatch.setattr(mp.subprocess, "run",
                        lambda cmd, *a, **k: _proc(1, stderr="boom"))
    with pytest.warns((RuntimeWarning, PerformanceWarning)):
        mp.build_tree(alignment_file, method="iqtree", bootstrap=5)
    assert made, "the test did not exercise a temporary directory"
    assert [d for d in made if os.path.isdir(d)] == []


def test_auto_without_tools_does_not_warn_on_every_call(alignment_file, monkeypatch):
    """The documented default is NJ, so it must not emit a warning each time."""
    import warnings

    monkeypatch.setattr(mp, "_has_tool", _tool_for())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mp.build_tree(alignment_file, method="auto", bootstrap=5)
    assert [w for w in caught if issubclass(w.category, PerformanceWarning)] == []
