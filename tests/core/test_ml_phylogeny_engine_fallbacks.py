"""Contracts for ml_phylogeny's degraded paths: missing Biopython, unreadable
alignments, IQ-TREE staging failures, and every external-engine result/fall-back.

Nothing here launches a real raxml-ng or iqtree: ``subprocess.run`` is intercepted,
so the assertions cover the command line that would have run, the parsed Newick, and
the warning that must accompany each failure instead of a silent NJ result.
"""
import importlib.util
import os
import pathlib
import subprocess
import sys
import warnings

import pytest
from Bio import AlignIO

from biosuite.core import ml_phylogeny as mp
from biosuite.core.utils import PerformanceWarning

SEQS = [("s1", "ACGTACGTAA"), ("s2", "ACGTACGTAC"), ("s3", "ACGAACGTAA"), ("s4", "ACGTGCGTAA")]


@pytest.fixture
def alignment_file(tmp_path):
    path = tmp_path / "aln.fasta"
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in SEQS), encoding="utf-8")
    return str(path)


@pytest.fixture
def biopython_free(monkeypatch):
    """Re-execute the module top level with ``Bio`` unimportable.

    ``importlib.reload`` would keep the attributes the first import bound, leaving
    ``HAS_BIO`` True.  The dotted name stays inside ``biosuite.core`` because the
    module uses relative imports.
    """
    monkeypatch.setitem(sys.modules, "Bio", None)
    name = "biosuite.core.ml_phylogeny_without_biopython"
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(mp.__file__))
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, fresh)
    spec.loader.exec_module(fresh)
    assert fresh.HAS_BIO is False, "the optional-dependency guard is what is under test"
    return fresh


@pytest.fixture
def completed():
    """Factory for a finished-subprocess stand-in."""
    def _make(returncode=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(args=["unused"], returncode=returncode,
                                           stdout=stdout, stderr=stderr)
    return _make


def test_biopython_free_module_degrades_on_every_entry_point(biopython_free):
    fresh = biopython_free
    assert fresh._nj_tree_from_alignment(object()) == (None, "Biopython not installed")
    assert fresh._bootstrap_support(None) == ({}, 0)
    result = fresh._builtin_phylogeny("does-not-exist.fasta")
    assert (result.engine, result.newick_tree, result.support_values) == ("builtin", "", {})
    assert result.message == "Biopython not installed"
    assert fresh.parse_newick("((s1:0.1,s2:0.2)s3:0.3);") is None
    with pytest.raises(RuntimeError, match="Bio.Phylo is not installed"):
        fresh._newick_from_tree(object())


def test_bootstrap_support_returns_nothing_for_a_missing_alignment(alignment_file):
    assert mp._bootstrap_support(None) == ({}, 0)


def test_neighbour_joining_builds_a_four_taxon_tree_serialised_as_newick(alignment_file):
    """The built-in engine's own happy path: NJ tree -> Newick text, no error."""
    alignment = AlignIO.read(alignment_file, "fasta")
    tree, error = mp._nj_tree_from_alignment(alignment)
    assert error is None
    assert [t.name for t in tree.get_terminals()] == ["s1", "s2", "s3", "s4"]
    newick = mp._newick_from_tree(tree)
    assert newick == "((s1:0.00000,s2:0.10000)Inner1:0.00000,s3:0.10000,s4:0.10000)Inner2:0.00000;"
    assert [t.name for t in mp.parse_newick(newick).get_terminals()] == ["s1", "s2", "s3", "s4"]


def test_builtin_phylogeny_returns_newick_clade_support_and_a_honest_message(alignment_file):
    """Support keys are sorted taxon tuples; the message still says neighbour-joining,
    and no likelihood is claimed for a tree that computed none."""
    result = mp._builtin_phylogeny(alignment_file, bootstrap=20, seed=7)
    assert result.engine == "builtin"
    assert result.newick_tree.startswith("((")
    assert result.support_values == {("s1", "s2"): 0.75, ("s1", "s3"): 0.2, ("s2", "s3"): 0.05}
    assert result.message == "Neighbor-Joining tree with 20 bootstrap replicates"
    assert result.tree_file == ""
    assert result.log_likelihood is None, \
        "0.0 would be a measured likelihood; neighbour-joining measures none"


def test_bootstrap_support_is_reproducible_only_when_seeded(alignment_file):
    alignment = AlignIO.read(alignment_file, "fasta")
    first, failed_first = mp._bootstrap_support(alignment, n_replicates=20, seed=7)
    second, _ = mp._bootstrap_support(alignment, n_replicates=20, seed=7)
    other, _ = mp._bootstrap_support(alignment, n_replicates=20, seed=8)
    assert failed_first == 0
    assert first == second, "seeded bootstrapping is the documented reproducibility handle"
    assert first != other
    assert all(0.0 <= v <= 1.0 for v in first.values())
    assert sum(first.values()) == pytest.approx(1.0), \
        "20 replicates contribute exactly 20 clade counts — one non-trivial split per tree"
    assert all(len(key) > 1 for key in first), "support is per non-trivial clade"


def test_bootstrap_ignores_replicates_whose_neighbour_joining_failed(alignment_file, monkeypatch):
    """A replicate with no tree contributes no clade — it must not raise, and it
    must be counted."""
    alignment = AlignIO.read(alignment_file, "fasta")
    monkeypatch.setattr(mp, "_nj_tree_from_alignment", lambda a: (None, "NJ exploded"))
    assert mp._bootstrap_support(alignment, n_replicates=4, seed=7) == ({}, 4)


def test_builtin_reports_an_unreadable_alignment_without_raising(tmp_path):
    junk = tmp_path / "junk.fasta"
    junk.write_text("# not fasta\n%%%%\n", encoding="utf-8")
    result = mp._builtin_phylogeny(str(junk))
    assert result.engine == "builtin"
    assert result.message.startswith("Error reading alignment:")
    assert "Pearson" in result.message, "Biopython's own hint must reach the caller"
    assert result.newick_tree == ""


def test_builtin_propagates_the_neighbour_joining_error_message(alignment_file, monkeypatch):
    monkeypatch.setattr(mp, "_nj_tree_from_alignment",
                        lambda a: (None, "distance matrix is singular"))
    result = mp._builtin_phylogeny(alignment_file, bootstrap=0)
    assert result.engine == "builtin"
    assert result.message == "distance matrix is singular"


def test_build_tree_refuses_an_absent_alignment_file(tmp_path):
    missing = str(tmp_path / "gone.fasta")
    result = mp.build_tree(missing)
    assert result.engine == "none"
    assert result.message == f"File not found: {missing}"
    assert result.newick_tree == ""


def test_raxml_success_returns_the_ml_tree_and_removes_its_directory(alignment_file,
                                                                    monkeypatch, completed):
    seen = {}
    prefix_dirs = []

    def fake_run(cmd, *args, **kwargs):
        seen["cmd"] = list(cmd)
        prefix = cmd[cmd.index("--prefix") + 1]
        prefix_dirs.append(os.path.dirname(prefix))
        with open(prefix + ".raxml.bestTree", "w", encoding="utf-8") as handle:
            handle.write("((s1:0.1,s2:0.2)0.95:0.05,(s3:0.2,s4:0.3)0.80:0.04)root;\n")
        return completed(0, stdout="best trees written")

    monkeypatch.setattr(mp.subprocess, "run", fake_run)
    monkeypatch.setattr(mp, "_has_tool", lambda name: name == "raxml-ng")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mp.build_tree(alignment_file, method="raxml", bootstrap=20)

    assert result.engine == "raxml"
    assert result.newick_tree == "((s1:0.1,s2:0.2)0.95:0.05,(s3:0.2,s4:0.3)0.80:0.04)root;"
    assert result.message == "RAxML-NG maximum likelihood tree"
    assert result.tree_file == "", "the tree is returned as text; the directory is gone"
    assert seen["cmd"][:2] == ["raxml-ng", "--all"]
    assert seen["cmd"][seen["cmd"].index("--model") + 1] == "GTRGAMMA"
    assert seen["cmd"][seen["cmd"].index("--bs-trees") + 1] == "20"
    assert [w.category for w in caught] == [], "a completed ML run must not warn"
    assert [d for d in prefix_dirs if os.path.isdir(d)] == []


def test_build_tree_returns_the_iqtree_result_when_the_engine_completes(alignment_file,
                                                                       monkeypatch,
                                                                       completed):
    seen = {}

    def fake_run(cmd, *args, **kwargs):
        seen["cmd"] = list(cmd)
        with open(cmd[cmd.index("-s") + 1] + ".treefile", "w", encoding="utf-8") as handle:
            handle.write("(s1:0.1,s2:0.2,(s3:0.2,s4:0.3)0.7);\n")
        return completed(0)

    monkeypatch.setattr(mp.subprocess, "run", fake_run)
    monkeypatch.setattr(mp, "_has_tool", lambda name: name == "iqtree")
    result = mp.build_tree(alignment_file, method="iqtree", model="GTR+I+G", bootstrap=50)
    assert result.engine == "iqtree"
    assert result.newick_tree == "(s1:0.1,s2:0.2,(s3:0.2,s4:0.3)0.7);"
    assert result.message == "IQ-TREE maximum likelihood tree"
    assert seen["cmd"][:2] == ["iqtree", "-s"]
    assert seen["cmd"][seen["cmd"].index("-m") + 1] == "GTR+I+G"
    assert seen["cmd"][seen["cmd"].index("-bb") + 1] == "50"
    assert not os.path.isdir(os.path.dirname(os.path.dirname(seen["cmd"][2])))


def test_iqtree_uses_an_alignment_that_is_already_staged_without_copying(monkeypatch,
                                                                         completed, tmp_path):
    """No copy when the staged path and the input path are the same file."""
    stage = tmp_path / "iqtree-stage"
    stage.mkdir()
    alignment_file = stage / "aln.fasta"
    alignment_file.write_text("".join(f">{n}\n{s}\n" for n, s in SEQS), encoding="utf-8")

    def no_copy(*args, **kwargs):
        raise AssertionError("the alignment is already where IQ-TREE will read it")

    seen = {}

    def fake_run(cmd, *args, **kwargs):
        seen["cmd"] = list(cmd)
        return completed(0)

    monkeypatch.setattr(mp.shutil, "copyfile", no_copy)
    monkeypatch.setattr(mp.subprocess, "run", fake_run)
    with pytest.warns(RuntimeWarning, match="was not produced"):
        assert mp._iqtree_run(str(alignment_file), str(tmp_path)) is None
    assert seen["cmd"][seen["cmd"].index("-s") + 1] == str(alignment_file)


def test_iqtree_reports_a_staging_failure_instead_of_running_anything(alignment_file,
                                                                      monkeypatch):
    def refusing_copy(src, dst, *args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(mp.shutil, "copyfile", refusing_copy)
    called = []
    monkeypatch.setattr(mp.subprocess, "run",
                        lambda cmd, *a, **k: called.append(cmd))
    with pytest.warns(RuntimeWarning, match="could not be staged"):
        assert mp._iqtree_run(alignment_file, str(os.path.dirname(alignment_file))) is None
    assert called == [], "a staging failure must not launch the binary"


@pytest.mark.parametrize("crash", [
    pytest.param(lambda cmd, *a, **k: (_ for _ in ()).throw(OSError(2, "No such file")),
                 id="missing-binary"),
    pytest.param(lambda cmd, *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd, 1)),
                 id="timeout"),
])
def test_iqtree_launch_failure_is_warned_and_returns_none(alignment_file, monkeypatch, crash):
    monkeypatch.setattr(mp.subprocess, "run", crash)
    with pytest.warns(RuntimeWarning, match="iqtree could not be executed"):
        assert mp._iqtree_run(alignment_file, str(os.path.dirname(alignment_file))) is None


def test_parse_newick_returns_none_for_text_biopython_cannot_read():
    assert mp.parse_newick("") is None, "an empty Newick string raises ValueError in Bio.Phylo"
    assert mp.parse_newick("((s1:0.1,s2:0.2") is None
    assert mp.parse_newick("(s1:0.1,s2:0.2);(s3:0.1,s4:0.2);") is None
    assert [t.name for t in mp.parse_newick("(s1:0.1,s2:0.2);").get_terminals()] == ["s1", "s2"]


def test_parse_newick_logs_why_it_returned_none(caplog):
    """Three different failures used to produce the same silent None."""
    import logging
    with caplog.at_level(logging.WARNING):
        assert mp.parse_newick("") is None
    assert "There are no trees in this file" in caplog.text


def test_parse_newick_of_real_text_leaves_the_log_quiet(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        assert mp.parse_newick("(s1:0.1,s2:0.2);") is not None
    assert caplog.text == ""


# ── the arguments build_tree accepts ────────────────────────────────────────

@pytest.mark.parametrize("bad_method", ["njx", "raxml-ng", "RAxML", "Maximum likelihood",
                                        "", "auto ", None, 0])
def test_a_method_outside_the_documented_set_is_refused(alignment_file, bad_method):
    """Asking for an engine by a name the module does not know used to be answered
    with a neighbour-joining tree and no word about it."""
    with pytest.raises(ValueError, match="method must be one of"):
        mp.build_tree(alignment_file, method=bad_method, bootstrap=0)


def test_the_documented_methods_all_reach_a_tree(alignment_file, monkeypatch):
    monkeypatch.setattr(mp, "_has_tool", _tool_for())
    for method in mp.METHODS:
        result = mp.build_tree(alignment_file, method=method, bootstrap=0)
        assert result.engine == "builtin", f"{method} asks for NJ and gets NJ"


def test_the_builtin_engine_keeps_its_historical_alias(alignment_file, monkeypatch):
    """'nj' is what older callers pass; refusing it would be a break, not a fix."""
    monkeypatch.setattr(mp, "_has_tool", _tool_for())
    assert mp.build_tree(alignment_file, method="nj", bootstrap=0).engine == "builtin"
    assert mp.build_tree(alignment_file, method="builtin", bootstrap=0).engine == "builtin"


@pytest.mark.parametrize("bad_bootstrap", [-5, -1, 1.5, "100", True, None])
def test_a_replicate_count_that_is_not_a_count_is_refused(alignment_file, bad_bootstrap):
    with pytest.raises(ValueError, match="bootstrap must be a whole number"):
        mp.build_tree(alignment_file, bootstrap=bad_bootstrap)


def test_the_replicate_count_reaches_the_tree_in_the_message(alignment_file):
    assert mp.build_tree(alignment_file, method="nj", bootstrap=0).message == \
        "Neighbor-Joining tree without bootstrap support"


def test_a_negative_replicate_count_never_claims_replicates(alignment_file):
    """-5 used to reach the message verbatim: 'with -5 bootstrap replicates'."""
    with pytest.raises(ValueError):
        mp.build_tree(alignment_file, bootstrap=-5)


# ── the working directory ───────────────────────────────────────────────────

def test_a_working_directory_that_cannot_be_removed_is_logged(alignment_file, monkeypatch,
                                                              caplog):
    """``ignore_errors=True`` hid a leaked directory; the tree is still the
    answer, but the leak has to be visible."""
    import logging

    def fake_run(cmd, *args, **kwargs):
        with open(cmd[cmd.index("--prefix") + 1] + ".raxml.bestTree", "w",
                  encoding="utf-8") as handle:
            handle.write("(s1,s2);\n")
        return subprocess.CompletedProcess(args=["unused"], returncode=0,
                                           stdout="", stderr="")

    def refuse(path, *args, **kwargs):
        raise OSError(16, "Device or resource busy")

    monkeypatch.setattr(mp.subprocess, "run", fake_run)
    monkeypatch.setattr(mp, "_has_tool", _tool_for("raxml-ng"))
    monkeypatch.setattr(mp.shutil, "rmtree", refuse)
    with caplog.at_level(logging.WARNING):
        result = mp.build_tree(alignment_file, method="raxml", bootstrap=0)
    assert result.engine == "raxml", "a cleanup failure must not lose the tree"
    assert "Could not remove the temporary analysis directory" in caplog.text
    assert "Device or resource busy" in caplog.text


def _tool_for(*names):
    return lambda name: name in names


@pytest.mark.parametrize("engine, failure, matches", [
    pytest.param("raxml", "launch-error", "raxml-ng could not be executed", id="raxml-launch"),
    pytest.param("raxml", "nonzero", "raxml-ng exited with 3: raxml-ng: exploded",
                 id="raxml-exit"),
    pytest.param("raxml", "no-output", "raxml-ng succeeded but.*was not produced",
                 id="raxml-no-tree"),
    pytest.param("iqtree", "launch-error", "iqtree could not be executed", id="iqtree-launch"),
    pytest.param("iqtree", "nonzero", "iqtree exited with 2: iqtree: exploded",
                 id="iqtree-exit"),
    pytest.param("iqtree", "no-output", "iqtree succeeded but.*was not produced",
                 id="iqtree-no-tree"),
])
def test_every_external_engine_failure_returns_no_tree(alignment_file, monkeypatch, completed,
                                                       engine, failure, matches):
    """A failed ML run must never hand back a tree — the caller falls back knowingly."""
    binary = "raxml-ng" if engine == "raxml" else "iqtree"

    def fake_run(cmd, *args, **kwargs):
        if failure == "launch-error":
            raise OSError(2, "No such file or directory")
        if failure == "nonzero":
            return completed(3 if engine == "raxml" else 2, stderr=f"{binary}: exploded")
        return completed(0)

    monkeypatch.setattr(mp.subprocess, "run", fake_run)
    call = mp._raxml_run if engine == "raxml" else mp._iqtree_run
    with pytest.warns(RuntimeWarning, match=matches):
        assert call(alignment_file, os.path.dirname(alignment_file), bootstrap=5) is None


def test_requested_engine_that_fails_warns_then_returns_the_builtin_tree(alignment_file,
                                                                        monkeypatch, completed):
    monkeypatch.setattr(mp, "_has_tool", _tool_for("raxml-ng"))
    monkeypatch.setattr(mp.subprocess, "run", lambda cmd, *a, **k: completed(1, stderr="boom"))
    with pytest.warns(PerformanceWarning, match="NOT maximum likelihood"):
        result = mp.build_tree(alignment_file, method="raxml", bootstrap=5, seed=1)
    assert result.engine == "builtin"
    assert result.newick_tree.startswith("((")


def test_requested_engine_that_is_absent_never_borrows_the_other_one(alignment_file,
                                                                    monkeypatch, completed):
    """method='raxml' must not quietly run iqtree, and must say that it fell back."""
    monkeypatch.setattr(mp, "_has_tool", _tool_for("iqtree"))
    launched = []
    monkeypatch.setattr(mp.subprocess, "run",
                        lambda cmd, *a, **k: launched.append(cmd) or completed(0))
    with pytest.warns(PerformanceWarning, match="method='raxml'"):
        result = mp.build_tree(alignment_file, method="raxml", bootstrap=0)
    assert launched == []
    assert result.engine == "builtin"
    assert result.support_values == {}


@pytest.mark.parametrize("installed", [True, False], ids=["engine-failed", "engine-absent"])
def test_iqtree_request_that_cannot_be_satisfied_warns_and_returns_builtin(alignment_file,
                                                                          monkeypatch,
                                                                          completed, installed):
    launched = []
    monkeypatch.setattr(mp, "_has_tool", _tool_for("iqtree") if installed else _tool_for())
    if installed:
        monkeypatch.setattr(mp.subprocess, "run",
                            lambda cmd, *a, **k: launched.append(cmd) or completed(1,
                                                                                   stderr="iqtree: exploded"))
    else:
        monkeypatch.setattr(mp.subprocess, "run", lambda cmd, *a, **k: launched.append(cmd))
    with pytest.warns(PerformanceWarning, match="method='iqtree'"):
        result = mp.build_tree(alignment_file, method="iqtree", bootstrap=0)
    assert result.engine == "builtin"
    assert result.newick_tree.startswith("((")
    assert (len(launched) == 1) is installed, "nothing is launched when the binary is absent"


def test_auto_uses_iqtree_when_only_iqtree_is_installed(alignment_file, monkeypatch, completed):
    def fake_run(cmd, *args, **kwargs):
        with open(cmd[cmd.index("-s") + 1] + ".treefile", "w", encoding="utf-8") as handle:
            handle.write("(s1,s2,s3,s4);\n")
        return completed(0)

    monkeypatch.setattr(mp, "_has_tool", _tool_for("iqtree"))
    monkeypatch.setattr(mp.subprocess, "run", fake_run)
    result = mp.build_tree(alignment_file, method="auto", bootstrap=5)
    assert result.engine == "iqtree"
    assert result.newick_tree == "(s1,s2,s3,s4);"
    assert result.message == "IQ-TREE maximum likelihood tree"


def test_auto_with_no_engine_returns_builtin_without_warning(alignment_file, monkeypatch):
    monkeypatch.setattr(mp, "_has_tool", _tool_for())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mp.build_tree(alignment_file, method="auto", bootstrap=5, seed=3)
    assert result.engine == "builtin"
    assert [w for w in caught if issubclass(w.category, PerformanceWarning)] == []


def test_auto_falls_back_loudly_when_the_detected_engine_cannot_finish(alignment_file,
                                                                       monkeypatch, completed):
    """Unlike the no-tool case, an installed engine that fails must be reported."""
    monkeypatch.setattr(mp, "_has_tool", _tool_for("raxml-ng"))
    monkeypatch.setattr(mp.subprocess, "run", lambda cmd, *a, **k: completed(0))
    with pytest.warns(PerformanceWarning, match="method='auto'"):
        result = mp.build_tree(alignment_file, method="auto", bootstrap=0)
    assert result.engine == "builtin"

