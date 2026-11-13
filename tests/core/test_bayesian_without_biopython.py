"""The module without Biopython: it has to stop, not limp along.

`HAS_BIO` is decided at import, so these tests re-execute the module top level with
`Bio` unimportable — a reload keeps the attributes the first import bound, which is why
the fixture builds a fresh module object instead of reloading in place.  Three things
were wrong here: the no-op paths reported `engine='builtin'` for work that never
happened, `_fasta_to_nexus` wrote nothing and said nothing, and `_mrbayes_run` then
launched `mb` on a script whose `execute` line named a file that did not exist, so the
user got a verdict on the tool instead of the reason.
"""
import importlib.util
import os
import pathlib
import sys
import tempfile

import pytest

from biosuite.core import bayesian_phylogeny as bp

FASTA_4 = ">A\nACGTACGT\n>B\nACGAACGT\n>C\nACGTGCGT\n>D\nTCGTACGA\n"


@pytest.fixture
def biopython_free(monkeypatch):
    """A fresh copy of the module, imported with `Bio` unavailable."""
    monkeypatch.setitem(sys.modules, "Bio", None)
    name = "biosuite.core.bayesian_phylogeny_without_biopython"
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(bp.__file__))
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, fresh)
    spec.loader.exec_module(fresh)
    assert fresh.HAS_BIO is False, "the optional-dependency guard is what is under test"
    return fresh


@pytest.fixture
def alignment_file(tmp_path):
    path = tmp_path / "aln.fasta"
    path.write_text(FASTA_4, encoding="utf-8")
    return str(path)


def test_the_builtin_chain_stops_before_reading_the_alignment(biopython_free,
                                                              alignment_file):
    result = biopython_free._builtin_bayesian(alignment_file)
    assert (result.engine, result.newick_tree, result.num_samples) == ("none", "", 0)
    assert result.message == "Biopython not installed"


def test_no_partial_nexus_is_written(biopython_free, alignment_file, tmp_path):
    nexus = tmp_path / "out.nex"
    assert biopython_free._fasta_to_nexus(alignment_file, str(nexus)) is None
    assert not nexus.exists(), "no partial NEXUS may be written without Biopython"


def test_mrBayes_is_not_launched_on_a_nexus_that_was_never_written(biopython_free,
                                                                  alignment_file,
                                                                  monkeypatch):
    launched = []
    monkeypatch.setattr(bp.subprocess, "run", lambda *a, **k: launched.append(a))
    temp_root = tempfile.gettempdir()

    def staging_dirs():
        return {name for name in os.listdir(temp_root)
                if name.startswith("biosuite_mrbayes_")}

    before = staging_dirs()
    with pytest.warns(RuntimeWarning, match=r"alignment\.nex.* was not written"):
        assert biopython_free._mrbayes_run(alignment_file, 50) is None
    assert launched == [], "mb must not be started on a script it cannot execute"
    assert staging_dirs() == before, \
        "the staging directory a refused run created is its own again"
