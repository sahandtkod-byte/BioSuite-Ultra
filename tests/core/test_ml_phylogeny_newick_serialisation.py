"""Turning a Biopython tree into Newick text, and reading Newick text back.

The bug this file exists for (N7) is the silent fall-back that serialised a tree with
``str(tree)`` and handed back a Python repr — a lenient Newick reader then parsed
``rooted=False`` into a taxon name.  Everything here asserts the serialiser produces
real Newick or raises, never a repr.
"""
import io
import os

import pytest
from Bio import Phylo

from biosuite.core import ml_phylogeny as mp

TAXA = ["s1_alt", "s2_alt", "s3_alt", "s4_alt"]
SEQS = ["ACGTACGTAA", "ACGTACGTAC", "ACGAACGTAA", "ACGTGCGTAA"]


def _write_alignment(tmp_path, names=None):
    names = names or ["s1", "s2", "s3", "s4"]
    path = os.path.join(str(tmp_path), "aln.fasta")
    with open(path, "w", encoding="utf-8") as handle:
        for name, seq in zip(names, SEQS):
            handle.write(f">{name}\n{seq}\n")
    return path


def _tree(*names):
    return Phylo.BaseTree.Tree(
        root=Phylo.BaseTree.Clade(
            clades=[Phylo.BaseTree.Clade(name=n) for n in names]))


# ── writing ─────────────────────────────────────────────────────────────────

def test_the_builtin_engine_writes_newick_not_a_repr(tmp_path):
    res = mp.build_tree(_write_alignment(tmp_path), method="nj", bootstrap=0)
    assert res.engine == "builtin"
    assert res.newick_tree.startswith("(")
    assert res.newick_tree.rstrip().endswith(";")
    assert "Tree(rooted" not in res.newick_tree
    assert "Clade(" not in res.newick_tree


def test_taxa_names_survive_the_round_trip(tmp_path):
    res = mp.build_tree(_write_alignment(tmp_path, names=TAXA), method="nj", bootstrap=0)
    tree = Phylo.read(io.StringIO(res.newick_tree), "newick")
    assert sorted(c.name for c in tree.get_terminals()) == sorted(TAXA)


def test_the_projects_own_parser_reads_the_trees_it_writes(tmp_path):
    res = mp.build_tree(_write_alignment(tmp_path, names=TAXA), method="nj", bootstrap=0)
    parsed = mp.parse_newick(res.newick_tree)
    assert parsed is not None
    assert sorted(c.name for c in parsed.get_terminals()) == sorted(TAXA)


@pytest.mark.parametrize("label", ["Homo sapiens", "a,b", "node:1", "(weird)", "16S rRNA"])
def test_labels_needing_quotes_round_trip(label):
    """Taxon names with spaces, commas, colons or parentheses are legal and must
    not corrupt the tree for an outside reader."""
    newick = mp._newick_from_tree(_tree(label, "other"))
    parsed = Phylo.read(io.StringIO(newick), "newick")
    assert sorted(c.name for c in parsed.get_terminals()) == sorted([label, "other"])


def test_branch_lengths_are_written():
    tree = Phylo.BaseTree.Tree(root=Phylo.BaseTree.Clade(
        clades=[Phylo.BaseTree.Clade(name="a", branch_length=0.25),
                Phylo.BaseTree.Clade(name="b", branch_length=0.75)]))
    newick = mp._newick_from_tree(tree)
    assert "0.25" in newick and "0.75" in newick
    parsed = Phylo.read(io.StringIO(newick), "newick")
    assert [round(c.branch_length, 2) for c in parsed.get_terminals()] == [0.25, 0.75]


def test_the_serialiser_does_not_swallow_errors(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("writer unavailable")

    monkeypatch.setattr(mp._BioPhylo, "write", boom)
    with pytest.raises(RuntimeError, match="writer unavailable"):
        mp._newick_from_tree(_tree("A"))


def test_trailing_newline_is_stripped(monkeypatch):
    def writes_a_block(tree, handle, fmt):
        handle.write("(A,B);\n\n")

    monkeypatch.setattr(mp._BioPhylo, "write", writes_a_block)
    assert mp._newick_from_tree(_tree("A")) == "(A,B);"


def test_serialising_without_biopython_says_which_package_is_missing():
    """It used to be a NameError for an attribute the import block never bound."""
    import importlib.util
    import sys

    name = "biosuite.core.ml_phylogeny_probe_no_bio"
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(mp.__file__))
    saved = sys.modules.get("Bio")
    sys.modules["Bio"] = None
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        assert module.HAS_BIO is False
        with pytest.raises(RuntimeError, match="Bio.Phylo is not installed"):
            module._newick_from_tree(_tree("A"))
    finally:
        sys.modules["Bio"] = saved
        sys.modules.pop(name, None)


# ── reading ─────────────────────────────────────────────────────────────────

def test_reading_a_simple_tree():
    tree = mp.parse_newick("((a:1,b:1):1,c:2);")
    assert tree is not None
    assert tree.count_terminals() == 3


def test_reading_text_that_is_not_a_tree_returns_none(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        assert mp.parse_newick("((s1:0.1,s2:0.2") is None
    assert "Could not read a Newick tree" in caplog.text


def test_the_reason_is_a_real_sentence(caplog):
    """Three different failures used to answer with the same silent None."""
    import logging
    with caplog.at_level(logging.WARNING):
        assert mp.parse_newick("") is None
    assert "There are no trees in this file" in caplog.text


def test_a_non_string_is_answered_with_a_message_about_what_it_was(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        assert mp.parse_newick(None) is None
    assert "None" in caplog.text


def test_reading_a_good_tree_leaves_no_log_line(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        assert mp.parse_newick("(a:0.1,b:0.2);") is not None
    assert caplog.text == ""


def test_a_lenient_reader_still_accepts_bare_words():
    """Biopython reads 'garbage' as a one-taxon tree; that is the reader's
    behaviour, recorded here so a future 'fix' cannot hide it."""
    tree = mp.parse_newick("garbage")
    assert [t.name for t in tree.get_terminals()] == ["garbage"]
