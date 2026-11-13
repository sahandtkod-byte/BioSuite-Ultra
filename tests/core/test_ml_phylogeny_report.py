"""`format_phylo_report()` — the text the CLI prints for a tree.

The report used to build its lines as a fixed list with an empty string in the
log-likelihood slot whenever no likelihood existed, so every neighbour-joining
report carried a blank line; it also printed "Tree: " with nothing after it, and
truncated a long tree without saying how much was hidden.
"""
import pytest

from biosuite.core import ml_phylogeny as mp

LONG_TREE = "(" + ",".join(f"taxon{i}:0.1" for i in range(30)) + ");"


def test_a_bare_result_carries_no_claims_at_all():
    """The defaults are the shape the report renders when an engine never ran:
    no tree, no support, no likelihood, and None rather than a measured zero."""
    result = mp.PhyloResult(engine="builtin")
    assert result.newick_tree == ""
    assert result.support_values == {}
    assert result.tree_file == ""
    assert result.log_likelihood is None
    assert result.message == ""


def test_the_header_is_two_lines_and_no_blank_one():
    text = mp.format_phylo_report(mp.PhyloResult(engine="builtin"))
    assert text.splitlines() == ["=== Phylogenetic Analysis Report ===",
                                 "Engine: builtin",
                                 "Tree: (none)"]
    assert "" not in text.splitlines()


def test_a_tree_with_no_likelihood_omits_the_line_rather_than_blanking_it():
    text = mp.format_phylo_report(mp.PhyloResult(engine="raxml", newick_tree="(a,b);"))
    assert "Log-likelihood" not in text
    assert text.splitlines()[1:3] == ["Engine: raxml", "Tree: (a,b);"]


def test_a_measured_likelihood_is_printed_to_two_places():
    res = mp.PhyloResult(engine="raxml", log_likelihood=-123.456, newick_tree="(a,b);",
                         support_values={("a", "b"): 1.0}, message="done")
    assert mp.format_phylo_report(res).splitlines() == [
        "=== Phylogenetic Analysis Report ===",
        "Engine: raxml",
        "Log-likelihood: -123.46",
        "Tree: (a,b);",
        "Bootstrap clusters: 1",
        "Note: done",
    ]


def test_a_likelihood_of_exactly_zero_is_still_a_measurement():
    """0.0 was falsy and disappeared with the 'not computed' lines; a real zero is
    a number the engine produced."""
    text = mp.format_phylo_report(mp.PhyloResult(engine="raxml", log_likelihood=0.0))
    assert "Log-likelihood: 0.00" in text


def test_a_long_tree_is_truncated_with_its_real_length():
    text = mp.format_phylo_report(mp.PhyloResult(engine="builtin",
                                                 newick_tree=LONG_TREE))
    line = [l for l in text.splitlines() if l.startswith("Tree:")][0]
    assert line.startswith(f"Tree: {LONG_TREE[:100]}...")
    assert f"({len(LONG_TREE)} characters in total)" in line
    assert len(LONG_TREE) > 100


def test_a_short_tree_is_printed_whole():
    text = mp.format_phylo_report(mp.PhyloResult(engine="builtin",
                                                 newick_tree="(a,b);"))
    assert "Tree: (a,b);" in text
    assert "characters in total" not in text


def test_the_support_line_counts_clades_not_reads():
    res = mp.PhyloResult(engine="builtin", newick_tree="(a,b);",
                         support_values={("a", "b"): 0.75, ("a", "c"): 0.2})
    assert "Bootstrap clusters: 2" in mp.format_phylo_report(res)


def test_no_support_and_no_message_add_nothing():
    res = mp.PhyloResult(engine="builtin", newick_tree="(a,b);")
    text = mp.format_phylo_report(res)
    assert "Bootstrap" not in text
    assert "Note:" not in text


def test_the_missing_file_case_explains_itself_end_to_end(tmp_path):
    result = mp.build_tree(str(tmp_path / "nope.fasta"))
    text = mp.format_phylo_report(result)
    assert "Engine: none" in text
    assert f"Note: File not found: {tmp_path / 'nope.fasta'}" in text
    assert "Tree: (none)" in text


def test_a_fallback_to_neighbour_joining_is_visible_in_the_report():
    """The report line is the last chance to tell the user the tree is not ML:
    the engine name has to be the truth."""
    res = mp.PhyloResult(engine="builtin", newick_tree="(a,b);",
                         message="Neighbor-Joining tree without bootstrap support")
    text = mp.format_phylo_report(res)
    assert "Engine: builtin" in text
    assert "Neighbor-Joining" in text
