"""Regression tests for ml_phylogeny bootstrap fixes."""
import pytest
from biosuite.core import ml_phylogeny as M

pytestmark = pytest.mark.skipif(not M.HAS_BIO, reason="biopython missing")

FASTA = """>human
ACGTACGTACGTACGTACGTACGTACGTACGT
>chimp
ACGTACGTACGTACGTACGTACGTACGTACGA
>mouse
AGGTACGGACGTACGTTCGTACGTACGTACGT
>rat
AGGTACGGACGTACGTTTGTACGTACGTACGT
"""


def _alignment():
    from io import StringIO
    from Bio import AlignIO
    return AlignIO.read(StringIO(FASTA), 'fasta')


def test_bootstrap_support_is_clade_frequency():
    aln = _alignment()
    supp = M._bootstrap_support(aln, n_replicates=30)
    assert supp
    for key, v in supp.items():
        assert 0 < v <= 1.0
        assert set(key) <= {'human', 'chimp', 'mouse', 'rat'}


def test_bootstrap_close_pair_supported():
    supp = M._bootstrap_support(_alignment(), n_replicates=50)
    pairs = [tuple(sorted(k)) for k in supp]
    assert any(set(k) == {'human', 'chimp'} or set(k) == {'mouse', 'rat'} for k in pairs)


def test_builtin_phylogeny_end_to_end(tmp_path):
    p = tmp_path / 'a.fasta'
    p.write_text(FASTA)
    res = M._builtin_phylogeny(str(p), bootstrap=10)
    assert res.engine == 'builtin'
    assert 'chimp' in res.newick_tree
    assert res.support_values


def test_parse_newick_roundtrip():
    tree = M.parse_newick("((a:1,b:1):1,c:2);")
    assert tree is not None
    assert tree.count_terminals() == 3
