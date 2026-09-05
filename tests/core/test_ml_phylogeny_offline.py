"""ml_phylogeny: builtin tree + graceful offline paths."""
import pytest

from biosuite.core import ml_phylogeny as mp


DIVERGENT = ['ACGTACGTACGT', 'ACGTACGAACGT', 'ACGTTCGAACGT',
             'TCGTTCGAACGT', 'TTGTTCGAACGT', 'TTGTTCGTACGT']


def _write_fasta(tmp_path, seqs, name='aln.fasta'):
    p = tmp_path / name
    p.write_text('\n'.join(f'>s{i}\n{s}' for i, s in enumerate(seqs)))
    return str(p)


def test_check_phylo_tools_shape():
    tools = mp.check_phylo_tools()
    assert isinstance(tools, dict)
    assert all(isinstance(v, bool) for v in tools.values())


def test_parse_newick_roundtrip():
    out = mp.parse_newick('(a:1,b:1,(c:2,d:2):0.5);')
    assert hasattr(out, 'root') or hasattr(out, 'clade') or out
    # chain via the same parser used for builtin output
    out2 = mp.parse_newick(str(out))
    assert out2


def test_builtin_phylogeny_shortalignment(tmp_path):
    aln = _write_fasta(tmp_path, DIVERGENT[:4])
    res = mp._builtin_phylogeny(aln, bootstrap=50)
    assert res is not None
    assert getattr(res, 'engine', None) == 'builtin'
    assert getattr(res, 'newick_tree', None)


def test_build_tree_from_alignment_builtin(tmp_path):
    aln = _write_fasta(tmp_path, DIVERGENT[:4])
    res = mp.build_tree(aln, method='nj')
    assert res is not None


def test_format_phylo_report_stringifier(tmp_path):
    aln = _write_fasta(tmp_path, DIVERGENT[:4])
    res = mp.build_tree(aln, method='nj')
    txt = mp.format_phylo_report(res)
    assert isinstance(txt, str) and txt


def test_raxml_missing_tool_graceful(tmp_path):
    aln = _write_fasta(tmp_path, DIVERGENT[:4])
    res = mp._raxml_run(aln, str(tmp_path), bootstrap=10)
    assert res is None or getattr(res, 'message', None) is not None
