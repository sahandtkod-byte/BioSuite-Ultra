"""BLAST builtin engine deep tests (k-mer index/seed/banded align) + tool checks."""
import pytest

from biosuite.core import blast as bl


def test_check_blast_installed_shape():
    tools = bl.check_blast_installed()
    assert isinstance(tools, dict)
    assert all(isinstance(v, bool) for v in tools.values())


def test_kmer_index_groups_positions():
    seqs = [(0, 'ACGTACGTACGTACGT')]
    index = bl._build_kmer_index(seqs, k=6)
    assert 'ACGTAC' in index
    assert len(index['ACGTAC']) >= 1


def test_find_seed_hits_above_threshold():
    seqs = [(0, 'ACGTACGTACGTACGTACGTACGTACGTACGT')]
    index = bl._build_kmer_index(seqs, k=10)
    hits = bl._find_seed_hits('ACGTACGTACGTACGTACGT', index, k=10, max_hits=10)
    assert hits


def test_banded_align_perfect_match_positive():
    result = bl._banded_align_score('ACGTACGT', 'ACGTACGT', start1=0, start2=0)
    vals = result if isinstance(result, (list, tuple)) else [result]
    assert any((v is not None) and (not isinstance(v, str)) and v > 0 for v in vals)


def test_banded_align_mismatch_penalty():
    r1 = bl._banded_align_score('ACGTACGT', 'ACGTACGT', 0, 0)
    r2 = bl._banded_align_score('ACGTACGT', 'ACGTTCGA', 0, 0)
    def score(r):
        if isinstance(r, (list, tuple)):
            nums = [v for v in r if isinstance(v, (int, float))]
            return max(nums) if nums else 0.0
        return r if isinstance(r, (int, float)) else 0.0
    assert score(r1) > score(r2)


def test_estimate_evalue_decreases_with_score():
    e_hi = bl._estimate_evalue(10, db_size=100, query_len=5)
    e_lo = bl._estimate_evalue(50, db_size=100, query_len=5)
    assert e_lo < e_hi


def test_read_fasta_helper(tmp_path):
    p = tmp_path / 'db.fa'
    p.write_text(">a desc\nACGTACGT\n>b\nTTTT\n")
    recs = bl._read_fasta(str(p))
    assert len(recs) == 2


def test_builtin_search_perfect_hits(tmp_path):
    q = tmp_path / 'q.fa'
    q.write_text(">q\nACGTACGTACGTACGTACGTACGTACGTACGT\n")
    d = tmp_path / 'db.fa'
    d.write_text(">t1\nACGTACGTACGTACGTACGTACGTACGTACGT\n>t2\n" + "T" * 60 + "\n")
    res = bl._builtin_search(str(q), str(d), evalue=1e-2, max_hits=10, k=10)
    assert res.hits
    assert res.hits[0].percent_identity > 90


def test_format_blast_result_displays(tmp_path):
    q = tmp_path / 'q.fa'
    q.write_text(">q\nACGTACGTACGTACGTACGTACGTACGTACGT\n")
    d = tmp_path / 'db.fa'
    d.write_text(">t1\nACGTACGTACGTACGTACGTACGTACGTACGT\n")
    res = bl._builtin_search(str(q), str(d), evalue=1e-2, max_hits=10, k=10)
    txt = bl.format_blast_result(res)
    assert isinstance(txt, str) and 't1' in txt


def test_run_blast_external_fallback(tmp_path, monkeypatch):
    q = tmp_path / 'q.fa'
    q.write_text(">q\n" + "ACGT" * 16 + "\n")
    dbf = tmp_path / 'db.fa'
    dbf.write_text(">t1\n" + "ACGT" * 16 + "\n")
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: False)
    res = bl.run_blast(str(q), str(dbf), evalue=1e-2)
    assert res is not None
