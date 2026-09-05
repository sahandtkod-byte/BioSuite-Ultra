"""Regression tests for blast.py builtin seed-and-extend engine."""
import warnings
import pytest
from biosuite.core import blast as B


def _write(tmp_path, name, recs):
    p = tmp_path / name
    with open(p, 'w') as f:
        for i, (hdr, seq) in enumerate(recs):
            f.write(f">{hdr}\n{seq}\n")
    return str(p)


@pytest.fixture
def demo_files(tmp_path):
    q_seq = "ATGCGATCGATCGATCGGGTTTTTAAACCCGGGATCGATCGATCGAAA"
    db_seq = "NNN" + q_seq + "NNNGG"
    q = _write(tmp_path, 'q.fasta', [('q1', q_seq)])
    db = _write(tmp_path, 'db.fasta', [('s1', db_seq), ('s2', 'ACGT' * 12)])
    return q, db


def test_builtin_finds_planted_match(demo_files):
    q, db = demo_files
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = B._builtin_search(q, db, k=15)
    assert res.engine == 'builtin'
    assert any(h.subject_id == 's1' and h.percent_identity > 90 for h in res.hits)


def test_seed_clamped_on_short_kmers(demo_files):
    q, db = demo_files
    # k larger than every sequence used to silently find ZERO hits
    # (no seed could ever form) — it is now clamped down to fit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = B._builtin_search(q, db, k=60)  # > shortest sequence (48)
    assert 'clamped' in (res.message or '').lower()
    assert any(h.subject_id == 's1' for h in res.hits)


def test_aligner_end_positions_bounded(demo_files):
    q, db = demo_files
    _qs, q_seq = B._read_fasta(q)[0]
    (_s, db_seq), (_s2, _) = B._read_fasta(db)
    out = B._banded_align_score(10 * 'ATCG' + q_seq, 3 * 'ATCG' + db_seq, 0, 0)
    _, matches, mism, q_st, t_st, qe, te = out
    assert qe <= len(10 * "ATCG" + q_seq) and te <= len(3 * "ATCG" + db_seq) or qe == 0
    assert qe > q_st and te > t_st


def test_evalue_younger_check():
    e = B._estimate_evalue(100, 1_000_000, 400)
    assert 0 < e < 1e-5
    assert B._estimate_evalue(0, 1_000_000, 400) == 1.0


def test_run_blast_missing_query(tmp_path):
    res = B.run_blast(str(tmp_path / 'missing.fasta'), 'db')
    assert 'not found' in res.message


def test_format_output_smoke(demo_files):
    q, db = demo_files
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = B.run_blast(q, db)
    txt = B.format_blast_result(res)
    assert 'Engine:' in txt and 'Database:' in txt
    assert str(res.num_hits) in txt
