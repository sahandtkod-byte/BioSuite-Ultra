"""Reading the FASTAs, building the k-mer index and finding seeds — the built-in engine's front half.

These three helpers decide what the built-in searcher can even see, so each is pinned on
its exact output rather than on "non-empty": the index's key-and-offset lists, the
diagonal arithmetic of a seed hit (`db_pos - i`, which can be negative), the 50-candidate
cap and the `max_hits` budget.  The old `test_blast.py`/`test_blast_builtin.py` versions
of these cases asserted `len(index) > 0` and
`'ACGT' in index or 'CGTA' in index or 'GTAC' in index`, which any wrong index would also
have satisfied.
"""
import pytest

from biosuite.core import blast as bl


# ── _read_fasta ──────────────────────────────────────────────────────────────

def test_one_record_comes_back_as_the_pair_it_holds(tmp_path):
    path = tmp_path / "one.fasta"
    path.write_text(">seq1\nACGTACGT\n", encoding="utf-8")
    assert bl._read_fasta(str(path)) == [("seq1", "ACGTACGT")]


def test_the_identifier_is_the_first_header_token_and_the_rest_is_dropped(tmp_path):
    """`record.id`, not `record.description`: the built-in engine keys its hits by
    whatever it stored, and a long description would otherwise appear in the report."""
    path = tmp_path / "desc.fasta"
    path.write_text(">s1 Homo sapiens chromosome 1\nACGT\n>s2\nTTTT\n", encoding="utf-8")
    assert bl._read_fasta(str(path)) == [("s1", "ACGT"), ("s2", "TTTT")]


def test_lowercase_sequence_survives_the_reader_unchanged(tmp_path):
    path = tmp_path / "case.fasta"
    path.write_text(">s\nacgtACGT\n", encoding="utf-8")
    assert bl._read_fasta(str(path)) == [("s", "acgtACGT")], \
        "the reader does not rewrite the data; normalisation belongs to the index"


# ── _build_kmer_index ────────────────────────────────────────────────────────

def test_the_index_maps_every_kmer_to_the_subject_and_offset(tmp_path):
    index = bl._build_kmer_index([("seq1", "ACGTACGT")], k=4)
    assert dict(index) == {
        'ACGT': [('seq1', 0), ('seq1', 4)],
        'CGTA': [('seq1', 1)],
        'GTAC': [('seq1', 2)],
        'TACG': [('seq1', 3)],
    }


def test_a_window_containing_n_is_not_indexed_but_the_clean_neighbours_are():
    """'ACGTNNACGT' contributes the two unambiguous 4-mers and nothing that overlaps an
    N, so an `N` cannot seed a hit but does not delete the rest of the record."""
    index = bl._build_kmer_index([("seq1", "ACGTNNACGT")], k=4)
    assert dict(index) == {'ACGT': [('seq1', 0), ('seq1', 6)]}
    assert not [kmer for kmer in index if 'N' in kmer]


@pytest.mark.parametrize("sequence, expected_keys", [
    pytest.param("acgtacgt", {'ACGT', 'CGTA', 'GTAC', 'TACG'}, id="lower-case-uppercased"),
    pytest.param("ACGUACGU", {'ACGT', 'CGTA', 'GTAC', 'TACG'}, id="rna-u-counts-as-t"),
])
def test_the_index_normalises_case_and_rna_uracil(sequence, expected_keys):
    index = bl._build_kmer_index([("s", sequence)], k=4)
    assert set(index) == expected_keys
    assert index['ACGT'] == [('s', 0), ('s', 4)]


def test_a_sequence_shorter_than_the_seed_contributes_nothing():
    assert dict(bl._build_kmer_index([("s", "")], k=4)) == {}
    assert dict(bl._build_kmer_index([("s", "ACG")], k=5)) == {}


# ── _find_seed_hits ──────────────────────────────────────────────────────────

def test_a_seed_hit_is_keyed_by_the_diagonal_it_sits_on():
    """Each candidate is `(subject_id, db_pos - query_pos)`, so every query position that
    agrees with the same diagonal lands in one list — that offset is what the extension
    step anchors both sequences to."""
    index = bl._build_kmer_index([("db", "ACGTACGTACGTACGT")], k=4)
    assert bl._find_seed_hits("ACGTACGT", index, k=4) == [
        (("db", 0), [0, 1, 2, 3, 4]),
        (("db", 4), [0, 1, 2, 3, 4]),
        (("db", 8), [0, 1, 2, 3, 4]),
        (("db", 12), [0]),
        (("db", -4), [4]),
    ], "the last entry is a diagonal that starts before the subject"


def test_a_shifted_query_reports_the_diagonal_offset_it_needs():
    index = bl._build_kmer_index([("db", "ACGTACGTACGTACGT")], k=4)
    assert bl._find_seed_hits("TTACGTAC", index, k=4) == [
        (("db", 2), [1, 2, 3, 4]),
        (("db", 6), [1, 2, 3, 4]),
        (("db", -2), [2, 3, 4]),
        (("db", 10), [1, 2]),
    ]


def test_a_query_with_no_shared_seed_yields_no_candidates():
    index = bl._build_kmer_index([("db", "ACGTACGTACGTACGT")], k=4)
    assert bl._find_seed_hits("TTTTTTTT", index, k=4) == []


def test_the_seed_budget_counts_occurrences_not_candidates():
    """`max_hits` stops the scan once that many seed occurrences have been recorded, so
    one query position matching twenty subject positions can spend the whole budget."""
    index = bl._build_kmer_index([("db", "ACGT" * 20)], k=4)
    candidates = bl._find_seed_hits("ACGT" * 12, index, k=4, max_hits=3)
    assert len(candidates) == 20
    assert all(positions == [0] for _key, positions in candidates), \
        "the scan stopped after the first query position spent the budget"


def test_only_the_fifty_best_diagonals_are_kept_for_extension():
    index = bl._build_kmer_index([("db", "ACGT" * 60)], k=4)
    candidates = bl._find_seed_hits("ACGT" * 100, index, k=4)
    assert len(candidates) == 50
    assert candidates[0][0] == ("db", 0)
    assert len(candidates[0][1]) == 4, "the busiest diagonal carries the most seeds"
