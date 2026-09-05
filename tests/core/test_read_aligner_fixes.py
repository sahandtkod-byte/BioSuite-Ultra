"""Regression tests for read_aligner.py review fixes."""
import random

import pytest

from biosuite.core.read_aligner import align_reads, _seed_and_extend, _build_suffix_index


@pytest.fixture()
def ref_and_reads(tmp_path):
    random.seed(2)
    genome = ''.join(random.choice('ACGT') for _ in range(5000))
    reads = []
    for i in range(20):
        st = random.randint(0, 4900)
        r = list(genome[st:st + 100])
        if i % 5 == 0:
            r[30] = random.choice([b for b in 'ACGT' if b != r[30]])
        reads.append((''.join(r), st))
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\n" + genome + "\n")
    fq = tmp_path / "reads.fq"
    fq.write_text(''.join(f"@r{j}\n{s}\n+\n{'I' * 100}\n" for j, (s, _) in enumerate(reads)))
    return str(fa), str(fq), reads, genome


def test_positions_exact(tmp_path, ref_and_reads):
    fa, fq, reads, _ = ref_and_reads
    rep = align_reads(fa, fq, str(tmp_path / "out.sam"), tool='builtin')
    assert rep.mapped_reads == 20
    for j, a in enumerate(rep.alignments):
        assert a.reference_id == 'chr1'
        assert a.position == reads[j][1], (j, a.position, reads[j][1])


def test_edit_distance_real_counts(tmp_path, ref_and_reads):
    fa, fq, reads, _ = ref_and_reads
    rep = align_reads(fa, fq, str(tmp_path / "out.sam"), tool='builtin')
    with_mm = [j for j in range(20) if j % 5 == 0]
    clean = [j for j in range(20) if j % 5 != 0]
    for j in with_mm:
        assert rep.alignments[j].edit_distance == 1  # before fix: always 0
    for j in clean:
        assert rep.alignments[j].edit_distance == 0


def test_leading_seedless_bases_dont_shift_position():
    # A read whose first seeds start at read offset 3 must still map to the
    # true offset (old code used seeds[0][1] and shifted the locus right).
    ref = "A" * 17 + "ACGTAGCTAGCTTGGAACCGGTTACCGGTAATCG" + "T" * 20
    read = "NNN" + "ACGTAGCTAGCTTGGAACCGGTTACCGGTAATCG"
    index = _build_suffix_index(ref, k=15)
    hit = _seed_and_extend(read, ref, "chr1", index, k=15, seed_threshold=2)
    # read[0] maps to 17-3=14; the old code reported seeds[0][1]=17 (shifted).
    assert hit is not None and hit.position == 14
    assert hit.edit_distance == 3  # the three N flanks


def test_sam_contains_sequence_and_quality(tmp_path, ref_and_reads):
    fa, fq, reads, _ = ref_and_reads
    sam = tmp_path / "out.sam"
    align_reads(fa, fq, str(sam), tool='builtin')
    rows = [l.rstrip('\n').split('\t') for l in open(sam) if not l.startswith('@')]
    assert all(len(r) >= 11 for r in rows)
    for j, r in enumerate(rows[:5]):
        assert r[9] == reads[j][0]          # sequence emitted
        assert r[10] == 'I' * 100           # flat qualities emitted


def test_sam_feeds_variant_caller(tmp_path, ref_and_reads):
    # End-to-end: aligner SAM must be parseable by the variant caller.
    from biosuite.core.variant_calling import _read_sam, _pileup_reads
    fa, fq, reads, _ = ref_and_reads
    sam = tmp_path / "out.sam"
    align_reads(fa, fq, str(sam), tool='builtin')
    rd = _read_sam(str(sam))
    assert len(rd) == 20
    piles = _pileup_reads(rd)
    assert len(piles['chr1']) > 0


def test_unmapped_reads_flagged(tmp_path):
    (tmp_path / "ref.fa").write_text(">chr1\n" + "ACGT" * 100 + "\n")
    (tmp_path / "reads.fq").write_text("@junk\n" + "T" * 100 + "\n+\n" + "I" * 100 + "\n")
    rep = align_reads(str(tmp_path / "ref.fa"), str(tmp_path / "reads.fq"),
                      str(tmp_path / "out.sam"), tool='builtin')
    assert rep.unmapped_reads == 1
    row = [l.split('\t') for l in open(str(tmp_path / "out.sam")) if not l.startswith('@')][0]
    assert int(row[1]) & 4


def test_missing_files_graceful(tmp_path):
    rep = align_reads(str(tmp_path / "no.fa"), str(tmp_path / "no.fq"))
    assert rep.engine == 'none'
