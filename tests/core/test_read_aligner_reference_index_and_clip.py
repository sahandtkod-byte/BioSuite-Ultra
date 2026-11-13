"""Reference indexing and the read window the built-in aligner actually aligns.

Three contracts that only show up at the edges: the suffix index must never seed
on an ambiguous base, the FASTA reader has to survive blank lines and a file with
no header at all, and a read that hangs off the reference must be soft-clipped —
a CIGAR that consumes more reference than the contig holds is not a valid SAM
record and the variant caller downstream parses those rows.
"""
import re

import pytest

from biosuite.core.read_aligner import (
    _build_fasta_index, _build_suffix_index, _builtin_align_reads,
    _seed_and_extend,
)

REFERENCE = "ACGTACGTACGT"


def _index(ref=REFERENCE, k=4):
    return _build_suffix_index(ref, k=k)


def cigar_operations(cigar):
    """[(operation, length)] of a SAM CIGAR string."""
    return [(op, int(n)) for n, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar)]


def test_ambiguous_bases_are_never_seeded():
    index = _build_suffix_index("ACGTACGTNNACGTACGT", k=4)

    assert [kmer for kmer in index if "N" in kmer] == []
    assert index["ACGT"] == [0, 4, 10, 14]
    assert index["CGTA"] == [1, 11]


def test_a_read_of_ambiguous_bases_has_no_seed_to_work_with():
    hit = _seed_and_extend("NNNNNNNNNNNN", REFERENCE, "chr1", _index(), k=4)
    assert hit is None


def test_fasta_index_joins_wrapped_lines_and_ignores_blank_ones(tmp_path):
    path = tmp_path / "multi.fa"
    path.write_text(">chr1 a description\nACGT\n\nTGCA\n>chr2\nGG\n", encoding="utf-8")
    assert _build_fasta_index(str(path)) == {"chr1": "ACGTTGCA", "chr2": "GG"}


def test_fasta_index_without_a_header_is_empty(tmp_path):
    bare = tmp_path / "bare.fa"
    bare.write_text("ACGT\nTGCA\n", encoding="utf-8")
    empty = tmp_path / "empty.fa"
    empty.write_text("", encoding="utf-8")

    assert _build_fasta_index(str(bare)) == {}
    assert _build_fasta_index(str(empty)) == {}


def test_a_read_hanging_off_the_reference_end_is_soft_clipped():
    hit = _seed_and_extend("ACGTACGTGGGGGGGG", REFERENCE, "chr1", _index(), k=4)

    assert hit.cigar == "12M4S"
    assert hit.position == 0
    # the four hanging bases are not compared, so only three mismatches remain
    assert hit.edit_distance == 3
    assert hit.mapping_quality == 45


def test_a_read_reaching_before_the_reference_start_is_clipped_on_the_left():
    hit = _seed_and_extend("NNNNNNNNNNACGTACGT", REFERENCE, "chr1", _index(), k=4)

    assert hit.cigar == "10S8M"
    assert hit.position == 0
    assert hit.edit_distance == 0
    assert hit.mapping_quality == 60


@pytest.mark.parametrize("read", [
    "ACGTACGTGGGGGGGG", "NNNNNNNNNNACGTACGT", "TTTTACGTACGTAA", REFERENCE,
])
def test_no_cigar_ever_consumes_more_reference_than_exists(read):
    hit = _seed_and_extend(read, REFERENCE, "chr1", _index(), k=4)
    assert hit is not None

    consumed = sum(n for op, n in cigar_operations(hit.cigar) if op in "MDN=X")
    clipped = sum(n for op, n in cigar_operations(hit.cigar) if op == "S")
    assert consumed + clipped == len(read)
    assert hit.position + consumed <= len(REFERENCE)


def test_a_record_without_a_sequence_line_stops_the_read(tmp_path):
    """A blank sequence line is a truncated record; nothing after it is claimed.

    The built-in reader consumes four lines per record, so an empty sequence
    line ends the run rather than desynchronising it into the quality string.
    """
    long_reference = REFERENCE * 4          # 48 bases: long enough for k=15 seeds
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{long_reference}\n", encoding="utf-8")
    reads = tmp_path / "reads.fq"
    reads.write_text(
        f"@r1\n{long_reference}\n+\n{'I' * len(long_reference)}\n"
        f"@r2\n\n+\n{'I' * len(long_reference)}\n"
        f"@r3\n{long_reference}\n+\n{'I' * len(long_reference)}\n",
        encoding="utf-8")

    report = _builtin_align_reads(str(reads), str(ref))

    assert report.engine == "builtin"
    assert report.total_reads == 1
    assert report.mapped_reads == 1
    assert [a.read_id for a in report.alignments] == ["r1"]
    # no output path was given, so nothing was written next to the inputs
    assert report.output_file == ""
    assert not list(tmp_path.glob("*.sam"))
