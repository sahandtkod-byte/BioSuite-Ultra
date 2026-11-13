"""Audit-remediation regression tests — Phase 4 (I/O cluster).

Every compressed-format test is a round trip: the same bytes are written
uncompressed and as .gz/.bz2, parsed, and compared.  The uncompressed parse is the
oracle, so the assertion never leans on the code under test.
"""
import bz2
import gzip
import os

import pytest

from biosuite.core import file_formats as ff
from biosuite.core import sequence as seq
from biosuite.core.utils import read_fasta_simple, smart_open

FASTA_TWO = ">first record\nACGTACGT\n>second\nTTTTGGGG\n"
FASTA_FIRST_ONLY = "\ufeff>rec1\nACGTACGT\n>rec2\nTTTT\n"
FASTQ_GOOD = "@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nGGGCCCAA\n+\nIIIIIIII\n"


def _write(tmp_path, name, text, opener=None, mode="w"):
    path = os.path.join(str(tmp_path), name)
    handle = open(path, "w", encoding="utf-8") if opener is None else opener(path, mode)
    with handle as fh:
        fh.write(text)
    return path


class TestStep41CompressedInput:
    """FIX-15: .gz/.bz2 were detected by the sniffers but never decompressed."""

    def test_fasta_round_trips_through_gzip(self, tmp_path):
        plain = _write(tmp_path, "a.fa", FASTA_TWO)
        gz = _write(tmp_path, "a.fa.gz", FASTA_TWO, opener=gzip.open, mode="wt")
        assert seq.read_fasta(gz) == seq.read_fasta(plain)
        assert len(seq.read_fasta(gz)) == 2, "compressed FASTA must not come back empty"

    def test_fasta_round_trips_through_bzip2(self, tmp_path):
        plain = _write(tmp_path, "b.fa", FASTA_TWO)
        bz = _write(tmp_path, "b.fa.bz2", FASTA_TWO, opener=bz2.open, mode="wt")
        assert seq.read_fasta(bz) == seq.read_fasta(plain)

    def test_fastq_round_trips_through_gzip(self, tmp_path):
        plain = _write(tmp_path, "a.fq", FASTQ_GOOD)
        gz = _write(tmp_path, "a.fq.gz", FASTQ_GOOD, opener=gzip.open, mode="wt")
        assert seq.read_fastq(gz) == seq.read_fastq(plain)
        assert len(seq.read_fastq(gz)) == 2

    def test_simple_reader_and_bed_and_gff_also_read_gzip(self, tmp_path):
        assert read_fasta_simple(
            _write(tmp_path, "c.fa.gz", FASTA_TWO, opener=gzip.open, mode="wt")) == \
            read_fasta_simple(_write(tmp_path, "c.fa", FASTA_TWO))

        bed = "chr1\t100\t200\tgeneA\t0\t+\nchr1\t300\t400\tgeneB\t0\t-\n"
        assert ff.parse_bed(_write(tmp_path, "a.bed.gz", bed, opener=gzip.open, mode="wt")) == \
            ff.parse_bed(_write(tmp_path, "a.bed", bed))

        gff = ("##gff-version 3\nchr1\t.\tgene\t100\t200\t.\t+\t.\tID=g1\n"
               "chr1\t.\tgene\t300\t400\t.\t-\t.\tID=g2\n")
        assert [r.start for r in ff.parse_gff(_write(tmp_path, "a.gff3.gz", gff,
                                                     opener=gzip.open, mode="wt"))] == \
            [100, 300]

    def test_no_silent_empty_result_on_real_ngs_input(self, tmp_path):
        """The pre-fix failure mode was [] or None with no exception at all."""
        gz = _write(tmp_path, "silent.fa.gz", FASTA_TWO, opener=gzip.open, mode="wt")
        got = seq.read_fasta(gz)
        assert got and all(seq_text for _h, seq_text in got)

    def test_smart_open_picks_the_right_backend(self, tmp_path):
        plain = _write(tmp_path, "p.txt", "hello\n")
        assert smart_open(plain).read() == "hello\n"
        gz = _write(tmp_path, "p.txt.gz", "hello\n", opener=gzip.open, mode="wt")
        assert smart_open(gz).read() == "hello\n"


class TestStep42BomAndEmptyHeaders:
    """N10: a UTF-8 BOM deleted the first record; a '>'-only header deleted a record."""

    def test_bom_does_not_swallow_the_first_record(self, tmp_path):
        path = os.path.join(str(tmp_path), "bom.fa")
        with open(path, "wb") as fh:
            fh.write(b"\xef\xbb\xbf>rec1\nACGTACGT\n>rec2\nTTTT\n")
        headers = [h for h, _s in seq.read_fasta(path)]
        assert headers == ["rec1", "rec2"], headers

    def test_empty_header_keeps_its_sequence(self, tmp_path):
        got = seq.read_fasta(_write(tmp_path, "empty.fa", ">a\nACGT\n>\nTTTT\n>b\nGGGG\n"))
        assert [h for h, _s in got] == ["a", "", "b"]
        assert dict(got)[""] == "TTTT"

    def test_lowercase_sequence_is_preserved(self, tmp_path):
        got = seq.read_fasta(_write(tmp_path, "lc.fa", ">x\nacgtACGT\n"))
        assert got == [("x", "acgtACGT")]


class TestStep43SimpleReader:
    """N11: read_fasta_simple crashed on a lone '>' and diverged from its own docs."""

    def test_lone_angle_bracket_does_not_raise(self, tmp_path):
        assert read_fasta_simple(_write(tmp_path, "only.fa", ">\n")) == [("", "")]

    def test_identifier_is_the_first_header_token(self, tmp_path):
        got = read_fasta_simple(_write(tmp_path, "sp.fa", ">name with spaces\nACGT\n"))
        assert got == [("name", "ACGT")]

    def test_missing_file_still_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_fasta_simple(os.path.join(str(tmp_path), "nope.fa"))

    def test_read_fasta_yields_the_full_header_by_design(self, tmp_path):
        path = _write(tmp_path, "full.fa", ">name with spaces\nACGT\n")
        assert seq.read_fasta(path) == [("name with spaces", "ACGT")]


class TestStep44FastqValidation:
    """N12: malformed FASTQ records were returned as data."""

    def test_truncated_final_record_is_dropped(self, tmp_path):
        got = seq.read_fastq(_write(tmp_path, "t.fq", "@r1\nACGT\n+\nIIII\n@r2\nACG\n"))
        assert got == [("r1", "ACGT", "IIII")]

    def test_wrong_separator_line_is_rejected(self, tmp_path):
        got = seq.read_fastq(_write(tmp_path, "s.fq", "@r1\nACGT\nWRONGSEP\nIIII\n"))
        assert got == []

    def test_quality_length_mismatch_is_rejected(self, tmp_path):
        got = seq.read_fastq(_write(tmp_path, "m.fq", "@r1\nACGTACGT\n+\nII\n"))
        assert got == []

    def test_valid_records_are_kept_beside_bad_ones(self, tmp_path):
        text = "@ok\nACGT\n+\nIIII\n@bad\nACGTACGT\n+\nII\n@ok2\nGGGG\n+\nIIII\n"
        got = seq.read_fastq(_write(tmp_path, "mix.fq", text))
        assert [n for n, _s, _q in got] == ["ok", "ok2"]

    def test_malformations_are_reported_once_not_silently(self, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            seq.read_fastq(_write(tmp_path, "warn.fq", "@r1\nACGTACGT\n+\nII\n"))
        assert any("malformed" in r.message for r in caplog.records), caplog.text

    def test_gzip_round_trip_still_works_after_validation(self, tmp_path):
        plain = _write(tmp_path, "v.fq", FASTQ_GOOD)
        gz = _write(tmp_path, "v.fq.gz", FASTQ_GOOD, opener=gzip.open, mode="wt")
        assert seq.read_fastq(gz) == seq.read_fastq(plain)


class TestStep45GffAttributeCollision:
    """N9: INFO attributes overwrote the record's real coordinates."""

    HOSTILE = ("##gff-version 3\nchr1\t.\tgene\t100\t200\t.\t+\t.\t"
               "ID=g1;start=999;strand=-;length=5\n")

    def test_core_columns_cannot_be_overwritten(self, tmp_path):
        df = ff.gff_to_dataframe(ff.parse_gff(_write(tmp_path, "h.gff3", self.HOSTILE)))
        row = df.iloc[0]
        assert int(row["start"]) == 100, row.to_dict()
        assert row["strand"] == "+", row.to_dict()
        assert int(row["length"]) == 101, row.to_dict()

    def test_attributes_are_still_reachable_under_a_prefix(self, tmp_path):
        df = ff.gff_to_dataframe(ff.parse_gff(_write(tmp_path, "h2.gff3", self.HOSTILE)))
        row = df.iloc[0]
        assert row["info_ID"] == "g1"
        assert row["info_start"] == "999"

    def test_ordinary_gff_still_produces_a_row_per_record(self, tmp_path):
        gff = ("##gff-version 3\nchr1\t.\tgene\t1\t10\t.\t+\t.\tID=a\n"
               "chr1\t.\tgene\t11\t20\t.\t-\t.\tID=b\n")
        df = ff.gff_to_dataframe(ff.parse_gff(_write(tmp_path, "ok.gff3", gff)))
        assert len(df) == 2 and list(df["start"]) == [1, 11]


class TestStep46NewickQuoting:
    """N15 / F2-8: the hand-written parser could not read quoted labels."""

    def test_quoted_label_with_colon_parses(self):
        node = ff.parse_newick("('node:1':0.1,x:0.2);")
        assert sorted(c.name for c in node.children) == ["node:1", "x"]

    def test_quoted_label_with_comma_and_space_parses(self):
        node = ff.parse_newick("('a,b':0.1,'Homo sapiens':0.2);")
        assert sorted(c.name for c in node.children) == ["Homo sapiens", "a,b"]

    def test_escaped_inner_quote_parses(self):
        node = ff.parse_newick("('it''s':0.1,x:0.2);")
        assert [c.name for c in node.children] == ["it's", "x"]

    def test_round_trip_preserves_awkward_names(self):
        names = ["Homo sapiens", "a,b", "node:1", "(weird)", "it's"]
        root = ff.TreeNode()
        root.children = []
        for i, name in enumerate(names):
            leaf = ff.TreeNode()
            leaf.name, leaf.branch_length, leaf.is_leaf = name, 0.1 * (i + 1), True
            root.children.append(leaf)
        text = ff.tree_to_newick(root)
        assert text.endswith(";"), text
        back = ff.parse_newick(text)
        assert sorted(c.name for c in back.children) == sorted(names)
        assert [round(c.branch_length, 6) for c in back.children] == \
            [round(0.1 * (i + 1), 6) for i in range(len(names))]

    def test_tiny_branch_length_parses(self):
        node = ff.parse_newick("(a:1e-05,b:0.2);")
        assert node.children[0].branch_length == pytest.approx(1e-05)

    def test_tiny_branch_length_survives_write_and_read(self):
        """%.4f used to render 1e-05 as 0.0000, silently erasing the branch."""
        leaves = []
        for name, bl in (("a", 1e-05), ("b", 0.2)):
            leaf = ff.TreeNode()
            leaf.name, leaf.branch_length, leaf.is_leaf = name, bl, True
            leaves.append(leaf)
        root = ff.TreeNode()
        root.children, root.branch_length, root.is_leaf = leaves, 0.3, False

        text = ff.tree_to_newick(root)
        assert "1e-05" in text, text
        back = ff.parse_newick(text)
        assert back.children[0].branch_length == pytest.approx(1e-05)
        assert back.children[1].branch_length == pytest.approx(0.2)

    def test_empty_string_raises_value_error_not_index_error(self):
        with pytest.raises(ValueError, match="empty Newick"):
            ff.parse_newick("")

    def test_unbalanced_parentheses_raise_value_error(self):
        with pytest.raises(ValueError, match="unbalanced"):
            ff.parse_newick("((a:1,b:1):0.5")

    def test_bad_branch_length_names_the_offending_token(self):
        with pytest.raises(ValueError, match="invalid branch length"):
            ff.parse_newick("(a:notanumber,b:0.2);")

    def test_plain_newick_still_parses(self):
        node = ff.parse_newick("(A:0.1,B:0.2)C:0.3;")
        assert [c.name for c in node.children] == ["A", "B"]
        assert node.branch_length == pytest.approx(0.3)
