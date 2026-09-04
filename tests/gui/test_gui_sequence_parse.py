"""Regression tests for the sequence-input parser used by the GUI.

The old parser stripped lines starting with '>' '@' '+' but then treated
every other line as sequence — so a FASTQ quality line beginning with
A/C/G/T (perfectly legal Phred) was silently included in the DNA string,
corrupting downstream analysis. These tests pin the fixed behavior.
"""
from biosuite.gui.tabs.sequence_analysis import _parse_sequence_text


class TestRawSequence:
    def test_plain_sequence(self):
        assert _parse_sequence_text("ACGTACGT") == "ACGTACGT"

    def test_multiline_raw(self):
        assert _parse_sequence_text("ACGT\nTTTT\n") == "ACGTTTTT"

    def test_spaces_removed(self):
        assert _parse_sequence_text("ACG TAC GT") == "ACGTACGT"

    def test_empty(self):
        assert _parse_sequence_text("") == ""
        assert _parse_sequence_text("   \n  ") == ""


class TestFasta:
    def test_single_record(self):
        text = ">seq1 description\nACGTACGT\nTTTT"
        assert _parse_sequence_text(text) == "ACGTACGTTTTT"

    def test_multi_record_concatenated(self):
        text = ">a\nAAAA\n>b\nCCCC"
        assert _parse_sequence_text(text) == "AAAACCCC"


class TestFastq:
    def test_quality_line_not_included(self):
        # Quality line starts with 'A' — must NOT leak into the sequence.
        text = "@read1\nACGTACGT\n+\nAAAFFFFF"
        assert _parse_sequence_text(text) == "ACGTACGT"

    def test_multi_record(self):
        text = "@r1\nACGT\n+\nFFFF\n@r2\nTTTT\n+\nFFFF"
        assert _parse_sequence_text(text) == "ACGTTTTT"

    def test_quality_with_symbols(self):
        text = "@r1\nGATTACA\n+\nI$%&*+I"
        assert _parse_sequence_text(text) == "GATTACA"

    def test_not_confused_with_plain_at(self):
        # A lone '@' line without the + separator is not FASTQ.
        text = "@something\nACGT"
        result = _parse_sequence_text(text)
        assert "ACGT" in result
