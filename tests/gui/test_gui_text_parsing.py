"""The pasted-text sequence parser that every sequence handler reads the box through.

FASTA loses its header, FASTQ loses header + separator + quality, raw lines are
joined as they arrived; the quality line is the trap, because legal Phred starts
with A/C/G/T and looks exactly like sequence.  Each expectation is spelled out
from the input's own lines, so any change to the record walk breaks one of them.
"""
from biosuite.gui.text_parsing import _parse_sequence_text

SEQ = "ACGTACGT"
QUALITY = "AAAFFFFF"


def test_an_empty_box_yields_no_sequence():
    assert _parse_sequence_text("") == ""
    assert _parse_sequence_text("   \n  ") == ""
    assert _parse_sequence_text(SEQ) == SEQ, "the same call on real text is not blank"


def test_raw_lines_are_joined_and_their_spaces_dropped():
    assert _parse_sequence_text("ACG TAC GT\nTT TT") == "ACGTACGTTTTT"


def test_tabs_inside_a_pasted_sequence_are_dropped_like_spaces():
    assert _parse_sequence_text("AC\tGT\nTT\tTT") == "ACGTTTTT"


def test_tabs_inside_a_fastq_sequence_are_dropped():
    assert _parse_sequence_text("@r1\nAC\tGT\n+\nIIII") == "ACGT"


def test_a_tab_and_a_space_in_one_line_are_both_dropped():
    assert _parse_sequence_text("AC \tGT\nTT") == "ACGTTT"


def test_blank_lines_inside_a_paste_contribute_nothing():
    assert _parse_sequence_text("ACGT\n\nTTTT") == "ACGTTTTT"


def test_the_paste_is_trimmed_before_it_is_walked():
    assert _parse_sequence_text("\n  ACGT  \n\n") == "ACGT"


def test_a_fasta_header_is_dropped_and_every_sequence_line_kept():
    assert _parse_sequence_text(">seq1 description\nACGTACGT\nTTTT") == "ACGTACGTTTTT"


def test_each_fasta_record_in_one_paste_contributes_its_sequence():
    assert _parse_sequence_text(">a\nAAAA\n>b\nCCCC") == "AAAACCCC"


def test_a_fasta_record_that_is_only_a_header_yields_nothing():
    assert _parse_sequence_text(">empty record") == ""


def test_a_fastq_quality_line_never_reaches_the_sequence():
    text = "\n".join(["@read1", SEQ, "+", QUALITY])
    assert _parse_sequence_text(text) == SEQ


def test_consecutive_fastq_records_are_walked_four_lines_at_a_time():
    first, second = "ACGT", "TTTT"
    text = "\n".join(["@r1", first, "+", "IIIII", "@r2", second, "+", "!!!!!"])
    assert _parse_sequence_text(text) == first + second


def test_the_fastq_separator_is_still_found_when_it_is_indented():
    assert _parse_sequence_text("@r1\nACGT\n +\nIIII") == "ACGT"


def test_a_fastq_record_cut_off_after_the_separator_still_yields_its_sequence():
    assert _parse_sequence_text("@r1\nACGT\n+") == "ACGT"


def test_a_stray_separator_line_is_dropped_and_the_flanks_are_kept():
    assert _parse_sequence_text("ACGT\n+\nTTTT") == "ACGTTTTT"


def test_an_at_line_without_a_separator_is_not_consumed_as_a_record():
    """Without the '+' nothing is skipped, so the lines after the '@' all survive —
    whereas the very same last line is dropped once the separator is there."""
    kept = _parse_sequence_text("@r1\nACGT\nTTTT")
    assert "ACGT" in kept and "TTTT" in kept
    assert _parse_sequence_text("@r1\nACGT\n+\nTTTT") == "ACGT"


def test_fasta_and_raw_lines_arrive_in_the_order_they_were_pasted():
    assert _parse_sequence_text(">h\nACGT\nTTTT\nGGGG") == "ACGTTTTTGGGG"


def test_a_fastq_record_following_a_fasta_record_does_not_lose_its_sequence():
    text = ">h\nACGT\n@r1\nTTTT\n+\nIIII"
    assert _parse_sequence_text(text) == "ACGTTTTT"
