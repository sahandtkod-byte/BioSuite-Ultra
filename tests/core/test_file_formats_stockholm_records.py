"""parse_stockholm: one alignment block, its metadata, and what is not returned.

Stockholm files carry the alignment and its annotation in the same text, and they
can hold several alignments one after another, so a reader that stops at the first
has to say that the rest of the file went unread.
"""
import pytest

from biosuite.core import file_formats as ff


def _write(tmp_path, text, name="aln.sto"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_the_version_line_is_not_data_and_the_sequences_are_the_alignment(tmp_path):
    out = ff.parse_stockholm(_write(tmp_path, "# STOCKHOLM 1.0\n"
                                              "seq1 ACGTACGT\nseq2 AAGGTTGG\n//\n"))
    assert out["alignment"] == {"seq1": "ACGTACGT", "seq2": "AAGGTTGG"}
    assert out["metadata"] == {}


def test_global_metadata_is_taken_from_the_number_gf_lines(tmp_path):
    out = ff.parse_stockholm(_write(tmp_path, "# STOCKHOLM 1.0\n#=GF ID   PfX\n"
                                              "#=GF Author Foo Bar\n#=GF\nseq1 AC\n"))
    assert out["metadata"] == {"ID": "PfX", "Author": "Foo Bar"}


def test_a_gf_line_without_a_value_stores_nothing_under_an_empty_key(tmp_path):
    out = ff.parse_stockholm(_write(tmp_path, "#=GF ID\n#=GF   \nseq1 AC\n"))
    assert out["metadata"] == {}


def test_per_sequence_and_per_column_records_are_not_part_of_the_result(tmp_path):
    """The result carries global metadata only, which is what the docstring now says.

    '#=GS' and '#=GC' lines are read past rather than parsed, so a caller that
    needed a sequence's database cross-references would find the metadata dict
    holding only the #=GF entries.
    """
    out = ff.parse_stockholm(_write(tmp_path, "# STOCKHOLM 1.0\n#=GF ID   PfX\n"
                                              "#=GS seq1 OR   UniProt:Q9027\n"
                                              "#=GC SS_cons  <<<<>>>>\nseq1 AC GT\n"))
    assert out["metadata"] == {"ID": "PfX"}
    assert out["alignment"] == {"seq1": "AC GT"}


def test_a_sequence_split_across_blocks_is_joined_in_the_order_it_was_written(tmp_path):
    out = ff.parse_stockholm(_write(tmp_path, "# STOCKHOLM 1.0\n"
                                              "seq1 ACGT\nseq2 ACG-\n\nseq1 TTGG\nseq2 -TGG\n//\n"))
    assert out["alignment"] == {"seq1": "ACGTTTGG", "seq2": "ACG--TGG"}


def test_a_file_without_a_terminator_returns_every_block_it_holds(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        out = ff.parse_stockholm(_write(tmp_path, "seq1 ACGT\nseq2 ACG-\nseq1 TTAA\n"))
    assert out["alignment"] == {"seq1": "ACGTTTAA", "seq2": "ACG-"}
    assert caplog.text == ""


def test_a_second_alignment_block_is_reported_instead_of_quietly_dropped(tmp_path, caplog):
    """The reader stops at the first '//' — the file in front of it does not.

    A multi-alignment Stockholm file used to come back as its first alignment with
    no signal at all, so a count of sequences was a count of one block while the
    user read it as the file.
    """
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        out = ff.parse_stockholm(_write(tmp_path, "# STOCKHOLM 1.0\n#=GF ID first\n"
                                                  "seq1 ACGT\n//\n"
                                                  "# STOCKHOLM 1.0\n#=GF ID second\n"
                                                  "seq9 TTTT\n//\n"))
    assert out["alignment"] == {"seq1": "ACGT"}
    assert out["metadata"] == {"ID": "first"}
    assert "belong to a second alignment block and were not read" in caplog.text, caplog.text
    assert "1 Stockholm line(s)" in caplog.text, caplog.text


def test_trailing_comments_after_the_terminator_are_not_reported_as_lost_data(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="biosuite.core.file_formats"):
        ff.parse_stockholm(_write(tmp_path, "# STOCKHOLM 1.0\nseq1 AC\n//\n"
                                            "# a trailing comment\n\n"))
    assert caplog.text == ""


def test_a_line_that_holds_only_a_name_contributes_no_sequence(tmp_path):
    out = ff.parse_stockholm(_write(tmp_path, "# STOCKHOLM 1.0\nSEQNAME\nseq1 ACGT\n//\n"))
    assert out["alignment"] == {"seq1": "ACGT"}


def test_a_crlf_file_does_not_keep_the_carriage_return_in_the_sequence(tmp_path):
    path = tmp_path / "crlf.sto"
    path.write_bytes(b"# STOCKHOLM 1.0\r\nseq1   AAAA\r\n//\r\n")
    assert ff.parse_stockholm(str(path))["alignment"] == {"seq1": "AAAA"}


def test_the_result_always_offers_both_keys_even_for_an_empty_file(tmp_path):
    assert ff.parse_stockholm(_write(tmp_path, "# STOCKHOLM 1.0\n//\n")) == {
        "alignment": {}, "metadata": {}}


def test_a_missing_stockholm_file_raises_naming_the_path(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        ff.parse_stockholm(str(tmp_path / "gone.sto"))
    assert "gone.sto" in str(exc.value)


def test_the_stockholm_summary_counts_the_sequences_in_the_alignment(tmp_path):
    result = {"format": "stockholm", "data": ff.parse_stockholm(
        _write(tmp_path, "# STOCKHOLM 1.0\nseq1 AC\nseq2 AC\n//\n"))}
    assert ff.format_file_summary(result) == "Stockholm: 2 sequences"
