"""format_file_summary: the sentence the CLI, the GUI and the API hand to a user.

Its job is to say what happened to the file, so a read that failed inside a nested
'data' dict has to come out as a failure — the shape the engine readers actually
return, and the one this function used to walk past.
"""
import pandas as pd
import pytest

from biosuite.core import file_formats as ff


def test_a_bed_result_is_the_bed_summary_text():
    recs = [ff.BEDRecord(chrom="chr1", start=1, end=2),
            ff.BEDRecord(chrom="chr2", start=5, end=9)]
    assert ff.format_file_summary({"format": "bed", "records": recs}) == (
        "=== BED File Summary ===\nRecords: 2\nChromosomes: 2\nTotal bp: 5")


def test_a_gff_or_gtf_result_counts_the_feature_types():
    recs = [ff.GFFRecord(seqid="chr1", source="s", feature="gene", start=1, end=9,
                         score=0, strand="+", phase="."),
            ff.GFFRecord(seqid="chr1", source="s", feature="exon", start=1, end=3,
                         score=0, strand="+", phase=".")]
    for fmt in ("gff", "gtf"):
        assert ff.format_file_summary({"format": fmt, "records": recs}) == (
            "=== GFF Summary ===\nRecords: 2\nFeature types:\n  gene: 1\n  exon: 1")


def test_a_fasta_result_counts_sequences_and_bases():
    records = [("h1", "ACGT"), ("h2", "AA")]
    assert ff.format_file_summary({"format": "fasta", "records": records}) == (
        "FASTA: 2 sequences, 6 bp total")
    assert ff.format_file_summary({"format": "fasta", "records": []}) == "FASTA: 0 sequences"


def test_a_fastq_result_counts_reads_and_bases():
    records = [("r1", "ACGT", "IIII"), ("r2", "AC", "II")]
    assert ff.format_file_summary({"format": "fastq", "records": records}) == (
        "FASTQ: 2 reads, 6 bp total")
    assert ff.format_file_summary({"format": "fastq", "records": []}) == "FASTQ: 0 reads"


def test_a_genbank_result_counts_the_records_it_holds():
    """'File loaded: genbank format' said nothing while the record count sat unused.

    The list is already in the result, so the reader gets the number they asked for
    instead of a sentence that could follow any successful read of any format.
    """
    assert ff.format_file_summary({"format": "genbank", "records": ["a", "b", "c"]}) == (
        "GenBank: 3 records")
    assert ff.format_file_summary({"format": "genbank", "records": []}) == "GenBank: 0 records"


def test_a_bam_read_that_never_happened_is_not_reported_as_a_load():
    """The headline: an engine error nested in 'data' is still an error.

    read_bam, read_cram and both index readers answer with a dict whose error sits
    under data, and the summary walked straight past it to print "File loaded: bam
    format" over a file nothing had opened.
    """
    inner = {"error": "pysam required for BAM reading. pip install pysam", "format": "bam",
             "note": "Install pysam for full BAM support"}
    assert ff.format_file_summary({"format": "bam", "data": inner}) == (
        "Error reading bam file: pysam required for BAM reading. pip install pysam")


def test_a_missing_cram_index_is_reported_from_its_nested_error():
    inner = {"error": "Index file not found: /tmp/x.cram"}
    assert ff.format_file_summary({"format": "cram", "data": inner}) == (
        "Error reading cram file: Index file not found: /tmp/x.cram")


@pytest.mark.parametrize("fmt,inner,expected", [
    ("bam", {"total_reads": 12345, "mapped_reads": 6789}, "BAM: 12,345 reads (6,789 mapped)"),
    ("cram", {"total_reads": 0, "mapped_reads": 0}, "CRAM: 0 reads (0 mapped)"),
    ("cram", {"mapped_reads": 1}, "File loaded: cram format"),
    ("bam", {}, "File loaded: bam format"),
])
def test_an_alignment_read_with_counts_reports_them_and_without_them_says_so(
        fmt, inner, expected):
    assert ff.format_file_summary({"format": fmt, "data": inner}) == expected


def test_a_bai_result_states_whether_the_index_is_one():
    valid = {"format": "bai", "path": "/x.bam.bai", "file_size": 6, "valid": True,
             "note": "BAM index recognised. Use pysam for detailed queries."}
    assert ff.format_file_summary({"format": "bai", "data": valid}) == (
        "BAI index: valid (6 bytes)")
    invalid = {"format": "bai", "path": "/x.bai", "file_size": 4, "valid": False,
               "note": "Not a BAM index: expected the magic b'BAM\\x01'"}
    assert ff.format_file_summary({"format": "bai", "data": invalid}) == (
        "Not a BAM index: expected the magic b'BAM\\x01'")
    assert "BAI index: Not a BAM index" not in ff.format_file_summary(
        {"format": "bai", "data": invalid})


def test_an_index_result_that_earned_no_note_still_says_what_it_was():
    assert ff.format_file_summary({"format": "tbi", "data": {"valid": False}}) == (
        "TBI index: not a valid tbi file")


def test_a_tbi_result_carries_the_contig_count_it_found():
    data = {"format": "tbi", "path": "/x.vcf.gz.tbi", "file_size": 4, "valid": True,
            "chromosomes": ["chr1", "chr2"], "n_chromosomes": 2}
    assert ff.format_file_summary({"format": "tbi", "data": data}) == (
        "TBI index: valid (4 bytes, 2 contigs)")


def test_a_vcf_result_counts_variants_including_a_file_with_none():
    frame = pd.DataFrame({"CHROM": ["chr1", "chr2"], "POS": [1, 2]})
    assert ff.format_file_summary({"format": "vcf", "data": frame}) == "VCF: 2 variants"
    assert ff.format_file_summary({"format": "vcf", "data": pd.DataFrame()}) == (
        "VCF: 0 variants")


def test_the_stockholm_count_comes_from_the_alignment_it_read():
    assert ff.format_file_summary({"format": "stockholm",
                                   "data": {"alignment": {"s1": "AC", "s2": "AC"},
                                            "metadata": {}}}) == "Stockholm: 2 sequences"


def test_a_saf_result_counts_the_annotated_features():
    assert ff.format_file_summary({"format": "saf", "records": [{"gene_id": "g1"}]}) == (
        "SAF: 1 annotated features")


def test_a_top_level_error_is_reported_with_the_format_it_carries():
    assert ff.format_file_summary({"format": "tbi",
                                   "error": "Index file not found: /tmp/x.tbi"}) == (
        "Error reading tbi file: Index file not found: /tmp/x.tbi")
    assert ff.format_file_summary({"error": "no format key here"}) == (
        "Error reading unknown file: no format key here")


def test_a_missing_sam_file_is_an_error_not_a_load():
    out = ff.read_file("/data/alignments.sam")
    assert ff.format_file_summary(out) == (
        "Error reading sam file: /data/alignments.sam was not read as SAM: the "
        "file is missing or could not be parsed")


def test_a_result_this_function_has_nothing_to_count_still_says_what_it_was():
    assert ff.format_file_summary({"format": "bcf"}) == "File loaded: bcf format"
    assert ff.format_file_summary({}) == "File loaded: unknown format"


def test_a_vcf_result_without_any_data_falls_back_to_the_generic_line():
    """read_file no longer produces this shape: a None from read_vcf is now an error.

    Called with a result that carries neither an error nor a frame, the summary has
    nothing to count, and the generic line is the honest answer about what it was
    handed rather than an invented variant count.
    """
    assert ff.format_file_summary({"format": "vcf", "data": None}) == (
        "File loaded: vcf format")
    assert "variants" not in ff.format_file_summary({"format": "vcf", "data": None})
