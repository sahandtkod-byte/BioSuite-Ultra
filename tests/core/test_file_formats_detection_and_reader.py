"""detect_file_format and read_file: naming a format, then being able to open it.

The detector answers for the universal reader, the CLI and the API, so a format it
lists has to be one a caller can actually get data out of — or the error has to say
which half of the job is missing.
"""
import pandas as pd
import pytest

from biosuite.core import file_formats as ff


@pytest.mark.parametrize("name,expected", [
    ("x.fasta", "fasta"), ("x.fa", "fasta"), ("x.fna", "fasta"), ("x.fas", "fasta"),
    ("x.fastq", "fastq"), ("x.fq", "fastq"),
    ("x.gb", "genbank"), ("x.genbank", "genbank"), ("x.gbk", "genbank"),
    ("x.bam", "bam"), ("x.sam", "sam"), ("x.cram", "cram"),
    ("x.vcf", "vcf"), ("x.bcf", "bcf"),
    ("x.bed", "bed"), ("x.gff", "gff"), ("x.gff3", "gff"), ("x.gtf", "gtf"),
    ("x.newick", "newick"), ("x.nwk", "newick"), ("x.tree", "newick"), ("x.nhx", "newick"),
    ("x.stockholm", "stockholm"), ("x.sto", "stockholm"),
    ("x.bigwig", "bigwig"), ("x.bw", "bigwig"),
    ("x.bam.bai", "bai"), ("x.vcf.gz.tbi", "tbi"), ("x.saf", "saf"),
    ("archive.tar", "unknown"), ("noextension", "unknown"),
])
def test_every_advertised_extension_is_named(name, expected):
    assert ff.detect_file_format(name) == expected


def test_the_lookup_ignores_case_because_windows_paths_do_not_preserve_it():
    assert ff.detect_file_format("READS.FASTQ") == "fastq"
    assert ff.detect_file_format("reads.Cram") == "cram"
    assert ff.detect_file_format("A.BED") == "bed"


@pytest.mark.parametrize("name,expected", [
    ("genome.fasta.gz", "fasta"), ("genome.fa.gz", "fasta"), ("genome.fa.bz2", "fasta"),
    ("reads.fastq.gz", "fastq"), ("reads.fq.bz2", "fastq"),
    ("data.vcf.gz", "vcf"), ("regions.bed.gz", "bed"), ("ann.gff.gz", "gff"),
    ("genes.gtf.bz2", "gtf"), ("tree.nwk.gz", "newick"), ("genes.saf.gz", "saf"),
    ("x.txt.gz", "unknown"), ("x.gz", "unknown"),
])
def test_a_trailing_compression_suffix_is_looked_through(name, expected):
    """parse_bed and friends open .gz themselves, so a gzipped BED is a BED.

    The special cases this replaces knew about fasta, fastq and vcf only, so the
    same file two extensions over came back 'unknown' and the reader refused it.
    """
    assert ff.detect_file_format(name) == expected


def test_a_path_whose_whole_name_is_the_extension_is_still_a_cram():
    assert ff.detect_file_format(".cram") == "cram"


def test_a_directory_component_is_not_mistaken_for_an_extension():
    assert ff.detect_file_format("/data/bed/x.txt") == "unknown"
    assert ff.detect_file_format("/data/bigwig.x/file.txt") == "unknown"


def test_a_missing_sam_answers_with_the_detected_format_and_a_named_error(tmp_path):
    """Detection and reading are wired: the error names the file, not the format.

    The answer used to be 'was detected as sam but this build has no reader', so
    a caller could not tell an unsupported format from a mistyped path.
    """
    missing = str(tmp_path / "alignments.sam")
    out = ff.read_file(missing)
    assert out["format"] == "sam"
    assert f"{missing} was not read as SAM" in out["error"]
    assert "the file is missing or could not be parsed" in out["error"]


@pytest.mark.parametrize("suffix", [".bcf", ".nwk", ".bw"])
def test_the_other_wired_formats_answer_with_their_format_and_a_named_error(suffix):
    """Every detected format now has a reader; a missing file is a named error."""
    out = ff.read_file(f"/data/sample{suffix}")
    assert out["format"] == ff.detect_file_format(f"/data/sample{suffix}")
    payload = out.get("data") if isinstance(out.get("data"), dict) else out
    assert "error" in payload


def test_a_bed_is_read_through_the_universal_reader(tmp_path):
    path = tmp_path / "a.bed"
    path.write_text("chr1\t1\t2\tg\n")
    out = ff.read_file(str(path))
    assert out["format"] == "bed" and len(out["records"]) == 1
    assert out["records"] == ff.parse_bed(str(path))


def test_a_gff3_is_read_by_the_gff_parser_and_a_gtf_by_the_gtf_parser(tmp_path):
    body = "chr1\ts\tgene\t1\t9\t.\t+\t.\tID=g\n"
    gff = tmp_path / "a.gff3"
    gff.write_text(body)
    assert ff.read_file(str(gff))["format"] == "gff"
    gtf = tmp_path / "a.gtf"
    gtf.write_text(body)
    assert ff.read_file(str(gtf))["format"] == "gtf"


def test_a_fasta_and_a_fastq_are_read_through_the_universal_reader(tmp_path):
    fa = tmp_path / "a.fa"
    fa.write_text(">id\nACGT\n")
    assert ff.read_file(str(fa)) == {"format": "fasta", "records": [("id", "ACGT")]}
    fq = tmp_path / "a.fq"
    fq.write_text("@r1\nACGT\n+\nIIII\n")
    assert ff.read_file(str(fq))["format"] == "fastq"
    assert ff.read_file(str(fq))["records"] == [("r1", "ACGT", "IIII")]


def test_a_vcf_comes_back_as_a_dataframe_of_variants(tmp_path):
    vcf = tmp_path / "a.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                   "chr1\t100\t.\tA\tT\t.\t.\t.\n")
    out = ff.read_file(str(vcf))
    assert out["format"] == "vcf" and isinstance(out["data"], pd.DataFrame)
    assert len(out["data"]) == 1


@pytest.mark.parametrize("name,fmt", [
    ("gone.fa", "fasta"), ("gone.fq", "fastq"), ("gone.vcf", "vcf"),
])
def test_a_file_the_extension_promises_but_does_not_exist_says_it_was_not_read(
        tmp_path, name, fmt):
    """read_fasta, read_fastq and read_vcf answer None for a missing file.

    Passing that through left the summary calling a mistyped path "FASTA: 0
    sequences" — the same words an empty but real file produces.
    """
    out = ff.read_file(str(tmp_path / name))
    assert out["format"] == fmt
    assert "was not read as" in out["error"]
    assert "the file is missing or could not be parsed" in out["error"], out
    assert ff.format_file_summary(out) == (
        f"Error reading {fmt} file: {str(tmp_path / name)} was not read as "
        f"{fmt.upper()}: the file is missing or could not be parsed")


def test_an_empty_fasta_is_a_file_that_was_read_and_holds_nothing(tmp_path):
    fa = tmp_path / "empty.fa"
    fa.write_text("")
    assert ff.read_file(str(fa)) == {"format": "fasta", "records": []}


@pytest.mark.parametrize("fmt,options", [
    ("bam", "region"), ("cram", "region, reference"),
    ("bigwig", "chrom, start, end"),
])
def test_the_option_table_lists_exactly_what_each_reader_accepts(fmt, options):
    assert ", ".join(ff.READER_OPTIONS[fmt]) == options


def test_an_option_a_reader_does_not_take_is_refused_by_name(tmp_path):
    path = tmp_path / "a.bed"
    path.write_text("chr1\t1\t2\tg\n")
    out = ff.read_file(str(path), chrom="chr1")
    assert out["format"] == "bed"
    assert "The bed reader does not take chrom (it accepts: no options)" in out["error"]
    assert len(ff.parse_bed(str(path))) == 1, "the file itself was never the problem"


def test_an_option_meant_for_a_different_format_is_refused_not_guessed():
    out = ff.read_file("/data/x.bam", reference="ref.fa")
    assert out["format"] == "bam"
    assert "The bam reader does not take reference (it accepts: region)" in out["error"]


def test_a_region_is_carried_through_to_the_alignment_reader(tmp_path, monkeypatch):
    seen = {}

    def fake_bam(path, region=None):
        seen["region"] = region
        return {"format": "bam", "total_reads": 0, "mapped_reads": 0}

    monkeypatch.setattr(ff, "read_bam", fake_bam)
    assert ff.read_file("/data/x.bam", region="chr1:1-100")["format"] == "bam"
    assert seen == {"region": "chr1:1-100"}


def test_a_reader_that_raises_becomes_an_error_dict_that_names_the_format(
        tmp_path, monkeypatch, caplog):
    path = tmp_path / "a.bed"
    path.write_text("chr1\t1\t2\tg\n")
    monkeypatch.setattr(ff, "parse_bed",
                        lambda p: (_ for _ in ()).throw(RuntimeError("boom in the parser")))
    with caplog.at_level("ERROR", logger="biosuite.core.file_formats"):
        assert ff.read_file(str(path)) == {"format": "bed", "error": "boom in the parser"}
    assert "The bed reader failed" in caplog.text, caplog.text


def test_a_missing_bed_file_comes_back_as_an_error_naming_the_path(tmp_path):
    """parse_bed raises where read_fasta returns None, and both end up an error.

    The two reader families fail differently — one raises, the other answers None —
    so the universal reader is the place that turns either into one shape.
    """
    missing = str(tmp_path / "gone.bed")
    out = ff.read_file(missing)
    assert out["format"] == "bed" and "No such file or directory" in out["error"]
    assert ff.format_file_summary(out).startswith("Error reading bed file: ")


def test_the_stockholm_result_travels_under_the_data_key(tmp_path):
    path = tmp_path / "a.sto"
    path.write_text("# STOCKHOLM 1.0\nseq1 AC\n//\n")
    out = ff.read_file(str(path))
    assert out == {"format": "stockholm", "data": {"alignment": {"seq1": "AC"},
                                                   "metadata": {}}}
