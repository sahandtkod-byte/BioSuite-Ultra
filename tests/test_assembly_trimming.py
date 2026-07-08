"""
Tests for Assembly, Trimming, and Quantification modules.
"""
import pytest
import os
import tempfile


class TestTrimming:
    def test_trim_fastq(self, tmp_path):
        from biosuite.core.trimming import trim_fastq, format_trim_report
        # Create a minimal FASTQ file
        fastq = tmp_path / "test.fastq"
        fastq.write_text(
            "@read1\n"
            "ACGTACGTACGT\n"
            "+\n"
            "IIIIIIIIIIII\n"
            "@read2\n"
            "GGCCTTAAGGCCT\n"
            "+\n"
            "IIIIIIIIIIIII\n"
        )
        out = tmp_path / "trimmed.fastq"
        report = trim_fastq(str(fastq), str(out), quality_threshold=20, min_length=4)
        assert report.total_reads == 2
        assert os.path.exists(str(out))

    def test_format_report(self, tmp_path):
        from biosuite.core.trimming import trim_fastq, format_trim_report
        fastq = tmp_path / "test.fastq"
        fastq.write_text("@r1\nACGT\n+\nIIII\n")
        out = tmp_path / "out.fastq"
        report = trim_fastq(str(fastq), str(out))
        text = format_trim_report(report)
        assert isinstance(text, str)
        assert "reads" in text.lower() or "total" in text.lower()


class TestAssembly:
    def test_assemble_builtin(self, tmp_path):
        from biosuite.core.assembly import assemble
        fastq = tmp_path / "reads.fastq"
        fastq.write_text(
            "@r1\nACGTACGTACGTACGTACGTACGT\n+\nIIIIIIIIIIIIIIIIIIIIIIIIII\n"
            "@r2\nTGCATGCATGCATGCATGCATGCA\n+\nIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        )
        result = assemble(str(fastq))
        assert hasattr(result, 'num_contigs')
        assert result.num_contigs >= 1

    def test_format_report(self, tmp_path):
        from biosuite.core.assembly import assemble, format_assembly_report
        fastq = tmp_path / "reads.fastq"
        fastq.write_text("@r1\nACGTACGT\n+\nIIIIIIII\n")
        result = assemble(str(fastq))
        report = format_assembly_report(result)
        assert isinstance(report, str)


class TestQuantification:
    def test_quantify_reads(self, tmp_path):
        from biosuite.core.quantification import quantify_reads
        # Create reads
        reads = tmp_path / "reads.fastq"
        reads.write_text(
            "@r1\nACGTACGTACGT\n+\nIIIIIIIIIIII\n"
            "@r2\nTGCATGCATGCA\n+\nIIIIIIIIIIII\n"
        )
        # Create transcriptome
        trans = tmp_path / "transcripts.fasta"
        trans.write_text(">t1\nACGTACGTACGTACGTACGTACGT\n>t2\nTGCATGCATGCATGCATGCATGCA\n")
        result = quantify_reads(str(reads), str(trans))
        assert hasattr(result, 'num_transcripts')
        assert result.num_transcripts == 2
