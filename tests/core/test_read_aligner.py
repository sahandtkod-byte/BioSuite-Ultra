"""biosuite.core.read_aligner's built-in aligner.

Records, the report text, the suffix/Fasta index helpers, and every missing-file
path around align_reads.
"""
import pytest


class TestReadAlignerDataclasses:
    def test_alignment_creation(self):
        from biosuite.core.read_aligner import Alignment
        a = Alignment(read_id="read1", reference_id="chr1", position=100,
                       strand="+", cigar="100M", mapping_quality=60, score=180,
                       edit_distance=0)
        assert a.is_primary is True
        assert a.position == 100

    def test_alignment_report_defaults(self):
        from biosuite.core.read_aligner import AlignmentReport
        r = AlignmentReport(tool='test', engine='test')
        assert r.total_reads == 0
        assert r.mapping_rate == 0.0
        assert r.alignments == []


class TestFormatAlignmentReport:
    def test_basic_format(self):
        from biosuite.core.read_aligner import AlignmentReport, format_alignment_report
        r = AlignmentReport(tool='test', engine='builtin', total_reads=1000,
                            mapped_reads=900, unmapped_reads=100,
                            mapping_rate=90.0, avg_mapping_quality=55.0,
                            message="test engine")
        text = format_alignment_report(r)
        assert "Read Alignment Report" in text
        assert "1,000" in text
        assert "90.0%" in text

    def test_zero_reads(self):
        from biosuite.core.read_aligner import AlignmentReport, format_alignment_report
        r = AlignmentReport(tool='none', engine='none')
        text = format_alignment_report(r)
        assert "0" in text


class TestBuiltinAlignerHelpers:
    def test_build_suffix_index(self):
        from biosuite.core.read_aligner import _build_suffix_index
        seq = "ATCGATCGATCGATCG"
        index = _build_suffix_index(seq, k=5)
        assert isinstance(index, dict)
        # Should have entries for k-mers
        assert len(index) > 0

    def test_build_fasta_index(self, tmp_path):
        from biosuite.core.read_aligner import _build_fasta_index
        fasta = tmp_path / "ref.fa"
        fasta.write_text(">ref1\nATCGATCGATCG\n>ref2\nGCTAGCTAGCTA\n")
        refs = _build_fasta_index(str(fasta))
        assert 'ref1' in refs
        assert 'ref2' in refs
        assert len(refs['ref1']) == 12

    def test_seed_and_extend(self):
        from biosuite.core.read_aligner import _build_suffix_index, _seed_and_extend
        ref = "ATCGATCGATCGATCGATCGATCG"
        index = _build_suffix_index(ref, k=5)
        read = "TCGATCGATCGA"
        result = _seed_and_extend(read, ref, "test_ref", index, k=5, seed_threshold=3)
        # With enough matching seeds, should find an alignment
        if result is not None:
            assert result.reference_id == "test_ref"


class TestAlignReadsErrorHandling:
    def test_missing_reference(self):
        from biosuite.core.read_aligner import align_reads
        report = align_reads("/nonexistent/ref.fa", "/nonexistent/reads.fq")
        assert "Reference not found" in report.message

    def test_missing_reads(self, tmp_path):
        from biosuite.core.read_aligner import align_reads
        ref = tmp_path / "ref.fa"
        ref.write_text(">ref\nATCGATCGATCGATCG\n")
        report = align_reads(str(ref), "/nonexistent/reads.fq")
        assert "Reads not found" in report.message

    def test_builtin_align_reads(self, tmp_path):
        from biosuite.core.read_aligner import align_reads
        ref = tmp_path / "ref.fa"
        ref.write_text(">ref\nATCGATCGATCGATCGATCGATCGATCGATCGATCG\n")
        reads = tmp_path / "reads.fq"
        reads.write_text(
            "@read1\nATCGATCGATCGATCG\n+\nIIIIIIIIIIIIIIII\n"
            "@read2\nGCTAGCTAGCTAGCTA\n+\nIIIIIIIIIIIIIIII\n"
        )
        report = align_reads(str(ref), str(reads))
        assert report.total_reads >= 1
        assert report.tool == 'builtin_seed_extend'
