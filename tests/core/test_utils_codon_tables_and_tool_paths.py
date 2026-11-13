"""
Tests for shared utilities: GENETIC_CODE, has_tool, read_fasta_simple.
"""

import pytest


class TestGeneticCode:
    def test_has_all_64_codons(self):
        from biosuite.core.utils import GENETIC_CODE
        assert len(GENETIC_CODE) == 64

    def test_stop_codons(self):
        from biosuite.core.utils import GENETIC_CODE, STOP_CODONS
        assert GENETIC_CODE['TAA'] == '*'
        assert GENETIC_CODE['TAG'] == '*'
        assert GENETIC_CODE['TGA'] == '*'
        assert STOP_CODONS == {'TAA', 'TAG', 'TGA'}

    def test_start_codon(self):
        from biosuite.core.utils import GENETIC_CODE
        assert GENETIC_CODE['ATG'] == 'M'

    def test_all_values_single_char(self):
        from biosuite.core.utils import GENETIC_CODE
        for codon, aa in GENETIC_CODE.items():
            assert len(codon) == 3
            assert len(aa) == 1


class TestHasTool:
    def test_nonexistent_tool(self):
        from biosuite.core.utils import has_tool
        assert has_tool('nonexistent_tool_xyz_123') is False

    def test_python_is_available(self):
        from biosuite.core.utils import has_tool
        # pytest itself runs on python, so the interpreter is on PATH here.
        assert has_tool('python') is True


class TestReadFastaSimple:
    def test_single_sequence(self, tmp_path):
        from biosuite.core.utils import read_fasta_simple
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq1\nACGTACGT\n")
        result = read_fasta_simple(str(fasta))
        assert len(result) == 1
        assert result[0] == ("seq1", "ACGTACGT")

    def test_multiple_sequences(self, tmp_path):
        from biosuite.core.utils import read_fasta_simple
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq1\nACGT\n>seq2\nTGCA\n")
        result = read_fasta_simple(str(fasta))
        assert len(result) == 2
        assert result[0] == ("seq1", "ACGT")
        assert result[1] == ("seq2", "TGCA")

    def test_multiline_sequence(self, tmp_path):
        from biosuite.core.utils import read_fasta_simple
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq1\nACGT\nTGCA\nACGT\n")
        result = read_fasta_simple(str(fasta))
        assert result[0] == ("seq1", "ACGTTGCAACGT")

    def test_header_with_description(self, tmp_path):
        from biosuite.core.utils import read_fasta_simple
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq1 some description\nACGT\n")
        result = read_fasta_simple(str(fasta))
        assert result[0][0] == "seq1"  # Only first word kept

    def test_empty_file(self, tmp_path):
        from biosuite.core.utils import read_fasta_simple
        fasta = tmp_path / "test.fasta"
        fasta.write_text("")
        result = read_fasta_simple(str(fasta))
        assert result == []

    def test_file_not_found(self):
        from biosuite.core.utils import read_fasta_simple
        with pytest.raises(FileNotFoundError):
            read_fasta_simple("/nonexistent/file.fasta")


class TestAutosaveSession:
    def test_save_and_load(self, tmp_path, monkeypatch):
        """The session lands where `user_config_dir()` points, which the env
        override decides — the module constant this test used to set
        (`SESSION_FILE`) is read by nothing and was removed as a decoy."""
        from biosuite.core.utils import load_session, save_session
        monkeypatch.setenv("BIOSUITE_CONFIG_DIR", str(tmp_path))
        data = {"key": "value", "count": 42}

        save_session(data)

        assert (tmp_path / "biosuite_session.json").exists()
        assert load_session() == data
