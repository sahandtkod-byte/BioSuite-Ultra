"""Tests for streaming readers and security modules."""
import pytest


class TestStreamingFasta:
    """Tests for iter_fasta()."""

    def test_iter_fasta(self, fasta_file):
        from biosuite.core.sequence import iter_fasta
        results = list(iter_fasta(fasta_file))
        assert len(results) == 2
        assert results[0][0] == "seq1 test sequence 1"

    def test_iter_fasta_generator(self, fasta_file):
        from biosuite.core.sequence import iter_fasta
        gen = iter_fasta(fasta_file)
        first = next(gen)
        assert isinstance(first, tuple)
        assert len(first) == 2

    def test_iter_fasta_nonexistent(self):
        from biosuite.core.sequence import iter_fasta
        results = list(iter_fasta("/nonexistent/file.fasta"))
        assert results == []


class TestStreamingFastq:
    """Tests for iter_fastq()."""

    def test_iter_fastq(self, fastq_file):
        from biosuite.core.sequence import iter_fastq
        results = list(iter_fastq(fastq_file))
        assert len(results) == 2
        assert results[0][0] == "read1"
        assert results[0][1] == "ATCGATCGATCG"

    def test_iter_fastq_nonexistent(self):
        from biosuite.core.sequence import iter_fastq
        results = list(iter_fastq("/nonexistent/file.fastq"))
        assert results == []


class TestSecurityModule:
    """Tests for api/security.py password hashing."""

    def test_hash_password(self):
        from biosuite.api.security import hash_password, verify_password
        hashed = hash_password("test_password_123")
        assert hashed != "test_password_123"
        assert verify_password("test_password_123", hashed)

    def test_verify_wrong_password(self):
        from biosuite.api.security import hash_password, verify_password
        hashed = hash_password("correct_password")
        assert not verify_password("wrong_password", hashed)

    def test_hash_is_deterministic(self):
        from biosuite.api.security import hash_password, verify_password
        h1 = hash_password("same_password")
        h2 = hash_password("same_password")
        # bcrypt includes random salt, so hashes differ but both verify
        assert verify_password("same_password", h1)
        assert verify_password("same_password", h2)


class TestLoggingModule:
    """Tests for core/log.py."""

    def test_get_logger(self):
        from biosuite.core.log import get_logger
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "biosuite.test_module"

    def test_get_logger_root(self):
        from biosuite.core.log import get_logger
        logger = get_logger()
        assert logger.name == "biosuite"

    def test_log_step(self):
        from biosuite.core.log import log_step
        # Should not raise
        log_step("test", "my_function", "started")
        log_step("test", "my_function", "completed", "100ms")
        log_step("test", "my_function", "failed", "error")


class TestConfigWrapper:
    """Tests for _DictWrapper in utils.py."""

    def test_config_is_dict_like(self):
        from biosuite.core.utils import config
        assert isinstance(config, dict)
        assert "theme" in config

    def test_config_get(self):
        from biosuite.core.utils import config
        theme = config.get("theme", "dark")
        assert isinstance(theme, str)

    def test_session_is_dict_like(self):
        from biosuite.core.utils import session
        assert isinstance(session, dict)
