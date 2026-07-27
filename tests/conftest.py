"""Shared pytest fixtures for BioSuite Ultra tests."""
import os
import tempfile
import pytest


# ── Sequence Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_dna():
    """Standard DNA sequence for testing."""
    return "ATCGATCGATCGATCGATCG"


@pytest.fixture
def sample_dna_long():
    """Longer DNA sequence (1000 bp) for performance tests."""
    import random
    random.seed(42)
    return ''.join(random.choices('ACGT', k=1000))


@pytest.fixture
def sample_protein():
    """Standard protein sequence for testing."""
    return "MKTAYIAKQRQISFVKSHFSRQISFVKSHFSR"


@pytest.fixture
def sample_rna():
    """Standard RNA sequence for testing."""
    return "AUCGAUCGAUCGAUCGAUCG"


@pytest.fixture
def fasta_content():
    """FASTA format string with multiple sequences."""
    return """>seq1 test sequence 1
ATCGATCGATCGATCG
ATCGATCGATCGATCG
>seq2 test sequence 2
GCTAGCTAGCTAGCTA
"""


@pytest.fixture
def fasta_file(tmp_path, fasta_content):
    """Write a temporary FASTA file and return its path."""
    p = tmp_path / "test.fasta"
    p.write_text(fasta_content)
    return str(p)


@pytest.fixture
def fastq_content():
    """FASTQ format string."""
    return """@read1
ATCGATCGATCG
+
IIIIIIIIIIII
@read2
GCTAGCTAGCTA
+
IIIIIIIIIIII
"""


@pytest.fixture
def fastq_file(tmp_path, fastq_content):
    """Write a temporary FASTQ file and return its path."""
    p = tmp_path / "test.fastq"
    p.write_text(fastq_content)
    return str(p)


# ── Alignment Fixtures ────────────────────────────────────────────────

@pytest.fixture
def alignment_pair():
    """Pair of sequences for pairwise alignment."""
    return "ACGTACGT", "ACGACGT"


@pytest.fixture
def msa_sequences():
    """Multiple sequences for MSA testing."""
    return {
        "seq1": "ACGTACGT",
        "seq2": "ACGACGT",
        "seq3": "ACGTTCGT",
    }


# ── Expression Fixtures ──────────────────────────────────────────────

@pytest.fixture
def expression_matrix():
    """Small expression matrix for testing."""
    import numpy as np
    np.random.seed(42)
    return np.random.rand(100, 6)


@pytest.fixture
def deg_results():
    """Differential expression results."""
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    return pd.DataFrame({
        "gene": [f"gene_{i}" for i in range(100)],
        "log2fc": np.random.normal(0, 1.5, 100),
        "pvalue": np.random.uniform(0, 1, 100),
        "padj": np.random.uniform(0, 1, 100),
    })


# ── Utility Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    """Provide a temporary directory, cleaned up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def biosuite_config():
    """Minimal BioSuite config dict for testing."""
    return {
        "theme": "dark-green",
        "default_dpi": 150,
        "save_format": "png",
        "interactive": False,
        "quiet": True,
    }
