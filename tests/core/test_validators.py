"""Tests for core/validators.py decorators + InputValidator."""
import os
import tempfile

import pandas as pd
import pytest

from biosuite.core import validators as v


# ── sequence validation ──────────────────────────────────────────────────────

def test_validate_sequence_ok_and_bad():
    @v.validate_sequence(min_length=2)
    def f(seq):
        return seq.upper()

    assert f("ACGT") == "ACGT"
    with pytest.raises(ValueError):
        f("A")
    with pytest.raises(ValueError):
        f("AXGC")          # X not in ATCGN
    # non-string positional args bypass validation by design (no str arg
    # to check); the wrapped function receives the raw value and raises
    # its own error — observed behavior, documented here
    with pytest.raises(AttributeError):
        f(123)



def test_validate_sequence_custom_chars():
    @v.validate_sequence(allowed_chars='AT', min_length=1)
    def g(seq):
        return len(seq)

    assert g('ATTA') == 4
    with pytest.raises(ValueError):
        g('ATCG')


def test_validate_sequence_kwarg_targeting():
    @v.validate_sequence(min_length=2)
    def h(sequence=None):
        return sequence
    with pytest.raises(ValueError):
        h(sequence='A')


# ── file extension validation ────────────────────────────────────────────────

def test_validate_file_extension():
    @v.validate_file_extension(['.fa', '.fasta'])
    def load(path):
        return path

    with tempfile.NamedTemporaryFile(suffix='.fa', delete=False) as fh:
        assert load(fh.name) == fh.name
    with pytest.raises(ValueError):
        load('/tmp/genome.txt')


# ── dataframe columns ───────────────────────────────────────────────────────

def test_validate_dataframe_columns():
    @v.validate_dataframe_columns(['a', 'b'])
    def agg(df):
        return df.shape

    df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    assert agg(df) == (1, 3)
    with pytest.raises((ValueError, KeyError)):
        agg(pd.DataFrame({'a': [1]}))


# ── range validation ────────────────────────────────────────────────────────

def test_validate_range():
    @v.validate_range(min_val=0.0, max_val=1.0)
    def pr(x):
        return x

    assert pr(0.5) == 0.5
    with pytest.raises(ValueError):
        pr(1.5)
    with pytest.raises(ValueError):
        pr(-0.1)


# ── retry_on_error / safe_execute ───────────────────────────────────────────

def test_retry_eventually_succeeds():
    calls = {'n': 0}

    @v.retry_on_error(max_retries=5, delay=0.001)
    def flaky():
        calls['n'] += 1
        if calls['n'] < 3:
            raise ConnectionError("transient")
        return 'ok'

    assert flaky() == 'ok'
    assert calls['n'] == 3


def test_retry_raises_after_exhaustion():
    @v.retry_on_error(max_retries=2, delay=0.001)
    def always():
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError):
        always()


def test_safe_execute_success_and_failure():
    result, err = v.safe_execute(lambda x: x * 2, 21)
    assert result == 42 and err is None
    result, err = v.safe_execute(lambda: 1 / 0, _default=-1)
    assert result == -1 and isinstance(err, Exception)


# ── InputValidator class ────────────────────────────────────────────────────

def test_input_validator_sequences():
    iv = v.InputValidator()
    assert iv.validate_sequence('ACGTACGT') == []
    errs = iv.validate_sequence('ATXCG')
    assert errs and 'Invalid DNA' in errs[0]
    assert iv.validate_sequence('MKVKGTLS', seq_type='amino_acid') == []
    assert iv.validate_sequence('', seq_type='dna') != []


def test_input_validator_fasta_fastq_vcf():
    iv = v.InputValidator()
    assert iv.validate_fasta('>s1\nACGT\n>s2\nTTGA\n') == []
    assert iv.validate_fasta('ACGT\n') != []
    good_fq = '@r1\nACGT\n+\nFFFF\n'
    assert iv.validate_fastq(good_fq) == []
    assert iv.validate_fastq('engel\n') != []
    vcf = ('##fileformat=VCFv4.2\n#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO\n'
           'chr1\t100\trs1\tA\tG\t50\tPASS\t.\n')
    assert iv.validate_vcf(vcf) == []
    assert iv.validate_vcf('garbage\n') != []



