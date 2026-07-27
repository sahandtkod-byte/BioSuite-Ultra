"""Input validation decorators, error recovery utilities, and bio-format validators.

This module provides:
- Decorators for validating sequences, file extensions, DataFrames, and numeric ranges
- Error recovery: retry with exponential backoff, safe execution wrappers
- InputValidator class for bioinformatics file formats (FASTA, FASTQ, VCF)
"""

import os
import re
import time
import functools
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ── Input Validation Decorators ─────────────────────────────────────────────


def validate_sequence(min_length: int = 1, allowed_chars: str = 'ATCGN'):
    """Decorator that validates a DNA/RNA sequence argument.

    Args:
        min_length: Minimum sequence length (inclusive).
        allowed_chars: Set of allowed characters (case-insensitive).

    Raises:
        ValueError: If the sequence is too short or contains invalid characters.

    Example::

        @validate_sequence(min_length=4, allowed_chars='ATCG')
        def reverse_complement(seq: str) -> str:
            comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
            return ''.join(comp[b] for b in reversed(seq))
    """
    allowed = set(allowed_chars.upper())

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to find the first positional sequence argument.
            # Convention: if the first positional arg is a string, treat it as the sequence.
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    _check_sequence(arg, min_length, allowed, arg_index=i)
                    break
            else:
                # Check keyword arguments — look for common names
                for key in ('sequence', 'seq', 'dna', 'rna', 'sequence_str'):
                    if key in kwargs and isinstance(kwargs[key], str):
                        _check_sequence(kwargs[key], min_length, allowed, kwarg=key)
                        break
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _check_sequence(seq: str, min_length: int, allowed: set, arg_index: int = 0, kwarg: str = None):
    """Internal helper to validate a sequence string."""
    name = f"argument '{kwarg}'" if kwarg else f"argument at index {arg_index}"
    if len(seq) < min_length:
        raise ValueError(
            f"Sequence in {name} is too short: {len(seq)} < {min_length} (min_length={min_length})"
        )
    invalid = set(seq.upper()) - allowed
    if invalid:
        raise ValueError(
            f"Sequence in {name} contains invalid characters: {invalid}. "
            f"Allowed characters: {allowed}"
        )


def validate_file_extension(exts: List[str] = None):
    """Decorator that validates a file path has an allowed extension.

    Args:
        exts: List of allowed extensions (e.g. ['.fasta', '.fa', '.fastq', '.fq']).
              Defaults to common bioinformatics formats.

    Raises:
        ValueError: If the file extension is not in the allowed list.

    Example::

        @validate_file_extension(['.fasta', '.fa', '.fastq'])
        def parse_file(filepath: str):
            ...
    """
    if exts is None:
        exts = ['.fasta', '.fa', '.fastq', '.fq']
    allowed_exts = {e.lower() for e in exts}

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i, arg in enumerate(args):
                if isinstance(arg, str) and ('.' in os.path.basename(arg) or '/' in arg or '\\' in arg):
                    _check_file_extension(arg, allowed_exts, arg_index=i)
                    break
            else:
                for key in ('filepath', 'path', 'file', 'filename', 'input_file'):
                    if key in kwargs and isinstance(kwargs[key], str):
                        _check_file_extension(kwargs[key], allowed_exts, kwarg=key)
                        break
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _check_file_extension(filepath: str, allowed_exts: set, arg_index: int = 0, kwarg: str = None):
    """Internal helper to validate file extension."""
    name = f"argument '{kwarg}'" if kwarg else f"argument at index {arg_index}"
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in allowed_exts:
        raise ValueError(
            f"File in {name} has unsupported extension '{ext}'. "
            f"Allowed extensions: {sorted(allowed_exts)}"
        )


def validate_dataframe_columns(required_cols: List[str] = None):
    """Decorator that validates a pandas DataFrame has required columns.

    Args:
        required_cols: Column names that must be present.
                       Defaults to ['gene'].

    Raises:
        ValueError: If required columns are missing from the DataFrame.

    Example::

        @validate_dataframe_columns(['gene', 'expression'])
        def normalize_expression(df):
            ...
    """
    if required_cols is None:
        required_cols = ['gene']
    required = set(required_cols)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import pandas as pd
            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame):
                    _check_dataframe_columns(arg, required, arg_index=i)
                    break
            else:
                for key in ('df', 'dataframe', 'data', 'table'):
                    if key in kwargs and isinstance(kwargs[key], pd.DataFrame):
                        _check_dataframe_columns(kwargs[key], required, kwarg=key)
                        break
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _check_dataframe_columns(df, required: set, arg_index: int = 0, kwarg: str = None):
    """Internal helper to validate DataFrame columns."""
    name = f"argument '{kwarg}'" if kwarg else f"argument at index {arg_index}"
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame in {name} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_range(min_val: float = None, max_val: float = None):
    """Decorator that validates a numeric argument is within a range.

    Args:
        min_val: Minimum allowed value (inclusive). None means no lower bound.
        max_val: Maximum allowed value (inclusive). None means no upper bound.

    Raises:
        ValueError: If the numeric value is outside the allowed range.

    Example::

        @validate_range(min_val=0.0, max_val=1.0)
        def set_threshold(value: float):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i, arg in enumerate(args):
                if isinstance(arg, (int, float)) and not isinstance(arg, bool):
                    _check_range(arg, min_val, max_val, arg_index=i)
                    break
            else:
                for key in ('value', 'threshold', 'ratio', 'score', 'pvalue', 'cutoff'):
                    if key in kwargs and isinstance(kwargs[key], (int, float)):
                        _check_range(kwargs[key], min_val, max_val, kwarg=key)
                        break
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _check_range(value: float, min_val: float, max_val: float, arg_index: int = 0, kwarg: str = None):
    """Internal helper to validate numeric range."""
    name = f"argument '{kwarg}'" if kwarg else f"argument at index {arg_index}"
    if min_val is not None and value < min_val:
        raise ValueError(
            f"Value in {name} ({value}) is below minimum ({min_val})"
        )
    if max_val is not None and value > max_val:
        raise ValueError(
            f"Value in {name} ({value}) is above maximum ({max_val})"
        )


# ── Error Recovery ──────────────────────────────────────────────────────────


def retry_on_error(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = None):
    """Decorator that retries a function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Initial delay between retries (seconds). Doubles each retry.
        exceptions: Tuple of exception types to catch. Defaults to Exception.

    Returns:
        The function's return value on success, or re-raises the last exception.

    Example::

        @retry_on_error(max_retries=3, delay=0.5)
        def fetch_remote(url):
            return requests.get(url)
    """
    if exceptions is None:
        exceptions = (Exception,)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= 2  # exponential backoff
                    else:
                        break
            raise last_exception
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, _default=None, _log_errors: bool = True, **kwargs) -> Tuple[Any, Optional[Exception]]:
    """Execute a function safely, returning (result, error) tuple.

    Args:
        func: Function to execute.
        *args: Positional arguments to pass.
        _default: Default value if execution fails (default: None).
        _log_errors: If True, print traceback on error.
        **kwargs: Keyword arguments to pass.

    Returns:
        Tuple of (result, error). If successful, error is None.
        If failed, result is _default and error is the caught exception.

    Example::

        result, err = safe_execute(complex_analysis, data)
        if err:
            print(f"Analysis failed: {err}")
    """
    try:
        return (func(*args, **kwargs), None)
    except Exception as e:
        if _log_errors:
            traceback.print_exc()
        return (_default, e)


# ── Bioinformatics Format Validators ───────────────────────────────────────


class InputValidator:
    """Comprehensive input validator for bioinformatics file formats and data.

    Provides static methods for validating sequences, FASTA, FASTQ, and VCF data
    without requiring file I/O.

    Example::

        validator = InputValidator()
        errors = validator.validate_fasta('>seq1\\nATCGATCG')
        if errors:
            print(f"Validation errors: {errors}")
    """

    VALID_DNA = set('ATCGN')
    VALID_RNA = set('ACGNU')
    VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY*')

    def validate_sequence(self, sequence: str, seq_type: str = 'dna',
                          min_length: int = 1) -> List[str]:
        """Validate a biological sequence string.

        Args:
            sequence: The sequence string to validate.
            seq_type: One of 'dna', 'rna', 'amino_acid'.
            min_length: Minimum allowed length.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []

        if not sequence or not isinstance(sequence, str):
            errors.append("Sequence must be a non-empty string")
            return errors

        cleaned = sequence.strip().replace(' ', '').replace('\n', '').replace('\r', '')

        if len(cleaned) < min_length:
            errors.append(f"Sequence length {len(cleaned)} is below minimum {min_length}")

        if seq_type == 'dna':
            valid = self.VALID_DNA
            name = 'DNA'
        elif seq_type == 'rna':
            valid = self.VALID_RNA
            name = 'RNA'
        elif seq_type == 'amino_acid':
            valid = self.VALID_AMINO_ACIDS
            name = 'amino acid'
        else:
            errors.append(f"Unknown sequence type: {seq_type}")
            return errors

        invalid_chars = set(cleaned.upper()) - valid
        if invalid_chars:
            errors.append(f"Invalid {name} characters: {sorted(invalid_chars)}")

        return errors

    def validate_fasta(self, content: str) -> List[str]:
        """Validate FASTA format content.

        Checks:
        - At least one sequence entry
        - Headers start with '>'
        - No empty sequences between headers
        - No invalid characters in sequences

        Args:
            content: Raw FASTA file content as string.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []

        if not content or not content.strip():
            errors.append("FASTA content is empty")
            return errors

        lines = content.strip().splitlines()
        headers = [i for i, line in enumerate(lines) if line.startswith('>')]

        if not headers:
            errors.append("No FASTA headers found (lines starting with '>')")
            return errors

        # Check that the first non-empty line is a header
        first_content_line = next(
            (i for i, line in enumerate(lines) if line.strip()), None
        )
        if first_content_line is not None and not lines[first_content_line].startswith('>'):
            errors.append(
                f"Line {first_content_line + 1}: First non-empty line must be a header (starting with '>')"
            )

        # Check each sequence block
        for idx, header_line_idx in enumerate(headers):
            header_text = lines[header_line_idx].strip()
            if header_text == '>':
                errors.append(f"Line {header_line_idx + 1}: Empty header")

            # Gather sequence lines following this header
            seq_start = header_line_idx + 1
            seq_end = (
                headers[idx + 1] if idx + 1 < len(headers) else len(lines)
            )
            seq_lines = [l.strip() for l in lines[seq_start:seq_end] if l.strip()]

            if not seq_lines:
                errors.append(
                    f"Line {header_line_idx + 1}: No sequence data after header '{header_text[:40]}'"
                )
                continue

            full_seq = ''.join(seq_lines).upper()
            invalid = set(full_seq) - self.VALID_DNA
            if invalid:
                errors.append(
                    f"Header '{header_text[:40]}': invalid characters {sorted(invalid)}"
                )

        return errors

    def validate_fastq(self, content: str) -> List[str]:
        """Validate FASTQ format content.

        Checks:
        - Lines come in groups of 4
        - Headers start with '@'
        - Quality scores have same length as sequence
        - Quality score characters are valid (ASCII 33-126)

        Args:
            content: Raw FASTQ file content as string.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []

        if not content or not content.strip():
            errors.append("FASTQ content is empty")
            return errors

        lines = [l.rstrip('\n\r') for l in content.strip().splitlines()]

        if len(lines) % 4 != 0:
            errors.append(
                f"FASTQ must have 4 lines per record; got {len(lines)} total "
                f"({len(lines) % 4} extra line(s))"
            )

        records = len(lines) // 4
        for i in range(records):
            base = i * 4
            header = lines[base] if base < len(lines) else ''
            sequence = lines[base + 1] if base + 1 < len(lines) else ''
            separator = lines[base + 2] if base + 2 < len(lines) else ''
            quality = lines[base + 3] if base + 3 < len(lines) else ''

            # Check header
            if not header.startswith('@'):
                errors.append(f"Record {i + 1} (line {base + 1}): Header must start with '@'")

            # Check separator
            if separator != '+':
                errors.append(f"Record {i + 1} (line {base + 3}): Separator line must be '+'")

            # Check sequence/quality length match
            if len(sequence) != len(quality):
                errors.append(
                    f"Record {i + 1} (line {base + 2}): Sequence length ({len(sequence)}) "
                    f"!= quality length ({len(quality)})"
                )

            # Check quality score validity (Phred+33: ASCII 33-126)
            if quality:
                invalid_q = [c for c in quality if ord(c) < 33 or ord(c) > 126]
                if invalid_q:
                    errors.append(
                        f"Record {i + 1} (line {base + 4}): Invalid quality characters"
                    )

        return errors

    def validate_vcf(self, content: str) -> List[str]:
        """Validate VCF (Variant Call Format) content.

        Checks:
        - Has meta-information lines starting with '##'
        - Has a header line starting with '#CHROM'
        - Required columns present: CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO
        - Data lines have correct number of tab-separated fields

        Args:
            content: Raw VCF file content as string.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []

        if not content or not content.strip():
            errors.append("VCF content is empty")
            return errors

        lines = content.strip().splitlines()

        # Check for header line
        header_line_idx = None
        for i, line in enumerate(lines):
            if line.startswith('#CHROM') or line.startswith('#chrom'):
                header_line_idx = i
                break

        if header_line_idx is None:
            errors.append("Missing VCF header line (must start with '#CHROM')")
            return errors

        # Parse header columns (strip leading # from CHROM)
        header_parts = lines[header_line_idx].split('\t')
        required_cols = {'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'}
        header_upper = {c.strip().lstrip('#').upper() for c in header_parts}

        missing = required_cols - header_upper
        if missing:
            errors.append(f"Missing required VCF columns: {sorted(missing)}")

        # Determine expected column count
        expected_cols = len(header_parts)

        # Check data lines
        data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
        for i, line in enumerate(data_lines):
            parts = line.split('\t')
            if len(parts) != expected_cols:
                errors.append(
                    f"Data line {i + header_line_idx + 2}: Expected {expected_cols} "
                    f"columns, got {len(parts)}"
                )

            # Validate POS is numeric
            if len(parts) >= 2:
                try:
                    pos = int(parts[1])
                    if pos < 1:
                        errors.append(
                            f"Data line {i + header_line_idx + 2}: POS must be >= 1, got {pos}"
                        )
                except ValueError:
                    errors.append(
                        f"Data line {i + header_line_idx + 2}: POS is not an integer: '{parts[1]}'"
                    )

        return errors


# ── Convenience aliases ─────────────────────────────────────────────────────

# A pre-built validator instance for quick use
default_validator = InputValidator()
