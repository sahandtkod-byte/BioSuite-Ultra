"""Input validation decorators, error recovery utilities, and bio-format validators.

This module provides:
- Decorators for validating sequences, file extensions, DataFrames, and numeric ranges
- Error recovery: retry with exponential backoff, safe execution wrappers
- InputValidator class for bioinformatics file formats (FASTA, FASTQ, VCF)

Each decorator decides **once, at decoration time**, which parameters of the
wrapped function it inspects — see `_resolve_targets`.  Scanning the call for "the
first argument of a matching type" cannot work: a function that takes a label and
a sequence would have its label checked and its sequence let through, and a call
that carries no matching value at all would be validated by nobody.
"""

import functools
import inspect
import math
import numbers
import os
import time
from typing import Any, Callable, List, Optional, Tuple

from .log import get_logger

logger = get_logger(__name__)

#: Extensions `validate_file_extension` accepts when the caller names none.
DEFAULT_EXTENSIONS = ('.fasta', '.fa', '.fastq', '.fq')

#: The keyword names each decorator recognises as its own argument.
SEQUENCE_ARGUMENT_NAMES = ('sequence', 'seq', 'dna', 'rna', 'sequence_str')
PATH_ARGUMENT_NAMES = ('filepath', 'path', 'file', 'filename', 'input_file')
DATAFRAME_ARGUMENT_NAMES = ('df', 'dataframe', 'data', 'table')
NUMBER_ARGUMENT_NAMES = ('value', 'threshold', 'ratio', 'score', 'pvalue', 'cutoff')

_SEQUENCE_COLLECTIONS = (list, tuple, set, frozenset)


# ── Argument targeting ──────────────────────────────────────────────────────


def _where(arg: str, func_name: str) -> str:
    """How a refusal names the argument it looked at."""
    return f"argument '{arg}' of {func_name}()"


def _resolve_targets(func: Callable, names: tuple, annotation, decorator: str):
    """Return the ``(signature, parameter names)`` pair *decorator* inspects.

    The parameters are chosen once, in a fixed order:

    1. **every** parameter whose name is in *names* — the decorator's own
       vocabulary, in that order;
    2. if no name matches, every parameter annotated with *annotation*;
    3. if neither, the first ordinary parameter of the function.

    ``*args`` / ``**kwargs`` are not ordinary parameters: a function that takes
    nothing else has no argument a decorator could name, and saying so while the
    decorator is applied beats validating nothing at run time.

    Raises:
        TypeError: if the signature cannot be read or has no ordinary parameter.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{decorator} cannot read the signature of {func!r}: {exc}") from exc

    ordinary = [p for p in signature.parameters.values()
                if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                  inspect.Parameter.VAR_KEYWORD)
                and p.name not in ('self', 'cls')]
    if not ordinary:
        raise TypeError(f"{decorator} cannot decorate {func.__name__}(): its signature "
                        f"{signature} has no ordinary parameter to inspect.")

    present = {p.name for p in ordinary}
    targets = [name for name in names if name in present]
    if targets:
        return signature, targets
    targets = [p.name for p in ordinary if p.annotation is annotation]
    if targets:
        return signature, targets
    return signature, [ordinary[0].name]


def _bound_arguments(signature, args: tuple, kwargs: dict):
    """Return ``(arguments, supplied)``, or ``(None, None)`` for a call that does not fit.

    ``arguments`` maps every parameter to the value the call gives it (defaults
    filled in), and ``supplied`` names the ones the caller actually passed.  The
    two are needed together because ``None`` means two different things: a
    parameter the caller left out is nothing to validate, while a parameter the
    caller deliberately set to ``None`` is a value of the wrong type and says so.

    A call that does not fit the signature at all is left to Python: the decorated
    function's own "takes 1 positional argument but 2 were given" reaches the
    caller instead of a validator's opinion about it.
    """
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return None, None
    supplied = set(bound.arguments)
    bound.apply_defaults()
    return bound.arguments, supplied


def _checked(func: Callable, decorator: str, names: tuple, annotation, check: Callable):
    """Build the wrapper that resolves the targets once and hands their values to *check*."""
    signature, targets = _resolve_targets(func, names, annotation, decorator)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        arguments, supplied = _bound_arguments(signature, args, kwargs)
        if arguments is not None:
            for name in targets:
                if arguments[name] is None and name not in supplied:
                    continue
                check(arguments[name], name, func.__name__)
        return func(*args, **kwargs)

    return wrapper


def _text_choices(values) -> str:
    """Quote a collection of names for an error message, in a stable order."""
    return ', '.join(repr(v) for v in sorted(values, key=repr))


# ── Input Validation Decorators ─────────────────────────────────────────────


def validate_sequence(min_length: int = 1, allowed_chars: str = 'ATCGN'):
    """Decorator that validates the DNA/RNA sequence argument of a function.

    Args:
        min_length: Minimum sequence length (inclusive); a non-negative integer.
        allowed_chars: Set of allowed characters (case-insensitive).

    Raises:
        ValueError: if ``min_length`` is not a non-negative integer, or if the
            sequence is too short or contains characters outside *allowed_chars*.
        TypeError: if ``allowed_chars`` is not a string, or the decorated function
            has no parameter to inspect.

    Example::

        @validate_sequence(min_length=4, allowed_chars='ATCG')
        def reverse_complement(seq: str) -> str:
            comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
            return ''.join(comp[b] for b in reversed(seq))
    """
    if isinstance(min_length, bool) or not isinstance(min_length, numbers.Integral) \
            or min_length < 0:
        raise ValueError(f"validate_sequence min_length must be a non-negative integer, "
                         f"got {min_length!r}.")
    if not isinstance(allowed_chars, str):
        raise TypeError(f"validate_sequence allowed_chars must be a string of characters, "
                        f"got {type(allowed_chars).__name__}.")
    min_length = int(min_length)
    allowed = frozenset(allowed_chars.upper())

    def check(seq, arg, func_name):
        _check_sequence(seq, min_length, allowed, arg, func_name)

    def decorator(func: Callable) -> Callable:
        return _checked(func, 'validate_sequence', SEQUENCE_ARGUMENT_NAMES, str, check)
    return decorator


def _check_sequence(seq, min_length: int, allowed: frozenset, arg: str, func_name: str):
    """Refuse anything that is not a long enough string over *allowed*."""
    where = _where(arg, func_name)
    if not isinstance(seq, str):
        raise ValueError(f"{where} must be a string, got {type(seq).__name__}.")
    if len(seq) < min_length:
        raise ValueError(f"Sequence in {where} is too short: {len(seq)} < {min_length} "
                         f"(min_length={min_length})")
    invalid = sorted(set(seq.upper()) - allowed)
    if invalid:
        raise ValueError(f"Sequence in {where} contains invalid characters: {invalid}. "
                         f"Allowed characters: {sorted(allowed)}")


def validate_file_extension(exts: List[str] = None):
    """Decorator that validates a file path has an allowed extension.

    Args:
        exts: Allowed extensions, with or without the leading dot (e.g.
            ['.fasta', '.fa', '.fastq', '.fq']).
            Defaults to the common bioinformatics formats.

    Raises:
        ValueError: if ``exts`` is empty, holds a non-string or a blank string, or
            if the file extension is not in the allowed list.
        TypeError: if ``exts`` is not a collection of strings (a bare string would
            silently be read one character at a time) or the decorated function has
            no parameter to inspect.

    Example::

        @validate_file_extension(['.fasta', '.fa', '.fastq'])
        def parse_file(filepath: str):
            ...
    """
    if exts is None:
        exts = list(DEFAULT_EXTENSIONS)
    if isinstance(exts, (str, bytes)):
        raise TypeError(f"validate_file_extension exts must be a list of extension strings, "
                        f"got the single string {exts!r}.")
    if not isinstance(exts, _SEQUENCE_COLLECTIONS):
        raise TypeError(f"validate_file_extension exts must be a list of extension strings, "
                        f"got {type(exts).__name__}.")
    if not exts:
        raise ValueError("validate_file_extension exts must list at least one extension.")
    allowed_exts = frozenset(_normalise_extension(ext) for ext in exts)

    def check(filepath, arg, func_name):
        _check_file_extension(filepath, allowed_exts, arg, func_name)

    def decorator(func: Callable) -> Callable:
        return _checked(func, 'validate_file_extension', PATH_ARGUMENT_NAMES, str, check)
    return decorator


def _normalise_extension(ext):
    """Lower-case *ext* and give it the leading dot its comparison needs."""
    if not isinstance(ext, str) or not ext.strip():
        raise ValueError(f"validate_file_extension exts must contain non-empty strings, "
                         f"got {ext!r}.")
    ext = ext.strip().lower()
    return ext if ext.startswith('.') else '.' + ext


def _check_file_extension(filepath, allowed_exts: frozenset, arg: str, func_name: str):
    where = _where(arg, func_name)
    if isinstance(filepath, os.PathLike):
        filepath = os.fspath(filepath)
    if not isinstance(filepath, str):
        raise ValueError(f"{where} must be a path string or a path object, got "
                         f"{type(filepath).__name__}.")
    if not filepath.strip():
        raise ValueError(f"{where} is an empty path.")
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in allowed_exts:
        raise ValueError(f"File in {where} has unsupported extension {ext!r}. "
                         f"Allowed extensions: {sorted(allowed_exts)}")


def validate_dataframe_columns(required_cols: List[str] = None):
    """Decorator that validates a pandas DataFrame argument has required columns.

    Args:
        required_cols: Column names that must be present. Defaults to ['gene'].

    Raises:
        ValueError: if ``required_cols`` is empty or holds a non-string, or if the
            DataFrame is missing required columns.
        TypeError: if ``required_cols`` is not a collection of strings, or the
            decorated function has no parameter to inspect.

    Example::

        @validate_dataframe_columns(['gene', 'expression'])
        def normalize_expression(df):
            ...
    """
    import pandas as pd

    if required_cols is None:
        required_cols = ['gene']
    if isinstance(required_cols, (str, bytes)):
        raise TypeError(f"validate_dataframe_columns required_cols must be a list of column "
                        f"names, got the single string {required_cols!r}.")
    if not isinstance(required_cols, _SEQUENCE_COLLECTIONS):
        raise TypeError(f"validate_dataframe_columns required_cols must be a list of column "
                        f"names, got {type(required_cols).__name__}.")
    if not required_cols:
        raise ValueError("validate_dataframe_columns required_cols must list at least one "
                         "column name.")
    for col in required_cols:
        if not isinstance(col, str) or not col:
            raise ValueError(f"validate_dataframe_columns required_cols must contain non-empty "
                             f"strings, got {col!r}.")
    required = set(required_cols)

    def check(df, arg, func_name):
        _check_dataframe_columns(df, required, pd, arg, func_name)

    def decorator(func: Callable) -> Callable:
        return _checked(func, 'validate_dataframe_columns', DATAFRAME_ARGUMENT_NAMES,
                        pd.DataFrame, check)
    return decorator


def _check_dataframe_columns(df, required: set, pd, arg: str, func_name: str):
    where = _where(arg, func_name)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{where} must be a pandas DataFrame, got {type(df).__name__}.")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame in {where} is missing required columns: {sorted(missing)}. "
                         f"Available columns: {list(df.columns)}")


def validate_range(min_val: float = None, max_val: float = None):
    """Decorator that validates a numeric argument is within a range.

    Args:
        min_val: Minimum allowed value (inclusive). None means no lower bound.
        max_val: Maximum allowed value (inclusive). None means no upper bound.

    Raises:
        ValueError: if a bound is not a finite number or ``min_val`` is above
            ``max_val``, or if the value is outside the allowed range.
        TypeError: if the decorated function has no parameter to inspect.

    Example::

        @validate_range(min_val=0.0, max_val=1.0)
        def set_threshold(value: float):
            ...
    """
    for label, bound in (('min_val', min_val), ('max_val', max_val)):
        if bound is None:
            continue
        if isinstance(bound, bool) or not isinstance(bound, numbers.Real) \
                or not math.isfinite(bound):
            raise ValueError(f"validate_range {label} must be a finite number or None, "
                             f"got {bound!r}.")
    if min_val is not None and max_val is not None and min_val > max_val:
        raise ValueError(f"validate_range needs min_val <= max_val, got min_val={min_val!r} "
                         f"and max_val={max_val!r}; no number could satisfy both.")

    def check(value, arg, func_name):
        _check_range(value, min_val, max_val, arg, func_name)

    def decorator(func: Callable) -> Callable:
        return _checked(func, 'validate_range', NUMBER_ARGUMENT_NAMES, None, check)
    return decorator


def _check_range(value, min_val, max_val, arg: str, func_name: str):
    """Refuse a non-number, a non-finite number, and a number outside the bounds."""
    where = _where(arg, func_name)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{where} must be a number, got {type(value).__name__}.")
    if not math.isfinite(value):
        raise ValueError(f"{where} must be a finite number, got {value!r}.")
    if min_val is not None and value < min_val:
        raise ValueError(f"Value in {where} ({value}) is below minimum ({min_val})")
    if max_val is not None and value > max_val:
        raise ValueError(f"Value in {where} ({value}) is above maximum ({max_val})")


# ── Error Recovery ──────────────────────────────────────────────────────────


def retry_on_error(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = None):
    """Decorator that retries a function on failure with exponential backoff.

    Args:
        max_retries: Number of retries after the first attempt, so the function is
            called at most ``max_retries + 1`` times.  A non-negative integer.
        delay: First delay between retries (seconds); it doubles each retry.
            A finite, non-negative number.
        exceptions: Exception class, or non-empty tuple of classes, to retry on.
            Defaults to ``Exception``.

    Raises:
        ValueError: if a parameter is out of contract.
        BaseException: the wrapped function's own error, once the retries are used
            up or when it raises something outside *exceptions*.

    Every retry is logged as a warning, so a function that only succeeds on its
    third attempt is visible rather than silently flaky.

    Example::

        @retry_on_error(max_retries=3, delay=0.5)
        def fetch_remote(url):
            return requests.get(url)
    """
    if isinstance(max_retries, bool) or not isinstance(max_retries, numbers.Integral) \
            or max_retries < 0:
        raise ValueError(f"retry_on_error max_retries must be a non-negative integer, "
                         f"got {max_retries!r}.")
    if isinstance(delay, bool) or not isinstance(delay, numbers.Real) \
            or not math.isfinite(delay) or delay < 0:
        raise ValueError(f"retry_on_error delay must be a finite, non-negative number, "
                         f"got {delay!r}.")
    if exceptions is None:
        caught: tuple = (Exception,)
    elif _is_exception_class(exceptions):
        caught = (exceptions,)
    elif isinstance(exceptions, tuple) and exceptions and all(
            _is_exception_class(e) for e in exceptions):
        caught = exceptions
    else:
        raise ValueError(f"retry_on_error exceptions must be an exception class or a "
                         f"non-empty tuple of them, got {exceptions!r}.")
    max_retries = int(max_retries)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except caught as exc:
                    logger.warning(f"{func.__name__} failed with {type(exc).__name__}: {exc}; "
                                   f"retry {attempt + 1} of {max_retries} in {current_delay} "
                                   f"second(s)")
                    time.sleep(current_delay)
                    current_delay *= 2
            # The budget is spent: this last call's outcome, good or bad, is the answer.
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _is_exception_class(candidate) -> bool:
    return isinstance(candidate, type) and issubclass(candidate, BaseException)


def safe_execute(func: Callable, /, *args, _default=None, _log_errors: bool = True,
                 **kwargs) -> Tuple[Any, Optional[Exception]]:
    """Execute a function safely, returning a ``(result, error)`` tuple.

    Args:
        func: Callable to execute.  Anything else is a programming error and is
            raised, not reported as a failed analysis.
        *args: Positional arguments to pass.
        _default: Value to return if execution fails (default: None).
        _log_errors: If True, log the failure and its traceback through the
            package logger, which is where the CLI/GUI report goes.
        **kwargs: Keyword arguments to pass.

    Returns:
        Tuple of (result, error). If successful, error is None.
        If failed, result is _default and error is the caught exception.

    Example::

        result, err = safe_execute(complex_analysis, data)
        if err:
            logger.error(f"Analysis failed: {err}")
    """
    if not callable(func):
        raise TypeError(f"safe_execute needs a callable, got {type(func).__name__}.")
    try:
        return (func(*args, **kwargs), None)
    except Exception as exc:
        if _log_errors:
            logger.error(f"{getattr(func, '__name__', repr(func))} failed: "
                         f"{type(exc).__name__}: {exc}", exc_info=True)
        return (_default, exc)


# ── Bioinformatics Format Validators ───────────────────────────────────────


class InputValidator:
    """Comprehensive input validator for bioinformatics file formats and data.

    Validates sequences and FASTA, FASTQ and VCF content without any file I/O: the
    content is the string you already read.

    The nucleotide and amino-acid alphabets are the IUPAC ones, gaps included, so
    an aligned or soft-masked FASTA passes — the same alphabet
    ``biosuite.core.sequence`` uses.  ``sequence.IUPAC_NUCLEOTIDES`` and these
    ``VALID_*`` sets are pinned against each other by a test, because two answers
    in one package is how a bad file gets read twice.

    Example::

        validator = InputValidator()
        errors = validator.validate_fasta('>seq1\\nATCGATCG')
        if errors:
            logger.error(f"Validation errors: {errors}")
    """

    #: IUPAC nucleotide codes, ``N`` for anything unknown, and the gap/mask
    #: characters that legitimately appear in aligned or soft-masked records.
    VALID_DNA = frozenset('ACGTNRYSWKMBDHVN-.*')
    VALID_RNA = frozenset('ACGUNRYSWKMBDHVN-.*')
    #: The 20 standard residues plus ``*`` (stop) and the IUPAC ambiguity codes.
    VALID_AMINO_ACIDS = frozenset('ACDEFGHIKLMNPQRSTVWY*BJOUXZ-')

    SEQ_TYPES = {'dna': ('DNA', VALID_DNA),
                 'rna': ('RNA', VALID_RNA),
                 'amino_acid': ('amino acid', VALID_AMINO_ACIDS)}

    def _seq_type(self, seq_type):
        """The ``(display name, alphabet)`` pair for *seq_type*, case-insensitively."""
        if isinstance(seq_type, str):
            return self.SEQ_TYPES.get(seq_type.strip().lower())
        return None

    @staticmethod
    def _unknown_seq_type(seq_type) -> str:
        return (f"Unknown sequence type: {seq_type!r}; use 'dna', 'rna' or 'amino_acid'.")

    @staticmethod
    def _bad_min_length(min_length) -> str:
        return f"min_length must be a non-negative integer, got {min_length!r}."

    def validate_sequence(self, sequence: str, seq_type: str = 'dna',
                          min_length: int = 1) -> List[str]:
        """Validate a biological sequence string.

        Args:
            sequence: The sequence string to validate.
            seq_type: One of 'dna', 'rna', 'amino_acid'; upper-case spellings are
                accepted because the sequence itself is matched case-insensitively.
            min_length: Minimum allowed length; a non-negative integer.

        Returns:
            List of error messages (empty if valid).  Every rejected input comes
            back as a message, including a ``min_length`` or ``seq_type`` that the
            caller got wrong.
        """
        errors = []

        if isinstance(min_length, bool) or not isinstance(min_length, numbers.Integral) \
                or min_length < 0:
            return [self._bad_min_length(min_length)]
        seq_alphabet = self._seq_type(seq_type)
        if seq_alphabet is None:
            return [self._unknown_seq_type(seq_type)]
        name, valid = seq_alphabet

        if not isinstance(sequence, str):
            return [f"Sequence must be a string, got {type(sequence).__name__}."]
        if not sequence:
            return ["Sequence must be a non-empty string"]

        # Any whitespace a caller pasted in is formatting, not sequence content.
        cleaned = ''.join(sequence.split())

        if len(cleaned) < min_length:
            errors.append(f"Sequence length {len(cleaned)} is below minimum {min_length}")

        invalid_chars = set(cleaned.upper()) - valid
        if invalid_chars:
            errors.append(f"Invalid {name} characters: {sorted(invalid_chars)}")

        return errors

    def validate_fasta(self, content: str, seq_type: str = 'dna') -> List[str]:
        """Validate FASTA format content.

        Checks:
        - At least one sequence entry
        - The first non-empty line is a header
        - Headers start with '>' and carry a name
        - No empty sequences between headers
        - No characters outside the alphabet of *seq_type*

        Args:
            content: Raw FASTA file content as string.
            seq_type: 'dna' (default), 'rna' or 'amino_acid' — the alphabet the
                sequence lines are checked against.  A protein file needs
                ``seq_type='amino_acid'``; without it its residues are reported as
                invalid characters.

        Returns:
            List of error messages (empty if valid).  Line numbers count the lines
            of *content* as it came in, leading blank lines included.
        """
        errors = []

        if not isinstance(content, str) or not content.strip():
            errors.append("FASTA content is empty")
            return errors
        seq_alphabet = self._seq_type(seq_type)
        if seq_alphabet is None:
            return [self._unknown_seq_type(seq_type)]
        name, valid = seq_alphabet

        lines = content.splitlines()
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
                f"Line {first_content_line + 1}: First non-empty line must be a header "
                f"(starting with '>')"
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
                    f"Line {header_line_idx + 1}: No sequence data after header "
                    f"'{header_text[:40]}'"
                )
                continue

            full_seq = ''.join(seq_lines).upper()
            invalid = set(full_seq) - valid
            if invalid:
                errors.append(
                    f"Header '{header_text[:40]}': invalid {name} characters "
                    f"{sorted(invalid)}"
                )

        return errors

    def validate_fastq(self, content: str) -> List[str]:
        """Validate FASTQ format content.

        Checks:
        - Lines come in groups of four, allowing blank lines between records
        - Headers start with '@'
        - The separator line is '+' or the header repeated, as the format allows
        - Quality scores have the same length as the sequence
        - Quality score characters are valid (Phred+33, ASCII 33-126)

        Args:
            content: Raw FASTQ file content as string.

        Returns:
            List of error messages (empty if valid), each naming the record and the
            line of *content* it came from.  A file this accepts is a file
            ``biosuite.core.trimming`` can read, and one it refuses is one the
            trimmer refuses too.
        """
        errors = []

        if not isinstance(content, str) or not content.strip():
            errors.append("FASTQ content is empty")
            return errors

        lines = content.splitlines()
        index = 0
        record = 0
        while True:
            # A blank line where a header belongs separates records; it is not a field.
            while index < len(lines) and not lines[index].strip():
                index += 1
            if index >= len(lines):
                break

            record += 1
            remaining = len(lines) - index
            if remaining < 4:
                errors.append(
                    f"FASTQ must have 4 lines per record; lines {index + 1}-{len(lines)} "
                    f"are a truncated final record ({remaining} of 4 lines)"
                )
                return errors

            header = lines[index]
            sequence = lines[index + 1]
            separator = lines[index + 2]
            quality = lines[index + 3]

            if not header.startswith('@'):
                errors.append(f"Record {record} (line {index + 1}): Header must start with '@'")

            if not _is_separator_line(separator, header):
                errors.append(
                    f"Record {record} (line {index + 3}): Separator line must be '+' or a "
                    f"repeat of the header, got {separator!r}"
                )

            if len(sequence) != len(quality):
                errors.append(
                    f"Record {record} (line {index + 2}): Sequence length ({len(sequence)}) "
                    f"!= quality length ({len(quality)})"
                )

            bad = sorted({c for c in quality if not 33 <= ord(c) <= 126})
            if bad:
                errors.append(
                    f"Record {record} (line {index + 4}): quality characters outside "
                    f"Phred+33: {_text_choices(bad)}"
                )

            index += 4

        return errors

    def validate_vcf(self, content: str) -> List[str]:
        """Validate VCF (Variant Call Format) content.

        Checks:
        - Has meta-information lines starting with '##'
        - Has a header line starting with '#CHROM'
        - Required columns present: CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO
        - No record line appears before that header
        - Data lines have the same number of tab-separated fields as the header
        - POS is an integer at least 1

        Args:
            content: Raw VCF file content as string.

        Returns:
            List of error messages (empty if valid).  Data lines are reported by
            their number in *content*, so a blank or comment line between records
            does not move the numbers of the lines after it.
        """
        errors = []

        if not isinstance(content, str) or not content.strip():
            errors.append("VCF content is empty")
            return errors

        lines = content.splitlines()

        # Check for header line
        header_line_idx = None
        for i, line in enumerate(lines):
            if line.startswith('#CHROM') or line.startswith('#chrom'):
                header_line_idx = i
                break

        if header_line_idx is None:
            errors.append("Missing VCF header line (must start with '#CHROM')")
            return errors

        if not any(line.startswith('##') for line in lines):
            errors.append("Missing VCF meta-information lines (none start with '##'; "
                          "the first line should be '##fileformat=VCFv4.2')")

        for i, line in enumerate(lines[:header_line_idx]):
            if line.strip() and not line.startswith('#'):
                errors.append(f"Line {i + 1}: data before the '#CHROM' header line "
                              f"(line {header_line_idx + 1})")

        # Parse header columns (strip leading # from CHROM)
        header_parts = lines[header_line_idx].split('\t')
        required_cols = {'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'}
        header_upper = {c.strip().lstrip('#').upper() for c in header_parts}

        missing = required_cols - header_upper
        if missing:
            errors.append(f"Missing required VCF columns: {sorted(missing)}")

        # Determine expected column count
        expected_cols = len(header_parts)

        # Check data lines, numbered by their place in the file
        for i, line in enumerate(lines):
            if line.startswith('#') or not line.strip():
                continue
            number = i + 1
            parts = line.split('\t')
            if len(parts) != expected_cols:
                errors.append(
                    f"Data line {number}: Expected {expected_cols} "
                    f"columns, got {len(parts)}"
                )

            # Validate POS is numeric
            if len(parts) >= 2:
                try:
                    pos = int(parts[1])
                    if pos < 1:
                        errors.append(f"Data line {number}: POS must be >= 1, got {pos}")
                except ValueError:
                    errors.append(f"Data line {number}: POS is not an integer: '{parts[1]}'")

        return errors


def _is_separator_line(separator: str, header: str) -> bool:
    """True for a legal FASTQ '+' line: the bare separator or the header repeated.

    The Sanger FASTQ specification allows the third line to carry the read
    identifier again, and ``biosuite.core.trimming`` reads such files, so treating
    ``+`` and ``+read1`` as different files here would make the package disagree
    with itself.
    """
    if separator == '+':
        return True
    return header.startswith('@') and separator == '+' + header[1:].split()[0]


# ── Convenience aliases ─────────────────────────────────────────────────────

# A pre-built validator instance for quick use
default_validator = InputValidator()
