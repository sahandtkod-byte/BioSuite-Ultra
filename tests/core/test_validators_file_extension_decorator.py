"""``@validate_file_extension`` — the path it inspects and the list it accepts.

Two failures were measured on the old code: a `pathlib.Path` argument was never
looked at at all (the scan only matched `str`), and `exts=['fasta']` — the natural
way to write the list — made every real `.fasta` file "unsupported".  The list is
normalised now, and anything that is not a path is refused rather than skipped.
Which parameter the decorator picks is the subject of
`test_validators_argument_targeting.py`.
"""
import pathlib

import pytest

from biosuite.core import validators as v


@pytest.fixture
def bio_formats():
    """``@validate_file_extension()`` — the default bioinformatics formats."""
    @v.validate_file_extension()
    def load(path):
        return path

    return load


def test_the_default_list_covers_the_four_common_formats(bio_formats):
    for name in ("a.fasta", "a.fa", "a.fastq", "a.fq", "UPPER.FASTA", "/data/genomes/a.fq"):
        assert bio_formats(name) == name
    assert tuple(v.DEFAULT_EXTENSIONS) == ('.fasta', '.fa', '.fastq', '.fq')


def test_a_path_object_is_checked_like_any_other_path(bio_formats, tmp_path):
    """Measured on the old code: `load(Path('/data/a.vcf'))` came back untouched —
    the scan asked for a `str`, so the most common path type in Python was never
    validated and the function received the file the decorator was meant to refuse."""
    assert bio_formats(pathlib.Path('/data/a.fa')) == pathlib.Path('/data/a.fa')
    with pytest.raises(ValueError) as excinfo:
        bio_formats(pathlib.Path('/data/a.vcf'))
    assert str(excinfo.value) == (
        "File in argument 'path' of load() has unsupported extension '.vcf'. "
        "Allowed extensions: ['.fa', '.fasta', '.fastq', '.fq']"
    )
    assert bio_formats(tmp_path / 'x.fq') == tmp_path / 'x.fq'


def test_an_unsupported_extension_is_reported_with_the_allowed_list(bio_formats):
    with pytest.raises(ValueError) as excinfo:
        bio_formats("/data/genomes/a.txt")
    assert str(excinfo.value) == (
        "File in argument 'path' of load() has unsupported extension '.txt'. "
        "Allowed extensions: ['.fa', '.fasta', '.fastq', '.fq']"
    )


def test_a_directory_or_a_bare_name_has_no_extension_and_is_refused(bio_formats):
    for bad, label in (("/data/genomes", "a directory"), ("/data/genomes/", "a trailing slash"),
                       ("reads", "a bare name")):
        with pytest.raises(ValueError) as excinfo:
            bio_formats(bad)
        assert str(excinfo.value) == (
            "File in argument 'path' of load() has unsupported extension ''. "
            "Allowed extensions: ['.fa', '.fasta', '.fastq', '.fq']"
        ), label


@pytest.mark.parametrize("bad, message", [
    pytest.param("", "argument 'path' of load() is an empty path.", id="empty"),
    pytest.param("   ", "argument 'path' of load() is an empty path.", id="blank"),
    pytest.param(5, "argument 'path' of load() must be a path string or a path object, "
                    "got int.", id="number"),
    pytest.param(['a.fasta'], "argument 'path' of load() must be a path string or a path "
                              "object, got list.", id="list"),
])
def test_a_value_that_is_not_a_path_is_refused(bio_formats, bad, message):
    with pytest.raises(ValueError) as excinfo:
        bio_formats(bad)
    assert str(excinfo.value) == message


def test_extensions_are_accepted_with_or_without_their_dot_and_their_case():
    """Measured on the old code: `exts=['fasta']` compared against `splitext()`,
    which always returns a dotted extension, so a correct file was refused with a
    message showing a list that could never match anything."""
    @v.validate_file_extension(['FASTA', '.fq', ' fastq '])
    def load(path):
        return path

    for name in ('x.fasta', 'x.FASTA', 'x.fq', 'x.FASTQ'):
        assert load(name) == name
    with pytest.raises(ValueError) as excinfo:
        load('x.fa')
    assert str(excinfo.value) == (
        "File in argument 'path' of load() has unsupported extension '.fa'. "
        "Allowed extensions: ['.fasta', '.fastq', '.fq']"
    )


@pytest.mark.parametrize("exts, error, message", [
    pytest.param([], ValueError,
                 "validate_file_extension exts must list at least one extension.",
                 id="empty-list"),
    pytest.param('.fasta', TypeError,
                 "validate_file_extension exts must be a list of extension strings, "
                 "got the single string '.fasta'.", id="bare-string"),
    pytest.param(('.fasta',), None, None, id="tuple-is-fine"),
    pytest.param([''], ValueError,
                 "validate_file_extension exts must contain non-empty strings, got ''.",
                 id="blank-entry"),
    pytest.param([3], ValueError,
                 "validate_file_extension exts must contain non-empty strings, got 3.",
                 id="number-entry"),
    pytest.param({'fasta': True}, TypeError,
                 "validate_file_extension exts must be a list of extension strings, "
                 "got dict.", id="mapping"),
])
def test_the_allowed_list_is_checked_when_the_decorator_is_built(exts, error, message):
    """A string passed where a list was meant would otherwise be read one
    character at a time: `exts='fasta'` used to allow the extensions '.f', '.a',
    '.s', '.t'."""
    if error is None:
        assert callable(v.validate_file_extension(exts))
        return
    with pytest.raises(error) as excinfo:
        v.validate_file_extension(exts)
    assert str(excinfo.value) == message


def test_a_dotfile_name_is_read_as_a_name_not_an_extension():
    @v.validate_file_extension(['.gz'])
    def load(path):
        return path

    assert load('data.tar.gz') == 'data.tar.gz'
    with pytest.raises(ValueError) as excinfo:
        load('.gitignore')
    assert "unsupported extension ''" in str(excinfo.value)
