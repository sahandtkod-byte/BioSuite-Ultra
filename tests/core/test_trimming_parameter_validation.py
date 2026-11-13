"""The arguments of ``trimming.py``'s entry points are checked before the module opens a
single file.

Measured on the pre-cluster code: a ``min_length`` that arrived as text - which is exactly
what a caller gets from a web form or a CSV header - reached the record loop as
``TypeError: '<' not supported between instances of 'int' and 'str'``, raised **after** the
output file had been created and one record had already been written into it.
"""
import re

import pytest

from biosuite.core import trimming as tr

FASTQ8 = "".join(f"@r{i}\nACGTACGT\n+\nIIIIIIII\n" for i in range(3))


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("kwargs, message", [
    pytest.param(dict(quality_threshold=20, min_length="36"),
                 "trim_fastq min_length must be a non-negative integer, got '36'.",
                 id="str-min-length"),
    pytest.param(dict(quality_threshold="20", min_length=36),
                 "trim_fastq quality_threshold must be an integer Phred score, got '20'.",
                 id="str-threshold"),
    pytest.param(dict(quality_threshold=20.5, min_length=36),
                 "quality_threshold must be an integer Phred score, got 20.5.",
                 id="float-threshold"),
    pytest.param(dict(quality_threshold=True, min_length=36),
                 "quality_threshold must be an integer Phred score, got True.",
                 id="bool-threshold"),
    pytest.param(dict(quality_threshold=20, min_length=-1),
                 "min_length must be a non-negative integer, got -1.", id="negative-length"),
])
def test_cut_parameters_are_refused_before_any_file_is_written(tmp_path, kwargs, message):
    src = _write(tmp_path, "in.fastq", FASTQ8)
    out = tmp_path / "out.fastq"

    with pytest.raises(ValueError, match=re.escape(message)):
        tr.trim_fastq(src, str(out), **kwargs)
    assert not out.exists(), "the refusal happens before the output file is created"


def test_trim_pair_end_refuses_the_same_parameters(tmp_path):
    src = _write(tmp_path, "p.fastq", FASTQ8)

    with pytest.raises(ValueError, match=re.escape(
            "trim_pair_end min_length must be a non-negative integer, got '36'.")):
        tr.trim_pair_end(src, src, min_length="36")


@pytest.mark.parametrize("limit", [0, -1, True, '10', None])
def test_the_quality_profile_refuses_an_impossible_read_limit(tmp_path, limit):
    src = _write(tmp_path, "g.fastq", FASTQ8)
    assert tr.analyze_fastq_quality(src, max_reads=limit) == {
        "error": f"max_reads must be a positive integer, got {limit!r}."}


def test_an_unknown_adapter_name_is_refused_rather_than_read_as_auto(tmp_path):
    """Measured before: a name that is not in ``ADAPTERS`` fell through to the 'auto'
    list, so a caller who asked for one adapter got all five - on a run whose report
    still said the trim had used the caller's choice."""
    src = _write(tmp_path, "in.fastq", FASTQ8)

    with pytest.raises(ValueError, match=re.escape(
            "unknown adapter_name 'truseq'; known names are illumina_nextera, "
            "illumina_truseq_dna, illumina_truseq_rna, polya, polyg, or pass the "
            "adapter sequence in adapter=")):
        tr.trim_fastq(src, str(tmp_path / "out.fastq"), adapter_name='truseq')

    assert tr._resolve_adapters('auto', 'polya') == ['AAAAAAAAAA']
    assert len(tr._resolve_adapters('auto', None)) == len(tr.ADAPTERS)
    assert tr._resolve_adapters('ACGT', None) == ['ACGT'], "a literal sequence passes through"
