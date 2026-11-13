"""`InputValidator.validate_vcf` — the header, the meta lines, and true line numbers.

The method's docstring listed four checks and performed three: nothing ever looked
for the `##` meta-information lines it promised, so a file with no `##fileformat`
at all validated cleanly.  Its data-line numbers were also counted from a filtered
list rather than the file, so a blank line between two records moved every number
after it — `audit_output/line_by_line_core.md` records this as finding C-148/C-149.
"""
import pytest

from biosuite.core.validators import InputValidator

VCF_HEADER = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
META = "##fileformat=VCFv4.2"


@pytest.fixture
def validator():
    return InputValidator()


def record(chrom='chr1', pos='100', identifier='.', ref='A', alt='G', qual='50',
           filt='PASS', info='.'):
    return '\t'.join((chrom, pos, identifier, ref, alt, qual, filt, info))


# ── a file that is a VCF ────────────────────────────────────────────────────

def test_a_minimal_file_with_meta_header_and_one_record_passes(validator):
    content = f"{META}\n{VCF_HEADER}\n{record()}\n"
    assert validator.validate_vcf(content) == []
    assert validator.validate_vcf(content.replace('#CHROM', '#chrom', 1)) == []


def test_a_sample_column_in_the_header_raises_the_expected_field_count(validator):
    header = VCF_HEADER + "\tFORMAT\tSAMPLE1"
    line = record() + "\tGT:0/1\t0/1"
    assert validator.validate_vcf(f"{META}\n{header}\n{line}\n") == []


def test_a_comment_line_between_records_is_a_comment_and_not_a_record(validator):
    content = f"{META}\n{VCF_HEADER}\n{record()}\n##interlude\n{record(pos='200')}\n"
    assert validator.validate_vcf(content) == []


# ── the checks the docstring promised ───────────────────────────────────────

def test_meta_information_lines_are_now_required(validator):
    """Measured on the old code: a file that began with its `#CHROM` line and had no
    `##` line at all returned `[]` — the check the docstring listed was never
    written."""
    content = f"{VCF_HEADER}\n{record()}\n"
    assert validator.validate_vcf(content) == [
        "Missing VCF meta-information lines (none start with '##'; the first line should "
        "be '##fileformat=VCFv4.2')"
    ]


def test_a_record_line_before_the_header_is_named(validator):
    content = f"{META}\n{record(pos='1')}\n{VCF_HEADER}\n"
    assert validator.validate_vcf(content) == [
        "Line 2: data before the '#CHROM' header line (line 3)"
    ]


@pytest.mark.parametrize("content", ["", "  \n", None])
def test_content_that_holds_nothing_is_not_a_file(validator, content):
    assert validator.validate_vcf(content) == ["VCF content is empty"]


def test_a_file_without_its_header_line_stops_at_that(validator):
    assert validator.validate_vcf(f"{META}\n") == [
        "Missing VCF header line (must start with '#CHROM')"
    ]


def test_every_missing_column_is_listed_at_once(validator):
    assert validator.validate_vcf("#CHROM\tPOS\tREF\tALT\n") == [
        "Missing VCF meta-information lines (none start with '##'; the first line should "
        "be '##fileformat=VCFv4.2')",
        "Missing required VCF columns: ['FILTER', 'ID', 'INFO', 'QUAL']",
    ]


# ── the record lines ────────────────────────────────────────────────────────

def test_a_record_with_too_few_or_too_many_fields_says_both_numbers(validator):
    too_few = validator.validate_vcf(f"{META}\n{VCF_HEADER}\nchr1\t100\t.\tA\tG\n")
    assert too_few == ["Data line 3: Expected 8 columns, got 5"]
    too_many = validator.validate_vcf(f"{META}\n{VCF_HEADER}\n{record()}\tEXTRA\n")
    assert too_many == ["Data line 3: Expected 8 columns, got 9"]


def test_a_line_with_one_field_is_reported_once_not_also_as_a_position(validator):
    assert validator.validate_vcf(f"{META}\n{VCF_HEADER}\nnocolons\n") == [
        "Data line 3: Expected 8 columns, got 1"
    ]


@pytest.mark.parametrize("pos, message", [
    pytest.param("0", "Data line 3: POS must be >= 1, got 0"),
    pytest.param("-5", "Data line 3: POS must be >= 1, got -5"),
    pytest.param("foo", "Data line 3: POS is not an integer: 'foo'"),
    pytest.param("1.5", "Data line 3: POS is not an integer: '1.5'"),
    pytest.param("", "Data line 3: POS is not an integer: ''"),
])
def test_the_position_column_has_to_be_a_positive_integer(validator, pos, message):
    content = f"{META}\n{VCF_HEADER}\n{record(pos=pos)}\n"
    assert validator.validate_vcf(content) == [message]


def test_line_numbers_point_at_the_line_the_user_sees(validator):
    """Measured on the old code: the content was `strip()`ed before splitting and
    the data lines were numbered from a filtered list, so with two leading blank
    lines and one between records a bad line was reported five lines early."""
    content = (f"\n\n{META}\n{VCF_HEADER}\n{record()}\n\n{record(pos='bad')}\n")
    assert validator.validate_vcf(content) == [
        "Data line 7: POS is not an integer: 'bad'"
    ]


def test_several_problems_are_all_reported_in_line_order(validator):
    content = (f"{META}\n{VCF_HEADER}\n{record(pos='0')}\n\n"
               f"{record(pos='bad', alt='G', info='.')}\tX\n")
    assert validator.validate_vcf(content) == [
        "Data line 3: POS must be >= 1, got 0",
        "Data line 5: Expected 8 columns, got 9",
        "Data line 5: POS is not an integer: 'bad'",
    ]
