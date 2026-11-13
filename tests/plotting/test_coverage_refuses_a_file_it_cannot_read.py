"""The coverage reader refuses a file it cannot read and arguments it cannot use.

Without pysam — this machine's state — the text fallback was handed a real BAM and
died inside the UTF-8 decoder, and ``bin_size=0`` divided by zero three frames
down.  A bare string region was read as chrom 'c', start 'h', end 'r', which
filtered every read out and returned an empty plot without a word.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import genome_browser as gb

SAM = ("@SQ\tSN:chr1\tLN:1000\n"
       "r1\t0\tchr1\t100\t60\t50M\t*\t0\t0\t" + "A" * 50 + "\t" + "I" * 50 + "\n")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def sam(tmp_path):
    path = tmp_path / "x.sam"
    path.write_text(SAM, encoding="utf-8")
    return str(path)


def test_a_bam_read_as_text_names_pysam_instead_of_the_decoder(tmp_path):
    path = tmp_path / "x.bam"
    path.write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xffBAM\x01")

    with pytest.raises(ValueError) as excinfo:
        gb.compute_coverage(str(path))

    message = str(excinfo.value)
    assert "is binary" in message and "pip install pysam" in message, message


def test_a_sam_file_is_still_read(sam):
    """The positive control for the refusal above."""
    positions, coverage = gb.compute_coverage(sam, None, 10)

    assert coverage.sum() == 6, "a 50 bp read from 0-based 99 touches bins 9..14"


@pytest.mark.parametrize("bin_size, reason", [
    (0, "at least one base"),
    (-5, "at least one base"),
    (2.5, "whole number"),
    (True, "whole number"),
])
def test_a_bin_size_that_is_not_a_positive_whole_number_is_refused(bin_size, reason):
    with pytest.raises(ValueError, match=reason):
        gb.compute_coverage("anything.sam", None, bin_size)


@pytest.mark.parametrize("region, reason", [
    ("chr1", r"region must be \(chrom, start, end\)"),
    (("chr1", "100", 200), "whole base position"),
    (("chr1", 0, "100"), "whole base position or None"),
    (("chr1", 100, 50), "ends before it starts"),
])
def test_a_region_that_is_not_a_window_is_refused_by_name(region, reason):
    with pytest.raises(ValueError, match=reason):
        gb.compute_coverage("anything.sam", region)


def test_an_open_region_is_legal_and_filled_in_from_the_header(sam):
    """The pysam path fills a missing end from the header; the fallback accepts it."""
    positions, coverage = gb.compute_coverage(sam, ("chr1", 100, None), 10)

    assert len(positions) == len(coverage)
    assert coverage.sum() == 5
    assert positions[0] == 100
