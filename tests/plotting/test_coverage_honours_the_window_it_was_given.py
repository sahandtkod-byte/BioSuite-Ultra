"""A coverage window is a window: the text fallback now clips to it like pysam does.

`_compute_coverage_sam` honoured only the chromosome name: asked for
(chr1, 4900, 5100) it returned bins from 0, counted a read at position 100, and
stopped at the last read instead of the requested end — while the pysam path
spanned the window.  The same call drew two different axes depending on whether
pysam was installed.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biosuite.plotting import genome_browser as gb

SAM = ("@SQ\tSN:chr1\tLN:10000\n"
       "early\t0\tchr1\t100\t60\t50M\t*\t0\t0\t" + "A" * 50 + "\t" + "I" * 50 + "\n"
       "inwin\t0\tchr1\t5000\t60\t50M\t*\t0\t0\t" + "A" * 50 + "\t" + "I" * 50 + "\n"
       "other\t0\tchr9\t5000\t60\t50M\t*\t0\t0\t" + "A" * 50 + "\t" + "I" * 50 + "\n")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def sam(tmp_path):
    path = tmp_path / "x.sam"
    path.write_text(SAM, encoding="utf-8")
    return str(path)


def test_the_axis_spans_the_window_that_was_asked_for(sam):
    positions, coverage = gb._compute_coverage_sam(sam, ("chr1", 4900, 5100), 100)

    assert positions[0] == 4900
    assert positions[-1] == 5100, "the track ends where the window ends"


def test_a_read_before_the_window_does_not_count(sam):
    _, whole = gb._compute_coverage_sam(sam, ("chr1", 0, 5200), 100)
    _, window = gb._compute_coverage_sam(sam, ("chr1", 4900, 5100), 100)

    assert window.sum() == 2, "only the read inside the window is counted"
    assert whole.sum() == 4, "both chr1 reads count when the window holds them"


def test_a_read_on_another_chromosome_does_not_count(sam):
    _, coverage = gb._compute_coverage_sam(sam, ("chr1", 4900, 5100), 100)

    assert coverage.sum() == 2, "the chr9 read is not in this window"


def test_the_read_is_clipped_to_the_window_edge(tmp_path):
    """A read straddling the window start contributes only its inside part."""
    path = tmp_path / "straddle.sam"
    path.write_text("@SQ\tSN:chr1\tLN:10000\n"
                    "r\t0\tchr1\t1001\t60\t200M\t*\t0\t0\t" + "A" * 200 + "\t"
                    + "I" * 200 + "\n", encoding="utf-8")

    positions, coverage = gb._compute_coverage_sam(str(path), ("chr1", 1100, 1300), 100)

    assert list(positions) == [1100, 1200, 1300]
    assert list(coverage) == [1, 0, 0], \
        "the read ends at 1199, so only the first bin of the window holds it"


def test_a_window_with_no_reads_is_an_empty_track_of_the_right_span(sam):
    positions, coverage = gb._compute_coverage_sam(sam, ("chr1", 8000, 8300), 100)

    assert coverage.sum() == 0
    assert positions[0] == 8000 and positions[-1] == 8300


def test_a_read_that_ends_before_the_window_is_skipped(tmp_path):
    path = tmp_path / "early.sam"
    path.write_text("@SQ\tSN:chr1\tLN:10000\n"
                    "r\t0\tchr1\t100\t60\t50M\t*\t0\t0\t" + "A" * 50 + "\t"
                    + "I" * 50 + "\n", encoding="utf-8")

    _, coverage = gb._compute_coverage_sam(str(path), ("chr1", 9000, 9200), 100)

    assert coverage.sum() == 0, "a read that ends at 148 cannot reach bin 9000"
