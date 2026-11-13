"""A row the parser cannot use is counted, not thrown away and not fatal.

One non-numeric start used to abort the whole BED file with ``int()``'s own
message and no line number (every later row lost), a backwards interval was
parsed into a box with negative width, and both parsers cut at their limit
without saying so.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import genome_browser as gb

GOOD_BED = "chr1\t100\t200\tok\t5\n"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _bed(tmp_path, text):
    path = tmp_path / "x.bed"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_an_unreadable_row_is_named_by_line_and_the_rest_still_read(tmp_path):
    problems = []

    regions = gb.parse_bed(_bed(tmp_path, GOOD_BED +
                                "chr1\tabc\t200\tbroken\t5\n"
                                "chr1\t300\t400\tafter\n"), problems=problems)

    assert [r[3] for r in regions] == ["ok", "after"], "one bad row lost the good ones"
    assert problems == ["line 2: invalid literal for int() with base 10: 'abc'"], problems


def test_a_short_row_says_how_short_it_was(tmp_path):
    problems = []

    gb.parse_bed(_bed(tmp_path, GOOD_BED + "chr1\t300\n"), problems=problems)

    assert problems == ["line 2: fewer than three columns"], problems


def test_a_backwards_interval_is_refused_rather_than_drawn(tmp_path):
    problems = []

    regions = gb.parse_bed(_bed(tmp_path, "chr1\t500\t300\tback\t5\n"), problems=problems)

    assert regions == []
    assert problems == ["line 1: interval 500-300 does not run forwards"], problems


def test_the_limit_is_reported_when_it_cuts_the_file(tmp_path):
    rows = "".join(f"chr1\t{i}\t{i + 10}\tn{i}\n" for i in range(12))
    problems = []

    regions = gb.parse_bed(_bed(tmp_path, rows), max_regions=5, problems=problems)

    assert len(regions) == 5
    assert problems == ["stopped after 5 regions"], problems


def test_the_track_carries_the_rows_it_could_not_use(tmp_path):
    track = gb.create_bed_track(_bed(tmp_path, GOOD_BED +
                                     "chr1\tabc\t200\tbroken\n" + "chr2\t1\t2\n"))
    # 'chr2\t1\t2' is a valid three-column record, so two problems is one bad row
    assert len(track['skipped']) == 1, track['skipped']
    assert track['data']['regions'][0][3] == "ok"


def test_the_browser_says_how_many_rows_it_lost(tmp_path):
    track = gb.create_bed_track(_bed(tmp_path, GOOD_BED + "chr1\tabc\t200\tbroken\n"))

    fig = gb.plot_genome_tracks([track])

    assert fig.axes[0].get_ylabel() == "BED — 1 row(s) skipped", \
        fig.axes[0].get_ylabel()


def test_a_clean_file_carries_no_note():
    """The positive control: the label is untouched when nothing was skipped."""
    track = {'type': 'bed', 'name': 'BED', 'data': {'regions': [('chr1', 1, 5, 'a', 0)]},
             'skipped': []}

    fig = gb.plot_genome_tracks([track])

    assert fig.axes[0].get_ylabel() == "BED"


def test_vcf_rows_are_counted_the_same_way(tmp_path):
    path = tmp_path / "x.vcf"
    path.write_text("##fileformat=VCFv4.2\n"
                    "chr1\t100\t.\tA\tG\t30\tPASS\t.\n"
                    "chr1\tx\t.\tA\tG\t30\tPASS\t.\n"
                    "chr1\t0\t.\tA\tG\t30\tPASS\t.\n"
                    "short\trow\n", encoding="utf-8")
    problems = []

    variants = gb.parse_vcf(str(path), problems=problems)

    assert len(variants) == 1
    assert problems == ["line 3: invalid literal for int() with base 10: 'x'",
                        "line 4: POS 0 is not a 1-based position",
                        "line 5: fewer than five columns"], problems


def test_the_vcf_limit_is_reported_too(tmp_path):
    rows = "".join(f"chr1\t{i}\t.\tA\tG\t.\t.\t.\n" for i in range(1, 8))
    path = tmp_path / "x.vcf"
    path.write_text("##fileformat=VCFv4.2\n" + rows, encoding="utf-8")
    problems = []

    variants = gb.parse_vcf(str(path), max_variants=3, problems=problems)

    assert len(variants) == 3
    assert problems == ["stopped after 3 variants"], problems
