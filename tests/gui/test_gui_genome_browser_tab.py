"""The Genome Browser tab: the track list, the loaders it calls, and what it reports.

``_gb_view`` reads ``type:path`` lines, builds a track per line with
``biosuite.plotting.genome_browser``, and shows them together.  The tests write real
BED and VCF files into the temporary directory and compute the expected track counts
with the module's own parsers, so the box is checked against the files rather than
against a remembered message.
"""
import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import genome_browser

BED_LINES = [
    "chr1\t1000\t2000\tgene_a\t500",
    "chr1\t2500\t3400\tgene_b\t120",
    "chr2\t500\t1500\tgene_c\t900",
]

VCF_LINES = [
    "##fileformat=VCFv4.2",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    "chr1\t1200\t.\tA\tG\t99.5\tPASS\tDP=12",
    "chr1\t3000\trs1\tC\tT\t.\tq0\tDP=3",
]


@pytest.fixture
def bed_file(tmp_path):
    path = tmp_path / "genes.bed"
    path.write_text("\n".join(BED_LINES) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def vcf_file(tmp_path):
    path = tmp_path / "calls.vcf"
    path.write_text("\n".join(VCF_LINES) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def empty_browser(ui):
    ui.clear("gb_tracks_text")
    ui.clear("gb_info")
    return ui


def test_add_bed_records_the_picked_file_as_a_track(empty_browser, bed_file):
    empty_browser.open_path = str(bed_file)

    empty_browser.app._gb_add_bed()

    assert f"bed:{bed_file}" in empty_browser.text("gb_tracks_text")


def test_a_cancelled_bed_dialog_adds_nothing(empty_browser):
    empty_browser.open_path = ""

    empty_browser.app._gb_add_bed()

    assert empty_browser.text("gb_tracks_text").strip() == ""


def test_add_vcf_records_the_picked_file_as_a_track(empty_browser, vcf_file):
    empty_browser.open_path = str(vcf_file)

    empty_browser.app._gb_add_vcf()

    assert f"vcf:{vcf_file}" in empty_browser.text("gb_tracks_text")


def test_a_cancelled_vcf_dialog_adds_nothing(empty_browser):
    empty_browser.open_path = ""

    empty_browser.app._gb_add_vcf()

    assert empty_browser.text("gb_tracks_text").strip() == ""


def test_viewing_draws_every_track_the_list_names(empty_browser, bed_file, vcf_file):
    expected_regions = len(genome_browser.parse_bed(str(bed_file)))
    expected_variants = len(genome_browser.parse_vcf(str(vcf_file)))
    assert expected_regions == 3 and expected_variants == 2
    empty_browser.set("gb_tracks_text", f"bed:{bed_file}\nvcf:{vcf_file}")
    plt.close("all")

    empty_browser.app._gb_view()

    assert f"Loaded {2} tracks\n" in empty_browser.text("gb_info"), empty_browser.text("gb_info")
    assert empty_browser.plots, "the browser never displayed the tracks"
    title, fig = empty_browser.plots[-1]
    assert title == "Genome Browser"
    assert any(t.get_text() == "Genome Browser" for t in fig.texts),             "the figure ignores the title it was given"
    assert empty_browser.messages == []


def test_a_missing_track_file_is_reported_in_the_box(empty_browser, bed_file, tmp_path):
    absent = tmp_path / "deleted.bed"
    empty_browser.set("gb_tracks_text", f"bed:{absent}")

    empty_browser.app._gb_view()

    info = empty_browser.text("gb_info")
    assert f"Error loading {absent}" in info, info
    assert empty_browser.messages_of("warning") == ["No valid tracks found."]


def test_a_track_plotter_failure_is_reported_instead_of_escaping(empty_browser, bed_file, monkeypatch):
    def broken(tracks, **kwargs):
        raise RuntimeError("the renderer is unwell")

    monkeypatch.setattr("biosuite.plotting.genome_browser.plot_genome_tracks", broken)
    empty_browser.set("gb_tracks_text", f"bed:{bed_file}")

    empty_browser.app._gb_view()

    assert empty_browser.messages_of("error") == ["the renderer is unwell"]
    assert empty_browser.text("gb_info").strip() == ""


def test_a_broken_track_next_to_a_good_one_is_still_visible(empty_browser, bed_file, tmp_path):
    """The summary used to be written with delete-then-insert, which erased every
    error the same click had reported and left the user thinking both tracks
    loaded."""
    absent = tmp_path / "deleted.bed"
    empty_browser.set("gb_tracks_text", f"bed:{bed_file}\nvcf:{absent}")
    plt.close("all")

    empty_browser.app._gb_view()

    info = empty_browser.text("gb_info")
    assert "Loaded 1 tracks" in info, info
    assert f"Error loading {absent}" in info, info


def test_lines_the_browser_cannot_use_are_skipped(empty_browser):
    empty_browser.set("gb_tracks_text", "coverage:x.bw\njust some text\nvcf:\t")

    empty_browser.app._gb_view()

    assert empty_browser.messages_of("warning") == ["No valid tracks found."]
    assert empty_browser.plots == []


def test_an_empty_track_list_is_refused(empty_browser):
    empty_browser.app._gb_view()

    assert empty_browser.messages_of("warning") == ["Add BED or VCF files first."]


def test_the_sidebar_entry_shows_the_browser_frame(empty_browser):
    empty_browser.app._gui_genome_browser()

    assert empty_browser.app._current_frame == "genomebrowser"
