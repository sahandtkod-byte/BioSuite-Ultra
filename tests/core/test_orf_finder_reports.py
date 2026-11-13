"""The ORF finder's report text and the two input edges in its scan.

The three ``format_*`` functions are what a user reads, so they are tested on the
rendered lines rather than on a return type: a header, one row per item, and the
amplicon size only when both primers exist.  In the scan itself, a segment that
opens on an unknown codon is trimmed to the first codon that means something, and
an enzyme name that is not in the table is skipped instead of matching nothing.
"""
import pytest

from biosuite.core.orf_finder import (
    ORF,
    Primer,
    RestrictionSite,
    find_orfs,
    find_restriction_sites,
    format_orf_results,
    format_primers,
    format_restriction_sites,
)


def _orf(**kwargs):
    fields = {'frame': 1, 'start': 0, 'end': 30, 'length': 10, 'protein': 'MKABCDEFGHI',
              'has_start_codon': True, 'has_stop_codon': True}
    fields.update(kwargs)
    return ORF(**fields)


def _primer(**kwargs):
    fields = {'name': 'F1', 'sequence': 'ACGTACGTAA', 'length': 10, 'gc_content': 50.0,
              'tm': 54.2, 'position': 100, 'strand': '+'}
    fields.update(kwargs)
    return Primer(**fields)


def _site(**kwargs):
    fields = {'enzyme': 'EcoRI', 'sequence': 'GAATTC', 'cut_position': 1,
              'position': 42, 'strand': '+'}
    fields.update(kwargs)
    return RestrictionSite(**fields)


# ─── the scan ────────────────────────────────────────────────────────────────

def test_a_segment_opening_on_an_unknown_codon_is_trimmed_to_the_first_real_one():
    """'N' translates to 'X', which is not an amino acid a report should lead with."""
    orfs = find_orfs("NNNAAACCCGGGTTTAAATGA", min_length=2, include_start=False)

    frame1 = next(o for o in orfs if o.frame == 1)
    assert 'X' not in frame1.protein, frame1.protein
    assert frame1.protein == "KPGFK", "the NNN codon was dropped from the front"
    assert frame1.start == 3, "and the reported start moved with it"


def test_an_enzyme_that_is_not_in_the_table_is_skipped_not_matched():
    sites = find_restriction_sites("GAATTC", enzymes=["NotARealEnzyme", "EcoRI"])

    assert [s.enzyme for s in sites] == ["EcoRI"], sites
    assert sites[0].sequence == "GAATTC"


def test_with_no_enzyme_list_the_whole_table_is_searched():
    sites = find_restriction_sites("GAATTCGGATCC")

    found = {s.enzyme for s in sites}
    assert {"EcoRI", "BamHI"} <= found, found
    assert all(s.sequence in "GAATTCGGATCC" for s in sites), \
        "every reported site is text the sequence really contains"
    assert sorted(s.position for s in sites if s.enzyme == "EcoRI") == [0]


# ─── the reports ─────────────────────────────────────────────────────────────

def test_the_orf_table_lists_the_header_even_for_no_orfs():
    report = format_orf_results([])

    lines = report.splitlines()
    assert lines[0] == "=== Open Reading Frames ==="
    assert lines[1] == "Found 0 ORFs", lines[1]
    assert "Frame" in lines[3] and "Stop?" in lines[3], "the column header is still printed"
    assert len([line for line in lines if line.startswith(("1", "2"))]) == 0


def test_one_orf_row_carries_its_own_frame_positions_and_codon_flags():
    report = format_orf_results([_orf(frame=2, start=11, end=41, length=10,
                                      has_start_codon=False, has_stop_codon=True)])

    rows = [line for line in report.splitlines() if line[:1].isdigit()]
    assert len(rows) == 1, report
    row = rows[0]
    assert row.split()[0] == "1", "the rows are numbered from one"
    assert "2" in row.split()[1:], "frame 2 is what the sequence says"
    assert "11" in row and "41" in row and "10" in row
    assert "No" in row and "Yes" in row, "start absent, stop present"


def test_the_orf_table_prints_at_most_twenty_rows_of_a_longer_list():
    report = format_orf_results([_orf(frame=f) for f in range(1, 26)])

    rows = [line for line in report.splitlines() if line[:1].isdigit()]
    assert "Found 25 ORFs" in report, "the count is the real one, not the printed one"
    assert len(rows) == 20, rows


def test_the_restriction_table_prints_one_row_per_site():
    sites = [_site(position=10), _site(enzyme="BamHI", sequence="GGATCC", position=25)]

    report = format_restriction_sites(sites)

    assert "Found 2 cut sites" in report
    assert "EcoRI" in report and "BamHI" in report
    assert "10" in report and "25" in report
    assert len([line for line in report.splitlines() if line.startswith(('EcoRI', 'BamHI'))]) == 2


def test_the_restriction_table_caps_its_rows_at_thirty():
    report = format_restriction_sites([_site(position=i) for i in range(40)])

    assert "Found 40 cut sites" in report
    assert len([line for line in report.splitlines() if line.startswith("EcoRI")]) == 30


def test_the_primer_report_measures_an_amplicon_only_when_it_has_both_ends():
    fwd, rev = _primer(strand='+'), _primer(name='R1', strand='-', position=150)

    both = format_primers(fwd, rev)
    only_forward = format_primers(fwd, None)
    only_reverse = format_primers(None, rev)
    neither = format_primers(None, None)

    assert "Forward: ACGTACGTAA" in both and "Reverse: ACGTACGTAA" in both
    assert "Amplicon size: ~60 bp" in both, both
    assert "Forward" in only_forward and "Reverse" not in only_forward
    assert "Amplicon" not in only_forward, "one end alone is not a distance"
    assert "Reverse" in only_reverse and "Amplicon" not in only_reverse
    assert neither.splitlines() == ["=== Designed Primers ==="]


def test_a_primer_row_reports_its_own_length_gc_and_melting_point():
    report = format_primers(_primer(length=21, gc_content=48.5, tm=60.1, position=7), None)

    assert "Length: 21 bp, GC: 48.5%, Tm: 60.1°C" in report, report
    assert "Position: 7" in report, report
