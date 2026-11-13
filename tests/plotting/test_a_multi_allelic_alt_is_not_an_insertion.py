"""A multi-allelic ALT is a set of alleles, not one long string.

The variant track classified by comparing ``len(alt)`` with ``len(ref)``, so a
record written ``A`` -> ``A,C`` (two SNPs) was three characters against one and
came out as an insertion, coloured green on the map.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import genome_browser as gb

COLORS = {'SNP': '#ff6b6b', 'DEL': '#ffd93d', 'INS': '#6bcb77'}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("ref, alt, kind", [
    ("A", "G", "SNP"),
    ("A", "A,C", "SNP"),          # two SNPs, one of them the reference allele
    ("A", "A,T,G", "SNP"),
    ("A", "A,*", "SNP"),          # the spanning-deletion star is not a base
    ("A", ".", "SNP"),            # no ALT at all is not an insertion either
    ("A", "AT", "INS"),
    ("A", "A,AT", "INS"),         # one insertion among SNPs
    ("ACG", "A", "DEL"),
    ("ACG", "A,AC", "DEL"),       # one deletion among shorter alleles
])
def test_the_kind_comes_from_every_allele_not_from_the_string_length(ref, alt, kind):
    assert gb._variant_kind(ref, alt) == kind


def test_the_marker_is_coloured_by_that_kind():
    fig, ax = plt.subplots()
    variants = [{'pos': 10, 'ref': 'A', 'alt': 'A,C'},
                {'pos': 20, 'ref': 'A', 'alt': 'AT'},
                {'pos': 30, 'ref': 'ACG', 'alt': 'A'}]

    gb._draw_variant_track(ax, {'variants': variants}, ('chr1', 0, 50), 'grey')

    assert [line.get_markerfacecolor() for line in ax.get_lines()] == \
        [COLORS['SNP'], COLORS['INS'], COLORS['DEL']]
    assert sorted(t.get_text() for t in ax.texts) == ['A>A,C', 'A>AT', 'ACG>A']
