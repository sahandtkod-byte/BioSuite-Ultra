"""The sample plasmid contains the sites it names, and says what its sequence is.

``create_sample_plasmid`` advertised "pUC19-like" with EcoRI, BamHI, HindIII and
PstI features while its sequence was unseeded random bases: none of the four
recognition sites existed at the offsets the features named (measured: EcoRI only
at 682, BamHI and HindIII absent), and two calls built two different plasmids.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import plasmid_map as pm

SITES = [("EcoRI", "GAATTC", 396), ("BamHI", "GGATCC", 412),
         ("HindIII", "AAGCTT", 433), ("PstI", "CTGCAG", 440)]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("name, motif, offset", SITES)
def test_every_restriction_site_is_where_its_feature_says(name, motif, offset):
    plasmid = pm.create_sample_plasmid()

    assert plasmid.sequence[offset:offset + len(motif)] == motif


@pytest.mark.parametrize("name, motif, offset", SITES)
def test_the_feature_spans_the_site_it_names(name, motif, offset):
    plasmid = pm.create_sample_plasmid()
    feature = next(f for f in plasmid.features if f.name == name)

    assert (feature.start, feature.end) == (offset, offset + len(motif))


def test_two_builds_are_the_same_plasmid():
    assert pm.create_sample_plasmid().sequence == pm.create_sample_plasmid().sequence


def test_the_sequence_is_flagged_and_reported_as_synthetic():
    plasmid = pm.create_sample_plasmid()

    assert plasmid.sequence_is_synthetic is True
    assert "Sequence: synthetic filler" in pm.format_plasmid_report(plasmid)
    assert "Synthetic" in plasmid.description


def test_a_size_only_map_is_flagged_too_and_a_real_sequence_is_not():
    assert pm.PlasmidMap(name="p", size=10).sequence_is_synthetic is True
    assert pm.PlasmidMap(name="p", sequence="ATGC").sequence_is_synthetic is False


def test_the_report_of_a_stored_sequence_carries_no_filler_line():
    plasmid = pm.PlasmidMap(name="p", sequence="ATGC" * 10)

    assert "synthetic filler" not in pm.format_plasmid_report(plasmid)


def test_the_sample_still_has_its_ten_features_and_size():
    plasmid = pm.create_sample_plasmid()

    assert (plasmid.size, len(plasmid.features)) == (2686, 10)
    assert len(plasmid.sequence) == plasmid.size
