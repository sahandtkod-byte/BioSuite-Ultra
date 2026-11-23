"""Discovery-layer tests: probes must not import or execute implementations."""
from __future__ import annotations

import sys

import pytest

from biosuite import capabilities as caps
from biosuite.capabilities import discovery


def test_discover_returns_one_record_per_capability():
    records = discovery.discover_availability(caps.list_capabilities())
    assert len(records) == len(caps.list_capabilities())
    assert set(records) == {c.id for c in caps.list_capabilities()}


def test_ready_reflects_tool_and_package_probes():
    cap = caps.get_capability("sequence.gc_content")
    record = discovery.discover_availability([cap])["sequence.gc_content"]
    assert record.capability_id == "sequence.gc_content"
    assert record.tools == ()
    bio = [p for p in record.packages if p.name == "Bio"]
    assert len(bio) == 1
    assert record.ready is bio[0].available


def test_unavailable_tool_marks_record_not_ready():
    cap = caps.get_capability("msa.auto_align")
    record = discovery.discover_availability([cap])["msa.auto_align"]
    tool_names = {t.name: t.available for t in record.tools}
    assert {"clustalo", "mafft", "muscle"} == set(tool_names)
    if not any(tool_names.values()):
        assert record.ready is False


def test_unknown_package_name_is_handled_not_fatal():
    status = discovery._probe_package("definitely_not_installed_pkg_xyz")
    assert status.available is False


@pytest.mark.parametrize("error", [ImportError, ValueError, ModuleNotFoundError])
def test_package_probe_handles_find_spec_errors(monkeypatch, error):
    def fail(name):
        raise error(name)

    monkeypatch.setattr(discovery.importlib.util, "find_spec", fail)
    assert discovery._probe_package("unavailable").available is False


def test_discover_defaults_to_full_catalog():
    assert discovery.discover_availability() == discovery.discover_availability(
        caps.list_capabilities()
    )
    assert discovery.discover_availability([]) == {}


def test_package_probe_survives_malformed_names():
    status = discovery._probe_package("not a valid module name!")
    assert status.available is False


def test_probes_never_import_biosuite_core_implementations():
    before = set(sys.modules)
    cap = caps.get_capability("sequence.gc_content")
    discovery.discover_availability([cap])
    leaked = {
        name
        for name in sys.modules
        if name.startswith("biosuite.core") and name not in before
    }
    assert not leaked, f"discovery imported core modules: {sorted(leaked)}"


def test_records_are_frozen():
    cap = caps.get_capability("sequence.gc_content")
    record = discovery.discover_availability([cap])["sequence.gc_content"]
    with pytest.raises(Exception):
        record.ready = True


def test_discover_accepts_arbitrary_iterables():
    first_two = list(caps.list_capabilities())[:2]
    records = discovery.discover_availability(iter(first_two))
    assert [r.capability_id for r in records.values()] == [c.id for c in first_two]
