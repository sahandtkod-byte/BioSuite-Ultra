"""WS7: the packaged static manifest matches the generated one, byte for byte.

``biosuite/capabilities/capabilities.json`` is generated from
``manifest_json(build_manifest())`` and committed as a static artifact so a
packaged wheel or sdist carries the catalog without executing any code.  These
tests fail if the committed file drifts from the live catalog, or if runtime
state (versions, paths, timestamps, availability) ever leaks into it.
"""

from __future__ import annotations

import json
from importlib import resources

from biosuite.capabilities.manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    manifest_json,
)

PACKAGED_MANIFEST = "capabilities.json"

RUNTIME_STATE_MARKERS = (
    "python_version",
    "platform",
    "machine",
    "hostname",
    "timestamp",
    "generated_at",
    "generated_by",
    "installed",
    "status",
)


def _packaged_bytes() -> bytes:
    return (
        resources.files("biosuite.capabilities")
        .joinpath(PACKAGED_MANIFEST)
        .read_bytes()
    )


def test_packaged_manifest_is_byte_identical_to_generated() -> None:
    assert _packaged_bytes().decode("utf-8") == manifest_json(build_manifest())


def test_regenerating_the_manifest_does_not_drift() -> None:
    # Drift test: if the catalog changes, the static file must be regenerated.
    assert manifest_json(build_manifest()) == manifest_json(build_manifest())
    assert _packaged_bytes() == manifest_json(build_manifest()).encode("utf-8")


def test_packaged_manifest_has_exactly_the_expected_top_level_keys() -> None:
    data = json.loads(_packaged_bytes())
    assert set(data) == {
        "schema_version",
        "total",
        "mcp_exposed_total",
        "capabilities",
        "mcp_tools",
    }
    assert data["schema_version"] == MANIFEST_SCHEMA_VERSION == "1.0.0"


def test_packaged_manifest_carries_no_runtime_state() -> None:
    text = _packaged_bytes().decode("utf-8")
    data = json.loads(text)

    def keys_of(value: object) -> set:
        if isinstance(value, dict):
            return set(value)
        return set()

    top_keys = keys_of(data)
    assert not top_keys & set(RUNTIME_STATE_MARKERS)
    entry_keys: set = set()
    tool_keys: set = set()
    for entry in data["capabilities"]:
        entry_keys |= keys_of(entry) | keys_of(entry.get("requires", {}))
    for tool in data["mcp_tools"]:
        tool_keys |= keys_of(tool)
    assert not entry_keys & set(RUNTIME_STATE_MARKERS)
    assert not tool_keys & set(RUNTIME_STATE_MARKERS)
    # No host paths, venvs or ISO timestamps anywhere in the artifact.
    assert ".venv" not in text and "/home/" not in text and "T00:00" not in text
