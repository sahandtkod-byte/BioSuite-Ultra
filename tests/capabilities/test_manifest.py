"""Manifest tests: deterministic build, JSON stability, legacy view."""
from __future__ import annotations

import json

import pytest

from biosuite import capabilities as caps
from biosuite.capabilities import manifest


@pytest.fixture()
def built():
    return manifest.build_manifest(caps.list_capabilities())


def test_manifest_shape_and_totals(built):
    assert built["schema_version"] == manifest.MANIFEST_SCHEMA_VERSION
    assert "biosuite_version" not in built
    assert built["total"] == 341
    assert built["mcp_exposed_total"] == 4
    assert len(built["capabilities"]) == 341
    assert len(built["mcp_tools"]) == 4


def test_manifest_json_is_byte_stable(built):
    first = manifest.manifest_json(built)
    rebuilt = manifest.build_manifest(caps.list_capabilities())
    second = manifest.manifest_json(rebuilt)
    assert first == second
    assert first.endswith("\n")
    assert json.loads(first)["total"] == 341


def test_build_manifest_omits_runtime_version(monkeypatch):
    import biosuite

    monkeypatch.setattr(biosuite, "__version__", "runtime-version-must-not-leak")
    assert "runtime-version-must-not-leak" not in manifest.manifest_json(manifest.build_manifest())
    assert "biosuite_version" not in manifest.build_manifest()


def test_manifest_defaults_to_catalog():
    assert manifest.build_manifest() == manifest.build_manifest(caps.list_capabilities())


def test_manifest_accepts_iterators_and_empty_catalog():
    entries = caps.list_capabilities()
    assert manifest.build_manifest(iter(entries)) == manifest.build_manifest(entries)
    empty = manifest.build_manifest([])
    assert empty["total"] == 0
    assert empty["mcp_tools"] == []


def test_mcp_tools_defaults_to_catalog():
    assert caps.mcp_tools() == caps.mcp_tools(caps.list_capabilities())
    assert caps.mcp_tools([]) == []


def test_mcp_tools_shape():
    tools = caps.mcp_tools(caps.list_capabilities())
    assert [t["name"] for t in tools] == [
        "sequence.gc_content",
        "sequence.reverse_complement",
        "sequence.translate",
        "sequence.sequence_stats",
    ]
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"
        assert tool["outputSchema"]["type"] == "object"
        assert tool["description"]
        assert set(tool) == {"name", "description", "inputSchema", "outputSchema"}


def test_mcp_tools_input_schemas_are_declared_for_every_exposed_tool():
    for tool in caps.mcp_tools(caps.list_capabilities()):
        assert tool["name"] in caps.mcp.MCP_INPUT_SCHEMAS
        assert tool["name"] in caps.mcp.MCP_OUTPUT_SCHEMAS


def test_mcp_tools_ignores_unexposed_entries():
    unexposed = [c for c in caps.list_capabilities() if not c.mcp_exposed]
    assert caps.mcp_tools(unexposed) == []


def test_legacy_modules_view_has_exactly_ten_ordered_entries():
    view = caps.legacy_modules_view(total_endpoints=38)
    assert len(view["modules"]) == 10
    names = [m["name"] for m in view["modules"]]
    assert names == [
        "sequence",
        "alignment",
        "expression",
        "plotting",
        "crispr",
        "metagenomics",
        "epitope",
        "gwas",
        "phylogeny",
        "popgen",
    ]
    for entry in view["modules"]:
        assert set(entry) == {"name", "endpoints", "description"}
        assert entry["endpoints"][0].startswith("/api/v1/")


def test_legacy_modules_view_caller_supplied_total_and_fresh_objects():
    first = caps.legacy_modules_view(total_endpoints=38)
    second = caps.legacy_modules_view(total_endpoints=41)
    assert first["total_endpoints"] == 38
    assert second["total_endpoints"] == 41
    first["modules"][0]["name"] = "mutated"
    assert caps.legacy_modules_view(total_endpoints=38)["modules"][0]["name"] == "sequence"


def test_legacy_endpoints_are_not_shared():
    view = caps.legacy_modules_view(total_endpoints=38)
    view["modules"][0]["endpoints"].append("/mutated")
    assert caps.legacy_modules_view(total_endpoints=38)["modules"][0]["endpoints"] == [
        "/api/v1/sequence/*"
    ]


def test_mcp_schemas_are_not_shared():
    tools = caps.mcp_tools()
    tools[0]["inputSchema"]["properties"]["seq"]["type"] = "integer"
    tools[0]["outputSchema"]["required"].clear()
    fresh = caps.mcp_tools()[0]
    assert fresh["inputSchema"]["properties"]["seq"]["type"] == "string"
    assert fresh["outputSchema"]["required"] == ["gc_percent"]


@pytest.mark.parametrize("tool", caps.mcp_tools(), ids=lambda t: t["name"])
def test_mcp_inputs_match_core_signatures(tool):
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).parents[2] / "biosuite/core/sequence.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == tool["name"].split(".")[1])
    names = [arg.arg for arg in function.args.args]
    schema = tool["inputSchema"]
    assert set(schema["properties"]) == set(names)
    required_count = len(names) - len(function.args.defaults)
    assert schema["required"] == names[:required_count]
    for name, default in zip(names[required_count:], function.args.defaults):
        assert schema["properties"][name]["default"] == ast.literal_eval(default)


@pytest.mark.parametrize("frame", [1, 2, 3, -1, -2, -3])
def test_translation_schema_advertises_valid_frames(frame):
    from biosuite.core.sequence import translate

    properties = caps.mcp.MCP_INPUT_SCHEMAS["sequence.translate"]["properties"]
    assert set(properties["frame"]["enum"]) == {1, 2, 3, -1, -2, -3}
    assert properties["table"]["enum"] == [1]
    assert isinstance(translate("ATGCGT", frame=frame, table=1), str)


def test_statistics_output_schema_matches_real_result():
    from biosuite.core.sequence import sequence_stats

    schema = caps.mcp.MCP_OUTPUT_SCHEMAS["sequence.sequence_stats"]
    result = sequence_stats("ATGCGT")
    assert set(schema["properties"]) == set(result)
    assert set(schema["required"]) == set(result)
    for key, definition in schema["properties"].items():
        assert isinstance(result[key], int if definition["type"] == "integer" else float)


def test_manifest_json_round_trips(built):
    parsed = json.loads(manifest.manifest_json(built))
    assert parsed == built


def test_json_serialization_uses_sorted_keys(built):
    text = manifest.manifest_json(built)
    json.loads(text, object_pairs_hook=_assert_sorted_keys)


def _assert_sorted_keys(pairs):
    keys = [key for key, _ in pairs]
    assert keys == sorted(keys)
    return dict(pairs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_manifest_json_rejects_nonfinite_numbers(value):
    with pytest.raises(ValueError):
        manifest.manifest_json({"invalid": value})


def test_manifest_bytes_match_across_processes_without_runtime_metadata():
    import os
    import subprocess
    import sys

    outputs = []
    for seed in ("17", "83"):
        code = (
            "import biosuite, platform, sys; "
            f"biosuite.__version__ = 'package-{seed}'; "
            f"platform.python_version = lambda: 'python-{seed}'; "
            f"platform.platform = lambda: 'platform-{seed}'; "
            f"platform.node = lambda: 'host-{seed}'; "
            "from biosuite.capabilities.manifest import build_manifest, manifest_json; "
            "sys.stdout.buffer.write(manifest_json(build_manifest()).encode('utf-8'))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True, check=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    data = json.loads(outputs[0])
    assert data["schema_version"] == manifest.MANIFEST_SCHEMA_VERSION

    def check_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key != "schema_version":
                    assert "version" not in key.lower()
                assert key not in {"availability", "available", "ready", "timestamp", "host", "platform", "path"}
                check_keys(child)
        elif isinstance(value, list):
            for child in value:
                check_keys(child)

    check_keys(data)
