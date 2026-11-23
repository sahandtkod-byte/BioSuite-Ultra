"""Catalog inventory tests: counts, order, uniqueness, AST validation."""
from __future__ import annotations

import ast
import pathlib

import pytest

from biosuite import capabilities as caps

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CORE = REPO_ROOT / "biosuite" / "core"

MCP_EXPOSED_IDS = frozenset(
    {
        "sequence.gc_content",
        "sequence.reverse_complement",
        "sequence.translate",
        "sequence.sequence_stats",
    }
)

EXPECTED_MODULES = 47
EXPECTED_ENTRIES = 341

LAYER_NAMES = frozenset({"api", "cli", "gui", "notebook", "plotting", "python"})


def _core_public_functions(module: str) -> set:
    """Top-level public functions of ``biosuite.core.<module>`` per AST."""
    path = CORE / f"{module}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


def _expected_module_names() -> set:
    return {
        p.relative_to(CORE).with_suffix("").as_posix()
        for p in CORE.rglob("*.py")
        if p.name != "__init__.py"
    }


def test_inventory_counts_are_exact():
    capabilities = caps.list_capabilities()
    assert len(capabilities) == EXPECTED_ENTRIES
    assert len({c.module for c in capabilities}) == EXPECTED_MODULES
    assert {c.module for c in capabilities} == _expected_module_names()


def test_mcp_flag_count_and_identity():
    capabilities = caps.list_capabilities()
    exposed = [c for c in capabilities if c.mcp_exposed]
    assert len(exposed) == 4
    assert {c.id for c in exposed} == MCP_EXPOSED_IDS


def test_ids_are_unique_and_catalog_order_is_deterministic():
    first = caps.list_capabilities()
    second = caps.list_capabilities()
    ids = [c.id for c in first]
    assert len(ids) == len(set(ids))
    assert [c.id for c in second] == ids


@pytest.mark.parametrize("capability", caps.list_capabilities(), ids=lambda c: c.id)
def test_every_entry_has_an_ast_validated_implementation(capability):
    assert capability.implementation == f"{capability.module}:{capability.function}"
    assert capability.function in _core_public_functions(capability.module)


def _interface_imports() -> dict:
    """Map each interface layer to the core modules it imports, per AST."""
    imported = {layer: set() for layer in LAYER_NAMES - {"python"}}
    for layer in imported:
        for path in (REPO_ROOT / "biosuite" / layer).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0:
                        names = [node.module] if node.module else []
                    else:
                        pkg = ".".join(
                            path.relative_to(REPO_ROOT).with_suffix("").parts[:-1]
                        )
                        for _ in range(node.level - 1):
                            pkg = pkg.rsplit(".", 1)[0]
                        if node.module:
                            names = [f"{pkg}.{node.module}"]
                        else:
                            names = [f"{pkg}.{a.name}" for a in node.names]
                for name in names:
                    if name.startswith("biosuite.core."):
                        imported[layer].add(
                            name[len("biosuite.core."):].replace(".", "/")
                        )
    return imported


INTERFACE_IMPORTS = _interface_imports()


@pytest.mark.parametrize("capability", caps.list_capabilities(), ids=lambda c: c.id)
def test_every_entry_has_a_valid_interface_binding(capability):
    assert "python" in capability.interfaces
    assert set(capability.interfaces) <= LAYER_NAMES
    assert len(capability.interfaces) == len(set(capability.interfaces))
    assert list(capability.interfaces) == sorted(capability.interfaces)
    for layer in set(capability.interfaces) - {"python"}:
        assert capability.module in INTERFACE_IMPORTS[layer], (
            f"{capability.id}: no {layer} file imports biosuite.core.{capability.module}"
        )
    for layer, modules in INTERFACE_IMPORTS.items():
        if capability.module in modules:
            assert layer in capability.interfaces, (
                f"{capability.id}: {layer} imports the module but binding omits it"
            )


@pytest.mark.parametrize("capability", caps.list_capabilities(), ids=lambda c: c.id)
def test_every_entry_has_a_non_empty_reviewed_summary(capability):
    assert capability.summary
    assert not capability.summary.startswith("TODO")


def test_get_capability_round_trip():
    cap = caps.get_capability("sequence.gc_content")
    assert cap.mcp_exposed is True
    assert cap.module == "sequence"
    assert cap.function == "gc_content"


def test_get_capability_unknown_id_raises_key_error():
    with pytest.raises(KeyError, match="unknown capability id"):
        caps.get_capability("sequence.does_not_exist")


def test_get_capability_unknown_or_malformed_ids_raise_key_error():
    with pytest.raises(KeyError, match="unknown capability id"):
        caps.get_capability("sequence.does_not_exist")
    with pytest.raises(KeyError, match="unknown capability id"):
        caps.get_capability("not-a-catalog-entry")


def test_invalid_catalog_definitions_raise_value_error():
    from biosuite.capabilities import catalog

    with pytest.raises(ValueError, match="malformed capability id"):
        catalog._split_id("not-a-catalog-entry")
    with pytest.raises(ValueError, match="duplicate capability id"):
        catalog._index([("m.f", "s", False, (), (), ()), ("m.f", "t", False, (), (), ())])


def test_get_capability_rejects_non_string_ids():
    with pytest.raises(TypeError, match="must be a string"):
        caps.get_capability(341)


def test_returned_structures_cannot_mutate_the_catalog():
    first = caps.list_capabilities()
    with pytest.raises(Exception):
        first[0].mcp_exposed = True  # frozen dataclass: mutation must fail
    again = caps.list_capabilities()
    assert all(not c.mcp_exposed for c in again if c.id not in MCP_EXPOSED_IDS)


@pytest.mark.parametrize("implementation", [None, "missing_separator"])
def test_capability_model_rejects_malformed_references(implementation):
    from biosuite.capabilities.models import Capability

    with pytest.raises(ValueError, match="implementation must be"):
        Capability(id="m.f", module="m", function="f", summary="s",
                   implementation=implementation)


def test_interface_set_is_immutable():
    cap = caps.get_capability("sequence.gc_content")
    assert cap.interface_set == frozenset(cap.interfaces)


def test_inventory_matches_independent_ast_scan():
    expected = {
        f"{module}.{function}"
        for module in _expected_module_names()
        for function in _core_public_functions(module)
    }
    assert len(expected) == EXPECTED_ENTRIES
    assert {cap.id for cap in caps.list_capabilities()} == expected


def test_capability_model_rejects_bad_definitions():
    from biosuite.capabilities.models import Capability

    with pytest.raises(ValueError, match="non-empty string"):
        Capability(id="", module="m", function="f", summary="s", implementation="m:f")
    with pytest.raises(ValueError, match="implementation"):
        Capability(id="m.f", module="m", function="f", summary="s", implementation="other:fn")
    with pytest.raises(ValueError, match="does not match"):
        Capability(id="m.f", module="m", function="f", summary="s", implementation="m:g")
    with pytest.raises(ValueError, match="summary"):
        Capability(id="m.f", module="m", function="f", summary="", implementation="m:f")
