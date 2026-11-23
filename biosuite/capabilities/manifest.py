"""Deterministic manifest generation for the capability catalog (stdlib only).

The manifest is the stable, machine-readable view later workstreams serve over
HTTP and package as a static artifact.  It must be byte-stable for the same
catalog: entries stay in catalog order, mappings are sorted, and JSON is
emitted with fixed separators and a trailing newline.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .mcp import mcp_tools

__all__ = [
    "build_manifest",
    "manifest_json",
    "legacy_modules_view",
    "MANIFEST_SCHEMA_VERSION",
]

MANIFEST_SCHEMA_VERSION = "1.0.0"

_LEGACY_MODULES_VIEW = (
    {"name": "sequence", "endpoints": ["/api/v1/sequence/*"], "description": "Sequence analysis"},
    {"name": "alignment", "endpoints": ["/api/v1/alignment/*"], "description": "Pairwise alignment"},
    {"name": "expression", "endpoints": ["/api/v1/expression/*"], "description": "Differential expression"},
    {"name": "plotting", "endpoints": ["/api/v1/plotting/*"], "description": "Visualization"},
    {"name": "crispr", "endpoints": ["/api/v1/crispr/*"], "description": "CRISPR guide design"},
    {"name": "metagenomics", "endpoints": ["/api/v1/metagenomics/*"], "description": "Taxonomic classification"},
    {"name": "epitope", "endpoints": ["/api/v1/epitope/*"], "description": "Epitope prediction"},
    {"name": "gwas", "endpoints": ["/api/v1/gwas/*"], "description": "GWAS analysis"},
    {"name": "phylogeny", "endpoints": ["/api/v1/phylogeny/*"], "description": "Phylogenetic analysis"},
    {"name": "popgen", "endpoints": ["/api/v1/popgen/*"], "description": "Population genetics"},
)


def build_manifest(capabilities=None, *, version: str | None = None) -> Dict[str, Any]:
    """Build a manifest, defaulting to the full catalog and package version."""
    if capabilities is None:
        from .catalog import list_capabilities

        capabilities = list_capabilities()
    if version is None:
        from biosuite import __version__

        version = __version__
    capabilities = tuple(capabilities)
    entries: List[Dict[str, Any]] = []
    for cap in capabilities:
        entries.append(
            {
                "id": cap.id,
                "module": cap.module,
                "function": cap.function,
                "summary": cap.summary,
                "implementation": cap.implementation,
                "mcp_exposed": cap.mcp_exposed,
                "interfaces": list(cap.interfaces),
                "requires": {
                    "tools": list(cap.requires_tools),
                    "packages": list(cap.requires_packages),
                },
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "biosuite_version": version,
        "total": len(entries),
        "mcp_exposed_total": sum(1 for e in entries if e["mcp_exposed"]),
        "capabilities": entries,
        "mcp_tools": mcp_tools(capabilities),
    }


def manifest_json(manifest: Dict[str, Any]) -> str:
    """Serialize a manifest deterministically (sorted keys, fixed separators)."""
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def legacy_modules_view(*, total_endpoints: int) -> Dict[str, Any]:
    """The ten-entry discovery view ``GET /api/v1/modules`` serves today.

    ``total_endpoints`` is supplied by the caller because the live endpoint
    derives it from the running application's route table; this package stays
    decoupled from any HTTP framework.  Returns a fresh mapping each call so
    callers cannot mutate the shared view.
    """
    return {
        "modules": [{**entry, "endpoints": list(entry["endpoints"])} for entry in _LEGACY_MODULES_VIEW],
        "total_endpoints": total_endpoints,
        "documentation": "/docs",
    }
