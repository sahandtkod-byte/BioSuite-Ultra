"""BioSuite capability catalog — stdlib-only public surface.

Importing this package pulls in nothing beyond the Python standard library
and never imports, parses or executes any ``biosuite.core`` implementation.
It exists so agents and later interfaces can answer "what can BioSuite do,
and can it run here?" without booting the heavy stack.
"""
from __future__ import annotations

from .catalog import CATALOG_VERSION, get_capability, list_capabilities
from .discovery import Availability, PackageStatus, ToolStatus, discover_availability
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    legacy_modules_view,
    manifest_json,
)
from .models import Capability
from .mcp import mcp_tools

__all__ = [
    "Availability",
    "Capability",
    "CATALOG_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "PackageStatus",
    "ToolStatus",
    "build_manifest",
    "discover_availability",
    "get_capability",
    "legacy_modules_view",
    "list_capabilities",
    "manifest_json",
    "mcp_tools",
]
