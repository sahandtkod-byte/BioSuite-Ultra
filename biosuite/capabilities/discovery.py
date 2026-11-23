"""Availability discovery for catalogued capabilities (stdlib only).

Discovery answers two questions per capability without importing or running
any ``biosuite.core`` implementation:

* which interface layers (``api``, ``cli``, ``gui``, ``notebook``) statically
  import the capability's home module — recorded in the catalog at review
  time, not re-derived at runtime;
* whether the optional external tools and third-party packages the module
  probes for are present right now, checked with :func:`shutil.which` and
  :func:`importlib.util.find_spec`.
"""
from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = ["ToolStatus", "PackageStatus", "Availability", "discover_availability"]


@dataclass(frozen=True)
class ToolStatus:
    name: str
    available: bool


@dataclass(frozen=True)
class PackageStatus:
    name: str
    available: bool


@dataclass(frozen=True)
class Availability:
    capability_id: str
    interfaces: Tuple[str, ...]
    tools: Tuple[ToolStatus, ...]
    packages: Tuple[PackageStatus, ...]

    @property
    def ready(self) -> bool:
        return all(t.available for t in self.tools) and all(
            p.available for p in self.packages
        )


def _probe_tool(name: str) -> ToolStatus:
    return ToolStatus(name=name, available=shutil.which(name) is not None)


def _probe_package(name: str) -> PackageStatus:
    try:
        available = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        available = False
    return PackageStatus(name=name, available=available)


def discover_availability(capabilities=None) -> Dict[str, Availability]:
    """Return availability for the full catalog or a supplied iterable."""
    if capabilities is None:
        from .catalog import list_capabilities

        capabilities = list_capabilities()
    by_tools: Dict[Tuple[str, ...], Tuple[ToolStatus, ...]] = {}
    by_pkgs: Dict[Tuple[str, ...], Tuple[PackageStatus, ...]] = {}
    records: Dict[str, Availability] = {}
    for cap in capabilities:
        tools = by_tools.get(cap.requires_tools)
        if tools is None:
            tools = tuple(_probe_tool(t) for t in cap.requires_tools)
            by_tools[cap.requires_tools] = tools
        packages = by_pkgs.get(cap.requires_packages)
        if packages is None:
            packages = tuple(_probe_package(p) for p in cap.requires_packages)
            by_pkgs[cap.requires_packages] = packages
        records[cap.id] = Availability(
            capability_id=cap.id,
            interfaces=tuple(cap.interfaces),
            tools=tools,
            packages=packages,
        )
    return records
