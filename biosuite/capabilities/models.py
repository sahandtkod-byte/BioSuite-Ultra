"""Typed models for the BioSuite capability catalog (stdlib only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Tuple

__all__ = ["Capability"]


@dataclass(frozen=True)
class Capability:
    """One catalogued public function of ``biosuite.core``.

    ``implementation`` is a ``"module:function"`` reference string.  Building a
    ``Capability`` never imports or executes that target; the string is
    resolved only by callers that explicitly choose to.
    """

    id: str
    module: str
    function: str
    summary: str
    implementation: str
    mcp_exposed: bool = False
    interfaces: Tuple[str, ...] = ()
    requires_tools: Tuple[str, ...] = ()
    requires_packages: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("capability id must be a non-empty string")
        if not isinstance(self.implementation, str) or ":" not in self.implementation:
            raise ValueError(
                f"capability {self.id!r}: implementation must be 'module:function'"
            )
        if self.implementation != f"{self.module}:{self.function}":
            raise ValueError(
                f"capability {self.id!r}: implementation "
                f"{self.implementation!r} does not match module/function"
            )
        if not isinstance(self.summary, str) or not self.summary:
            raise ValueError(f"capability {self.id!r}: summary must be a non-empty string")
        object.__setattr__(self, "interfaces", tuple(self.interfaces))
        object.__setattr__(self, "requires_tools", tuple(self.requires_tools))
        object.__setattr__(self, "requires_packages", tuple(self.requires_packages))

    @property
    def interface_set(self) -> FrozenSet[str]:
        return frozenset(self.interfaces)
