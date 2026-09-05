"""GUI module for BioSuite.

The GUI needs ``tkinter``/``customtkinter``, which are frequently absent on
headless machines (CI runners, servers, slim containers).  Importing this
package therefore must NOT pull those in eagerly: ``biosuite.gui.text_parsing``
and other headless helpers stay importable, and the ImportError for a real GUI
launch is raised at the point of use with an actionable message.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .main_window import BioSuiteApp

__all__ = ["BioSuiteApp"]


def __getattr__(name: str) -> Any:
    """Import GUI symbols lazily (PEP 562)."""
    if name == "BioSuiteApp":
        try:
            from .main_window import BioSuiteApp
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "The BioSuite GUI requires tkinter and customtkinter. "
                "Install the Tk bindings for your Python (e.g. "
                "`apt-get install python3-tk`) and `pip install customtkinter`. "
                f"Original error: {exc}"
            ) from exc
        return BioSuiteApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
