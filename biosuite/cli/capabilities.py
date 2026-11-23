"""Non-interactive ``manifest`` and ``doctor`` subcommands (WS6).

Both commands answer questions about the capability catalog without running
any analysis, importing any optional dependency or launching any executable:

``biosuite manifest``
    Print the canonical deterministic JSON produced by
    :func:`biosuite.capabilities.manifest.manifest_json` to stdout and exit 0.
    The bytes are stable for the same catalog: no timestamps, host paths,
    versions, availability state, network access or analysis execution.

``biosuite doctor``
    Report availability for the catalog using
    :func:`biosuite.capabilities.discovery.discover_availability` only.
    Tools are labelled ``located`` (found on ``PATH``) or ``not located``;
    packages are labelled ``importable`` or ``not importable``.  Nothing here
    proves that a located tool can actually run — runnability is never
    claimed, so no executable is ever launched (in particular
    ``has_tool``/``--version`` probes are never used).  Missing *optional*
    tools and packages are warnings and never change the exit code.

Exit codes (the contract, also documented in ``docs/for_agents.md``):

* ``0`` — the catalog loaded and no required engine is missing;
* ``1`` — the catalog loaded but a required engine is missing;
* ``3`` — the catalog could not be loaded or the manifest could not be
  serialized (any :class:`Exception` from the capabilities layer).

Errors go to stderr; stdout is written only on success.
"""
from __future__ import annotations

import sys

__all__ = ["REQUIRED_ENGINES", "cli_manifest", "cli_doctor", "run_manifest", "run_doctor"]

#: External executables that must be present for this installation to be
#: considered usable.  Every catalogued tool requirement is per-capability
#: and therefore optional (reported as a warning by ``doctor``); the base
#: install has no hard external engine requirement, so this is empty by
#: default and a deployment may extend it.
REQUIRED_ENGINES: tuple = ()


def run_manifest():
    """Return ``(exit_code, stdout, stderr)`` for the manifest subcommand."""
    try:
        from ..capabilities.manifest import build_manifest, manifest_json
        return 0, manifest_json(build_manifest()), ""
    except Exception as exc:  # noqa: BLE001 — the CLI boundary reports, never traces
        return 3, "", f"manifest failed: {type(exc).__name__}: {exc}\n"


def run_doctor():
    """Return ``(exit_code, stdout, stderr)`` for the doctor subcommand."""
    try:
        from ..capabilities.catalog import list_capabilities
        from ..capabilities.discovery import discover_availability
    except Exception as exc:  # noqa: BLE001
        return 3, "", f"doctor failed: {type(exc).__name__}: {exc}\n"

    lines = []
    try:
        capabilities = list_capabilities()
        availability = discover_availability(capabilities)
    except Exception as exc:  # noqa: BLE001
        return 3, "", f"doctor failed: {type(exc).__name__}: {exc}\n"

    lines.append(f"catalog: {len(capabilities)} capabilities")

    # "located" is exactly what discovery reports: the name was found on PATH.
    # Runnability is deliberately never claimed because no executable is run.
    located = {}
    for record in availability.values():
        for tool in record.tools:
            located[tool.name] = tool.available
    missing_required = []
    for name in sorted(located):
        state = "located" if located[name] else "not located"
        lines.append(f"tool {name}: {state}")
        if name in REQUIRED_ENGINES and not located[name]:
            missing_required.append(name)
    for name in REQUIRED_ENGINES:
        if name not in located:
            missing_required.append(name)
            lines.append(f"tool {name}: not located")

    package_states = {}
    for record in availability.values():
        for package in record.packages:
            package_states[package.name] = package.available
    for name in sorted(package_states):
        state = "importable" if package_states[name] else "not importable"
        lines.append(f"package {name}: {state}")

    for name in sorted(set(n for n, ok in located.items() if not ok)
                       | set(n for n, ok in package_states.items() if not ok)):
        lines.append(f"warning: optional dependency {name} was not found "
                     "(capabilities that need it are unavailable; the exit code is unchanged)")

    if missing_required:
        for name in missing_required:
            lines.append(f"error: required engine {name} is missing")
        return 1, "\n".join(lines) + "\n", "missing required engine(s): " + ", ".join(missing_required) + "\n"
    return 0, "\n".join(lines) + "\n", ""


def cli_manifest():
    """Console entry point for ``biosuite manifest``."""
    code, out, err = run_manifest()
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    if code:
        raise SystemExit(code)
    return None


def cli_doctor():
    """Console entry point for ``biosuite doctor``."""
    code, out, err = run_doctor()
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    if code:
        raise SystemExit(code)
    return None
