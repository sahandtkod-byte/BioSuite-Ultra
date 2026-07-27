# ADR-005: Plugin System Architecture

## Status
Accepted

## Context
BioSuite needs extensibility without modifying core code — custom analysis modules, third-party tool wrappers, community contributions.

## Decision
Use a registry-based plugin system with `biosuite.core.plugin` as the entry point.

## Rationale
- Plugins register via `@register_plugin("name")` decorator
- Plugin discovery: scan `biosuite_plugins/` directory + entry points
- Each plugin defines: name, version, description, dependencies, commands
- CLI and API auto-discover registered plugins
- No circular imports — plugins import from core, not vice versa

## Consequences
- Plugins must declare optional dependencies explicitly
- Plugin conflicts detected at registration time
- Backward-compatible: existing modules remain in `biosuite.core.*`
