# BioSuite Ultra — Guide for Agents

This document is for automated agents and scripts that drive BioSuite
Ultra locally. Everything here works offline; the tool never phones home.

## Discovery

- `biosuite manifest` prints the canonical deterministic capability manifest as
  JSON and exits 0. The bytes are identical across processes for the same
  catalog: no timestamps, host paths, package/platform versions or availability
  state. The `schema_version` field is a fixed constant (`"1.0.0"`) describing
  the manifest layout, not the package version.
- `biosuite doctor` reports what is installed: external executables are labelled
  `located` or `not located` (found on `PATH`), Python packages `importable` or
  `not importable`. It never launches an executable, so "located" does not prove
  a tool can run — only a real invocation can.
- Python: `from biosuite.capabilities.catalog import list_capabilities` and
  `from biosuite.capabilities.discovery import discover_availability`. The
  manifest lives in `biosuite.capabilities.manifest.manifest_json`.
- Scope note (6.5.0): structured input/output schemas and fallback metadata
  exist only for the curated MCP-exposed subset (the four `mcp_exposed`
  catalog entries, projected by `biosuite.capabilities.mcp`). The full catalog
  carries id, description, purity, surfaces and requirements only; complete
  I/O and fallback metadata for every capability is deferred to a later
  release, not silently absent.

## Exit codes for `manifest` and `doctor`

| Code | Meaning |
|------|---------|
| 0 | Catalog loaded; no required engine is missing |
| 1 | Catalog loaded but a required engine is missing (`doctor` only) |
| 3 | Catalog failed to load or the manifest could not be serialized |

Errors go to stderr; stdout carries output only on success. Missing *optional*
tools/packages are warnings and never change the exit code. Deployments that
require specific external engines can extend
`biosuite.cli.capabilities.REQUIRED_ENGINES`.

## Authentication and scopes

The local REST API (started with `biosuite api --port 8000`) uses API keys:
a legacy master key (full access) or, when `BIOSUITE_SCOPED_KEYS` is set,
scoped keys of the form `name:scope` with `read < execute < admin`. Scoped
keys are denied anything outside their scope (403); unknown keys get 401.
Admin JWT routes are separate and never satisfied by an API key.

## Jobs

Long analyses run as persistent local jobs over SQLite
(`.biosuite-jobs/jobs.sqlite3`): `POST /api/v1/jobs` (202),
`GET /api/v1/jobs/{id}`, `GET /api/v1/jobs/{id}/result`,
`POST /api/v1/jobs/{id}/cancel`. Jobs are owner-isolated; execution is
subject to a per-job time budget. The allowlist is intentionally small
(currently `sequence.gc_content`).

## Provenance

Every API job records how its result was produced (capability id, payload
digest, engine state) through the provenance bridge
(`biosuite/api/provenance.py`), and core refusals are recorded with the same
fidelity as successful runs.

## Limits

- No network access at runtime; all engines are local.
- Do not parse human-readable CLI output; use `manifest` for machine-readable
  discovery and the REST API for structured results.
- The interactive menu has 117 options; only the 22 documented subcommands are
  non-interactive.
- Manifest content is deterministic: do not add versions, timestamps or
  availability to it — those belong to `doctor`/`discover_availability`.

## Examples

```console
$ biosuite manifest | python -c "import json,sys; m=json.load(sys.stdin); print(m['total'], m['schema_version'])"
$ biosuite doctor; echo "exit=$?"     # 0 = ready, 1 = required engine missing
```
