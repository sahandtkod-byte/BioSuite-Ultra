# ADR-002: Use Typer for CLI

## Status
Accepted

## Context
BioSuite needs a modern CLI with subcommands, auto-completion, and rich output. Options: argparse, click, typer.

## Decision
Use **Typer** (built on Click) for the CLI.

## Rationale
- Auto-generated help from type hints
- Shell completion (bash/zsh/fish/powershell) out of the box
- Subcommand groups map cleanly to analysis modules
- Rich integration for colored output and tables
- Minimal boilerplate vs raw Click

## Consequences
- Backward-compatible menu-based CLI retained as legacy
- Typer is a thin wrapper; underlying Click limitations still apply
