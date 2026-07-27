# ADR-004: Lazy Imports for Heavy Dependencies

## Status
Accepted

## Context
`import biosuite.core.sequence` takes 10+ seconds because numpy, pandas, scipy, and biopython are imported eagerly at module level. CLI startup needs to be <0.5s.

## Decision
Move heavy dependency imports (`numpy`, `pandas`, `scipy`, `matplotlib`, `Bio`) into function bodies or `TYPE_CHECKING` blocks.

## Rationale
- `import biosuite` (top-level) stays instant (<10ms)
- Heavy deps loaded on-demand when analysis functions are called
- `TYPE_CHECKING` guard preserves IDE autocompletion without runtime cost
- `core/__init__.py` no longer wildcard-imports all submodules

## Consequences
- First call to `gc_content()` incurs numpy import (~0.3s); subsequent calls are free
- Type checkers (mypy, pyright) still see full type information
- No change to public API — all functions remain importable
