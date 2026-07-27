# ADR-001: Use PyQt6 for GUI Framework

## Status
Accepted

## Context
BioSuite needs a desktop GUI that supports matplotlib embedding, custom widgets, and tabbed interface. Options considered: Tkinter, PyQt6, PySide6, DearPyGui.

## Decision
Use **PyQt6** as the primary GUI framework.

## Rationale
- **PyQt6** provides native look on Windows/macOS/Linux
- Mature matplotlib integration via `FigureCanvasQTAgg`
- `QThreadPool`/`QRunnable` for background analysis
- Rich widget set (docking, tab bars, tree views)
- Qt Designer available for rapid prototyping

## Consequences
- GPL-licensed (commercial use requires license) — acceptable for open-source
- Heavy dependency (~150MB) — moved to `[gui]` optional extra
- PySide6 is the LGPL alternative but has less community tooling
