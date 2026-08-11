# Changelog

All notable changes to BioSuite Ultra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.5] - 2026-08-09

### Fixed
- Reverted from the unstable v5.0.1 line back to the stable 4.2.x series
- All 41 registered plotting entries verified working end-to-end
- Version strings unified across `pyproject.toml`, `biosuite/__init__.py`, the REST API,
  the CLI banner, the GUI labels, `CITATION.cff`, `docs/conf.py` and the README
- `tests/test_api.py` no longer depends on a hard-coded absolute path, and reads the
  API key from the `BIOSUITE_API_KEY` environment variable
- Release archives (`*.zip`) are now distributed as GitHub Release assets instead of
  being committed into the repository

## [4.2.4] - 2026-07-22

### Fixed
- Missing `tkinter` import in `biosuite/gui/tabs/visualization.py`

## [4.2.3] - 2026-07-22

### Changed
- Plots now open in a dedicated window with a "Save As..." action instead of blocking
  the tkinter event loop via `plt.show()`

## [4.2.2] - 2026-07-22

### Fixed
- Deep bug-fix pass across the plotting and GUI layers

## [4.2.1] - 2026-07-21

### Fixed
- Heatmap and other plotting crashes

## [4.2.0] - 2026-07-21

### Changed
- General bug fixes and stability work

## [4.1.1] - 2026-07-09

### Added
- JOSS paper and bibliography for academic submission
- Zenodo DOI for citation
- Docker and Binder quick start instructions
- Comprehensive test suite
- CI/CD pipeline with GitHub Actions
- Extended validation against BioPython and R
- Supplementary File S1 for journal submission

### Changed
- Updated `CITATION.cff` with correct module counts
- Updated benchmark results on real hardware (i5-4590)
- Improved README with badges and quick start

### Fixed
- Corrected module counts in documentation
- Fixed dependency claims (external bioinformatics tools)
- Updated machine specifications for benchmarks

## [4.1.0] - 2026-07-07

### Added
- Dual-Mode architecture (external tools + pure Python fallback)
- 47 analysis modules
- 123 visualization functions
- GUI with 11 tabs
- CLI with 117 menu options
- REST API with 40 endpoints
- BioSuite Ultra publication paper

### Changed
- Renamed packages for PyPI compatibility
- Updated all documentation

## [4.0.0] - 2026-06-15

### Added
- Initial release of BioSuite Ultra
- Complete bioinformatics platform
- Multi-omics analysis support
