# Changelog

All notable changes to BioSuite Ultra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.1.0] - 2026-07-07

### Added
- **Parallel Processing Module** (`bioplatter/core/parallel.py`):
  - `parallel_map()` — Apply functions to items in parallel (CPU or I/O bound)
  - `parallel_submit()` — Submit heterogeneous tasks in parallel
  - `ParallelBatchProcessor` — Process large datasets in batches with progress tracking
  - Convenience functions: `parallel_gc_content()`, `parallel_reverse_complement()`, `parallel_translate()`, `parallel_align_pairs()`
  - Auto-detection of optimal worker count based on system resources
  - Graceful fallback to sequential processing on failure

- **Expanded Restriction Enzyme Database** (`bioplatter/core/utils.py`):
  - From 18 to 100+ Type II restriction enzymes
  - Comprehensive coverage of enzymes used in molecular biology labs worldwide
  - Includes: AatII, Acc65I, AflII, AflIII, AgeI, AluI, ApaLI, ApoI, AscI, AvaI, AvaII, AvrII, BaeI, BanI, BanII, BbsI, BcgI, BciVI, BclI, BfaI, BfuAI, BglI, BglII, BmtI, BsaI, BsaHI, BsaWI, BseRI, BsgI, BsiWI, BslI, BsmAI, BsmFI, BsoBI, Bsp1286I, BspEI, BspHI, BspMI, BsrFI, BsrGI, BstBI, BstEI, BstNI, BstUI, BstXI, BstZ17I, BsuRI, BtgI, BtgZI, BtsI, BveI, Cac8I, CviAII, CviJI, DdeI, DpnI, DraI, DraIII, DrdI, EaeI, EagI, EarI, EciI, EcoNI, EcoO109I, EcoT22I, FatI, FauI, Fnu4HI, FokI, FseI, FspI, HaeII, HaeIII, HgaI, HhaI, HinfI, HpaI, HpaII, Hpy188I, Hpy188III, HpyAV, HpyCH4III, HpyCH4IV, KasI, MboI, MboII, MfeI, MluCI, MmeI, MnlI, MseI, MslI, MspA1I, MspI, MwoI, NaeI, NarI, NdeI, NgoMIV, NlaIII, NlaIV, NruI, NsiI, NspI, NspBIII, PacI, PaeR7I, PciI, PmeI, PmlI, PpuMI, PvuI, PvuII, RsaI, RsrII, SacII, SapI, Sau3AI, Sau96I, SbfI, ScaI, ScrFI, SexAI, SfaNI, SfiI, SfoI, SgrAI, SmlI, SnaBI, SplI, SrfI, Sse8387I, SspI, StuI, StyI, StyD4I, TaiI, TaqI, TfiI, TseI, Tsp45I, Tsp509I, TsrI, Tth111I, XcmI, XmaCI, XmnI, ZraI

### Improved
- **Bayesian Phylogeny** (`bioplatter/core/bayesian_phylogeny.py`):
  - Implemented proper MCMC tree sampler with Jukes-Cantor substitution model
  - Real log-likelihood calculations using substitution matrices
  - Proper burn-in and thinning for MCMC chains
  - Actual ESS (Effective Sample Size) computation from autocorrelation
  - Better convergence diagnostics (PSRF, log-likelihood trace)

- **MD Simulation** (`bioplatter/core/md_simulation.py`):
  - Improved Lennard-Jones potential with proper cutoff radius
  - Velocity Verlet integrator (replaces Euler method for better energy conservation)
  - Berendsen thermostat for temperature coupling
  - Energy minimization step (steepest descent)
  - Better PDB file parsing with residue handling
  - Proper radius of gyration calculation
  - Energy decomposition (kinetic, potential, total)

- **Molecular Cloning** (`bioplatter/core/cloning.py`):
  - Fixed restriction digestion cut position calculations
  - Improved PCR simulation with proper annealing temperature
  - Better ligation efficiency modeling
  - Enhanced virtual gel electrophoresis edge case handling
  - Added insert verification after ligation

- **Database Searches** (`bioplatter/core/databases.py`):
  - Added retry logic with exponential backoff for HTTP requests
  - Improved rate limiting implementation
  - Better cache invalidation with TTL support
  - Proper timeout handling for all network requests
  - Connection error handling and graceful degradation

- **Expression Analysis** (`bioplatter/core/expression.py`):
  - Fixed statistical calculation edge cases with zero counts
  - Improved multiple testing correction (BH, Bonferroni, Holm)
  - Better DESeq2-style median-of-ratios normalization
  - Added variance stabilization transform (VST)
  - Added fold change calculation with log2 transformation

### Fixed
- Fixed `read_fasta` returning `None` instead of empty list for empty files
- Fixed `translate` not handling ambiguous codons correctly
- Fixed `needleman_wunsch` traceback returning incorrect alignment for certain gap patterns
- Fixed `smith_waterman` not finding the true maximum score cell
- Fixed `multiple_alignment` crashing with single-character sequences
- Fixed `bl` (truncated filename) — multiple alignment module initialization
- Fixed `metagenomics.py` classify_16s_rna missing abundance table calculation
- Fixed `enrichment.py` ORA not handling empty gene lists
- Fixed `popgen.py` FST calculation for single-population input
- Fixed `crispr.py` not warning when CRISPOR API is unavailable
- Fixed `epitope.py` B-cell prediction scoring above threshold for low-quality peptides
- Fixed `single_cell.py` scanpy import check returning wrong boolean
- Fixed `structure_prediction.py` pLDDT extraction for malformed PDB lines
- Fixed `docking.py` Vina scoring function using wrong units
- Fixed `survival.py` Kaplan-Meier estimator not handling censored data correctly
- Fixed `metabolomics.py` PCA returning wrong component count
- Fixed `metabolism.py` FBA solver not handling infeasible models
- Fixed `pathway_viz.py` KEGG pathway download failing silently
- Fixed `go_browser.py` OBO file parsing for missing definitions
- Fixed `file_formats.py` GenBank parser not handling multi-feature records
- Fixed `validators.py` retry_on_error not preserving function signature
- Fixed `log.py` ColorFormatter not handling Unicode characters
- Fixed `utils.py` load_dataframe_safe not supporting parquet format
- Fixed `utils.py` load_session returning stale data
- Fixed `utils.py` apply_glass_ax crashing on empty axes
- Fixed `run.py` --config not saving theme changes
- Fixed `run.py` --export-folder not creating output directory
- Fixed `api/__init__.py` CORS middleware not reading env var correctly
- Fixed `api/__init__.py` health endpoint returning wrong module count
- Fixed `docker-compose.yml` missing volume definitions
- Fixed `Dockerfile` not copying config files
- Fixed `pyproject.toml` missing optional dependencies in full extra
- Fixed `requirements.txt` missing version pins for critical packages

## [4.0.0] - 2026-07-01

### Added
- 52 analysis modules
- 36+ visualization types
- 29 GUI tabs
- 99+ CLI options
- 42 REST API endpoints
- 1,089 tests
- Docker support
- Plugin system
- Provenance tracking
- Jupyter integration

## [3.0.0] - 2026-06-01

### Added
- Molecular cloning toolkit
- Plasmid maps
- Virtual gel electrophoresis
- Sequence viewer
- CRISPR guide RNA design
- Epitope prediction
- GWAS analysis
- Metagenomics
- Machine learning

## [2.0.0] - 2026-05-01

### Added
- NGS analysis (BAM, VCF, Coverage)
- Variant calling
- Peak calling
- RNA-seq quantification
- Differential expression
- Multiple sequence alignment
- Phylogenetics

## [1.0.0] - 2026-04-01

### Added
- Initial release
- Sequence analysis
- Pairwise alignment
- GC content, reverse complement, translation
- Basic plotting
