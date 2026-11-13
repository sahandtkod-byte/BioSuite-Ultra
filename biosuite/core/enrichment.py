"""
Gene Ontology and pathway enrichment analysis.

Provides over-representation analysis (ORA) and gene set enrichment analysis
(GSEA) using goatools and gseapy libraries.

All tools are free and open source:
- goatools: https://github.com/tanghaibao/goatools
- gseapy: https://github.com/zqfang/GSEApy
"""
from dataclasses import dataclass, field
import math

import pandas as pd

from .log import get_logger

logger = get_logger(__name__)


# The two engines are optional, and *why* one is missing matters: the old guard
# lumped the statsmodels compatibility shim, goatools' own import and gseapy's into
# one try/except ImportError, so a half-installed dependency reported the engine as
# absent and told the user to install a package that was already there.
GOATOOLS_IMPORT_ERROR = ""
GSEAPY_IMPORT_ERROR = ""

try:
    # statsmodels >= 0.15 moved multipletests out of the legacy sandbox
    # path; goatools (<= 1.6.x) still imports it from there, so provide
    # the relocated symbol before goatools' multiple_testing module loads.
    # A failure here is recorded separately: it must not decide whether
    # goatools itself can be used.
    import statsmodels.sandbox.stats.multicomp as _sm_legacy
    if not hasattr(_sm_legacy, 'multipletests'):
        from statsmodels.stats.multitest import multipletests as _sm_new

        _sm_legacy.multipletests = _sm_new
except Exception as _shim_exc:      # noqa: BLE001 - the reason is reported, not swallowed
    logger.warning("[enrichment] the statsmodels sandbox shim did not apply: %s",
                   _shim_exc)

try:
    from goatools.go_enrichment import GOEnrichmentStudy
    from goatools.obo_parser import GODag
    HAS_GOATOOLS = True
except ImportError as _exc:
    HAS_GOATOOLS = False
    GOATOOLS_IMPORT_ERROR = str(_exc)
    logger.info("[enrichment] goatools is not usable: %s", _exc)

try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError as _exc:
    HAS_GSEAPY = False
    GSEAPY_IMPORT_ERROR = str(_exc)
    logger.info("[enrichment] gseapy is not usable: %s", _exc)


# Which correction `run_ora` reports, and the human label for it.  The attribute
# goatools names it under is `p_<method>`; reading `p_fdr_bh` first whatever the
# caller asked made the `method` argument a decoration.
GOATOOLS_CORRECTIONS = {'bonferroni': 'Bonferroni', 'fdr_bh': 'FDR'}

#: Every analysis type the module can actually run, and its backend.
ENGINE_BY_ANALYSIS = {'ORA': 'goatools', 'GSEA': 'gseapy', 'KEGG': 'gseapy'}

#: The gseapy calls take no caller-settable threshold, so the cutoff the analysis
#: counts and the cutoff the header prints come from the same number.
GSEA_FDR_THRESHOLD = 0.05
KEGG_CUTOFF = 0.05


def _missing_engine_text(analysis_type):
    """Advice that names the engine the analysis needs and why it is unusable.

    The old text said "not installed" for every import failure, including a
    dependency of the engine failing to load — which sent users to install a
    package they already had.
    """
    engine = ENGINE_BY_ANALYSIS.get(analysis_type)
    if engine == 'goatools':
        return ("goatools is not usable. Run: pip install goatools"
                + (f" (import failed: {GOATOOLS_IMPORT_ERROR})"
                   if GOATOOLS_IMPORT_ERROR else ""))
    return ("gseapy is not usable. Run: pip install gseapy"
            + (f" (import failed: {GSEAPY_IMPORT_ERROR})" if GSEAPY_IMPORT_ERROR else ""))


@dataclass
class EnrichmentResult:
    """Container for enrichment analysis results."""
    analysis_type: str  # 'ORA' or 'GSEA'
    term_name: str
    term_id: str
    p_value: float
    adjusted_p_value: float
    enrichment_score: float = 0.0
    gene_count: int = 0
    genes: list = field(default_factory=list)
    category: str = ""  # 'BP', 'CC', 'MF', 'KEGG', etc.
    description: str = ""


def _engine_for(analysis_type: str) -> str:
    """Which backend can actually run this analysis in this installation.

    Every name that was not 'ORA' used to answer gseapy, so a report built with a
    typo'd analysis type claimed a backend that cannot produce it.
    """
    engine = ENGINE_BY_ANALYSIS.get(analysis_type)
    if engine is None:
        return 'none'
    return engine if (HAS_GOATOOLS if engine == 'goatools' else HAS_GSEAPY) else 'none'


@dataclass
class EnrichmentReport:
    """Complete enrichment analysis report.

    ``engine`` separates "the backend is missing" from "nothing was enriched":
    an empty result with ``engine='none'`` is not a biological finding (N18).
    ``threshold`` is the cutoff the analysis actually applied and ``correction``
    names the number that cutoff was applied to; the text report used to print
    ``FDR < 0.05`` whatever the caller had passed.
    """
    analysis_type: str
    num_input_genes: int
    num_significant: int
    results: list = field(default_factory=list)
    background_count: int = 0
    message: str = ""
    engine: str = ""
    threshold: float = 0.05
    correction: str = "adjusted p"

    def __post_init__(self):
        if not self.engine:
            self.engine = _engine_for(self.analysis_type)

    def top_terms(self, n=20):
        """The first *n* terms, strongest first.

        ``n`` is a count, not a slice index: ``top_terms(-1)`` used to answer every
        term but the last, so the strongest-looking list was quietly short one.
        """
        if n is None:
            return list(self.results)
        if isinstance(n, bool) or not isinstance(n, int):
            raise ValueError(f"top_terms(n) must be an integer >= 0 or None, got {n!r}.")
        if n < 0:
            raise ValueError(f"top_terms(n) must be >= 0, got {n}.")
        return self.results[:n]

    def significant_terms(self, fdr_threshold=0.05):
        return [r for r in self.results if r.adjusted_p_value < fdr_threshold]


def check_enrichment_tools():
    """Check which enrichment tools are available."""
    return {
        'goatools': HAS_GOATOOLS,
        'gseapy': HAS_GSEAPY,
    }


# ── argument domains shared by the three analyses ────────────────────────────

def _gene_arguments(gene_list, name="gene_list"):
    """A concrete list of gene identifiers, or an error naming what arrived.

    A bare string used to be read as a list of one-character genes — `run_ora`
    reported the string's characters, found none of them in the database and said
    "No input genes found" — and a generator died outside every guard on `len()`.
    """
    if isinstance(gene_list, (str, bytes)):
        raise ValueError(
            f"{name} must be a list of gene identifiers, not one bare "
            f"{type(gene_list).__name__} ({gene_list[:20]!r}…). "
            f"Pass ['BRCA1', 'TP53'], not 'BRCA1TP53'.")
    try:
        genes = [str(gene) for gene in gene_list]
    except TypeError:
        raise ValueError(f"{name} must be an iterable of gene identifiers, "
                         f"got {gene_list!r}.") from None
    blank = [gene for gene in genes if not gene.strip()]
    if blank:
        raise ValueError(f"{name} contains {len(blank)} blank gene identifier(s).")
    return genes


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _probability(value, name="cutoff"):
    """A threshold in (0, 1]; outside that range the filter keeps all or nothing."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number in (0, 1], got {value!r}.") from None
    if not 0 < number <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value!r}.")
    return number


def _size_limits(min_size, max_size):
    min_size = _positive_int(min_size, "min_size")
    max_size = _positive_int(max_size, "max_size")
    if max_size < min_size:
        raise ValueError(f"max_size ({max_size}) must be >= min_size ({min_size}).")
    return min_size, max_size


def _engine_number(value, name, term=""):
    """A float from an engine table cell, NaN when the cell never held one.

    `np.isnan('0.001')` raises, so a string-typed column used to abort the whole
    run with ``GSEA error: ufunc 'isnan' not supported for the input types``, and a
    missing number used to be filled with 1.0 — a measurement nobody made.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("[enrichment] %s: %s is not a number (%r), reported as missing",
                       term or "engine row", name, value)
        return float('nan')


def _is_missing(number):
    return number is None or (isinstance(number, float) and math.isnan(number))


def _significance_key(result):
    """Sort key that keeps a missing adjusted p-value last, deterministically."""
    value = result.adjusted_p_value
    ordered = math.inf if _is_missing(value) else value
    return (ordered, result.term_id)


def run_ora(
    gene_list,
    organism='human',
    ontology='BP',
    obo_file=None,
    associations_file=None,
    method='bonferroni',
    cutoff=0.05
):
    """Over-Representation Analysis (ORA) for Gene Ontology terms.

    Tests whether genes in the input list are enriched for specific
    GO terms compared to a background gene set.

    Args:
        gene_list: List of gene identifiers.  The background this module builds
            comes from NCBI gene2go, so its keys are **Entrez gene IDs as text**;
            a symbol ('BRCA1') is not in that key space and is now reported as
            unmatched instead of vanishing.
        organism: Organism name ('human', 'mouse', 'yeast', etc.).
        ontology: GO subontology ('BP' for Biological Process,
                  'CC' for Cellular Component, 'MF' for Molecular Function).
        obo_file: Path to go-basic.obo file (downloaded if not provided).
        associations_file: Path to gene-GO associations file.
        method: The multiple-testing correction each term is reported under
                ('bonferroni' or 'fdr_bh'); it also names the column the
                significance threshold is applied to.
        cutoff: Significance threshold (default: 0.05).

    Returns:
        EnrichmentReport object.

    Raises:
        ValueError: when *gene_list* is one bare string, contains a blank
            identifier, when *method* names a correction this module cannot read
            or when *cutoff* is not a probability.  All three used to be answered
            with an analysis that had quietly tested something else.
    """
    genes = _gene_arguments(gene_list)
    if not isinstance(method, str) or not method.strip():
        raise ValueError(
            f"method must name a multiple-testing correction, got {method!r}; "
            f"the corrections this module labels are "
            f"{', '.join(sorted(GOATOOLS_CORRECTIONS))}.")
    method = method.strip()
    correction_attribute = f"p_{method}"
    correction_label = GOATOOLS_CORRECTIONS.get(method, method)
    cutoff = _probability(cutoff)

    if not genes:
        return EnrichmentReport(
            analysis_type='ORA', num_input_genes=0, num_significant=0,
            threshold=cutoff, correction=correction_label,
            message="Empty gene list provided.")

    if not HAS_GOATOOLS:
        return EnrichmentReport(
            analysis_type='ORA', num_input_genes=len(genes), num_significant=0,
            threshold=cutoff, correction=correction_label,
            message=_missing_engine_text('ORA'))

    notes = []

    try:
        # Load GO ontology
        if obo_file is None:
            obo_file = _download_go_obo()
            if obo_file is None:
                # A missing ontology is not a biological result.  It used to fall
                # through to GODag(None) and come back as "ORA analysis error:
                # expected string or bytes-like object, not 'NoneType'".
                return EnrichmentReport(
                    analysis_type='ORA', num_input_genes=len(genes),
                    num_significant=0, threshold=cutoff, correction=correction_label,
                    message=("No GO ontology available: go-basic.obo is not cached "
                             "and could not be downloaded. Pass obo_file=<path>."))

        obodag = GODag(obo_file)

        # Load associations.  The two sources are named differently, because
        # "Could not load gene associations for human." used to be the answer to
        # an unreadable file the caller had named themselves.
        if associations_file is None:
            associations = _get_associations(organism)
            problem = (f"no gene/GO-term associations are available for organism "
                       f"{organism!r}" if not associations else "")
        else:
            associations = _load_associations_file(associations_file)
            problem = (f"association file {associations_file!r} yielded no "
                       f"gene/GO-term rows" if not associations else "")
        if problem:
            return EnrichmentReport(
                analysis_type='ORA', num_input_genes=len(genes),
                num_significant=0, threshold=cutoff, correction=correction_label,
                message=f"Could not load gene associations: {problem}.")

        background = set(associations.keys())
        matched = set(genes) & background

        if not matched:
            return EnrichmentReport(
                analysis_type='ORA',
                num_input_genes=len(genes),
                num_significant=0,
                background_count=len(background),
                threshold=cutoff, correction=correction_label,
                message=(f"None of the {len(genes)} input identifier(s) is in the "
                         f"{len(background)}-gene background. That background is keyed "
                         f"by Entrez gene IDs (numbers as text), so a symbol such as "
                         f"{genes[0]!r} can never match."))
        unknown = len(set(genes)) - len(matched)
        if unknown:
            notes.append(f"{unknown} of {len(set(genes))} input identifier(s) are not "
                         f"in the {len(background)}-gene background and were left out")

        # Create GO enrichment study.  Request BOTH corrections: the
        # downstream filter reads r.p_fdr_bh, but a bonferroni-only run
        # (the DEFAULT!) produces records that lack p_fdr_bh, so every
        # default call used to die with AttributeError and surface as
        # "ORA analysis error" with zero results.
        goeaobj = GOEnrichmentStudy(
            background,
            associations,
            obodag,
            methods=sorted({method, 'bonferroni', 'fdr_bh'}),
            cutoff=cutoff
        )

        goea_results = goeaobj.run_study(matched)

        # Filter by significance and depth (depth ≥ 2 keeps it term-specific)
        results = []
        dropped_shallow = 0
        substituted = 0

        for r in goea_results:
            p_adj = getattr(r, correction_attribute, None)
            if p_adj is None:
                # The caller named one correction.  If the engine did not produce
                # it, the number reported is a substitution, and the report says
                # so instead of quietly reading whichever column came first.
                for fallback in ('p_fdr_bh', 'p_bonferroni'):
                    p_adj = getattr(r, fallback, None)
                    if p_adj is not None:
                        substituted += 1
                        break
            if p_adj is None or not p_adj < cutoff:
                continue
            # Check ontology category from GO term prefix
            go_id = r.GO
            if go_id in obodag:
                term_depth = obodag[go_id].depth
                if term_depth is None or term_depth < 2:
                    dropped_shallow += 1
                    continue

            enrichment = EnrichmentResult(
                analysis_type='ORA',
                term_name=r.name if hasattr(r, 'name') else '',
                term_id=r.GO,
                # r.p_value is the nominal (uncorrected) hypergeometric p that
                # goatools computed.  Reading 'p_bonferroni' here filled the
                # p_value slot with a SECOND correction and threw the nominal p
                # away, so the field disagreed with run_gsea (nominal 'NOM
                # p-val') and run_kegg_enrichment ('P-value') and no caller
                # could ever see the uncorrected number.
                p_value=getattr(r, 'p_value', p_adj),
                adjusted_p_value=p_adj,
                gene_count=r.study_count,
                genes=list(r.study_items) if hasattr(r, 'study_items') else [],
                category=ontology,
                description=r.name if hasattr(r, 'name') else ''
            )
            results.append(enrichment)

        # Ties break on the term id: the incoming order comes from a set
        # traversal in the upstream engine, so sorting on the p-value alone let
        # PYTHONHASHSEED decide how two equally significant terms were listed,
        # and one input produced a different report in each run.
        results.sort(key=_significance_key)

        if dropped_shallow:
            notes.append(f"{dropped_shallow} term(s) cleared the cutoff but were "
                         f"dropped for a GO depth below 2 (too general to act on)")
        if substituted:
            notes.append(f"{substituted} term(s) carried no {correction_attribute} "
                         f"from the engine and are reported under a fallback correction")

        return EnrichmentReport(
            analysis_type='ORA',
            num_input_genes=len(genes),
            num_significant=len(results),
            results=results,
            background_count=len(background),
            message="; ".join(notes),
            threshold=cutoff, correction=correction_label
        )

    except Exception as e:
        logger.exception("[enrichment] ORA failed")
        return EnrichmentReport(
            analysis_type='ORA',
            num_input_genes=len(genes),
            num_significant=0,
            threshold=cutoff, correction=correction_label,
            message=f"ORA analysis error: {str(e)}"
        )


def run_gsea(
    gene_list,
    gene_scores=None,
    organism='Human',
    gene_sets='GO_Biological_Process_2023',
    min_size=15,
    max_size=500,
    permutation_num=1000,
    seed=42
):
    """Gene Set Enrichment Analysis (GSEA) using gseapy.

    Unlike ORA, GSEA uses ranked gene lists with scores (e.g., fold changes)
    to detect coordinated changes in predefined gene sets.

    Args:
        gene_list: List of gene identifiers (ranked by score).
        gene_scores: Numeric scores for ranking (e.g., log2FC).
            If None, genes are ranked by position in list.
        organism: 'Human', 'Mouse', or 'Yeast'.
        gene_sets: Gene set database name (default: GO_Biological_Process_2023).
            Options include: KEGG_2021_Human, Reactome_2022, MSigDB_Hallmark_2020, etc.
        min_size: Minimum gene set size (default: 15).
        max_size: Maximum gene set size (default: 500).
        permutation_num: Number of permutations (default: 1000).
        seed: Random seed for reproducibility.

    Returns:
        EnrichmentReport object.

    Raises:
        ValueError: for a bare string gene list, a score list whose length
            disagrees with the gene list, or unusable size/permutation counts.  A
            length mismatch used to be answered with `GSEA error: Length of values
            (1) does not match length of index (3)`, which names neither argument.
    """
    genes = _gene_arguments(gene_list)
    min_size, max_size = _size_limits(min_size, max_size)
    permutation_num = _positive_int(permutation_num, "permutation_num")
    seed = _positive_int(seed, "seed", minimum=0)

    if gene_scores is not None:
        if isinstance(gene_scores, (str, bytes)):
            raise ValueError(f"gene_scores must be a list of numbers, got the string "
                             f"{gene_scores!r}.")
        scores = list(gene_scores)
        if len(scores) != len(genes):
            raise ValueError(f"gene_scores has {len(scores)} value(s) for "
                             f"{len(genes)} gene(s); one score per gene.")
    else:
        scores = None

    if not genes:
        return EnrichmentReport(
            analysis_type='GSEA', num_input_genes=0, num_significant=0,
            threshold=GSEA_FDR_THRESHOLD, message="Empty gene list provided.")

    if not HAS_GSEAPY:
        return EnrichmentReport(
            analysis_type='GSEA', num_input_genes=len(genes), num_significant=0,
            threshold=GSEA_FDR_THRESHOLD, message=_missing_engine_text('GSEA'))

    notes = []

    try:
        # Build ranked gene list
        if scores is not None:
            ranking = pd.Series(scores, index=genes).sort_values(ascending=False)
        else:
            ranking = pd.Series(range(len(genes), 0, -1), index=genes)

        # Run GSEA
        gsea_results = gp.prerank(
            rnk=ranking,
            gene_sets=gene_sets,
            organism=organism,
            min_size=min_size,
            max_size=max_size,
            permutation_num=permutation_num,
            seed=seed,
            no_plot=True,
            silent=True
        )

        if gsea_results is None or gsea_results.res2d is None:
            return EnrichmentReport(
                analysis_type='GSEA',
                num_input_genes=len(genes),
                num_significant=0,
                threshold=GSEA_FDR_THRESHOLD,
                message="GSEA returned no results."
            )

        df = gsea_results.res2d
        if 'Term' not in df.columns:
            # A table the module does not recognise used to become one row per
            # entry with an empty term id, a p-value of 1.0 and no message —
            # indistinguishable from an analysis that found nothing significant.
            return EnrichmentReport(
                analysis_type='GSEA',
                num_input_genes=len(genes),
                num_significant=0,
                threshold=GSEA_FDR_THRESHOLD,
                message=(f"GSEA returned a result table without a 'Term' column; "
                         f"it carries {', '.join(map(str, df.columns)) or 'no columns'}.")
            )
        missing_p = missing_fdr = geneless = 0
        results = []

        for _, row in df.iterrows():
            term = row.get('Term', '')
            nes = _engine_number(row.get('NES'), 'NES', term)
            pval = _engine_number(row.get('NOM p-val'), 'NOM p-val', term)
            fdr = _engine_number(row.get('FDR q-val'), 'FDR q-val', term)
            genes_str = row.get('Genes', '')
            gene_names = ([g.strip() for g in genes_str.split(';') if g.strip()]
                          if isinstance(genes_str, str) else [])

            if _is_missing(pval):
                missing_p += 1
            if _is_missing(fdr):
                missing_fdr += 1
            if not gene_names:
                geneless += 1

            enrichment = EnrichmentResult(
                analysis_type='GSEA',
                term_name=term,
                term_id=term,
                p_value=pval,
                adjusted_p_value=fdr,
                enrichment_score=nes,
                gene_count=len(gene_names),
                genes=gene_names,
                category=gene_sets
            )
            results.append(enrichment)

        results.sort(key=_significance_key)
        sig_count = sum(1 for r in results if r.adjusted_p_value < GSEA_FDR_THRESHOLD)

        if missing_p or missing_fdr:
            notes.append(f"{missing_p} term(s) had no 'NOM p-val' and {missing_fdr} "
                         f"no 'FDR q-val' from the engine; those cells are blank in "
                         f"the report and count as not significant")
        if geneless:
            notes.append(f"{geneless} term(s) carried no member genes "
                         f"(the engine's 'Genes' cell was empty)")

        return EnrichmentReport(
            analysis_type='GSEA',
            num_input_genes=len(genes),
            num_significant=sig_count,
            results=results,
            message="; ".join(notes),
            threshold=GSEA_FDR_THRESHOLD, correction='FDR'
        )

    except Exception as e:
        logger.exception("[enrichment] GSEA failed")
        return EnrichmentReport(
            analysis_type='GSEA',
            num_input_genes=len(genes),
            num_significant=0,
            threshold=GSEA_FDR_THRESHOLD, correction='FDR',
            message=f"GSEA error: {str(e)}"
        )


def run_kegg_enrichment(
    gene_list,
    organism='hsa',
    gene_sets=None,
    min_size=10,
    max_size=500
):
    """KEGG pathway enrichment analysis.

    Args:
        gene_list: List of Entrez gene IDs.
        organism: KEGG organism code ('hsa' for human, 'mmu' for mouse).
        gene_sets: Custom gene sets dict (default: a KEGG library name built
            from the organism code and handed to gseapy unchanged).
        min_size: Minimum pathway size.
        max_size: Maximum pathway size.

    Returns:
        EnrichmentReport object.

    Raises:
        ValueError: for a bare string gene list or unusable size limits.  An empty
            list is answered with a message, as ORA and GSEA already did; it used
            to reach the engine, which is the only one of the three that did.
    """
    genes = _gene_arguments(gene_list)
    min_size, max_size = _size_limits(min_size, max_size)

    if not genes:
        return EnrichmentReport(
            analysis_type='KEGG', num_input_genes=0, num_significant=0,
            threshold=KEGG_CUTOFF, message="Empty gene list provided.")

    if not HAS_GSEAPY:
        return EnrichmentReport(
            analysis_type='KEGG',
            num_input_genes=len(genes),
            num_significant=0,
            threshold=KEGG_CUTOFF,
            message=_missing_engine_text('KEGG')
        )

    notes = []

    try:
        # Use gseapy's KEGG module.  gene_sets / min_size / max_size were
        # advertised in the signature and docstring but never reached
        # gp.enrich(), so a caller supplying a custom GMT was analysed against
        # the built-in KEGG library anyway and the size limits gseapy defaults
        # to (10/500) silently replaced the requested ones.
        kegg_sets = gene_sets if gene_sets is not None else f'KEGG_{organism.upper()}_2021'
        enrichment = gp.enrich(
            gene_list=genes,
            gene_sets=kegg_sets,
            organism=organism,
            outdir=None,
            no_plot=True,
            cutoff=KEGG_CUTOFF,
            min_size=min_size,
            max_size=max_size
        )

        if enrichment is None:
            return EnrichmentReport(
                analysis_type='KEGG',
                num_input_genes=len(genes),
                num_significant=0,
                threshold=KEGG_CUTOFF,
                message="KEGG enrichment returned no results."
            )

        df = enrichment.res2d if hasattr(enrichment, 'res2d') else enrichment
        columns = set(df.columns)
        if not ({'Term', 'Description'} & columns) or 'P-value' not in columns:
            return EnrichmentReport(
                analysis_type='KEGG',
                num_input_genes=len(genes),
                num_significant=0,
                threshold=KEGG_CUTOFF,
                message=("KEGG enrichment returned a table without a 'Term' "
                         "(or 'Description') and 'P-value' column; it carries "
                         f"{', '.join(map(str, df.columns)) or 'no columns'}.")
            )
        unreadable_overlap = 0
        missing_adjusted = 0
        results = []

        for _, row in df.iterrows():
            term = row.get('Term', row.get('Description', ''))
            overlap = str(row.get('Overlap', ''))
            if '/' in overlap and overlap.split('/')[0].strip().isdigit():
                gene_count = int(overlap.split('/')[0].strip())
            else:
                # An Overlap the module cannot read is counted, not quietly
                # reported as a pathway holding zero genes.
                gene_count = 0
                unreadable_overlap += 1
            p_value = _engine_number(row.get('P-value'), 'P-value', term)
            adjusted_raw = row.get('Adjusted P-value', row.get('FDR'))
            if _is_missing(adjusted_raw):
                missing_adjusted += 1
            adjusted = _engine_number(adjusted_raw, 'Adjusted P-value', term)
            member_genes = [g.strip() for g in str(row.get('Genes', '')).split(';')
                            if g.strip()]

            results.append(EnrichmentResult(
                analysis_type='KEGG',
                term_name=term,
                term_id=row.get('Term', ''),
                p_value=p_value,
                adjusted_p_value=adjusted,
                gene_count=gene_count,
                genes=member_genes,
                description=row.get('Description', ''),
                category='KEGG'
            ))

        results.sort(key=_significance_key)

        if unreadable_overlap:
            notes.append(f"{unreadable_overlap} pathway(s) had an 'Overlap' cell the "
                         f"module could not read; their gene count is reported as 0")
        if missing_adjusted:
            notes.append(f"{missing_adjusted} pathway(s) carried no adjusted p-value "
                         f"from the engine")

        return EnrichmentReport(
            analysis_type='KEGG',
            num_input_genes=len(genes),
            num_significant=sum(1 for r in results if r.adjusted_p_value < KEGG_CUTOFF),
            results=results,
            message="; ".join(notes),
            threshold=KEGG_CUTOFF, correction='adjusted p'
        )

    except Exception as e:
        logger.exception("[enrichment] KEGG enrichment failed")
        return EnrichmentReport(
            analysis_type='KEGG',
            num_input_genes=len(genes),
            num_significant=0,
            threshold=KEGG_CUTOFF, correction='adjusted p',
            message=f"KEGG enrichment error: {str(e)}"
        )


def _format_stat(number):
    """A right-aligned statistic, or an explicit blank for a missing one."""
    if _is_missing(number):
        return 'n/a'.rjust(10)
    return f"{number:>10.2e}"


def format_enrichment_report(report, max_terms=30):
    """Format EnrichmentReport as readable string.

    Args:
        report: EnrichmentReport object.
        max_terms: Maximum terms to display (>= 0).

    Returns:
        Formatted string.

    The header names the correction and the cutoff the analysis actually applied,
    and the trailing count is the number of result rows the table left out — not
    `num_significant - max_terms`, which mixed two different populations: a GSEA
    report with 200 rows, 5 shown and 3 significant used to claim nothing had been
    omitted, while the same arithmetic with 100 significant claimed 95.
    """
    if report is None:
        return "No enrichment results available."
    if isinstance(max_terms, bool) or not isinstance(max_terms, int):
        raise ValueError(f"max_terms must be an integer >= 0, got {max_terms!r}.")
    if max_terms < 0:
        raise ValueError(f"max_terms must be >= 0, got {max_terms}.")

    lines = [
        f"=== {report.analysis_type} Enrichment Analysis ===",
        f"Input genes: {report.num_input_genes}",
        f"Significant terms ({report.correction} < {report.threshold:g}): "
        f"{report.num_significant}",
        ""
    ]

    if report.message:
        lines.append(f"Note: {report.message}")

    if not report.results:
        if not report.message:
            lines.append("No significant enriched terms found.")
        return '\n'.join(lines)

    lines.append(f"{'#':<4} {'ID':<12} {'Term':<38} {'Adj p':>10} {'NES':>8} {'Genes':>6}")
    lines.append("-" * 88)

    shown = report.top_terms(max_terms)
    for i, r in enumerate(shown):
        # The term ID is the action item for citations/queries — the old
        # report displayed only the human-readable name.
        term_id = (r.term_id or '')[:12]
        term_display = r.term_name[:35] if r.term_name else (r.term_id or '')[:35]
        # NES is a GSEA statistic.  For ORA and KEGG there is no such number, and
        # a GSEA term whose NES the engine never filled is printed as a blank,
        # not as the literal "nan".
        if report.analysis_type != 'GSEA':
            nes_str = "N/A"
        else:
            nes_str = "n/a" if _is_missing(r.enrichment_score) else f"{r.enrichment_score:.2f}"
        lines.append(
            f"{i+1:<4} {term_id:<12} {term_display:<38} "
            f"{_format_stat(r.adjusted_p_value)} "
            f"{nes_str:>8} "
            f"{r.gene_count:>6}"
        )

    omitted = len(report.results) - len(shown)
    if omitted > 0:
        lines.append(f"\n... and {omitted} more term(s) not shown.")

    return '\n'.join(lines)


def _download_go_obo():
    """Download go-basic.obo into the ``~/.biosuite`` cache, or return None.

    A failure is a warning, not an info line: every caller that needs the
    ontology stops here, and it used to be the least visible line in the run.
    """
    import os
    obo_path = os.path.join(os.path.expanduser('~'), '.biosuite', 'go-basic.obo')
    if os.path.exists(obo_path):
        return obo_path

    os.makedirs(os.path.dirname(obo_path), exist_ok=True)
    try:
        import urllib.request
        url = 'http://purl.obolibrary.org/obo/go-basic.obo'
        req = urllib.request.urlopen(url, timeout=60)
        body = req.read()
    except Exception as e:
        logger.warning("Could not download the GO ontology from %s: %s",
                       'http://purl.obolibrary.org/obo/go-basic.obo', e)
        return None

    try:
        with open(obo_path, 'wb') as f:
            f.write(body)
    except OSError as e:
        logger.warning("Downloaded the GO ontology but could not write %s: %s",
                        obo_path, e)
        return None
    return obo_path


#: NCBI taxonomy ids for the organisms `_get_associations` can build a background
#: for.  The names are matched case-insensitively.
ORGANISM_TAXID = {
    'human': 9606, 'homo sapiens': 9606, '9606': 9606,
    'mouse': 10090, 'mus musculus': 10090, '10090': 10090,
    'yeast': 4932, 'saccharomyces cerevisiae': 4932, '4932': 4932,
}


def _get_associations(organism):
    """Get gene-GO associations for an organism from the cached NCBI gene2go file.

    gene2go is downloaded into ``~/.biosuite/`` on first use and refreshed after
    30 days.  The module used to download NCBI's ``gene_info.gz`` beside it —
    per organism, on the same 30-day cycle — and never open the file: every
    association here comes from gene2go's own columns, so that second download was
    pure transfer cost on a metered connection.

    Supported organisms: 'human' (taxid 9606), 'mouse' (taxid 10090),
    'yeast' (taxid 4932).

    Returns:
        Dict mapping Entrez gene id (str) -> set of GO term IDs (str), or an
        empty dict.  The reason an answer is empty is logged: an unknown organism,
        a file that could not be fetched and a file with no rows for the taxid are
        three different problems that used to look identical to the caller.
    """
    import gzip
    import os

    taxid = ORGANISM_TAXID.get(str(organism).lower())
    if taxid is None:
        logger.error("Unknown organism %r. Supported: %s.", organism,
                     ", ".join(sorted(set(ORGANISM_TAXID))))
        return {}

    cache_dir = os.path.join(os.path.expanduser('~'), '.biosuite')
    os.makedirs(cache_dir, exist_ok=True)
    gene2go_path = os.path.join(cache_dir, 'gene2go.gz')

    if not _download_if_needed(
            gene2go_path,
            'ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz'):
        logger.error("No readable gene2go cache at %s for organism %r.",
                     gene2go_path, organism)
        return {}

    # Parse gene2go: filter to our taxid
    associations = {}
    unreadable = 0
    try:
        opener = gzip.open if gene2go_path.endswith('.gz') else open
        with opener(gene2go_path, 'rt') as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 4:
                    unreadable += 1
                    continue
                try:
                    file_taxid = int(parts[0])
                except ValueError:
                    unreadable += 1
                    continue
                if file_taxid != taxid:
                    continue
                associations.setdefault(parts[1], set()).add(parts[2])
    except (OSError, EOFError, UnicodeDecodeError) as e:
        logger.error("Could not read the gene2go cache %s: %s", gene2go_path, e)
        return {}

    if unreadable:
        logger.warning("%s: skipped %d gene2go row(s) that were not a usable "
                       "taxid/gene/term record", gene2go_path, unreadable)
    if not associations:
        logger.warning("No GO associations found for taxid %d in %s.", taxid,
                       gene2go_path)

    return associations


def _download_if_needed(filepath, url, max_age_days=30):
    """Download *url* into *filepath* when the cache is missing or stale.

    Returns True when a usable file is present afterwards.  The body lands in a
    temporary file next to the destination and is renamed only once it is
    complete: `urlretrieve` writes straight into the destination, so an
    interrupted transfer left half a gene2go.gz behind that then counted as fresh
    cache for 30 days, and the caller's error became "Error parsing gene2go".
    """
    import os
    import tempfile
    import time
    import urllib.request

    if os.path.exists(filepath):
        age_days = (time.time() - os.path.getmtime(filepath)) / 86400
        if age_days < max_age_days:
            return True  # cached and fresh

    logger.info("Downloading %s ...", url)
    directory = os.path.dirname(filepath) or '.'
    fd, partial = tempfile.mkstemp(suffix='.part', dir=directory)
    os.close(fd)
    try:
        # urllib supports ftp:// and http://
        urllib.request.urlretrieve(url, partial)
        if os.path.getsize(partial) == 0:
            raise OSError("the transfer produced an empty file")
        os.replace(partial, filepath)
    except Exception as e:
        logger.error("Download of %s failed: %s", url, e)
        try:
            os.unlink(partial)
        except OSError as unlink_exc:
            logger.debug("could not remove the partial download %s: %s",
                         partial, unlink_exc)
        return os.path.exists(filepath)
    logger.info("  -> saved to %s", filepath)
    return True


def _load_associations_file(filepath):
    """Load gene-GO associations from a two-column file (gene_id, GO_term).

    A row that is not tab-separated is counted and reported: a space-separated
    file used to answer "Could not load gene associations for human" — the
    organism was blamed for a file the caller had named.
    """
    associations = {}
    malformed = 0
    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                    associations.setdefault(parts[0].strip(), set()).add(parts[1].strip())
                elif line.strip():
                    malformed += 1
    except OSError as e:
        logger.error("Could not read the association file %s: %s", filepath, e)
        return {}
    if malformed:
        logger.warning("%s: skipped %d row(s) that were not a tab-separated "
                       "gene/GO-term pair", filepath, malformed)
    return associations
