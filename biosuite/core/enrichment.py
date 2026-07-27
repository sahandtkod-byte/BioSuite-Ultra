"""
Gene Ontology and pathway enrichment analysis.

Provides over-representation analysis (ORA) and gene set enrichment analysis
(GSEA) using goatools and gseapy libraries.

All tools are free and open source:
- goatools: https://github.com/tanghaibao/goatools
- gseapy: https://github.com/zqfang/GSEApy
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

try:
    import goatools
    from goatools.go_enrichment import GOEnrichmentStudy
    from goatools.obo_parser import GODag
    from goatools.anno.idtogos_reader import IdToGosReader
    HAS_GOATOOLS = True
except ImportError:
    HAS_GOATOOLS = False

try:
    import gseapy as gp
    HAS_GSEAPY = True
except ImportError:
    HAS_GSEAPY = False


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


@dataclass
class EnrichmentReport:
    """Complete enrichment analysis report."""
    analysis_type: str
    num_input_genes: int
    num_significant: int
    results: list = field(default_factory=list)
    background_count: int = 0
    message: str = ""

    def top_terms(self, n=20):
        return self.results[:n]

    def significant_terms(self, fdr_threshold=0.05):
        return [r for r in self.results if r.adjusted_p_value < fdr_threshold]


def check_enrichment_tools():
    """Check which enrichment tools are available."""
    return {
        'goatools': HAS_GOATOOLS,
        'gseapy': HAS_GSEAPY,
    }


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
        gene_list: List of gene IDs (e.g., UniProt IDs or Entrez IDs).
        organism: Organism name ('human', 'mouse', 'yeast', etc.).
        ontology: GO subontology ('BP' for Biological Process,
                  'CC' for Cellular Component, 'MF' for Molecular Function).
        obo_file: Path to go-basic.obo file (downloaded if not provided).
        associations_file: Path to gene-GO associations file.
        method: Multiple testing correction ('bonferroni', 'fdr_bh').
        cutoff: Significance threshold (default: 0.05).

    Returns:
        EnrichmentReport object.
    """
    if not HAS_GOATOOLS:
        return EnrichmentReport(
            analysis_type='ORA',
            num_input_genes=len(gene_list),
            num_significant=0,
            message="goatools not installed. Run: pip install goatools"
        )

    if not gene_list:
        return EnrichmentReport(
            analysis_type='ORA',
            num_input_genes=0,
            num_significant=0,
            message="Empty gene list provided."
        )

    try:
        # Load GO ontology
        if obo_file is None:
            obo_file = _download_go_obo()

        obodag = GODag(obo_file)

        # Load associations
        if associations_file is None:
            associations = _get_associations(organism)
        else:
            associations = _load_associations_file(associations_file)

        if not associations:
            return EnrichmentReport(
                analysis_type='ORA',
                num_input_genes=len(gene_list),
                num_significant=0,
                message=f"Could not load gene associations for {organism}."
            )

        # Get study genes that are in the association
        study_genes = set(gene_list) & set(associations.keys())
        if not study_genes:
            return EnrichmentReport(
                analysis_type='ORA',
                num_input_genes=len(gene_list),
                num_significant=0,
                message="No input genes found in association database."
            )

        # Background: all genes in associations
        background = set(associations.keys())

        # Create GO enrichment study
        goeaobj = GOEnrichmentStudy(
            background,
            associations,
            obodag,
            methods=[method],
            cutoff=cutoff
        )

        goea_results = goeaobj.run_study(study_genes)

        # Filter by ontology and significance
        ontology_prefix = {'BP': 'GO:000', 'CC': 'GO:000', 'MF': 'GO:000'}
        results = []

        for r in goea_results:
            if not r.p_fdr_bh < cutoff:
                continue
            # Check ontology category from GO term prefix
            go_id = r.GO
            if go_id in obodag:
                term_depth = obodag[go_id].depth
                if term_depth is None or term_depth < 2:
                    continue

            enrichment = EnrichmentResult(
                analysis_type='ORA',
                term_name=r.name if hasattr(r, 'name') else '',
                term_id=r.GO,
                p_value=r.p_bonferroni if hasattr(r, 'p_bonferroni') else r.p_fdr_bh,
                adjusted_p_value=r.p_fdr_bh,
                gene_count=r.study_count,
                genes=list(r.study_items) if hasattr(r, 'study_items') else [],
                category=ontology,
                description=r.name if hasattr(r, 'name') else ''
            )
            results.append(enrichment)

        # Sort by adjusted p-value
        results.sort(key=lambda x: x.adjusted_p_value)

        return EnrichmentReport(
            analysis_type='ORA',
            num_input_genes=len(gene_list),
            num_significant=len(results),
            results=results,
            background_count=len(background)
        )

    except Exception as e:
        return EnrichmentReport(
            analysis_type='ORA',
            num_input_genes=len(gene_list),
            num_significant=0,
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
    """
    if not HAS_GSEAPY:
        return EnrichmentReport(
            analysis_type='GSEA',
            num_input_genes=len(gene_list),
            num_significant=0,
            message="gseapy not installed. Run: pip install gseapy"
        )

    if not gene_list:
        return EnrichmentReport(
            analysis_type='GSEA',
            num_input_genes=0,
            num_significant=0,
            message="Empty gene list provided."
        )

    try:
        # Build ranked gene list
        if gene_scores is not None:
            ranking = pd.Series(gene_scores, index=gene_list)
            ranking = ranking.sort_values(ascending=False)
        else:
            ranking = pd.Series(range(len(gene_list), 0, -1), index=gene_list)

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
                num_input_genes=len(gene_list),
                num_significant=0,
                message="GSEA returned no results."
            )

        df = gsea_results.res2d
        results = []

        for _, row in df.iterrows():
            term = row.get('Term', '')
            nes = row.get('NES', 0)
            pval = row.get('NOM p-val', 1.0)
            fdr = row.get('FDR q-val', 1.0)
            genes_str = row.get('Genes', '')

            gene_list_for_term = genes_str.split(';') if isinstance(genes_str, str) else []

            enrichment = EnrichmentResult(
                analysis_type='GSEA',
                term_name=term,
                term_id=term,
                p_value=float(pval) if not np.isnan(pval) else 1.0,
                adjusted_p_value=float(fdr) if not np.isnan(fdr) else 1.0,
                enrichment_score=float(nes),
                gene_count=len(gene_list_for_term),
                genes=gene_list_for_term,
                category=gene_sets
            )
            results.append(enrichment)

        results.sort(key=lambda x: x.adjusted_p_value)

        sig_count = sum(1 for r in results if r.adjusted_p_value < 0.05)

        return EnrichmentReport(
            analysis_type='GSEA',
            num_input_genes=len(gene_list),
            num_significant=sig_count,
            results=results
        )

    except Exception as e:
        return EnrichmentReport(
            analysis_type='GSEA',
            num_input_genes=len(gene_list),
            num_significant=0,
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
        gene_sets: Custom gene sets dict (default: KEGG from gseapy).
        min_size: Minimum pathway size.
        max_size: Maximum pathway size.

    Returns:
        EnrichmentReport object.
    """
    if not HAS_GSEAPY:
        return EnrichmentReport(
            analysis_type='KEGG',
            num_input_genes=len(gene_list),
            num_significant=0,
            message="gseapy not installed."
        )

    try:
        # Use gseapy's KEGG module
        kegg_name = f'KEGG_{organism.upper()}_2021'
        enrichment = gp.enrich(
            gene_list=gene_list,
            gene_sets=kegg_name,
            organism=organism,
            outdir=None,
            no_plot=True,
            cutoff=0.05
        )

        if enrichment is None:
            return EnrichmentReport(
                analysis_type='KEGG',
                num_input_genes=len(gene_list),
                num_significant=0,
                message="KEGG enrichment returned no results."
            )

        df = enrichment.res2d if hasattr(enrichment, 'res2d') else enrichment
        results = []

        for _, row in df.iterrows():
            result = EnrichmentResult(
                analysis_type='KEGG',
                term_name=row.get('Term', row.get('Description', '')),
                term_id=row.get('Term', ''),
                p_value=float(row.get('P-value', 1.0)),
                adjusted_p_value=float(row.get('Adjusted P-value', row.get('FDR', 1.0))),
                gene_count=int(row.get('Overlap', '0/0').split('/')[0]) if '/' in str(row.get('Overlap', '0')) else 0,
                category='KEGG'
            )
            results.append(result)

        results.sort(key=lambda x: x.adjusted_p_value)

        return EnrichmentReport(
            analysis_type='KEGG',
            num_input_genes=len(gene_list),
            num_significant=sum(1 for r in results if r.adjusted_p_value < 0.05),
            results=results
        )

    except Exception as e:
        return EnrichmentReport(
            analysis_type='KEGG',
            num_input_genes=len(gene_list),
            num_significant=0,
            message=f"KEGG enrichment error: {str(e)}"
        )


def format_enrichment_report(report, max_terms=30):
    """Format EnrichmentReport as readable string.

    Args:
        report: EnrichmentReport object.
        max_terms: Maximum terms to display.

    Returns:
        Formatted string.
    """
    if not report:
        return "No enrichment results available."

    lines = [
        f"=== {report.analysis_type} Enrichment Analysis ===",
        f"Input genes: {report.num_input_genes}",
        f"Significant terms (FDR < 0.05): {report.num_significant}",
        ""
    ]

    if report.message:
        lines.append(f"Note: {report.message}")
        return '\n'.join(lines)

    if not report.results:
        lines.append("No significant enriched terms found.")
        return '\n'.join(lines)

    lines.append(f"{'#':<4} {'Term':<45} {'FDR':>10} {'NES':>8} {'Genes':>6}")
    lines.append("-" * 80)

    for i, r in enumerate(report.top_terms(max_terms)):
        term_display = r.term_name[:42] if r.term_name else r.term_id[:42]
        nes_str = f"{r.enrichment_score:.2f}" if r.enrichment_score != 0 else "N/A"
        lines.append(
            f"{i+1:<4} {term_display:<45} "
            f"{r.adjusted_p_value:>10.2e} "
            f"{nes_str:>8} "
            f"{r.gene_count:>6}"
        )

    if report.num_significant > max_terms:
        lines.append(f"\n... and {report.num_significant - max_terms} more terms.")

    return '\n'.join(lines)


def _download_go_obo():
    """Download go-basic.obo if not present."""
    import os
    obo_path = os.path.join(os.path.expanduser('~'), '.biosuite', 'go-basic.obo')
    if os.path.exists(obo_path):
        return obo_path

    os.makedirs(os.path.dirname(obo_path), exist_ok=True)
    try:
        import urllib.request
        url = 'http://purl.obolibrary.org/obo/go-basic.obo'
        req = urllib.request.urlopen(url, timeout=60)
        with open(obo_path, 'wb') as f:
            f.write(req.read())
        return obo_path
    except Exception as e:
        print(f"Could not download GO ontology: {e}")
        return None


def _get_associations(organism):
    """Get gene-GO associations for an organism.

    Downloads gene2go and gene_info from NCBI FTP and caches them
    in ``~/.biosuite/``.  The first run may take a minute; subsequent
    calls use the cached files.

    Supported organisms: 'human' (taxid 9606), 'mouse' (taxid 10090),
    'yeast' (taxid 4932).

    Returns:
        Dict mapping gene_id (str) -> set of GO term IDs (str), or
        empty dict if the download fails.
    """
    import os
    import gzip
    import urllib.request

    cache_dir = os.path.join(os.path.expanduser('~'), '.biosuite')
    os.makedirs(cache_dir, exist_ok=True)

    # Map organism names to NCBI taxonomy IDs
    _TAXID = {
        'human': 9606, 'homo sapiens': 9606, '9606': 9606,
        'mouse': 10090, 'mus musculus': 10090, '10090': 10090,
        'yeast': 4932, 'saccharomyces cerevisiae': 4932, '4932': 4932,
    }
    taxid = _TAXID.get(organism.lower())
    if taxid is None:
        print(f"Unknown organism '{organism}'. Supported: human, mouse, yeast.")
        return {}

    gene2go_path = os.path.join(cache_dir, 'gene2go.gz')
    gene_info_path = os.path.join(cache_dir, 'gene_info.gz')

    # Download gene2go if not cached (or stale > 30 days)
    _download_if_needed(
        gene2go_path,
        'ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz',
    )
    _download_if_needed(
        gene_info_path,
        'ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz'
        if taxid == 9606 else
        'ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz'
        if taxid == 10090 else
        'ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Arthropoda/Saccharomyces_cerevisiae.gene_info.gz',
    )

    # Parse gene2go: filter to our taxid
    associations = {}
    try:
        opener = gzip.open if gene2go_path.endswith('.gz') else open
        with opener(gene2go_path, 'rt') as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                try:
                    file_taxid = int(parts[0])
                except ValueError:
                    continue
                if file_taxid != taxid:
                    continue
                gene_id = parts[1]
                go_term = parts[2]
                if gene_id not in associations:
                    associations[gene_id] = set()
                associations[gene_id].add(go_term)
    except Exception as e:
        print(f"Error parsing gene2go: {e}")

    if not associations:
        print(f"No GO associations found for taxid {taxid}.")

    return associations


def _download_if_needed(filepath, url, max_age_days=30):
    """Download a file from URL if it doesn't exist or is older than max_age_days.

    Args:
        filepath: Local cache path.
        url: Remote URL (ftp:// or http://).
        max_age_days: Re-download if file is older than this.
    """
    import os
    import time
    import urllib.request

    if os.path.exists(filepath):
        age_days = (time.time() - os.path.getmtime(filepath)) / 86400
        if age_days < max_age_days:
            return  # cached and fresh

    print(f"Downloading {url} ...")
    try:
        # urllib supports ftp:// and http://
        urllib.request.urlretrieve(url, filepath)
        print(f"  -> saved to {filepath}")
    except Exception as e:
        print(f"  Download failed: {e}")


def _load_associations_file(filepath):
    """Load gene-GO associations from a two-column file (gene_id, GO_term)."""
    associations = {}
    try:
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    gene, go_term = parts[0], parts[1]
                    if gene not in associations:
                        associations[gene] = set()
                    associations[gene].add(go_term)
    except Exception as e:
        print(f"Error loading associations: {e}")
    return associations
