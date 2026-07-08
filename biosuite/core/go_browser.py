"""
Gene Ontology browser — browse GO hierarchy, query term relationships,
perform GO enrichment analysis. Pure Python with optional goatools.
"""
import os
from collections import defaultdict

try:
    import goatools
    from goatools.obo_parser import GODag
    HAS_GOATOOLS = True
except ImportError:
    HAS_GOATOOLS = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ─── Built-in GO subset (for offline use) ───────────────────────────────────

BASIC_GO_TERMS = {
    # Biological Process
    "GO:0008150": {"name": "biological_process", "namespace": "BP", "parents": [], "def": "A biological process."},
    "GO:0009987": {"name": "cellular process", "namespace": "BP", "parents": ["GO:0008150"], "def": "Any process that is carried out at the cellular level."},
    "GO:0006457": {"name": "protein folding", "namespace": "BP", "parents": ["GO:0009987"], "def": "The process of polypeptide chain assuming its functional conformation."},
    "GO:0006281": {"name": "DNA repair", "namespace": "BP", "parents": ["GO:0009987"], "def": "The restoration of DNA after genetic damage."},
    "GO:0007155": {"name": "cell adhesion", "namespace": "BP", "parents": ["GO:0009987"], "def": "Attachment of a cell to a surface or substrate."},
    "GO:0006915": {"name": "apoptotic process", "namespace": "BP", "parents": ["GO:0009987"], "def": "Programmed cell death."},
    "GO:0008283": {"name": "cell proliferation", "namespace": "BP", "parents": ["GO:0009987"], "def": "The multiplication or reproduction of cells."},
    "GO:0006954": {"name": "inflammatory response", "namespace": "BP", "parents": ["GO:0009987"], "def": "Inflammation, the immediate defense response."},
    "GO:0006468": {"name": "protein phosphorylation", "namespace": "BP", "parents": ["GO:0009987"], "def": "Addition of a phosphate group to a protein."},
    "GO:0007165": {"name": "signal transduction", "namespace": "BP", "parents": ["GO:0009987"], "def": "Process of transmitting a signal."},
    "GO:0015031": {"name": "protein transport", "namespace": "BP", "parents": ["GO:0009987"], "def": "Directing proteins to their destination."},
    "GO:0006355": {"name": "regulation of transcription", "namespace": "BP", "parents": ["GO:0009987"], "def": "Modulation of DNA-templated transcription."},
    # Molecular Function
    "GO:0003674": {"name": "molecular_function", "namespace": "MF", "parents": [], "def": "A molecular function."},
    "GO:0003824": {"name": "catalytic activity", "namespace": "MF", "parents": ["GO:0003674"], "def": "Catalysis of a biochemical reaction."},
    "GO:0005524": {"name": "ATP binding", "namespace": "MF", "parents": ["GO:0003674"], "def": "Binding to ATP."},
    "GO:0005515": {"name": "protein binding", "namespace": "MF", "parents": ["GO:0003674"], "def": "Binding to a protein."},
    "GO:0003700": {"name": "transcription factor activity", "namespace": "MF", "parents": ["GO:0003674"], "def": "Activity of a transcription factor."},
    "GO:0004672": {"name": "protein kinase activity", "namespace": "MF", "parents": ["GO:0003824"], "def": "Catalysis of protein phosphorylation."},
    "GO:0004674": {"name": "serine/threonine kinase activity", "namespace": "MF", "parents": ["GO:0004672"], "def": "Catalysis of serine/threonine phosphorylation."},
    "GO:0016787": {"name": "hydrolase activity", "namespace": "MF", "parents": ["GO:0003824"], "def": "Catalysis of hydrolysis."},
    "GO:0008270": {"name": "zinc ion binding", "namespace": "MF", "parents": ["GO:0003674"], "def": "Binding to zinc ions."},
    # Cellular Component
    "GO:0005575": {"name": "cellular_component", "namespace": "CC", "parents": [], "def": "A component of a cell."},
    "GO:0005634": {"name": "nucleus", "namespace": "CC", "parents": ["GO:0005575"], "def": "The membrane-bounded organelle containing chromosomes."},
    "GO:0005737": {"name": "cytoplasm", "namespace": "CC", "parents": ["GO:0005575"], "def": "The part of the cell excluding the nucleus."},
    "GO:0016020": {"name": "membrane", "namespace": "CC", "parents": ["GO:0005575"], "def": "A lipid bilayer membrane."},
    "GO:0005829": {"name": "cytosol", "namespace": "CC", "parents": ["GO:0005737"], "def": "The cytoplasm excluding organelles."},
    "GO:0005783": {"name": "endoplasmic reticulum", "namespace": "CC", "parents": ["GO:0005575"], "def": "The ER membrane system."},
    "GO:0005794": {"name": "Golgi apparatus", "namespace": "CC", "parents": ["GO:0005575"], "def": "The Golgi complex."},
    "GO:0005654": {"name": "nucleoplasm", "namespace": "CC", "parents": ["GO:0005634"], "def": "The nucleus excluding chromosomes."},
    "GO:0070062": {"name": "extracellular exosome", "namespace": "CC", "parents": ["GO:0005575"], "def": "Extracellular vesicle."},
    "GO:0005886": {"name": "plasma membrane", "namespace": "CC", "parents": ["GO:0016020"], "def": "The cell membrane."},
}


class GOTerm:
    """Represents a single GO term."""

    def __init__(self, go_id, name, namespace, parents=None, definition=""):
        self.go_id = go_id
        self.name = name
        self.namespace = namespace
        self.parents = parents or []
        self.definition = definition

    def __repr__(self):
        return f"GOTerm({self.go_id}, {self.name}, {self.namespace})"


class GOBrowser:
    """Browse and query Gene Ontology terms."""

    def __init__(self, obo_file=None):
        self.terms = {}
        self.children = defaultdict(list)

        if obo_file and HAS_GOATOOLS and os.path.exists(obo_file):
            self._load_obo(obo_file)
        else:
            self._load_builtin()

    def _load_obo(self, obo_file):
        dag = GODag(obo_file)
        for go_id, item in dag.items():
            ns = item.namespace.split("_")[0].upper() if hasattr(item, 'namespace') else "BP"
            parents = [p for p in item.parents] if hasattr(item, 'parents') else []
            self.terms[go_id] = GOTerm(
                go_id=go_id, name=item.name, namespace=ns,
                parents=[p.id for p in parents] if hasattr(parents[0], 'id') else parents,
                definition=getattr(item, 'def', '')
            )
        for go_id, term in self.terms.items():
            for parent_id in term.parents:
                self.children[parent_id].append(go_id)

    def _load_builtin(self):
        for go_id, data in BASIC_GO_TERMS.items():
            self.terms[go_id] = GOTerm(
                go_id=go_id, name=data["name"], namespace=data["namespace"],
                parents=data["parents"], definition=data["def"]
            )
        for go_id, term in self.terms.items():
            for parent_id in term.parents:
                self.children[parent_id].append(go_id)

    def search(self, query):
        query_lower = query.lower()
        results = []
        for go_id, term in self.terms.items():
            if query_lower in term.name.lower() or query_lower in go_id.lower():
                results.append(term)
        return results

    def get_term(self, go_id):
        return self.terms.get(go_id)

    def get_parents(self, go_id):
        term = self.terms.get(go_id)
        if not term:
            return []
        return [self.terms[p] for p in term.parents if p in self.terms]

    def get_children(self, go_id):
        return [self.terms[c] for c in self.children.get(go_id, []) if c in self.terms]

    def get_ancestors(self, go_id):
        visited = set()
        queue = [go_id]
        ancestors = []
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            term = self.terms.get(current)
            if term:
                ancestors.append(term)
                queue.extend(term.parents)
        return ancestors

    def get_namespace_terms(self, namespace):
        return [t for t in self.terms.values() if t.namespace == namespace]

    def get_dag(self, go_id, depth=3):
        """Get a DAG subgraph rooted at go_id."""
        nodes = []
        queue = [(go_id, 0)]
        while queue:
            current, d = queue.pop(0)
            if d > depth or current in [n[0] for n in nodes]:
                continue
            nodes.append((current, d))
            for child_id in self.children.get(current, []):
                queue.append((child_id, d + 1))
        return nodes


def go_enrichment(gene_list, go_terms_map, background_size=None):
    """Simple GO enrichment using Fisher's exact test.

    Args:
        gene_list: list of gene IDs with GO annotations.
        go_terms_map: dict mapping GO term -> list of gene IDs.
        background_size: total number of genes in background.

    Returns:
        list of dicts with go_term, p_value, enrichment, genes.
    """
    from scipy.stats import fisher_exact
    import numpy as np

    gene_set = set(gene_list)
    n_total = background_size or len(gene_set) * 10
    results = []

    for go_term, annotated_genes in go_terms_map.items():
        go_set = set(annotated_genes)
        a = len(gene_set & go_set)
        b = len(gene_set - go_set)
        c = len(go_set - gene_set)
        d = n_total - a - b - c
        table = [[a, b], [c, d]]
        _, pval = fisher_exact(table, alternative='greater')
        n_annotated = len(go_set)
        enrichment = (a / max(len(gene_set), 1)) / (n_annotated / max(n_total, 1))
        results.append({
            "go_term": go_term,
            "genes": list(gene_set & go_set),
            "count": a,
            "p_value": pval,
            "enrichment": round(enrichment, 2),
        })

    results.sort(key=lambda x: x["p_value"])
    return results


def format_go_results(terms):
    """Format GO term search results."""
    if not terms:
        return "No GO terms found."
    lines = [f"{'GO ID':<15} {'Name':<35} {'Namespace':<5} {'Definition'}"]
    lines.append("-" * 100)
    for t in terms:
        defn = t.definition[:40] + "..." if len(t.definition) > 40 else t.definition
        lines.append(f"{t.go_id:<15} {t.name:<35} {t.namespace:<5} {defn}")
    lines.append(f"\nFound {len(terms)} terms")
    return "\n".join(lines)
