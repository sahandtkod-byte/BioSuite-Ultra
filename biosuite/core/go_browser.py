"""
Gene Ontology browser — browse GO hierarchy, query term relationships,
perform GO enrichment analysis. Pure Python with optional goatools.
"""
import os
from collections import defaultdict

from .log import get_logger

logger = get_logger(__name__)

GODag = None
try:
    from goatools.obo_parser import GODag  # noqa: F811
    HAS_GOATOOLS = True
except ImportError:
    HAS_GOATOOLS = False


# ─── Built-in GO subset (for offline use) ───────────────────────────────────

# goatools reports GO namespaces as their long names, while every consumer in
# BioSuite (CLI option 95, the GUI GO browser, the built-in subset below)
# filters on the two-letter codes.
GO_NAMESPACE_CODES = {
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
}

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

# Named after the table it counts, so the browser can tell the user how many terms
# it really holds when no OBO file was loaded.
BUILTIN_SOURCE = f"built-in GO subset ({len(BASIC_GO_TERMS)} terms, offline)"


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
    """Browse and query Gene Ontology terms.

    ``source`` says which ontology is actually loaded.  Without it a browser built
    on the built-in subset — 31 terms, offline — is indistinguishable on screen
    from one built on the ~45,000-term OBO file the user asked for, and a search
    that returns nothing reads as "that term is not in GO".
    """

    def __init__(self, obo_file=None):
        self.terms = {}
        self.children = defaultdict(list)
        self.source = BUILTIN_SOURCE

        if not obo_file:
            self._load_builtin()
            return
        if not HAS_GOATOOLS:
            logger.warning(
                "%s was asked for, but goatools is not installed, so this browser "
                "holds the built-in subset only (%d terms). Run: pip install goatools",
                obo_file, len(BASIC_GO_TERMS))
            self._load_builtin()
            return
        if not os.path.exists(obo_file):
            logger.warning("%s was asked for but is not there, so this browser holds "
                           "the built-in subset only (%d terms)", obo_file,
                           len(BASIC_GO_TERMS))
            self._load_builtin()
            return
        # A file that is there but cannot be parsed lets goatools' own error
        # through: the caller named that file, so answering with a 31-term subset
        # would be a guess about what they wanted.
        self._load_obo(obo_file)

    def _load_obo(self, obo_file):
        dag = GODag(obo_file)
        self.source = f"GO OBO file {obo_file}"
        unnamed_parents = 0
        for go_id, item in dag.items():
            raw_ns = getattr(item, "namespace", "biological_process")
            ns = GO_NAMESPACE_CODES.get(raw_ns, raw_ns.split("_")[0].upper())
            parents = []
            for parent in getattr(item, "parents", ()) or ():
                # Every parent is mapped on its own.  Reading only the first one's
                # `id` decided for the whole list, so a mixed list kept the rest as
                # term objects, and those became children-map keys no lookup with
                # a GO id could ever reach.
                parent_id = getattr(parent, "id", None)
                if parent_id is None and isinstance(parent, str):
                    parent_id = parent
                if parent_id is None:
                    unnamed_parents += 1
                    continue
                if parent_id not in parents:
                    # An OBO file can assert the same parent twice; keeping it twice
                    # puts the child into the parent's children list twice, and every
                    # count of "how many terms are under GO:x" then over-reports.
                    parents.append(parent_id)
            self.terms[go_id] = GOTerm(
                go_id=go_id, name=item.name, namespace=ns,
                parents=parents,
                definition=getattr(item, 'def', '')
            )
        if unnamed_parents:
            logger.warning("%s: %d parent reference(s) carried no GO id and were "
                           "dropped from the hierarchy", obo_file, unnamed_parents)
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
        """Find terms whose name or id contains ``query``, case-insensitively.

        The query is trimmed, and an empty one is refused rather than answered:
        an empty box used to match every term in the ontology, and both front ends
        printed that as the result of a search.  Callers that want the whole
        namespace ask for it with :meth:`get_namespace_terms`.
        """
        if not isinstance(query, str):
            raise TypeError(f"search() needs a string query, got {type(query).__name__}")
        needle = query.strip().lower()
        if not needle:
            raise ValueError("search() needs a non-empty query; use "
                             "get_namespace_terms() to list a whole namespace")
        results = []
        for go_id, term in self.terms.items():
            if needle in term.name.lower() or needle in go_id.lower():
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
        """Every term reachable upwards from ``go_id``, starting with the term itself.

        The list is the path the DAG walk took, so element 0 is the queried term and
        the rest are its parents and their parents; each term appears once even when
        the graph reaches it by two routes.
        """
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
        """Every term of one namespace ('BP', 'MF' or 'CC'), in load order."""
        return [t for t in self.terms.values() if t.namespace == namespace]

    def get_dag(self, go_id, depth=3):
        """Get a DAG subgraph rooted at go_id, as (go_id, level) pairs.

        ``go_id`` has to be a term this browser knows.  An unknown id used to come
        back as a one-node subgraph for a term that does not exist, which the
        caller could only read as an ontology containing it.
        """
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise ValueError(f"depth must be a whole number of levels, got {depth!r}")
        if depth < 0:
            raise ValueError(f"depth must be 0 or more, got {depth}")
        if go_id not in self.terms:
            return []
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


def _gene_names(values, term=None):
    """Turn one gene collection into a set of usable identifiers.

    The rules are the ones `enrichment._gene_arguments` already applies, so the two
    enrichment paths in the package refuse the same mistakes in the same words: a
    bare string is refused because iterating it yields characters, and a blank entry
    is refused because a gene named '' matches no annotation and would silently
    disappear from every count.
    """
    where = f" for {term!r}" if term else ""
    if isinstance(values, (str, bytes)):
        raise ValueError("go_enrichment() needs the genes"
                         f"{where} as a list of gene identifiers, not one bare "
                         f"{type(values).__name__} ({values[:20]!r}…)")
    try:
        # bool is an int subclass, and str(True) is 'True' — a name no annotation
        # table can hold, so a flag slipped into a gene list is refused by value.
        unreadable = [value for value in values
                      if isinstance(value, bool) or not isinstance(value, (str, int, float))]
    except TypeError:
        raise ValueError(f"go_enrichment() needs the genes{where} as an iterable of "
                         f"gene identifiers, got {values!r}") from None
    if unreadable:
        raise ValueError(f"go_enrichment() found {len(unreadable)} gene "
                         f"identifier(s){where} that is not a name, e.g. "
                         f"{unreadable[0]!r}")
    names = [str(value) for value in values]
    blank = [name for name in names if not name.strip()]
    if blank:
        raise ValueError(f"go_enrichment() found {len(blank)} blank gene "
                         f"identifier(s){where}.")
    return set(names)


def go_enrichment(gene_list, go_terms_map, background_size):
    """Score each GO term against a gene list with Fisher's exact test.

    Args:
        gene_list: the query genes.
        go_terms_map: dict mapping GO term -> the genes annotated with it.
        background_size: the number of genes the background holds — the population
            the query was drawn from.  **There is no default for this.**  A missing
            value used to be answered with ``len(gene_list) * 10``, and every p-value
            and enrichment ratio that came back was computed from that invention:
            measured, 4 query genes against a term annotating all of them reported
            ``enrichment=10.0``, which is the magic factor, not the data.

    Returns:
        list of dicts with go_term, genes, count, p_value and enrichment, ranked by
        p_value and then by term id, and with the gene names in a fixed order, so
        the same input always produces the same report.

    Raises:
        ValueError: when the gene list is empty, when ``background_size`` is not a
            whole number of at least 1, or when it is too small to hold the query
            and the term's annotations together — a 2x2 table with a negative cell
            has no p-value to report.
    """
    from scipy.stats import fisher_exact

    gene_set = _gene_names(gene_list)
    if not gene_set:
        raise ValueError("go_enrichment() needs at least one gene in gene_list")
    if (isinstance(background_size, bool) or not isinstance(background_size, int)
            or background_size < 1):
        raise ValueError("background_size must be a whole number of genes of at least "
                         f"1, got {background_size!r}")

    results = []
    for go_term, annotated_genes in go_terms_map.items():
        go_set = _gene_names(annotated_genes, go_term)
        a = len(gene_set & go_set)
        b = len(gene_set - go_set)
        c = len(go_set - gene_set)
        d = background_size - a - b - c
        if d < 0:
            raise ValueError(
                f"background_size={background_size} cannot hold the {a + b + c} genes "
                f"that the query and term {go_term!r} together describe")
        if not go_set:
            # A term annotating no gene cannot be enriched.  The row stays so a
            # broken annotation file is visible, but no ratio is invented for it.
            logger.warning("%r annotates no genes; reported as p_value=1.0, "
                           "enrichment 0.0", go_term)
            results.append({"go_term": go_term, "genes": [], "count": 0,
                            "p_value": 1.0, "enrichment": 0.0})
            continue
        _, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
        enrichment = (a / len(gene_set)) / (len(go_set) / background_size)
        results.append({
            "go_term": go_term,
            "genes": sorted(gene_set & go_set),
            "count": a,
            "p_value": float(pval),
            "enrichment": round(enrichment, 2),
        })

    results.sort(key=lambda row: (row["p_value"], row["go_term"]))
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
