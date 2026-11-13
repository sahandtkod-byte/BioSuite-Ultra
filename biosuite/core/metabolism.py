"""
Metabolic network analysis with dual-mode execution.

Pure Python stoichiometric analysis as default, COBRApy as optional.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np

from .log import get_logger

logger = get_logger(__name__)

try:
    import cobra
    HAS_COBRA = True
except ImportError:
    HAS_COBRA = False

from .utils import PerformanceWarning

FLUX_REPORT_ROWS = 10


@dataclass
class FluxResult:
    engine: str
    objective_value: float = None
    fluxes: dict = field(default_factory=dict)
    reaction_count: int = 0
    metabolite_count: int = 0
    gene_count: int = 0
    message: str = ""


@dataclass
class KnockoutResult:
    gene: str
    wild_type_flux: float | None
    knockout_flux: float | None
    growth_reduction: float | None
    essential: bool


def check_metabolism_tools():
    return {'cobra': HAS_COBRA}


# ── Pure Python FBA ─────────────────────────────────────────────────────────

def _sbml_local_name(element):
    """The tag of an SBML element without its version-specific namespace."""
    return element.tag.rsplit('}', 1)[-1].lower()


def _sbml_references(reaction, wanted_suffix):
    """Species references under this reaction's reactant or product list.

    SBML puts its element names in a namespace that changes with the level (and
    omits it entirely in some exports), and capitalises the list containers
    differently between L2 and L3, so matching the local name is the only way that
    reads more than one flavour.  A version-specific `findall` here returned an
    empty substrate/product list for every file it met, including the level-3 ones
    whose reaction it did count.
    """
    refs = []
    unreadable = []
    for group in reaction:
        if not _sbml_local_name(group).endswith(wanted_suffix):
            continue
        for ref in group:
            if 'speciesreference' not in _sbml_local_name(ref):
                continue
            sid = ref.get('species') or ref.get('reference')
            if sid is None:
                continue
            raw = ref.get('stoichiometry', '1')
            try:
                coeff = float(raw)
            except (TypeError, ValueError):
                # Defaulting to 1.0 here would invent the one number this element
                # exists to state, so the reference is skipped and counted instead.
                unreadable.append((sid, raw))
                continue
            refs.append((sid, coeff))
    return refs, unreadable


def _parse_sbml_simple(filepath):
    """Read the reactions and species of an SBML file.

    Raises what `xml.etree` raises — an unreadable file or malformed XML is a fact
    about the model the caller asked for, and swallowing it here reported a broken
    file as a model with zero reactions.
    """
    import xml.etree.ElementTree as ET

    reactions = []
    metabolites = []
    stoich = {}
    unreadable = []

    tree = ET.parse(filepath)
    root = tree.getroot()

    for element in root.iter():
        if _sbml_local_name(element) != 'reaction':
            continue
        rid = element.get('id')
        reactions.append({'id': rid, 'name': element.get('name', rid),
                          'reversible': element.get('reversible', 'false') == 'true'})
        substrates, bad_substrates = _sbml_references(element, 'reactants')
        products, bad_products = _sbml_references(element, 'products')
        unreadable += [(rid, sid, raw) for sid, raw in bad_substrates + bad_products]
        stoich[rid] = {'substrates': substrates, 'products': products}

    for element in root.iter():
        if _sbml_local_name(element) == 'species':
            mid = element.get('id')
            metabolites.append({'id': mid, 'name': element.get('name', mid)})

    if unreadable:
        logger.warning("%s: %d species reference(s) carried a stoichiometry that is "
                       "not a number and were left out: %s", filepath, len(unreadable),
                       ", ".join(f"{rid}/{sid}={raw!r}"
                                 for rid, sid, raw in unreadable[:5]))
    return reactions, metabolites, stoich


def _checked_bounds(flux_bounds):
    """A (low, high) pair of finite numbers with low <= high, or a named refusal."""
    try:
        low, high = flux_bounds
    except (TypeError, ValueError):
        raise ValueError("flux_bounds must be a (low, high) pair, got "
                         f"{flux_bounds!r}") from None
    low, high = _flux_number(low, "lower"), _flux_number(high, "upper")
    if low > high:
        raise ValueError(f"flux_bounds has its lower bound above its upper one: "
                         f"{flux_bounds!r}")
    return (low, high)


def _flux_number(value, which):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"flux_bounds: the {which} bound must be a number, got {value!r}")
    if not np.isfinite(value):
        raise ValueError(f"flux_bounds: the {which} bound must be finite, got {value!r}")
    return float(value)


def _builtin_fba(model_file=None, stoich_matrix=None, flux_bounds=(-1000, 1000)):
    """Solve max x[0] subject to S·x = 0 with the built-in linear program.

    Reaction 0 of the supplied matrix is the objective, because a bare matrix
    carries no other indication of what should grow.
    """
    import xml.etree.ElementTree as ET

    bounds = _checked_bounds(flux_bounds)

    if stoich_matrix is not None:
        # accept plain Python lists too (used to AttributeError on .shape)
        try:
            S = np.asarray(stoich_matrix, dtype=float)
        except (TypeError, ValueError) as exc:
            # numpy's own wording here is about array construction ("inhomogeneous
            # shape after 1 dimensions"), which a caller of run_fba cannot act on.
            raise ValueError("stoich_matrix must be a rectangular grid of numbers: "
                             f"{exc}") from None
        if S.ndim != 2:
            raise ValueError("stoich_matrix must be 2-dimensional")
        n_reactions = S.shape[1]
        n_metabolites = S.shape[0]
        if n_reactions == 0:
            return FluxResult(
                engine='builtin', reaction_count=0, metabolite_count=n_metabolites,
                message="FBA did not run: the matrix has no reaction columns to optimise.")
        obj = np.zeros(n_reactions)
        # FBA MAXIMISES the objective: linprog minimises c.x, so c = -1
        # (the old +1 MINIMISED reaction 0's flux — usually to -1000 or 0).
        obj[0] = -1

        from scipy.optimize import linprog
        try:
            result = linprog(obj, A_eq=S, b_eq=np.zeros(n_metabolites),
                             bounds=[bounds] * n_reactions, method='highs')
        except (ValueError, TypeError, ArithmeticError) as exc:
            # The matrix or the bounds were something highs could not work with at
            # all.  Saying so beats falling through to "No model provided", which
            # told the caller they had passed nothing when they had passed a matrix.
            logger.error("The built-in LP solver rejected a %dx%d matrix: %s",
                         n_metabolites, n_reactions, exc)
            return FluxResult(
                engine='builtin', reaction_count=n_reactions,
                metabolite_count=n_metabolites,
                message=f"FBA did not run: the solver rejected this matrix ({exc}).")

        if not result.success:
            logger.error("The built-in LP found no optimum: %s", result.message)
            return FluxResult(
                engine='builtin', reaction_count=n_reactions,
                metabolite_count=n_metabolites,
                message=f"FBA found no optimum: {result.message}")

        # + 0.0 turns the negation's -0.0 into the 0.0 the model actually reached.
        obj_val = round(-float(result.fun), 4) + 0.0
        return FluxResult(
            engine='builtin',
            objective_value=obj_val,
            fluxes={f'R{i}': round(float(v), 4) + 0.0 for i, v in enumerate(result.x)},
            reaction_count=n_reactions,
            metabolite_count=n_metabolites,
            # report the *maximised* objective, not the negated LP value
            message=f"FBA solved: objective={obj_val:.4f} (reaction 0 maximised)"
        )

    if model_file:
        try:
            reactions, metabolites, _stoich = _parse_sbml_simple(model_file)
        except OSError as exc:
            logger.error("Could not open the SBML model %s: %s", model_file, exc)
            return FluxResult(engine='builtin',
                              message=f"The SBML model {model_file} could not be read: {exc}")
        except ET.ParseError as exc:
            logger.error("The SBML model %s is not well-formed XML: %s", model_file, exc)
            return FluxResult(engine='builtin',
                              message=f"The SBML model {model_file} is not well-formed "
                                      f"XML: {exc}")
        if not reactions:
            return FluxResult(
                engine='builtin',
                message=f"No reactions were found in {model_file}, so no model was "
                        "built from it.")
        return FluxResult(
            engine='builtin',
            reaction_count=len(reactions),
            metabolite_count=len(metabolites),
            message=(f"Parsed {len(reactions)} reactions, {len(metabolites)} metabolites "
                     "from the model; the built-in engine does not solve a file, it "
                     "solves a supplied stoichiometric matrix. Install COBRApy (pip "
                     "install cobra) to run FBA on the model itself."))

    return FluxResult(engine='builtin',
                      message="No model was given: pass a stoichiometric matrix or an "
                              "SBML file.")


# ── COBRApy Wrapper ─────────────────────────────────────────────────────────

def _cobra_fba(model_file):
    """(FluxResult, None) when COBRApy solved the model, (None, reason) when it did not.

    The reason travels because the caller has to say what happened: answering None
    let run_fba report "COBRApy not found" on a machine that had COBRApy installed
    whenever the model itself was the problem.
    """
    if not HAS_COBRA:
        return None, "COBRApy is not installed"
    try:
        model = cobra.io.read_sbml_model(model_file)
        solution = model.optimize()
        objective = solution.objective_value
        if objective is None or not np.isfinite(objective):
            return None, ("the solver returned no objective value for "
                          f"{model_file} (status {getattr(solution, 'status', 'unknown')})")
        # `solution.fluxes` is a pandas Series indexed by reaction id, so its items
        # are (id, value) — reading `.id` off the key raised AttributeError inside
        # the handler below, which made this branch return None for every model and
        # sent every cobra-backed run to the built-in solver.
        fluxes = {rid: round(float(value), 4) + 0.0
                  for rid, value in solution.fluxes.items()}
        return FluxResult(
            engine='cobra',
            objective_value=round(float(objective), 4) + 0.0,
            fluxes=fluxes,
            reaction_count=len(model.reactions),
            metabolite_count=len(model.metabolites),
            gene_count=len(model.genes),
            message=f"COBRApy FBA: objective={objective:.4f}"
        ), None
    except Exception as exc:  # COBRApy surfaces model problems as many exception types
        logger.error("COBRApy could not read or solve %s: %s", model_file, exc)
        return None, str(exc)


def _cobra_knockout(model_file, gene_ids):
    """One KnockoutResult per gene COBRApy knows, plus the names it did not.

    A gene the model does not hold is counted and named rather than silently
    shrinking the result list, and a wild type that does not grow has no ratio to
    be reduced by: `growth_reduction` is then None and `essential` is False, which
    says "this test could not decide", instead of the old fabricated 0 % reduction
    that read as "not essential".
    """
    if not HAS_COBRA:
        return [], []
    missing = []
    results = []
    try:
        model = cobra.io.read_sbml_model(model_file)
        wt_flux = model.optimize().objective_value
    except Exception as exc:  # COBRApy surfaces model problems as many exception types
        logger.error("COBRApy could not prepare the knockout run for %s: %s",
                     model_file, exc)
        return [], []
    if wt_flux is None or not np.isfinite(wt_flux) or wt_flux <= 0:
        logger.warning("%s: the wild-type flux is %r, so no gene can be shown "
                       "essential by growth reduction here", model_file, wt_flux)
    for gene_id in gene_ids:
        try:
            gene = model.genes.get_by_id(gene_id)
        except KeyError:
            # COBRApy's element collections raise for an id they do not hold, so a
            # mistyped gene name is counted here instead of ending the batch: the
            # broad handler around this loop used to throw away every result that
            # had already been computed.
            missing.append(gene_id)
            continue
        if not gene:
            missing.append(gene_id)
            continue
        with model:
            gene.knock_out()
            ko_flux = model.optimize().objective_value
        comparable = (wt_flux is not None and np.isfinite(wt_flux) and wt_flux > 0
                      and ko_flux is not None and np.isfinite(ko_flux))
        results.append(KnockoutResult(
            gene=gene_id,
            wild_type_flux=round(float(wt_flux), 4) if wt_flux is not None else None,
            knockout_flux=round(float(ko_flux), 4) if ko_flux is not None else None,
            growth_reduction=round((wt_flux - ko_flux) / wt_flux * 100, 2) if comparable else None,
            essential=bool(comparable and ko_flux < 0.01 * wt_flux)
        ))
    if missing:
        logger.warning("%s: %d gene(s) were not in the model and were not tested: %s",
                       model_file, len(missing), ", ".join(map(str, missing[:10])))
    return results, missing


# ── Public API ──────────────────────────────────────────────────────────────

def run_fba(model_file=None, stoich_matrix=None) -> FluxResult:
    """Maximise an objective flux, with COBRApy when a model file can be solved.

    A supplied matrix is always solved by the built-in linear program — COBRApy
    takes a model file, not a bare array — so no warning is raised for that case.
    The `PerformanceWarning` says which engine was missing, and it never claims
    COBRApy is absent when COBRApy was in fact the one that refused the model.
    """
    if HAS_COBRA and model_file:
        result, reason = _cobra_fba(model_file)
        if result is not None:
            return result
        warnings.warn(
            f"COBRApy did not solve {model_file} ({reason}); the built-in linear "
            "programming solver was used instead.",
            PerformanceWarning, stacklevel=2)
        return _builtin_fba(model_file, stoich_matrix)

    if not HAS_COBRA and (model_file is not None or stoich_matrix is not None):
        warnings.warn(
            "COBRApy not found. Using built-in linear programming FBA. "
            "For production metabolic modeling, install COBRApy (https://opencobra.github.io/cobrapy/).",
            PerformanceWarning, stacklevel=2)
    return _builtin_fba(model_file, stoich_matrix)


def knockout_analysis(model_file, gene_ids) -> list:
    """Simulate single-gene knockouts.  Needs COBRApy: without it, nothing is tested.

    The empty list this returns when the engine is missing is the same shape as a
    run that found no essential genes, so the fact is said in the log as well.
    """
    if HAS_COBRA and model_file:
        results, _missing = _cobra_knockout(model_file, gene_ids)
        return results
    if gene_ids:
        logger.warning("Gene knockouts were asked for (%d gene id(s)) but COBRApy is "
                       "not installed, so no gene was tested: %s",
                       len(list(gene_ids)), model_file)
    return []


def create_stoichiometric_matrix(reactions_dict, metabolites_list):
    """Build the S matrix (metabolites x reactions) of a parsed reaction dict.

    A species that is both substrate and product is netted, not overwritten: one
    column entry per reaction-metabolite pair, so a catalyst cancels to zero
    instead of keeping whichever side was read last.

    Raises:
        ValueError: when a stoichiometric coefficient is not a positive finite
            number.  A negative one used to be turned into a positive by ``abs()``
            on the substrate side and left to flip the product side, so the matrix
            silently described a different reaction.
    """
    n_mets = len(metabolites_list)
    n_rxns = len(reactions_dict)
    met_idx = {m: i for i, m in enumerate(metabolites_list)}
    S = np.zeros((n_mets, n_rxns))
    skipped = 0

    for j, (rid, rxn) in enumerate(reactions_dict.items()):
        for side, sign in (('substrates', -1.0), ('products', 1.0)):
            for met, coeff in rxn.get(side, []):
                if (isinstance(coeff, bool) or not isinstance(coeff, (int, float))
                        or not np.isfinite(coeff) or coeff <= 0):
                    raise ValueError(f"reaction {rid!r} has a {side} coefficient of "
                                     f"{coeff!r} for {met!r}; stoichiometry must be a "
                                     "positive finite number")
                if met not in met_idx:
                    skipped += 1
                    continue
                S[met_idx[met], j] += sign * float(coeff)

    if skipped:
        logger.warning("%d species reference(s) named a metabolite outside the given "
                       "list and are absent from the matrix, which can leave a "
                       "reaction column empty", skipped)
    return S


def format_flux_report(result):
    """The text a user reads as the result of an FBA run.

    An objective that was never computed says so.  Printing it as `0.0000` made a
    model that was only parsed — or one the solver could not bring to an optimum —
    read as a viable network that simply did not grow.
    """
    objective = ("not computed" if result.objective_value is None
                 else f"{result.objective_value + 0.0:.4f}")
    lines = [
        "=== Flux Balance Analysis Report ===",
        f"Engine: {result.engine}",
        f"Reactions: {result.reaction_count}",
        f"Metabolites: {result.metabolite_count}",
        f"Objective value: {objective}",
    ]
    if result.gene_count:
        lines.append(f"Genes: {result.gene_count}")
    if result.fluxes:
        non_zero = {k: v for k, v in result.fluxes.items() if abs(v) > 0.001}
        lines.append(f"Non-zero fluxes: {len(non_zero)}")
        ranked = sorted(non_zero.items(), key=lambda x: (-abs(x[1]), x[0]))
        for k, v in ranked[:FLUX_REPORT_ROWS]:
            lines.append(f"  {k}: {v:.4f}")
        if len(ranked) > FLUX_REPORT_ROWS:
            lines.append(f"... and {len(ranked) - FLUX_REPORT_ROWS} more flux(es).")
    if result.message:
        lines.append(f"\nNote: {result.message}")
    return '\n'.join(lines)
