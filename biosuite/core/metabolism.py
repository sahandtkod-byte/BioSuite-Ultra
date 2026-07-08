"""
Metabolic network analysis with dual-mode execution.

Pure Python stoichiometric analysis as default, COBRApy as optional.
"""
import os
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field

try:
    import cobra
    HAS_COBRA = True
except ImportError:
    HAS_COBRA = False

from .utils import PerformanceWarning


@dataclass
class FluxResult:
    engine: str
    objective_value: float = 0.0
    fluxes: dict = field(default_factory=dict)
    reaction_count: int = 0
    metabolite_count: int = 0
    gene_count: int = 0
    message: str = ""


@dataclass
class KnockoutResult:
    gene: str
    wild_type_flux: float
    knockout_flux: float
    growth_reduction: float
    essential: bool


def check_metabolism_tools():
    return {'cobra': HAS_COBRA}


# ── Pure Python FBA ─────────────────────────────────────────────────────────

def _parse_sbml_simple(filepath):
    reactions = []
    metabolites = []
    stoich = {}
    objective = None

    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(filepath)
        root = tree.getroot()
        ns = {'sbml': 'http://www.sbml.org/sbml/level3/version1/core'}

        for reaction in root.findall('.//sbml:reaction', ns):
            rid = reaction.get('id')
            name = reaction.get('name', rid)
            rev = reaction.get('reversible', 'false') == 'true'
            reactions.append({'id': rid, 'name': name, 'reversible': rev})

            substrates = []
            products = []
            for species in reaction.findall('.//sbml:listOfProducts/sbml:speciesReference', ns):
                sid = species.get('species')
                coeff = float(species.get('stoichiometry', 1))
                products.append((sid, coeff))

            for species in reaction.findall('.//sbml:listOfReactants/sbml:speciesReference', ns):
                sid = species.get('species')
                coeff = float(species.get('stoichiometry', 1))
                substrates.append((sid, coeff))

            stoich[rid] = {'substrates': substrates, 'products': products}

        for metabolite in root.findall('.//sbml:species', ns):
            mid = metabolite.get('id')
            name = metabolite.get('name', mid)
            metabolites.append({'id': mid, 'name': name})

    except (ET.ParseError, KeyError, ValueError, AttributeError):
        pass

    return reactions, metabolites, stoich


def _builtin_fba(model_file=None, stoich_matrix=None, flux_bounds=(-1000, 1000)):
    if stoich_matrix is not None:
        S = stoich_matrix
        n_reactions = S.shape[1]
        n_metabolites = S.shape[0]
        obj = np.zeros(n_reactions)
        obj[0] = 1

        try:
            from scipy.optimize import linprog
            result = linprog(obj, A_eq=S, b_eq=np.zeros(n_metabolites),
                           bounds=[flux_bounds] * n_reactions, method='highs')
            if result.success:
                return FluxResult(
                    engine='builtin',
                    objective_value=round(result.fun, 4),
                    fluxes={f'R{i}': round(v, 4) for i, v in enumerate(result.x)},
                    reaction_count=n_reactions,
                    metabolite_count=n_metabolites,
                    message=f"FBA solved: objective={result.fun:.4f}"
                )
        except Exception:  # scipy.optimize can fail in many ways
            pass

    if model_file and os.path.exists(model_file):
        reactions, metabolites, stoich = _parse_sbml_simple(model_file)
        return FluxResult(
            engine='builtin',
            reaction_count=len(reactions),
            metabolite_count=len(metabolites),
            message=f"Parsed {len(reactions)} reactions, {len(metabolites)} metabolites"
        )

    return FluxResult(engine='builtin', message="No model provided")


# ── COBRApy Wrapper ─────────────────────────────────────────────────────────

def _cobra_fba(model_file):
    if not HAS_COBRA:
        return None
    try:
        model = cobra.io.read_sbml_model(model_file)
        solution = model.optimize()
        fluxes = {r.id: round(v, 4) for r, v in solution.fluxes.items()}
        return FluxResult(
            engine='cobra',
            objective_value=round(solution.objective_value, 4),
            fluxes=fluxes,
            reaction_count=len(model.reactions),
            metabolite_count=len(model.metabolites),
            gene_count=len(model.genes),
            message=f"COBRApy FBA: objective={solution.objective_value:.4f}"
        )
    except Exception:  # COBRApy can fail in many ways
        return None


def _cobra_knockout(model_file, gene_ids):
    if not HAS_COBRA:
        return []
    try:
        model = cobra.io.read_sbml_model(model_file)
        wt_flux = model.optimize().objective_value
        results = []
        for gene_id in gene_ids:
            gene = model.genes.get_by_id(gene_id)
            if gene:
                with model:
                    gene.knock_out()
                    ko_flux = model.optimize().objective_value
                    reduction = (wt_flux - ko_flux) / wt_flux * 100 if wt_flux > 0 else 0
                    results.append(KnockoutResult(
                        gene=gene_id,
                        wild_type_flux=round(wt_flux, 4),
                        knockout_flux=round(ko_flux, 4),
                        growth_reduction=round(reduction, 2),
                        essential=ko_flux < 0.01 * wt_flux
                    ))
        return results
    except Exception:  # COBRApy knockout can fail in many ways
        return []


# ── Public API ──────────────────────────────────────────────────────────────

def run_fba(model_file=None, stoich_matrix=None) -> FluxResult:
    tools = check_metabolism_tools()
    if tools['cobra'] and model_file:
        result = _cobra_fba(model_file)
        if result:
            return result

    warnings.warn(
        "COBRApy not found. Using built-in linear programming FBA. "
        "For production metabolic modeling, install COBRApy (https://opencobra.github.io/cobrapy/).",
        PerformanceWarning, stacklevel=2
    )
    return _builtin_fba(model_file, stoich_matrix)


def knockout_analysis(model_file, gene_ids) -> list:
    tools = check_metabolism_tools()
    if tools['cobra'] and model_file:
        return _cobra_knockout(model_file, gene_ids)

    return []


def create_stoichiometric_matrix(reactions_dict, metabolites_list):
    n_mets = len(metabolites_list)
    n_rxns = len(reactions_dict)
    met_idx = {m: i for i, m in enumerate(metabolites_list)}
    S = np.zeros((n_mets, n_rxns))

    for j, (rid, rxn) in enumerate(reactions_dict.items()):
        for met, coeff in rxn.get('substrates', []):
            if met in met_idx:
                S[met_idx[met], j] = -abs(coeff)
        for met, coeff in rxn.get('products', []):
            if met in met_idx:
                S[met_idx[met], j] = abs(coeff)

    return S


def format_flux_report(result):
    lines = [
        "=== Flux Balance Analysis Report ===",
        f"Engine: {result.engine}",
        f"Reactions: {result.reaction_count}",
        f"Metabolites: {result.metabolite_count}",
        f"Objective value: {result.objective_value:.4f}",
    ]
    if result.fluxes:
        non_zero = {k: v for k, v in result.fluxes.items() if abs(v) > 0.001}
        lines.append(f"Non-zero fluxes: {len(non_zero)}")
        for k, v in sorted(non_zero.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
            lines.append(f"  {k}: {v:.4f}")
    if result.message:
        lines.append(f"\nNote: {result.message}")
    return '\n'.join(lines)
