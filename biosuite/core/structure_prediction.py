"""
Protein structure prediction with dual-mode execution.

Pure Python ESMFold via the esm library as default, AlphaFold DB API as optional.
Both are free for academic use.
"""
import tempfile
from dataclasses import dataclass, field

import numpy as np

try:
    import esm
    import torch
    HAS_ESM = True
except ImportError:
    HAS_ESM = False

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


@dataclass
class PredictionResult:
    engine: str
    pdb_string: str = ""
    plddt_scores: list = field(default_factory=list)
    sequence: str = ""
    num_residues: int = 0
    confidence: float = 0.0
    output_file: str = ""
    message: str = ""


def check_prediction_tools():
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False
    return {'esmfold': HAS_ESM, 'torch': has_torch}


# ── Pure Python ESMFold ─────────────────────────────────────────────────────

def _esmfold_predict(sequence, output_file=None):
    if not HAS_ESM:
        return PredictionResult(engine='esmfold', message="esm library not installed. Run: pip install esm")

    try:
        model = esm.pretrained.esmfold_v1()
        model = model.eval()

        with torch.no_grad():
            output = model.infer_pdb(sequence)

        pdb_string = output
        plddt = _extract_plddt(pdb_string)
        confidence = float(np.mean(plddt)) if plddt else 0.0

        if output_file:
            with open(output_file, 'w') as f:
                f.write(pdb_string)

        return PredictionResult(
            engine='esmfold',
            pdb_string=pdb_string,
            plddt_scores=plddt,
            sequence=sequence,
            num_residues=len(sequence),
            confidence=confidence,
            output_file=output_file or '',
            message=f"ESMFold prediction: {len(sequence)} residues, confidence {confidence:.1f}%"
        )
    except Exception as e:
        return PredictionResult(engine='esmfold', message=f"ESMFold error: {e}")


def _extract_plddt(pdb_string):
    """One pLDDT per RESIDUE (not per atom).

    The old version appended the B-factor of every ATOM record, inflating
    the list ~9x relative to the residue count: the report then claimed
    e.g. 280 confident residues for a 30-residue protein.
    """
    scores = []
    seen = set()
    for line in pdb_string.split('\n'):
        if line.startswith(('ATOM', 'HETATM')) and len(line) >= 66:
            chain = line[21]
            resseq = line[22:26].strip()
            try:
                bf = float(line[60:66].strip())
            except ValueError:
                continue
            if (chain, resseq) in seen:
                continue
            seen.add((chain, resseq))
            scores.append(bf)
    return scores


# ── AlphaFold DB API ────────────────────────────────────────────────────────

def _alphafold_fetch(uniprot_id, output_file=None):
    if not HAS_URLLIB:
        return PredictionResult(engine='alphafold', message="urllib not available")

    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = __import__('json').loads(response.read().decode())

        pdb_url = data[0].get('pdbUrl')
        if pdb_url:
            pdb_req = urllib.request.Request(pdb_url)
            with urllib.request.urlopen(pdb_req, timeout=60) as pdb_response:
                pdb_string = pdb_response.read().decode()

            if output_file:
                with open(output_file, 'w') as f:
                    f.write(pdb_string)

            plddt = _extract_plddt(pdb_string)
            confidence = float(np.mean(plddt)) if plddt else 0.0

            return PredictionResult(
                engine='alphafold',
                pdb_string=pdb_string,
                plddt_scores=plddt,
                confidence=confidence,
                num_residues=len(plddt),
                output_file=output_file or '',
                message=f"AlphaFold DB: {uniprot_id}, confidence {confidence:.1f}%"
            )
    except Exception as e:
        return PredictionResult(engine='alphafold', message=f"AlphaFold API error: {e}")

    return PredictionResult(engine='alphafold', message="No prediction found")


# ── Public API ──────────────────────────────────────────────────────────────

def predict_structure(sequence=None, uniprot_id=None, output_file=None):
    if output_file is None:
        output_file = tempfile.mktemp(suffix='.pdb')

    last_result = None
    if uniprot_id:
        last_result = _alphafold_fetch(uniprot_id, output_file)
        if last_result.pdb_string:
            return last_result

    if sequence:
        last_result = _esmfold_predict(sequence, output_file)
        if last_result.pdb_string:
            return last_result

    if last_result is not None:
        return last_result  # surface the engine error, not a generic text
    return PredictionResult(engine='none', message="No sequence or UniProt ID provided")


def format_prediction_report(result):
    lines = [
        "=== Structure Prediction Report ===",
        f"Engine: {result.engine}",
        f"Residues: {result.num_residues}",
        f"Confidence (pLDDT): {result.confidence:.1f}%",
        f"Output: {result.output_file}",
    ]
    if result.plddt_scores:
        high = sum(1 for s in result.plddt_scores if s > 90)
        good = sum(1 for s in result.plddt_scores if 70 < s <= 90)
        low = sum(1 for s in result.plddt_scores if s <= 70)
        lines.append(f"Confident (>90): {high} | Good (70-90): {good} | Low (<70): {low}")
    if result.message:
        lines.append(f"Note: {result.message}")
    return '\n'.join(lines)
