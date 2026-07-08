"""
Protein structure prediction with dual-mode execution.

Pure Python ESMFold via the esm library as default, AlphaFold DB API as optional.
Both are free for academic use.
"""
import os
import tempfile
import numpy as np
from dataclasses import dataclass, field

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
    return {'esmfold': HAS_ESM, 'torch': HAS_ESM}


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
    scores = []
    for line in pdb_string.split('\n'):
        if line.startswith('ATOM') and len(line) >= 66:
            try:
                bf = float(line[60:66].strip())
                scores.append(bf)
            except ValueError:
                pass
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

    if uniprot_id:
        result = _alphafold_fetch(uniprot_id, output_file)
        if result.pdb_string:
            return result

    if sequence:
        result = _esmfold_predict(sequence, output_file)
        if result.pdb_string:
            return result

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
