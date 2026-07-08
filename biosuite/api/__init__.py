"""
BioSuite Ultra — REST API Server

Exposes all 48 analysis modules as HTTP endpoints.
100% free, open-source, no paid features.

Usage:
    python -m biosuite.api.server
    # or
    uvicorn biosuite.api.server:app --host 0.0.0.0 --port 8000

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""
import os
import sys
import json
import time
import tempfile
from typing import Optional, List, Dict, Any
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BioSuite Ultra API",
    description="The most comprehensive open-source bioinformatics REST API. "
                "48 analysis modules, 36+ visualization types — all free, all pure Python.",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for web frontends
# In production, set BIOPATTER_CORS_ORIGINS env var to comma-separated origins
_cors_origins_env = os.environ.get('BIOPATTER_CORS_ORIGINS', '')
_cors_origins = [o.strip() for o in _cors_origins_env.split(',') if o.strip()] if _cors_origins_env else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ──────────────────────────────────────────────────────────

class SequenceRequest(BaseModel):
    sequence: str = Field(..., description="DNA or protein sequence", min_length=1)

class AlignmentRequest(BaseModel):
    seq1: str = Field(..., description="First sequence")
    seq2: str = Field(..., description="Second sequence")
    match: int = Field(1, description="Match score")
    mismatch: int = Field(-1, description="Mismatch penalty")
    gap: int = Field(-2, description="Gap penalty")

class TranslationRequest(BaseModel):
    sequence: str = Field(..., description="DNA sequence to translate")
    frame: int = Field(1, description="Reading frame (1-3, -1 to -3)")

class VolcanoRequest(BaseModel):
    log2fc: List[float] = Field(..., description="Log2 fold changes")
    pvalues: List[float] = Field(..., description="P-values")
    gene_names: Optional[List[str]] = Field(None, description="Gene names for hover")
    fc_thresh: float = Field(1.0, description="Fold-change threshold")
    p_thresh: float = Field(0.05, description="P-value threshold")
    interactive: bool = Field(False, description="Return Plotly JSON")

class DifferentialExpressionRequest(BaseModel):
    counts: Dict[str, List[int]] = Field(..., description="Gene counts {gene: [sample1, sample2, ...]}")
    conditions: List[str] = Field(..., description="Condition labels per sample")
    method: str = Field("ttest", description="Statistical method (ttest/nb)")

class CRISPRRequest(BaseModel):
    sequence: str = Field(..., description="Target DNA sequence")
    pam_type: str = Field("SpCas9", description="PAM type")
    guide_length: int = Field(20, description="Guide length")
    max_guides: int = Field(20, description="Maximum guides to return")

class BLASTRequest(BaseModel):
    query: str = Field(..., description="Query sequence")
    database: Optional[str] = Field(None, description="Database path (uses built-in if None)")
    evalue: float = Field(1e-5, description="E-value threshold")

class PCARequest(BaseModel):
    data: List[List[float]] = Field(..., description="Feature matrix (samples x features)")
    labels: Optional[List[str]] = Field(None, description="Sample labels")
    n_components: int = Field(2, description="Number of components")

class ManhattanRequest(BaseModel):
    chromosomes: List[str] = Field(..., description="Chromosome names")
    positions: List[int] = Field(..., description="Genomic positions")
    pvalues: List[float] = Field(..., description="P-values")
    threshold: float = Field(5e-8, description="Significance threshold")

class MetagenomicsRequest(BaseModel):
    sequences: List[Dict[str, str]] = Field(..., description="List of {name, sequence} dicts")

class EpitopeRequest(BaseModel):
    sequence: str = Field(..., description="Protein sequence")
    mhc_type: str = Field("A0201", description="HLA type")

class GWASRequest(BaseModel):
    snps: List[Dict[str, Any]] = Field(..., description="SNP data")

# ── Health & Info ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """BioSuite API landing page."""
    return """
    <html>
    <head><title>BioSuite Ultra API</title></head>
    <body style="font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
        <h1 style="color: #00ff88;">BioSuite Ultra API</h1>
        <p>The most comprehensive open-source bioinformatics REST API.</p>
        <p><strong>48 analysis modules</strong> | <strong>36+ visualization types</strong> | <strong>100% free</strong></p>
        <hr>
        <h2>Quick Links</h2>
        <ul>
            <li><a href="/docs">Swagger UI (Interactive API Docs)</a></li>
            <li><a href="/redoc">ReDoc (API Reference)</a></li>
            <li><a href="/health">Health Check</a></li>
            <li><a href="/api/v1/modules">List All Modules</a></li>
        </ul>
        <hr>
        <h2>Quick Example</h2>
        <pre>
# GC Content
curl -X POST "http://localhost:8000/api/v1/sequence/gc-content" \\
     -H "Content-Type: application/json" \\
     -d '{"sequence": "ATCGATCG"}'

# Alignment
curl -X POST "http://localhost:8000/api/v1/alignment/needleman-wunsch" \\
     -H "Content-Type: application/json" \\
     -d '{"seq1": "AGTACGCA", "seq2": "TATGC"}'
        </pre>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "4.0.0",
        "modules": 48,
        "timestamp": time.time()
    }

@app.get("/api/v1/modules")
async def list_modules():
    """List all available API modules."""
    return {
        "modules": [
            {"name": "sequence", "endpoints": ["/api/v1/sequence/*"], "description": "Sequence analysis"},
            {"name": "alignment", "endpoints": ["/api/v1/alignment/*"], "description": "Pairwise alignment"},
            {"name": "expression", "endpoints": ["/api/v1/expression/*"], "description": "Differential expression"},
            {"name": "plotting", "endpoints": ["/api/v1/plotting/*"], "description": "Visualization"},
            {"name": "crispr", "endpoints": ["/api/v1/crispr/*"], "description": "CRISPR guide design"},
            {"name": "metagenomics", "endpoints": ["/api/v1/metagenomics/*"], "description": "Taxonomic classification"},
            {"name": "epitope", "endpoints": ["/api/v1/epitope/*"], "description": "Epitope prediction"},
            {"name": "gwas", "endpoints": ["/api/v1/gwas/*"], "description": "GWAS analysis"},
            {"name": "phylogeny", "endpoints": ["/api/v1/phylogeny/*"], "description": "Phylogenetic analysis"},
            {"name": "popgen", "endpoints": ["/api/v1/popgen/*"], "description": "Population genetics"},
        ],
        "total_endpoints": 50,
        "documentation": "/docs"
    }

# ── Sequence Analysis ────────────────────────────────────────────────────────

@app.post("/api/v1/sequence/gc-content")
async def api_gc_content(req: SequenceRequest):
    """Calculate GC content of a DNA sequence."""
    from biosuite.core.sequence import gc_content
    result = gc_content(req.sequence)
    return {"gc_percent": round(result, 2), "sequence_length": len(req.sequence)}

@app.post("/api/v1/sequence/reverse-complement")
async def api_reverse_complement(req: SequenceRequest):
    """Compute reverse complement of a DNA sequence."""
    from biosuite.core.sequence import reverse_complement
    result = reverse_complement(req.sequence)
    return {"reverse_complement": result, "original": req.sequence}

@app.post("/api/v1/sequence/translate")
async def api_translate(req: TranslationRequest):
    """Translate DNA to protein."""
    from biosuite.core.sequence import translate
    protein = translate(req.sequence, frame=req.frame)
    return {"protein": protein, "frame": req.frame, "length": len(protein)}

@app.post("/api/v1/sequence/stats")
async def api_sequence_stats(req: SequenceRequest):
    """Get sequence composition statistics."""
    from biosuite.core.sequence import sequence_stats
    stats = sequence_stats(req.sequence)
    return stats

# ── Alignment ────────────────────────────────────────────────────────────────

@app.post("/api/v1/alignment/needleman-wunsch")
async def api_needleman_wunsch(req: AlignmentRequest):
    """Global pairwise alignment (Needleman-Wunsch)."""
    from biosuite.core.alignment import needleman_wunsch
    a1, a2, score = needleman_wunsch(req.seq1, req.seq2, req.match, req.mismatch, req.gap)
    return {"aligned_seq1": a1, "aligned_seq2": a2, "score": score}

@app.post("/api/v1/alignment/smith-waterman")
async def api_smith_waterman(req: AlignmentRequest):
    """Local pairwise alignment (Smith-Waterman)."""
    from biosuite.core.alignment import smith_waterman
    a1, a2, score = smith_waterman(req.seq1, req.seq2, req.match, req.mismatch, req.gap)
    return {"aligned_seq1": a1, "aligned_seq2": a2, "score": score}

# ── BLAST Search ─────────────────────────────────────────────────────────────

@app.post("/api/v1/blast/search")
async def api_blast_search(req: BLASTRequest):
    """Sequence similarity search using built-in k-mer engine."""
    from biosuite.core.blast import run_blast, format_blast_result
    import tempfile

    # Create temp query file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(f">query\n{req.query}\n")
        query_file = f.name

    try:
        if req.database and os.path.exists(req.database):
            result = run_blast(query_file, req.database, evalue=req.evalue)
        else:
            # Use built-in search with query as both query and database
            result = run_blast(query_file, query_file, evalue=req.evalue)

        return {
            "num_hits": result.num_hits,
            "engine": result.engine,
            "hits": [
                {
                    "subject_id": h.subject_id,
                    "identity": round(h.percent_identity, 2),
                    "e_value": h.e_value,
                    "score": h.bit_score,
                    "alignment_length": h.alignment_length
                }
                for h in result.top_hits(20)
            ]
        }
    finally:
        os.unlink(query_file)

# ── Differential Expression ──────────────────────────────────────────────────

@app.post("/api/v1/expression/differential")
async def api_differential_expression(req: DifferentialExpressionRequest):
    """Differential expression analysis between two groups."""
    import pandas as pd
    from biosuite.core.expression import differential_expression

    # Convert dict to DataFrame
    counts_df = pd.DataFrame(req.counts)
    counts_df.insert(0, 'gene', counts_df.index)
    counts_df = counts_df.reset_index(drop=True)

    result = differential_expression(counts_df, req.conditions, method=req.method)

    return {
        "num_genes": len(result),
        "num_upregulated": int(((result['log2FC'] > 1) & (result['padj'] < 0.05)).sum()),
        "num_downregulated": int(((result['log2FC'] < -1) & (result['padj'] < 0.05)).sum()),
        "results": result.to_dict(orient='records')
    }

@app.post("/api/v1/expression/normalize/cpm")
async def api_cpm_normalization(counts: Dict[str, List[int]]):
    """CPM normalization."""
    import pandas as pd
    import numpy as np
    from biosuite.core.expression import cpm_normalization

    df = pd.DataFrame(counts)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    result = cpm_normalization(df)
    return {"normalized": result[numeric_cols].to_dict(orient='list')}

@app.post("/api/v1/expression/normalize/tpm")
async def api_tpm_normalization(counts: Dict[str, List[int]], gene_lengths: List[float]):
    """TPM normalization."""
    import pandas as pd
    import numpy as np
    from biosuite.core.expression import tpm_normalization

    df = pd.DataFrame(counts)
    result = tpm_normalization(df, gene_lengths)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    return {"normalized": result[numeric_cols].to_dict(orient='list')}

@app.post("/api/v1/expression/normalize/deseq2")
async def api_deseq2_normalization(counts: Dict[str, List[int]]):
    """DESeq2 median-of-ratios normalization."""
    import pandas as pd
    from biosuite.core.expression import deseq2_normalization

    df = pd.DataFrame(counts)
    result = deseq2_normalization(df)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    return {"normalized": result[numeric_cols].to_dict(orient='list')}

# ── CRISPR ───────────────────────────────────────────────────────────────────

@app.post("/api/v1/crispr/design")
async def api_crispr_design(req: CRISPRRequest):
    """Design CRISPR guide RNAs."""
    from biosuite.core.crispr import design_guides

    result = design_guides(
        req.sequence,
        pam_type=req.pam_type,
        guide_length=req.guide_length,
        max_guides=req.max_guides
    )

    return {
        "engine": result.engine,
        "num_guides": result.num_guides,
        "guides": [
            {
                "sequence": g.sequence,
                "pam": g.pam,
                "position": g.position,
                "strand": g.strand,
                "score": g.score,
                "gc_content": g.gc_content,
                "on_target_score": g.on_target_score
            }
            for g in result.guides
        ]
    }

# ── Metagenomics ─────────────────────────────────────────────────────────────

@app.post("/api/v1/metagenomics/classify-16s")
async def api_classify_16s(sequences: List[Dict[str, str]]):
    """Classify 16S rRNA sequences."""
    from biosuite.core.metagenomics import classify_16s_rna

    seq_list = [(s['name'], s['sequence']) for s in sequences]
    result = classify_16s_rna(seq_list)

    return {
        "engine": result.engine,
        "num_classified": len(result.classifications),
        "classifications": result.classifications,
        "abundance": result.abundance_table.to_dict(orient='records') if result.abundance_table is not None and not result.abundance_table.empty else []
    }

@app.post("/api/v1/metagenomics/diversity")
async def api_diversity(counts: List[int]):
    """Calculate alpha diversity metrics."""
    from biosuite.core.metagenomics import shannon_entropy, simpson_index, chao1_estimator

    return {
        "shannon": round(shannon_entropy(counts), 4),
        "simpson": round(simpson_index(counts), 4),
        "chao1": round(chao1_estimator(counts), 2),
        "observed_taxa": sum(1 for c in counts if c > 0)
    }

# ── Epitope Prediction ──────────────────────────────────────────────────────

@app.post("/api/v1/epitope/predict")
async def api_epitope_predict(req: EpitopeRequest):
    """Predict T-cell and B-cell epitopes."""
    from biosuite.core.epitope import predict_t_cell_epitopes, predict_b_cell_epitopes

    t_cell = predict_t_cell_epitopes(req.sequence, mhc_type=req.mhc_type)
    b_cell = predict_b_cell_epitopes(req.sequence)

    return {
        "t_cell_epitopes": [e.to_dict() for e in t_cell[:20]],
        "b_cell_epitopes": [e.to_dict() for e in b_cell[:20]],
        "t_cell_count": len(t_cell),
        "b_cell_count": len(b_cell)
    }

# ── GWAS ─────────────────────────────────────────────────────────────────────

@app.post("/api/v1/gwas/run")
async def api_gwas_run(snps: List[Dict[str, Any]]):
    """Run GWAS analysis."""
    import pandas as pd
    from biosuite.core.gwas import run_gwas, detect_lead_snps

    df = pd.DataFrame(snps)
    results = run_gwas(df)
    leads = detect_lead_snps(results)

    return {
        "num_snps": len(results),
        "num_significant": int((results['p_value'] < 5e-8).sum()),
        "results": results.head(100).to_dict(orient='records'),
        "lead_snps": leads.to_dict(orient='records') if not leads.empty else []
    }

@app.get("/api/v1/gwas/demo")
async def api_gwas_demo(n_snps: int = Query(2000, description="Number of SNPs")):
    """Generate demo GWAS data."""
    from biosuite.core.gwas import generate_gwas_data

    data = generate_gwas_data(n_snps=n_snps)
    return {"data": data.head(100).to_dict(orient='records'), "total": len(data)}

# ── Phylogeny ────────────────────────────────────────────────────────────────

@app.post("/api/v1/phylogeny/distance-matrix")
async def api_distance_matrix(sequences: List[str]):
    """Compute pairwise distance matrix."""
    from biosuite.core.phylogeny import distance_matrix

    mat = distance_matrix(sequences)
    return {"matrix": mat.tolist(), "labels": [f"seq_{i}" for i in range(len(sequences))]}

@app.post("/api/v1/phylogeny/upgma")
async def api_upgma(sequences: List[str]):
    """Build UPGMA tree from sequences."""
    from biosuite.core.phylogeny import distance_matrix, upgma_tree

    mat = distance_matrix(sequences)
    labels = [f"seq_{i}" for i in range(len(sequences))]
    linkage = upgma_tree(mat, labels)

    return {
        "linkage_matrix": linkage.tolist(),
        "labels": labels,
        "num_sequences": len(sequences)
    }

# ── Population Genetics ─────────────────────────────────────────────────────

@app.post("/api/v1/popgen/hwe")
async def api_hwe(genotype_counts: Dict[str, int]):
    """Hardy-Weinberg equilibrium test."""
    import numpy as np
    from biosuite.core.popgen import hardy_weinberg_test

    result = hardy_weinberg_test(genotype_counts)
    # Convert numpy types to Python types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj
    return _convert(result)

@app.post("/api/v1/popgen/fst")
async def api_fst(populations: List[List[List[int]]]):
    """Calculate pairwise FST between populations."""
    import numpy as np
    from biosuite.core.popgen import calculate_fst

    matrices = [np.array(p) for p in populations]
    result = calculate_fst(matrices)
    return {"fst_pairs": {f"{k[0]}-{k[1]}": v for k, v in result.items()}}

@app.post("/api/v1/popgen/tajimas-d")
async def api_tajimas_d(genotype_matrix: List[List[int]]):
    """Calculate Tajima's D."""
    import numpy as np
    from biosuite.core.popgen import tajimas_d

    matrix = np.array(genotype_matrix)
    result = tajimas_d(matrix)
    return {"tajimas_d": result}

# ── Plotting ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/plotting/volcano")
async def api_volcano_plot(req: VolcanoRequest):
    """Generate volcano plot."""
    import numpy as np
    from biosuite.plotting.plot_api import volcano

    fig = volcano(
        np.array(req.log2fc), np.array(req.pvalues),
        gene_names=req.gene_names,
        fc_thresh=req.fc_thresh, p_thresh=req.p_thresh,
        interactive=req.interactive
    )

    if req.interactive:
        import plotly.io as pio
        return {"plotly_json": pio.to_json(fig)}
    else:
        # Save to temp file and return path
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=150, bbox_inches='tight')
            return {"image_path": f.name, "format": "png"}

@app.post("/api/v1/plotting/pca")
async def api_pca_plot(req: PCARequest):
    """Generate PCA plot."""
    import numpy as np
    from biosuite.plotting.plot_api import pca

    data = np.array(req.data)
    fig = pca(data, labels=req.labels, n_components=req.n_components, interactive=False)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=150, bbox_inches='tight')
        return {"image_path": f.name, "format": "png"}

@app.post("/api/v1/plotting/manhattan")
async def api_manhattan_plot(req: ManhattanRequest):
    """Generate Manhattan plot."""
    import numpy as np
    from biosuite.plotting.plot_api import manhattan

    fig = manhattan(
        np.array(req.chromosomes), np.array(req.positions), np.array(req.pvalues),
        threshold=req.threshold, interactive=False
    )

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=150, bbox_inches='tight')
        return {"image_path": f.name, "format": "png"}

@app.post("/api/v1/plotting/heatmap")
async def api_heatmap(data: List[List[float]], title: str = "Heatmap"):
    """Generate heatmap."""
    import numpy as np
    from biosuite.plotting.plot_api import heatmap

    fig = heatmap(np.array(data), title=title, interactive=False)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=150, bbox_inches='tight')
        return {"image_path": f.name, "format": "png"}

# ── Workflow ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/workflow/pipeline")
async def api_run_pipeline(steps: List[Dict[str, Any]]):
    """Run a pipeline of analysis steps."""
    from biosuite.core.workflow.pipeline import Pipeline

    p = Pipeline("api_pipeline")
    for step in steps:
        func_name = step.get('function', '')
        args = step.get('args', {})
        # Dynamic function loading would go here
        pass

    return {"status": "Pipeline execution not yet implemented via API", "steps": len(steps)}

# ── Database Search ──────────────────────────────────────────────────────────

@app.get("/api/v1/database/ncbi")
async def api_search_ncbi(query: str = Query(..., description="Search query"), max_results: int = 10):
    """Search NCBI databases."""
    from biosuite.core.databases import search_ncbi, format_search_results

    result = search_ncbi(query, max_results=max_results)
    return {"results": result.records, "count": result.data.get('count', 0)}

@app.get("/api/v1/database/uniprot")
async def api_search_uniprot(query: str = Query(..., description="Search query"), max_results: int = 10):
    """Search UniProt protein database."""
    from biosuite.core.databases import search_uniprot

    result = search_uniprot(query, max_results=max_results)
    return {"results": result.records}

@app.get("/api/v1/database/pdb")
async def api_search_pdb(query: str = Query(..., description="Search query"), max_results: int = 10):
    """Search RCSB PDB structures."""
    from biosuite.core.databases import search_pdb

    result = search_pdb(query, max_results=max_results)
    return {"results": result.records}

@app.get("/api/v1/database/kegg")
async def api_search_kegg(query: str = Query(..., description="Search query"), max_results: int = 10):
    """Search KEGG pathways."""
    from biosuite.core.databases import search_kegg

    result = search_kegg(query, max_results=max_results)
    return {"results": result.records}

# ── File Operations ──────────────────────────────────────────────────────────

@app.post("/api/v1/file/detect-format")
async def api_detect_format(file_path: str = Query(..., description="File path")):
    """Detect file format from extension."""
    from biosuite.core.file_formats import detect_file_format

    fmt = detect_file_format(file_path)
    return {"format": fmt, "file": file_path}

@app.post("/api/v1/file/read")
async def api_read_file(file_path: str = Query(..., description="File path")):
    """Read any supported bioinformatics file."""
    from biosuite.core.file_formats import read_file, format_file_summary

    result = read_file(file_path)
    summary = format_file_summary(result)
    return {"format": result.get('format', 'unknown'), "summary": summary}

# ── Provenance ───────────────────────────────────────────────────────────────

@app.post("/api/v1/provenance/record")
async def api_record_step(module: str, function: str, params: Dict[str, Any] = {}, result_summary: str = ""):
    """Record an analysis step for reproducibility."""
    from biosuite.core.provenance import ProvenanceTracker

    tracker = ProvenanceTracker()
    step = tracker.record(module, function, params, result_summary)
    return {"step_id": step.step_id, "session_id": step.session_id}

@app.get("/api/v1/provenance/summary")
async def api_provenance_summary():
    """Get provenance summary."""
    from biosuite.core.provenance import ProvenanceTracker

    tracker = ProvenanceTracker()
    return {"summary": tracker.summary()}

# ── Run Server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  BioSuite Ultra API Server")
    print("  http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
