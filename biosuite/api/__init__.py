"""
BioSuite Ultra — REST API Server

Exposes the analysis modules as HTTP endpoints.

Usage:
    python -m biosuite.api.server
    # or
    uvicorn biosuite.api.server:app --host 127.0.0.1 --port 8000

Both entry points require BIOSUITE_API_KEY / BIOSUITE_JWT_SECRET (and
BIOSUITE_ADMIN_PASSWORD for the admin routes) unless BIOSUITE_DEV_MODE=1.

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""
import logging
import functools
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from biosuite import __version__
from biosuite.api.auth import verify_api_key
from biosuite.api.config import dev_mode
from biosuite.api.security import (
    ADMIN_USERNAME,
    admin_login_enabled,
    authenticate_admin,
    create_access_token,
    verify_admin_token,
)

logger = logging.getLogger(__name__)

# ── App Setup ────────────────────────────────────────────────────────────────

# Interactive documentation publishes the full endpoint surface, so it is only
# mounted anonymously in explicit dev mode.  Otherwise it is re-registered
# below behind the same API-key dependency as every other route.
_DOCS_PUBLIC = dev_mode() or os.environ.get("BIOSUITE_PUBLIC_DOCS", "").strip() in {"1", "true", "yes"}

app = FastAPI(
    title="BioSuite Ultra API",
    description="REST API for the BioSuite Ultra bioinformatics platform: sequence analysis, alignment, phylogenetics, expression analysis, population genetics, molecular cloning and plotting.",
    version=__version__,
    docs_url="/docs" if _DOCS_PUBLIC else None,
    redoc_url="/redoc" if _DOCS_PUBLIC else None,
    openapi_url="/openapi.json" if _DOCS_PUBLIC else None,
    dependencies=[Depends(verify_api_key)],
)
#Limiter for rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS for web frontends.
#
# Reflecting an arbitrary Origin while allow_credentials is on lets any web
# page a user visits drive their local API, so the default is a small
# loopback-only allow list.  Set BIOSUITE_CORS_ORIGINS to a comma-separated
# list for real deployments.  A literal "*" is honoured only with credentials
# disabled, because "*" + credentials is rejected by browsers anyway.
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:8000", "http://127.0.0.1:8000",
    "http://localhost:8080", "http://127.0.0.1:8080",
]
_cors_origins_env = os.environ.get('BIOSUITE_CORS_ORIGINS', '')
_cors_origins = [o.strip() for o in _cors_origins_env.split(',') if o.strip()] or _DEFAULT_CORS_ORIGINS
_cors_allow_credentials = "*" not in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Authorization", "Content-Type"],
)


# ── Data-access confinement ──────────────────────────────────────────────────

def _data_root() -> Path:
    """Directory that file endpoints are allowed to read from."""
    return Path(os.environ.get("BIOSUITE_DATA_DIR", os.getcwd())).resolve()


# A single path component that is safe to look up. The first character must be
# alphanumeric or "_", which rejects "..", "~" and dotfiles such as ".env"
# outright; the remainder is a conservative allowlist.
_SAFE_COMPONENT = re.compile(r"\A[A-Za-z0-9_][A-Za-z0-9._+\-]{0,254}\Z")
# Any leading separator (POSIX root or a UNC share) or a drive letter means the
# caller is not asking for a path relative to the data directory. This is
# checked explicitly rather than via ntpath.isabs(), whose result for a bare
# UNC root such as "\\\\server\\share" differs between Python 3.10 and 3.11+.
_ABSOLUTE_PREFIX = re.compile(r"\A(?:[/\\]|[A-Za-z]:)")
_MAX_PATH_DEPTH = 16


def _validated_components(user_path: str) -> List[str]:
    """Split *user_path* into components that are provably safe to look up.

    This runs **before** any filesystem call. Percent-encoding is unwrapped
    first, so ``..%2f..%2f`` cannot smuggle traversal past the check, and every
    remaining component must match a strict allowlist. NUL bytes, absolute
    POSIX paths, Windows drive letters and UNC paths, ``..``, ``~`` and
    dotfiles are all rejected here.
    """
    if not isinstance(user_path, str):
        raise HTTPException(status_code=400, detail="Invalid file path")

    decoded = user_path
    for _ in range(4):                        # unwrap repeated percent-encoding
        once = unquote(decoded)
        if once == decoded:
            break
        decoded = once

    if not decoded or "\x00" in decoded:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if _ABSOLUTE_PREFIX.match(decoded):
        raise HTTPException(
            status_code=400,
            detail="file_path must be inside the configured data directory")

    components = [c for c in re.split(r"[\\/]+", decoded) if c not in ("", ".")]
    if not components or len(components) > _MAX_PATH_DEPTH:
        raise HTTPException(status_code=400, detail="Invalid file path")
    for component in components:
        if not _SAFE_COMPONENT.match(component):
            raise HTTPException(
                status_code=400,
                detail="file_path must be inside the configured data directory")
    return components


def resolve_user_path(user_path: str) -> Path:
    """Resolve *user_path* strictly inside :func:`_data_root`.

    Untrusted text is never concatenated into a path expression. After the
    allowlist check in :func:`_validated_components`, each component is matched
    against the *actual* directory listing, so the path handed to the
    filesystem is assembled entirely from names the filesystem itself reported;
    the user-supplied string is only ever used on the right-hand side of an
    equality test. A final ``startswith`` containment check catches symlinks
    that live inside the root but point outside it.

    Raises 400 for anything that is not a plain relative path inside the data
    directory, and 404 when the target does not exist.
    """
    components = _validated_components(user_path)
    root_real = os.path.realpath(str(_data_root()))

    current = root_real
    for component in components:
        entry_path = None
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    # `component` is compared, never joined: no untrusted value
                    # reaches a path expression.
                    if entry.name == component:
                        entry_path = entry.path
                        break
        except (NotADirectoryError, FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=404,
                                detail="Requested file was not found") from exc
        if entry_path is None:
            raise HTTPException(status_code=404,
                                detail="Requested file was not found")
        current = entry_path

    # Defence in depth: a symlink inside the root may still point outside it.
    final = os.path.realpath(current)
    prefix = root_real if root_real.endswith(os.sep) else root_real + os.sep
    if not final.startswith(prefix):
        raise HTTPException(
            status_code=400,
            detail="file_path must be inside the configured data directory")
    return Path(final)


# ── Managed temporary artifacts ──────────────────────────────────────────────

_ARTIFACT_DIR = Path(tempfile.gettempdir()) / "biosuite-api-artifacts"
_ARTIFACT_MAX_FILES = 64
_ARTIFACT_MAX_AGE_S = 3600


def _prune_artifacts() -> None:
    """Bound the on-disk artifact cache by age and count."""
    try:
        files = sorted(_ARTIFACT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    now = time.time()
    for path in files:
        try:
            if now - path.stat().st_mtime > _ARTIFACT_MAX_AGE_S:
                path.unlink()
        except OSError:
            pass
    try:
        files = sorted(_ARTIFACT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    for path in files[:-_ARTIFACT_MAX_FILES] if len(files) > _ARTIFACT_MAX_FILES else []:
        try:
            path.unlink()
        except OSError:
            pass


def save_figure(fig) -> str:
    """Persist *fig* into the managed artifact directory and close it.

    Closing is mandatory: matplotlib keeps every figure alive in its global
    registry otherwise, which leaks memory in a long-running server.
    """
    import matplotlib.pyplot as plt
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', dir=str(_ARTIFACT_DIR),
                                         delete=False) as handle:
            fig.savefig(handle.name, dpi=150, bbox_inches='tight')
            path = handle.name
    finally:
        plt.close(fig)
    _prune_artifacts()
    return path


# ── Error handling ───────────────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def _value_error_handler(request: Request, exc: ValueError):
    """Malformed scientific input is a client error, not a server fault."""
    logger.info("422 on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=422,
                        content={"detail": f"Invalid input: {exc}"})


@app.exception_handler(FileNotFoundError)
async def _not_found_handler(request: Request, exc: FileNotFoundError):
    logger.info("404 on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=404, content={"detail": "Requested file was not found"})


@app.exception_handler(KeyError)
async def _key_error_handler(request: Request, exc: KeyError):
    logger.info("422 on %s: missing key %s", request.url.path, exc)
    return JSONResponse(status_code=422, content={"detail": f"Missing required field: {exc}"})


@app.exception_handler(IndexError)
async def _index_error_handler(request: Request, exc: IndexError):
    logger.info("422 on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=422,
                        content={"detail": "Input arrays have inconsistent lengths"})

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

    @field_validator("frame")
    @classmethod
    def _check_frame(cls, value: int) -> int:
        if value not in (1, 2, 3, -1, -2, -3):
            raise ValueError("frame must be one of 1, 2, 3, -1, -2, -3")
        return value

class VolcanoRequest(BaseModel):
    log2fc: List[float] = Field(..., description="Log2 fold changes")
    pvalues: List[float] = Field(..., description="P-values")
    gene_names: Optional[List[str]] = Field(None, description="Gene names for hover")
    fc_thresh: float = Field(1.0, description="Fold-change threshold")
    p_thresh: float = Field(0.05, description="P-value threshold")
    interactive: bool = Field(False, description="Return Plotly JSON")

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.log2fc) != len(self.pvalues):
            raise ValueError(
                f"log2fc ({len(self.log2fc)}) and pvalues ({len(self.pvalues)}) "
                "must have the same length")
        if self.gene_names is not None and len(self.gene_names) != len(self.log2fc):
            raise ValueError(
                f"gene_names ({len(self.gene_names)}) must match log2fc ({len(self.log2fc)})")
        return self

class DifferentialExpressionRequest(BaseModel):
    counts: Dict[str, List[int]] = Field(..., description="Gene counts {gene: [sample1, sample2, ...]}")
    conditions: List[str] = Field(..., description="Condition labels per sample")
    method: str = Field("ttest", description="Statistical method (ttest/nb)")

    @model_validator(mode="after")
    def _check_shape(self):
        if not self.counts:
            raise ValueError("counts must contain at least one gene")
        widths = {len(v) for v in self.counts.values()}
        if len(widths) != 1:
            raise ValueError("every gene must have the same number of count columns")
        width = widths.pop()
        if len(self.conditions) != width:
            raise ValueError(
                f"conditions ({len(self.conditions)}) must match the number of "
                f"count columns ({width})")
        if len(set(self.conditions)) != 2:
            raise ValueError("conditions must contain exactly two distinct groups")
        return self

class CRISPRRequest(BaseModel):
    sequence: str = Field(..., description="Target DNA sequence")
    pam_type: str = Field("SpCas9", description="PAM type")
    guide_length: int = Field(20, ge=1, le=50, description="Guide length")
    max_guides: int = Field(20, ge=1, le=1000, description="Maximum guides to return")

class BLASTRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query sequence")
    database: Optional[str] = Field(None, description="Database path inside BIOSUITE_DATA_DIR (uses built-in if None)")
    evalue: float = Field(1e-5, gt=0, description="E-value threshold")

class PCARequest(BaseModel):
    data: List[List[float]] = Field(..., description="Feature matrix (samples x features)")
    labels: Optional[List[str]] = Field(None, description="Sample labels")
    n_components: int = Field(2, ge=1, description="Number of components")

    @model_validator(mode="after")
    def _check_matrix(self):
        if not self.data:
            raise ValueError("data must contain at least one sample")
        widths = {len(row) for row in self.data}
        if len(widths) != 1:
            raise ValueError("all rows of data must have the same number of features")
        n_features = widths.pop()
        if n_features == 0:
            raise ValueError("data rows must contain at least one feature")
        limit = min(len(self.data), n_features)
        if self.n_components > limit:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed min(n_samples, "
                f"n_features) = {limit}")
        if self.labels is not None and len(self.labels) != len(self.data):
            raise ValueError("labels must have one entry per sample")
        return self

class ManhattanRequest(BaseModel):
    chromosomes: List[str] = Field(..., description="Chromosome names")
    positions: List[int] = Field(..., description="Genomic positions")
    pvalues: List[float] = Field(..., description="P-values")
    threshold: float = Field(5e-8, description="Significance threshold")

    @model_validator(mode="after")
    def _check_lengths(self):
        if not (len(self.chromosomes) == len(self.positions) == len(self.pvalues)):
            raise ValueError(
                "chromosomes, positions and pvalues must all have the same length "
                f"(got {len(self.chromosomes)}, {len(self.positions)}, {len(self.pvalues)})")
        return self

class MetagenomicsRequest(BaseModel):
    sequences: List[Dict[str, str]] = Field(..., description="List of {name, sequence} dicts")

class EpitopeRequest(BaseModel):
    sequence: str = Field(..., description="Protein sequence")
    mhc_type: str = Field("A0201", description="HLA type")

class GWASRequest(BaseModel):
    snps: List[Dict[str, Any]] = Field(..., description="SNP data")


class AdminLoginRequest(BaseModel):
    """Admin credentials.  Send these in the request *body*, never in the URL."""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


# ── Protected API documentation ──────────────────────────────────────────────

if not _DOCS_PUBLIC:
    @app.get("/openapi.json", include_in_schema=False)
    async def _protected_openapi():
        """OpenAPI schema — requires the same API key as every other route."""
        return JSONResponse(get_openapi(
            title=app.title, version=app.version,
            description=app.description, routes=app.routes))

    @app.get("/docs", include_in_schema=False)
    async def _protected_docs():
        """Swagger UI — requires the same API key as every other route."""
        return get_swagger_ui_html(openapi_url="/openapi.json",
                                   title=f"{app.title} — docs")

# ── Health & Info ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """BioSuite API landing page."""
    return """
    <html>
    <head><title>BioSuite Ultra API</title></head>
    <body style="font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
        <h1 style="color: #00ff88;">BioSuite Ultra API</h1>
        <p>REST API for the BioSuite Ultra bioinformatics platform: sequence analysis, alignment, phylogenetics, expression analysis, population genetics, molecular cloning and plotting.</p>
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

@functools.lru_cache(maxsize=1)
def _count_analysis_modules() -> int:
    """Number of analysis modules actually present in this installation.

    Counted from the package rather than hard-coded, so /health cannot drift
    away from reality when modules are added or removed.
    """
    from pathlib import Path
    import biosuite.core as core_pkg
    root = Path(core_pkg.__file__).parent
    modules = {p.stem for p in root.glob("*.py") if p.stem != "__init__"}
    modules |= {f"workflow.{p.stem}" for p in (root / "workflow").glob("*.py")
                if p.stem != "__init__"}
    return len(modules)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "modules": _count_analysis_modules(),
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
# ── Admin (JWT) ──────────────────────────────────────────────────────────────

# Simple in-process lockout for the admin login route.  slowapi caps the request
# *rate*; this additionally blocks an address after repeated wrong passwords so
# a slow-and-low guessing campaign cannot run indefinitely.
_LOGIN_MAX_FAILURES = 5
_LOGIN_LOCKOUT_S = 300
_login_failures: Dict[str, List[float]] = {}


def _client_key(request: Request) -> str:
    return get_remote_address(request) or "unknown"


def _login_locked_out(request: Request) -> bool:
    key = _client_key(request)
    now = time.time()
    recent = [t for t in _login_failures.get(key, []) if now - t < _LOGIN_LOCKOUT_S]
    _login_failures[key] = recent
    return len(recent) >= _LOGIN_MAX_FAILURES


def _record_login_failure(request: Request) -> None:
    key = _client_key(request)
    _login_failures.setdefault(key, []).append(time.time())
    logger.warning("Failed admin login attempt from %s", key)


def _clear_login_failures(request: Request) -> None:
    _login_failures.pop(_client_key(request), None)


@app.post("/api/v1/admin/login")
@limiter.limit("5/minute")
async def admin_login(
    request: Request,
    payload: Optional[AdminLoginRequest] = None,
    username: Optional[str] = Query(None, deprecated=True,
                                    description="Deprecated: send credentials in the JSON body"),
    password: Optional[str] = Query(None, deprecated=True,
                                    description="Deprecated: send credentials in the JSON body"),
):
    """Exchange admin credentials for a JWT access token.

    Credentials belong in the JSON body (``{"username": ..., "password": ...}``).
    Query parameters are still accepted for backwards compatibility but are
    deprecated because URLs are recorded in proxy and browser logs.

    The route is rate limited (5 requests/minute/client) and additionally locks
    an address out for a short period after repeated failures.
    """
    if payload is not None:
        user, pwd = payload.username, payload.password
    else:
        if username is not None or password is not None:
            logger.warning("Admin login used deprecated query-parameter credentials "
                           "from %s; use the JSON body instead.", _client_key(request))
        user, pwd = username or "", password or ""

    if not admin_login_enabled():
        raise HTTPException(
            status_code=503,
            detail="Admin login is disabled: no admin password is configured "
                   "(set BIOSUITE_ADMIN_PASSWORD).")

    if _login_locked_out(request):
        raise HTTPException(status_code=429,
                            detail="Too many failed login attempts; try again later.")

    if not authenticate_admin(user, pwd):
        _record_login_failure(request)
        raise HTTPException(status_code=401, detail="Invalid admin credentials")

    _clear_login_failures(request)
    return {"access_token": create_access_token(user), "token_type": "bearer"}

@app.get("/api/v1/admin/status")
async def admin_status(user: str = Depends(verify_admin_token)):
    """Example protected admin route."""
    return {"admin": user, "status": "ok"}
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
    import tempfile

    from biosuite.core.blast import format_blast_result, run_blast

    # Create temp query file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(f">query\n{req.query}\n")
        query_file = f.name

    try:
        database = None
        if req.database:
            resolved_db = resolve_user_path(req.database)
            if not resolved_db.is_file():
                raise HTTPException(status_code=404, detail="BLAST database not found")
            database = str(resolved_db)
        if database:
            result = run_blast(query_file, database, evalue=req.evalue)
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

    # ``counts`` maps gene -> per-sample counts, so the DataFrame must be built
    # with genes as ROWS.  ``pd.DataFrame(req.counts)`` produced the transpose
    # (samples as rows, genes as columns), which made every request fail with
    # "positional indexers are out-of-bounds" deep inside pandas.
    n_samples = len(req.conditions)
    counts_df = pd.DataFrame.from_dict(
        req.counts, orient='index',
        columns=[f"sample{i + 1}" for i in range(n_samples)])
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
    import numpy as np
    import pandas as pd

    from biosuite.core.expression import cpm_normalization

    df = pd.DataFrame(counts)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    result = cpm_normalization(df)
    return {"normalized": result[numeric_cols].to_dict(orient='list')}

@app.post("/api/v1/expression/normalize/tpm")
async def api_tpm_normalization(counts: Dict[str, List[int]], gene_lengths: List[float]):
    """TPM normalization."""
    import numpy as np
    import pandas as pd

    from biosuite.core.expression import tpm_normalization

    df = pd.DataFrame(counts)
    result = tpm_normalization(df, gene_lengths)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    return {"normalized": result[numeric_cols].to_dict(orient='list')}

@app.post("/api/v1/expression/normalize/deseq2")
async def api_deseq2_normalization(counts: Dict[str, List[int]]):
    """DESeq2 median-of-ratios normalization."""
    import numpy as np
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
    from biosuite.core.metagenomics import (
        chao1_estimator,
        shannon_entropy,
        simpson_index,
    )

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
    from biosuite.core.epitope import predict_b_cell_epitopes, predict_t_cell_epitopes

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

    from biosuite.core.gwas import detect_lead_snps, run_gwas

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
        # Save into the managed artifact directory; the figure is closed there.
        return {"image_path": save_figure(fig), "format": "png"}

@app.post("/api/v1/plotting/pca")
async def api_pca_plot(req: PCARequest):
    """Generate PCA plot."""
    import numpy as np

    from biosuite.plotting.plot_api import pca

    data = np.array(req.data)
    fig = pca(data, labels=req.labels, n_components=req.n_components, interactive=False)

    return {"image_path": save_figure(fig), "format": "png"}

@app.post("/api/v1/plotting/manhattan")
async def api_manhattan_plot(req: ManhattanRequest):
    """Generate Manhattan plot."""
    import numpy as np

    from biosuite.plotting.plot_api import manhattan

    fig = manhattan(
        np.array(req.chromosomes), np.array(req.positions), np.array(req.pvalues),
        threshold=req.threshold, interactive=False
    )

    return {"image_path": save_figure(fig), "format": "png"}

@app.post("/api/v1/plotting/heatmap")
async def api_heatmap(data: List[List[float]], title: str = "Heatmap"):
    """Generate heatmap."""
    import numpy as np

    from biosuite.plotting.plot_api import heatmap

    fig = heatmap(np.array(data), title=title, interactive=False)

    return {"image_path": save_figure(fig), "format": "png"}

# ── Workflow ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/workflow/pipeline")
async def api_run_pipeline(steps: List[Dict[str, Any]]):
    """Run a pipeline of analysis steps."""
    from biosuite.core.workflow.pipeline import Pipeline

    # build the pipeline object; per-step dynamic dispatch not yet wired
    Pipeline("api_pipeline")
    for step in steps:
        step.get('function', '')
        step.get('args', {})

    return {"status": "Pipeline execution not yet implemented via API", "steps": len(steps)}

# ── Database Search ──────────────────────────────────────────────────────────

@app.get("/api/v1/database/ncbi")
async def api_search_ncbi(query: str = Query(..., description="Search query"), max_results: int = 10):
    """Search NCBI databases."""
    from biosuite.core.databases import format_search_results, search_ncbi

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
async def api_detect_format(file_path: str = Query(..., description="Path relative to BIOSUITE_DATA_DIR")):
    """Detect file format from extension (path confined to the data directory)."""
    from biosuite.core.file_formats import detect_file_format

    resolved = resolve_user_path(file_path)
    fmt = detect_file_format(str(resolved))
    return {"format": fmt, "file": file_path}

@app.post("/api/v1/file/read")
async def api_read_file(file_path: str = Query(..., description="Path relative to BIOSUITE_DATA_DIR")):
    """Read any supported bioinformatics file from inside the data directory."""
    from biosuite.core.file_formats import format_file_summary, read_file

    resolved = resolve_user_path(file_path)
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Requested file was not found")
    result = read_file(str(resolved))
    summary = format_file_summary(result)
    return {"format": result.get('format', 'unknown'), "summary": summary}

# ── Provenance ───────────────────────────────────────────────────────────────

def get_provenance_tracker():
    """Return the process-wide provenance tracker.

    A tracker created per request records into a private session that is
    discarded immediately, which made ``/provenance/summary`` permanently
    report zero steps.  One application-scoped instance keeps the recorded
    history addressable.
    """
    from biosuite.core.provenance import ProvenanceTracker

    tracker = getattr(app.state, "provenance_tracker", None)
    if tracker is None:
        tracker = ProvenanceTracker()
        app.state.provenance_tracker = tracker
    return tracker


@app.post("/api/v1/provenance/record")
async def api_record_step(module: str, function: str, params: Optional[Dict[str, Any]] = None, result_summary: str = ""):
    """Record an analysis step for reproducibility."""
    tracker = get_provenance_tracker()
    step = tracker.record(module, function, params or {}, result_summary)
    return {"step_id": step.step_id, "session_id": step.session_id}

@app.get("/api/v1/provenance/summary")
async def api_provenance_summary():
    """Get provenance summary for this server process."""
    tracker = get_provenance_tracker()
    return {"summary": tracker.summary()}

# ── Run Server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Delegates to the single hardened entry point so that every start path
    # performs the same production-configuration check.
    from biosuite.api.server import main

    raise SystemExit(main())
