"""
Sequence similarity search with dual-mode execution.

Uses BLAST+ if installed for speed, otherwise falls back to a pure Python
k-mer indexed search engine. Works out of the box with just pip install.
"""
import os
import subprocess
import tempfile
import numpy as np
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

try:
    from Bio import SeqIO, AlignIO
    from Bio.Blast import NCBIXML
    from Bio.Align import PairwiseAligner
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

from .utils import PerformanceWarning


@dataclass
class BlastHit:
    query_id: str
    subject_id: str
    subject_description: str
    percent_identity: float
    alignment_length: int
    mismatches: int
    gap_opens: int
    query_start: int
    query_end: int
    subject_start: int
    subject_end: int
    e_value: float
    bit_score: float
    query_cov: float = 0.0

    def __str__(self):
        return (f"{self.subject_id} | {self.percent_identity:.1f}% identity | "
                f"E={self.e_value:.2e} | Score={self.bit_score:.0f}")


@dataclass
class BlastResult:
    program: str
    database: str
    query_length: int
    hits: list = field(default_factory=list)
    message: str = ""
    engine: str = "builtin"  # 'blast+' or 'builtin'

    @property
    def num_hits(self):
        return len(self.hits)

    def top_hits(self, n=10):
        return self.hits[:n]

    def significant_hits(self, evalue_threshold=1e-5):
        return [h for h in self.hits if h.e_value < evalue_threshold]


# ── External BLAST+ Detection ──────────────────────────────────────────────

def _has_blast_plus():
    try:
        r = subprocess.run(['blastn', '-version'], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_blast_installed():
    tools = ['blastn', 'blastp', 'blastx', 'tblastn', 'tblastx', 'makeblastdb']
    available = {}
    for t in tools:
        try:
            r = subprocess.run([t, '-version'], capture_output=True, text=True, timeout=10)
            available[t] = r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            available[t] = False
    return available


# ── Pure Python K-mer Search Engine ─────────────────────────────────────────

def _build_kmer_index(sequences, k=15):
    index = defaultdict(list)
    for seq_id, seq in sequences:
        seq_upper = seq.upper().replace('U', 'T')
        for i in range(len(seq_upper) - k + 1):
            kmer = seq_upper[i:i + k]
            if 'N' not in kmer:
                index[kmer].append((seq_id, i))
    return index


def _find_seed_hits(query, index, k=15, max_hits=200):
    query_upper = query.upper().replace('U', 'T')
    hits = defaultdict(list)
    count = 0
    for i in range(len(query_upper) - k + 1):
        if count >= max_hits:
            break
        kmer = query_upper[i:i + k]
        if kmer in index:
            for seq_id, db_pos in index[kmer]:
                hits[(seq_id, db_pos - i)].append(i)
                count += 1
    # Sort by number of seed hits (best first)
    return sorted(hits.items(), key=lambda x: len(x[1]), reverse=True)[:50]


def _banded_align_score(seq1, seq2, start1, start2, match=1, mismatch=-1, gap=-2):
    """Quick banded local alignment score between two subsequences."""
    max_len = min(len(seq1) - start1, len(seq2) - start2, 500)
    if max_len <= 0:
        return 0, 0, 0, 0, 0, 0, 0

    s1 = seq1[start1:start1 + max_len]
    s2 = seq2[start2:start2 + max_len]
    n, m = len(s1), len(s2)

    best_score = 0
    best_i, best_j = 0, 0

    if HAS_BIO:
        try:
            aligner = PairwiseAligner()
            aligner.mode = 'local'
            aligner.match_score = match
            aligner.mismatch_score = mismatch
            aligner.open_gap_score = gap
            aligner.extend_gap_score = gap
            alignments = aligner.align(s1, s2)
            if alignments:
                best = alignments[0]
                score = best.score
                # Estimate positions from alignment
                q_start = start1
                t_start = start2
                q_end = start1 + len(s1)
                t_end = start2 + len(s2)
                matches = sum(1 for a, b in zip(s1[:min(n, m)], s2[:min(n, m)]) if a == b)
                return score, matches, min(n, m) - matches, q_start, t_start, q_end, t_end
        except (ImportError, AttributeError, TypeError):
            pass

    # Fallback: simple scoring
    score = 0
    matches = 0
    gaps = 0
    for i in range(min(n, m)):
        if s1[i] == s2[i]:
            score += match
            matches += 1
        else:
            score += mismatch
    return score, matches, max_len - matches, start1, start2, start1 + max_len, start2 + max_len


def _estimate_evalue(score, db_size, query_len, k=15):
    """Rough E-value approximation."""
    if score <= 0:
        return 1.0
    lambda_param = 0.332  # Approximate for nucleotide search
    K = 0.133
    effective_db = db_size * max(query_len - k + 1, 1)
    return K * effective_db * np.exp(-lambda_param * score)


def _read_fasta(filepath):
    if not HAS_BIO:
        from .utils import read_fasta_simple
        return read_fasta_simple(filepath)
    return [(r.id, str(r.seq)) for r in SeqIO.parse(filepath, 'fasta')]


def _builtin_search(query_file, database_file, evalue=1e-5, max_hits=500, k=15):
    queries = _read_fasta(query_file)
    db_seqs = _read_fasta(database_file)

    if not queries or not db_seqs:
        return BlastResult(program='builtin_search', database=database_file,
                          query_length=0, message='No sequences found.', engine='builtin')

    total_db_len = sum(len(s) for _, s in db_seqs)
    index = _build_kmer_index(db_seqs, k=k)
    first_query_len = len(queries[0][1])

    result = BlastResult(
        program='builtin_search',
        database=os.path.basename(database_file),
        query_length=first_query_len,
        engine='builtin'
    )

    for q_id, q_seq in queries:
        seed_hits = _find_seed_hits(q_seq, index, k=k)
        seen_subjects = set()

        for (subject_id, offset), positions in seed_hits:
            if subject_id in seen_subjects:
                continue
            seen_subjects.add(subject_id)

            db_seq = next((s for sid, s in db_seqs if sid == subject_id), None)
            if db_seq is None:
                continue

            seed_pos = positions[0]
            db_start = max(0, offset - 5)
            score, matches, mismatches, qs, ts, qe, te = _banded_align_score(
                q_seq, db_seq, seed_pos, db_start
            )

            align_len = min(qe - qs, te - ts)
            if align_len <= 0:
                continue

            computed_evalue = _estimate_evalue(score, total_db_len, len(q_seq), k=k)
            if computed_evalue > evalue:
                continue

            pct_identity = matches / align_len * 100 if align_len > 0 else 0

            hit = BlastHit(
                query_id=q_id,
                subject_id=subject_id,
                subject_description=subject_id,
                percent_identity=pct_identity,
                alignment_length=align_len,
                mismatches=mismatches,
                gap_opens=0,
                query_start=qs + 1,
                query_end=qe,
                subject_start=ts + 1,
                subject_end=te,
                e_value=computed_evalue,
                bit_score=score,
                query_cov=align_len / len(q_seq) * 100
            )
            result.hits.append(hit)

            if len(result.hits) >= max_hits:
                break

    result.hits.sort(key=lambda x: x.e_value)
    return result


# ── External BLAST+ Execution ───────────────────────────────────────────────

def _blast_plus_search(query_file, database, program='blastn', evalue=1e-5,
                       max_target_seqs=500, num_threads=1):
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tmp:
        outfile = tmp.name

    cmd = [program, '-query', query_file, '-db', database,
           '-out', outfile, '-outfmt', '5', '-evalue', str(evalue),
           '-max_target_seqs', str(max_target_seqs),
           '-num_threads', str(num_threads)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return BlastResult(program=program, database=database,
                              query_length=0, message=f"BLAST+ error: {r.stderr[:300]}",
                              engine='blast+')

        with open(outfile) as f:
            records = list(NCBIXML.parse(f))

        result = BlastResult(program=program, database=database,
                            query_length=records[0].query_length or 0,
                            engine='blast+')

        for record in records:
            for alignment in record.alignments:
                for hsp in alignment.hsps:
                    hit = BlastHit(
                        query_id=record.query,
                        subject_id=alignment.accession or alignment.title.split()[0],
                        subject_description=alignment.title,
                        percent_identity=hsp.identities / hsp.align_length * 100 if hsp.align_length > 0 else 0,
                        alignment_length=hsp.align_length,
                        mismatches=hsp.mismatches,
                        gap_opens=hsp.gaps,
                        query_start=hsp.query_start,
                        query_end=hsp.query_end,
                        subject_start=hsp.sbjct_start,
                        subject_end=hsp.sbjct_end,
                        e_value=hsp.expect,
                        bit_score=hsp.bits,
                        query_cov=hsp.align_length / (record.query_length or 1) * 100
                    )
                    result.hits.append(hit)

        result.hits.sort(key=lambda x: x.e_value)
        return result

    except Exception as e:
        return BlastResult(program=program, database=database,
                          query_length=0, message=str(e), engine='blast+')
    finally:
        if os.path.exists(outfile):
            try:
                os.unlink(outfile)
            except OSError:
                pass


# ── Public API ──────────────────────────────────────────────────────────────

def make_database(sequence_file, dbtype='nucl', output_dir=None, title='mydb'):
    if _has_blast_plus():
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='blastdb_')
        db_prefix = os.path.join(output_dir, 'blastdb')
        cmd = ['makeblastdb', '-in', sequence_file, '-dbtype', dbtype,
               '-out', db_prefix, '-title', title]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                return db_prefix
        except (OSError, subprocess.SubprocessError):
            pass
    # For builtin search, just return the FASTA file path as the "database"
    return sequence_file


def run_blast(query_file, database, program='blastn', evalue=1e-5,
              max_target_seqs=500, num_threads=1, extra_args=None):
    if not os.path.exists(query_file):
        return BlastResult(program=program, database=database,
                          query_length=0, message=f"Query not found: {query_file}")

    if _has_blast_plus() and os.path.exists(database):
        engine_msg = "Using BLAST+ (external)"
        result = _blast_plus_search(query_file, database, program, evalue,
                                    max_target_seqs, num_threads)
        result.message = engine_msg
        return result
    else:
        warnings.warn(
            "BLAST+ not found. Using built-in k-mer search engine. "
            "For production use, install BLAST+ (https://blast.ncbi.nlm.nih.gov/) "
            "for 10-100x speedup and accurate E-values.",
            PerformanceWarning, stacklevel=2
        )
        result = _builtin_search(query_file, database, evalue, max_target_seqs)
        result.message = "Using built-in k-mer search engine (BLAST+ recommended for production)"
        return result


def run_blast_quick(query_seq, program='blastn', database='nt', evalue=1e-5):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(f">query\n{query_seq}\n")
        query_file = f.name
    try:
        return run_blast(query_file, database, program=program, evalue=evalue)
    finally:
        os.unlink(query_file)


def format_blast_result(result, max_hits=20):
    if result is None:
        return "No BLAST results available."
    lines = [
        f"Engine: {result.engine}",
        f"Program: {result.program}",
        f"Database: {result.database}",
        f"Query length: {result.query_length} bp",
        f"Total hits: {result.num_hits}",
        ""
    ]
    if not result.hits:
        lines.append("No significant hits found.")
        return '\n'.join(lines)
    lines.append(f"{'#':<4} {'Subject ID':<20} {'Identity%':>10} {'E-value':>12} {'Score':>8} {'Length':>8}")
    lines.append("-" * 72)
    for i, hit in enumerate(result.top_hits(max_hits)):
        lines.append(
            f"{i+1:<4} {hit.subject_id[:20]:<20} "
            f"{hit.percent_identity:>9.1f}% "
            f"{hit.e_value:>12.2e} "
            f"{hit.bit_score:>8.0f} "
            f"{hit.alignment_length:>8}"
        )
    if result.num_hits > max_hits:
        lines.append(f"\n... and {result.num_hits - max_hits} more hits.")
    return '\n'.join(lines)
