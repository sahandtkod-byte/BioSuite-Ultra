"""
Sequence similarity search with dual-mode execution.

Uses BLAST+ if installed for speed, otherwise falls back to a pure Python
k-mer seeded search over a FASTA file.  The fallback is a nucleotide-only,
approximate engine: it finds seeded local alignments and estimates E-values with
fixed Karlin-Altschul constants, and it cannot read a BLAST database, run a
protein program, or search a remote name like ``nt``.  A caller who needs those
needs BLAST+ on ``PATH``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from xml.parsers.expat import ExpatError

import numpy as np

try:
    from Bio import SeqIO
    from Bio.Align import PairwiseAligner
    from Bio.Blast import NCBIXML
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

from .log import get_logger
from .utils import PerformanceWarning, has_tool as _has_tool, wrote_output

logger = get_logger(__name__)

__all__ = [
    "check_blast_installed",
    "make_database",
    "run_blast",
    "run_blast_quick",
    "format_blast_result"
]

#: The programs BLAST+ ships.  ``run_blast`` refuses anything outside this set,
#: because a typo used to be answered by running the built-in engine instead.
BLAST_PROGRAMS = ('blastn', 'blastp', 'blastx', 'tblastn', 'tblastx')




@dataclass
class BlastHit:
    """A single BLAST hit against one database subject sequence.

    Attributes:
        query_id: Name of the query sequence this hit came from.
        subject_id: Database sequence identifier.
        subject_description: Full subject title (BLAST+ only; the built-in engine
            has no description to offer and repeats the id).
        percent_identity: Identical bases as a percentage of `alignment_length`.
        alignment_length: Columns in the aligned segment.  BLAST+ reports the HSP's
            `align_length`; the built-in engine scores a gapless local alignment, so
            its length is matches plus mismatches (see `gap_opens`).
        mismatches: Aligned columns that are not identical.
        gap_opens: Gap columns in the alignment.  The built-in engine scores with
            match +1 / mismatch -1 / gap -2, a scheme under which opening a gap is
            never cheaper than a run of mismatches, so its alignments are gapless
            and this is always 0 there.
        query_start, query_end: 1-based inclusive span on the query.
        subject_start, subject_end: 1-based inclusive span on the subject.
        e_value: Expect value — read from BLAST+ when it ran, estimated from fixed
            Karlin-Altschul constants by the built-in engine.
        bit_score: Bit score (BLAST+), or the raw alignment score (built-in engine,
            which has no normalized bit score to offer).
        query_cov: Aligned length as a percentage of the query length.
    """
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

    def __str__(self) -> str:
        """Compact one-line representation used in reports."""
        return (f"{self.subject_id} | {self.percent_identity:.1f}% identity | "
                f"E={self.e_value:.2e} | Score={self.bit_score:.0f}")


@dataclass
class BlastResult:
    """Aggregated BLAST search output for one query.

    Attributes:
        program: The program that was asked for (`blastn`, `blastp`, …).  It used to
            be the literal ``'builtin_search'`` on the built-in path, which threw
            away the caller's choice.
        database: Base name of the file or database searched.
        query_length: Length of the first query sequence, 0 when nothing was parsed.
        hits: `BlastHit` list in ascending `e_value` order — the order both engines
            leave them in, not a bit-score order (the two agree for a fixed query
            against a fixed database, which is why nothing showed the difference).
        message: The engine line, plus whatever the engine itself reported.
        engine: ``'blast+'`` or ``'builtin'`` — the engine that ran.
    """
    program: str
    database: str
    query_length: int
    hits: List[BlastHit] = field(default_factory=list)
    message: str = ""
    engine: str = "builtin"

    @property
    def num_hits(self) -> int:
        """Number of hits retained in this result."""
        return len(self.hits)

    def top_hits(self, n: int = 10) -> List[BlastHit]:
        """Return the first n hits, i.e. the lowest E-values.

        It used to promise "sorted by bit score" and then slice whatever order the
        engine left, which is the same list for a fixed query/database pair but is
        not what the sentence said.
        """
        return self.hits[:n]

    def significant_hits(self, evalue_threshold: float = 1e-5) -> List[BlastHit]:
        """Hits passing an E-value cutoff.

        Args:
            evalue_threshold: Maximum acceptable E-value (default 1e-5).  The
                boundary itself counts as acceptable; ``h.e_value == threshold`` used
                to be dropped by a strict ``<`` while the sentence said "maximum
                acceptable".
        """
        return [h for h in self.hits if h.e_value <= evalue_threshold]


# ── External BLAST+ Detection ──────────────────────────────────────────────

#: The binaries `check_blast_installed` reports on, in reporting order.
BLAST_TOOLS = ('blastn', 'blastp', 'blastx', 'tblastn', 'tblastx', 'makeblastdb')


def _has_blast_plus() -> bool:
    """Whether `blastn` is present and startable.

    This used to be a local `subprocess.run(['blastn', '-version'])` with
    `returncode == 0` as the definition of installed and only
    `FileNotFoundError`/`TimeoutExpired` caught.  Two consequences were measured: a
    BLAST+ build that exits non-zero for a version flag answered "not installed",
    and a `blastn` on ``PATH`` without the execute bit raised `PermissionError` out
    of `run_blast` itself — the caller got a traceback instead of a fallback.  The
    package's own probe answers both.
    """
    return _has_tool('blastn')


def check_blast_installed() -> Dict[str, bool]:
    """Public availability probe mirroring other core modules.

    One entry per BLAST+ binary.  A name that is nowhere on ``PATH`` costs no
    subprocess at all now: the previous version started six, one per tool, on every
    call, so a machine without BLAST+ paid six spawns to learn nothing.
    """
    return {tool: _has_tool(tool) for tool in BLAST_TOOLS}


# ── Pure Python K-mer Search Engine ─────────────────────────────────────────

def _build_kmer_index(sequences: List[Tuple[str, str]],
                      k: int = 15) -> Dict[str, List[Tuple[str, int]]]:
    """Build a k-mer to (subject id, offset) index over the database sequences.

    Used by the built-in seed-and-extend searcher as a lightweight alternative to a
    real BLAST install.  `sequences` is the ``(id, sequence)`` pairs `_read_fasta`
    returns — the annotation used to say ``List[str]``, which is not what the loop
    unpacks.

    Only ``N`` is skipped as uninformative.  A gap character in a database sequence
    is indexed like any other symbol, so ``-`` can seed a hit; that is why the
    built-in engine wants a gap-free database.
    """
    index = defaultdict(list)
    for seq_id, seq in sequences:
        seq_upper = seq.upper().replace('U', 'T')
        for i in range(len(seq_upper) - k + 1):
            kmer = seq_upper[i:i + k]
            if 'N' not in kmer:
                index[kmer].append((seq_id, i))
    return index


def _find_seed_hits(query: str, index: Dict[str, List[Tuple[str, int]]], k: int = 15,
                   max_hits: int = 200) -> List[Tuple[Tuple[str, int], List[int]]]:
    """Locate candidate match regions for the query via seeded lookup.

    Returns ``((subject_id, diagonal_offset), [query positions])``, most-seeded
    first, capped at 50 candidates.  `max_hits` bounds the number of *seed
    occurrences* examined on the way, not the number of returned candidates.
    """
    query_upper = query.upper().replace('U', 'T')
    hits = defaultdict(list)
    count = 0
    for i in range(len(query_upper) - k + 1):
        if count >= max_hits:
            break
        kmer = query_upper[i:i + k]
        for seq_id, db_pos in index.get(kmer, ()):
            hits[(seq_id, db_pos - i)].append(i)
            count += 1
    # Sort by number of seed hits (best first)
    return sorted(hits.items(), key=lambda x: len(x[1]), reverse=True)[:50]


def _banded_align_score(seq1: str, seq2: str, start1: int, start2: int, match: int = 1, mismatch: int = -1, gap: int = -2) -> Tuple[float, int]:
    """Quick banded local alignment score between two subsequences."""
    max_len = min(len(seq1) - start1, len(seq2) - start2, 500)
    if max_len <= 0:
        # No room to extend from these anchors.  Every other return below is a
        # 7-tuple (score, matches, mismatches, q_start, t_start, q_end, t_end)
        # and callers unpack seven, so this guard must yield seven too — it
        # used to yield six, which surfaced as
        # "ValueError: not enough values to unpack" out of _builtin_search.
        # Zero aligned columns make the hit unusable, which is what the
        # caller's ``align_len <= 0`` test is there to drop.
        return 0, 0, 0, start1, start2, start1, start2

    s1 = seq1[start1:start1 + max_len]
    s2 = seq2[start2:start2 + max_len]
    n, m = len(s1), len(s2)

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
                # Exact positions and identity from the alignment's aligned
                # coordinate blocks (the old code compared s through IDENTITY
                # — it only counted matches on UNALIGNED index positions,
                # making percent_identity essentially random).
                qal, tal = best.aligned
                matches = 0
                for (qs, qe), (ts, te) in zip(qal, tal):
                    matches += sum(1 for a, b in zip(
                        s1[qs:qe], s2[ts:te]) if a == b)
                aln_len = sum(qe - qs for qs, qe in qal)
                q_start = start1 + qal[0][0]
                q_end = start1 + qal[-1][1]
                t_start = start2 + tal[0][0]
                t_end = start2 + tal[-1][1]
                return float(best.score), int(matches), int(aln_len - matches), \
                    int(q_start), int(t_start), int(q_end), int(t_end)
        except (ImportError, AttributeError, TypeError) as exc:
            # Deliberate fall-back to the naive scorer, but not a silent one: the
            # two paths score differently (the naive one cannot find an offset past
            # the seed), and a caller comparing hit lengths across versions would
            # otherwise have no way to see which scorer produced the number.
            logger.debug("Biopython local alignment unavailable for this pair (%s: %s); "
                         "using the fallback scorer", type(exc).__name__, exc)

    # Fallback: simple scoring
    score = 0
    matches = 0
    for i in range(min(n, m)):
        if s1[i] == s2[i]:
            score += match
            matches += 1
        else:
            score += mismatch
    return score, matches, min(n, m) - matches, start1, start2, start1 + min(n, m), start2 + min(n, m)


def _estimate_evalue(score: float, db_size: int, query_len: int, k: int = 15) -> float:
    """Rough E-value approximation, in plain Python float.

    Karlin-Altschul with constants fitted to this scorer (match=+1, mismatch=-1,
    gap=-2).  The previous constants (0.332/0.133 are protein/BLOSUM62-ish values)
    grossly overestimated E-values for short built-in hits and caused true planted
    hits to be filtered out.  This is an estimate over `db_size` *bases* times the
    number of query windows; it is not BLAST's effective search space, so a built-in
    E-value is comparable to another built-in E-value and to nothing else.
    """
    if score <= 0:
        return 1.0
    lambda_param = 1.19
    K = 0.025
    effective_db = db_size * max(query_len - k + 1, 1)
    return float(K * effective_db * np.exp(-lambda_param * score))


def _read_fasta(filepath: str) -> List[Tuple[str, str]]:
    """Read a FASTA file into ``(id, sequence)`` pairs.

    Raises for a missing or unparsable file: the public entry point checks the paths
    first, and this function has no result shape in which to report it.
    """
    if not HAS_BIO:
        from .utils import read_fasta_simple
        return read_fasta_simple(filepath)
    return [(r.id, str(r.seq)) for r in SeqIO.parse(filepath, 'fasta')]


def _builtin_search(query_file: str, database_file: str, evalue: float = 1e-5,
                    max_hits: int = 500, k: int = 15,
                    program: str = 'blastn') -> BlastResult:
    """Pure-Python approximate search (seed-and-extend fallback).

    Slower but dependency-free; used when BLAST+ is not installed.  It is a
    nucleotide engine whatever `program` says — it has no translation table and no
    protein scoring matrix — so `run_blast` says so when a protein program reaches it.
    """
    queries = _read_fasta(query_file)
    db_seqs = _read_fasta(database_file)
    # First record wins on a repeated label, which is what the per-hit linear scan
    # through `db_seqs` did before it became a dict.
    db_by_id: Dict[str, str] = {}
    for seq_id, seq in db_seqs:
        db_by_id.setdefault(seq_id, seq)

    if not queries or not db_seqs:
        return BlastResult(program=program, database=os.path.basename(database_file),
                           query_length=0, message='No sequences found.', engine='builtin')

    total_db_len = sum(len(s) for _, s in db_seqs)
    first_query_len = len(queries[0][1])

    # Clamp k to the shortest sequence: a k-mer larger than every sequence
    # can never seed, silently returning ZERO hits for short inputs.
    shortest = min(len(s) for _, s in queries + db_seqs)
    clamp_note = ""
    if k > shortest:
        k = max(3, shortest)
        clamp_note = f'k clamped to {k} (longest possible seed)'
    index = _build_kmer_index(db_seqs, k=k)

    result = BlastResult(
        program=program,
        database=os.path.basename(database_file),
        query_length=first_query_len,
        engine='builtin',
        message=clamp_note,
    )

    for q_id, q_seq in queries:
        if len(result.hits) >= max_hits:
            break
        seed_hits = _find_seed_hits(q_seq, index, k=k)
        seen_subjects = set()

        for (subject_id, offset), positions in seed_hits:
            if subject_id in seen_subjects:
                continue
            seen_subjects.add(subject_id)

            db_seq = db_by_id.get(subject_id)
            if db_seq is None:
                continue

            # Anchor BOTH sequences to the same diagonal: the windows must
            # start at the SAME distance back from the seed.  The previous
            # "sync" applied max(0, x - slack) independently on each side —
            # when one side clamped at 0 the diagonals still diverged
            # (seed at db offset 3 anchored q=0/db=0 instead of q=0/db=3,
            # scoring a perfect planted hit as ~2 matches and dropping it).
            seed_pos = positions[0]
            shift = min(5, seed_pos, offset)   # shared back-shift
            q_start = seed_pos - shift
            db_start = offset - shift
            score, matches, mismatches, qs, ts, qe, te = _banded_align_score(
                q_seq, db_seq, q_start, db_start
            )

            # The scorer returns matches and non-matching aligned columns, so the
            # column count is their sum.  `min(qe - qs, te - ts)` — the span — is a
            # third number, and it was what `alignment_length` and the identity
            # percentage reported.  They coincide for this scorer (see
            # `BlastHit.gap_opens`: a -2 gap is never cheaper than a -1 mismatch, so
            # the local alignment is gapless), but the column count is the definition
            # BLAST uses, so the hit is built from it.
            columns = matches + mismatches
            if columns <= 0:
                continue

            computed_evalue = _estimate_evalue(score, total_db_len, len(q_seq), k=k)
            if computed_evalue > evalue:
                continue

            hit = BlastHit(
                query_id=q_id,
                subject_id=subject_id,
                subject_description=subject_id,
                percent_identity=matches / columns * 100,
                alignment_length=columns,
                mismatches=mismatches,
                gap_opens=0,
                query_start=qs + 1,
                query_end=qe,
                subject_start=ts + 1,
                subject_end=te,
                e_value=computed_evalue,
                bit_score=score,
                query_cov=columns / max(len(q_seq), 1) * 100
            )
            result.hits.append(hit)

            if len(result.hits) >= max_hits:
                break

    result.hits.sort(key=lambda x: x.e_value)
    return result


# ── External BLAST+ Execution ───────────────────────────────────────────────

def _blast_plus_search(query_file: str, database: str, program: str = 'blastn',
                       evalue: float = 1e-5, max_target_seqs: int = 500,
                       num_threads: int = 1,
                       extra_args: Optional[List[str]] = None) -> BlastResult:
    """Delegate the search to NCBI BLAST+ via subprocess.

    Reads the XML (-outfmt 5) report back with Biopython's NCBIXML parser into
    BlastHit objects.  Nothing here raises: the caller is a CLI menu option or a REST
    endpoint, and each failure is described in `message` instead — the report still
    says which engine ran, so a blast+ result with no hits and an error message can
    never be read as "the database has nothing in it".

    The parse step is the only broad catch in this function, and it is bounded by
    measurement: on Biopython 1.87 an empty report raises `ValueError`, non-XML or
    truncated input raises `UnboundLocalError` from inside the parser (it reads
    `expat_parser` before assigning it), and a report missing a section the parser
    expects arrives as `xml.parsers.expat.ExpatError` whose own message is the useless
    "parsing finished" with the real cause one level down.  Building the hits is kept
    outside that try, so a mistake in this module's own mapping is not laundered into
    a BLAST error — that is exactly how the `hsp.mismatches` AttributeError once
    reached the user as "no hits found".
    """
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as tmp:
        outfile = tmp.name

    cmd = [program, '-query', query_file, '-db', database,
           '-out', outfile, '-outfmt', '5', '-evalue', str(evalue),
           '-max_target_seqs', str(max_target_seqs),
           '-num_threads', str(num_threads)]
    if extra_args:
        # Appended after the fixed options so a caller can add flags BLAST+ would
        # not otherwise see.  The list form means no shell is involved.
        cmd.extend(extra_args)

    result = BlastResult(program=program, database=database,
                         query_length=0, engine='blast+')
    try:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            result.message = (f"{program} could not be started: "
                              f"{type(exc).__name__}: {exc}")
            logger.warning(result.message)
            return result
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or 'no output').strip()
            result.message = f"BLAST+ error ({program} exited {r.returncode}): {tail[:300]}"
            logger.warning(result.message)
            return result
        if not wrote_output(outfile):
            result.message = (f"{program} exited 0 but wrote no report to {outfile}; "
                              f"the database is probably not a BLAST+ database")
            logger.warning(result.message)
            return result

        try:
            with open(outfile, encoding="utf-8") as handle:
                records = list(NCBIXML.parse(handle))
        except (ValueError, NameError, ExpatError) as exc:
            cause = exc.__context__ or exc.__cause__
            detail = f"{type(exc).__name__}: {exc}"
            if cause is not None:
                detail += (f" (while reading the report: "
                           f"{type(cause).__name__}: {cause})")
            result.message = f"Could not parse the BLAST+ XML report: {detail}"
            logger.warning(result.message)
            return result

        if not records:
            result.message = ("BLAST+ produced a report with no query records "
                              f"(nothing matched at evalue={evalue:g})")
            logger.warning(result.message)
            return result

        result.query_length = int(records[0].query_length or 0)

        for record in records:
            for alignment in record.alignments:
                for hsp in alignment.hsps:
                    # Biopython's NCBIXML HSP has no ``mismatches`` attribute
                    # (it exposes identities, gaps and align_length only), so
                    # reading it raised AttributeError on every real hit.
                    # BLAST's own definition:
                    # align_length = identities + mismatches + gap columns.
                    mismatches = max(hsp.align_length - hsp.identities - hsp.gaps, 0)
                    hit = BlastHit(
                        query_id=record.query,
                        subject_id=alignment.accession or alignment.title.split()[0],
                        subject_description=alignment.title,
                        percent_identity=hsp.identities / hsp.align_length * 100 if hsp.align_length > 0 else 0.0,
                        alignment_length=int(hsp.align_length),
                        mismatches=int(mismatches),
                        gap_opens=int(hsp.gaps),
                        query_start=int(hsp.query_start),
                        query_end=int(hsp.query_end),
                        subject_start=int(hsp.sbjct_start),
                        subject_end=int(hsp.sbjct_end),
                        e_value=float(hsp.expect),
                        bit_score=float(hsp.bits),
                        query_cov=hsp.align_length / (record.query_length or 1) * 100
                    )
                    result.hits.append(hit)

        result.hits.sort(key=lambda x: x.e_value)
        return result

    finally:
        try:
            os.unlink(outfile)
        except FileNotFoundError:
            # BLAST+ itself can be configured to move the report; the file being gone
            # is the outcome the caller wants, not an error.
            pass
        except OSError as exc:
            # The search has already been answered by now.  A report this process
            # cannot delete (Windows: a handle still open) is a cleanup detail — it is
            # logged, not raised into the caller's result and not swallowed.
            logger.debug("Temporary BLAST report %s could not be removed (%s: %s)",
                         outfile, type(exc).__name__, exc)


# ── Public API ──────────────────────────────────────────────────────────────

def _engine_message(engine_line: str, own: str) -> str:
    """One message that keeps both what ran and what the engine itself reported.

    `run_blast` used to assign its engine line straight over `result.message`, which
    destroyed the two things the engines put there: the `BLAST+ error: …`/parse-failure
    text of the external path (a crash came back as "Using BLAST+ (external)" with no
    hits at all), and the `k clamped to 4 …` note of the built-in path (a user whose
    seeds were silently shortened could not see that it had happened).
    """
    return f"{engine_line}; {own}" if own else engine_line


def make_database(sequence_file: str, dbtype: str = 'nucl', output_dir: Optional[str] = None,
                  title: str = 'mydb') -> str:
    """Create a BLAST database from a FASTA file, and return what to pass as `database`.

    Args:
        sequence_file: Source FASTA path.
        dbtype: ``'nucl'`` or ``'prot'``.  Anything else is refused: `makeblastdb`
            prints its own complaint and exits non-zero, and the caller used to get a
            usable-looking path back with no word about either.
        output_dir: Where to build.  A directory created here is removed again when the
            build fails; failed builds used to leave ``blastdb_*`` directories in the
            system temp folder with nothing pointing at them.
        title: The title recorded in the database.

    Returns:
        str: The database prefix when BLAST+ built one, otherwise `sequence_file`
            itself — the built-in engine reads FASTA directly, which is what lets the
            same return value work in `run_blast` on either engine.
    """
    if dbtype not in ('nucl', 'prot'):
        raise ValueError(f"dbtype must be 'nucl' or 'prot', got {dbtype!r}.")
    if not os.path.isfile(sequence_file):
        raise FileNotFoundError(f"No FASTA to build a database from: {sequence_file!r}")
    if not _has_blast_plus():
        return sequence_file

    owns_dir = output_dir is None
    if owns_dir:
        output_dir = tempfile.mkdtemp(prefix='blastdb_')
    db_prefix = os.path.join(output_dir, 'blastdb')

    def _fall_back(reason: str) -> str:
        warnings.warn(f"{reason}; using {sequence_file!r} as the database with the "
                      f"built-in engine", PerformanceWarning, stacklevel=3)
        logger.warning(reason)
        if owns_dir:
            # ignore_errors because the build has already failed, and a file BLAST+
            # left open must not turn that into a second, unrelated error.
            shutil.rmtree(output_dir, ignore_errors=True)
        return sequence_file

    cmd = ['makeblastdb', '-in', sequence_file, '-dbtype', dbtype,
           '-out', db_prefix, '-title', title]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        return _fall_back(f"makeblastdb could not be started: {type(exc).__name__}: {exc}")
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or 'no output').strip()
        return _fall_back(f"makeblastdb exited {r.returncode}: {tail[:200]}")
    # BLAST+ writes the search index next to the prefix; a zero exit with no `.nin` is
    # the same "the tool said it worked" case `wrote_output` exists for.
    if not wrote_output(db_prefix + '.nin'):
        return _fall_back(f"makeblastdb exited 0 but left no {db_prefix}.nin")
    return db_prefix


def run_blast(query_file: str, database: str, program: str = 'blastn', evalue: float = 1e-5,
              max_target_seqs: int = 500, num_threads: int = 1,
              extra_args: Optional[List[str]] = None) -> BlastResult:
    """Run a BLAST search, auto-selecting the engine that is available.

    Prefers BLAST+ binaries; falls back to the built-in k-mer searcher so callers never
    need engine checks.  Every outcome comes back as a `BlastResult` — a bad path, an
    unreadable file or an unknown program is reported in `message`, not raised at the
    caller, because this is what the CLI menu option and the REST endpoint print.

    Args:
        query_file: FASTA of the query sequences.
        database: A BLAST+ database (a prefix, or a name like ``nt`` resolved through
            ``BLASTDB``) when BLAST+ is installed; otherwise a FASTA file, which is what
            `make_database` returns when there is no BLAST+ to build with.
        program: One of `BLAST_PROGRAMS`.  Anything else is refused — a typo used to be
            answered by running the built-in nucleotide engine and reporting
            ``program='builtin_search'``, so the caller's choice simply disappeared.
        evalue: Report threshold.
        max_target_seqs: Subject sequences to keep (BLAST+), and an upper bound on the
            built-in engine's whole hit list.
        num_threads: Passed to BLAST+; the built-in engine is single-threaded.
        extra_args: Appended to the BLAST+ command line.  The built-in engine has no
            equivalent options, so when it takes over the result names the flags that
            were not applied.
    """
    def _no_result(message: str) -> BlastResult:
        return BlastResult(program=program, database=os.path.basename(str(database)),
                           query_length=0, message=message, engine="builtin")

    if program not in BLAST_PROGRAMS:
        return _no_result(f"Unknown program {program!r}; use "
                          f"{' or '.join(repr(p) for p in BLAST_PROGRAMS)}.")
    if not os.path.isfile(query_file):
        return _no_result(f"Query not found: {query_file}")

    if _has_blast_plus():
        result = _blast_plus_search(query_file, database, program, evalue,
                                    max_target_seqs, num_threads, extra_args)
        result.message = _engine_message("Using BLAST+ (external)", result.message)
        return result

    if not os.path.isfile(database):
        return _no_result(
            f"The built-in engine searches a FASTA file, and {database!r} is not one. "
            f"Pass a FASTA path (or what make_database returned), or install BLAST+ to "
            f"search a named database.")

    warnings.warn(
        "BLAST+ not found. Using built-in k-mer search engine. "
        "For production use, install BLAST+ (https://blast.ncbi.nlm.nih.gov/) "
        "for 10-100x speedup and accurate E-values.",
        PerformanceWarning, stacklevel=2
    )
    try:
        result = _builtin_search(query_file, database, evalue, max_target_seqs,
                                 program=program)
    except (OSError, ValueError) as exc:
        return _no_result(f"Could not read the search inputs: "
                          f"{type(exc).__name__}: {exc}")

    notes = ["Using built-in k-mer search engine (BLAST+ recommended for production)"]
    if program != 'blastn':
        notes.append(f"{program} is a BLAST+ program: the built-in engine is a "
                     f"nucleotide-only k-mer search and did not apply it")
    if extra_args:
        notes.append(f"extra_args {list(extra_args)} are BLAST+ options and were not applied")
    notes.append(result.message)
    result.message = "; ".join(note for note in notes if note)
    return result


def run_blast_quick(query_seq: str, program: str = 'blastn', database: str = 'nt',
                    evalue: float = 1e-5) -> BlastResult:
    """One-liner convenience search of a raw query string.

    ``database`` defaults to ``'nt'``, which only means something to BLAST+ with
    ``BLASTDB`` pointing at a built copy of it; without BLAST+ installed the result
    says so instead of raising ``FileNotFoundError: 'nt'``.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False,
                                     encoding="utf-8") as handle:
        handle.write(f">query\n{query_seq}\n")
        query_file = handle.name
    try:
        return run_blast(query_file, database, program=program, evalue=evalue)
    finally:
        os.unlink(query_file)


def format_blast_result(result: BlastResult, max_hits: int = 20) -> str:
    """Format a BlastResult as a fixed-width text table."""
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
