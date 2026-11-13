"""
Metagenomics analysis with dual-mode execution.

Pure Python k-mer based taxonomic classifier as default, Kraken2 as optional.
Includes diversity metrics and abundance analysis.
"""
import os
import subprocess
import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .log import get_logger
from .utils import (
    PerformanceWarning,
    has_tool as _has_tool,
    secure_temp_path,
    wrote_output,
)

logger = get_logger(__name__)


@dataclass
class TaxonomyResult:
    engine: str
    classifications: list = field(default_factory=list)
    abundance_table: pd.DataFrame | None = None
    message: str = ""


@dataclass
class DiversityResult:
    # One dict per sample, in input order — a list, not a mapping.
    alpha_diversity: list = field(default_factory=list)
    beta_diversity: pd.DataFrame | None = None
    message: str = ""


# The engines classify_reads() will work with.  'auto' prefers kraken2 when it is
# installed and falls back to the built-in classifier; 'builtin' never looks at
# the PATH.
TOOLS = ('auto', 'builtin', 'kraken2')

# Taxa shown by format_metagenomics_report() before it says how many it left out.
TAXA_REPORT_ROWS = 10

ABUNDANCE_COLUMNS = ('taxon', 'count', 'relative_abundance')

def check_metagenomics_tools():
    """Which external metagenomics engines this machine can run.

    Only kraken2 is wrapped; the built-in classifier needs nothing beyond this
    package, so ``classify_reads(tool='builtin')`` never calls in here.
    """
    return {'kraken2': _has_tool('kraken2')}


# ── Pure Python Taxonomic Classifier ────────────────────────────────────────

KNOWN_TAXA = {
    'ATCGATCGATCGATCG': 'Escherichia coli',
    'GCTAGCTAGCTAGCTA': 'Staphylococcus aureus',
    'TTTTAAAACCCCGGGG': 'Bacillus subtilis',
    'AAAACCCCGGGGTTTT': 'Lactobacillus acidophilus',
}

# Every signature is the same length, and the classifier walks the read with a
# window of exactly that length; test_signature_table_is_uniform_length() is what
# keeps this number honest if the table is ever edited.
SIGNATURE_KMER_LENGTH = len(next(iter(KNOWN_TAXA)))


def _check_kmer_length(k):
    """Refuse a window length the signature table can never satisfy."""
    if k is None:
        return SIGNATURE_KMER_LENGTH
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a whole k-mer length of 1 or more, got {k!r}")
    if k != SIGNATURE_KMER_LENGTH:
        raise ValueError(
            f"the built-in classifier matches {SIGNATURE_KMER_LENGTH}-mer signatures, "
            f"so k={k} would classify nothing; leave k at its default")
    return k


def _read_fastq_records(reads_file):
    """Yield (read_id, sequence) from a FASTQ file.

    The header must start with '@' and the separator with '+': the previous
    reader took any four lines in order, so a plain FASTA alignment was parsed
    into read ids like '>s1' and half of its records were silently swallowed by
    the lines it mistook for qualities.
    """
    with open(reads_file, encoding="utf-8") as f:
        first_line = True
        while True:
            header = f.readline()
            if not header:
                break
            header = header.strip()
            if first_line:
                first_line = False
                if not header.startswith('@'):
                    raise ValueError(
                        f"{reads_file} is not a FASTQ file: its first line is "
                        f"{header[:40]!r}, which does not start with '@'")
            elif not header.startswith('@'):
                _warn_unread(f, reads_file, "a read header without its leading '@'")
                break
            seq = f.readline().strip()
            separator = f.readline().strip()
            f.readline()
            if not seq:
                _warn_unread(f, reads_file, "a record with an empty sequence line")
                break
            if not separator.startswith('+'):
                _warn_unread(f, reads_file,
                             f"a quality separator line that reads {separator[:20]!r}")
                break
            yield header.lstrip('@').split()[0], seq


def _warn_unread(handle, reads_file, reason):
    """Say that the file was abandoned mid-way, and how much of it was left."""
    unread = sum(1 for _ in handle)
    logger.warning("%s stopped being read after %s; %d line(s) of the file were "
                   "not classified", reads_file, reason, unread)


def _builtin_classify(reads_file, k=None):
    """Assign each read to the signature taxon with the most k-mer votes.

    Votes are counted PER READ against the signature table; comparing a read
    against the running totals of the whole file made every later read collapse
    onto whichever taxon happened to be first.
    """
    k = _check_kmer_length(k)
    classifications = []
    taxon_counts = Counter()  # READS per taxon (not k-mer hits)

    for read_id, seq in _read_fastq_records(reads_file):
        seq = seq.upper()
        read_votes = Counter()
        for i in range(len(seq) - k + 1):
            taxon = KNOWN_TAXA.get(seq[i:i + k])
            if taxon:
                read_votes[taxon] += 1

        best_taxon = "Unknown"
        if read_votes:
            best_taxon = read_votes.most_common(1)[0][0]
            taxon_counts[best_taxon] += 1

        classifications.append({'read_id': read_id, 'taxon': best_taxon})

    total = len(classifications)
    # Abundance = share of READS (k-mer hits produced arbitrary percentages
    # that could sum to far more than 100%).
    df = _abundance_frame(taxon_counts, total)
    assigned = sum(taxon_counts.values())
    if total and not assigned:
        logger.warning("None of the %d read(s) in %s matched a %d-mer signature; "
                       "the built-in classifier knows %d reference taxon(s) only",
                       total, reads_file, k, len(KNOWN_TAXA))

    return TaxonomyResult(
        engine='builtin',
        classifications=classifications,
        abundance_table=df,
        message=f"Built-in classifier: {total} reads, {len(taxon_counts)} taxa"
    )


def _abundance_frame(taxon_counts, total):
    """Build the abundance table, keeping its columns even when it has no rows."""
    if not taxon_counts:
        return pd.DataFrame(columns=list(ABUNDANCE_COLUMNS))
    return pd.DataFrame([
        {'taxon': taxon, 'count': count, 'relative_abundance': count / total * 100}
        for taxon, count in taxon_counts.most_common()
    ])


# ── Kraken2 Wrapper ─────────────────────────────────────────────────────────

def _kraken2_classify(reads_file, database=None, output_file=None):
    """Run kraken2 and parse its report, or explain why it did not.

    Returns ``(result, failure)``.  A bare ``None`` used to be indistinguishable
    from "kraken2 is not installed", so a kraken2 that was found and then timed
    out was reported to the user as a missing program.
    """
    # Reserve the output path once: ``secure_temp_path`` creates the file, so
    # parsing it after a zero exit says nothing unless wrote_output() checks it.
    out_path = output_file or secure_temp_path()
    cmd = ['kraken2', '--threads', '1', '--output', out_path]
    if database:
        cmd.extend(['--db', database])
    cmd.append(reads_file)
    result = None
    failure = ""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or '').strip().splitlines()[-1:]
            failure = f"it exited with {r.returncode}: {tail[0] if tail else 'no output'}"
        elif not wrote_output(out_path):
            failure = "it exited cleanly but wrote nothing to its report"
        else:
            try:
                result = _parse_kraken2_output(out_path)
            except (OSError, UnicodeDecodeError) as exc:
                failure = f"its report could not be read: {exc}"
    except (OSError, subprocess.SubprocessError) as exc:
        failure = f"it could not be executed: {exc}"
    finally:
        # The reserved path belongs to this call only.  Leaving it behind would
        # cost one file per run that failed, and a caller who named a file keeps
        # it.
        if output_file is None:
            try:
                os.remove(out_path)
            except OSError as exc:
                logger.warning("Could not remove the reserved kraken2 report %s: %s",
                               out_path, exc)
    return result, failure


def _parse_kraken2_output(output_file):
    """Read a kraken2 ``--output`` report.

    The per-read lines are ``<status>\\t<read id>\\t<taxid>\\t<length>\\t<k-mer
    hits>``; the assignment lives in column 3 and the taxid is a number, not a
    name.  Reading parts[4] handed back the k-mer hit list (e.g. "561:0 561:1")
    as the taxon, and requiring five columns dropped every 4-field unclassified
    read.

    ``C`` rows are classified reads, ``U`` rows are unclassified reads, and ``D``
    rows are clade summaries that name no read at all — counting a ``D`` row as a
    read invented a read whose id was a taxid.
    """
    classifications = []
    taxon_counts = Counter()
    unclassified = 0
    summary_rows = 0
    malformed = 0
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            status = parts[0]
            if status in ('C', 'U'):
                if len(parts) < 3:
                    malformed += 1
                    continue
                read_id, taxid = parts[1], parts[2]
                if status == 'C':
                    taxon_counts[taxid] += 1
                else:
                    unclassified += 1
                classifications.append({'read_id': read_id, 'taxon': taxid})
            elif status == 'D':
                summary_rows += 1
            else:
                malformed += 1

    if malformed or summary_rows:
        logger.warning(
            "%s: %d clade-summary row(s) and %d row(s) that are not kraken2 read "
            "rows (unknown status or too few fields) were not counted as reads",
            output_file, summary_rows, malformed)

    total = len(classifications)
    classified = sum(taxon_counts.values())
    df = _abundance_frame(taxon_counts, total)
    message = f"Kraken2: {total} reads ({classified} classified, {unclassified} not)"
    if summary_rows:
        message += f"; {summary_rows} clade-summary line(s) ignored"

    return TaxonomyResult(
        engine='kraken2',
        classifications=classifications,
        abundance_table=df,
        message=message
    )


# ── Diversity Metrics ───────────────────────────────────────────────────────

def _counts_vector(counts):
    """Validate an abundance vector before any metric reads it.

    A negative count used to be folded into the probabilities of
    ``shannon_entropy`` and made ``simpson_index`` return a value outside its own
    [0, 1] range; a NaN silently dropped its taxon from the sum.
    """
    try:
        vector = np.asarray(counts, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"abundance counts must be numbers, got {counts!r}: {exc}") \
            from exc
    if vector.ndim != 1:
        raise ValueError(
            f"abundance counts must be a flat sequence, got a {vector.ndim}-D array")
    for index, value in enumerate(vector):
        if not np.isfinite(value):
            raise ValueError(f"abundance counts must be finite; index {index} is {value!r}")
        if value < 0:
            raise ValueError(
                f"abundance counts are read counts and cannot be negative; "
                f"index {index} is {value!r}")
    return vector


def _round_result(value) -> float:
    """A Python float, and 0.0 rather than the -0.0 numpy produces for a zero sum."""
    value = float(value)
    return 0.0 if value == 0 else value


def shannon_entropy(counts) -> float:
    vector = _counts_vector(counts)
    total = vector.sum()
    if total == 0:
        return 0.0
    p = vector / total
    p = p[p > 0]
    return _round_result(-np.sum(p * np.log(p)))


def simpson_index(counts) -> float:
    vector = _counts_vector(counts)
    total = vector.sum()
    if total <= 1:
        return 0.0
    return _round_result(1 - np.sum(vector * (vector - 1)) / (total * (total - 1)))


def chao1_estimator(counts) -> float:
    """Chao1 richness: the singleton/doubleton counts decide the extrapolation.

    Fractional values simply do not form singletons or doubletons, so an estimate
    from a non-integer vector collapses onto the observed richness; the metric is
    defined on read counts.
    """
    vector = _counts_vector(counts)
    s_obs = np.sum(vector > 0)
    f1 = np.sum(vector == 1)
    f2 = np.sum(vector == 2)
    if f2 == 0:
        return _round_result(s_obs + f1 * (f1 - 1) / 2)
    return _round_result(s_obs + f1 ** 2 / (2 * f2))


def bray_curtis_distance(sample1, sample2) -> float:
    s1 = _counts_vector(sample1)
    s2 = _counts_vector(sample2)
    if s1.shape != s2.shape:
        raise ValueError(
            f"the two samples must hold the same number of taxa, got "
            f"{s1.size} and {s2.size}; align them on a shared taxon list first")
    total = s1.sum() + s2.sum()
    if total == 0:
        return 0.0
    return _round_result(np.sum(np.abs(s1 - s2)) / total)


def _require_columns(table, columns, what):
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(
            f"{what} needs the column(s) {', '.join(missing)}; this table has "
            f"{list(table.columns) or 'no columns at all'}")


def compute_alpha_diversity(abundance_table):
    if abundance_table is None or abundance_table.empty:
        return {}
    _require_columns(abundance_table, ['count'], "an abundance table")
    counts = abundance_table['count'].values
    return {
        'observed_taxa': int(np.sum(counts > 0)),
        'shannon': round(shannon_entropy(counts), 4),
        'simpson': round(simpson_index(counts), 4),
        'chao1': round(chao1_estimator(counts), 4),
    }


def compute_beta_diversity(sample_tables):
    """Pairwise Bray-Curtis distances between samples.

    Counts are aligned on the UNION of taxon names — zipping raw position
    vectors (the old code) compared mismatched taxa whenever the samples
    had different taxon sets or order, corrupting every distance.

    A sample with no rows contributes an all-zero vector, which is the truth
    about it; a sample that has rows but no taxon/count columns is a malformed
    table and raises instead of raising a bare KeyError from pandas.
    """
    tables = list(sample_tables)
    present = []
    taxa = set()
    for table in tables:
        if table.empty:
            # A sample with no rows contributes nothing to the taxon union and
            # an all-zero vector below; it has no columns to check either.
            present.append(None)
            continue
        _require_columns(table, ['taxon', 'count'], "a sample table")
        taxa.update(table['taxon'].values)
        present.append(table)

    n = len(tables)
    ordered_taxa = sorted(taxa)
    taxon_index = {t: i for i, t in enumerate(ordered_taxa)}

    aligned = []
    for table in present:
        vec = np.zeros(len(ordered_taxa))
        if table is not None:
            for taxon, count in zip(table['taxon'].values, table['count'].values):
                vec[taxon_index[taxon]] += count
        aligned.append(vec)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = bray_curtis_distance(aligned[i], aligned[j])
            dist_matrix[i][j] = dist_matrix[j][i] = d
    return dist_matrix


# ── Public API ──────────────────────────────────────────────────────────────

def classify_reads(reads_file, database=None, tool='auto'):
    """Classify the reads of a FASTQ file.

    Args:
        reads_file: path to a FASTQ file.
        database: kraken2 database directory, passed through as ``--db``.
        tool: one of ``TOOLS``.  A name that is not in it raises, because a
            misspelled engine used to be answered with the built-in classifier
            and a warning that blamed kraken2 for being missing.

    Returns:
        TaxonomyResult.  ``engine`` says which classifier actually ran.
    """
    if tool not in TOOLS:
        raise ValueError(f"tool must be one of {', '.join(TOOLS)}, got {tool!r}")

    if not os.path.exists(reads_file):
        return TaxonomyResult(engine='none', message=f"File not found: {reads_file}")

    if tool == 'builtin':
        return _builtin_classify(reads_file)

    failure = None
    if check_metagenomics_tools()['kraken2']:
        result, failure = _kraken2_classify(reads_file, database)
        if result is not None:
            return result

    if failure is None:
        warnings.warn(
            "Kraken2 not found. Using built-in k-mer classifier. "
            "For production use, install Kraken2 (https://github.com/DerrickWood/kraken2) "
            "for accurate taxonomic classification.",
            PerformanceWarning, stacklevel=2
        )
    else:
        warnings.warn(
            f"Kraken2 is installed but did not classify {reads_file} ({failure}). "
            f"Using the built-in {SIGNATURE_KMER_LENGTH}-mer classifier, which "
            f"recognises {len(KNOWN_TAXA)} taxon(s) only.",
            PerformanceWarning, stacklevel=2
        )
    return _builtin_classify(reads_file)


def analyze_diversity(abundance_tables):
    """Per-sample alpha diversity plus the pairwise beta matrix.

    A sample with no abundance rows keeps the same metric keys as every other
    sample, all of them legitimately zero, and is counted in the message so the
    caller can tell "no data" from "a monoculture".
    """
    tables = list(abundance_tables)
    results = []
    empty = 0
    for i, table in enumerate(tables):
        alpha = compute_alpha_diversity(table)
        if not alpha:
            empty += 1
            alpha = {'observed_taxa': 0, 'shannon': 0.0, 'simpson': 0.0, 'chao1': 0.0}
        alpha['sample'] = f"Sample_{i+1}"
        results.append(alpha)

    beta = compute_beta_diversity(tables) if len(tables) > 1 else None

    message = f"Analyzed {len(tables)} samples"
    if empty:
        message += f" ({empty} with no abundance data)"

    return DiversityResult(
        alpha_diversity=results,
        beta_diversity=pd.DataFrame(beta) if beta is not None else None,
        message=message
    )


def format_metagenomics_report(result):
    lines = [
        "=== Metagenomics Report ===",
        f"Engine: {result.engine}",
        f"Reads: {len(result.classifications)}",
    ]
    table = result.abundance_table
    has_rows = table is not None and not table.empty
    if has_rows:
        assigned = int(table['count'].sum())
        lines.append(f"Assigned to a taxon: {assigned} of {len(result.classifications)} read(s)")
    elif result.classifications:
        lines.append("Assigned to a taxon: 0 read(s)")
    if has_rows:
        label = "Top taxa (kraken2 taxids):" if result.engine == 'kraken2' else "Top taxa:"
        lines.append(f"\n{label}")
        shown = table.head(TAXA_REPORT_ROWS)
        for _, row in shown.iterrows():
            lines.append(f"  {row['taxon']}: {row['relative_abundance']:.1f}%")
        hidden = len(table) - len(shown)
        if hidden:
            lines.append(f"  ... and {hidden} more taxon(s).")
    if result.message:
        lines.append(f"\nNote: {result.message}")
    return '\n'.join(lines)


# ── 16S rRNA Pipeline ────────────────────────────────────────────────────────

# Simplified 16S rRNA reference database (common gut bacteria)
SILVA_16S_DB = {
    'Escherichia coli': 'TGGAGGAAGGTGGGGACGACGTCAGTATCGAATCTTGGATCAGGATCACCTCCGGA',
    'Staphylococcus aureus': 'AGCCATGCAGCACCTGTCTCAGCTTCCCGAAGGCACTATACGTAGATCGAAAGTTGAT',
    'Bacillus subtilis': 'TGGAGAGTTTGATCATGGCTCAGATTGAACGCTGGCGGCAACCCTGATACAGGAT',
    'Lactobacillus acidophilus': 'TGCGGTCGACCGTCTGGAAAGTCACCTTCTTTCCGGATCGAAAGTTGATGGCTCAT',
    'Clostridium difficile': 'TGGGGAATATTGGGCAATGGGGGGAACCCTGACCCAGCAATGCCGCGTGTGTGAAGA',
    'Pseudomonas aeruginosa': 'TGGAGAGTTTGATCCTGGCTCAGATTGAACGCTGGCGGTAATCCTGATACAGGAT',
    'Streptococcus pneumoniae': 'ATGGAGAGTTTGATCCTGGCTCAGGATGAACGCTGGCGGTATCCTGATACAGGAT',
    'Helicobacter pylori': 'TGAGGATGAAGGCTTGAGGCTTAACCTGGTGAATTTTGGCTTAGATCGAAAGTTGAT',
}

# The window the identity scorer slides; a read shorter than this has no window
# of its own, so its "identity" is a prefix comparison of a few bases.
IDENTITY_WINDOW = 20


def _read_16s_fasta(path):
    """Parse (name, sequence) pairs from a FASTA file.

    A header line of bare '>' used to raise IndexError from inside the parser,
    and sequence lines before the first header were dropped in silence.
    """
    records = []
    name = None
    seq = ""
    orphan_lines = 0
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                tokens = line[1:].split()
                if not tokens:
                    raise ValueError(f"{path} line {number}: a FASTA header with no name")
                if name:
                    records.append((name, seq))
                name = tokens[0]
                seq = ""
            elif name is None:
                orphan_lines += 1
            else:
                seq += line
    if name:
        records.append((name, seq))
    if orphan_lines:
        logger.warning("%s: %d sequence line(s) before the first FASTA header "
                       "belong to no record and were not classified",
                       path, orphan_lines)
    return records


def classify_16s_rna(sequences, db=None):
    """Classify 16S rRNA sequences against a reference database.

    Args:
        sequences: list of (name, sequence) tuples or FASTA file path.
        db: reference database dict {name: 16S sequence}. Uses built-in if None.

    Returns:
        TaxonomyResult with classifications.  ``status`` is 'too_short' when a
        read is shorter than the scoring window: its identity is a prefix
        comparison and cannot support a call.
    """
    if db is None:
        db = SILVA_16S_DB
    if not isinstance(db, dict):
        raise TypeError(f"db must be a mapping of taxon to reference sequence, "
                        f"got {type(db).__name__}")
    if not db:
        logger.warning("classify_16s_rna() was given an empty reference database; "
                       "no sequence can match it")

    if isinstance(sequences, str):
        # Read from file
        if not os.path.exists(sequences):
            return TaxonomyResult(engine="16S", message=f"File not found: {sequences}")
        seqs = _read_16s_fasta(sequences)
    else:
        seqs = list(sequences)

    classifications = []
    for name, seq in seqs:
        best_match = "Unclassified"
        best_identity = 0.0
        seq_upper = str(seq).upper()

        for taxon, ref_seq in db.items():
            identity = _compute_identity(seq_upper, str(ref_seq).upper())
            if identity > best_identity:
                best_identity = identity
                best_match = taxon

        if len(seq_upper) < IDENTITY_WINDOW:
            status = "too_short"
        elif best_identity > 0.7:
            status = "classified"
        else:
            status = "low_confidence"

        classifications.append({
            "name": name,
            "taxonomy": best_match,
            "identity": best_identity,
            "confidence": best_identity * 100,
            "status": status
        })

    # Build abundance table
    taxon_counts = Counter(c["taxonomy"] for c in classifications)
    total = len(classifications)
    abundance = [
        {"taxon": taxon, "count": count, "relative_abundance": count / total * 100}
        for taxon, count in taxon_counts.most_common()
    ]

    return TaxonomyResult(
        engine="16S",
        classifications=classifications,
        abundance_table=pd.DataFrame(abundance, columns=list(ABUNDANCE_COLUMNS)),
        message=f"Classified {len(seqs)} 16S sequences"
    )


def _compute_identity(seq1, seq2, window=IDENTITY_WINDOW):
    """Best fraction of matching positions over any aligned window.

    This is a local-similarity score, not an alignment identity: it is the best
    of every window pair, which is why a short query can score 1.0 against a
    reference it barely resembles.  classify_16s_rna() labels those reads
    'too_short' instead of calling them classified.

    A window shorter than one base divides by zero.
    """
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError(f"window must be a whole number of bases, 1 or more, "
                         f"got {window!r}")
    if len(seq1) < window or len(seq2) < window:
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0
        matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a == b)
        return matches / min_len

    best = 0
    for i in range(len(seq1) - window + 1):
        chunk = seq1[i:i + window]
        # Find best alignment in seq2
        for j in range(len(seq2) - window + 1):
            matches = sum(1 for a, b in zip(chunk, seq2[j:j + window]) if a == b)
            identity = matches / window
            if identity > best:
                best = identity
    return best


DIVERSITY_METHODS = ('shannon', 'simpson', 'observed', 'chao1')


def alpha_diversity_single(sequence_list, method="shannon"):
    """Compute alpha diversity for a single sample.

    Args:
        sequence_list: the taxa assigned to one sample, one entry per read.
        method: one of ``DIVERSITY_METHODS``.  Anything else raises; a typo used
            to be answered with 0.0, which is a real diversity value and a lie
            about the name.

    Returns:
        float diversity value (0.0 for an empty sample).
    """
    if method not in DIVERSITY_METHODS:
        raise ValueError(f"method must be one of {', '.join(DIVERSITY_METHODS)}, "
                         f"got {method!r}")

    counts = Counter(sequence_list)
    n = sum(counts.values())
    if n == 0:
        return 0.0

    if method == "shannon":
        entropy = 0.0
        for count in counts.values():
            # A Counter built from a list only holds positive counts, so p > 0 here
            # and no zero-log guard is needed.
            p = count / n
            entropy -= p * np.log(p)
        return _round_result(entropy)

    if method == "simpson":
        total = n * (n - 1)
        if total == 0:
            return 0.0
        return 1 - sum(c * (c - 1) for c in counts.values()) / total

    if method == "observed":
        return len(counts)

    s = len(counts)
    f1 = sum(1 for c in counts.values() if c == 1)
    f2 = sum(1 for c in counts.values() if c == 2)
    if f2 > 0:
        return s + (f1 * f1) / (2 * f2)
    if f1 > 0:
        return s + f1 * (f1 - 1) / 2
    return s


def format_16s_report(result):
    """Format 16S rRNA classification results."""
    lines = [
        "=== 16S rRNA Classification Report ===",
        f"Engine: {result.engine}",
        f"Sequences read: {len(result.classifications)}",
    ]
    if result.abundance_table is not None and not result.abundance_table.empty:
        lines.append("\nTaxonomic Composition:")
        for _, row in result.abundance_table.iterrows():
            bar = '#' * int(row['relative_abundance'] / 2)
            lines.append(f"  {row['taxon']:<30} {row['relative_abundance']:5.1f}% {bar}")
    if result.classifications:
        low = sum(1 for c in result.classifications
                  if c.get('status') in ('low_confidence', 'too_short'))
        if low:
            lines.append(f"\nWarning: {low} sequences not confident enough to call")
        short = sum(1 for c in result.classifications if c.get('status') == 'too_short')
        if short:
            lines.append(f"  {short} of them are shorter than the {IDENTITY_WINDOW}-base "
                         f"scoring window, so their identity is a prefix comparison")
    if result.message:
        lines.append(f"\nNote: {result.message}")
    return "\n".join(lines)
