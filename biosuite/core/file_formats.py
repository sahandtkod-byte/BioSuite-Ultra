"""
File format parsers: BED, GFF/GTF, Newick tree, Stockholm.

Pure Python parsers for common bioinformatics file formats.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .log import get_logger
from .utils import smart_open

logger = get_logger(__name__)

# htslib writes these four bytes at the head of a BAM index and a tabix index, so
# they are the only evidence available without linking against libhts.
BAM_INDEX_MAGIC = b"BAM\x01"
TABIX_INDEX_MAGIC = b"TBI\x01"
BIGWIG_MAGIC = b"\xc7\x4b\x6c\x27"
BIGWIG_REPORT_ROWS = 20
# A real BCF on disk is BGZF-compressed, so its raw head is the gzip variant
# below; the uncompressed BCF magic is five bytes (BCF + version 2.2).
_BCF_MAGIC = b"BCF\x02\x02"
_BGZF_MAGIC = b"\x1f\x8b\x08\x04"

# Options the universal reader actually forwards.  Every other format is read by a
# parser that takes only the path, so an option for it would be silently ignored.
READER_OPTIONS = {"bam": ("region",), "cram": ("region", "reference"),
                  "bigwig": ("chrom", "start", "end")}


def _report_skipped(filepath, count, kind):
    """Say once, per file, how many rows a parser refused to read.

    A record count the reader trusts is the whole product of these parsers, so a
    dropped row has to be reported rather than disappear into the total.
    """
    if count:
        logger.warning("%s: skipped %d row(s) that were not a usable %s record",
                       filepath, count, kind)


@dataclass
class BEDRecord:
    chrom: str
    start: int
    end: int
    name: str = ""
    score: float = 0
    strand: str = "."
    thick_start: int = 0
    thick_end: int = 0
    item_rgb: str = "0,0,0"


@dataclass
class GFFRecord:
    seqid: str
    source: str
    feature: str
    start: int
    end: int
    score: float
    strand: str
    phase: str
    attributes: dict = field(default_factory=dict)


@dataclass
class TreeNode:
    name: str = ""
    branch_length: float = 0.0
    children: list = field(default_factory=list)
    is_leaf: bool = True


def parse_bed(filepath):
    """Parse BED file into list of BEDRecord objects.

    Columns 1-9 are read, which covers BED3, BED6 and BED9; the three extra
    columns of a BED12 record (blockCount, blockSizes, blockSaved) have no place
    in ``BEDRecord`` and are dropped.  A row whose coordinates are not usable —
    not a number, a negative start, or an end before its start — is skipped and
    counted, because an inverted interval would otherwise subtract from the total
    length the summary reports.
    """
    records = []
    skipped = 0
    with smart_open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) < 3:
                skipped += 1
                continue
            try:
                start, end = int(parts[1]), int(parts[2])
                if start < 0 or end < start:
                    raise ValueError(f"invalid BED interval {start}-{end}")
                rec = BEDRecord(
                    chrom=parts[0],
                    start=start,
                    end=end,
                    name=parts[3] if len(parts) > 3 else "",
                    score=(float(parts[4]) if len(parts) > 4 and parts[4] != '.' else 0),
                    strand=parts[5] if len(parts) > 5 else ".",
                )
                if len(parts) > 6:
                    rec.thick_start = int(parts[6])
                if len(parts) > 7:
                    rec.thick_end = int(parts[7])
                if len(parts) > 8:
                    rec.item_rgb = parts[8]
            except ValueError:
                skipped += 1  # malformed row: counted, and the rest of the file survives
                continue
            records.append(rec)
    _report_skipped(filepath, skipped, "BED")
    return records


def _parse_attributes(field_text):
    """Split a GFF/GTF ninth column into attributes.

    Both separator styles are honoured: GFF3 writes ``key=value`` and GTF writes
    ``key "value"``, and real files mix them.  Reading only one of the two lost
    ``gene_id`` from an ``=``-style record without any signal that the attribute
    column had been ignored.
    """
    attrs = {}
    for item in field_text.split(';'):
        item = item.strip()
        if not item:
            continue
        if '=' in item:
            key, val = item.split('=', 1)
            attrs[key.strip()] = val.strip()
        elif ' ' in item:
            key, val = item.split(' ', 1)
            attrs[key.strip()] = val.strip().strip('"')
    return attrs


def _feature_row(parts):
    """Build a GFFRecord from nine tab-separated columns.

    Raises ValueError when the coordinates or the score are not numbers, which the
    callers count as a malformed row instead of letting it end the file.
    """
    return GFFRecord(
        seqid=parts[0],
        source=parts[1],
        feature=parts[2],
        start=int(parts[3]),
        end=int(parts[4]),
        score=float(parts[5]) if parts[5] != '.' else 0,
        strand=parts[6],
        phase=parts[7],
        attributes=_parse_attributes(parts[8]),
    )


def parse_gff(filepath):
    """Parse GFF3 file into list of GFFRecord objects.

    A row that does not carry nine columns, or whose coordinates or score are not
    numbers, is skipped and counted rather than aborting the file: one truncated
    annotation must not destroy an otherwise readable annotation set.
    """
    records = []
    skipped = 0
    with smart_open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 9:
                skipped += 1
                continue
            try:
                records.append(_feature_row(parts))
            except ValueError:
                skipped += 1
                continue
    _report_skipped(filepath, skipped, "GFF")
    return records


def parse_newick(newick_str):
    """Parse Newick format tree string into TreeNode objects.

    Quoted labels are honoured, with '' as the escaped quote, so a name
    containing a space, comma or colon survives instead of being mistaken for
    tree structure.

    Args:
        newick_str: Newick format string (e.g., "(A:0.1,B:0.2)C:0.3;")

    Returns:
        Root TreeNode.

    Raises:
        ValueError: if the string is empty, the parentheses are unbalanced, a
            branch length is not a number, or anything follows the first tree —
            a concatenated forest is not one tree, and reading only its first
            element silently drops the rest.
        TypeError: if the argument is not a string at all.
    """
    if not isinstance(newick_str, str):
        raise TypeError("parse_newick() needs a Newick string, got "
                        f"{type(newick_str).__name__}")
    newick_str = newick_str.strip().rstrip(';').strip()
    if not newick_str:
        raise ValueError("empty Newick string")
    pos = [0]

    def _parse():
        node = TreeNode()
        if pos[0] < len(newick_str) and newick_str[pos[0]] == '(':
            pos[0] += 1  # skip '('
            while pos[0] < len(newick_str) and newick_str[pos[0]] != ')':
                child = _parse()
                node.children.append(child)
                if pos[0] < len(newick_str) and newick_str[pos[0]] == ',':
                    pos[0] += 1
            if pos[0] >= len(newick_str):
                raise ValueError("unbalanced parentheses in Newick string")
            pos[0] += 1  # skip ')'

        # Read name and branch length
        name = ''
        bl = ''
        if pos[0] < len(newick_str) and newick_str[pos[0]] == "'":
            pos[0] += 1
            while pos[0] < len(newick_str):
                if newick_str[pos[0]] == "'":
                    if (pos[0] + 1 < len(newick_str)
                            and newick_str[pos[0] + 1] == "'"):
                        name += "'"
                        pos[0] += 2
                        continue
                    pos[0] += 1
                    break
                name += newick_str[pos[0]]
                pos[0] += 1
        else:
            while pos[0] < len(newick_str) and newick_str[pos[0]] not in '(),;:':
                name += newick_str[pos[0]]
                pos[0] += 1
        if pos[0] < len(newick_str) and newick_str[pos[0]] == ':':
            pos[0] += 1
            while pos[0] < len(newick_str) and newick_str[pos[0]] not in '(),;':
                bl += newick_str[pos[0]]
                pos[0] += 1

        node.name = name.strip()
        try:
            node.branch_length = float(bl) if bl else 0.0
        except ValueError as exc:
            raise ValueError(
                f"invalid branch length {bl!r} after label {name.strip()!r}") from exc
        node.is_leaf = len(node.children) == 0
        return node

    tree = _parse()
    rest = newick_str[pos[0]:]
    if rest.startswith(';'):
        rest = rest[1:]
    if rest.strip():
        raise ValueError("unexpected text after the first Newick tree at "
                         f"position {pos[0]}: {rest.strip()[:40]!r}")
    return tree


def _newick_label(name):
    """Quote a Newick label when it holds a character the format reserves."""
    if name and any(c in name for c in " ,:()[];'"):
        return "'" + name.replace("'", "''") + "'"
    return name


def tree_to_newick(node):
    """Convert TreeNode to a Newick format string.

    Labels that need quoting are quoted and the terminating ';' is emitted, so
    the result can be handed back to ``parse_newick`` or to an outside reader
    unchanged.  ``%.6g`` keeps a short branch such as 1e-5 from rounding to 0.
    """
    def _emit(n):
        label = _newick_label(n.name) if n.name else ''
        if not n.children:
            return f"{label}:{n.branch_length:.6g}"
        inner = ','.join(_emit(c) for c in n.children)
        return f"({inner}){label}:{n.branch_length:.6g}"

    return _emit(node) + ';'


def tree_to_ascii(node, prefix="", is_last=True):
    """Convert tree to ASCII art for display.

    A branch length is shown whenever one is set, including a negative one:
    neighbor-joining and similar methods do produce negative branches, and
    printing the taxon with no number at all leaves the reader to assume the
    source gave none.
    """
    lines = []
    connector = "└── " if is_last else "├── "
    name = node.name if node.name else f"({len(node.children)} children)"
    length = "" if node.branch_length == 0 else f" ({node.branch_length:.3f})"
    lines.append(f"{prefix}{connector}{name}{length}")

    new_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(node.children):
        lines.extend(tree_to_ascii(child, new_prefix, i == len(node.children) - 1))
    return lines


def _read_newick_file(filepath):
    """Read a Newick file into a result dict that names the file in failures.

    A missing file, an empty one and a malformed tree each become an error dict
    carrying the format, so read_file hands the caller one shape instead of
    letting OSError or the parser's ValueError surface as a bare exception.
    """
    try:
        with smart_open(filepath) as handle:
            text = handle.read()
    except OSError as exc:
        return {"format": "newick",
                "error": f"Could not read the Newick file {filepath}: {exc}"}
    if not text.strip():
        return {"format": "newick",
                "error": f"{filepath} is empty, so there is no Newick tree to read"}
    try:
        tree = parse_newick(text)
    except ValueError as exc:
        return {"format": "newick", "error": f"Not a Newick tree: {exc}"}
    return {"format": "newick", "tree": tree}


def parse_stockholm(filepath):
    """Parse Stockholm format alignment file.

    Returns:
        Dict with 'alignment' (dict of name->sequence) and 'metadata'.

    Only the first alignment block is returned: 'metadata' holds the '#=GF'
    global lines of that block, and '#=GS' per-sequence and '#=GC' per-column
    records are not represented in the result.  A file holding several blocks is
    therefore read as its first one, and the remaining lines are reported.
    """
    alignment = {}
    metadata = {}

    with smart_open(filepath) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('# STOCKHOLM'):
                continue
            if line.startswith('#=GF'):
                parts = line.split(None, 2)
                if len(parts) >= 3:
                    metadata[parts[1]] = parts[2]
            elif line.startswith('#=GS'):
                pass  # Per-sequence metadata
            elif line.startswith('#'):
                continue
            elif line.strip() == '//':
                break  # end of the first alignment block
            elif line.strip():
                parts = line.split(None, 1)
                if len(parts) == 2:
                    name, seq = parts
                    alignment[name] = alignment.get(name, '') + seq

        unread = sum(1 for line in f
                     if line.strip() and not line.startswith('#') and line.strip() != '//')
    if unread:
        logger.warning("%s: %d Stockholm line(s) after the first '//' terminator "
                       "belong to a second alignment block and were not read",
                       filepath, unread)
    return {'alignment': alignment, 'metadata': metadata}


BED_COLUMNS = ("chrom", "start", "end", "name", "score", "strand", "length")
GFF_COLUMNS = ("seqid", "source", "feature", "start", "end", "score", "strand", "length")
BIGWIG_COLUMNS = ("chrom", "start", "end", "mean", "max")


def bed_to_dataframe(records):
    """Convert BED records to pandas DataFrame.

    An empty record list still yields the columns: a caller that indexes
    ``df['chrom']`` after a file with no usable rows should see an empty column,
    not a KeyError that hides where the emptiness came from.
    """
    if not records:
        return pd.DataFrame(columns=list(BED_COLUMNS))
    data = [{
        'chrom': r.chrom, 'start': r.start, 'end': r.end,
        'name': r.name, 'score': r.score, 'strand': r.strand,
        'length': r.end - r.start
    } for r in records]
    return pd.DataFrame(data)


def gff_to_dataframe(records):
    """Convert GFF records to pandas DataFrame.

    Attribute keys are prefixed with ``info_``: spreading them bare let a record
    carrying ``start=999`` or ``strand=-`` in its ninth column overwrite the real
    coordinates and strand, which are the reason the row exists.  An empty record
    list still yields the coordinate columns.
    """
    if not records:
        return pd.DataFrame(columns=list(GFF_COLUMNS))
    data = []
    for r in records:
        row = {
            'seqid': r.seqid, 'source': r.source, 'feature': r.feature,
            'start': r.start, 'end': r.end, 'score': r.score,
            'strand': r.strand, 'length': r.end - r.start + 1,
        }
        for key, value in r.attributes.items():
            row[f"info_{key}"] = value
        data.append(row)
    return pd.DataFrame(data)


def format_bed_summary(records):
    lines = ["=== BED File Summary ===", f"Records: {len(records)}"]
    if records:
        chroms = set(r.chrom for r in records)
        lines.append(f"Chromosomes: {len(chroms)}")
        total_bp = sum(r.end - r.start for r in records)
        lines.append(f"Total bp: {total_bp:,}")
    return '\n'.join(lines)


def format_gff_summary(records):
    lines = ["=== GFF Summary ===", f"Records: {len(records)}"]
    if records:
        features = {}
        for r in records:
            features[r.feature] = features.get(r.feature, 0) + 1
        lines.append("Feature types:")
        for feat, count in sorted(features.items(), key=lambda x: -x[1]):
            lines.append(f"  {feat}: {count}")
    return '\n'.join(lines)


# ── BigWig Reader ────────────────────────────────────────────────────────────

def read_bigwig(path, chrom=None, start=None, end=None):
    """Read BigWig file as numpy array.

    A region is read only when chrom, start and end are all given.  Half a region
    used to fall through to the whole-file branch, so a caller that asked for one
    chromosome got every contig in the file back under a different key.

    Args:
        path: path to BigWig file.
        chrom: chromosome to read (None for all).
        start: start position (0-based).
        end: end position.

    Returns:
        dict with 'chroms' (contig -> values array) for a whole-file read, or
        'chrom', 'start', 'end' and 'values' for a region, or {"error": ...} when
        the file cannot be read or no decoder is installed.

    Raises:
        ValueError: when only some of chrom, start and end are given.
    """
    given = [name for name, value in (("chrom", chrom), ("start", start), ("end", end))
             if value is not None]
    if given and len(given) != 3:
        raise ValueError("a BigWig region needs chrom, start and end together; "
                         f"only {', '.join(given)} were given")
    try:
        import bigWigReader
    except ImportError:
        return _bigwig_without_reader(path)

    try:
        bw = bigWigReader.open(path)
        if given:
            values = bw.getEntries(chrom, start, end)
            return {"chrom": chrom, "start": start, "end": end,
                    "values": np.array([v[2] for v in values]) if values else np.array([])}
        result = {}
        for c in bw.getChroms():
            entries = bw.getEntries(c, 0, bw.getChromSize(c))
            result[c] = np.array([e[2] for e in entries]) if entries else np.array([])
        return {"chroms": result}
    except OSError as exc:
        logger.error("Could not read the BigWig file %s: %s", path, exc)
        return {"error": f"Could not read BigWig file {path}: {exc}"}


def _bigwig_without_reader(path):
    """Report why a BigWig could not be read instead of returning it as empty.

    The fallback this replaces unpacked the header and came back with an empty
    values array, so the CLI and the GUI both printed "No BigWig data available."
    for a populated file whenever the decoder package was simply not installed.
    The magic number is still checked, because a file that does not start with
    BigWig's 0x276C4BC7 (little-endian on disk) is not a BigWig at all.
    """
    try:
        with open(path, 'rb') as handle:
            magic = handle.read(4)
    except OSError as exc:
        logger.error("Could not open the BigWig file %s: %s", path, exc)
        return {"error": f"Could not open BigWig file {path}: {exc}"}
    if magic != BIGWIG_MAGIC:
        return {"error": f"{path} does not start with the BigWig magic number "
                         f"{BIGWIG_MAGIC.hex()} (found {magic.hex() or 'nothing'}), so it is "
                         "not a BigWig file"}
    return {"error": f"{path} is a BigWig file, but BioSuite cannot decode it without "
                     "the bigWigReader package. Run: pip install bigWigReader"}


def bigwig_summary(path, bin_size=1000):
    """Summarize BigWig data into bins.

    Args:
        path: path to BigWig file.
        bin_size: bin size in bp: a whole number of at least 1.

    Returns:
        DataFrame with chrom, start, end, mean, max columns.  The frame is empty
        when the file holds no coverage or could not be read, which the log says
        in words; the reason is never left for the reader to guess at.

    Raises:
        ValueError: when bin_size is not a positive whole number.  A zero bin size
        divided by zero, and a negative one produced a single row whose end was
        negative and whose mean covered an arbitrary slice of the contig.
    """
    if isinstance(bin_size, bool) or not isinstance(bin_size, int) or bin_size < 1:
        raise ValueError(f"bin_size must be a whole number of at least 1 bp, got {bin_size!r}")

    data = read_bigwig(path)
    if "error" in data:
        logger.error("No BigWig summary of %s: %s", path, data["error"])
        return pd.DataFrame(columns=list(BIGWIG_COLUMNS))
    if "chroms" not in data:
        logger.error("No BigWig summary of %s: the reader returned no per-contig "
                     "coverage", path)
        return pd.DataFrame(columns=list(BIGWIG_COLUMNS))

    rows = []
    for chrom, values in data["chroms"].items():
        if len(values) == 0:
            continue
        n_bins = -(-len(values) // bin_size)  # ceil: the tail bin is a bin too
        for i in range(n_bins):
            start = i * bin_size
            end = min((i + 1) * bin_size, len(values))
            chunk = values[start:end]
            rows.append({
                "chrom": chrom, "start": start, "end": end,
                "mean": float(np.mean(chunk)),
                "max": float(np.max(chunk)),
            })
    if not rows:
        return pd.DataFrame(columns=list(BIGWIG_COLUMNS))
    return pd.DataFrame(rows)


def format_bigwig_summary(df):
    """Format BigWig summary as text, listing at most BIGWIG_REPORT_ROWS bins.

    The bins left out are counted on a last line: a header that announces 25 bins
    over a list of 20 has to say that five of them are not on the screen.
    """
    if df.empty:
        return "No BigWig data available."
    lines = [f"BigWig Summary: {len(df)} bins"]
    shown = df.head(BIGWIG_REPORT_ROWS)
    for _, row in shown.iterrows():
        lines.append(f"  {row['chrom']}:{row['start']}-{row['end']} "
                     f"mean={row['mean']:.2f} max={row['max']:.2f}")
    omitted = len(df) - len(shown)
    if omitted:
        lines.append(f"... and {omitted} more bin(s).")
    return "\n".join(lines)


# ── CRAM Reader ──────────────────────────────────────────────────────────────

def _read_alignment(path, reference, region, fmt):
    """Shared pysam-backed reader for CRAM ('rc'+reference) and BAM ('rb').

    The old read_file() routed .bam files through the CRAM-only mode string
    'rc', which made every BAM read fail whenever pysam was installed.

    The 'alignments' list is filled only for a region read: without one pysam is
    not asked for any read, so the counts are complete while the list is empty.
    The 'region' key records which of the two the caller asked for.
    """
    try:
        import pysam
        has_pysam = True
    except ImportError:
        has_pysam = False

    if not has_pysam:
        return {"error": f"pysam required for {fmt.upper()} reading. pip install pysam",
                "format": fmt, "note": f"Install pysam for full {fmt.upper()} support"}

    try:
        kwargs = {}
        if fmt == 'cram':
            kwargs["format"] = "cram"
            if reference:
                kwargs["reference_filename"] = reference

        with pysam.AlignmentFile(path, "rc" if fmt == 'cram' else "rb", **kwargs) as cram:
            header = dict(cram.header)
            n_mapped = cram.mapped
            n_unmapped = cram.unmapped

            alignments = []
            if region:
                for read in cram.fetch(region=region):
                    alignments.append({
                        "query_name": read.query_name,
                        "reference_start": read.reference_start,
                        "reference_end": read.reference_end,
                        "mapping_quality": read.mapping_quality,
                        "query_alignment_length": read.query_alignment_length,
                        "is_unmapped": read.is_unmapped,
                        "is_reverse": read.is_reverse,
                    })

            return {
                "format": fmt,
                "header": header,
                "mapped_reads": n_mapped,
                "unmapped_reads": n_unmapped,
                "total_reads": n_mapped + n_unmapped,
                "region": region,
                "alignments": alignments,
                "reference": reference,
            }
    except Exception as e:
        logger.error("Could not read the %s file %s: %s", fmt, path, e)
        return {"error": str(e), "format": fmt}


def read_cram(cram_path, reference=None, region=None):
    """Read CRAM file and return alignment records.

    CRAM is a compressed alignment format that requires a reference genome.
    Decoding needs pysam; without it this returns an error dict that says so, and
    no alignment data at all.

    Args:
        cram_path: path to CRAM file.
        reference: path to reference FASTA (required for decoding).
        region: genomic region (e.g., 'chr1:1000-2000').  Reads are only listed
            for a region; without one the counts are still reported.

    Returns:
        dict with headers, alignments count, and optional alignment data.
    """
    return _read_alignment(cram_path, reference, region, "cram")


def read_bam(bam_path, region=None):
    """Read BAM file and return alignment records (pysam-backed).

    ``region`` is the only option this reader takes; reads are listed only when
    one is given.
    """
    return _read_alignment(bam_path, None, region, "bam")


# ── BCF Reader ───────────────────────────────────────────────────────────────

def read_bcf(path):
    """Read a BCF file (the binary VCF) into a list of variant records.

    The magic check comes before any decoder is touched, so a file that is not
    BCF is refused as such instead of surfacing a decoder's complaint.  Decoding
    needs pysam, and its absence names the install command.
    """
    try:
        with open(path, 'rb') as handle:
            head = handle.read(len(_BCF_MAGIC))
    except OSError as exc:
        logger.error("Could not open the BCF file %s: %s", path, exc)
        return {"error": f"Could not open BCF file {path}: {exc}"}
    if head != _BCF_MAGIC and head[:len(_BGZF_MAGIC)] != _BGZF_MAGIC:
        return {"error": f"{path} does not start with the BCF magic {_BCF_MAGIC.hex()} "
                         f"or a BGZF gzip header (found {head.hex() or 'nothing'}), so "
                         "it is not a BCF file"}
    try:
        import pysam
    except ImportError:
        return {"error": f"{path} is a BCF file, but BioSuite cannot decode it without "
                         "pysam. Run: pip install pysam (or convert it with bcftools)"}
    records = []
    try:
        with pysam.VariantFile(path) as handle:
            for rec in handle:
                records.append({
                    "chrom": rec.contig,
                    "pos": rec.pos,
                    "id": rec.id or "",
                    "ref": rec.ref,
                    "alts": list(rec.alts or []),
                })
    except (OSError, ValueError) as exc:
        logger.error("Could not decode the BCF file %s: %s", path, exc)
        return {"error": f"Could not decode BCF file {path}: {exc}"}
    return {"records": records}


# ── GTF Parser ───────────────────────────────────────────────────────────────

def parse_gtf(filepath):
    """Parse GTF (Gene Transfer Format) file.

    GTF is a stricter dialect of GFF: its records conventionally carry gene_id and
    transcript_id in the ninth column.  Those two are not enforced here — a record
    without them is still a feature with usable coordinates — but both attribute
    styles are read, because files in the wild write ``gene_id "X";`` and
    ``gene_id=X;`` side by side.

    Returns:
        list of GFFRecord objects with GTF-specific attribute parsing.
    """
    records = []
    skipped = 0
    with smart_open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 9:
                skipped += 1
                continue
            try:
                records.append(_feature_row(parts))
            except ValueError:
                skipped += 1
                continue
    _report_skipped(filepath, skipped, "GTF")
    return records


# ── SAF (Simple Annotation Format) ──────────────────────────────────────────

def parse_saf(filepath):
    """Parse SAF (Simple Annotation Format) file.

    SAF is the featureCounts summary: GeneID, Chr, Start, End and an optional
    Strand column.  A record written without the strand is reported as '.', which
    is what the file actually says; assuming '+' would give every unstranded
    feature a direction it never had.

    Returns:
        list of dicts with gene_id, chr, start, end, strand.
    """
    records = []
    skipped = 0
    with smart_open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                skipped += 1
                continue
            # SAF spec includes a header row ("GeneID Chr Start End Strand");
            # featureCounts standard output always carries it and the naive
            # int() used to blow up on "Start".
            if parts[0] == 'GeneID':
                continue
            try:
                start, end = int(parts[2]), int(parts[3])
            except ValueError:
                skipped += 1
                continue
            records.append({
                "gene_id": parts[0],
                "chr": parts[1],
                "start": start,
                "end": end,
                "strand": parts[4] if len(parts) > 4 else ".",
            })
    _report_skipped(filepath, skipped, "SAF")
    return records


# ── BAM Index Reader ─────────────────────────────────────────────────────────

def _index_head(path, count=4):
    """Return the first *count* bytes of an index file, or None when it cannot be read."""
    try:
        with open(path, 'rb') as handle:
            return handle.read(count)
    except OSError as exc:
        logger.error("Could not read the index file %s: %s", path, exc)
        return None


def read_bam_index(bai_path):
    """Read BAM index (.bai) file and return basic metadata.

    ``valid`` is the file's own answer: it is true only when the file starts with
    htslib's BAM index magic.  Which packages happen to be installed says nothing
    about the bytes in front of us, so the pysam branch checks them too instead of
    declaring the index sound because an import succeeded.

    Args:
        bai_path: path to .bai index file.

    Returns:
        dict with file info and whether the file is a BAM index at all.
    """
    import os
    if not os.path.exists(bai_path):
        return {"error": f"Index file not found: {bai_path}"}
    head = _index_head(bai_path)
    if head is None:
        return {"format": "bai", "path": bai_path,
                "error": f"Could not read the BAM index {bai_path}"}

    result = {"format": "bai", "path": bai_path,
              "file_size": os.path.getsize(bai_path), "valid": head == BAM_INDEX_MAGIC}
    if not result["valid"]:
        result["note"] = (f"Not a BAM index: expected the magic {BAM_INDEX_MAGIC!r} at the "
                          f"head of the file, found {head!r}")
        return result
    try:
        import pysam  # noqa: F401  (reported so the reader knows what detail is available)
    except ImportError:
        result["note"] = "BAM index recognised. Install pysam to query it in detail."
    else:
        result["note"] = "BAM index recognised. Use pysam for detailed queries."
    return result


# ── VCF Index Reader ─────────────────────────────────────────────────────────

def _tabix_data_file(tbi_path):
    """Return the data file a tabix index sits beside, or None when it cannot be named.

    Cutting the suffix off the end is the whole of it: replacing every '.tbi' in
    the string also rewrote directory names that happened to contain it, and a path
    whose suffix was not the lowercase '.tbi' was quietly handed the index itself.
    """
    if tbi_path.lower().endswith('.tbi'):
        return tbi_path[:-len('.tbi')]
    return None


def read_vcf_index(tbi_path):
    """Read VCF tabix index (.tbi) file.

    The index is validated by its own magic bytes; the contig list comes from the
    data file beside it, which has to be present and bgzip-compressed.  When either
    is missing that is said plainly rather than reported as a working index.

    Args:
        tbi_path: path to .tbi index file.

    Returns:
        dict with file info and basic index structure.
    """
    import os
    if not os.path.exists(tbi_path):
        return {"error": f"Index file not found: {tbi_path}"}
    head = _index_head(tbi_path)
    if head is None:
        return {"format": "tbi", "path": tbi_path,
                "error": f"Could not read the tabix index {tbi_path}"}

    result = {"format": "tbi", "path": tbi_path,
              "file_size": os.path.getsize(tbi_path), "valid": head == TABIX_INDEX_MAGIC}
    if not result["valid"]:
        result["note"] = (f"Not a tabix index: expected the magic {TABIX_INDEX_MAGIC!r} at "
                          f"the head of the file, found {head!r}")
        return result

    data_file = _tabix_data_file(tbi_path)
    if data_file is None:
        result["note"] = ("Index recognised; its data file cannot be named because the "
                          "index path does not end in '.tbi'.")
        return result
    if not os.path.exists(data_file):
        result["note"] = (f"Index recognised; the data file {data_file} is not there, so no "
                          "contigs could be listed.")
        return result
    try:
        import pysam
    except ImportError:
        result["note"] = "Index recognised. Install pysam to list its contigs."
        return result
    try:
        tbx = pysam.TabixFile(data_file)
    except (OSError, ValueError) as exc:
        logger.error("Could not query %s with pysam: %s", data_file, exc)
        result["note"] = (f"Index recognised but pysam could not query {data_file}: {exc}. "
                          "The data file must be bgzip-compressed for tabix queries.")
        return result
    try:
        chromosomes = list(tbx.contigs or [])
    finally:
        tbx.close()
    result["chromosomes"] = chromosomes
    result["n_chromosomes"] = len(chromosomes)
    return result


# ── Format Detection ─────────────────────────────────────────────────────────

FORMAT_EXTENSIONS = {
    '.fasta': 'fasta', '.fa': 'fasta', '.fna': 'fasta', '.fas': 'fasta',
    '.fastq': 'fastq', '.fq': 'fastq',
    '.gb': 'genbank', '.genbank': 'genbank', '.gbk': 'genbank',
    '.bam': 'bam', '.sam': 'sam', '.cram': 'cram',
    '.vcf': 'vcf', '.bcf': 'bcf',
    '.bed': 'bed',
    '.gff': 'gff', '.gff3': 'gff', '.gtf': 'gtf',
    '.newick': 'newick', '.nwk': 'newick', '.tree': 'newick', '.nhx': 'newick',
    '.stockholm': 'stockholm', '.sto': 'stockholm',
    '.bigwig': 'bigwig', '.bw': 'bigwig',
    '.bai': 'bai',
    '.tbi': 'tbi',
    '.saf': 'saf',
}

# Codecs every text parser in this module can already open through smart_open.
COMPRESSED_SUFFIXES = ('.gz', '.bz2')


def detect_file_format(filepath):
    """Detect bioinformatics file format from extension.

    A trailing .gz or .bz2 is looked through, because the parsers here open
    compressed files transparently: an annotation saved as the usual 'x.bed.gz'
    used to come back as 'unknown' even though parse_bed reads it without trouble.

    Detection and reading are wired together: read_file has a reader for every
    format named here.  Decoding a BAM/CRAM/BCF still needs pysam and a BigWig
    needs the bigWigReader package, and those readers say so when a package is
    missing.

    Args:
        filepath: path to file.

    Returns:
        format string (e.g., 'fasta', 'bam', 'vcf') or 'unknown'.
    """
    import os
    low = str(filepath).lower()
    for suffix in COMPRESSED_SUFFIXES:
        if low.endswith(suffix):
            low = low[:-len(suffix)]
            break
    if low.endswith('.cram'):
        # Also covers a file whose whole name is the extension: splitext('.cram')
        # reports no extension at all.
        return 'cram'
    return FORMAT_EXTENSIONS.get(os.path.splitext(low)[1], 'unknown')


def _read_or_error(fmt, filepath, payload, key):
    """Wrap a reader whose failure is a returned None instead of a raised error.

    read_fasta, read_fastq, read_genbank and read_vcf all answer None for a file
    that is missing or unparseable.  Handing that straight through let a mistyped
    path report as "FASTA: 0 sequences", so an absent file and an empty one looked
    the same to whoever read the summary.
    """
    if payload is None:
        return {"format": fmt,
                "error": f"{filepath} was not read as {fmt.upper()}: the file is "
                         "missing or could not be parsed"}
    return {"format": fmt, key: payload}


def read_file(filepath, **kwargs):
    """Universal file reader that auto-detects format from the file's extension.

    Args:
        filepath: path to file.
        **kwargs: reader options.  'bam' takes 'region', 'cram' takes 'region'
            and 'reference', and 'bigwig' takes 'chrom', 'start' and 'end';
            every other parser here reads the whole file and is handed the path
            alone, so an option for one of them is refused by name instead of
            being silently ignored.

    Returns:
        dict with 'format' key and parsed data, or error dict.  'format' is the
        detected format even when no reader is available for it.
    """
    from .ngs import read_sam, read_vcf
    from .sequence import read_fasta, read_fastq, read_genbank

    fmt = detect_file_format(filepath)
    allowed = READER_OPTIONS.get(fmt, ())
    unexpected = sorted(set(kwargs) - set(allowed))
    if unexpected:
        accepts = ", ".join(allowed) if allowed else "no options"
        return {"format": fmt,
                "error": f"The {fmt} reader does not take {', '.join(unexpected)} "
                         f"(it accepts: {accepts})"}

    readers = {
        'fasta': lambda: _read_or_error('fasta', filepath, read_fasta(filepath), 'records'),
        'fastq': lambda: _read_or_error('fastq', filepath, read_fastq(filepath), 'records'),
        'genbank': lambda: _read_or_error('genbank', filepath, read_genbank(filepath), 'records'),
        'bed': lambda: {"format": "bed", "records": parse_bed(filepath)},
        'gff': lambda: {"format": "gff", "records": parse_gff(filepath)},
        'gtf': lambda: {"format": "gtf", "records": parse_gtf(filepath)},
        'saf': lambda: {"format": "saf", "records": parse_saf(filepath)},
        'stockholm': lambda: {"format": "stockholm", "data": parse_stockholm(filepath)},
        'newick': lambda: _read_newick_file(filepath),
        'vcf': lambda: _read_or_error('vcf', filepath, read_vcf(filepath), 'data'),
        'sam': lambda: _read_or_error('sam', filepath, read_sam(filepath), 'records'),
        'bcf': lambda: {"format": "bcf", "data": read_bcf(filepath)},
        'bam': lambda: {"format": "bam", "data": read_bam(filepath, **kwargs)},
        'cram': lambda: {"format": "cram", "data": read_cram(filepath, **kwargs)},
        'bai': lambda: {"format": "bai", "data": read_bam_index(filepath)},
        'tbi': lambda: {"format": "tbi", "data": read_vcf_index(filepath)},
        'bigwig': lambda: {"format": "bigwig", "data": read_bigwig(filepath, **kwargs)},
    }

    if fmt in readers:
        try:
            return readers[fmt]()
        except Exception as e:
            logger.error("The %s reader failed on %s: %s", fmt, filepath, e)
            return {"format": fmt, "error": str(e)}

    return {"format": fmt,
            "error": f"{filepath} was detected as {fmt} but this build has no reader "
                     f"for {fmt} files"}


# ── Format Summary ───────────────────────────────────────────────────────────

def format_file_summary(result):
    """Format file read result as human-readable text.

    A reader that failed stays a failure even when its error sits inside 'data'
    rather than at the top of the result.  BAM, CRAM and both index readers answer
    with a nested error dict, and reporting one of those as "File loaded: bam
    format" told the user the opposite of what had happened.
    """
    if "error" in result:
        return f"Error reading {result.get('format', 'unknown')} file: {result['error']}"

    fmt = result.get("format", "unknown")
    data = result.get("data")
    if isinstance(data, dict) and "error" in data:
        return f"Error reading {fmt} file: {data['error']}"

    if fmt == "bed" and "records" in result:
        return format_bed_summary(result["records"])
    elif fmt in ("gff", "gtf") and "records" in result:
        return format_gff_summary(result["records"])
    elif fmt == "fasta" and "records" in result:
        records = result["records"]
        if records:
            total_len = sum(len(s) for _, s in records)
            return f"FASTA: {len(records)} sequences, {total_len:,} bp total"
        return "FASTA: 0 sequences"
    elif fmt == "fastq" and "records" in result:
        records = result["records"]
        if records:
            total_len = sum(len(s) for _, s, _ in records)
            return f"FASTQ: {len(records)} reads, {total_len:,} bp total"
        return "FASTQ: 0 reads"
    elif fmt == "genbank" and "records" in result:
        return f"GenBank: {len(result['records'])} records"
    elif fmt == "saf" and "records" in result:
        return f"SAF: {len(result['records'])} annotated features"
    elif fmt == "stockholm" and isinstance(data, dict):
        return f"Stockholm: {len(data.get('alignment', {}))} sequences"
    elif fmt in ("bai", "tbi") and isinstance(data, dict):
        if data.get("valid"):
            contigs = (f", {data['n_chromosomes']} contigs"
                       if "n_chromosomes" in data else "")
            return f"{fmt.upper()} index: valid ({data.get('file_size', 0):,} bytes{contigs})"
        # The reader's own sentence already names which index it rejected and why,
        # so repeating "BAI index:" in front of it adds nothing a user needs.
        return data.get("note") or f"{fmt.upper()} index: not a valid {fmt} file"
    elif fmt in ("bam", "cram") and isinstance(data, dict) and "total_reads" in data:
        return (f"{fmt.upper()}: {data['total_reads']:,} reads "
                f"({data.get('mapped_reads', 0):,} mapped)")
    elif fmt == "vcf" and data is not None:
        return f"VCF: {len(data)} variants"

    return f"File loaded: {fmt} format"
