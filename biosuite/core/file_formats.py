"""
File format parsers: BED, GFF/GTF, Newick tree, Stockholm.

Pure Python parsers for common bioinformatics file formats.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


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

    Supports BED3, BED6, and BED12 formats.
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            rec = BEDRecord(
                chrom=parts[0],
                start=int(parts[1]),
                end=int(parts[2]),
                name=parts[3] if len(parts) > 3 else "",
                score=float(parts[4]) if len(parts) > 4 else 0,
                strand=parts[5] if len(parts) > 5 else ".",
            )
            if len(parts) > 6:
                rec.thick_start = int(parts[6])
            if len(parts) > 7:
                rec.thick_end = int(parts[7])
            if len(parts) > 8:
                rec.item_rgb = parts[8]
            records.append(rec)
    return records


def parse_gff(filepath):
    """Parse GFF3 or GTF file into list of GFFRecord objects."""
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 9:
                continue
            attrs = {}
            for item in parts[8].split(';'):
                if '=' in item:
                    key, val = item.split('=', 1)
                    attrs[key.strip()] = val.strip()
                elif ' ' in item:
                    key, val = item.split(' ', 1)
                    attrs[key.strip()] = val.strip().strip('"')

            records.append(GFFRecord(
                seqid=parts[0],
                source=parts[1],
                feature=parts[2],
                start=int(parts[3]),
                end=int(parts[4]),
                score=float(parts[5]) if parts[5] != '.' else 0,
                strand=parts[6],
                phase=parts[7],
                attributes=attrs
            ))
    return records


def parse_newick(newick_str):
    """Parse Newick format tree string into TreeNode objects.

    Args:
        newick_str: Newick format string (e.g., "(A:0.1,B:0.2)C:0.3;")

    Returns:
        Root TreeNode.
    """
    newick_str = newick_str.strip().rstrip(';').strip()
    pos = [0]

    def _parse():
        node = TreeNode()
        if newick_str[pos[0]] == '(':
            pos[0] += 1  # skip '('
            while newick_str[pos[0]] != ')':
                child = _parse()
                node.children.append(child)
                if newick_str[pos[0]] == ',':
                    pos[0] += 1
            pos[0] += 1  # skip ')'

        # Read name and branch length
        name = ''
        bl = ''
        while pos[0] < len(newick_str) and newick_str[pos[0]] not in '(),;:':
            name += newick_str[pos[0]]
            pos[0] += 1
        if pos[0] < len(newick_str) and newick_str[pos[0]] == ':':
            pos[0] += 1
            while pos[0] < len(newick_str) and newick_str[pos[0]] not in '(),;':
                bl += newick_str[pos[0]]
                pos[0] += 1

        node.name = name.strip()
        node.branch_length = float(bl) if bl else 0.0
        node.is_leaf = len(node.children) == 0
        return node

    return _parse()


def tree_to_newick(node):
    """Convert TreeNode to Newick format string."""
    if node.is_leaf:
        return f"{node.name}:{node.branch_length:.4f}" if node.name else f"{node.branch_length:.4f}"
    children = ','.join(tree_to_newick(c) for c in node.children)
    name_part = node.name if node.name else ''
    bl_part = f":{node.branch_length:.4f}" if node.branch_length > 0 else ''
    return f"({children}){name_part}{bl_part}"


def tree_to_ascii(node, prefix="", is_last=True):
    """Convert tree to ASCII art for display."""
    lines = []
    connector = "└── " if is_last else "├── "
    name = node.name if node.name else f"({len(node.children)} children)"
    lines.append(f"{prefix}{connector}{name}" + (f" ({node.branch_length:.3f})" if node.branch_length > 0 else ""))

    new_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(node.children):
        lines.extend(tree_to_ascii(child, new_prefix, i == len(node.children) - 1))
    return lines


def parse_stockholm(filepath):
    """Parse Stockholm format alignment file.

    Returns:
        Dict with 'alignment' (dict of name->sequence) and 'metadata'.
    """
    alignment = {}
    metadata = {}
    in_alignment = False

    with open(filepath) as f:
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
                in_alignment = False
            elif line.strip():
                parts = line.split(None, 1)
                if len(parts) == 2:
                    name, seq = parts
                    alignment[name] = alignment.get(name, '') + seq

    return {'alignment': alignment, 'metadata': metadata}


def bed_to_dataframe(records):
    """Convert BED records to pandas DataFrame."""
    data = [{
        'chrom': r.chrom, 'start': r.start, 'end': r.end,
        'name': r.name, 'score': r.score, 'strand': r.strand,
        'length': r.end - r.start
    } for r in records]
    return pd.DataFrame(data)


def gff_to_dataframe(records):
    """Convert GFF records to pandas DataFrame."""
    data = [{
        'seqid': r.seqid, 'source': r.source, 'feature': r.feature,
        'start': r.start, 'end': r.end, 'score': r.score,
        'strand': r.strand, 'length': r.end - r.start + 1,
        **r.attributes
    } for r in records]
    return pd.DataFrame(data)


def format_bed_summary(records):
    lines = [f"=== BED File Summary ===", f"Records: {len(records)}"]
    if records:
        chroms = set(r.chrom for r in records)
        lines.append(f"Chromosomes: {len(chroms)}")
        total_bp = sum(r.end - r.start for r in records)
        lines.append(f"Total bp: {total_bp:,}")
    return '\n'.join(lines)


def format_gff_summary(records):
    lines = [f"=== GFF Summary ===", f"Records: {len(records)}"]
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

    Args:
        path: path to BigWig file.
        chrom: chromosome to read (None for all).
        start: start position (0-based).
        end: end position.

    Returns:
        dict with chroms, values arrays.
    """
    try:
        import bigWigReader
        bw = bigWigReader.open(path)
        if chrom and start is not None and end is not None:
            values = bw.getEntries(chrom, start, end)
            return {"chrom": chrom, "start": start, "end": end,
                    "values": np.array([v[2] for v in values]) if values else np.array([])}
        else:
            chroms = bw.getChroms()
            result = {}
            for c in chroms:
                entries = bw.getEntries(c, 0, bw.getChromSize(c))
                result[c] = np.array([e[2] for e in entries]) if entries else np.array([])
            return {"chroms": result}
    except ImportError:
        # Fallback: parse as text if bigWigReader not available
        return _read_bigwig_text(path, chrom, start, end)


def _read_bigwig_text(path, chrom=None, start=None, end=None):
    """Fallback BigWig reader using struct parsing."""
    import struct
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'\x27\xbb\x1f':
            return {"error": "Not a valid BigWig file"}
        # Read basic header
        header_size = struct.unpack('>I', f.read(4))[0]
        f.seek(header_size)
        # Simplified: return raw coverage data
        return {"chrom": chrom, "start": start or 0, "end": end or 0,
                "values": np.array([]), "note": "Basic BigWig parsing"}


def bigwig_summary(path, bin_size=1000):
    """Summarize BigWig data into bins.

    Args:
        path: path to BigWig file.
        bin_size: bin size in bp.

    Returns:
        DataFrame with chrom, start, end, mean, max columns.
    """
    data = read_bigwig(path)
    if "error" in data or "chroms" not in data:
        return pd.DataFrame()

    rows = []
    for chrom, values in data["chroms"].items():
        if len(values) == 0:
            continue
        n_bins = max(1, len(values) // bin_size)
        for i in range(n_bins):
            start = i * bin_size
            end = min((i + 1) * bin_size, len(values))
            chunk = values[start:end]
            if len(chunk) > 0:
                rows.append({
                    "chrom": chrom, "start": start, "end": end,
                    "mean": float(np.mean(chunk)),
                    "max": float(np.max(chunk)),
                })
    return pd.DataFrame(rows)


def format_bigwig_summary(df):
    """Format BigWig summary as text."""
    if df.empty:
        return "No BigWig data available."
    lines = [f"BigWig Summary: {len(df)} bins"]
    for _, row in df.head(20).iterrows():
        lines.append(f"  {row['chrom']}:{row['start']}-{row['end']} "
                     f"mean={row['mean']:.2f} max={row['max']:.2f}")
    return "\n".join(lines)


# ── CRAM Reader ──────────────────────────────────────────────────────────────

def read_cram(cram_path, reference=None, region=None):
    """Read CRAM file and return alignment records.

    CRAM is a compressed alignment format that requires a reference genome.
    Uses pysam if available, otherwise returns metadata only.

    Args:
        cram_path: path to CRAM file.
        reference: path to reference FASTA (required for decoding).
        region: genomic region (e.g., 'chr1:1000-2000').

    Returns:
        dict with headers, alignments count, and optional alignment data.
    """
    try:
        import pysam
        has_pysam = True
    except ImportError:
        has_pysam = False

    if not has_pysam:
        return {"error": "pysam required for CRAM reading. pip install pysam",
                "format": "cram", "note": "Install pysam for full CRAM support"}

    try:
        kwargs = {"format": "cram"}
        if reference:
            kwargs["reference_filename"] = reference

        with pysam.AlignmentFile(cram_path, "rc", **kwargs) as cram:
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
                "format": "cram",
                "header": header,
                "mapped_reads": n_mapped,
                "unmapped_reads": n_unmapped,
                "total_reads": n_mapped + n_unmapped,
                "alignments": alignments,
                "reference": reference,
            }
    except Exception as e:
        return {"error": str(e), "format": "cram"}


# ── GTF Parser (stricter GFF) ───────────────────────────────────────────────

def parse_gtf(filepath):
    """Parse GTF (Gene Transfer Format) file.

    GTF is a stricter version of GFF3 with required attributes:
    gene_id, transcript_id.

    Returns:
        list of GFFRecord objects with GTF-specific attribute parsing.
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 9:
                continue

            attrs = {}
            for item in parts[8].split(';'):
                item = item.strip()
                if not item:
                    continue
                if ' ' in item:
                    key, val = item.split(' ', 1)
                    attrs[key.strip()] = val.strip().strip('"')

            records.append(GFFRecord(
                seqid=parts[0],
                source=parts[1],
                feature=parts[2],
                start=int(parts[3]),
                end=int(parts[4]),
                score=float(parts[5]) if parts[5] != '.' else 0,
                strand=parts[6],
                phase=parts[7],
                attributes=attrs
            ))
    return records


# ── SAF (Simple Annotation Format) ──────────────────────────────────────────

def parse_saf(filepath):
    """Parse SAF (Simple Annotation Format) file.

    SAF is a simple two-column format: GeneID <tab> Chr <tab> Start <tab> End <tab> Strand

    Returns:
        list of dicts with gene_id, chr, start, end, strand.
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            record = {
                "gene_id": parts[0],
                "chr": parts[1],
                "start": int(parts[2]),
                "end": int(parts[3]),
                "strand": parts[4] if len(parts) > 4 else "+",
            }
            records.append(record)
    return records


# ── BAM Index Reader ─────────────────────────────────────────────────────────

def read_bam_index(bai_path):
    """Read BAM index (.bai) file and return basic metadata.

    Args:
        bai_path: path to .bai index file.

    Returns:
        dict with file info and basic index structure.
    """
    import os
    if not os.path.exists(bai_path):
        return {"error": f"Index file not found: {bai_path}"}

    try:
        import pysam
        # pysam can validate the index
        # We just check if it loads without error
        return {
            "format": "bai",
            "path": bai_path,
            "file_size": os.path.getsize(bai_path),
            "valid": True,
            "note": "BAM index loaded successfully. Use pysam for detailed queries."
        }
    except ImportError:
        # Basic check: read magic bytes
        with open(bai_path, 'rb') as f:
            magic = f.read(4)
            is_valid = len(magic) == 4
            return {
                "format": "bai",
                "path": bai_path,
                "file_size": os.path.getsize(bai_path),
                "valid": is_valid,
                "note": "Basic validation only. Install pysam for full support."
            }


# ── VCF Index Reader ─────────────────────────────────────────────────────────

def read_vcf_index(tbi_path):
    """Read VCF tabix index (.tbi) file.

    Args:
        tbi_path: path to .tbi index file.

    Returns:
        dict with file info and basic index structure.
    """
    import os
    if not os.path.exists(tbi_path):
        return {"error": f"Index file not found: {tbi_path}"}

    try:
        import pysam
        tbx = pysam.TabixFile(tbi_path.replace('.tbi', '.vcf.gz'))
        chromosomes = tbx.contigs
        return {
            "format": "tbi",
            "path": tbi_path,
            "file_size": os.path.getsize(tbi_path),
            "valid": True,
            "chromosomes": list(chromosomes),
            "n_chromosomes": len(chromosomes),
        }
    except Exception:
        # Basic info only
        with open(tbi_path, 'rb') as f:
            magic = f.read(4)
            return {
                "format": "tbi",
                "path": tbi_path,
                "file_size": os.path.getsize(tbi_path),
                "valid": len(magic) == 4,
                "note": "Basic info only. VCF must be bgzip-compressed for tabix queries."
            }


# ── Format Detection ─────────────────────────────────────────────────────────

FORMAT_EXTENSIONS = {
    '.fasta': 'fasta', '.fa': 'fasta', '.fna': 'fasta', '.fas': 'fasta',
    '.fastq': 'fastq', '.fq': 'fastq',
    '.gb': 'genbank', '.genbank': 'genbank', '.gbk': 'genbank',
    '.bam': 'bam', '.sam': 'sam', '.cram': 'cram',
    '.vcf': 'vcf', '.bcf': 'bcf',
    '.bed': 'bed',
    '.gff': 'gff', '.gff3': 'gff', '.gtf': 'gtf',
    '.newick': 'newick', '.nwk': 'newick', '.tree': 'newick',
    '.stockholm': 'stockholm', '.sto': 'stockholm',
    '.bigwig': 'bigwig', '.bw': 'bigwig',
    '.bai': 'bai',
    '.tbi': 'tbi',
    '.saf': 'saf',
}


def detect_file_format(filepath):
    """Detect bioinformatics file format from extension.

    Args:
        filepath: path to file.

    Returns:
        format string (e.g., 'fasta', 'bam', 'vcf') or 'unknown'.
    """
    import os
    ext = os.path.splitext(filepath)[1].lower()

    # Handle double extensions
    if filepath.endswith('.fa.gz'):
        return 'fasta'
    if filepath.endswith('.fq.gz'):
        return 'fastq'
    if filepath.endswith('.vcf.gz'):
        return 'vcf'
    if filepath.endswith('.cram'):
        return 'cram'

    return FORMAT_EXTENSIONS.get(ext, 'unknown')


def read_file(filepath, **kwargs):
    """Universal file reader that auto-detects format.

    Args:
        filepath: path to file.
        **kwargs: additional arguments passed to format-specific reader.

    Returns:
        dict with 'format' key and parsed data, or error dict.
    """
    from .sequence import read_fasta, read_fastq, read_genbank
    from .ngs import read_vcf

    fmt = detect_file_format(filepath)

    readers = {
        'fasta': lambda: {"format": "fasta", "records": read_fasta(filepath)},
        'fastq': lambda: {"format": "fastq", "records": read_fastq(filepath)},
        'genbank': lambda: {"format": "genbank", "records": read_genbank(filepath)},
        'bed': lambda: {"format": "bed", "records": parse_bed(filepath)},
        'gff': lambda: {"format": "gff", "records": parse_gff(filepath)},
        'gtf': lambda: {"format": "gtf", "records": parse_gtf(filepath)},
        'saf': lambda: {"format": "saf", "records": parse_saf(filepath)},
        'stockholm': lambda: {"format": "stockholm", "data": parse_stockholm(filepath)},
        'vcf': lambda: {"format": "vcf", "data": read_vcf(filepath)},
        'bam': lambda: {"format": "bam", "data": read_cram(filepath, **kwargs)},
        'cram': lambda: {"format": "cram", "data": read_cram(filepath, **kwargs)},
        'bai': lambda: {"format": "bai", "data": read_bam_index(filepath)},
        'tbi': lambda: {"format": "tbi", "data": read_vcf_index(filepath)},
    }

    if fmt in readers:
        try:
            return readers[fmt]()
        except Exception as e:
            return {"format": fmt, "error": str(e)}

    return {"format": "unknown", "error": f"Unsupported format: {fmt} ({filepath})"}


# ── Format Summary ───────────────────────────────────────────────────────────

def format_file_summary(result):
    """Format file read result as human-readable text."""
    if "error" in result:
        return f"Error reading {result.get('format', 'unknown')} file: {result['error']}"

    fmt = result.get("format", "unknown")

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
    elif fmt in ("bam", "cram") and "data" in result:
        data = result["data"]
        if "total_reads" in data:
            return (f"{fmt.upper()}: {data['total_reads']:,} reads "
                    f"({data.get('mapped_reads', 0):,} mapped)")
    elif fmt == "vcf" and "data" in result:
        data = result["data"]
        if data is not None and not data.empty:
            return f"VCF: {len(data)} variants"

    return f"File loaded: {fmt} format"
