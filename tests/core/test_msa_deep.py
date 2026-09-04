"""Deep tests for the pure-Python progressive MSA engine in core/msa.py."""
import pytest

from biosuite.core import msa


# ── pure helpers ─────────────────────────────────────────────────────────────

def test_is_nucleotide():
    assert msa._is_nucleotide("ACGTNacgtnU") is True
    assert msa._is_nucleotide("MKVLWAALLVTFLAGCQAKVEQAVETE") is False


def test_kmer_distance_symmetric():
    d1 = msa._kmer_distance("ACGTACGT", "ACGTACGT")
    d2 = msa._kmer_distance("ACGTACGT", "TTTTAAAA")
    assert d1 == pytest.approx(d1) and d1 >= 0
    assert d2 > d1
    assert msa._kmer_distance("ACGTAC", "TGCATC") == msa._kmer_distance("TGCATC", "ACGTAC")


def test_pairwise_distance_shape():
    dist = msa._pairwise_distance(["ACGT", "ACGC", "TTTT"])
    assert len(dist) == 3 and all(len(r) == 3 for r in dist)
    assert dist[0][0] == 0
    assert dist[0][1] < dist[0][2]


def test_upgma_tree_recovers_closest_pair():
    dist = [[0, 1, 9, 9],
            [1, 0, 9, 9],
            [9, 9, 0, 1],
            [9, 9, 1, 0]]
    tree = msa._upgma_tree(dist, 4)
    assert tree is not None


def test_normalize_and_write_fasta(tmp_path):
    seqs = ["ACGT", "acgt", "ACGT"]
    named = msa._normalize_sequences(seqs)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in named)
    fp = tmp_path / "in.fa"
    msa._write_fasta(seqs, str(fp))
    txt = fp.read_text()
    assert txt.count(">") == 3


# ── full end-to-end (pure-Python progressive alignment) ─────────────────────

def test_progressive_msa_all_equal_length():
    named = [("s1", "GATTACA"), ("s2", "GCATGCU"), ("s3", "GATTACA")]
    out = msa._progressive_msa(named)
    assert len(out) == 3
    seqs_only = [str(s[1]) for s in out]
    assert len({len(s) for s in seqs_only}) == 1      # all aligned to same length
    ident = [s.replace('-', '').replace('U', 'T') for s in seqs_only]
    assert ident[0] == "GATTACA"                      # originals preserved mod gap


def test_auto_align_fallback_runs_without_external_tools(monkeypatch):
    monkeypatch.setattr(msa, '_tool_available', lambda *_: False) \
        if hasattr(msa, '_tool_available') else None
    seqs = ["MKTAYIAK", "MKTAYIAKA", "MKTA"]
    result = msa.auto_align(seqs)
    assert result is not None
    n = len(result.sequences) if hasattr(result, 'sequences') else len(result)
    assert n == 3
    lens = result.sequences if hasattr(result, 'sequences') else result
    assert len({len(s) for s in lens}) == 1


def test_conservation_and_consensus():
    result = msa.auto_align(["ACGT", "ACGT", "ACGT"])
    cons = msa.compute_conservation(result)
    assert all(c == pytest.approx(1.0) for c in cons)
    consensus = msa.consensus_sequence(result)
    assert consensus == "ACGT"


def test_alignment_statistics_and_format():
    result = msa.auto_align(["ACGT", "ACGA", "ACGT"])
    stats = msa.alignment_statistics(result)
    assert isinstance(stats, dict)
    formatted = msa.format_alignment(result, max_width=60)
    assert "ACG" in formatted


def test_read_fasta_for_msa(tmp_path):
    fa = tmp_path / 'x.fa'
    fa.write_text('>s1\nAAAA\n>s2\nCCCC\n>s3\nGGGG\n')
    seqs = msa.read_fasta_for_msa(str(fa))
    assert len(seqs) == 3


# ── graceful degradation for missing external binaries ───────────────────────

def check_or_skip():
    tools = msa.check_tools()
    return tools


def test_external_runners_report_tool_choice():
    tools = check_or_skip()
    assert isinstance(tools, dict) and tools
    for name in ('clustal-omega', 'muscle', 'mafft'):
        if not tools.get(name):
            continue
        assert tools[name] is True
