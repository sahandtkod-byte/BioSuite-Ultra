"""End-to-end enrichment tests using small offline GO fixtures (goatools + gseapy)."""
import pytest

go = pytest.importorskip("goatools")
gp = pytest.importorskip("gseapy")

from biosuite.core import enrichment as en

OBO = """format-version: 1.2

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process

[Term]
id: GO:0000001
name: process alpha
namespace: biological_process
is_a: GO:0008150

[Term]
id: GO:0000002
name: process beta
namespace: biological_process
is_a: GO:0000001

[Term]
id: GO:0000003
name: process gamma
namespace: biological_process
is_a: GO:0000001

[Term]
id: GO:0000004
name: process delta
namespace: biological_process
is_a: GO:0008150
"""

# 20 background genes; GO:0000002 (depth>=2) carries exactly genes G1..G4.
ASSOC_ROWS = []
for i in range(1, 21):
    g = f"G{i}"
    ASSOC_ROWS.append((g, "GO:0000001"))          # all in alpha (depth 1 -> filtered)
for i in range(1, 5):
    ASSOC_ROWS.append((f"G{i}", "GO:0000002"))     # beta: G1-G4
for i in range(5, 13):
    ASSOC_ROWS.append((f"G{i}", "GO:0000003"))     # gamma: G5-G12
for i in range(13, 21):
    ASSOC_ROWS.append((f"G{i}", "GO:0000004"))     # delta: G13-G20

GMT = "\n".join([
    "SetA\toffline\t" + "\t".join(f"G{i}" for i in range(1, 6)),
    "SetB\toffline\t" + "\t".join(f"G{i}" for i in range(6, 11)),
]) + "\n"


@pytest.fixture()
def go_fixtures(tmp_path):
    obo = tmp_path / "go-basic.obo"
    obo.write_text(OBO)
    assoc = tmp_path / "assoc.tsv"
    assoc.write_text("\n".join(f"{g}\t{t}" for g, t in ASSOC_ROWS) + "\n")
    gmt = tmp_path / "sets.gmt"
    gmt.write_text(GMT)
    return str(obo), str(assoc), str(gmt)


def test_ora_detects_enriched_term(go_fixtures):
    obo, assoc, _ = go_fixtures
    rep = en.run_ora(["G1", "G2", "G3", "G4"], obo_file=obo, associations_file=assoc)
    assert rep.analysis_type == 'ORA'
    assert rep.background_count == 20
    assert rep.num_significant >= 1
    best = rep.results[0]
    assert best.term_id == "GO:0000002"
    assert best.adjusted_p_value < 0.05
    assert set(best.genes) == {"G1", "G2", "G3", "G4"}


def test_ora_no_overlap_graceful(go_fixtures):
    obo, assoc, _ = go_fixtures
    rep = en.run_ora(["X1", "X2"], obo_file=obo, associations_file=assoc)
    assert rep.num_significant == 0
    assert "No input genes" in (rep.message or "")


def test_ora_empty_list_graceful():
    rep = en.run_ora([])
    assert rep.num_input_genes == 0


def test_format_enrichment_report(go_fixtures):
    obo, assoc, _ = go_fixtures
    rep = en.run_ora(["G1", "G2", "G3", "G4"], obo_file=obo, associations_file=assoc)
    txt = en.format_enrichment_report(rep)
    assert "GO:0000002" in txt
    assert "process beta" in txt


def test_load_associations_file(go_fixtures):
    _, assoc, _ = go_fixtures
    loaded = en._load_associations_file(assoc)
    assert len(loaded) == 20
    assert loaded["G1"] == {"GO:0000001", "GO:0000002"}
    assert en._load_associations_file("/nonexistent/file.tsv") == {}


def test_gsea_real_prerank(go_fixtures):
    _, _, gmt = go_fixtures
    genes = [f"G{i}" for i in range(1, 11)]
    scores = [10.0, 9, 8, 7, 6, 1, 0.5, 0.4, 0.3, 0.2]   # G1..G5 rank top
    rep = en.run_gsea(genes, gene_scores=scores, gene_sets=gmt,
                      min_size=2, max_size=500, permutation_num=50, seed=7)
    assert rep.analysis_type == 'GSEA'
    assert rep.num_significant >= 1
    assert "SetA" in [r.term_name or r.term_id for r in rep.results]


def test_gsea_empty_graceful():
    rep = en.run_gsea([])
    assert rep.num_input_genes == 0
