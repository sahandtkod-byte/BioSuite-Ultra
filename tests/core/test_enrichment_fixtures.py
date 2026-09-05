"""ORA/GSEA with local synthetic OBO + associations (no network, real goatools statistics)."""
import pytest

from biosuite.core import enrichment as en


OBO = """format-version: 1.2
data-version: test/1.0

[Term]
id: GO:0000001
name: root process
namespace: biological_process

[Term]
id: GO:0000002
name: mitochondrion maintenance
is_a: GO:0000001 ! root process
namespace: biological_process

[Term]
id: GO:0000003
name: DNA repair
is_a: GO:0000002 ! mitochondrion maintenance
namespace: biological_process
"""

# genes G1..G3 map strongly to the deep term; background genes spread on root/shallow
ASSOC = (
    "G1\tGO:0000003\n"
    "G2\tGO:0000003\n"
    "G3\tGO:0000003\n"
    "B1\tGO:0000001\n"
    "B2\tGO:0000001\n"
    "B3\tGO:0000001\n"
    "B4\tGO:0000002\n"
    "B5\tGO:0000002\n"
    "B6\tGO:0000002\n"
    "B7\tGO:0000002\n"
    "B8\tGO:0000001\n"
    "B9\tGO:0000001\n"
)


@pytest.fixture
def obo(tmp_path):
    p = tmp_path / 'go.obo'
    p.write_text(OBO)
    return str(p)


@pytest.fixture
def assoc(tmp_path):
    p = tmp_path / 'assoc.tsv'
    p.write_text(ASSOC)
    return str(p)


def test_ora_finds_enriched_deep_term(obo, assoc):
    rep = en.run_ora(['G1', 'G2', 'G3'], obo_file=obo,
                     associations_file=assoc, cutoff=0.05)
    assert rep.num_significant >= 1
    hit = rep.results[0]
    assert hit.term_id == 'GO:0000003'
    assert hit.adjusted_p_value < 0.05
    assert rep.analysis_type == 'ORA'


def test_ora_empty_list():
    rep = en.run_ora([], organism='human')
    assert rep.num_significant == 0
    assert 'empty' in rep.message.lower()


def test_ora_genes_not_in_background(obo, assoc):
    rep = en.run_ora(['X', 'Y'], obo_file=obo, associations_file=assoc)
    assert rep.num_significant == 0
    assert 'no input genes' in rep.message.lower()


def test_ora_bad_associations_file(obo):
    rep = en.run_ora(['G1'], obo_file=obo, associations_file='/nonexistent/p')
    assert rep.num_significant == 0


def test_format_enrichment_report_filled(obo, assoc):
    rep = en.run_ora(['G1', 'G2', 'G3'], obo_file=obo, associations_file=assoc)
    txt = en.format_enrichment_report(rep)
    assert 'GO:0000003' in txt or 'ORA' in txt


def test_load_associations_file_nonexistent():
    assert en._load_associations_file('/nonexistent/x') == {}


def test_get_associations_unknown_organism():
    assert en._get_associations('martian') == {}
