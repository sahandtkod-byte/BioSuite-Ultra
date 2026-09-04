"""Regression tests for go_browser.py review fixes."""
import pytest
from biosuite.core.go_browser import (
    GOBrowser, GOTerm, go_enrichment, format_go_results,
)


def test_builtin_hierarchy_navigation():
    b = GOBrowser()
    nuc = b.get_term("GO:0005634")
    assert nuc.name == "nucleus"
    kids = [t.go_id for t in b.get_children("GO:0005575")]
    assert "GO:0005634" in kids
    anc = [t.go_id for t in b.get_ancestors("GO:0005829")]   # cytosol
    assert "GO:0005737" in anc and "GO:0005575" in anc


def test_obo_load_handles_root_terms(monkeypatch):
    """Root terms (no parents) used to IndexError during OBO loading."""
    from biosuite.core import go_browser as G
    monkeypatch.setattr(G, 'HAS_GOATOOLS', True)

    class Item:
        def __init__(self, name, ns, parents):
            self.name, self.namespace, self.parents = name, ns, parents

        @property
        def def_(self):
            return ''

    root = Item('root', 'biological_process', [])
    leaf_p = Item.__new__(Item)
    leaf_p.id = 'GO:0000000'
    child = Item('child', 'biological_process', [leaf_p])

    monkeypatch.setattr(G, 'GODag', lambda p: {'GO:0000000': root, 'GO:0000001': child})
    monkeypatch.setattr(G.os.path, 'exists', lambda p: True)
    gb = G.GOBrowser(obo_file='fake.obo')
    assert gb.get_term('GO:0000000').parents == []
    assert gb.get_term('GO:0000001').parents == ['GO:0000000']


def test_enrichment_fisher_table_consistency():
    genes = ['g%d' % i for i in range(10)]
    m = {'GO:1': ['g0', 'g1', 'g2', 'g3'], 'GO:2': ['other_a', 'other_b']}
    res = go_enrichment(genes, m, background_size=100)
    by = {r['go_term']: r for r in res}
    assert by['GO:1']['count'] == 4
    assert by['GO:1']['p_value'] < by['GO:2']['p_value']  # enrichment ranks strong hits first
    assert by['GO:2']['count'] == 0
    assert res[0]['go_term'] == 'GO:1'


def test_search_and_format():
    b = GOBrowser()
    hits = b.search('kinase')
    assert hits
    txt = format_go_results(hits)
    assert 'GO ID' in txt and 'kinase' in txt.lower()
    assert format_go_results([]) == "No GO terms found."
