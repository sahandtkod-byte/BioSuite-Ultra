"""
Unit tests for biosuite.core.file_formats module.
"""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biosuite.core.file_formats import (
    parse_bed, parse_gff, parse_newick, parse_stockholm,
    tree_to_newick, tree_to_ascii, TreeNode,
    bed_to_dataframe, gff_to_dataframe,
    format_bed_summary, format_gff_summary
)


class TestBEDParser:
    def test_basic_bed(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as f:
            f.write("chr1\t1000\t2000\tgene1\t0\t+\n")
            f.write("chr2\t3000\t4000\tgene2\t10\t-\n")
            f.flush()
            records = parse_bed(f.name)
        os.unlink(f.name)
        assert len(records) == 2
        assert records[0].chrom == 'chr1'
        assert records[0].start == 1000
        assert records[0].end == 2000
        assert records[0].strand == '+'

    def test_bed_with_comments(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as f:
            f.write("# comment\n")
            f.write("chr1\t100\t200\n")
            f.flush()
            records = parse_bed(f.name)
        os.unlink(f.name)
        assert len(records) == 1

    def test_empty_bed(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as f:
            f.write("")
            f.flush()
            records = parse_bed(f.name)
        os.unlink(f.name)
        assert len(records) == 0


class TestGFFParser:
    def test_basic_gff(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gff', delete=False) as f:
            f.write("chr1\tensembl\tgene\t1000\t2000\t.\t+\t.\tID=gene1;Name=TP53\n")
            f.flush()
            records = parse_gff(f.name)
        os.unlink(f.name)
        assert len(records) == 1
        assert records[0].seqid == 'chr1'
        assert records[0].feature == 'gene'
        assert records[0].attributes.get('ID') == 'gene1'


class TestNewickParser:
    def test_simple_tree(self):
        tree = parse_newick("(A:0.1,B:0.2)C:0.3;")
        assert tree.name == 'C'
        assert len(tree.children) == 2
        assert tree.children[0].name == 'A'
        assert tree.children[0].branch_length == 0.1

    def test_nested_tree(self):
        tree = parse_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);")
        assert len(tree.children) == 2
        assert len(tree.children[0].children) == 2

    def test_roundtrip(self):
        tree = parse_newick("(A:0.1,B:0.2):0.3;")
        newick = tree_to_newick(tree)
        assert 'A' in newick
        assert 'B' in newick


class TestTreeNode:
    def test_leaf(self):
        node = TreeNode(name='A', branch_length=0.1)
        assert node.is_leaf
        assert node.name == 'A'

    def test_internal(self):
        node = TreeNode(name='parent')
        node.children.append(TreeNode(name='child1'))
        node.children.append(TreeNode(name='child2'))
        assert len(node.children) == 2


class TestASCII:
    def test_simple_tree(self):
        tree = parse_newick("(A:0.1,B:0.2);")
        lines = tree_to_ascii(tree)
        assert len(lines) > 0
        assert any('A' in line for line in lines)


class TestFormatters:
    def test_bed_summary(self):
        records = parse_bed.__doc__  # placeholder
        formatted = format_bed_summary([])
        assert 'BED' in formatted

    def test_gff_summary(self):
        formatted = format_gff_summary([])
        assert 'GFF' in formatted
