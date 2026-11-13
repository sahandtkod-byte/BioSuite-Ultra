"""Regression tests for the MSA module bugs found in the deep review.

Bugs pinned here:
1. auto_align advertised List[str] but engines expected (name, seq)
   tuples — plain strings were silently mis-parsed (first two letters
   of each sequence became the "name" and "sequence").
2. The single-pass progressive merge could leave several clusters
   behind, producing alignments with DIFFERENT lengths for 4+ inputs.
3. compute_conservation called a nonexistent get_alignment_length()
   and subscripted the MSA object — it crashed 100% of the time.
4. alignment_statistics silently reported zeros when the engine had
   not pre-populated the conservation field.
"""
from biosuite.core.msa import (auto_align, compute_conservation,
                                consensus_sequence, alignment_statistics)


class TestStringInput:
    def test_plain_strings_produce_named_records(self):
        aln = auto_align(["ACGTGGCATTA", "ACGTGGCATGA", "ACGTGGCTTTA"])
        assert aln.num_sequences == 3
        assert all(n.startswith("seq") for n in aln.names)

    def test_conserved_prefix_stays_in_register(self):
        aln = auto_align(["ACGTGGCATTA", "ACGTGGCATGA", "ACGTGGCTTTA", "ACGTGGCATTAC"])
        seqs = aln.sequences_only
        assert {s[0] for s in seqs} == {'A'}
        assert {s[1] for s in seqs} == {'C'}

    def test_tuple_input_still_works(self):
        aln = auto_align([("a", "ACGT"), ("b", "AGGT")])
        assert aln.names == ["a", "b"]


class TestUniformAlignmentLength:
    def test_four_sequences_same_length(self):
        aln = auto_align(["ACGTGGCATTA", "ACGTGGCATGA", "ACGTGGCTTTA", "ACGTGGCATTAC"])
        assert len({len(s) for s in aln.sequences_only}) == 1

    def test_many_varying_lengths_same_length(self):
        import random
        random.seed(7)
        pool = [(f"s{i}", "".join(random.choice("ACGT")
                                 for _ in range(random.randint(20, 40))))
                for i in range(8)]
        aln = auto_align(pool)
        assert aln.num_sequences == 8
        assert len({len(s) for s in aln.sequences_only}) == 1


class TestConservationAndStats:
    def _aln(self):
        return auto_align(["ACGTGGCATTA", "ACGTGGCATGA", "ACGTGGCTTTA", "ACGTGGCATTAC"])

    def test_compute_conservation_does_not_crash(self):
        cons = compute_conservation(self._aln())
        assert len(cons) == self._aln().alignment_length
        assert cons[0] == 1.0 and cons[1] == 1.0

    def test_statistics_not_zeroed(self):
        stats = alignment_statistics(self._aln())
        assert stats['mean_conservation'] > 0.7
        assert stats['highly_conserved'] >= 6
        assert stats['num_sequences'] == 4

    def test_consensus_majority_rule(self):
        c = consensus_sequence(self._aln())
        assert c.startswith("ACGTGGC")


class TestEdgeCases:
    def test_empty_and_single(self):
        assert auto_align([]).method == 'none'
        assert auto_align(["ACGT"]).method == 'none'

    def test_two_sequences(self):
        aln = auto_align(["ACGT", "ACGT"])
        assert aln.sequences_only == ["ACGT", "ACGT"]
        assert compute_conservation(aln) == [1.0, 1.0, 1.0, 1.0]
