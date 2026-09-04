"""Core performance benchmarks for BioSuite Ultra.

Runs with plain pytest (uses wall-clock timing, no pytest-benchmark
dependency required) so it works in CI and local dev identically.

Each test asserts the operation completes inside a generous time budget;
budgets are regression guards, not tight performance targets — tighten
them as the code gets faster.

Run:
    pytest benchmarks/ -v -s
"""
from __future__ import annotations

import random
import time

import pytest

from biosuite.core.sequence import gc_content, reverse_complement, translate
from biosuite.core.alignment import needleman_wunsch, smith_waterman
from biosuite.core.cloning import simulate_digestion

random.seed(42)

DNA = "ACGT"


def _rand_seq(n: int) -> str:
    return "".join(random.choice(DNA) for _ in range(n))


@pytest.fixture(scope="module")
def seq_10k() -> str:
    return _rand_seq(10_000)


@pytest.fixture(scope="module")
def pair_1k() -> tuple[str, str]:
    return _rand_seq(1_000), _rand_seq(1_000)


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


class TestSequenceBenchmarks:
    def test_gc_content_10k(self, seq_10k):
        _, elapsed = _timed(gc_content, seq_10k)
        print(f"\ngc_content(10 kb): {elapsed*1000:.2f} ms")
        assert elapsed < 0.5

    def test_reverse_complement_10k(self, seq_10k):
        _, elapsed = _timed(reverse_complement, seq_10k)
        print(f"\nreverse_complement(10 kb): {elapsed*1000:.2f} ms")
        assert elapsed < 0.5

    def test_translate_10k(self, seq_10k):
        _, elapsed = _timed(translate, seq_10k)
        print(f"\ntranslate(10 kb): {elapsed*1000:.2f} ms")
        assert elapsed < 1.0


class TestAlignmentBenchmarks:
    def test_needleman_wunsch_1k(self, pair_1k):
        s1, s2 = pair_1k
        (_, _, score), elapsed = _timed(needleman_wunsch, s1, s2)
        print(f"\nneedleman_wunsch(1 kb x 1 kb): {elapsed:.2f} s  score={score}")
        assert elapsed < 10.0

    def test_smith_waterman_1k(self, pair_1k):
        s1, s2 = pair_1k
        (_, _, score), elapsed = _timed(smith_waterman, s1, s2)
        print(f"\nsmith_waterman(1 kb x 1 kb): {elapsed:.2f} s  score={score}")
        assert elapsed < 10.0


class TestCloningBenchmarks:
    def test_digest_plasmid_ecori(self):
        plasmid = _rand_seq(5_000)
        result, elapsed = _timed(simulate_digestion, plasmid, enzyme="EcoRI")
        print(f"\nsimulate_digestion(5 kb, EcoRI): {elapsed*1000:.2f} ms  "
              f"cuts={len(result['cuts'])}")
        assert elapsed < 1.0
