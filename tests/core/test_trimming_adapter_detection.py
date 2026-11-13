"""How the built-in trimmer finds adapters and cuts reads.

The adapter search is an exact substring match and takes the earliest hit across the
candidate list; the quality cut runs backwards from the 3' end only, so a read that is
junk at the 5' end survives intact - the module docstring says so now, and the last two
tests here pin both properties.  Refusals (bad FASTQ files, bad arguments) live in
``test_trimming_fastq_validation.py`` and ``test_trimming_parameter_validation.py``;
what each report number means lives in ``test_trimming_report_statistics.py``.
"""
import pytest

from biosuite.core import trimming as tr
from biosuite.core.trimming import (
    trim_fastq, trim_pair_end, _pure_python_trim, _resolve_adapters,
    _find_adapter, ADAPTERS,
)


def _fq(recs):
    return ''.join(f"@{h}\n{s}\n+\n{q}\n" for h, s, q in recs)


def test_auto_adapter_detected_builtin(tmp_path):
    ad = ADAPTERS['illumina_truseq_rna']
    seq = "ACGTACGTACGTACGTACGT" + ad + "TTTT"
    qual = "I" * len(seq)
    inp = tmp_path / "in.fq"
    inp.write_text(_fq([("r1", seq, qual)]))
    out = tmp_path / "out.fq"
    rep = _pure_python_trim(str(inp), str(out), 20, 10,
                            _resolve_adapters('auto', None))
    assert rep.adapter_trimmed == 1  # 'auto' previously meant NO adapters at all
    body = out.read_text().split('\n')
    assert ad not in body[1] and len(body[1]) == 20


def test_explicit_adapter_name_and_value(tmp_path):
    assert _resolve_adapters('auto', 'polya') == [ADAPTERS['polya']]
    assert ADAPTERS['illumina_nextera'] in _resolve_adapters('auto', None)
    assert _resolve_adapters('CUSTOMADAPTER', None) == ['CUSTOMADAPTER']


def test_find_adapter_picks_earliest():
    assert _find_adapter("TTGCCTGGCC", ["CCTGG", "GCC"]) == 2  # earliest hit wins
    assert _find_adapter("AAAA", [ADAPTERS['polyg']]) == -1


def test_quality_trim_from_three_prime(tmp_path):
    seq = "ACGT" * 10
    qual = "I" * 36 + "!" * 4
    inp = tmp_path / "in.fq"
    inp.write_text(_fq([("r1", seq, qual)]))
    out = tmp_path / "out.fq"
    rep = _pure_python_trim(str(inp), str(out), 20, 10, [])
    lines = out.read_text().split('\n')
    assert lines[1] == "ACGT" * 9 and rep.reads_trimmed == 1


def test_pair_end_mate_sync_lockstep(tmp_path):
    # R1 read #2 is junk-quality (will be dropped): mate #2 must ALSO leave.
    r1s = [("p1", "ACGT" * 15, "I" * 60),
           ("p2", "ACGT" * 15, "!" * 60),
           ("p3", "ACGT" * 15, "I" * 60)]
    r2s = [("p1", "TTGG" * 15, "I" * 60),
           ("p2", "TTGG" * 15, "I" * 60),
           ("p3", "TTGG" * 15, "I" * 60)]
    i1, i2 = tmp_path / "r1.fq", tmp_path / "r2.fq"
    o1, o2 = tmp_path / "o1.fq", tmp_path / "o2.fq"
    i1.write_text(_fq(r1s))
    i2.write_text(_fq(r2s))
    rep = trim_pair_end(str(i1), str(i2), str(o1), str(o2), 20, 36, 'none')
    h1 = [l for l in o1.read_text().split('\n') if l.startswith('@')]
    h2 = [l for l in o2.read_text().split('\n') if l.startswith('@')]
    assert h1 == h2 == ['@p1', '@p3']  # synchronized
    assert rep.reads_removed == 2  # counted in reads, not pairs


def test_pair_end_adapter_in_r2(tmp_path):
    ad = ADAPTERS['illumina_nextera']
    r1s = [("p1", "ACGT" * 15, "I" * 60)]
    r2s = [("p1", "TTGG" * 10 + ad, "I" * (40 + len(ad)))]
    i1, i2 = tmp_path / "r1.fq", tmp_path / "r2.fq"
    o1, o2 = tmp_path / "o1.fq", tmp_path / "o2.fq"
    i1.write_text(_fq(r1s))
    i2.write_text(_fq(r2s))
    trim_pair_end(str(i1), str(i2), str(o1), str(o2), 20, 36, 'auto')
    assert ad not in o2.read_text()
    h1 = [l for l in o1.read_text().split('\n') if l.startswith('@')]
    h2 = [l for l in o2.read_text().split('\n') if l.startswith('@')]
    assert h1 == h2 == ['@p1']


def test_min_length_drops_whole_read(tmp_path):
    inp = tmp_path / "in.fq"
    inp.write_text(_fq([("r1", "ACGT", "IIII"), ("r2", "ACGT" * 15, "I" * 60)]))
    out = tmp_path / "out.fq"
    rep = trim_fastq(str(inp), str(out), 20, 36, 'none')
    assert rep.reads_removed == 1
    assert out.read_text().count('@') == 1


def test_missing_input_file():
    rep = trim_fastq("/nonexistent.fq")
    assert 'not found' in rep.message.lower()


def test_the_three_prime_tail_is_trimmed_and_the_five_prime_end_is_left_alone(tmp_path):
    """The built-in engine trims the 3' quality tail and cuts adapters; it is not a
    sliding window, which is what this module's docstring used to promise.  A caller who
    needs 5' cleaning has to run cutadapt, and the docstring says so."""
    inp = tmp_path / "in.fq"
    inp.write_text(_fq([("r1", "ACGTACGTAC", "!" * 8 + "II")]))
    out = tmp_path / "out.fq"
    rep = trim_fastq(str(inp), str(out), 20, 2, 'polyg')

    assert (rep.reads_trimmed, rep.reads_removed) == (0, 0)
    assert out.read_text() == "@r1\nACGTACGTAC\n+\n" + "!" * 8 + "II" + "\n"
    docstring = " ".join(tr.__doc__.split())
    assert "not a sliding window" in docstring
    assert "does not touch the 5' end" in docstring


def test_a_valid_file_still_trims_adapters_after_the_validation_pass(tmp_path):
    """Positive control for the FASTQ validator: a well-formed file must not pay for it,
    and what it writes must be a FASTQ another reader can open."""
    adapter = ADAPTERS['illumina_nextera']
    seq = "ACGT" * 15
    inp = tmp_path / "in.fq"
    inp.write_text(_fq([("r%d" % i, seq + adapter, "I" * (len(seq) + len(adapter)))
                        for i in range(2)]))
    out = tmp_path / "out.fq"
    rep = trim_fastq(str(inp), str(out), 20, 10, 'auto', 'illumina_nextera')

    assert (rep.total_reads, rep.reads_trimmed, rep.adapter_trimmed,
            rep.reads_removed) == (2, 2, 2, 0)
    lines = out.read_text().splitlines()
    assert lines[1] == seq and len(lines[1]) == len(lines[3])
    assert lines[0].startswith('@') and lines[2] == '+'
    assert adapter not in out.read_text()
