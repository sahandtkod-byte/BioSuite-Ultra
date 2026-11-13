"""Assembly: read/contig loading, N50/L50 edges and the SPAdes/MEGAHIT wrappers.

Pins the FASTA/FASTQ reader on blank lines, sequence-less records and empty
files, the N50/L50 answer for an empty contig list, the exact commands the two
external assemblers are asked to run, and every failure path of those wrappers
(non-zero exit, empty output, missing tool) — which must fall back to the
built-in engine and say so in the log rather than report a bogus assembly.
No external binary is ever executed here.
"""
import os
import subprocess
import tempfile

import pytest

from biosuite.core import assembly as asm
from biosuite.core.assembly import (AssemblyResult, _compute_l50, _compute_n50,
                                    _load_reads_fasta, _megahit_assemble,
                                    _parse_assembly_fasta, _spades_assemble,
                                    assemble, check_assembly_tools,
                                    format_assembly_report)

CONTIGS_FASTA = ">NODE_1_length_100\nACGTACGTAC\n>NODE_2_length_50\nGGGGG\n"


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_tool(write_contigs, returncode=0, record=None):
    """subprocess.run stand-in that plants a SPAdes/MEGHIT output tree."""
    def run(cmd, **kwargs):
        if record is not None:
            record.append((list(cmd), kwargs))
        if write_contigs is not None:
            out_dir = cmd[cmd.index('-o') + 1]
            name = 'contigs.fasta' if cmd[0] == 'spades.py' else 'final.contigs.fa'
            with open(os.path.join(out_dir, name), 'w', encoding='utf-8') as fh:
                fh.write(write_contigs)
        return _Completed(returncode, "", "tool exploded" if returncode else "")
    return run


def _reads_file(tmp_path, text, name="reads.fa"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


@pytest.mark.parametrize("lengths,n50,l50", [
    ([8, 7, 6, 5, 4], 7, 2),
    ([10], 10, 1),
    ([6, 6, 6, 6], 6, 2),
    ([100, 1], 100, 1),
    ([1], 1, 1),
    ([], 0, 0),
])
def test_n50_and_l50_answer_for_known_contig_length_lists(lengths, n50, l50):
    assert _compute_n50(lengths) == n50
    assert _compute_l50(lengths) == l50


def test_a_toolless_installation_reports_both_external_assemblers_absent():
    assert check_assembly_tools() == {'spades': False, 'megahit': False}


def test_fasta_reads_join_wrapped_lines_and_skip_blank_ones(tmp_path):
    path = _reads_file(tmp_path, ">r1\n\nAC\nGT\n\n>r2\nTTGG\n>r3\n")
    assert _load_reads_fasta(path) == [("r1", "ACGT"), ("r2", "TTGG")]


def test_fastq_reads_with_an_empty_sequence_line_are_dropped(tmp_path):
    path = _reads_file(tmp_path, "@r1\n\n+\n\n@r2\nACGT\n+\nIIII\n", name="r.fq")
    assert _load_reads_fasta(path) == [("r2", "ACGT")]


def test_an_empty_reads_file_loads_no_reads(tmp_path):
    assert _load_reads_fasta(_reads_file(tmp_path, "")) == []


def test_assembling_an_empty_reads_file_reports_no_contigs_without_temp_litter(
        tmp_path, monkeypatch):
    reserved = []
    monkeypatch.setattr(asm, "secure_temp_path",
                        lambda suffix="", prefix="biosuite_": reserved.append(suffix)
                        or os.path.join(str(tmp_path), "unused" + suffix))
    res = asm._builtin_assembly(_reads_file(tmp_path, ""))

    assert (res.engine, res.num_contigs, res.output_fasta, res.message) \
        == ("builtin", 0, "", "No contigs")
    assert reserved == [], "a read file with no reads must not reserve an output path"
    assert list(tmp_path.iterdir()) == [tmp_path / "reads.fa"]


def test_spades_contigs_are_parsed_into_the_reported_stats(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(asm, "_has_tool", lambda name: name in ('spades.py',))
    monkeypatch.setattr(asm.subprocess, "run",
                        _fake_tool(CONTIGS_FASTA, 0, record=calls))

    res = assemble(_reads_file(tmp_path, ">r1\nACGTACGTAC\n"), tool='spades')

    cmd, kwargs = calls[0]
    assert cmd[:3] == ['spades.py', '-s', str(tmp_path / "reads.fa")]
    assert cmd[3] == '-o' and os.path.isabs(cmd[4])
    assert cmd[5:] == ['--only-assembler', '-t', '4']
    assert kwargs == {'capture_output': True, 'text': True, 'timeout': 14400}
    assert (res.engine, res.num_contigs, res.total_length, res.n50, res.l50) \
        == ("spades", 2, 15, 10, 1)
    assert res.gc_content == pytest.approx(10 / 15 * 100), \
        "five GC in ACGTACGTAC plus five in GGGGG, over fifteen bases"
    assert res.message == "spades: 2 contigs, N50=10"
    assert res.output_fasta.endswith("contigs.fasta")


def test_a_contig_soft_masked_in_lower_case_is_still_counted_for_gc():
    """Assemblers mask low-quality regions by writing them in lower case; a
    statistic that follows the case would call those bases neither G nor C."""
    written = asm._compute_assembly_stats(["ACGTACGTAC", "GGGGG"])
    masked = asm._compute_assembly_stats(["acgtACGTac", "GgGgG"])

    assert written.gc_content == pytest.approx(10 / 15 * 100), written
    assert masked.gc_content == pytest.approx(written.gc_content), masked
    assert masked.total_length == written.total_length


def test_megahit_command_is_the_metagenomic_core_invocation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(asm, "_has_tool", lambda name: name == 'megahit-core')
    monkeypatch.setattr(asm.subprocess, "run",
                        _fake_tool(CONTIGS_FASTA, 0, record=calls))

    res = assemble(_reads_file(tmp_path, ">r1\nACGT\n"), tool='megahit', threads=8)

    assert calls[0][0][:3] == ['megahit-core', '-1', str(tmp_path / "reads.fa")]
    assert calls[0][0][3] == '-o' and os.path.isabs(calls[0][0][4])
    assert calls[0][0][5:] == ['--min-contig-len', '200', '-t', '8']
    assert res.engine == "megahit" and res.num_contigs == 2


@pytest.mark.parametrize("engine", ["spades", "megahit"])
def test_a_nonzero_exit_is_reported_and_the_builtin_engine_takes_over(
        tmp_path, monkeypatch, caplog, engine):
    monkeypatch.setattr(asm, "_has_tool", lambda name: True)
    monkeypatch.setattr(asm.subprocess, "run", _fake_tool(CONTIGS_FASTA, 1))

    with caplog.at_level("WARNING", logger="biosuite.core.assembly"):
        res = assemble(_reads_file(tmp_path, ">r1\nACGTACGTACGTACGTACGTAC\n"),
                       tool=engine)

    assert res.engine == "builtin", "the failed external assembly must not be reported"
    assert res.num_contigs == 1
    verb = "SPAdes" if engine == "spades" else "MEGAHIT"
    assert f"{verb} assembly failed (exit 1): tool exploded" in caplog.text


@pytest.mark.parametrize("engine", ["spades", "megahit"])
def test_an_empty_contig_file_from_the_tool_is_not_an_assembly(tmp_path, monkeypatch,
                                                               caplog, engine):
    monkeypatch.setattr(asm, "_has_tool", lambda name: True)
    monkeypatch.setattr(asm.subprocess, "run", _fake_tool(""))

    with caplog.at_level("WARNING", logger="biosuite.core.assembly"):
        res = assemble(_reads_file(tmp_path, ">r1\nACGTACGTACGTACGTACGTAC\n"),
                       tool=engine)

    assert res.engine == "builtin"
    assert "assembly failed (exit 0)" in caplog.text


@pytest.mark.parametrize("engine", ["spades", "megahit"])
def test_a_tool_that_never_ran_leaves_no_output_tree(tmp_path, monkeypatch, engine):
    monkeypatch.setattr(asm, "_has_tool", lambda name: True)
    monkeypatch.setattr(asm.subprocess, "run", _fake_tool(None, 0))
    tmpdir = tempfile.gettempdir()
    before = {p for p in os.listdir(tmpdir) if p.startswith("biosuite_" + engine + "_")}

    res = assemble(_reads_file(tmp_path, ">r1\nACGTACGTACGTACGTACGTAC\n"), tool=engine)

    assert res.engine == "builtin"
    after = {p for p in os.listdir(tmpdir) if p.startswith("biosuite_" + engine + "_")}
    assert after - before == set(), f"{engine} left {sorted(after - before)} behind"


@pytest.mark.parametrize("exc", [OSError("spades.py not found"),
                                 subprocess.TimeoutExpired(cmd="spades.py", timeout=1)])
def test_a_tool_that_cannot_be_spawned_is_logged_and_replaced_by_the_builtin(
        tmp_path, monkeypatch, caplog, exc):
    def raiser(cmd, **kwargs):
        raise exc

    monkeypatch.setattr(asm, "_has_tool", lambda name: True)
    monkeypatch.setattr(asm.subprocess, "run", raiser)

    with caplog.at_level("WARNING", logger="biosuite.core.assembly"):
        res = assemble(_reads_file(tmp_path, ">r1\nACGTACGTACGTACGTACGTAC\n"),
                       tool='spades')

    assert res.engine == "builtin"
    assert "SPAdes could not be run:" in caplog.text


def test_the_wrapper_itself_returns_none_when_the_tool_produces_nothing(tmp_path):
    out_dir = tmp_path / "spades_out"
    out_dir.mkdir()
    assert _spades_assemble(str(tmp_path / "reads.fa"), str(out_dir)) is None
    assert _megahit_assemble(str(tmp_path / "reads.fa"), str(out_dir)) is None


def test_assemble_writes_the_builtin_contigs_to_an_explicit_output_path(tmp_path):
    out = tmp_path / "chosen.fasta"
    res = assemble(_reads_file(tmp_path, ">r1\nACGTACGTACGTACGTACGTAC\n"), str(out))

    assert res.output_fasta == str(out) and res.n50 == 22
    assert out.read_text().splitlines() == [">contig_1 len=22", "ACGTACGTACGTACGTACGTAC"]


def test_assemble_refuses_a_missing_read_file_without_reserving_output(tmp_path):
    res = assemble(str(tmp_path / "never_written.fa"))

    assert res.engine == "none"
    assert res.message == f"Reads file not found: {tmp_path / 'never_written.fa'}"
    assert list(tmp_path.iterdir()) == []


def test_a_header_without_sequence_is_not_counted_as_a_contig(tmp_path):
    """Blank lines between the header and the sequence are noise, but a header
    that carries no sequence at all used to be parsed into a zero-length contig:
    num_contigs was inflated, min_contig became 0 and the N50/L50 arithmetic read
    that empty entry as part of the assembly."""
    path = tmp_path / "contigs.fa"
    path.write_text(">c1\n\nACGT\nACGT\n>c2\n\n")
    res = _parse_assembly_fasta(str(path), "test-engine")

    assert res.contig_lengths == [8]
    assert (res.num_contigs, res.total_length, res.n50, res.l50) == (1, 8, 8, 1)
    assert (res.min_contig, res.max_contig) == (8, 8)
    assert res.gc_content == pytest.approx(50.0)
    assert res.message == "test-engine: 1 contigs, N50=8"


def test_parsing_an_empty_contig_file_yields_zero_contigs(tmp_path):
    path = tmp_path / "empty.fa"
    path.write_text("")
    res = _parse_assembly_fasta(str(path), "megahit")

    assert (res.engine, res.num_contigs, res.contig_lengths, res.n50) == ("megahit", 0, [], 0)
    assert res.message == "megahit: 0 contigs, N50=0"
    assert res.output_fasta == str(path)


def test_report_has_no_note_line_when_the_result_carries_no_message():
    lines = format_assembly_report(AssemblyResult(engine='builtin')).splitlines()

    assert lines == ["=== Genome Assembly Report ===",
                     "Engine: builtin",
                     "Contigs: 0",
                     "Total length: 0 bp",
                     "N50: 0 bp",
                     "L50: 0",
                     "GC content: 0.0%",
                     "Longest contig: 0 bp"]
    assert not any(line.startswith("Note:") for line in lines)
