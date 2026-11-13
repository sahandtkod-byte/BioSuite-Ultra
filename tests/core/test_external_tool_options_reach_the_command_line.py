"""Options that were accepted and dropped on the floor before reaching a tool.

`run_blast(extra_args=...)` never appended them to the BLAST+ command line and
`call_peaks(genome=...)` never forwarded the species to MACS2's `-g`.  Measured
before the change: `call_peaks(..., genome='mm')` produced exactly the same
argv as the default — `-g hs` — while the MACS2 background model is derived from
that number, so a mouse experiment was scored against the human genome size.
"""
import re
import subprocess

import pytest

from biosuite.core import assembly as asm
from biosuite.core import blast as bl
from biosuite.core import peak_calling as pc

QUERY = ">q\nACGTACGTACGT\n"
DATABASE = ">d\nACGTACGTACGT\n"


class _Run:
    """Stub subprocess.run that records the argv it was handed."""

    def __init__(self, returncode=0, stdout="", stderr="stub"):
        self.calls = []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, cmd, capture_output=True, text=True, timeout=None):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self.returncode,
                                           stdout=self.stdout, stderr=self.stderr)


def _fasta(tmp_path, text, name):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_extra_args_reach_the_blast_command_line_in_order(tmp_path, monkeypatch):
    run = _Run(returncode=1)
    monkeypatch.setattr(bl.subprocess, "run", run)
    query = _fasta(tmp_path, QUERY, "q.fa")
    db = _fasta(tmp_path, DATABASE, "db.fa")

    bl._blast_plus_search(query, db, extra_args=['-soft_masking', 'false',
                                                 '-max_hsps', '2'])
    cmd = run.calls[-1]
    assert cmd[-4:] == ['-soft_masking', 'false', '-max_hsps', '2']
    assert cmd[:2] == ['blastn', '-query']
    assert cmd.count('-soft_masking') == 1


def test_without_extra_args_the_command_is_unchanged(tmp_path, monkeypatch):
    run = _Run(returncode=1)
    monkeypatch.setattr(bl.subprocess, "run", run)
    query = _fasta(tmp_path, QUERY, "q.fa")
    db = _fasta(tmp_path, DATABASE, "db.fa")

    bl._blast_plus_search(query, db)
    assert run.calls[-1][-3:] == ['500', '-num_threads', '1']


def test_run_blast_forwards_extra_args_and_keeps_what_the_tool_said(
        tmp_path, monkeypatch):
    query = _fasta(tmp_path, QUERY, "q.fa")
    db = _fasta(tmp_path, DATABASE, "db.fa")
    run = _Run(returncode=1)
    monkeypatch.setattr(bl.subprocess, "run", run)
    monkeypatch.setattr(bl, "_has_blast_plus", lambda: True)

    result = bl.run_blast(query, db, extra_args=['-task', 'blastn-short'])
    assert run.calls[-1][-2:] == ['-task', 'blastn-short']
    # The engine line, and then what the engine reported.  `run_blast` used to assign
    # its own line straight over `result.message`, so a blastn that exited 1 came back
    # as "Using BLAST+ (external)" with no hits and no reason — this pin now proves the
    # reason survives the dispatch.
    assert result.message.startswith("Using BLAST+ (external); BLAST+ error "
                                     "(blastn exited 1):")
    assert result.num_hits == 0 and result.engine == 'blast+'


def test_the_builtin_engine_names_the_options_it_did_not_apply(tmp_path, monkeypatch):
    query = _fasta(tmp_path, QUERY, "q.fa")
    db = _fasta(tmp_path, DATABASE, "db.fa")
    monkeypatch.setattr(bl, "_has_blast_plus", lambda: False)

    with pytest.warns(bl.PerformanceWarning):
        result = bl.run_blast(query, db, extra_args=['-task', 'blastn-short'])
    assert "extra_args ['-task', 'blastn-short'] are BLAST+ options and were " \
        "not applied" in result.message


def test_macs2_receives_the_genome_the_caller_asked_for(tmp_path, monkeypatch):
    run = _Run(returncode=1)
    monkeypatch.setattr(pc.subprocess, "run", run)
    monkeypatch.setattr(pc, "check_peak_tools", lambda: {'macs2': True})
    reads = _fasta(tmp_path, "", "r.bam")

    report = pc.call_peaks(reads, genome='mm')
    cmd = run.calls[-1]
    assert cmd[cmd.index('-g') + 1] == 'mm', "MACS2 built its background for mouse"
    assert cmd[:4] == ['macs2', 'callpeak', '-t', reads]
    # the run above exits non-zero, so the honest fallback message is the builtin one
    assert (report.engine, report.message) == ('builtin', 'No mapped reads found.')


def test_macs2_keeps_the_human_default_when_nothing_is_given(tmp_path, monkeypatch):
    run = _Run(returncode=1)
    monkeypatch.setattr(pc.subprocess, "run", run)
    monkeypatch.setattr(pc, "check_peak_tools", lambda: {'macs2': True})
    reads = _fasta(tmp_path, "", "r.bam")
    pc.call_peaks(reads)
    assert run.calls[-1][run.calls[-1].index('-g') + 1] == 'hs'


def test_a_successful_macs2_run_reports_the_genome_it_used(tmp_path, monkeypatch):
    run = _Run(returncode=0)
    monkeypatch.setattr(pc.subprocess, "run", run)
    monkeypatch.setattr(pc, "check_peak_tools", lambda: {'macs2': True})
    reads = _fasta(tmp_path, "", "r.bam")
    report = pc.call_peaks(reads, genome='mm')
    assert report.engine == 'macs2'
    assert report.message == "Using MACS2 (external, genome=mm)"


def test_an_explicit_request_for_a_failing_macs2_is_reported_not_hidden(
        tmp_path, monkeypatch):
    """`tool='macs2'` used to fall through to the built-in caller when MACS2
    failed, so the caller believed its engine had run."""
    reads = _fasta(tmp_path, "", "r.bam")
    monkeypatch.setattr(pc, "check_peak_tools", lambda: {'macs2': True})
    monkeypatch.setattr(pc, "_macs2_call_peaks", lambda *a, **k: False)
    report = pc.call_peaks(reads, tool='macs2')
    assert report.engine == 'none'
    assert "MACS2 was requested but its run failed" in report.message


@pytest.mark.parametrize("tool", ["mac", "MACS2", "", "built"])
def test_call_peaks_refuses_an_unknown_tool_name(tool):
    with pytest.raises(ValueError,
                       match=rf"^Unknown tool: {re.escape(tool)}\. "
                             rf"Use 'auto', 'macs2' or 'builtin'\.$"):
        pc.call_peaks("absent.bam", tool=tool)


@pytest.mark.parametrize("tool", ["spdes", "SPADES", "", "megigit"])
def test_assemble_refuses_an_unknown_tool_name(tool):
    with pytest.raises(ValueError,
                       match=rf"^Unknown tool: {re.escape(tool)}\. Use 'auto', "
                             rf"'spades', 'megahit' or 'builtin'\.$"):
        asm.assemble("absent.fa", tool=tool)


def test_assemble_builtin_never_consults_the_external_tools(monkeypatch):
    monkeypatch.setattr(asm, "check_assembly_tools",
                        lambda: pytest.fail("the external probe must not run"))
    result = asm.assemble("absent.fa", tool='builtin')
    assert result.engine == 'none'
    assert result.message == "Reads file not found: absent.fa"
