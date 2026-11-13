"""Pins core/quantification.py: the built-in k-mer quantifier and the tool wrappers.

Salmon/Kallisto are never executed here - subprocess.run is replaced at the
module's own reference so the command line, the parsed abundance table and the
failure paths are all asserted.  The k-mer engine's vote rules and its TPM
arithmetic are checked against an independent recomputation.
"""
import os
import subprocess as _sp
import types

import numpy as np
import pandas as pd
import pytest

from biosuite.core import quantification as qf
from biosuite.core import utils as core_utils

TX_A = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"          # 40 bp
TX_B = "TTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGG"            # 40 bp
KMER_K = 8

SALMON_COLUMNS = ['Name', 'Length', 'EffectiveLength', 'TPM', 'NumReads']
KALLISTO_COLUMNS = ['target_id', 'length', 'eff_length', 'est_counts', 'tpm']


@pytest.fixture()
def no_external_tools(monkeypatch):
    """Force the availability probe to report neither Salmon nor Kallisto."""
    _tools_available(monkeypatch, False)
    return qf


def _tools_available(monkeypatch, value):
    """Say the external tools are (or are not) on PATH, for every probe site.

    ``quantification`` imports ``has_tool`` at module level, so patching only
    ``biosuite.core.utils`` would let the version probe see the fake and the index
    builders still see the real machine.
    """
    monkeypatch.setattr(core_utils, 'has_tool', lambda name: value, raising=False)
    monkeypatch.setattr(qf, 'has_tool', lambda name: value, raising=False)


@pytest.fixture()
def fastq(tmp_path):
    def _write(records, name="reads.fq"):
        p = tmp_path / name
        p.write_text("".join(f"@{rid}\n{seq}\n+\n{'I' * len(seq)}\n"
                              for rid, seq in records), encoding="utf-8")
        return str(p)
    return _write


# ── a subprocess stand-in that never launches a tool ─────────────────────────

class _ToolRunner:
    """Records the argv it is handed and lays down the tool's output file.

    Nothing is executed: every branch of both wrappers is driven from here.
    """

    def __init__(self, *, returncode=0, write_salmon=True, write_kallisto=True,
                 index_writes=True, fail_commands=(), raise_error=None,
                 salmon_rows=None, kallisto_rows=None):
        self.commands = []
        self.kwargs = []
        self.returncode = returncode
        self.write_salmon = write_salmon
        self.write_kallisto = write_kallisto
        self.index_writes = index_writes
        self.fail_commands = [tuple(c) for c in fail_commands]
        self.raise_error = raise_error
        self.salmon_rows = salmon_rows if salmon_rows is not None else {
            'Name': ['t1', 't2'], 'Length': [40, 40], 'EffectiveLength': [40, 40],
            'TPM': [30.0, 70.0], 'NumReads': [3.0, 7.0]}
        self.kallisto_rows = kallisto_rows if kallisto_rows is not None else {
            'target_id': ['t1', 't2'], 'length': [40, 40], 'eff_length': [38, 38],
            'est_counts': [1.25, 3.75], 'tpm': [25.0, 75.0]}

    @staticmethod
    def _result(returncode, stdout='', stderr=''):
        return types.SimpleNamespace(returncode=returncode, stdout=stdout,
                                     stderr=stderr)

    def run(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        self.kwargs.append(kwargs)
        if self.raise_error is not None:
            raise self.raise_error
        tool, subcommand = cmd[0], cmd[1]
        if (tool, subcommand) in self.fail_commands:
            return self._result(77, stderr='refused')
        if self.returncode != 0:
            return self._result(self.returncode, stderr='tool blew up')
        if subcommand == 'index':
            target = cmd[cmd.index('-i') + 1]
            if self.index_writes:
                if tool == 'salmon':
                    os.makedirs(target, exist_ok=True)
                else:
                    with open(target, 'w', encoding='utf-8') as fh:
                        fh.write('fake index bytes\n')
        elif subcommand == 'quant':
            out_dir = cmd[cmd.index('-o') + 1]
            if tool == 'salmon' and self.write_salmon:
                os.makedirs(out_dir, exist_ok=True)
                pd.DataFrame(self.salmon_rows).to_csv(
                    os.path.join(out_dir, 'quant.sf'), sep='\t', index=False)
            elif tool == 'kallisto' and self.write_kallisto:
                os.makedirs(out_dir, exist_ok=True)
                pd.DataFrame(self.kallisto_rows).to_csv(
                    os.path.join(out_dir, 'abundance.tsv'), sep='\t', index=False)
        return self._result(0)

    @property
    def namespace(self):
        return types.SimpleNamespace(
            run=self.run,
            TimeoutExpired=_sp.TimeoutExpired,
            SubprocessError=_sp.SubprocessError)


def _install(monkeypatch, runner):
    """Swap the subprocess reference quantification.py itself uses."""
    monkeypatch.setattr(qf, 'subprocess', runner.namespace)
    return runner


# ── QuantResult plumbing ─────────────────────────────────────────────────────

def test_quant_result_without_transcript_ids_has_no_rows():
    df = qf.QuantResult(tool='builtin', sample_name='s').to_dataframe()
    assert df.empty and list(df.columns) == [], \
        "an unrun quantification is an empty frame, not a frame of NaNs"


def test_quant_result_backfills_missing_numeric_columns():
    df = qf.QuantResult(tool='builtin', sample_name='s',
                        transcript_ids=['t1', 't2', 't3']).to_dataframe()
    assert df['tpm'].tolist() == [0, 0, 0]
    assert df['num_reads'].tolist() == [0, 0, 0]
    empty_lists = qf.QuantResult(tool='builtin', sample_name='s',
                                 transcript_ids=['t1', 't2'], tpm_values=[],
                                 num_reads_values=[]).to_dataframe()
    assert empty_lists['tpm'].tolist() == [0, 0], \
        "an empty result list is falsy and is backfilled to one zero per transcript"
    partial = qf.QuantResult(tool='builtin', sample_name='s',
                             transcript_ids=['t1', 't2'], tpm_values=[5.0, 6.0],
                             num_reads_values=[1, 2]).to_dataframe()
    assert partial.columns.tolist() == ['transcript_id', 'tpm', 'num_reads']
    assert partial['tpm'].tolist() == [5.0, 6.0]


def test_format_quant_report_omits_a_note_when_there_is_no_message():
    bare = qf.format_quant_report(qf.QuantResult(
        tool='builtin_kmer', sample_name='S7', num_transcripts=12,
        num_mapped_reads=6, mapping_rate=50.0))
    assert bare.splitlines() == ['=== BUILTIN_KMER Quantification ===',
                                 'Engine: builtin',
                                 'Sample: S7',
                                 'Transcripts: 12',
                                 'Mapped reads: 6',
                                 'Mapping rate: 50.0%']
    noted = qf.format_quant_report(qf.QuantResult(
        tool='salmon', sample_name='S1', message="Salmon not available."))
    assert noted.endswith('Note: Salmon not available.')


def test_merge_quantification_results_union_of_transcripts():
    r1 = qf.QuantResult(tool='builtin', sample_name='ctrl', transcript_ids=['a', 'b'],
                        tpm_values=[1.0, 2.0], num_reads_values=[1, 2])
    r2 = qf.QuantResult(tool='builtin', sample_name='trt', transcript_ids=['b', 'c'],
                        tpm_values=[3.0, 4.0], num_reads_values=[3, 4])
    merged = qf.merge_quantification_results([r1, r2])
    assert list(merged.columns) == ['ctrl', 'trt']
    assert merged.index.tolist() == ['a', 'b', 'c']
    assert merged['ctrl'].tolist() == [1.0, 2.0, 0.0], \
        "a transcript absent from a sample is a zero, not a hole"


def test_merge_quantification_results_edge_cases():
    assert qf.merge_quantification_results([]).empty
    only_empty = qf.QuantResult(tool='builtin', sample_name='x')
    assert qf.merge_quantification_results([only_empty]).empty, \
        "results without transcript_ids contribute no frames at all"
    mixed = qf.QuantResult(tool='builtin', sample_name='y', transcript_ids=['a'],
                           tpm_values=[9.0], num_reads_values=[1])
    merged = qf.merge_quantification_results([only_empty, mixed])
    assert merged['y'].tolist() == [9.0]


# ── availability probe ───────────────────────────────────────────────────────

@pytest.mark.parametrize('tool_on_path,returncode,expected', [
    (True, 0, True),
    (True, 1, False),
    (False, 0, False),
])
def test_check_quantification_tools_tracks_the_version_probe(
        monkeypatch, tool_on_path, returncode, expected):
    runner = _ToolRunner(returncode=returncode)
    _install(monkeypatch, runner)
    _tools_available(monkeypatch, tool_on_path)
    assert qf.check_quantification_tools() == {'salmon': expected,
                                               'kallisto': expected}
    if tool_on_path:
        assert runner.commands == [['salmon', '--version'],
                                  ['kallisto', '--version']], \
            "'salmon version' is not a salmon subcommand; only --version is"


@pytest.mark.parametrize('error', [
    FileNotFoundError("salmon: not found"),
    _sp.TimeoutExpired(cmd='salmon', timeout=10),
    OSError("cannot execute binary file"),
])
def test_check_quantification_tools_degrades_when_the_probe_raises(
        monkeypatch, error):
    _install(monkeypatch, _ToolRunner(raise_error=error))
    _tools_available(monkeypatch, True)
    assert qf.check_quantification_tools() == {'salmon': False, 'kallisto': False}


# ── k-mer index and pseudo-alignment ─────────────────────────────────────────

def test_build_transcript_index_excludes_n_kmers_and_sorts_votes():
    seq = "ACGTACGTNNACGTACGTACGT"
    index, lengths = qf._build_transcript_index([('t1', seq)], k=6)
    assert lengths == {'t1': len(seq)}
    assert not any('N' in kmer for kmer in index), \
        "an N-containing k-mer carries no sequence information"
    assert seq[0:6] in index, "the clean prefix k-mer is indexed"
    assert seq[5:11] not in index, "the k-mer straddling the N run is not"
    assert all(ids == sorted(ids) for ids in index.values())


def test_a_shared_kmer_carries_every_transcript_that_has_it():
    first = "AAAAAAAA" + "ACGTACGTACGT"
    second = "ACGTACGT" + "TTTTTTTTTTTT"
    index, _ = qf._build_transcript_index([('zeta', first), ('alpha', second)], k=8)
    assert index['ACGTACGT'] == ['alpha', 'zeta'], \
        "vote lists are sorted by name, not in transcript-file order"


def test_repeated_kmer_inside_one_transcript_votes_once():
    index, _ = qf._build_transcript_index([('dup', TX_A + TX_A)], k=8)
    assert index['ACGTACGT'] == ['dup'], "a set, not a list: repeats do not multiply"


def test_pseudo_align_read_requires_min_hits_or_returns_nothing():
    index, _ = qf._build_transcript_index([('t1', TX_A)], k=8)
    assert qf._pseudo_align_read("TTTTTTTTTTTTTTTT", index, k=8, min_hits=2) == (None, 0)
    one_hit = "ACGTACGT" + "TTTT" * 8
    assert qf._pseudo_align_read(one_hit, index, k=8, min_hits=2) == (None, 0), \
        "a single shared 8-mer is not evidence of a transcript"
    best, hits = qf._pseudo_align_read(TX_A[:24], index, k=8, min_hits=2)
    assert (best, hits) == ('t1', 17), "40 - 8 + 1 k-mers for a 24 bp read is 17 votes"


def test_pseudo_align_read_picks_the_majority_transcript():
    index, _ = qf._build_transcript_index([('a', TX_A), ('b', TX_B)], k=8)
    mixed = TX_A[:16] + TX_B[:24]
    best, hits = qf._pseudo_align_read(mixed, index, k=8, min_hits=2)
    assert best == 'b' and hits == 17, "the longer run of matching k-mers wins"


# ── built-in quantifier ──────────────────────────────────────────────────────

def test_builtin_quantify_maps_reads_and_normalises_to_a_million(fastq):
    fq = fastq([('r1', TX_A[:24]), ('r2', TX_A[:20]), ('r3', TX_B[:24])])
    res = qf._builtin_quantify(fq, [('a', TX_A), ('b', TX_B)], k=KMER_K,
                               sample_name='S1')
    assert res.num_transcripts == 2 and res.engine == 'builtin'
    assert res.num_reads_values == [2, 1] and res.num_mapped_reads == 3
    assert res.mapping_rate == pytest.approx(100.0)
    counts = np.array([2.0, 1.0])
    lengths_kb = np.array([len(TX_A), len(TX_B)], dtype=float) / 1000.0
    rpk = counts / lengths_kb
    expected_tpm = rpk / (rpk.sum() / 1e6)
    assert res.tpm_values == pytest.approx(expected_tpm.tolist()), \
        "TPM must be count/lengthkb divided by the per-sample sum of that ratio"
    assert sum(res.tpm_values) == pytest.approx(1e6)
    assert res.transcript_ids == ['a', 'b'] and res.num_transcripts == 2
    assert res.message == "Using built-in k-mer quantifier (3/3 reads mapped)"


def test_builtin_quantify_skips_a_blank_record_and_keeps_reading(tmp_path):
    """Regression: an empty sequence line used to end the read loop."""
    p = tmp_path / "blank_record.fq"
    p.write_text("@r1\n" + TX_A[:24] + "\n+\n" + "I" * 24 + "\n"
                 "@blank\n\n+\n\n"
                 "@r2\n" + TX_B[:24] + "\n+\n" + "I" * 24 + "\n", encoding="utf-8")
    res = qf._builtin_quantify(str(p), [('a', TX_A), ('b', TX_B)], k=KMER_K)
    assert res.num_mapped_reads == 2, "the read after the blank record must still count"
    assert res.num_reads_values == [1, 1]
    assert res.message == "Using built-in k-mer quantifier (2/2 reads mapped)", \
        "total_reads must not stop at the malformed record either"


def test_builtin_quantify_ignores_a_truncated_final_record(tmp_path):
    p = tmp_path / "truncated.fq"
    p.write_text("@r1\n" + TX_A[:24] + "\n+\n" + "I" * 24 + "\n@r2\n", encoding="utf-8")
    res = qf._builtin_quantify(str(p), [('a', TX_A)], k=KMER_K)
    assert res.num_reads_values == [1] and res.mapping_rate == pytest.approx(100.0), \
        "a header with no sequence is not a read"


def test_builtin_quantify_clips_k_for_reads_shorter_than_k(fastq):
    fq = fastq([('long', TX_A[:24]), ('short', TX_A[:10]), ('tiny', 'ACGT')])
    res = qf._builtin_quantify(fq, [('a', TX_A), ('b', TX_B)], k=16)
    assert res.num_reads_values == [2, 0], \
        "a 10 bp read is quantified with k clipped to 10, a 4 bp read is dropped"
    assert res.num_mapped_reads == 2
    assert res.mapping_rate == pytest.approx(200 / 3), \
        "reads under 8 bases are counted as total but can never be mapped"


def test_builtin_quantify_unmatched_reads_lower_the_mapping_rate(fastq):
    fq = fastq([('r1', TX_A[:24]), ('r2', 'C' * 24)])
    res = qf._builtin_quantify(fq, [('a', TX_A)], k=KMER_K)
    assert res.num_mapped_reads == 1 and res.num_reads_values == [1]
    assert res.mapping_rate == pytest.approx(50.0)


def test_builtin_quantify_on_an_empty_file_reports_zero_rate(fastq):
    fq = fastq([])
    res = qf._builtin_quantify(fq, [('a', TX_A)], k=KMER_K)
    assert res.num_mapped_reads == 0 and res.mapping_rate == 0, \
        "0/0 reads is reported as 0% rather than a division by zero"
    assert res.tpm_values == [0.0]


def test_builtin_quantify_warns_about_transcripts_shorter_than_k(fastq):
    fq = fastq([('r1', TX_A[:24])])
    with pytest.warns(UserWarning, match="shorter than k=31"):
        res = qf._builtin_quantify(
            fq, [('short', 'ACGTACGT'), ('a', TX_A)], k=31)
    assert res.num_transcripts == 2
    assert res.tpm_values[0] == 0.0, "a transcript with no indexable k-mer is TPM 0"


def test_builtin_quantify_zero_length_transcript_does_not_divide_by_zero(fastq):
    fq = fastq([('r1', TX_A[:24])])
    with pytest.warns(UserWarning):
        res = qf._builtin_quantify(fq, [('empty', ''), ('a', TX_A)], k=KMER_K)
    assert res.num_reads_values == [0, 1]
    assert res.tpm_values[0] == 0.0 and res.tpm_values[1] == pytest.approx(1e6)


# ── quantify_reads public entry ──────────────────────────────────────────────

@pytest.fixture()
def transcriptome(tmp_path):
    p = tmp_path / "tx.fa"
    p.write_text(f">a\n{TX_A}\n>b\n{TX_B}\n", encoding="utf-8")
    return str(p)


def test_quantify_reads_missing_inputs(tmp_path, transcriptome, no_external_tools,
                                       fastq):
    fq = fastq([('r1', TX_A[:24])])
    gone = qf.quantify_reads(str(tmp_path / "ghost.fq"), transcriptome)
    assert gone.message.startswith("Reads file not found:")
    assert gone.num_transcripts == 0
    bad_tx = qf.quantify_reads(fq, str(tmp_path / "ghost.fa"))
    assert bad_tx.message.startswith("Transcriptome not found:")
    empty_tx = tmp_path / "empty.fa"
    empty_tx.write_text("there is no record here\n", encoding="utf-8")
    no_records = qf.quantify_reads(fq, str(empty_tx))
    assert no_records.message == "No transcripts found in FASTA file."


def test_quantify_reads_falls_back_to_the_builtin_engine(
        tmp_path, transcriptome, fastq, monkeypatch):
    _tools_available(monkeypatch, True)
    _install(monkeypatch, _ToolRunner(returncode=1))
    fq = fastq([('r1', TX_A[:24]), ('r2', TX_B[:24])])
    res = qf.quantify_reads(fq, transcriptome, sample_name='S9', k=KMER_K)
    assert res.engine == 'builtin' and res.sample_name == 'S9'
    assert res.num_reads_values == [1, 1], \
        "a failing external tool must not lose the quantification"


def test_quantify_reads_uses_salmon_when_it_and_its_index_work(
        tmp_path, transcriptome, fastq, monkeypatch):
    _tools_available(monkeypatch, True)
    runner = _install(monkeypatch, _ToolRunner())
    fq = fastq([('r1', TX_A[:24])])
    res = qf.quantify_reads(fq, transcriptome, k=KMER_K)
    assert res.engine == 'salmon' and res.num_transcripts == 2
    assert res.tpm_values == [30.0, 70.0]
    assert res.num_reads_values == [3.0, 7.0]
    quant_cmds = [c for c in runner.commands if c[:2] == ['salmon', 'quant']]
    assert len(quant_cmds) == 1
    assert quant_cmds[0][quant_cmds[0].index('-o') + 1].endswith('sample'), \
        "salmon writes into <output>/<sample_name>/"
    assert ['kallisto', 'quant'] not in [c[:2] for c in runner.commands], \
        "salmon succeeding must short-circuit kallisto"


def test_quantify_reads_uses_kallisto_when_salmon_cannot_be_indexed(
        tmp_path, transcriptome, fastq, monkeypatch):
    _tools_available(monkeypatch, True)
    runner = _install(monkeypatch, _ToolRunner(fail_commands=[('salmon', 'index')]))
    fq = fastq([('r1', TX_A[:24])])
    res = qf.quantify_reads(fq, transcriptome, sample_name='K', k=KMER_K)
    assert res.engine == 'kallisto' and res.tool == 'kallisto'
    assert res.transcript_ids == ['t1', 't2']
    assert res.num_reads_values == [1.25, 3.75], \
        "abundance.tsv carries est_counts, not est_count"
    assert res.tpm_values == [25.0, 75.0]
    assert res.sample_name == 'K'
    assert [c[:2] for c in runner.commands] == [['salmon', '--version'],
                                                ['kallisto', '--version'],
                                                ['salmon', 'index'],
                                                ['kallisto', 'index'],
                                                ['kallisto', 'quant']], \
        "a failed salmon index must not stop the kallisto attempt"


# ── external tool wrappers (marked: subprocess is faked) ─────────────────────

def test_salmon_quant_builds_a_paired_end_command_and_parses_quant_sf(
        tmp_path, monkeypatch):
    out = tmp_path / "sal"
    runner = _install(monkeypatch, _ToolRunner())
    res = qf._salmon_quant('r1.fq', 'r2.fq', 'idx_dir', str(out), 'S1', 8)
    cmd = runner.commands[-1]
    assert cmd[:2] == ['salmon', 'quant']
    assert cmd[cmd.index('-i') + 1] == 'idx_dir'
    assert cmd[cmd.index('-l') + 1] == 'A'
    assert cmd[cmd.index('--threads') + 1] == '8'
    assert cmd[-2:] == ['-r', 'r2.fq'] and '--paired' in cmd
    assert str(out / 'S1') in cmd[cmd.index('-o') + 1]
    assert runner.kwargs[-1]['timeout'] == 7200
    assert res.engine == 'salmon' and res.num_transcripts == 2
    assert res.tpm_values == [30.0, 70.0]
    assert res.num_reads_values == [3.0, 7.0]
    assert res.transcript_ids == ['t1', 't2']
    assert res.abundance_file.endswith('quant.sf')


def test_salmon_quant_single_end_command_has_no_paired_flag(tmp_path, monkeypatch):
    _install(monkeypatch, _ToolRunner())
    res = qf._salmon_quant('r1.fq', None, 'idx_dir', str(tmp_path / 's'), 'S2', 1)
    assert res.engine == 'salmon'


@pytest.mark.parametrize('returncode,writes_table', [(1, True), (0, False)])
def test_salmon_quant_returns_none_without_a_usable_table(
        tmp_path, monkeypatch, returncode, writes_table, caplog):
    _install(monkeypatch, _ToolRunner(returncode=returncode,
                                      write_salmon=writes_table))
    assert qf._salmon_quant('r1.fq', None, 'idx', str(tmp_path / 'x'), 'S', 1) is None


def test_salmon_quant_reports_a_broken_abundance_table(tmp_path, monkeypatch, caplog):
    _install(monkeypatch, _ToolRunner(
        salmon_rows={'Name': ['t1'], 'Length': [40], 'EffectiveLength': [40]}))
    with caplog.at_level('ERROR', logger='biosuite.core.quantification'):
        assert qf._salmon_quant('r1.fq', None, 'idx', str(tmp_path / 'x'), 'S', 1) is None
    assert 'TPM' in caplog.text, "a missing column is logged, not silently 'unavailable'"


def test_kallisto_quant_builds_a_single_end_command_and_parses_abundance(
        tmp_path, monkeypatch):
    runner = _install(monkeypatch, _ToolRunner())
    res = qf._kallisto_quant('r1.fq', None, 'k.idx', str(tmp_path / 'k'), 'K1', 2)
    cmd = runner.commands[-1]
    assert cmd[:2] == ['kallisto', 'quant']
    assert cmd[cmd.index('-i') + 1] == 'k.idx'
    assert cmd[cmd.index('-b') + 1] == '100'
    assert cmd[cmd.index('--threads') + 1] == '2'
    assert cmd[-6:-1] == ['--single', '-l', '200', '-s', '20']
    assert cmd[-1] == 'r1.fq'
    assert res.engine == 'kallisto' and res.num_transcripts == 2
    assert res.tpm_values == [25.0, 75.0]
    assert res.num_reads_values == [1.25, 3.75]
    assert res.transcript_ids == ['t1', 't2']
    assert res.abundance_file.endswith('abundance.tsv')


def test_kallisto_quant_paired_command_appends_both_mates(tmp_path, monkeypatch):
    runner = _install(monkeypatch, _ToolRunner())
    qf._kallisto_quant('m1.fq', 'm2.fq', 'k.idx', str(tmp_path / 'k2'), 'K2', 4)
    assert runner.commands[-1][-2:] == ['m1.fq', 'm2.fq']


@pytest.mark.parametrize('returncode,writes_table', [(1, True), (0, False)])
def test_kallisto_quant_returns_none_without_a_usable_table(
        tmp_path, monkeypatch, returncode, writes_table):
    _install(monkeypatch, _ToolRunner(returncode=returncode,
                                      write_kallisto=writes_table))
    assert qf._kallisto_quant('r1.fq', None, 'i', str(tmp_path / 'x'), 'K', 1) is None


def test_external_wrappers_report_a_failing_process(tmp_path, monkeypatch):
    _install(monkeypatch, _ToolRunner(raise_error=OSError("salmon: cannot exec")))
    assert qf._salmon_quant('r1.fq', None, 'i', str(tmp_path / 'a'), 'S', 1) is None
    assert qf._kallisto_quant('r1.fq', None, 'i', str(tmp_path / 'b'), 'K', 1) is None


# ── index builders ───────────────────────────────────────────────────────────

def test_build_salmon_index_creates_the_directory_and_returns_it(
        tmp_path, monkeypatch):
    _tools_available(monkeypatch, True)
    runner = _install(monkeypatch, _ToolRunner())
    target = tmp_path / "salmon_index"
    got = qf.build_salmon_index(str(tmp_path / "tx.fa"), str(target), threads=4)
    assert got == str(target) and target.is_dir()
    cmd = runner.commands[-1]
    assert cmd[:2] == ['salmon', 'index']
    assert cmd[cmd.index('-t') + 1] == str(tmp_path / "tx.fa")
    assert cmd[cmd.index('-k') + 1] == '31'
    assert cmd[cmd.index('--threads') + 1] == '4'


def test_build_salmon_index_failure_paths(tmp_path, monkeypatch):
    # The tools are reported installed so that these None results can only come
    # from the failing subprocess, not from the availability probe.
    _tools_available(monkeypatch, True)
    _install(monkeypatch, _ToolRunner(returncode=1))
    assert qf.build_salmon_index('tx.fa', str(tmp_path / 'i1')) is None
    _install(monkeypatch, _ToolRunner(raise_error=_sp.TimeoutExpired('salmon', 3600)))
    assert qf.build_salmon_index('tx.fa', str(tmp_path / 'i2')) is None, \
        "a timed-out index build must not be handed to a quantifier"


def test_kallisto_index_returns_the_file_only_on_success(tmp_path, monkeypatch):
    _tools_available(monkeypatch, True)
    runner = _install(monkeypatch, _ToolRunner(write_kallisto=False))
    idx = tmp_path / "tx.idx"
    got = qf.kallisto_index(str(tmp_path / "tx.fa"), str(idx), threads=3)
    assert got == str(idx)
    cmd = runner.commands[-1]
    assert cmd[:2] == ['kallisto', 'index']
    assert cmd[cmd.index('-i') + 1] == str(idx)
    assert cmd[cmd.index('-k') + 1] == '31'
    assert cmd[-1] == str(tmp_path / "tx.fa")
    assert cmd[-2] == '3'

    _install(monkeypatch, _ToolRunner(returncode=2, write_kallisto=False))
    assert qf.kallisto_index('tx.fa', str(tmp_path / 'other.idx')) is None
    _install(monkeypatch, _ToolRunner(raise_error=OSError("no kallisto"),
                                      write_kallisto=False))
    assert qf.kallisto_index('tx.fa', str(tmp_path / 'third.idx')) is None


# ── public salmon_quant / kallisto_quant ─────────────────────────────────────

def test_builtin_quantify_reuses_the_index_built_for_a_clipped_k(fastq):
    """Two short reads share one lazily-built index at the clipped k."""
    fq = fastq([('r1', TX_A[:10]), ('r2', TX_A[:10]), ('r3', TX_B[:10])])
    res = qf._builtin_quantify(fq, [('a', TX_A), ('b', TX_B)], k=16)
    assert res.num_reads_values == [2, 1], \
        "the second 10 bp read must be quantified, not silently dropped"
    assert res.mapping_rate == pytest.approx(100.0)


def test_quantify_reads_falls_back_when_kallisto_cannot_be_indexed(
        tmp_path, transcriptome, fastq, monkeypatch):
    _tools_available(monkeypatch, True)
    runner = _install(monkeypatch, _ToolRunner(
        fail_commands=[('salmon', 'index'), ('kallisto', 'index')]))
    fq = fastq([('r1', TX_A[:24])])
    res = qf.quantify_reads(fq, transcriptome, k=KMER_K)
    assert res.engine == 'builtin' and res.num_reads_values == [1, 0]
    assert [c[:2] for c in runner.commands] == [['salmon', '--version'],
                                                ['kallisto', '--version'],
                                                ['salmon', 'index'],
                                                ['kallisto', 'index']], \
        "two refused index builds and no quantification attempt in between"


def test_quantify_reads_falls_back_when_both_tools_quantify_nothing(
        tmp_path, transcriptome, fastq, monkeypatch):
    _tools_available(monkeypatch, True)
    runner = _install(monkeypatch, _ToolRunner(write_salmon=False,
                                               write_kallisto=False))
    fq = fastq([('r1', TX_A[:24]), ('r2', TX_B[:24])])
    res = qf.quantify_reads(fq, transcriptome, k=KMER_K)
    assert res.engine == 'builtin' and res.num_mapped_reads == 2, \
        "a tool that exits 0 without a table is a failed run, not a zero-abundance one"
    assert [c[:2] for c in runner.commands] == [['salmon', '--version'],
                                                ['kallisto', '--version'],
                                                ['salmon', 'index'],
                                                ['salmon', 'quant'],
                                                ['kallisto', 'index'],
                                                ['kallisto', 'quant']]


def test_salmon_quant_builds_an_index_when_only_a_transcriptome_is_given(
        tmp_path, monkeypatch):
    _tools_available(monkeypatch, True)
    runner = _install(monkeypatch, _ToolRunner())
    res = qf.salmon_quant('reads.fq', transcriptome_fasta=str(tmp_path / 'tx.fa'),
                          sample_name='auto', threads=2)
    assert res.engine == 'salmon' and res.sample_name == 'auto'
    index_cmds = [c for c in runner.commands if c[:2] == ['salmon', 'index']]
    assert len(index_cmds) == 1, "a missing index dir is built, not assumed"


def test_salmon_quant_without_any_index_is_refused(monkeypatch):
    """The old answer to a missing argument was the same "Salmon not available"
    message an uninstalled tool produces, which sent the caller hunting for a PATH
    problem they did not have."""
    _install(monkeypatch, _ToolRunner())
    with pytest.raises(ValueError, match=(
            r"^salmon_quant needs a Salmon index: pass index_dir=\.\.\., or "
            r"transcriptome_fasta=\.\.\. so one can be built\.$")):
        qf.salmon_quant('reads.fq', transcriptome_fasta=None, index_dir=None)


def test_build_salmon_index_creates_nothing_when_the_tools_are_absent(tmp_path,
                                                                      monkeypatch):
    """Three builds on a machine without salmon left three empty `salmon_idx_*`
    directories and three 0-byte `.idx` files behind; the reservation now happens
    after the availability probe."""
    import glob as _glob
    import tempfile as _tempfile
    _tools_available(monkeypatch, False)
    _install(monkeypatch, _ToolRunner())
    before = set(_glob.glob(os.path.join(_tempfile.gettempdir(), "*")))
    assert qf.build_salmon_index(str(tmp_path / "tx.fa")) is None
    assert qf.kallisto_index(str(tmp_path / "tx.fa")) is None
    leaked = [p for p in _glob.glob(os.path.join(_tempfile.gettempdir(), "*"))
              if p not in before]
    assert leaked == [], f"failed index builds left {leaked} behind"


def test_salmon_quant_falls_back_when_the_tool_produces_no_table(
        tmp_path, monkeypatch):
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    _install(monkeypatch, _ToolRunner(write_salmon=False))
    res = qf.salmon_quant('reads.fq', index_dir=str(index_dir),
                          output_dir=str(tmp_path / "out"), sample_name='S')
    assert "Salmon not available" in res.message, \
        "an existing index dir is not proof that salmon can quantify"


def test_kallisto_quant_needs_a_non_empty_index(tmp_path, monkeypatch):
    _install(monkeypatch, _ToolRunner())
    empty_idx = tmp_path / "zero.idx"
    empty_idx.write_bytes(b"")
    res = qf.kallisto_quant('reads.fq', index_file=str(empty_idx))
    assert "Kallisto not available" in res.message, \
        "a 0-byte index is a failed build, not a usable one"


def test_quantify_reads_skips_salmon_when_it_indexes_but_quantifies_nothing(
        tmp_path, transcriptome, fastq, monkeypatch):
    _tools_available(monkeypatch, True)
    runner = _install(monkeypatch, _ToolRunner(write_salmon=False))
    fq = fastq([('r1', TX_A[:24])])
    res = qf.quantify_reads(fq, transcriptome, k=KMER_K)
    assert res.engine == 'kallisto', \
        "an empty salmon run must not be reported as a quantification"
    assert [c[:2] for c in runner.commands] == [['salmon', '--version'],
                                                 ['kallisto', '--version'],
                                                 ['salmon', 'index'],
                                                 ['salmon', 'quant'],
                                                 ['kallisto', 'index'],
                                                 ['kallisto', 'quant']]


def test_kallisto_quant_builds_an_index_and_quantifies(tmp_path, monkeypatch):
    _tools_available(monkeypatch, True)
    tx = tmp_path / "tx.fa"
    tx.write_text(f">a\n{TX_A}\n", encoding="utf-8")
    runner = _install(monkeypatch, _ToolRunner(
        kallisto_rows={'target_id': ['a'], 'length': [40], 'eff_length': [40],
                       'est_counts': [2.0], 'tpm': [1e6]}))
    res = qf.kallisto_quant('reads.fq', transcriptome_fasta=str(tx),
                            output_dir=str(tmp_path / 'o'), sample_name='K9')
    assert res.engine == 'kallisto' and res.num_transcripts == 1
    assert res.tpm_values == [1e6] and res.num_reads_values == [2.0]
    assert res.abundance_file.endswith('abundance.tsv')
    built = [c for c in runner.commands if c[:2] == ['kallisto', 'index']]
    assert len(built) == 1, "no index file was given, so one must be built first"


def test_kallisto_quant_without_any_index_is_refused(monkeypatch):
    _install(monkeypatch, _ToolRunner())
    with pytest.raises(ValueError, match=(
            r"^kallisto_quant needs a Kallisto index: pass index_file=\.\.\., or "
            r"transcriptome_fasta=\.\.\. so one can be built\.$")):
        qf.kallisto_quant('reads.fq', index_file=None, transcriptome_fasta=None)


def test_kallisto_quant_builds_nothing_when_the_index_build_fails(
        tmp_path, monkeypatch):
    _install(monkeypatch, _ToolRunner(fail_commands=[('kallisto', 'index')]))
    tx = tmp_path / "tx.fa"
    tx.write_text(f">a\n{TX_A}\n", encoding="utf-8")
    res = qf.kallisto_quant('reads.fq', transcriptome_fasta=str(tx),
                            output_dir=str(tmp_path / 'o'))
    assert "Kallisto not available" in res.message, \
        "a refused index build must not be quantified against"


def test_kallisto_quant_falls_back_when_quantification_produces_nothing(
        tmp_path, monkeypatch):
    idx = tmp_path / "tx.idx"
    idx.write_bytes(b"not really an index")
    _install(monkeypatch, _ToolRunner(write_kallisto=False))
    res = qf.kallisto_quant('reads.fq', index_file=str(idx), sample_name='K0')
    assert res.tool == 'kallisto' and "Kallisto not available" in res.message
