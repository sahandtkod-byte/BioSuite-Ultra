"""Engine selection and output parsing of ``core.blast``: BLAST+ and built-in.

BLAST+ is an external binary and the built-in engine is a seed-and-extend
approximation, so both are exercised here without spawning a process:
``subprocess.run`` is replaced by a fake that writes a canned BLAST+ XML report,
and the tests assert the command line that was built, how every HSP was parsed,
and what a non-zero exit, a timeout, an unreadable report or a missing binary
turns into.  Also pinned: the built-in engine works without Biopython installed
and never crashes on a seed it cannot extend.
"""
import contextlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

from biosuite.core import blast as bl
from biosuite.core import utils
from biosuite.core.utils import PerformanceWarning

CORE = 'ACGTACGTACGTACGT'                     # 16 bases, the built-in engine's seed
QUERY_SEQ = 'ACGT' * 10                       # 40 bases, used by the BLAST+ report


# ── canned BLAST+ XML report ────────────────────────────────────────────────

SUBJECT_A = 'ACGT' * 9 + 'ACGA'              # 38 of 40 columns identical
SUBJECT_B = 'ACGA' + 'ACGT' * 9              # 37 identities, 1 gap column
SHORT_SEQ = QUERY_SEQ[4:]                    # 36 bases, second hit


def _hsp(num, bits, score, evalue, q_from, q_to, h_from, h_to, identities,
         align_length, gaps, qseq, hseq):
    midline = ''.join('|' if a == b else ' ' for a, b in zip(qseq, hseq))
    return (
        f"<Hsp><Hsp_num>{num}</Hsp_num><Hsp_bit-score>{bits}</Hsp_bit-score>"
        f"<Hsp_score>{score}</Hsp_score><Hsp_evalue>{evalue}</Hsp_evalue>"
        f"<Hsp_query-from>{q_from}</Hsp_query-from><Hsp_query-to>{q_to}</Hsp_query-to>"
        f"<Hsp_hit-from>{h_from}</Hsp_hit-from><Hsp_hit-to>{h_to}</Hsp_hit-to>"
        f"<Hsp_query-frame>1</Hsp_query-frame><Hsp_identity>{identities}</Hsp_identity>"
        f"<Hsp_positive>{identities}</Hsp_positive><Hsp_gaps>{gaps}</Hsp_gaps>"
        f"<Hsp_align-len>{align_length}</Hsp_align-len>"
        f"<Hsp_qseq>{qseq}</Hsp_qseq><Hsp_hseq>{hseq}</Hsp_hseq>"
        f"<Hsp_midline>{midline}</Hsp_midline></Hsp>")


def _hit(num, hit_id, accession, definition, hsps):
    return (f"<Hit><Hit_num>{num}</Hit_num><Hit_id>{hit_id}</Hit_id>"
            f"<Hit_def>{definition}</Hit_def><Hit_accession>{accession}</Hit_accession>"
            f"<Hit_len>1000</Hit_len><Hit_hsps>{''.join(hsps)}</Hit_hsps></Hit>")


BLAST_XML = (
    '<?xml version="1.0"?>'
    '<!DOCTYPE BlastOutput PUBLIC "-//NCBI//NCBI BlastOutput/EN" '
    '"http://www.ncbi.nlm.nih.gov/dtd/NCBI_BlastOutput.dtd">'
    '<BlastOutput><BlastOutput_program>blastn</BlastOutput_program>'
    '<BlastOutput_version>BLASTN 2.14.0+</BlastOutput_version>'
    '<BlastOutput_reference>Reference</BlastOutput_reference>'
    '<BlastOutput_db>nt</BlastOutput_db>'
    '<BlastOutput_query-ID>Query_1</BlastOutput_query-ID>'
    '<BlastOutput_query-def>myquery</BlastOutput_query-def>'
    '<BlastOutput_query-len>40</BlastOutput_query-len>'
    f'<BlastOutput_query-seq>{QUERY_SEQ}</BlastOutput_query-seq>'
    '<BlastOutput_matrix>0</BlastOutput_matrix>'
    '<Parameters><Parameters_expect>1e-05</Parameters_expect>'
    '<Parameters_sc-match>1</Parameters_sc-match>'
    '<Parameters_sc-mismatch>-2</Parameters_sc-mismatch>'
    '<Parameters_gap-open>0</Parameters_gap-open>'
    '<Parameters_gap-extend>0</Parameters_gap-extend>'
    '<Parameters_filter>F</Parameters_filter></Parameters>'
    '<BlastOutput_iterations><Iteration>'
    '<Iteration_iter-num>1</Iteration_iter-num>'
    '<Iteration_query-ID>Query_1</Iteration_query-ID>'
    '<Iteration_query-def>myquery</Iteration_query-def>'
    '<Iteration_query-len>40</Iteration_query-len><Iteration_hits>'
    + _hit(1, 'gn|NC_000001.1', 'NC_000001.1', 'Escherichia coli plasmid', [
        _hsp(1, 74.3, 40, '1e-10', 1, 40, 11, 50, 38, 40, 0, QUERY_SEQ, SUBJECT_A),
        _hsp(2, 60.0, 32, '2e-6', 1, 40, 101, 140, 37, 40, 1, QUERY_SEQ, SUBJECT_B)])
    + _hit(2, 'gn|NR_000002.1', '', 'Mitochondrial reference', [
        _hsp(1, 45.0, 24, '3e-2', 5, 40, 3, 38, 34, 36, 0, SHORT_SEQ, SUBJECT_A[:36])])
    + '</Iteration_hits><Iteration_msg></Iteration_msg>'
    '<Iteration_stat><Statistics><Statistics_db-num>2</Statistics_db-num>'
    '<Statistics_seq-num>2</Statistics_seq-num>'
    '<Statistics_db_positions>2000</Statistics_db_positions>'
    '<Statistics_db_sequences>2</Statistics_db_sequences>'
    '<Statistics_max_scores>1</Statistics_max_scores>'
    '<Statistics_query_frames>1</Statistics_query_frames>'
    '<Statistics_matrix-num>1</Statistics_matrix-num></Statistics></Iteration_stat>'
    '</Iteration></BlastOutput_iterations></BlastOutput>')


def staging_dirs():
    """The `blastdb_*` directories currently in the system temp folder."""
    return {name for name in os.listdir(tempfile.gettempdir())
            if name.startswith('blastdb_')}


class FakeBlastPlus:
    """Fake ``subprocess.run`` for the BLAST+ binaries.

    A version probe (``-version`` as the binaries take it, ``--version`` as
    :func:`biosuite.core.utils.has_tool` takes it) answers like an installed tool; a
    real search writes the canned report to the ``-out`` path the code chose, exactly
    as blastn would.
    """

    def __init__(self, report=BLAST_XML, returncode=0, stderr=''):
        self.report = report
        self.returncode = returncode
        self.stderr = stderr
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        if cmd[-1] in ('-version', '--version'):
            # A failed probe must look failed: the version branch answers with the
            # returncode this fake was built with, not a hardcoded success.
            return subprocess.CompletedProcess(
                cmd, self.returncode, 'blastn: 2.14.0+\n', self.stderr)
        if self.returncode == 0 and self.report is not None:
            with open(cmd[cmd.index('-out') + 1], 'w', encoding='utf-8') as handle:
                handle.write(self.report)
        return subprocess.CompletedProcess(cmd, self.returncode, '', self.stderr)


@pytest.fixture
def query_file(tmp_path):
    path = tmp_path / 'query.fasta'
    path.write_text(f'>q1\n{CORE}\n', encoding='utf-8')
    return str(path)


@pytest.fixture
def subject_file(tmp_path):
    """Three subjects sharing one seed, so the built-in engine has choices to make."""
    path = tmp_path / 'subject.fasta'
    path.write_text(f'>subjA\n{CORE}\n>subjB\n{CORE}\n>subjC\n{CORE}AAAA\n',
                    encoding='utf-8')
    return str(path)


def _hit_of(subject, evalue, identity=90.0, score=40.0):
    return bl.BlastHit(query_id='q1', subject_id=subject, subject_description=subject,
                       percent_identity=identity, alignment_length=10, mismatches=1,
                       gap_opens=0, query_start=1, query_end=10,
                       subject_start=1, subject_end=10, e_value=evalue,
                       bit_score=score)


# ── result object contract ───────────────────────────────────────────────────

def test_significant_hits_keeps_the_boundary_it_documents():
    """`evalue_threshold` is documented as the "maximum acceptable E-value", so a hit
    sitting exactly on it is acceptable.  The code compared with a strict `<` — and the
    pin said so in its own name — which threw away `e_value == 1e-5`, the single most
    common value a BLAST report carries."""
    result = bl.BlastResult(program='blastn', database='nt', query_length=10,
                            hits=[_hit_of('s1', 1e-9), _hit_of('s2', 1e-5),
                                  _hit_of('s3', 0.5)])

    assert [h.subject_id for h in result.significant_hits()] == ['s1', 's2']
    assert [h.subject_id for h in result.significant_hits(1e-4)] == ['s1', 's2']
    assert [h.subject_id for h in result.significant_hits(1.0)] == ['s1', 's2', 's3']
    assert [h.subject_id for h in result.significant_hits(1e-6)] == ['s1']
    assert result.num_hits == 3
    assert [h.subject_id for h in result.top_hits(2)] == ['s1', 's2']
    assert result.top_hits(0) == []


def test_hit_one_line_report_shows_identity_evalue_and_score():
    assert str(_hit_of('NC_000001.1', 1e-10)) == (
        'NC_000001.1 | 90.0% identity | E=1.00e-10 | Score=40')


# ── BLAST+ detection ─────────────────────────────────────────────────────────

@pytest.fixture
def on_path(monkeypatch):
    """`shutil.which` answers for a given set of names, and `subprocess.run` is `fake`."""
    def apply(names=('blastn', 'blastp', 'blastx', 'tblastn', 'tblastx', 'makeblastdb'),
              fake=None):
        present = set(names)
        monkeypatch.setattr(utils.shutil, 'which',
                            lambda name: f'/opt/blast/bin/{name}' if name in present else None)
        monkeypatch.setattr(bl.subprocess, 'run',
                            fake if fake is not None else FakeBlastPlus())
    return apply


def test_presence_comes_from_path_and_a_nonzero_version_exit_is_still_installed(on_path):
    """`blastn -version` exiting non-zero used to answer "not installed", so the
    package fell back to the approximate engine beside a working binary.  `has_tool`
    separates the two questions: `shutil.which` says whether the program is there,
    the probe only says whether it started."""
    class UsageExiter:
        """A BLAST+ build that prints its usage line and exits 2 for a version flag."""

        def __init__(self):
            self.calls = []

        def __call__(self, cmd, **kwargs):
            self.calls.append((list(cmd), kwargs))
            return subprocess.CompletedProcess(cmd, 2, 'Usage: blastn [options]', '')

    fake = UsageExiter()
    on_path(fake=fake)
    assert bl._has_blast_plus() is True
    assert fake.calls == [(['blastn', '--version'],
                           {'capture_output': True, 'text': True, 'timeout': 10})]


def test_a_name_that_is_nowhere_on_path_costs_no_subprocess(on_path):
    """The old probe started one process per tool even on a machine with no BLAST+ at
    all: six spawns to learn nothing, twice per search."""
    seen = []
    on_path(names=(), fake=lambda cmd, **kwargs: seen.append(cmd))
    assert bl._has_blast_plus() is False
    assert bl.check_blast_installed() == {tool: False for tool in bl.BLAST_TOOLS}
    assert seen == [], "no binary, no subprocess"


@pytest.mark.parametrize('failure', [
    FileNotFoundError('blastn'),
    PermissionError(13, 'Permission denied'),
    subprocess.TimeoutExpired(cmd='blastn', timeout=10),
])
def test_a_binary_that_cannot_be_started_is_reported_absent_not_raised(failure, on_path):
    """`PermissionError` — a `blastn` on `PATH` without the execute bit — escaped the
    old probe and took `run_blast` down with it: the CLI option showed a traceback
    instead of falling back."""
    def raiser(cmd, **kwargs):
        raise failure

    on_path(fake=raiser)
    assert bl._has_blast_plus() is False
    assert bl.check_blast_installed() == {tool: False for tool in bl.BLAST_TOOLS}


def test_check_blast_installed_reports_each_tool_separately(on_path):
    answers = {'blastn': 0, 'blastp': 1}

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, answers.get(cmd[0], 0), '', '')

    on_path(fake=fake_run)
    assert bl.check_blast_installed() == {
        'blastn': True, 'blastp': True, 'blastx': True,
        'tblastn': True, 'tblastx': True, 'makeblastdb': True,
    }, "a non-zero version exit is still an installed tool"


def test_check_blast_installed_marks_a_missing_tool_false(on_path):
    """One name absent from `PATH` while the rest are present."""
    on_path(names=('blastn', 'blastp', 'blastx', 'tblastn', 'tblastx'))
    assert bl.check_blast_installed() == {
        'blastn': True, 'blastp': True, 'blastx': True,
        'tblastn': True, 'tblastx': True, 'makeblastdb': False,
    }
    assert bl.BLAST_TOOLS == ('blastn', 'blastp', 'blastx', 'tblastn', 'tblastx',
                              'makeblastdb')


# ── BLAST+ search: command line and report parsing ──────────────────────────

def test_blast_plus_search_builds_the_command_line(query_file, tmp_path, monkeypatch):
    fake = FakeBlastPlus()
    monkeypatch.setattr(bl.subprocess, 'run', fake)

    bl._blast_plus_search(query_file, str(tmp_path / 'nt'), program='blastn',
                          evalue=1e-5, max_target_seqs=7, num_threads=4)

    cmd, kwargs = fake.calls[0]
    assert cmd[0] == 'blastn'
    assert cmd[1:3] == ['-query', query_file]
    assert cmd[3:5] == ['-db', str(tmp_path / 'nt')]
    assert cmd[5] == '-out' and cmd[6].endswith('.xml')
    assert cmd[7:9] == ['-outfmt', '5']                # XML, matching NCBIXML.parse
    assert cmd[9:11] == ['-evalue', '1e-05']
    assert cmd[11:13] == ['-max_target_seqs', '7']
    assert cmd[13:15] == ['-num_threads', '4']
    assert kwargs == {'capture_output': True, 'text': True, 'timeout': 600}


def test_blast_plus_search_parses_every_hsp_of_the_report(query_file, monkeypatch):
    monkeypatch.setattr(bl.subprocess, 'run', FakeBlastPlus())

    result = bl._blast_plus_search(query_file, 'nt', evalue=1e-5, max_target_seqs=7)

    assert (result.engine, result.program, result.database) == ('blast+', 'blastn', 'nt')
    assert result.query_length == 40 and result.message == '' and result.num_hits == 3
    first = result.hits[0]
    assert (first.query_id, first.subject_id) == ('myquery', 'NC_000001.1')
    assert first.subject_description == 'gn|NC_000001.1 Escherichia coli plasmid'
    assert first.percent_identity == pytest.approx(95.0)      # 38 identities / 40
    assert first.alignment_length == 40
    assert first.mismatches == 2                              # 40 - 38 - 0 gap columns
    assert first.gap_opens == 0
    assert (first.query_start, first.query_end) == (1, 40)
    assert (first.subject_start, first.subject_end) == (11, 50)
    assert first.e_value == pytest.approx(1e-10)
    assert first.bit_score == pytest.approx(74.3)
    assert first.query_cov == pytest.approx(100.0)
    # the gapped HSP: 40 - 37 identities - 1 gap column
    assert (result.hits[1].mismatches, result.hits[1].gap_opens,
            result.hits[1].percent_identity) == (2, 1, pytest.approx(92.5))
    # a hit whose <Hit_accession> is empty falls back to the first title token
    assert result.hits[2].subject_id == 'gn|NR_000002.1'
    assert result.hits[2].alignment_length == 36
    assert result.hits[2].query_cov == pytest.approx(90.0)
    assert [h.e_value for h in result.hits] == sorted(h.e_value for h in result.hits)


def test_blast_plus_search_deletes_the_report_it_asked_for(query_file, monkeypatch):
    fake = FakeBlastPlus()
    monkeypatch.setattr(bl.subprocess, 'run', fake)

    bl._blast_plus_search(query_file, 'nt')

    outfile = fake.calls[0][0][fake.calls[0][0].index('-out') + 1]
    assert os.path.exists(outfile) is False


def test_blast_plus_search_reports_a_nonzero_exit_with_the_tool_stderr(query_file,
                                                                      monkeypatch):
    """The exit code and the program name are in the message: `BLAST+ error:` alone did
    not say which tool failed or how, and the old code also quoted an empty stderr
    where BLAST+ often writes its complaint to stdout."""
    fake = FakeBlastPlus(report=None, returncode=1,
                         stderr='Database not found: nt ' * 40)
    monkeypatch.setattr(bl.subprocess, 'run', fake)

    result = bl._blast_plus_search(query_file, 'nt')

    assert result.engine == 'blast+' and result.num_hits == 0
    assert result.query_length == 0 and result.database == 'nt'
    head = 'BLAST+ error (blastn exited 1): '
    assert result.message.startswith(head)
    assert result.message == head + ('Database not found: nt ' * 40).strip()[:300]
    assert len(result.message) == len(head) + 300   # stderr[:300]


def test_blast_plus_search_reports_a_zero_exit_that_wrote_nothing(query_file, monkeypatch):
    """An empty report used to reach the XML parser and come back as its complaint
    ("Your XML file was empty"); the honest answer is that blastn exited 0 and produced
    no report, which is what a nonexistent -db argument looks like."""
    monkeypatch.setattr(bl.subprocess, 'run', FakeBlastPlus(report=''))

    result = bl._blast_plus_search(query_file, 'nt')

    assert result.engine == 'blast+' and result.num_hits == 0
    assert result.message.startswith('blastn exited 0 but wrote no report to ')
    assert 'the database is probably not a BLAST+ database' in result.message


@pytest.mark.parametrize('body, expected', [
    pytest.param('not xml at all',
                 r"Could not parse the BLAST\+ XML report: UnboundLocalError",
                 id="not-xml"),
    pytest.param('<BlastOutput><BlastOutput_program>blastn',
                 r"Could not parse the BLAST\+ XML report: UnboundLocalError",
                 id="truncated-xml"),
])
def test_blast_plus_search_reports_an_unparsable_report(query_file, monkeypatch, body,
                                                        expected):
    """Measured on Biopython 1.87: non-XML and truncated input both surface as
    `UnboundLocalError` from inside `NCBIXML.parse` (it reads `expat_parser` before
    assigning it), not as a parse error — the old broad `except Exception` was the only
    thing that could hold all three shapes, and it also held real bugs."""
    monkeypatch.setattr(bl.subprocess, 'run', FakeBlastPlus(report=body))

    result = bl._blast_plus_search(query_file, 'nt')

    assert result.num_hits == 0 and result.query_length == 0
    assert re.match(expected, result.message)
    assert "while reading the report" in result.message or 'expat_parser' in result.message


def test_blast_plus_search_keeps_the_real_cause_of_a_report_error(query_file, monkeypatch):
    """A BLAST+ report with no `<Parameters>` block makes Biopython's handler raise
    `AttributeError: 'Parameters' object has no attribute 'expect'`, which the parser
    then re-presents as `ExpatError: parsing finished` — a message with no information
    in it.  The cause is kept in the text the user sees."""
    body = re.sub(r'<Parameters>.*?</Parameters>', '', BLAST_XML, flags=re.S)
    assert '<Parameters_expect>' not in body
    monkeypatch.setattr(bl.subprocess, 'run', FakeBlastPlus(report=body))

    result = bl._blast_plus_search(query_file, 'nt')

    assert result.num_hits == 0
    assert result.message.startswith('Could not parse the BLAST+ XML report:')
    assert 'while reading the report: AttributeError' in result.message


def test_blast_plus_search_appends_the_callers_extra_arguments(query_file, monkeypatch):
    """`extra_args` has to arrive after the fixed options and as separate argv entries:
    the list form is what keeps a value like `-evalue 1e-50` from being shell-split."""
    fake = FakeBlastPlus()
    monkeypatch.setattr(bl.subprocess, 'run', fake)

    bl._blast_plus_search(query_file, 'nt', extra_args=['-soft_masking', 'false',
                                                        '-word_size', '11'])

    cmd = fake.calls[0][0]
    assert cmd[-4:] == ['-soft_masking', 'false', '-word_size', '11'], \
        "the caller's flags go after the fixed options, as separate argv entries"


def test_blast_plus_search_reports_a_parse_error_that_has_no_cause(query_file, monkeypatch):
    """The other parse test covers the chained case; this one is the bare
    `ExpatError: no element found`, where the report says everything there is to say."""
    monkeypatch.setattr(bl.subprocess, 'run', FakeBlastPlus(report='<?xml version="1.0"?>'))

    result = bl._blast_plus_search(query_file, 'nt')

    assert result.num_hits == 0 and result.query_length == 0
    assert result.message.startswith('Could not parse the BLAST+ XML report: ExpatError:')
    assert 'no element found' in result.message
    assert 'while reading the report' not in result.message


def test_blast_plus_search_reports_a_well_formed_report_with_no_records(query_file,
                                                                       monkeypatch):
    """A BLAST+ run that matched nothing writes a complete, parseable report with no
    query records — that is not an empty result and not an error either, and the caller
    cannot tell the three apart from `num_hits` alone."""
    monkeypatch.setattr(bl.subprocess, 'run',
                        FakeBlastPlus(report='<?xml version="1.0"?><BlastOutput>'
                                             '</BlastOutput>'))

    result = bl._blast_plus_search(query_file, 'nt')

    assert result.engine == 'blast+' and result.num_hits == 0
    assert result.message.startswith('BLAST+ produced a report with no query records')
    assert 'nothing matched at evalue=1e-05' in result.message


def test_blast_plus_search_survives_a_report_that_was_already_removed(query_file,
                                                                     monkeypatch):
    """`blastn` can be pointed at a path it renames or removes; the cleanup step then
    finds no file, which is the outcome the caller wanted, not an error."""
    class SelfRemoving(FakeBlastPlus):
        def __call__(self, cmd, **kwargs):
            result = super().__call__(cmd, **kwargs)
            os.unlink(cmd[cmd.index('-out') + 1])
            return result

    monkeypatch.setattr(bl.subprocess, 'run', SelfRemoving())

    result = bl._blast_plus_search(query_file, 'nt')

    assert result.num_hits == 0
    assert result.message.startswith('blastn exited 0 but wrote no report to ')


def test_blast_plus_search_reports_a_timeout(query_file, monkeypatch):
    def raiser(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=600)

    monkeypatch.setattr(bl.subprocess, 'run', raiser)

    result = bl._blast_plus_search(query_file, 'nt')

    assert result.engine == 'blast+' and result.num_hits == 0
    assert 'timed out after 600 seconds' in result.message
    assert result.message.startswith('blastn could not be started: TimeoutExpired:')


def test_blast_plus_search_still_returns_when_cleanup_fails(query_file, monkeypatch):
    monkeypatch.setattr(bl.subprocess, 'run', FakeBlastPlus())

    def locked(path):
        raise OSError('the report file is locked')

    monkeypatch.setattr(bl.os, 'unlink', locked)

    assert bl._blast_plus_search(query_file, 'nt').num_hits == 3


# ── make_database ───────────────────────────────────────────────────────────

class FakeMakeBlastdb:
    """Fake ``subprocess.run`` for `makeblastdb`.

    A successful build writes the index file BLAST+ writes (`<prefix>.nin`), because a
    zero exit that produced no database is exactly the case `make_database` now checks
    for — the same "the tool said it worked" case `wrote_output` was written for.
    """

    def __init__(self, returncode=0, stderr='', write_index=True):
        self.returncode = returncode
        self.stderr = stderr
        self.write_index = write_index
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        if self.returncode == 0 and self.write_index:
            prefix = cmd[cmd.index('-out') + 1]
            with open(prefix + '.nin', 'w', encoding='utf-8') as handle:
                handle.write('BLAST-index. Version 1.3\n')   # non-empty, as a real one is
            with open(prefix + '.nsq', 'w', encoding='utf-8') as handle:
                handle.write('sequences\n')
        return subprocess.CompletedProcess(cmd, self.returncode, 'Building DB.',
                                           self.stderr)


def test_make_database_runs_makeblastdb_and_returns_the_prefix(subject_file, tmp_path,
                                                               monkeypatch):
    fake = FakeMakeBlastdb()
    monkeypatch.setattr(bl.subprocess, 'run', fake)
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: True)

    prefix = bl.make_database(subject_file, dbtype='prot', output_dir=str(tmp_path),
                              title='gut MAGs')

    assert prefix == str(tmp_path / 'blastdb')
    assert fake.calls == [(['makeblastdb', '-in', subject_file, '-dbtype', 'prot',
                            '-out', prefix, '-title', 'gut MAGs'],
                           {'capture_output': True, 'text': True, 'timeout': 300})]
    assert os.path.exists(prefix + '.nin'), "the returned prefix is a real database"


def test_make_database_creates_its_own_output_directory(subject_file, monkeypatch):
    monkeypatch.setattr(bl.subprocess, 'run', FakeMakeBlastdb())
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: True)

    prefix = bl.make_database(subject_file)

    assert os.path.basename(prefix) == 'blastdb'
    parent = os.path.dirname(prefix)
    assert os.path.basename(parent).startswith('blastdb_')
    assert os.path.isdir(parent)
    shutil.rmtree(parent, ignore_errors=True)


def test_make_database_refuses_a_zero_exit_that_left_no_index(subject_file, tmp_path,
                                                             monkeypatch):
    """`returncode == 0` was the whole success test, and the caller got a prefix with
    no database behind it — which `run_blast` then handed to BLAST+ as `-db`."""
    monkeypatch.setattr(bl.subprocess, 'run', FakeMakeBlastdb(write_index=False))
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: True)

    assert bl.make_database(subject_file, output_dir=str(tmp_path)) == subject_file
    assert not (tmp_path / 'blastdb.nin').exists()


def test_make_database_removes_a_directory_it_created_on_a_failed_build(subject_file,
                                                                        monkeypatch):
    """Three failed builds used to leave three `blastdb_*` directories in the system
    temp folder with no return value pointing at any of them."""
    monkeypatch.setattr(bl.subprocess, 'run',
                        FakeMakeBlastdb(returncode=1, stderr='no sequences in FASTA file'))
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: True)
    before = staging_dirs()

    with pytest.warns(PerformanceWarning, match='no sequences in FASTA file'):
        assert bl.make_database(subject_file) == subject_file
    assert staging_dirs() == before


def test_make_database_keeps_a_directory_it_was_given(subject_file, tmp_path, monkeypatch):
    monkeypatch.setattr(bl.subprocess, 'run', FakeMakeBlastdb(returncode=1, stderr='bad'))
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: True)

    with pytest.warns(PerformanceWarning, match='exited 1'):
        assert bl.make_database(subject_file, output_dir=str(tmp_path)) == subject_file
    assert tmp_path.exists(), "the caller's directory is not ours to delete"


@pytest.mark.parametrize('dbtype', ['protein', 'nucl ', '', None])
def test_make_database_refuses_a_dbtype_makeblastdb_would_reject(subject_file, dbtype):
    """The tool prints its own complaint and exits non-zero; the caller used to be
    handed a path either way."""
    with pytest.raises(ValueError, match=re.escape(
            f"dbtype must be 'nucl' or 'prot', got {dbtype!r}.")):
        bl.make_database(subject_file, dbtype=dbtype)


def test_make_database_refuses_a_source_that_is_not_there(tmp_path):
    missing = str(tmp_path / 'absent.fasta')
    with pytest.raises(FileNotFoundError, match='No FASTA to build a database from'):
        bl.make_database(missing)


@pytest.mark.parametrize('outcome', [
    subprocess.CompletedProcess('makeblastdb', 1, '', 'no sequences in FASTA file'),
    OSError('makeblastdb is not executable'),
    subprocess.SubprocessError('fork failed'),
])
def test_make_database_falls_back_to_the_fasta_when_the_build_fails(outcome, subject_file,
                                                                    monkeypatch):
    def fake_run(cmd, **kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(bl.subprocess, 'run', fake_run)
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: True)

    assert bl.make_database(subject_file) == subject_file


def test_make_database_returns_the_fasta_when_blast_is_not_installed(subject_file,
                                                                     monkeypatch):
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: False)

    assert bl.make_database(subject_file) == subject_file


# ── run_blast dispatch ──────────────────────────────────────────────────────

def test_run_blast_prefers_blast_plus_and_labels_the_engine(query_file, subject_file,
                                                            monkeypatch):
    monkeypatch.setattr(bl.subprocess, 'run', FakeBlastPlus())
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: True)

    result = bl.run_blast(query_file, subject_file)

    assert result.engine == 'blast+'
    assert result.message == 'Using BLAST+ (external)'
    assert result.num_hits == 3


def test_run_blast_warns_and_uses_the_builtin_engine_without_binaries(
        query_file, subject_file, monkeypatch):
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: False)
    forwarded = {}

    def spy_builtin(query_file_arg, database_file, evalue, max_hits, k=15,
                    program='blastn'):
        forwarded.update(query_file=query_file_arg, database=database_file,
                         evalue=evalue, max_hits=max_hits, k=k, program=program)
        return bl.BlastResult(program=program, database='sp', query_length=16)

    monkeypatch.setattr(bl, '_builtin_search', spy_builtin)

    with pytest.warns(PerformanceWarning, match='BLAST\\+ not found'):
        result = bl.run_blast(query_file, subject_file, program='blastp',
                              evalue=0.5, max_target_seqs=9)

    assert result.engine == 'builtin'
    assert result.message == (
        'Using built-in k-mer search engine (BLAST+ recommended for production); '
        "blastp is a BLAST+ program: the built-in engine is a nucleotide-only k-mer "
        'search and did not apply it')
    # the cutoff, the hit limit and the requested program are handed over
    assert forwarded == {'query_file': query_file, 'database': subject_file,
                         'evalue': 0.5, 'max_hits': 9, 'k': 15, 'program': 'blastp'}


def test_run_blast_names_the_blast_options_the_builtin_engine_cannot_apply(
        query_file, subject_file, monkeypatch):
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: False)
    monkeypatch.setattr(bl, '_builtin_search',
                        lambda *a, **k: bl.BlastResult(program='blastn', database='sp',
                                                       query_length=16))

    with pytest.warns(PerformanceWarning):
        result = bl.run_blast(query_file, subject_file, extra_args=['-soft_masking', 'false'])

    assert ("extra_args ['-soft_masking', 'false'] are BLAST+ options and were not "
            "applied") in result.message


def test_run_blast_refuses_a_program_blast_does_not_have(query_file, subject_file,
                                                         monkeypatch):
    """Measured before: `'blastz'`, `'BLASTN'` and `''` each ran the built-in k-mer
    engine and reported `program='builtin_search'`, so the caller's choice disappeared
    and the report looked like a successful search."""
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: False)
    ran = []
    monkeypatch.setattr(bl, '_builtin_search',
                        lambda *a, **k: ran.append(a) or bl.BlastResult(
                            program='blastn', database='sp', query_length=16))

    for program in ('blastz', 'BLASTN', 'blastn-t', '', None):
        result = bl.run_blast(query_file, subject_file, program=program)
        assert result.engine == 'builtin'
        assert result.message == (
            f"Unknown program {program!r}; use 'blastn' or 'blastp' or 'blastx' or "
            f"'tblastn' or 'tblastx'.")
    assert ran == [], "a refused program must not start a search"
    assert bl.BLAST_PROGRAMS == ('blastn', 'blastp', 'blastx', 'tblastn', 'tblastx')


def test_run_blast_needs_a_fasta_for_the_builtin_engine(query_file, tmp_path, monkeypatch):
    """A database name the filesystem does not know used to be handed to the built-in
    reader, which then raised `FileNotFoundError` out of `run_blast` — and with it
    `run_blast_quick('ACGT…')`, whose default database is the BLAST+ name `nt`."""
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: False)

    result = bl.run_blast(query_file, str(tmp_path / 'no-such-db'))

    assert result.engine == 'builtin' and result.num_hits == 0
    assert result.message.startswith('The built-in engine searches a FASTA file, and')
    assert 'install BLAST+ to search a named database' in result.message


def test_run_blast_passes_a_named_database_to_blast_plus(query_file, monkeypatch):
    """`-db nt` is resolved by BLAST+ through `BLASTDB`, so it is not a file and must
    still reach the tool."""
    fake = FakeBlastPlus()
    monkeypatch.setattr(bl.subprocess, 'run', fake)
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: True)

    result = bl.run_blast(query_file, 'nt')

    assert result.engine == 'blast+' and result.num_hits == 3
    cmd = fake.calls[0][0]
    assert cmd[cmd.index('-db') + 1] == 'nt'
    assert result.message == 'Using BLAST+ (external)'


def test_run_blast_reports_an_unreadable_database_inst_of_crashing(query_file, tmp_path,
                                                                   monkeypatch):
    bad = tmp_path / 'bad.fasta'
    bad.write_text('# comment before sequence\n%%%%\n', encoding='utf-8')
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: False)

    result = bl.run_blast(query_file, str(bad))

    assert result.engine == 'builtin' and result.num_hits == 0
    assert result.message.startswith('Could not read the search inputs: ')
    assert 'FASTA file contains comments' in result.message


def test_run_blast_reports_a_missing_query_file(tmp_path):
    missing = str(tmp_path / 'absent.fasta')

    result = bl.run_blast(missing, 'nt')

    assert result.engine == 'builtin'
    assert result.message == f'Query not found: {missing}'
    assert result.database == 'nt' and result.num_hits == 0


def test_run_blast_quick_writes_and_removes_the_query_file(subject_file, monkeypatch):
    seen = {}

    def spy_run_blast(query_file, database, program='blastn', evalue=1e-5):
        seen['path'] = query_file
        with open(query_file, encoding='utf-8') as handle:
            seen['content'] = handle.read()
        seen['database'], seen['program'], seen['evalue'] = database, program, evalue
        return bl.BlastResult(program=program, database=os.path.basename(database),
                              query_length=len(QUERY_SEQ))

    monkeypatch.setattr(bl, 'run_blast', spy_run_blast)

    result = bl.run_blast_quick(QUERY_SEQ, program='blastx', database=subject_file,
                                evalue=0.25)

    assert seen['content'] == f'>query\n{QUERY_SEQ}\n'
    assert seen['database'] == subject_file
    assert seen['program'] == 'blastx' and seen['evalue'] == 0.25
    assert result.database == 'subject.fasta' and result.query_length == 40
    assert not os.path.exists(seen['path'])            # the temp query is always cleaned


# ── built-in engine ─────────────────────────────────────────────────────────

def test_builtin_search_finds_every_subject_carrying_the_seed(query_file, subject_file):
    result = bl._builtin_search(query_file, subject_file, evalue=1e-5, max_hits=500)

    assert (result.engine, result.program) == ('builtin', 'blastn')
    assert result.database == 'subject.fasta'          # basename, not the tmp path
    assert result.query_length == 16 and result.message == ''
    assert sorted(h.subject_id for h in result.hits) == ['subjA', 'subjB', 'subjC']
    hit = result.hits[0]
    assert (hit.query_id, hit.subject_id, hit.subject_description) == ('q1', 'subjA',
                                                                       'subjA')
    assert hit.percent_identity == pytest.approx(100.0)
    assert hit.alignment_length == 16 and hit.mismatches == 0
    assert hit.gap_opens == 0
    assert (hit.query_start, hit.query_end) == (1, 16)
    assert (hit.subject_start, hit.subject_end) == (1, 16)
    assert hit.e_value == pytest.approx(1.3996079881955784e-08)
    assert hit.bit_score == pytest.approx(16.0)
    assert hit.query_cov == pytest.approx(100.0)
    assert {type(name).__name__ for name in
            (hit.query_start, hit.query_end, hit.subject_start, hit.subject_end,
             hit.alignment_length, hit.mismatches)} == {'int'}, \
        "numpy scalars are not JSON serialisable, and the annotations promise int"


def test_builtin_search_reports_the_program_the_caller_asked_for(query_file, subject_file):
    """`program` used to be the literal 'builtin_search' whatever had been requested,
    so a caller who asked for blastp read back a program name that does not exist."""
    for program in ('blastn', 'blastp', 'tblastn'):
        result = bl._builtin_search(query_file, subject_file, evalue=1e-5,
                                   program=program)
        assert result.program == program


def test_builtin_search_returns_no_sequences_for_an_empty_input(tmp_path):
    empty = tmp_path / 'empty.fasta'
    empty.write_text('', encoding='utf-8')

    result = bl._builtin_search(str(empty), str(empty))

    assert result.message == 'No sequences found.'
    # Both return paths report the database the same way: the basename. The early one
    # used to hand back the full temporary path, so the same field held two shapes.
    assert result.database == 'empty.fasta'
    assert result.engine == 'builtin' and result.program == 'blastn'
    assert result.query_length == 0 and result.num_hits == 0


def test_builtin_search_clamps_the_seed_to_the_shortest_sequence(tmp_path):
    query = tmp_path / 'q.fasta'
    query.write_text(f'>q1\n{CORE}\n', encoding='utf-8')                  # 16 bases
    subject = tmp_path / 's.fasta'
    subject.write_text('>tiny\nACGTACGTACGTAC\n', encoding='utf-8')       # 14 bases

    result = bl._builtin_search(str(query), str(subject))

    assert result.message == 'k clamped to 14 (longest possible seed)'
    assert [(h.subject_id, h.alignment_length, h.percent_identity)
            for h in result.hits] == [('tiny', 14, pytest.approx(100.0))]


def test_builtin_search_stops_appending_hits_at_max_hits(query_file, subject_file):
    one = bl._builtin_search(query_file, subject_file, evalue=1e-5, max_hits=1)
    two = bl._builtin_search(query_file, subject_file, evalue=1e-5, max_hits=2)

    assert [h.subject_id for h in one.hits] == ['subjA']
    assert [h.subject_id for h in two.hits] == ['subjA', 'subjB']


def test_max_hits_bounds_the_whole_result_not_just_one_query(tmp_path, subject_file):
    """The limit used to `break` out of the seed loop of one query and then continue
    with the next, so four queries with `max_hits=1` kept four hits.  Measured before
    the fix: `max_hits=1 -> 4 hits kept`."""
    four = tmp_path / 'four.fasta'
    four.write_text(''.join(f'>q{i}\n{CORE}\n' for i in range(4)), encoding='utf-8')

    for limit in (1, 2):
        result = bl._builtin_search(str(four), subject_file, evalue=1e-5, max_hits=limit)
        assert result.num_hits == limit, f"max_hits={limit} kept {result.num_hits}"


def test_run_blast_keeps_the_engines_own_message(query_file, subject_file, tmp_path,
                                                 monkeypatch):
    """`result.message = engine_msg` used to overwrite whatever the engine had put
    there — including a BLAST+ failure and the built-in engine's k-clamp note."""
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: False)
    short = tmp_path / 'short.fasta'
    short.write_text('>q\nACGT\n', encoding='utf-8')
    short_db = tmp_path / 'shortdb.fasta'
    short_db.write_text('>s\nACGT\n>t\nACGA\n', encoding='utf-8')

    with pytest.warns(PerformanceWarning):
        result = bl.run_blast(str(short), str(short_db), evalue=1.0)

    assert result.message.endswith('k clamped to 4 (longest possible seed)')
    assert result.message.startswith('Using built-in k-mer search engine')

    monkeypatch.setattr(bl.subprocess, 'run',
                        FakeBlastPlus(report=None, returncode=1, stderr='boom'))
    monkeypatch.setattr(bl, '_has_blast_plus', lambda: True)
    result = bl.run_blast(query_file, subject_file)
    assert result.message == 'Using BLAST+ (external); BLAST+ error (blastn exited 1): boom'


def test_builtin_search_drops_hits_above_the_evalue_cutoff(query_file, subject_file):
    loose = bl._builtin_search(query_file, subject_file, evalue=1e-5)
    strict = bl._builtin_search(query_file, subject_file, evalue=1e-300)

    assert loose.num_hits == 3
    assert strict.num_hits == 0 and strict.message == ''


def test_builtin_search_skips_a_seed_that_names_an_unknown_subject(query_file,
                                                                  subject_file,
                                                                  monkeypatch):
    """A seed pointing at a subject the database does not carry is not a hit."""
    monkeypatch.setattr(bl, '_find_seed_hits',
                        lambda *a, **k: [(('ghost', 0), [0]), (('subjA', 0), [0])])

    result = bl._builtin_search(query_file, subject_file)

    assert [h.subject_id for h in result.hits] == ['subjA']


def test_builtin_search_extends_only_one_diagonal_per_subject(query_file, subject_file,
                                                              monkeypatch):
    """Two diagonals on the same subject must not both become hits."""
    monkeypatch.setattr(bl, '_find_seed_hits',
                        lambda *a, **k: [(('subjA', 0), [0]), (('subjA', 1), [1])])

    result = bl._builtin_search(query_file, subject_file)

    assert [h.subject_id for h in result.hits] == ['subjA']


def test_builtin_search_drops_a_seed_too_near_the_query_end_to_extend(tmp_path):
    """A tail-only seed anchors the window past the end of the query.

    The extension has no room, and that used to leave ``_banded_align_score``
    returning a 6-tuple where the caller unpacks seven, so a real FASTA pair
    raised ``ValueError: not enough values to unpack`` out of ``run_blast``.
    """
    subject = tmp_path / 'db.fasta'
    subject.write_text(f'>subj1\n{QUERY_SEQ}\n', encoding='utf-8')
    query = tmp_path / 'q.fasta'
    query.write_text(f'>q1\n{"N" * 25}{QUERY_SEQ[:15]}\n', encoding='utf-8')

    result = bl._builtin_search(str(query), str(subject), evalue=10.0, max_hits=50)

    assert result.num_hits == 0 and result.message == ''
    assert result.query_length == 40


def test_banded_align_score_reports_a_zero_width_alignment_without_room():
    assert bl._banded_align_score(CORE, CORE, 16, 0) == (0, 0, 0, 16, 0, 16, 0)
    assert bl._banded_align_score(CORE, CORE, 0, 16) == (0, 0, 0, 0, 16, 0, 16)
    assert len(bl._banded_align_score(CORE, CORE, 16, 16)) == 7


SCORED1 = 'TTTT' + CORE
SCORED2 = CORE + 'CCCC'


def test_banded_align_score_prefers_the_local_alignment():
    score, matches, mismatches, qs, ts, qe, te = bl._banded_align_score(SCORED1, SCORED2,
                                                                        0, 0)

    assert (score, matches, mismatches) == (pytest.approx(16.0), 16, 0)
    assert (qs, qe) == (4, 20) and (ts, te) == (0, 16)


@pytest.mark.parametrize('broken', ['raises', 'no-alignments'])
def test_banded_align_score_falls_back_to_pairwise_scoring_when_the_aligner_breaks(
        broken, monkeypatch):
    class BrokenAligner:
        def align(self, first, second):
            if broken == 'raises':
                raise TypeError('alphabets are not compatible')
            return []

    monkeypatch.setattr(bl, 'PairwiseAligner', BrokenAligner)

    assert bl._banded_align_score(SCORED1, SCORED2, 0, 0) == (8, 14, 6, 0, 0, 20, 20)


def test_read_fasta_returns_identifier_and_sequence(query_file, subject_file):
    assert bl._read_fasta(query_file) == [('q1', CORE)]
    assert bl._read_fasta(subject_file) == [('subjA', CORE), ('subjB', CORE),
                                            ('subjC', CORE + 'AAAA')]


def test_kmer_index_skips_ambiguous_kmers_and_seed_hits_report_the_diagonal():
    index = bl._build_kmer_index([('s1', 'ACGTNNACGTACGTACG')], k=8)

    assert sorted(index) == ['ACGTACGT', 'CGTACGTA', 'GTACGTAC', 'TACGTACG']
    assert index['ACGTACGT'] == [('s1', 6)]
    assert bl._find_seed_hits('NNNNNNNN', index, k=8) == []
    # an N-free tail still seeds; the key is (subject, subject_pos - query_pos)
    assert bl._find_seed_hits('NNNNNNNNACGTACGTACG', index, k=8) ==         [(('s1', -2), [8, 9, 10, 11])]
    # max_hits counts seed pairs, not subjects: two pairs, one diagonal
    assert bl._find_seed_hits('ACGTACGTACG', index, k=8, max_hits=2) ==         [(('s1', 6), [0, 1])]


# ── the module without Biopython ────────────────────────────────────────────

@contextlib.contextmanager
def _executed_without_biopython():
    """Run blast.py again with every ``Bio`` name hidden from the import system.

    ``HAS_BIO`` is decided while the module body executes, so the pure-Python
    side (``read_fasta_simple`` plus the fallback scorer) can only be reached by
    executing the file a second time — into a fresh module object, leaving the
    shared ``biosuite.core.blast`` untouched for the rest of the session.
    """
    class HiddenFinder:
        def find_spec(self, name, path=None, target=None):
            if name == 'Bio' or name.startswith('Bio.'):
                raise ImportError(f"{name} is hidden by this test")
            return None

    finder = HiddenFinder()
    parked = {name: module for name, module in list(sys.modules.items())
              if name == 'Bio' or name.startswith('Bio.')}
    for name in parked:
        del sys.modules[name]
    sys.meta_path.insert(0, finder)
    alias = f"{bl.__name__}_without_bio"
    spec = importlib.util.spec_from_file_location(alias, bl.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(parked)
        sys.modules.pop(alias, None)


def test_module_reads_fasta_without_biopython(subject_file):
    with _executed_without_biopython() as module:
        assert module.HAS_BIO is False
        assert module._read_fasta(subject_file) == [('subjA', CORE), ('subjB', CORE),
                                                    ('subjC', CORE + 'AAAA')]


def test_builtin_search_still_finds_the_subjects_without_biopython(query_file,
                                                                  subject_file):
    with _executed_without_biopython() as module:
        assert module._banded_align_score(SCORED1, SCORED2, 0, 0) == \
            (8, 14, 6, 0, 0, 20, 20)
        result = module._builtin_search(query_file, subject_file, evalue=1e-5)

    assert result.engine == 'builtin'
    assert sorted(h.subject_id for h in result.hits) == ['subjA', 'subjB', 'subjC']
    assert result.hits[0].percent_identity == pytest.approx(100.0)
    assert result.hits[0].alignment_length == 16


# ── report formatting ───────────────────────────────────────────────────────

def test_format_blast_result_handles_a_missing_result():
    assert bl.format_blast_result(None) == 'No BLAST results available.'


def test_format_blast_result_says_so_when_there_are_no_hits():
    result = bl.BlastResult(program='blastn', database='nt', query_length=40,
                            hits=[], engine='builtin')

    assert bl.format_blast_result(result) == (
        'Engine: builtin\nProgram: blastn\nDatabase: nt\nQuery length: 40 bp\n'
        'Total hits: 0\n\nNo significant hits found.')


def test_format_blast_result_tabs_hits_and_flags_the_ones_it_omitted():
    result = bl.BlastResult(program='blastn', database='nt', query_length=40,
                            hits=[_hit_of('NC_000001.1', 1e-10),
                                  _hit_of('NR_000002.1', 2e-6, identity=85.0,
                                          score=35.0)],
                            engine='blast+')

    text = bl.format_blast_result(result, max_hits=1)

    assert text.splitlines() == [
        'Engine: blast+', 'Program: blastn', 'Database: nt', 'Query length: 40 bp',
        'Total hits: 2', '',
        '#    Subject ID            Identity%      E-value    Score   Length',
        '-' * 72,
        '1    NC_000001.1               90.0%     1.00e-10       40       10',
        '',
        '... and 1 more hits.',
    ]
    assert 'NR_000002.1' not in text


def test_format_blast_result_lists_every_hit_when_the_limit_allows_it():
    result = bl.BlastResult(program='blastn', database='nt', query_length=40,
                            hits=[_hit_of('NC_000001.1', 1e-10)], engine='blast+')

    text = bl.format_blast_result(result)

    assert 'more hits' not in text
    assert text.splitlines()[8] == \
        '1    NC_000001.1               90.0%     1.00e-10       40       10'
