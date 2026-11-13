"""The MrBayes wrapper: the script it generates, the tree it parses, and what it leaves behind.

Every failure path of the external engine has to answer two questions — what went
wrong, and who cleans up.  `mb` is not installed on this machine, so nothing here runs
it: the assertions are on the staged NEXUS, the generated script, the command line, the
parsed consensus and the directories, which are all checkable without the binary.
"""
import os
import subprocess
import tempfile

import pytest

from biosuite.core import bayesian_phylogeny as bp

FASTA_4 = ">A\nACGTACGT\n>B\nACGAACGT\n>C\nACGTGCGT\n>D\nTCGTACGA\n"
FASTA_DUPES = ">A\nACGTACGT\n>A\nACGAACGT\n>C\nACGTGCGT\n"
FASTA_NAMED = ">Homo sapiens\nACGT\n>Mus musculus\nACGA\n>Rattus norvegicus\nACGC\n"


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["mb"], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def staging_dirs():
    return {name for name in os.listdir(tempfile.gettempdir())
            if name.startswith("biosuite_mrbayes_")}


@pytest.fixture
def alignment(tmp_path):
    path = tmp_path / "aln.fasta"
    path.write_text(FASTA_4, encoding="utf-8")
    return str(path)


@pytest.fixture
def work(tmp_path):
    directory = tmp_path / "mb"
    directory.mkdir()
    return str(directory)


# ── staging the NEXUS ────────────────────────────────────────────────────────

def test_fasta_to_nexus_writes_the_block_mrBayes_expects(tmp_path):
    alignment = tmp_path / "aln.fasta"
    alignment.write_text(">A\nAC-G\n>B\nATNT\n>C\nACGA\n", encoding="utf-8")
    nexus = tmp_path / "aln.nex"
    assert bp._fasta_to_nexus(str(alignment), str(nexus)) is None
    assert nexus.read_text(encoding="utf-8") == (
        "#nexus\nbegin data;\n  dimensions ntax=3 nchar=4;\n"
        "  format datatype=dna missing=? gap=-;\n  matrix\n"
        "    A                    AC-G\n"
        "    B                    ATNT\n"
        "    C                    ACGA\n"
        ";\nend;\n")


def test_a_header_with_a_description_stages_only_its_first_token(tmp_path):
    """`record.id` is the first whitespace-delimited token, so the matrix line cannot
    arrive as three tokens.  This is the measurement that removed a quoting fix which
    had nothing to quote."""
    alignment = tmp_path / "named.fasta"
    alignment.write_text(FASTA_NAMED, encoding="utf-8")
    nexus = tmp_path / "named.nex"
    bp._fasta_to_nexus(str(alignment), str(nexus))
    assert nexus.read_text(encoding="utf-8").splitlines()[5:8] == [
        "    Homo                 ACGT",
        "    Mus                  ACGA",
        "    Rattus               ACGC"]


def test_a_repeated_label_is_refused_before_it_reaches_the_tool(tmp_path):
    """MrBayes rejects the whole data block for a duplicated taxon name, and the exit
    status told the user nothing about which two records collided."""
    alignment = tmp_path / "dupes.fasta"
    alignment.write_text(FASTA_DUPES, encoding="utf-8")
    with pytest.raises(ValueError, match=r"labels must be unique; .* repeats A$"):
        bp._fasta_to_nexus(str(alignment), str(tmp_path / "dupes.nex"))


# ── generating and running the script ────────────────────────────────────────

def test_mrbayes_run_stages_nexus_and_parses_the_consensus(tmp_path, monkeypatch,
                                                           alignment):
    seen = {}

    def fake_run(cmd, *args, **kwargs):
        seen['cmd'] = list(cmd)
        seen['kwargs'] = dict(kwargs)
        (tmp_path / "mb" / "alignment.nex.tre").write_text(
            "((A:0.1,B:0.1)50:0.05,C:0.2)root;\n", encoding="utf-8")
        return _completed(0, stdout="MrBayes 3.2.7")

    work = tmp_path / "mb"
    work.mkdir()
    monkeypatch.setattr(bp.subprocess, "run", fake_run)
    result = bp._mrbayes_run(alignment, n_generations=250, output_dir=str(work))
    assert seen['cmd'] == ["mb", str(work / "run.nex")]
    assert seen['kwargs']['cwd'] == str(work) and seen['kwargs']['timeout'] == 7200
    assert (result.engine, result.tree_file) == ("mrbayes", str(work / "alignment.nex.tre"))
    assert result.newick_tree == "((A:0.1,B:0.1)50:0.05,C:0.2)root;"
    assert result.message == "MrBayes posterior consensus tree"
    script = (work / "run.nex").read_text(encoding="utf-8").splitlines()
    assert script[0] == f'execute "{work / "alignment.nex"}"'
    assert script[1] == "lset nst=6 rates=invgamma"
    assert script[2] == "mcmc ngen=250 samplefreq=10 nruns=2 nchains=4"
    assert script[3:] == ["sump", "sumt", "quit"]
    assert (work / "alignment.nex").read_text(encoding="utf-8").startswith("#nexus")


def test_the_execute_command_survives_a_staging_directory_with_a_space(tmp_path,
                                                                       monkeypatch,
                                                                       alignment):
    """An absolute temporary path is not one token, and the unquoted form reached `mb`
    as `execute C:\\Users\\me\\Temp\\with space\\alignment.nex` — several arguments
    where the command takes one filename."""
    seen = {}
    spaced = tmp_path / "with space"
    spaced.mkdir()

    def fake_run(cmd, *args, **kwargs):
        seen['cmd'] = list(cmd)
        (spaced / "alignment.nex.tre").write_text("(A,B,C);\n", encoding="utf-8")
        return _completed(0)

    monkeypatch.setattr(bp.subprocess, "run", fake_run)
    result = bp._mrbayes_run(alignment, 100, output_dir=str(spaced))
    assert result.engine == "mrbayes"
    line = (spaced / "run.nex").read_text(encoding="utf-8").splitlines()[0]
    assert line == f'execute "{spaced / "alignment.nex"}"'
    assert line != f'execute {spaced / "alignment.nex"}', \
        "the unquoted form is what the wrapper used to write"


def test_a_run_with_no_output_file_is_not_reported_as_a_tree(alignment, work, monkeypatch):
    monkeypatch.setattr(bp.subprocess, "run", lambda cmd, *a, **k: _completed(0))
    with pytest.warns(RuntimeWarning, match=r"mb succeeded but.*was not produced"):
        assert bp._mrbayes_run(alignment, 50, output_dir=work) is None


@pytest.mark.parametrize("failure, matches", [
    pytest.param("nonzero", r"mb exited with 1: Error: unsupported model", id="exit-status"),
    pytest.param("launch", "mb could not be executed", id="binary-missing"),
])
def test_mrbayes_failures_are_warned_about_and_return_nothing(alignment, work, monkeypatch,
                                                             failure, matches):
    def fake_run(cmd, *args, **kwargs):
        if failure == "launch":
            raise FileNotFoundError(2, "No such file or directory: 'mb'")
        return _completed(1, stderr="Error: unsupported model")

    monkeypatch.setattr(bp.subprocess, "run", fake_run)
    with pytest.warns(RuntimeWarning, match=matches):
        assert bp._mrbayes_run(alignment, 50, output_dir=work) is None
    assert not os.path.exists(os.path.join(work, "alignment.nex.tre"))


# ── whose directory, and who removes it ──────────────────────────────────────

def test_a_caller_owned_directory_is_left_alone_even_when_the_run_fails(alignment, work,
                                                                        monkeypatch):
    monkeypatch.setattr(bp.subprocess, "run",
                        lambda cmd, *a, **k: _completed(1, stderr="rejected"))
    with pytest.warns(RuntimeWarning, match="mb exited with 1"):
        assert bp._mrbayes_run(alignment, 50, output_dir=work) is None
    assert sorted(os.listdir(work)) == ["alignment.nex", "run.nex"], \
        "the caller's directory is not ours to delete"


def test_a_failing_run_does_not_leave_its_own_directory_behind(alignment, monkeypatch):
    """Measured before: three failed runs with no `output_dir` left three directories
    in the system temp folder, each holding an `alignment.nex` and a `run.nex` that
    nobody could find again — and no `tree_file` to point at them."""
    monkeypatch.setattr(bp.subprocess, "run",
                        lambda cmd, *a, **k: _completed(1, stderr="rejected"))
    before = staging_dirs()
    for _ in range(3):
        with pytest.warns(RuntimeWarning, match="mb exited with 1"):
            assert bp._mrbayes_run(alignment, 50) is None
    assert staging_dirs() == before


def test_mrbayes_creates_its_own_output_directory_when_none_is_given(alignment,
                                                                     monkeypatch):
    """The success path is the one case where the directory outlives the call, because
    `tree_file` is returned and the consensus lives in it."""
    def fake_run(cmd, *args, **kwargs):
        tempfile_dir = kwargs['cwd']
        with open(os.path.join(tempfile_dir, "alignment.nex.tre"), "w",
                  encoding="utf-8") as handle:
            handle.write("(A,B,C);\n")
        return _completed(0)

    monkeypatch.setattr(bp.subprocess, "run", fake_run)
    before = staging_dirs()
    result = bp._mrbayes_run(alignment, 50)
    try:
        assert result.newick_tree == "(A,B,C);"
        out_dir = os.path.dirname(result.tree_file)
        assert os.path.isdir(out_dir)
        assert os.path.basename(out_dir).startswith("biosuite_mrbayes_")
        assert staging_dirs() == before | {os.path.basename(out_dir)}
    finally:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
