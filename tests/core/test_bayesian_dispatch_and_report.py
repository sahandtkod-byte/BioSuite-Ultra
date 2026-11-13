"""Choosing the engine and printing what it produced: `run_bayesian` and the report.

`run_bayesian` is the only entry point the CLI uses
(`biosuite/cli/menu.py:954`), so its contract — always a `BayesianResult`, never a
traceback — is what the menu's "Bayesian phylogeny" option actually prints.  Two
measured defects live here: an unrecognised `tool` value ran the built-in chain without
a word, and a FASTA MrBayes could not stage raised `ValueError` straight through that
contract.
"""
import os
import subprocess

import pytest

from biosuite.core import bayesian_phylogeny as bp
from biosuite.core import utils
from biosuite.core.bayesian_phylogeny import BayesianResult, format_bayesian_report

FASTA_4 = ">A\nACGTACGT\n>B\nACGAACGT\n>C\nACGTGCGT\n>D\nTCGTACGA\n"
FASTA_DUPES = ">A\nACGTACGT\n>A\nACGAACGT\n>C\nACGTGCGT\n"


def _completed(returncode=0, stderr=""):
    return subprocess.CompletedProcess(["mb"], returncode, "", stderr)


@pytest.fixture
def alignment(tmp_path):
    path = tmp_path / "aln.fasta"
    path.write_text(FASTA_4, encoding="utf-8")
    return str(path)


@pytest.fixture
def mrbayes_installed(monkeypatch):
    monkeypatch.setattr(bp, "check_bayesian_tools", lambda: {'mrbayes': True})


@pytest.fixture
def mrbayes_absent(monkeypatch):
    monkeypatch.setattr(bp, "check_bayesian_tools", lambda: {'mrbayes': False})


@pytest.fixture
def no_chain(monkeypatch):
    """Fail loudly if the built-in MCMC is asked to run, so a refusal test cannot pass
    by quietly falling through to a 5,000-generation chain."""
    def refuse(*args, **kwargs):
        raise AssertionError("the built-in chain must not run for a refused engine name")
    monkeypatch.setattr(bp, "_builtin_bayesian", refuse)


@pytest.mark.parametrize("tool", [
    pytest.param('mrbay', id="truncated"),
    pytest.param('MrBayes', id="wrong-case"),
    pytest.param('BUILTIN', id="upper-case-builtin"),
    pytest.param('', id="empty"),
    pytest.param(None, id="none"),
    pytest.param('external', id="not-an-engine"),
])
def test_an_engine_name_that_does_not_exist_is_refused(alignment, mrbayes_installed,
                                                       no_chain, tool):
    """Measured: all of these returned a built-in MCMC result with no trace of the
    engine the caller had named, so a typo in a script read as a successful run."""
    result = bp.run_bayesian(alignment, n_generations=20, tool=tool)
    assert result.engine == "none"
    assert result.message == f"Unknown tool {tool!r}; use 'auto', 'mrbayes' or 'builtin'."
    assert result.newick_tree == ""


def test_the_three_engine_names_the_module_offers_all_reach_an_analysis(alignment,
                                                                        mrbayes_absent):
    for tool in bp.BAYESIAN_ENGINES:
        result = bp.run_bayesian(alignment, n_generations=20, tool=tool, seed=1)
        assert result.engine == "builtin", f"{tool} must reach the built-in chain"
        assert result.num_samples > 0


def test_run_bayesian_hands_the_alignment_to_mrbayes_when_it_is_installed(alignment,
                                                                        mrbayes_installed,
                                                                        monkeypatch,
                                                                        no_chain):
    recorded = []

    def fake_run(cmd, *args, **kwargs):
        recorded.append((cmd, kwargs.get('cwd')))
        with open(os.path.join(kwargs['cwd'], "alignment.nex.tre"), "w",
                  encoding="utf-8") as handle:
            handle.write("(A,B,C);\n")
        return _completed(0)

    monkeypatch.setattr(bp.subprocess, "run", fake_run)
    result = bp.run_bayesian(alignment, n_generations=40, tool='mrbayes')
    try:
        assert result.engine == "mrbayes"
        assert result.newick_tree == "(A,B,C);"
        script = open(os.path.join(recorded[0][1], "run.nex"),
                      encoding="utf-8").read()
        assert "mcmc ngen=400 samplefreq=10" in script, \
            "MrBayes gets ten times the requested generations, and its script says so"
    finally:
        import shutil
        shutil.rmtree(recorded[0][1], ignore_errors=True)


def test_run_bayesian_falls_through_to_the_builtin_chain_without_mrBayes(alignment,
                                                                        mrbayes_absent,
                                                                        monkeypatch):
    launched = []
    monkeypatch.setattr(bp.subprocess, "run", lambda *a, **k: launched.append(a))
    result = bp.run_bayesian(alignment, n_generations=60, tool='auto', seed=5)
    assert launched == [], "nothing is launched when the tool probe says no"
    assert result.engine == "builtin"
    assert result.num_samples > 0


def test_run_bayesian_labels_the_result_with_the_engine_that_actually_ran(
        alignment, mrbayes_installed, monkeypatch):
    """MrBayes installed but useless: the caller still gets engine='builtin'."""
    monkeypatch.setattr(bp.subprocess, "run",
                        lambda cmd, *a, **k: _completed(2, stderr="Error: nexus rejected"))
    with pytest.warns(RuntimeWarning, match="mb exited with 2"):
        result = bp.run_bayesian(alignment, n_generations=20, tool='mrbayes', seed=1)
    assert result.engine == "builtin", "the fallback must not be reported as mrbayes"
    assert result.newick_tree.startswith("(")
    assert result.num_samples > 0


def test_an_alignment_mrBayes_cannot_stage_comes_back_as_a_reason(alignment,
                                                                 mrbayes_installed,
                                                                 tmp_path, no_chain):
    """The file exists, so the early check passes, and then the NEXUS staging step
    raises.  It used to escape `run_bayesian` as `ValueError` and land in the CLI's
    per-action guard as a stack trace instead of a report line."""
    dupes = tmp_path / "dupes.fasta"
    dupes.write_text(FASTA_DUPES, encoding="utf-8")
    result = bp.run_bayesian(str(dupes), n_generations=10, tool='mrbayes')
    assert result.engine == "none"
    assert result.message.startswith("Could not stage the MrBayes run: ValueError:")
    assert "repeats A" in result.message


def test_a_junk_fasta_with_mrBayes_installed_is_not_a_crash(tmp_path, mrbayes_installed,
                                                            no_chain):
    junk = tmp_path / "junk.fasta"
    junk.write_text("# comment\n%%%%\n", encoding="utf-8")
    result = bp.run_bayesian(str(junk), n_generations=10, tool='mrbayes')
    assert result.engine == "none"
    assert result.message.startswith("Could not stage the MrBayes run: ValueError:")


def test_run_bayesian_refuses_an_absent_alignment(tmp_path):
    missing = str(tmp_path / "gone.fasta")
    result = bp.run_bayesian(missing)
    assert result.engine == "none"
    assert result.message == f"File not found: {missing}"


def test_check_bayesian_tools_probes_the_mb_binary(monkeypatch):
    probed = []

    def fake_has_tool(name):
        probed.append(name)
        return name == 'mb'

    monkeypatch.setattr(utils, "has_tool", fake_has_tool)
    assert bp.check_bayesian_tools() == {'mrbayes': True}
    assert probed == ['mb']


# ── the report ───────────────────────────────────────────────────────────────

def test_bayesian_report_prints_diagnostics_and_truncates_a_long_tree():
    long_newick = "(A:0.1," * 30 + "Z:0.1);"
    result = BayesianResult(engine="mrbayes", num_samples=120, log_likelihood=-450.5,
                            ess=42.0, psrf=1.004, newick_tree=long_newick,
                            message="MrBayes posterior consensus tree")
    lines = format_bayesian_report(result).splitlines()
    assert lines[:6] == [
        "=== Bayesian Phylogenetics Report ===", "Engine: mrbayes", "Samples: 120",
        "Log-likelihood: -450.50", "ESS: 42", "PSRF: 1.004"]
    assert lines[6].startswith("Tree: (A:0.1,") and lines[6].endswith("...")
    assert len(lines[6]) == len("Tree: ") + 80 + len("...")
    assert lines[7] == "Note: MrBayes posterior consensus tree"


def test_a_run_that_never_started_is_not_printed_as_a_converged_analysis():
    """Measured before: the same table came out around a parse error, with
    `Samples: 0, Log-likelihood: 0.00, ESS: 0, PSRF: 1.000` — a finished-looking
    analysis of nothing, and the reason last."""
    failed = BayesianResult(engine="none", message="Error: This FASTA file contains comments")
    assert format_bayesian_report(failed).splitlines() == [
        "=== Bayesian Phylogenetics Report ===", "Engine: none",
        "Note: Error: This FASTA file contains comments"]


def test_a_failed_run_with_no_message_still_says_which_engine_did_not_run():
    """`message` defaults to "" on the dataclass, and the report is public, so a
    caller-built failure result must not end with an empty `Note: ` line."""
    assert format_bayesian_report(BayesianResult(engine="none")).splitlines() == [
        "=== Bayesian Phylogenetics Report ===", "Engine: none"]


def test_the_convergence_proxy_is_not_printed_as_clade_support():
    """`posterior_probability` is the fraction of samples within 2 sigma of their own
    mean; printing it under a name like "Posterior support" is how a proxy turns into
    a claim about clades."""
    result = BayesianResult(engine="builtin", num_samples=20, log_likelihood=-51.27,
                            ess=5.1, psrf=1.752, posterior_probability=0.95,
                            newick_tree="(A:0.1,B:0.1,C:0.1);")
    text = format_bayesian_report(result)
    assert "support" not in text.lower()
    assert "0.95" not in text
    assert "PSRF: 1.752" in text
