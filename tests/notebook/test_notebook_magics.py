"""The IPython magic surface: ``%biosuite``, ``%biostats``, ``%bioimport``, extension hook.

Magics are the part of the notebook module reachable without a running kernel, and
each one has a fallback path — an argument missing from the line, a variable that is
not in the namespace, a module that advertises a function it does not have.  Inside
a notebook a broken magic shows the user a traceback and nothing else, so every test
here asserts both the text the magic printed and the value it returned.

Two of the paths these tests cover were broken: ``%biostats`` on a DataFrame died
with ``NameError: name 'display' is not defined`` whenever IPython was installed
without ipywidgets (that import sat in the ipywidgets block), and ``%bioimport
plotting`` could never work because its module path was rebuilt from the dictionary
key and pointed at the non-existent ``biosuite.core.plotting``.
"""
import builtins

import numpy as np
import pandas as pd
import pytest

import biosuite.notebook as nbmod

pytest.importorskip("IPython")

from IPython.core.interactiveshell import InteractiveShell  # noqa: E402

from biosuite.core import alignment, blast, expression, phylogeny, sequence  # noqa: E402
from biosuite.plotting import plot_api  # noqa: E402

#: ``%bioimport <target>`` -> the names it promises and the module that holds them.
BIOIMPORT_TARGETS = {
    "sequence": (sequence, ["gc_content", "reverse_complement", "translate",
                            "read_fasta", "read_fastq", "sequence_stats"]),
    "alignment": (alignment, ["needleman_wunsch", "smith_waterman"]),
    "phylogeny": (phylogeny, ["distance_matrix", "upgma_tree"]),
    "expression": (expression, ["differential_expression", "cpm_normalization",
                                "tpm_normalization"]),
    "blast": (blast, ["run_blast"]),
    "plotting": (plot_api, ["volcano", "pca", "manhattan", "heatmap", "scatter",
                            "boxplot"]),
}


@pytest.fixture
def shell():
    """The process-wide InteractiveShell, with its user namespace restored after.

    ``Magics`` is a traitlets class and rejects a mock parent, so the real singleton
    is used.  The save/restore keeps the functions a magic imported out of later
    tests' namespaces.
    """
    ip = InteractiveShell.instance()
    saved = dict(ip.user_ns)
    yield ip
    ip.user_ns.clear()
    ip.user_ns.update(saved)


@pytest.fixture
def magic(shell):
    return nbmod.BioSuiteMagics(shell=shell)


# ── %biosuite ────────────────────────────────────────────────────────────────

def test_biosuite_gc_prints_and_returns_the_percentage(magic, capsys):
    assert magic.biosuite("gc ATCGATCG") == 50.0
    assert "GC content: 50.00%" in capsys.readouterr().out


def test_biosuite_revcomp_prints_and_returns_the_complement(magic, capsys):
    assert magic.biosuite("revcomp ATCG") == "CGAT"
    assert "Reverse complement: CGAT" in capsys.readouterr().out


def test_biosuite_translate_prints_and_returns_the_protein(magic, capsys):
    assert magic.biosuite("translate ATGAAATTTTAA") == "MKF*"
    assert "Protein: MKF*" in capsys.readouterr().out


def test_biosuite_stats_prints_every_count_it_returns(magic, capsys):
    result = magic.biosuite("stats ATCGATCG")

    assert result["length"] == 8 and result["GC"] == 50.0
    out = capsys.readouterr().out
    for key, value in result.items():
        assert f"{key}: {value}" in out


@pytest.mark.parametrize("command,answer,expected", [
    ("gc", "ATCG", "GC content"),
    ("revcomp", "ATCG", "Reverse complement"),
    ("translate", "ATG", "Protein"),
    ("stats", "ATCG", "length"),
])
def test_a_subcommand_asks_for_the_sequence_it_was_not_given(magic, monkeypatch,
                                                             command, answer, expected,
                                                             capsys):
    """``%biosuite gc`` with no argument is a valid line: it prompts instead."""
    monkeypatch.setattr(builtins, "input", lambda prompt="": answer)

    magic.biosuite(command)

    assert expected in capsys.readouterr().out


def test_search_reports_the_hits_from_two_files(magic, tmp_path, capsys):
    query = tmp_path / "query.fasta"
    query.write_text(">q1\nATCGATCGATCGATCGATCGATCG\n", encoding="utf-8")
    database = tmp_path / "db.fasta"
    database.write_text(">d1\nATCGATCGATCGATCGATCGATCG\n>d2\nGGGGCCCCAAAATTTTGGGG\n",
                        encoding="utf-8")

    result = magic.biosuite(f"search {query} {database}")

    assert result.num_hits >= 1
    assert f"Found {result.num_hits} hits" in capsys.readouterr().out


def test_search_refuses_a_line_without_a_database(magic, capsys):
    assert magic.biosuite("search only_one_path.fasta") is None
    assert "Usage: %biosuite search <query.fasta> <database.fasta>" in \
        capsys.readouterr().out


def _blast_pair(directory):
    """A query/database pair with a real BLAST hit, under ``directory``."""
    query = directory / "query.fasta"
    query.write_text(">q1\nATCGATCGATCGATCGATCGATCG\n", encoding="utf-8")
    database = directory / "database.fasta"
    database.write_text(">d1\nATCGATCGATCGATCGATCGATCG\n>d2\nGGGGCCCCAAAATTTTGGGG\n",
                        encoding="utf-8")
    return query, database


def test_search_accepts_a_quoted_path_with_one_space(magic, tmp_path, capsys):
    """The parser used to split on whitespace, so ``my query.fasta`` reached
    BLAST as three arguments and the search died on a file named ``my``."""
    spaced = tmp_path / "my data"
    spaced.mkdir()
    query, database = _blast_pair(spaced)

    result = magic.biosuite(f'search "{query}" "{database}"')

    assert result.num_hits >= 1
    assert f"Found {result.num_hits} hits" in capsys.readouterr().out


def test_search_keeps_consecutive_spaces_inside_quotes(magic, tmp_path, capsys):
    spaced = tmp_path / "lab files"
    spaced.mkdir()
    query, database = _blast_pair(spaced)
    double_spaced = spaced / "two  spaces.fasta"
    double_spaced.write_text(query.read_text(encoding="utf-8"), encoding="utf-8")

    result = magic.biosuite(f'search "{double_spaced}" "{database}"')

    assert result.num_hits >= 1


def test_search_accepts_one_quoted_argument_among_plain_ones(magic, tmp_path, capsys):
    spaced = tmp_path / "my data"
    spaced.mkdir()
    query, _ = _blast_pair(spaced)
    database = tmp_path / "plain_db.fasta"
    database.write_text(">d1\nATCGATCGATCGATCGATCGATCG\n", encoding="utf-8")

    result = magic.biosuite(f'search "{query}" {database}')

    assert result.num_hits >= 1


def test_search_survives_windows_backslashes_in_a_quoted_path(magic, tmp_path, capsys):
    """The real path on Windows carries backslashes; if the parser treated them
    as escapes the file would never be found."""
    spaced = tmp_path / "my data"
    spaced.mkdir()
    query, database = _blast_pair(spaced)

    result = magic.biosuite(f'search "{query}" "{database}"')

    assert result.num_hits >= 1


def test_an_unterminated_quote_is_a_controlled_error_not_a_crash(magic, capsys):
    assert magic.biosuite('search "unterminated.fasta') is None

    out = capsys.readouterr().out
    assert "Error:" in out
    assert "unterminated" in out
    assert "Usage:" not in out


@pytest.mark.parametrize("line,expected", [
    ('search "my query.fasta" db.fasta', ["search", "my query.fasta", "db.fasta"]),
    ("search 'q one.fasta' db.fasta", ["search", "q one.fasta", "db.fasta"]),
    (r'search "C:\dir with space\q.fasta"', ["search", "C:\\dir with space\\q.fasta"]),
    ("gc    ATCG  ", ["gc", "ATCG"]),
    ('seq"mid token"x', ['seqmid tokenx']),
])
def test_the_tokenizer_rules_behind_the_magic(magic, line, expected):
    assert nbmod._split_magic_args(line) == expected


@pytest.mark.parametrize("line", ["", "help"])
def test_help_lists_every_subcommand_the_magic_understands(magic, line, capsys):
    assert magic.biosuite(line) is None

    out = capsys.readouterr().out
    for name in ("gc", "revcomp", "translate", "stats", "search"):
        assert f"%biosuite {name}" in out


def test_an_unknown_subcommand_is_named_in_the_error(magic, capsys):
    assert magic.biosuite("nonsense ATCG") is None
    assert "Unknown command: nonsense" in capsys.readouterr().out


# ── %biostats ────────────────────────────────────────────────────────────────

def test_biostats_without_a_variable_shows_its_usage(magic, capsys):
    assert magic.biostats("") is None
    assert "Usage: %biostats <variable_name>" in capsys.readouterr().out


def test_biostats_names_a_variable_that_is_not_there(magic, shell, capsys):
    shell.user_ns.pop("absent", None)

    assert magic.biostats("absent") is None
    assert "Variable 'absent' not found" in capsys.readouterr().out


def test_biostats_describes_a_dataframe_including_its_head(magic, shell, capsys):
    shell.user_ns["df"] = pd.DataFrame({"log2fc": [1.0, -2.0], "p": [0.1, 0.2]})

    magic.biostats("df")

    head = capsys.readouterr().out.split("First 5 rows:")
    assert "DataFrame: 2 rows x 2 columns" in head[0]
    assert "Columns: ['log2fc', 'p']" in head[0]
    assert "Columns: ['log2fc', 'p']..." not in head[0], \
        "a two-column frame must not claim truncated columns"
    assert head[1].strip(), "%biostats displayed nothing after promising the head"
    assert "log2fc" in head[1]


@pytest.mark.parametrize("array,expected", [
    (np.arange(6.0), "Max: 5.0000"),
    (np.array([]), None),
])
def test_biostats_summarises_an_array_and_skips_stats_it_cannot_compute(
        magic, shell, capsys, array, expected):
    shell.user_ns["arr"] = array

    magic.biostats("arr")

    out = capsys.readouterr().out
    assert "shape=" in out
    if expected is None:
        assert "Min:" not in out, "an empty array has no minimum to report"
    else:
        assert expected in out


def test_biostats_falls_back_to_the_repr_of_anything_else(magic, shell, capsys):
    shell.user_ns["num"] = 7

    magic.biostats("num")

    out = capsys.readouterr().out
    assert "Type: int" in out and "Value: 7" in out


# ── %bioimport ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("target", sorted(BIOIMPORT_TARGETS))
def test_bioimport_puts_every_advertised_function_in_the_namespace(magic, shell,
                                                                   target, capsys):
    """The promise of the magic is the notebook user's ``gc_content(...)`` call.

    Each name is asserted to be the *same object* the module exposes, so a magic
    that claims to have loaded a function it does not have cannot pass.
    """
    module, names = BIOIMPORT_TARGETS[target]

    magic.bioimport(target)

    out = capsys.readouterr().out
    assert f"Loaded {target}: {', '.join(names)}" in out
    assert "Not found" not in out
    for name in names:
        assert shell.user_ns[name] is getattr(module, name)


def test_bioimport_all_loads_every_target(magic, shell, capsys):
    expected = [name for _target, (_module, names) in sorted(BIOIMPORT_TARGETS.items())
                for name in names]

    magic.bioimport("all")

    out = capsys.readouterr().out
    assert "All functions loaded into namespace!" in out
    assert "Not loaded:" not in out
    for name in expected:
        assert name in shell.user_ns


def test_bioimport_reports_a_module_that_cannot_be_imported(magic, monkeypatch, capsys):
    """``Skipping`` plus the reason — the old code swallowed this in ``pass``."""
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "biosuite.core.blast":
            raise ImportError("no blast here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    magic.bioimport("all")

    out = capsys.readouterr().out
    assert "Skipping blast: no blast here" in out
    assert "Not loaded: blast" in out
    assert "All functions loaded into namespace!" not in out


def test_bioimport_names_a_function_the_module_does_not_have(magic, shell, monkeypatch,
                                                             capsys):
    """A name that was advertised but is absent is reported, not quietly dropped."""
    monkeypatch.delattr(phylogeny, "upgma_tree")

    magic.bioimport("phylogeny")

    out = capsys.readouterr().out
    assert "Not found in biosuite.core.phylogeny: upgma_tree" in out
    assert "distance_matrix" in shell.user_ns
    assert "upgma_tree" not in shell.user_ns


def test_bioimport_lists_its_targets_for_an_unknown_one(magic, capsys):
    magic.bioimport("nope")

    out = capsys.readouterr().out
    assert "Available modules:" in out
    for target in BIOIMPORT_TARGETS:
        assert target in out
    assert out.rstrip().endswith("all")


# ── extension registration ───────────────────────────────────────────────────

def test_load_ipython_extension_registers_the_line_magics(shell, capsys):
    """The registry is keyed by class name on a process-wide shell, so an earlier
    test's registration would satisfy a naive assertion: it is cleared first."""
    shell.magics_manager.registry.pop("BioSuiteMagics", None)

    nbmod.load_ipython_extension(shell)

    registered = shell.magics_manager.registry["BioSuiteMagics"]
    assert registered.__class__ is nbmod.BioSuiteMagics
    assert {"biosuite", "biostats", "bioimport"} <= set(
        shell.magics_manager.magics["line"])
    assert "BioSuite magic commands loaded!" in capsys.readouterr().out
