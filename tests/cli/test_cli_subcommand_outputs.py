"""The nineteen ``biosuite <command>`` subcommands, each with a real argument.

``_execute_command`` is the non-interactive face of the tool, and five of its commands
were broken in a way only a run could show: the registry passed the file path *string*
straight to functions expecting data — ``gwas`` answered
``AttributeError: 'str' object has no attribute 'iterrows'``, ``volcano`` and ``pca``
answered ``could not convert string to float: 'lfc.txt'``, ``manhattan`` answered
``missing 2 required positional arguments`` — and ``blast`` printed the repr of a
``BlastResult`` whose message was ``Query not found`` while exiting **0**, so any script
chaining it believed the search had succeeded.
"""
import io
from contextlib import redirect_stdout

import pytest

from biosuite.cli import menu

DNA = ("ATGCGTACGTTAGCGTACAGTACTGCATGCATCGTCGGGATCGATCGCGATCGATCGCGATCGATCG"
       "ATCGATCGATCGATCGATCG")
PROTEIN = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPMSIRSGKYVSAYRNVYTAHESTVVPIFGTAVVG"

FILES = {
    "query.fasta": f">query\n{DNA[:40]}\n",
    "database.fasta": f">subject\n{DNA[:40]}\n>other\n{DNA[10:50]}\n",
    "gwas.csv": "chrom,pos,snp_id,case_alt,case_ref,ctrl_alt,ctrl_ref\n"
                "chr1,1000,rsLead,90,10,10,90\nchr1,2000,rsNull,50,50,50,50\n",
    "lfc.txt": "2.0\n-2.0\n0.1\n1.5\n",
    "pvals.txt": "0.001\n0.002\n0.9\n0.04\n",
    "matrix.csv": "a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,2,4\n",
    "manhattan.csv": "chrom,pos,p_value\nchr1,1000,1e-9\nchr1,2000,0.4\nchr2,500,1e-12\n",
    "headerless.txt": "just words\nno numbers here\n",
}


def run_command(*argv):
    """Call ``main_cli`` the way the console script does and hand back (code, output)."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = menu.main_cli(list(argv))
    return code, buffer.getvalue()


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    for name, body in FILES.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # The launchers are covered by their own tests; here they must not open a window
    # or bind a port.
    monkeypatch.setattr(menu, "_launch_gui", lambda: print("gui requested"))
    monkeypatch.setattr(menu, "_launch_api",
                        lambda port=8000: print(f"api requested on {port}"))
    return tmp_path


# ── plain sequence commands ──────────────────────────────────────────────────

@pytest.mark.parametrize(("argv", "code", "expected"), [
    (["gc", "ATCGATCG"], 0, "50.0"),
    (["revcomp", "ATCG"], 0, "CGAT"),
    (["translate", "ATGCCCTAA"], 0, "MP*"),
])
def test_single_value_commands_print_the_value(argv, code, expected):
    status, out = run_command(*argv)

    assert (status, out.strip()) == (code, expected)


def test_stats_command_reports_the_composition():
    status, out = run_command("stats", "ATCG")

    assert status == 0
    for key in ("length", "A", "T", "G", "C", "N", "AT", "GC"):
        assert f"{key}:" in out


def test_alignment_commands_print_aligned_lines_not_a_tuple():
    """The registry used to print ``('ACGT', 'ACGA', 2)`` straight from the function."""
    for command in ("nw", "sw"):
        status, out = run_command(command, "ACGT", "ACGA")

        assert status == 0, command
        assert "Score:" in out, command
        assert "(" not in out, f"{command} printed a Python tuple at the user"


def test_hwe_command_takes_three_counts():
    status, out = run_command("hwe", "30", "50", "20")

    assert status == 0
    assert "p_value" in out
    assert "allele_freq_p" in out


def test_hwe_command_rejects_a_non_integer_count():
    status, out = run_command("hwe", "30", "many", "20")

    assert status == 2
    assert "must be integers" in out


# ── commands that read the user's report formatters ──────────────────────────

def test_crispr_command_uses_the_module_report():
    status, out = run_command("crispr", DNA)

    assert status == 0
    assert "CRISPR" in out and "Guide" in out
    assert "CRISPRResult(" not in out


def test_epitope_command_reports_both_cell_types():
    status, out = run_command("epitope", PROTEIN)

    assert status == 0
    assert "Epitope Prediction Report" in out
    assert "object at 0x" not in out


def test_digest_command_uses_the_digest_report():
    status, out = run_command("digest", "GAATTCGGATCC", "EcoRI")

    assert status == 0
    assert "DIGESTION REPORT" in out
    assert "fragments: [" not in out


def test_primers_command_uses_the_primer_report():
    status, out = run_command("primers", DNA)

    assert status == 0
    assert "PRIMER DESIGN REPORT" in out


def test_blast_command_reports_hits_and_exits_zero():
    status, out = run_command("blast", "query.fasta", "database.fasta")

    assert status == 0
    assert "Total hits:" in out
    assert "Query not found" not in out


def test_blast_command_fails_loudly_when_a_file_is_missing():
    """Exit code 0 with "Query not found" in the output used to be the behaviour."""
    for argv in (["blast", "absent.fasta", "database.fasta"],
                 ["blast", "query.fasta", "absent.fasta"]):
        status, out = run_command(*argv)

        assert status == 2, argv
        assert "File not found" in out, argv


def test_gwas_command_runs_the_association_test():
    status, out = run_command("gwas", "gwas.csv")

    assert status == 0
    assert "rsLead" in out
    assert "iterrows" not in out


def test_gwas_command_names_the_missing_columns():
    """A table without the case/control columns must not become a KeyError later."""
    status, out = run_command("gwas", "manhattan.csv")

    assert status in (1, 2)
    assert "missing required column" in out


# ── commands that write a figure ─────────────────────────────────────────────

@pytest.mark.parametrize(("command", "argv", "png"), [
    ("volcano", ["volcano", "lfc.txt", "pvals.txt"], "volcano.png"),
    ("pca", ["pca", "matrix.csv"], "pca.png"),
    ("manhattan", ["manhattan", "manhattan.csv"], "manhattan.png"),
])
def test_plot_commands_save_a_png(workspace, command, argv, png):
    status, out = run_command(*argv)

    assert status == 0, command
    assert png in out, command
    with (workspace / png).open("rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


def test_plot_commands_accept_an_output_name(workspace):
    status, out = run_command("volcano", "lfc.txt", "pvals.txt", "figure.png")

    assert status == 2
    assert "Too many arguments" in out


def test_a_numeric_file_without_numbers_is_reported(workspace):
    status, out = run_command("volcano", "headerless.txt", "pvals.txt")

    assert status in (1, 2)
    assert "No numbers found" in out


def test_manhattan_command_requires_its_three_columns(workspace):
    status, out = run_command("manhattan", "gwas.csv")

    assert status in (1, 2)
    assert "missing column(s)" in out


# ── plumbing around the registry ─────────────────────────────────────────────

def test_unknown_command_lists_what_exists():
    status, out = run_command("nosuchthing")

    assert status == 2
    assert "Unknown command: nosuchthing" in out
    for name in menu.COMMAND_REGISTRY:
        assert name in out


def test_a_missing_argument_prints_usage_instead_of_a_traceback():
    status, out = run_command("gc")

    assert status == 2
    assert "Usage: biosuite gc <sequence>" in out
    assert "Missing argument(s): sequence" in out


def test_theme_command_records_the_choice(workspace, monkeypatch):
    """The choice has to be applied *and* written, which the inline copy skipped."""
    monkeypatch.setitem(menu.config, "theme", "dark-green")

    status, out = run_command("theme", "dark-purple")

    assert status == 0
    assert "Theme changed to: dark-purple" in out
    assert menu.config["theme"] == "dark-purple"


def test_an_unknown_theme_is_refused_not_applied(workspace, monkeypatch):
    """``set_theme`` reads any non-dark name as light, so a typo changed the app."""
    monkeypatch.setitem(menu.config, "theme", "dark-green")

    status, out = run_command("theme", "dark-greene")

    assert status == 2
    assert "Unknown theme" in out
    assert menu.config["theme"] == "dark-green"


def test_the_interactive_theme_option_shares_the_command_line_rules(workspace, monkeypatch):
    """Option 99 and ``biosuite theme`` must not be two different implementations."""
    import builtins

    monkeypatch.setitem(menu.config, "theme", "dark-green")
    monkeypatch.setattr(builtins, "input", lambda prompt="": "2")

    menu._handle_menu_choice("99")

    assert menu.config["theme"] == "dark-purple"

    monkeypatch.setattr(builtins, "input", lambda prompt="": "7")
    status, out = run_command("theme", "7")
    assert status == 2


def test_gui_and_api_commands_reach_the_launchers(workspace):
    assert run_command("gui") == (0, "gui requested\n")
    assert run_command("api") == (0, "api requested on 8000\n")


def test_every_registered_command_is_callable_and_documented():
    """A registry entry that names a function that does not exist is a broken CLI.

    Every entry is dispatched through the same import path the command uses, so this
    fails the moment a rename leaves an entry behind.
    """
    for name, entry in menu.COMMAND_REGISTRY.items():
        assert entry.get("desc"), name
        target = entry["func"]
        if target.startswith("_"):
            assert hasattr(menu, target), f"{name} dispatches to a missing {target}"
        else:
            assert callable(menu._import_func(target)), f"{name} resolves to no function"
        assert all(argument.isidentifier() for argument in entry["args"]), name


# ── error policy of the command boundary ─────────────────────────────────────

def _register(monkeypatch, name, raiser, args=()):
    """Add a temporary command whose function is a test seam."""
    monkeypatch.setitem(menu.COMMAND_REGISTRY, name,
                        {"func": "test-only:run", "desc": "test seam", "args": list(args)})
    monkeypatch.setattr(menu, "_import_func", lambda path: raiser)


def test_a_keyboard_interrupt_at_the_command_boundary_exits_130(monkeypatch):
    _register(monkeypatch, "interrupting", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))

    status, out = run_command("interrupting")

    assert status == 130
    assert "Interrupted." in out


def test_an_unexpected_error_is_named_and_the_session_survives(monkeypatch):
    def explode(*args, **kwargs):
        raise ZeroDivisionError("division by zero")

    _register(monkeypatch, "exploding", explode)
    monkeypatch.delenv("BIOSUITE_DEBUG", raising=False)

    status, out = run_command("exploding")

    assert status == 1
    assert "Error: ZeroDivisionError: division by zero" in out
    assert "BIOSUITE_DEBUG=1" in out


def test_debug_mode_lets_the_traceback_through(monkeypatch):
    def explode(*args, **kwargs):
        raise ZeroDivisionError("division by zero")

    _register(monkeypatch, "exploding", explode)
    monkeypatch.setenv("BIOSUITE_DEBUG", "1")

    with pytest.raises(ZeroDivisionError):
        run_command("exploding")


def test_a_table_result_is_printed_as_a_table(monkeypatch):
    import pandas as pd

    _register(monkeypatch, "framing",
              lambda *a, **k: pd.DataFrame({"gene": ["G1", "G2"], "value": [1.5, 2.5]}))

    status, out = run_command("framing")

    assert status == 0
    assert "gene" in out and "G1" in out and "2.5" in out


def test_a_command_that_returns_nothing_prints_nothing(monkeypatch):
    _register(monkeypatch, "silent", lambda *a, **k: None)

    status, out = run_command("silent")

    assert (status, out.strip()) == (0, "")


# ── the file loaders behind the plot commands ────────────────────────────────

def test_number_reading_skips_comments_and_headers(workspace):
    (workspace / "mixed.txt").write_text("# a comment\n\nvalue\n2.5\n-2.0\n0.1\n1.5\n",
                                         encoding="utf-8")

    status, out = run_command("volcano", "mixed.txt", "pvals.txt")

    assert status == 0, out
    assert "Saved: volcano.png" in out


def test_the_two_volcano_files_must_line_up(workspace):
    """A length mismatch used to surface as a numpy broadcast error."""
    (workspace / "short.txt").write_text("2.5\n4.5\n", encoding="utf-8")

    status, out = run_command("volcano", "short.txt", "pvals.txt")

    assert status in (1, 2)
    assert "the two files need one value per gene" in out
    assert "broadcast" not in out


def test_pca_refuses_a_table_without_numeric_columns(workspace):
    (workspace / "names.csv").write_text("a,b\ncat,dog\nfish,bird\n", encoding="utf-8")

    status, out = run_command("pca", "names.csv")

    assert status in (1, 2)
    assert "no numeric columns" in out
