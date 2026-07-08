"""
Tests for the BioSuite CLI module (biosuite/cli/menu.py).

Covers:
  - Menu rendering (print_menu)
  - Argument parser construction (_build_parser)
  - Dynamic function import (_import_func)
  - Non-interactive command execution (_execute_command)
  - COMMAND_REGISTRY completeness
  - main_cli non-interactive path
  - Theme change command
"""
import sys
import argparse
from unittest import mock
from unittest.mock import patch, MagicMock

import pytest

# Ensure the project root is importable
sys.path.insert(0, "C:/Users/SAHAND/Desktop/python/BioSuite-Ultra")

from biosuite.cli.menu import (
    _build_parser,
    _import_func,
    _execute_command,
    _change_theme,
    main_cli,
    print_menu,
    COMMAND_REGISTRY,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def parser():
    """Return a freshly-built argparse parser."""
    return _build_parser()


@pytest.fixture
def seq_args():
    """A minimal namespace mimicking parsed args for a single-sequence command."""
    ns = argparse.Namespace()
    ns.sequence = "ATCGATCG"
    return ns


@pytest.fixture
def pair_args():
    """A minimal namespace mimicking parsed args for a two-sequence alignment."""
    ns = argparse.Namespace()
    ns.seq1 = "AGTACGCA"
    ns.seq2 = "TATGC"
    return ns


# ── COMMAND_REGISTRY tests ───────────────────────────────────────────────────


class TestCommandRegistry:
    """Verify the command registry is complete and well-formed."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "gc", "revcomp", "translate", "stats", "nw", "sw",
            "blast", "crispr", "epitope", "gwas", "hwe",
            "volcano", "pca", "manhattan", "digest", "primers",
            "gui", "api", "theme",
        ],
    )
    def test_expected_commands_exist(self, cmd):
        """All expected command keys should be present in the registry."""
        assert cmd in COMMAND_REGISTRY, f"Missing command '{cmd}' in COMMAND_REGISTRY"

    def test_registry_is_dict(self):
        """COMMAND_REGISTRY should be a plain dictionary."""
        assert isinstance(COMMAND_REGISTRY, dict)

    def test_each_entry_has_required_keys(self):
        """Every registry entry must have at least 'func' and 'args'."""
        for name, entry in COMMAND_REGISTRY.items():
            assert "func" in entry, f"Entry '{name}' missing 'func'"
            assert "args" in entry, f"Entry '{name}' missing 'args'"

    def test_seq_commands_reference_core_modules(self):
        """Core sequence commands should reference biosuite.core.sequence."""
        for cmd in ("gc", "revcomp", "translate", "stats"):
            assert "biosuite.core.sequence" in COMMAND_REGISTRY[cmd]["func"]


# ── _build_parser tests ─────────────────────────────────────────────────────


class TestBuildParser:
    """Validate the argparse parser returned by _build_parser."""

    def test_returns_argparse_parser(self, parser):
        assert isinstance(parser, argparse.ArgumentParser)

    def test_has_command_argument(self, parser):
        """Parser should accept an optional positional command."""
        args = parser.parse_args(["gc", "ATCGATCG"])
        assert args.command == "gc"
        assert args.cmd_args == ["ATCGATCG"]

    def test_has_gui_flag(self, parser):
        args = parser.parse_args(["--gui"])
        assert args.gui is True

    def test_has_api_flag(self, parser):
        args = parser.parse_args(["--api"])
        assert args.api is True

    def test_has_port_option(self, parser):
        args = parser.parse_args(["--port", "9000"])
        assert args.port == 9000

    def test_has_theme_option(self, parser):
        args = parser.parse_args(["--theme", "dark-purple"])
        assert args.theme == "dark-purple"

    def test_has_version_flag(self, parser):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_no_args_gives_none_command(self, parser):
        args = parser.parse_args([])
        assert args.command is None


# ── _import_func tests ──────────────────────────────────────────────────────


class TestImportFunc:
    """Dynamic function import via _import_func."""

    def test_import_gc_content(self):
        func = _import_func("biosuite.core.sequence:gc_content")
        assert callable(func)
        result = func("ATCGATCG")
        assert isinstance(result, float)

    def test_import_reverse_complement(self):
        func = _import_func("biosuite.core.sequence:reverse_complement")
        assert callable(func)
        result = func("ATCG")
        assert result == "CGAT"

    def test_import_needleman_wunsch(self):
        func = _import_func("biosuite.core.alignment:needleman_wunsch")
        assert callable(func)

    def test_import_nonexistent_raises(self):
        with pytest.raises(AttributeError):
            _import_func("biosuite.core.sequence:does_not_exist")

    def test_import_bad_module_raises(self):
        with pytest.raises(ModuleNotFoundError):
            _import_func("biosuite.core.nonexistent:func")


# ── _execute_command tests ──────────────────────────────────────────────────


class TestExecuteCommand:
    """Execute commands registered in COMMAND_REGISTRY."""

    def test_gc_command(self, capsys):
        """GC command should print a percentage."""
        ns = argparse.Namespace()
        ns.sequence = "ATCGATCG"
        _execute_command("gc", ns)
        captured = capsys.readouterr()
        assert "50.0" in captured.out  # ATCGATCG → 4/8 = 50%

    def test_revcomp_command(self, capsys):
        """Revcomp command should print reverse complement."""
        ns = argparse.Namespace()
        ns.sequence = "ATCG"
        _execute_command("revcomp", ns)
        captured = capsys.readouterr()
        assert "CGAT" in captured.out

    def test_translate_command(self, capsys):
        """Translate command should print a protein."""
        ns = argparse.Namespace()
        ns.sequence = "ATGAAATTTTAA"
        ns.frame = 1
        _execute_command("translate", ns)
        captured = capsys.readouterr()
        assert "M" in captured.out  # ATG → M

    def test_stats_command(self, capsys):
        """Stats command should print composition stats."""
        ns = argparse.Namespace()
        ns.sequence = "ATCGATCG"
        _execute_command("stats", ns)
        captured = capsys.readouterr()
        assert "length" in captured.out

    def test_unknown_command(self, capsys):
        """Unknown command should print available commands."""
        ns = argparse.Namespace()
        _execute_command("nonexistent", ns)
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_theme_change(self, capsys):
        """Theme command should change theme without error."""
        ns = argparse.Namespace()
        ns.theme_name = "dark-green"
        _execute_command("theme", ns)
        captured = capsys.readouterr()
        assert "dark-green" in captured.out


# ── print_menu tests ────────────────────────────────────────────────────────


class TestPrintMenu:
    """print_menu should render without errors."""

    def test_print_menu_runs(self, capsys):
        """Menu should render to stdout without exceptions."""
        print_menu()
        captured = capsys.readouterr()
        assert len(captured.out) > 100  # non-trivial output

    def test_print_menu_contains_title(self, capsys):
        """Menu should contain the BioSuite title."""
        print_menu()
        captured = capsys.readouterr()
        assert "BIOSUITE" in captured.out or "Bioinformatic" in captured.out


# ── _change_theme tests ─────────────────────────────────────────────────────


class TestChangeTheme:
    """Theme change via _change_theme helper."""

    def test_change_theme_prints(self, capsys):
        _change_theme("dark-green")
        captured = capsys.readouterr()
        assert "dark-green" in captured.out


# ── main_cli tests ──────────────────────────────────────────────────────────


class TestMainCli:
    """Test the main_cli entry point in non-interactive mode."""

    def test_gc_command(self, capsys):
        """main_cli(['gc', 'ATCGATCG']) should compute GC content."""
        main_cli(["gc", "ATCGATCG"])
        captured = capsys.readouterr()
        assert "50.0" in captured.out

    def test_revcomp_command(self, capsys):
        main_cli(["revcomp", "ATCG"])
        captured = capsys.readouterr()
        assert "CGAT" in captured.out

    def test_translate_command(self, capsys):
        main_cli(["translate", "ATGAAATTTTAA"])
        captured = capsys.readouterr()
        assert "M" in captured.out

    def test_stats_command(self, capsys):
        main_cli(["stats", "ATCGATCG"])
        captured = capsys.readouterr()
        assert "length" in captured.out

    def test_unknown_command(self, capsys):
        main_cli(["badcmd"])
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_theme_command(self, capsys):
        main_cli(["theme", "dark-purple"])
        captured = capsys.readouterr()
        assert "dark-purple" in captured.out

    def test_empty_cmd_args_raises(self):
        """Calling gc with no args raises an error (missing required argument)."""
        with pytest.raises((TypeError, AttributeError, SystemExit)):
            main_cli(["gc"])


# ── Interactive mode mock test ───────────────────────────────────────────────


class TestInteractiveMode:
    """Interactive mode (with mocked input)."""

    def test_interactive_exits_on_zero(self, capsys):
        """Interactive mode should exit when user types '0'."""
        import sys as _sys
        with patch("builtins.input", return_value="0"), \
             patch.object(_sys, "argv", ["biosuite"]):
            main_cli()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out or "Thank you" in captured.out

    def test_interactive_choice_gc(self, capsys):
        """Choice '27' should trigger GC calculation in interactive mode."""
        with patch("builtins.input", side_effect=["27", "ATCGATCG", "0"]), \
             patch.object(sys, 'argv', ["biosuite"]):
            main_cli(argv=None)
        captured = capsys.readouterr()
        assert "50.0" in captured.out or "GC" in captured.out
