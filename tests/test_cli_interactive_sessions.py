"""End-to-end interactive CLI sessions (menu loop driven with fake input)."""
import builtins
import io
import sys
import pytest

from biosuite.cli import menu


def _drive(menu_choices):
    answers = iter(menu_choices)

    def fake_input(prompt=''):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None
    return fake_input


def _run():
    """Drive one interactive session with the input the test already installed.

    ``'clean'`` means the session reached ``0`` and printed its goodbye;
    ``'eof-ended'`` means the answers ran out first — which is what happens when an
    option asks more questions than the test supplied, or crashes before returning
    to the menu.
    """
    argv_backup = sys.argv
    sys.argv = ['biosuite']
    try:
        menu.main_cli()
        return 'clean'
    except EOFError:
        return 'eof-ended'
    finally:
        sys.argv = argv_backup


def test_reverse_complement_session(monkeypatch, capsys):
    monkeypatch.setattr(builtins, 'input',
                        _drive(['25', 'ATGCGTACGTTAGCGTACAGTACTGCATGCATCGTC', '0']))

    assert _run() == 'clean'
    assert 'Result: GACGATGCATGCAGTACTGTACGCTAACGTACGCAT' in capsys.readouterr().out


def test_gc_calculator_session(monkeypatch):
    monkeypatch.setattr(builtins, 'input', _drive(
        ['27', 'ATGCGTACGTTAGCGTACAGTACTGCATGCATCGTCGGG', '0']))
    assert _run() == 'clean'


def test_invalid_option_handling(monkeypatch):
    monkeypatch.setattr(builtins, 'input', _drive(['notexist', 'xyz', '0']))
    assert _run() == 'clean'


def test_eof_ends_session_cleanly(monkeypatch, capsys):
    """Regression guard for BSU-014.

    Ctrl-D at the menu prompt used to escape as an EOFError traceback from
    menu.py; it must now end the session the same way option 0 does.
    """
    monkeypatch.setattr(builtins, 'input', _drive([]))
    assert _run() == 'clean'
    out = capsys.readouterr().out
    assert 'Goodbye' in out


def test_exception_in_handler_does_not_kill_session(monkeypatch, capsys):
    """Regression guard for BSU-014: a failing action returns to the menu.

    Option 75 with a malformed Newick string used to raise IndexError out of
    the loop and terminate the whole session.  The parser now rejects unbalanced
    parentheses with a ValueError naming the defect (N15), so the exception type
    reported changed — what this guard pins is that it is reported *and survived*.
    """
    monkeypatch.setattr(builtins, 'input', _drive(['75', '((((bad', '0']))
    assert _run() == 'clean'
    out = capsys.readouterr().out
    assert 'ValueError' in out                      # reported...
    assert 'session is still running' in out        # ...but survived
    assert 'Goodbye' in out                         # and reached the clean exit


def test_menu_has_no_code_evaluation(monkeypatch):
    """Regression guard for BSU-003: no eval()/exec() anywhere in the CLI."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(menu.__file__).read_text(encoding='utf8'))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert 'eval' not in called
    assert 'exec' not in called
    assert 'compile' not in called


def test_batch_builder_rejects_python_expressions(monkeypatch, capsys):
    """Regression guard for BSU-003: option 93 must not execute user code."""
    payload = '__import__("os").system("touch /tmp/biosuite-rce-probe")'
    monkeypatch.setattr(builtins, 'input', _drive(['93', payload, '0']))
    assert _run() == 'clean'
    out = capsys.readouterr().out
    assert 'Unknown function' in out
    import os
    assert not os.path.exists('/tmp/biosuite-rce-probe')


def test_resolve_safe_callable_rejects_non_biosuite_targets():
    """Only whitelisted names and biosuite.* paths resolve."""
    with pytest.raises(ValueError):
        menu.resolve_safe_callable('os:system')
    with pytest.raises(ValueError):
        menu.resolve_safe_callable('subprocess:run')
    with pytest.raises(ValueError):
        menu.resolve_safe_callable('lambda x: x')
    with pytest.raises(ValueError):
        menu.resolve_safe_callable('')
    assert menu.resolve_safe_callable('gc_content')('ATGC') == 50.0
    assert menu.resolve_safe_callable('biosuite.core.sequence:gc_content')('ATGC') == 50.0


def test_parser_help_non_interactive(monkeypatch, capsys):
    import sys
    argv = sys.argv
    sys.argv = ['biosuite', '--help']
    try:
        with pytest.raises(SystemExit):
            menu.main_cli()
        out, _ = capsys.readouterr()
        assert 'biosuite' in out.lower()
    finally:
        sys.argv = argv


def test_orf_finder_session(monkeypatch, capsys):
    monkeypatch.setattr(builtins, 'input', _drive(
        ['63', 'ATGACGTACGTTAAATGAAACGTACAGTTAAGG', '100', '0']))

    assert _run() == 'clean'
    assert 'Open Reading Frames' in capsys.readouterr().out


def test_translate_session(monkeypatch, capsys):
    monkeypatch.setattr(builtins, 'input', _drive(
        ['26', 'ATGGCACGTGCTACGTAAC', '', '0']))    # third answer: frame, Enter = 1

    assert _run() == 'clean'
    assert 'Protein: MARAT*' in capsys.readouterr().out


def test_bigwig_reader_graceful(monkeypatch, tmp_path):
    import struct
    bw = tmp_path / 'x.bw'
    # BBI magic: little-endian 0x888FFC26
    magic = struct.pack('<I', 0x888FFC26)
    bw.write_bytes(magic + b'\x00' * 60)
    monkeypatch.setattr(builtins, 'input', _drive(['c', str(bw), '0']))

    assert _run() == 'clean'


def test_kmer_counter_session(monkeypatch, capsys):
    monkeypatch.setattr(builtins, 'input', _drive(
        ['66', 'ACGTACGTACGTACGTACGTACGTACGTACGT', '6', '0']))

    assert _run() == 'clean'
    assert 'ACGTAC' in capsys.readouterr().out


def test_file_format_detector_session(monkeypatch, tmp_path):
    fa = tmp_path / 'demo.fa'
    fa.write_text('>s\nACGT\n')
    monkeypatch.setattr(builtins, 'input', _drive(['g', str(fa), '0']))
    out = _run()
    assert out == 'clean'


class _RecordingStream:
    """A stream that records ``reconfigure`` calls, like a Windows console."""

    def __init__(self):
        self.reconfigures = []

    def reconfigure(self, **kwargs):
        self.reconfigures.append(kwargs)

    def write(self, text):
        return len(text)


def test_utf8_stdio_helper_reconfigures_both_streams(monkeypatch):
    out, err = _RecordingStream(), _RecordingStream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    menu._enable_utf8_stdio()
    menu._enable_utf8_stdio()

    expected = {"encoding": "utf-8", "errors": "replace"}
    assert out.reconfigures == [expected, expected]
    assert err.reconfigures == [expected, expected]


def test_utf8_stdio_helper_leaves_streams_without_reconfigure_alone(monkeypatch):
    """pytest's own capture object has no reconfigure(); the helper must skip it."""
    out, err = object(), object()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    menu._enable_utf8_stdio()

    assert sys.stdout is out and sys.stderr is err


def test_banner_survives_a_cp1252_console_only_after_the_helper_runs(monkeypatch):
    """The regression itself: legacy Windows code pages cannot encode the box art."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)

    with pytest.raises(UnicodeEncodeError):
        menu._header()

    menu._enable_utf8_stdio()
    menu._header()
    stream.flush()

    written = raw.getvalue().decode("utf-8")
    # One banner survived; the first attempt died on its first box-drawing line.
    assert written.count("Bioinformatic Platform") == 1
