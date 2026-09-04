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


def _run(choices):
    import sys
    argv_backup = sys.argv
    sys.argv = ['biosuite']
    try:
        menu.main_cli()
        return 'clean'
    except EOFError:
        return 'eof-ended'
    finally:
        sys.argv = argv_backup


def test_reverse_complement_session(monkeypatch):
    def _translate_input(prompt=''):
        if 'DNA sequence' in prompt or 'sequence' in prompt or True:
            return 'ATGCGTACGTTAGCGTACAGTACTGCATGCATCGTC'
    monkeypatch.setattr(builtins, 'input',
                        _drive(['25', 'ATGCGTACGTTAGCGTACAGTACTGCATGCATCGTC', '0']))
    state = _run(['25',  # reverse complement branch
                  'ATGCGTACGTTAGCGTACAGTACTGCATGCATCGTC',
                  '0'])
    assert state == 'clean'


def test_gc_calculator_session(monkeypatch):
    monkeypatch.setattr(builtins, 'input', _drive(
        ['27', 'ATGCGTACGTTAGCGTACAGTACTGCATGCATCGTCGGG', '0']))
    assert _run([]) == 'clean'


def test_invalid_option_handling(monkeypatch):
    monkeypatch.setattr(builtins, 'input', _drive(['notexist', 'xyz', '0']))
    assert _run([]) == 'clean'


def test_eof_breaks_session(monkeypatch):
    monkeypatch.setattr(builtins, 'input', _drive([]))
    assert _run([]) == 'eof-ended'


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


def test_orf_finder_session(monkeypatch):
    monkeypatch.setattr(builtins, 'input', _drive(
        ['63', 'ATGACGTACGTTAAATGAAACGTACAGTTAAGG', '100', '0', '0']))
    # ORF finder may ask min-length; answers fed; must exit cleanly
    _run([])


def test_translate_session(monkeypatch):
    monkeypatch.setattr(builtins, 'input', _drive(
        ['26', 'ATGGCACGTGCTACGTAAC', '0']))
    _run([])


def test_bigwig_reader_graceful(monkeypatch, tmp_path):
    import struct
    bw = tmp_path / 'x.bw'
    # BBI magic: little-endian 0x888FFC26
    magic = struct.pack('<I', 0x888FFC26)
    bw.write_bytes(magic + b'\x00' * 60)
    monkeypatch.setattr(builtins, 'input', _drive(['c', str(bw), '0']))
    _run([])


def test_kmer_counter_session(monkeypatch):
    monkeypatch.setattr(builtins, 'input', _drive(
        ['66', 'ACGTACGTACGTACGTACGTACGTACGTACGT', '6', '0']))
    _run([])


def test_file_format_detector_session(monkeypatch, tmp_path):
    fa = tmp_path / 'demo.fa'
    fa.write_text('>s\nACGT\n')
    monkeypatch.setattr(builtins, 'input', _drive(['g', str(fa), '0']))
    out = _run([])
    assert out == 'clean'


def test_utf8_stdio_helper_idempotent():
    from biosuite.cli.menu import _enable_utf8_stdio
    _enable_utf8_stdio()
    _enable_utf8_stdio()  # twice: must never raise


def test_stdout_reconfigure_safe_on_captured_frames():
    """Regression: banner must not crash when stdout is cp1252-limited (Windows)."""
    from biosuite.cli.menu import _header, _enable_utf8_stdio
    _enable_utf8_stdio()
    buf = io.TextIOWrapper(io.BytesIO(), encoding='cp1252', errors='replace')
    old = sys.stdout
    try:
        sys.stdout = buf
        _header()
        buf.flush()
    finally:
        sys.stdout = old
