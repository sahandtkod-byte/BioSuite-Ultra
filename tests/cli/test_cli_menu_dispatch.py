"""Regression: cli/menu.py display options must map 1:1 to dispatch branches.

History: option 81 was BOTH "Synteny Dotplot" (display) and codon-usage
(first elif won -> Synteny unreachable, wrong action silently), and the
ANALYSIS block displayed 120-125 while dispatch lived at 81-86 (options
120-125 were silently "Invalid option").
"""
import re
from pathlib import Path

MENU = Path(__file__).resolve().parents[2] / 'biosuite' / 'cli' / 'menu.py'


def _dl_parts():
    src = MENU.read_text(encoding='utf8')
    body = src[src.index('def print_menu'):src.index('def main_cli')]
    return src, body


def test_no_duplicate_dispatch_keys():
    src, _ = _dl_parts()
    keys = re.findall(r"(?:el)?if choice == '([a-z0-9]+)'", src)
    dupes = {k for k in set(keys) if keys.count(k) > 1}
    assert not dupes, f"duplicate dispatch branches: {dupes}"


def test_displayed_options_are_dispatched():
    _, body = _dl_parts()
    displayed = set(re.findall(r"\{G\}\s*([0-9]{1,3})\{R\}", body))
    displayed |= set(re.findall(r"\{G\}\s*([a-q])\{R\}", body))
    src, _ = _dl_parts()
    dispatched = set(re.findall(r"(?:el)?if choice == '([a-z0-9]+)'", src)) | {'0'}
    missing = displayed - dispatched
    # 101-111 plot functions are dispatched as single-line elifs (see file)
    assert not missing, f"displayed but not dispatched: {sorted(missing)}"


def test_analysis_block_aligned():
    """Codon/kmer/complexity/survival/network must be at 120-125, not 81-86."""
    src, _ = _dl_parts()
    analysis = src[src.index('# ── Analysis'):src.index('# ── Advanced Visualization')]
    for num in ('120', '121', '122', '123', '124', '125'):
        assert f"elif choice == '{num}':" in analysis
    synteny = src[src.index('Advanced Visualization (77'):src.index('# ── Sequence Tools')]
    assert re.search(r"elif choice == '81':", synteny)


def test_registry_funcs_resolve():
    """Every COMMAND_REGISTRY module:function path must exist."""
    import importlib

    from biosuite.cli.menu import COMMAND_REGISTRY
    for name, entry in COMMAND_REGISTRY.items():
        path = entry['func']
        if path.startswith('_'):
            continue
        mod, func = path.rsplit(':', 1)
        assert hasattr(importlib.import_module(mod), func), name
